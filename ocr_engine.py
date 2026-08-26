"""
OCR Engine — Extracts Hindi + Sanskrit + English text from PDF files.

For each page we FIRST look at the embedded text layer and let
text_quality.choose_method() decide how to extract it:

    "direct"  → the text layer is clean → use it as-is (fast, no OCR)
    "ocr"     → no/garbled text layer → rasterize and run the selected OCR backend

Google Cloud Vision is the default OCR backend and uses Application Default
Credentials. Tesseract remains available as a fully local backend.
"""

import io
import random
import re
import time
import difflib
import statistics
import unicodedata

import pymupdf  # formerly imported as `fitz` (deprecated alias)
import pytesseract
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import (
    GOOGLE_VISION_LANGUAGE_HINTS,
    OCR_BACKEND,
    OCR_MAX_ATTEMPTS,
    OCR_BACKOFF_BASE_S,
    OCR_MAX_WORKERS,
    TESSERACT_LANG,
    PDF_DPI,
    LAYER_CHECK_SAMPLE,
    LAYER_CHECK_MIN_SIMILARITY,
)
from text_quality import choose_method

console = Console()


def _pixmap_to_image(pix) -> Image.Image:
    """Convert a PyMuPDF pixmap to a PIL image Tesseract can read."""
    mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(pix.n, "RGB")
    img = Image.frombytes(mode, (pix.width, pix.height), pix.samples)
    if mode == "RGBA":
        img = img.convert("RGB")
    return img


def _render_page(page) -> Image.Image:
    """Rasterize one PDF page at PDF_DPI into an RGB PIL image."""
    mat = pymupdf.Matrix(PDF_DPI / 72, PDF_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    return _pixmap_to_image(pix).convert("RGB")


def _ocr_with_tesseract(png: bytes) -> str:
    """Run the local Tesseract backend on encoded PNG bytes."""
    text = pytesseract.image_to_string(Image.open(io.BytesIO(png)), lang=TESSERACT_LANG)
    return text.strip()


def _with_retry(fn, what: str):
    """Retry a hosted-OCR call with exponential backoff.

    Hosted OCR fails transiently in two ways seen in practice: Vision 503
    "service is currently unavailable", and Vertex 429 RESOURCE_EXHAUSTED under
    per-minute quota. Neither means the page is unreadable — retrying works.
    Without this, a single blip aborts an entire multi-hour index.
    """
    last = None
    for attempt in range(OCR_MAX_ATTEMPTS):
        try:
            return fn()
        except Exception as exc:                     # noqa: BLE001 - re-raised below
            last = exc
            if attempt == OCR_MAX_ATTEMPTS - 1:
                break
            # Jitter matters once requests run concurrently: without it, N
            # workers hitting the same 503 window back off on an identical
            # schedule and retry in lockstep, amplifying the outage instead of
            # riding it out.
            wait = OCR_BACKOFF_BASE_S * (2 ** attempt) * random.uniform(0.5, 1.5)
            console.print(
                f"   [yellow]{what} attempt {attempt + 1}/{OCR_MAX_ATTEMPTS} "
                f"failed ({type(exc).__name__}); retrying in {wait:.1f}s[/yellow]"
            )
            time.sleep(wait)
    raise RuntimeError(f"{what} failed after {OCR_MAX_ATTEMPTS} attempts") from last


_vision_client = None


def _get_vision_client():
    """Build the Vision client once and reuse it.

    The previous code constructed ImageAnnotatorClient() inside the per-page
    function, so every page paid credential lookup and TLS setup.
    """
    global _vision_client
    if _vision_client is None:
        from google.cloud import vision
        _vision_client = vision.ImageAnnotatorClient()
    return _vision_client


def _ocr_with_google_vision(png: bytes) -> str:
    """Run Cloud Vision dense-document OCR using ADC authentication."""
    try:
        from google.cloud import vision
    except ImportError as exc:
        raise RuntimeError(
            "Google Cloud Vision OCR requires google-cloud-vision. "
            "Install it with: .venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    def _call():
        client = _get_vision_client()
        response = client.document_text_detection(
            image=vision.Image(content=png),
            image_context=vision.ImageContext(
                language_hints=GOOGLE_VISION_LANGUAGE_HINTS
            ),
        )
        # The in-band check MUST live inside the retried callable. Vision reports
        # transient failures ("The service is currently unavailable") on the
        # RESPONSE rather than by raising, so checking it outside the retry means
        # the retry sees success and the error escapes un-retried. Observed
        # exactly that way on 2026-08-25 before this was moved inward.
        if response.error.message:
            raise RuntimeError(
                f"Google Cloud Vision OCR failed: {response.error.message}"
            )
        return response

    try:
        response = _with_retry(_call, "Google Cloud Vision OCR")
    except Exception as exc:
        raise RuntimeError(
            "Google Cloud Vision OCR request failed. Confirm Application Default "
            "Credentials, Vision API enablement, billing, and quota."
        ) from exc

    return response.full_text_annotation.text.strip()


_gemini_client = None

# Transcription, not interpretation. Every rule here exists because the model
# otherwise "helps": translating, correcting spelling, or adding a preamble.
_GEMINI_OCR_PROMPT = """Transcribe ALL text visible in this scanned book page, exactly as printed.

Rules:
- Reproduce the text character-for-character. Do NOT translate, correct,
  modernize, complete, or normalize spelling.
- Preserve the original line breaks and reading order of the printed blocks.
- Include Devanagari numerals and dandas exactly as printed.
- If any part of the page is blank, obscured, or unreadable, write [ILLEGIBLE]
  at that position. Do NOT guess what belongs there.
- Output ONLY the transcribed text. No preamble, no commentary, no translation.
"""


def _ocr_with_gemini(png: bytes) -> str:
    """Run multimodal-LLM OCR (config.LLM_MODEL) on the page image.

    ⚠ This backend's failure mode differs IN KIND from the other two. Tesseract
    emits detectable garbage and Vision misorders blocks, but an LLM can emit
    fluent, plausible text that was never on the page — and, measured here, it
    SILENTLY DROPS regions it cannot read instead of writing [ILLEGIBLE] as the
    prompt asks. Verified 2026-08-25 by whiting out two words of a printed verse:
    the model omitted them without any marker. Quiet data loss is harder to
    detect downstream than noise, so prefer 'google_vision' for bulk indexing.

    thinking_budget=0 matters: 2.5-flash reasons by default and thinking tokens
    bill at the OUTPUT rate, buying nothing for transcription.
    """
    global _gemini_client
    try:
        from google.genai import types
    except ImportError as exc:
        raise RuntimeError(
            "Gemini OCR requires google-genai. "
            "Install it with: .venv/bin/python -m pip install -r requirements.txt"
        ) from exc

    from llm_client import get_client
    from config import LLM_MODEL

    if _gemini_client is None:
        _gemini_client = get_client()

    def _call():
        return _gemini_client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                types.Part.from_bytes(data=png, mime_type="image/png"),
                _GEMINI_OCR_PROMPT,
            ],
            config=types.GenerateContentConfig(
                temperature=0,
                thinking_config=types.ThinkingConfig(thinking_budget=0),
            ),
        )

    # 21 of 90 benchmark pages failed with 429 RESOURCE_EXHAUSTED (Vertex
    # express per-minute quota) purely for want of this.
    try:
        response = _with_retry(_call, "Gemini OCR")
    except Exception as exc:
        raise RuntimeError(
            "Gemini OCR request failed. Confirm LLM_BACKEND, the matching API "
            "key, and per-minute quota."
        ) from exc

    return (response.text or "").strip()


_OCR_BACKENDS = {
    "tesseract": _ocr_with_tesseract,
    "google_vision": _ocr_with_google_vision,
    "gemini": _ocr_with_gemini,
}


def _prepare_page(page) -> bytes:
    """Rasterize + encode one page. MUST run on the main thread.

    PyMuPDF's Document is not thread-safe, so page rendering cannot move into a
    worker. Splitting here is what makes concurrency safe AND cheap: rendering
    is ~0.25 s while a Vision round-trip is ~2.6 s, so the part that stays
    serial is ~10% of the work.

    Encoding here rather than inside the backend also bounds memory. A rendered
    10.5 MP RGB page is ~31 MB in RAM; the grayscale PNG it becomes is ~1.6 MB.
    With 8 workers in flight that is the difference between ~250 MB and ~13 MB.
    """
    img = _render_page(page)
    buf = io.BytesIO()
    if OCR_BACKEND == "tesseract":
        # Keep Tesseract's input byte-identical to what it saw before this
        # refactor, so its measured accuracy still applies.
        img.save(buf, format="PNG")
    else:
        # Hosted backends are upload-bound; grayscale is ~4x smaller for
        # 0.9994 text similarity (measured 2026-08-25).
        img.convert("L").save(buf, format="PNG", optimize=True)
    return buf.getvalue()


def _ocr_page(page) -> str:
    """Rasterize one PDF page and run the configured OCR backend.

    Single per-page entry point: the parallel path calls the same
    _prepare_page/_OCR_BACKENDS pair, so retry, grayscale encoding and backend
    dispatch cannot drift between the two.
    """
    return _OCR_BACKENDS[OCR_BACKEND](_prepare_page(page))


def _norm(s: str) -> str:
    """Normalize for comparison: Unicode NFC + collapse whitespace runs."""
    s = unicodedata.normalize("NFC", s)
    return re.sub(r"\s+", " ", s).strip()


def _verify_text_layer(doc) -> bool:
    """Language-agnostic corruption defense.

    Some PDFs have a broken font→Unicode map: the page renders fine but
    get_text() returns scrambled codepoints. We OCR a few 'direct'-routed
    pages and compare to their text layer. A clean layer agrees with OCR
    (high similarity); a corrupt layer disagrees strongly. Returns False if
    the layer looks corrupt → the caller should OCR the whole document.
    """
    cands = []
    for i in range(len(doc)):
        raw = doc[i].get_text().strip()
        if choose_method(raw) == "direct" and len(raw) > 200:
            cands.append(i)

    if not cands:
        return True  # nothing trustable to begin with (e.g. fully scanned)

    k = min(LAYER_CHECK_SAMPLE, len(cands))
    if k == 1:
        sample = [cands[0]]
    else:
        sample = [cands[round(j * (len(cands) - 1) / (k - 1))] for j in range(k)]

    ratios = []
    for i in sample:
        layer = _norm(doc[i].get_text().strip())
        ocr = _norm(_ocr_page(doc[i]))
        # autojunk=False is REQUIRED for Devanagari. difflib's default discards
        # any element occurring in >1% of a 200+ element sequence as "junk" — a
        # source-code heuristic. Devanagari's small effective alphabet means ~23
        # characters cover ~83% of a Hindi page, so ALL of them get discarded and
        # a CLEAN layer scores ~0.02. Measured 2026-08-25 on मनुस्मृति: 0.0235
        # with the default vs 0.8571 without it. English is barely affected,
        # which is why the original 0.02-vs-0.93 calibration looked bimodal —
        # it was measuring SCRIPT, not corruption.
        ratios.append(
            difflib.SequenceMatcher(None, layer, ocr, autojunk=False).ratio()
        )

    median = statistics.median(ratios)
    trusted = median >= LAYER_CHECK_MIN_SIMILARITY
    if trusted:
        console.print(f"   ✅ Text-layer spot check OK (median similarity {median:.2f})")
    else:
        console.print(
            f"   [red]⚠ Text layer looks corrupt (median similarity {median:.2f} < "
            f"{LAYER_CHECK_MIN_SIMILARITY}) — forcing OCR for this document.[/red]"
        )
    return trusted


def _ocr_pages(doc, page_nums: list[int], progress=None, task=None) -> dict[int, str]:
    """OCR the given pages, concurrently when OCR_MAX_WORKERS > 1.

    Only the network call is parallel. Each chunk is rasterized and encoded on
    THIS thread (PyMuPDF is not thread-safe), then handed to workers as bytes.

    Chunking bounds memory: at most `workers * 2` encoded pages are held at
    once, so a 900-page document costs the same RAM as a 16-page one.

    A page that fails every retry yields "" and is reported, rather than
    aborting the run — losing one page of a 3653-page index should not cost the
    other 3652.
    """
    backend = _OCR_BACKENDS[OCR_BACKEND]
    results: dict[int, str] = {}

    def _done(n, text):
        results[n] = text
        if progress is not None:
            progress.update(task, advance=1)

    # workers == 1 takes the plain sequential path, byte-for-byte the old
    # behaviour, so there is always a fallback if concurrency misbehaves.
    if OCR_MAX_WORKERS <= 1:
        for n in page_nums:
            try:
                _done(n, backend(_prepare_page(doc[n])))
            except Exception as exc:                       # noqa: BLE001
                console.print(f"   [red]page {n + 1}: OCR failed — {exc}[/red]")
                _done(n, "")
        return results

    # One pool for the whole document; chunking below is what bounds memory, so
    # the pool does not need to be rebuilt per chunk.
    chunk = max(1, OCR_MAX_WORKERS * 2)
    with ThreadPoolExecutor(max_workers=OCR_MAX_WORKERS) as pool:
        for start in range(0, len(page_nums), chunk):
            batch = page_nums[start:start + chunk]
            # Rasterize + encode HERE, on the main thread. Only `chunk` encoded
            # pages exist at once, so RAM is flat in document length.
            prepared = [(n, _prepare_page(doc[n])) for n in batch]
            futures = {pool.submit(backend, data): n for n, data in prepared}
            for fut in as_completed(futures):
                n = futures[fut]
                try:
                    _done(n, fut.result())
                except Exception as exc:                   # noqa: BLE001
                    console.print(f"   [red]page {n + 1}: OCR failed — {exc}[/red]")
                    _done(n, "")
    return results


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a PDF, routing each page to the right method.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys: 'page', 'text', 'method'
        'method' is 'direct' (text layer) or 'ocr' (rasterized + Tesseract).
        Example: [{'page': 1, 'text': 'पाठ 1...', 'method': 'direct'}]
    """
    doc = pymupdf.open(pdf_path)
    total_pages = len(doc)

    console.print(f"\n📄 Processing: [bold]{pdf_path}[/bold] ({total_pages} pages)")
    console.print(f"   🧠 OCR backend: [bold cyan]{OCR_BACKEND}[/bold cyan]")

    # Language-agnostic defense: is the embedded text layer trustworthy?
    trust_layer = _verify_text_layer(doc)

    # ── Pass 1: routing (cheap — reads the text layer only, no OCR) ──
    # Deciding routes up front is what lets the OCR pages be batched and run
    # concurrently; interleaving the decision with the work would serialise it.
    routes: dict[int, tuple[str, str]] = {}
    for page_num in range(total_pages):
        raw = doc[page_num].get_text().strip()
        method = choose_method(raw)
        if method == "direct" and not trust_layer:
            method = "ocr"  # layer proven unreliable → OCR even 'direct' pages
        routes[page_num] = (method, raw if method == "direct" else "")

    ocr_pages = [n for n, (m, _) in routes.items() if m == "ocr"]
    method_counts = {
        "direct": total_pages - len(ocr_pages),
        "ocr": len(ocr_pages),
    }
    if ocr_pages:
        console.print(
            f"   ⚡ OCR: {len(ocr_pages)} pages · {OCR_MAX_WORKERS} worker(s)"
        )

    # ── Pass 2: OCR the pages that need it ───────────────────────────
    ocr_text: dict[int, str] = {}
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 Extracting text", total=total_pages)
        progress.update(task, advance=method_counts["direct"])  # direct pages are free
        if ocr_pages:
            ocr_text = _ocr_pages(doc, ocr_pages, progress, task)

    # ── Assemble in page order ───────────────────────────────────────
    pages_text = []
    for page_num in range(total_pages):
        method, raw = routes[page_num]
        text = raw if method == "direct" else ocr_text.get(page_num, "")
        if text:
            pages_text.append({
                "page": page_num + 1,
                "text": text,
                "method": method,
            })

    doc.close()

    # Summary
    pages_with_text = len(pages_text)
    total_chars = sum(len(p["text"]) for p in pages_text)
    console.print(f"\n[green]✅ Extraction Complete![/green]")
    console.print(f"   📊 {pages_with_text}/{total_pages} pages had text")
    console.print(
        f"   🧭 Routing: [cyan]{method_counts['direct']} direct[/cyan] · "
        f"[magenta]{method_counts['ocr']} ocr[/magenta]"
    )
    console.print(f"   📝 {total_chars:,} characters extracted")

    return pages_text
