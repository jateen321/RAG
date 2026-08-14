"""
OCR Engine — Extracts Hindi + Sanskrit + English text from PDF files.

For each page we FIRST look at the embedded text layer and let
text_quality.choose_method() decide how to extract it:

    "direct"  → the text layer is clean → use it as-is (fast, no OCR)
    "ocr"     → no/garbled text layer → rasterize the page and run Tesseract

Tesseract is a binary (installed via Homebrew) called through the pytesseract
wrapper. Unlike EasyOCR there is no large model to preload — each page is a
fresh call to the engine with lang=TESSERACT_LANG (eng+hin+san).
"""

import re
import difflib
import statistics
import unicodedata

import fitz  # PyMuPDF
import pytesseract
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from config import (
    TESSERACT_LANG, PDF_DPI, LAYER_CHECK_SAMPLE, LAYER_CHECK_MIN_SIMILARITY,
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


def _ocr_page(page) -> str:
    """Rasterize one PDF page at PDF_DPI and run Tesseract on it."""
    mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
    pix = page.get_pixmap(matrix=mat)
    img = _pixmap_to_image(pix)
    text = pytesseract.image_to_string(img, lang=TESSERACT_LANG)
    return text.strip()


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
        ratios.append(difflib.SequenceMatcher(None, layer, ocr).ratio())

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
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    console.print(f"\n📄 Processing: [bold]{pdf_path}[/bold] ({total_pages} pages)")

    # Language-agnostic defense: is the embedded text layer trustworthy?
    trust_layer = _verify_text_layer(doc)

    pages_text = []
    method_counts = {"direct": 0, "ocr": 0}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 Extracting text", total=total_pages)

        for page_num in range(total_pages):
            page = doc[page_num]

            # Decide per page from its embedded text layer
            raw = page.get_text().strip()
            method = choose_method(raw)
            if method == "direct" and not trust_layer:
                method = "ocr"  # layer proven unreliable → OCR even 'direct' pages

            if method == "direct":
                text = raw
            else:
                text = _ocr_page(page)

            method_counts[method] += 1

            if text:
                pages_text.append({
                    "page": page_num + 1,
                    "text": text,
                    "method": method,
                })

            progress.update(task, advance=1)

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
