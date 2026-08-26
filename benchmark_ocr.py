"""Compare OCR backends over the same PDF pages: latency, cost, and text quality.

The point is a fair comparison, which means separating costs that belong to
different layers:

  * Rasterization (PDF page -> PNG) is work the LOCAL path must do before
    Tesseract sees anything. A hosted OCR API that accepts a PDF does it
    internally. Charging it to one side and not the other is the easiest way to
    get a misleading number, so it is timed ONCE per page and shared: both
    backends receive the identical PNG bytes.
  * The first call to a backend pays one-time cost (model load, TLS handshake,
    client construction). Cold and warm requests are therefore reported
    separately rather than averaged.

On measuring QUALITY without ground truth
-----------------------------------------
We have no transcribed reference for these scans, so this script does NOT claim
an accuracy number. It reports three things that are honestly measurable:

  * `chars` — text VOLUME, not quality. On a page with show-through (text from
    the reverse side bleeding into the scan) a HIGHER char count is plausibly
    WORSE: it means phantom text was read. Never rank backends by this alone.
  * `agreement` — difflib ratio between the two backends' output for the same
    page, after NFC + whitespace normalization. Agreement is not accuracy, but
    DISAGREEMENT LOCALIZES: the lowest-agreement pages are the manual-inspection
    queue, and that queue is the real deliverable.
  * `script_runs` — how many times the text switches between Devanagari and
    Latin. These books interleave an English translation, a Sanskrit verse and a
    Hindi commentary as separate BLOCKS. An engine that returns correct
    characters in scrambled reading order shreds those blocks into many short
    runs, which produces chunks that retrieve badly — the thing that actually
    matters downstream. Fewer runs = blocks kept intact.

Usage:
    python benchmark_ocr.py --backend tesseract --pages 5
    python benchmark_ocr.py --compare --pdf data/SRIMAD-BHAGAVAD-GITA.pdf \
        --start 250 --end 299
"""

from __future__ import annotations

import argparse
import io
import json
import re
import statistics
import time
import unicodedata
from pathlib import Path

import pymupdf

from config import PDF_DPI, TESSERACT_LANG, GOOGLE_VISION_LANGUAGE_HINTS

# Cloud Vision pricing, fetched from cloud.google.com/vision/pricing 2026-08-25.
# A "unit" is one image; for a multi-page file each PAGE is one image. Document
# Text Detection: first 1,000 units/month free, then $1.50 per 1,000 units
# (through 5M/month), then $0.60 per 1,000.
VISION_FREE_UNITS_PER_MONTH = 1_000
VISION_USD_PER_1K_UNITS = 1.50

# Gemini 2.5 Flash, fetched from ai.google.dev/gemini-api/docs/pricing 2026-08-25.
# Paid tier, per 1M tokens. Thinking tokens bill at the OUTPUT rate.
GEMINI_USD_PER_1M_INPUT = 0.30
GEMINI_USD_PER_1M_OUTPUT = 2.50


def _rasterize(page) -> bytes:
    """Render one PDF page to PNG bytes at the configured DPI."""
    zoom = PDF_DPI / 72          # PDF user space is 72 dpi
    pix = page.get_pixmap(matrix=pymupdf.Matrix(zoom, zoom))
    return pix.tobytes("png")


def _ocr_tesseract(png: bytes) -> str:
    """OCR PNG bytes with the local Tesseract binary."""
    import pytesseract
    from PIL import Image

    return pytesseract.image_to_string(
        Image.open(io.BytesIO(png)), lang=TESSERACT_LANG
    )


_vision_client = None


def _ocr_vision(png: bytes) -> str:
    """OCR PNG bytes with Cloud Vision's DENSE-document model.

    `document_text_detection`, not `text_detection`: the latter targets sparse
    text (signage, photos) and reads a scanned book page worse. The client is
    built once and reused — constructing it per page would charge every page the
    auth/TLS setup that only the first one really pays.
    """
    global _vision_client
    from google.cloud import vision
    from PIL import Image

    if _vision_client is None:
        _vision_client = vision.ImageAnnotatorClient()

    # Upload GRAYSCALE. Vision's latency here is upload-bound, not OCR-bound:
    # 6.4 MB RGB PNG ~9.0s vs 1.6 MB grayscale PNG 2.64s for 0.9994 similarity
    # (Gita p.250/254/255, 2026-08-25). Mirrors ocr_engine's production path.
    # NOTE this means Vision receives a grayscale copy while Tesseract reads the
    # RGB original -- deliberate, because it is what each path really does.
    buf = io.BytesIO()
    Image.open(io.BytesIO(png)).convert("L").save(buf, format="PNG", optimize=True)

    # content= takes RAW bytes: the client base64-encodes for the wire itself.
    response = _vision_client.document_text_detection(
        image=vision.Image(content=buf.getvalue()),
        image_context=vision.ImageContext(
            language_hints=GOOGLE_VISION_LANGUAGE_HINTS
        ),
    )
    # Vision reports some failures IN-BAND rather than raising, so an unchecked
    # response looks like "OCR returned nothing" instead of "request rejected".
    if response.error.message:
        raise RuntimeError(f"Vision API error: {response.error.message}")
    return response.full_text_annotation.text


_gemini_client = None
_LAST_META: dict = {}

# Transcription, not interpretation. Each rule exists because the model
# otherwise "helps": translating, correcting spelling, or adding a preamble
# that inflates the output length and corrupts any char-count comparison.
GEMINI_OCR_PROMPT = """Transcribe ALL text visible in this scanned book page, exactly as printed.

Rules:
- Reproduce the text character-for-character. Do NOT translate, correct,
  modernize, complete, or normalize spelling.
- Preserve the original line breaks and reading order of the printed blocks.
- Include Devanagari numerals and dandas exactly as printed.
- If any part of the page is blank, obscured, or unreadable, write [ILLEGIBLE]
  at that position. Do NOT guess what belongs there.
- Output ONLY the transcribed text. No preamble, no commentary, no translation.
"""


def _ocr_gemini(png: bytes) -> str:
    """OCR PNG bytes with a multimodal LLM (config.LLM_MODEL).

    Two settings are load-bearing:
      * thinking_budget=0 — 2.5-flash reasons by default and thinking tokens
        bill as OUTPUT ($2.50/1M). Transcription gains nothing from it, so
        leaving it on measures the wrong cost AND the wrong latency.
      * temperature=0 — otherwise the same page yields different text per run
        and the comparison is not reproducible.

    NOTE the failure mode differs in KIND from the other two backends.
    Tesseract emits detectable garbage; Vision misorders blocks; an LLM emits
    FLUENT PLAUSIBLE TEXT THAT WAS NOT ON THE PAGE. Only ground truth or
    three-way disagreement surfaces that, so never rank this backend on
    char counts or `script_runs` (it reformats cleanly whether or not it read
    the layout correctly).
    """
    global _gemini_client
    from google.genai import types
    from llm_client import get_client
    from config import LLM_MODEL

    if _gemini_client is None:
        _gemini_client = get_client()

    resp = _gemini_client.models.generate_content(
        model=LLM_MODEL,
        contents=[types.Part.from_bytes(data=png, mime_type="image/png"),
                  GEMINI_OCR_PROMPT],
        config=types.GenerateContentConfig(
            temperature=0,
            thinking_config=types.ThinkingConfig(thinking_budget=0),
        ),
    )
    u = resp.usage_metadata
    # Read REAL token usage rather than estimating from the image tiling math:
    # the tiling rules are model-version specific and easy to get wrong.
    _LAST_META["gemini"] = {
        "input_tokens": u.prompt_token_count,
        "output_tokens": u.candidates_token_count,
        "thinking_tokens": getattr(u, "thoughts_token_count", None) or 0,
    }
    return resp.text or ""


BACKENDS = {
    "tesseract": _ocr_tesseract,
    "vision": _ocr_vision,
    "gemini": _ocr_gemini,
}

# ── Quality probes (no ground truth required) ─────────────────────────

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LATIN = re.compile(r"[A-Za-z]")


def _norm(s: str) -> str:
    """NFC + collapse whitespace, so formatting differences are not counted
    as textual disagreement."""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFC", s)).strip()


def _agreement(a: str, b: str) -> float:
    """difflib similarity of two backends' output for the same page.

    autojunk=False is NOT optional here. difflib's default marks any element
    occurring in >1% of a 200+ element sequence as "junk" and excludes it from
    matching -- a heuristic tuned for source-code diffs. Devanagari has a small
    effective alphabet, so on a page of Hindi ~23 characters cover ~83% of the
    text and ALL of them get discarded. Two near-identical pages then score
    0.008 instead of 0.979. Measured on History p.301.
    """
    import difflib

    return round(
        difflib.SequenceMatcher(None, _norm(a), _norm(b), autojunk=False).ratio(), 4
    )


def _script_runs(s: str) -> int:
    """Count switches between Devanagari and Latin blocks.

    A page laid out as [English para][Sanskrit verse][Hindi commentary] should
    yield a SMALL number of long runs. A large number means the reading order
    interleaved the scripts — correct glyphs, shredded blocks.
    """
    runs, prev = 0, None
    for ch in s:
        cur = "dev" if _DEVANAGARI.match(ch) else ("lat" if _LATIN.match(ch) else None)
        if cur is not None and cur != prev:
            runs += 1
            prev = cur
    return runs


def _devanagari_ratio(s: str) -> float:
    letters = len(_DEVANAGARI.findall(s)) + len(_LATIN.findall(s))
    if not letters:
        return 0.0
    return round(len(_DEVANAGARI.findall(s)) / letters, 3)


def _page_stats(text: str) -> dict:
    t = text.strip()
    return {
        "chars": len(t),
        "words": len(t.split()),
        "devanagari_ratio": _devanagari_ratio(t),
        "script_runs": _script_runs(t),
    }


# ── Runner ────────────────────────────────────────────────────────────


def run(pdf_path: Path, backend_names: list[str], start: int, end: int) -> dict:
    """OCR pages [start, end] (1-based inclusive) with each backend.

    Each page is rasterized ONCE and the identical PNG handed to every backend,
    so raster cost is not charged twice and no backend gets a different input.
    """
    doc = pymupdf.open(pdf_path)
    start = max(1, start)
    end = min(end, len(doc))
    if start > end:
        raise SystemExit(f"empty page range {start}..{end} ({pdf_path.name} has {len(doc)})")
    page_nos = list(range(start, end + 1))

    print(f"{pdf_path.name} · pages {start}-{end} ({len(page_nos)}) · {PDF_DPI} DPI")
    print(f"backends: {', '.join(backend_names)}\n")

    header = f"{'page':>5} {'raster_s':>9}"
    for b in backend_names:
        header += f" {b[:6] + '_s':>10} {b[:6] + '_ch':>10}"
    if len(backend_names) == 2:
        header += f" {'agree':>7}"
    print(header)

    rows = []
    for page_no in page_nos:
        page = doc[page_no - 1]

        t0 = time.perf_counter()
        png = _rasterize(page)
        raster_s = time.perf_counter() - t0

        rect = page.rect
        zoom = PDF_DPI / 72
        px = int(rect.width * zoom) * int(rect.height * zoom)

        row = {
            "page": page_no,
            "raster_s": round(raster_s, 3),
            "megapixels": round(px / 1e6, 2),
            "png_kb": round(len(png) / 1024),
            "backends": {},
        }

        for name in backend_names:
            try:
                t1 = time.perf_counter()
                text = BACKENDS[name](png)
                ocr_s = time.perf_counter() - t1
                row["backends"][name] = {
                    "ocr_s": round(ocr_s, 3),
                    **_page_stats(text),
                    **_LAST_META.pop(name, {}),
                    "text": text,
                }
            except Exception as exc:                      # keep the run going
                row["backends"][name] = {
                    "ocr_s": None, "chars": 0, "words": 0,
                    "devanagari_ratio": 0.0, "script_runs": 0,
                    "text": "", "error": f"{type(exc).__name__}: {exc}",
                }
                print(f"  !! page {page_no} {name}: {type(exc).__name__}: {exc}")

        if len(backend_names) == 2:
            a, b = backend_names
            row["agreement"] = _agreement(
                row["backends"][a]["text"], row["backends"][b]["text"]
            )

        line = f"{row['page']:>5} {row['raster_s']:>9.3f}"
        for name in backend_names:
            r = row["backends"][name]
            s = f"{r['ocr_s']:.3f}" if r["ocr_s"] is not None else "ERR"
            line += f" {s:>10} {r['chars']:>10}"
        if "agreement" in row:
            line += f" {row['agreement']:>7.3f}"
        print(line)

        rows.append(row)

    doc.close()
    return {"summary": _summarize(pdf_path, backend_names, rows), "pages": rows}


def _summarize(pdf_path: Path, backend_names: list[str], rows: list[dict]) -> dict:
    raster = [r["raster_s"] for r in rows]
    per_backend = {}

    for name in backend_names:
        times = [r["backends"][name]["ocr_s"] for r in rows
                 if r["backends"][name]["ocr_s"] is not None]
        chars = [r["backends"][name]["chars"] for r in rows]
        runs = [r["backends"][name]["script_runs"] for r in rows]
        errors = sum(1 for r in rows if "error" in r["backends"][name])

        stats = {
            # Cold vs warm: the first page pays one-time cost. Averaging it in
            # hides that cost for a long run and overstates it for a short one.
            "cold_first_page_s": times[0] if times else None,
            "warm_median_s": round(statistics.median(times[1:]), 3) if len(times) > 1 else None,
            "ocr_total_s": round(sum(times), 3),
            "ocr_mean_s": round(statistics.mean(times), 3) if times else None,
            "total_chars": sum(chars),
            "mean_chars": round(statistics.mean(chars)) if chars else None,
            "mean_script_runs": round(statistics.mean(runs), 1) if runs else None,
            "errors": errors,
        }
        if name == "vision":
            units = len(rows)                    # 1 page = 1 unit
            stats["vision_units"] = units
            stats["usd_if_free_tier_exhausted"] = round(
                units / 1000 * VISION_USD_PER_1K_UNITS, 4
            )
            stats["free_units_per_month"] = VISION_FREE_UNITS_PER_MONTH
        elif name == "gemini":
            tin = sum(r["backends"][name].get("input_tokens", 0) or 0 for r in rows)
            tout = sum((r["backends"][name].get("output_tokens", 0) or 0)
                       + (r["backends"][name].get("thinking_tokens", 0) or 0)
                       for r in rows)
            stats["input_tokens"] = tin
            stats["output_tokens_incl_thinking"] = tout
            stats["mean_input_tokens"] = round(tin / len(rows)) if rows else None
            stats["usd_total"] = round(
                tin / 1e6 * GEMINI_USD_PER_1M_INPUT
                + tout / 1e6 * GEMINI_USD_PER_1M_OUTPUT, 5)
            stats["usd_per_page"] = round(stats["usd_total"] / len(rows), 6) if rows else None
        else:
            # Local OCR has no per-page fee; its cost is wall-clock, not dollars.
            stats["marginal_usd"] = 0.0
        per_backend[name] = stats

    summary = {
        "pdf": pdf_path.name,
        "pages": len(rows),
        "page_range": [rows[0]["page"], rows[-1]["page"]] if rows else None,
        "dpi": PDF_DPI,
        "raster_total_s": round(sum(raster), 3),
        "raster_mean_s": round(statistics.mean(raster), 3) if raster else None,
        "avg_megapixels": round(statistics.mean(r["megapixels"] for r in rows), 2) if rows else None,
        "avg_png_kb": round(statistics.mean(r["png_kb"] for r in rows)) if rows else None,
        "backends": per_backend,
    }

    if len(backend_names) == 2 and rows and "agreement" in rows[0]:
        ag = [r["agreement"] for r in rows]
        worst = sorted(rows, key=lambda r: r["agreement"])[:5]
        summary["agreement_mean"] = round(statistics.mean(ag), 4)
        summary["agreement_median"] = round(statistics.median(ag), 4)
        # The inspection queue: where the two engines disagree most is where at
        # least one of them is wrong, and it is the only place worth reading.
        summary["lowest_agreement_pages"] = [
            {"page": r["page"], "agreement": r["agreement"],
             **{n: r["backends"][n]["chars"] for n in backend_names}}
            for r in worst
        ]
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pdf", type=Path, default=Path("data/bhagya-bada-ya-karm.pdf"))
    parser.add_argument("--backend", choices=sorted(BACKENDS), default="tesseract")
    parser.add_argument("--compare", action="store_true",
                        help="Run tesseract AND vision on every page.")
    parser.add_argument("--pages", type=int, default=None,
                        help="Limit to the FIRST N pages (ignored if --start given).")
    parser.add_argument("--start", type=int, default=None, help="First page, 1-based inclusive.")
    parser.add_argument("--end", type=int, default=None, help="Last page, 1-based inclusive.")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()

    if args.start is not None:
        start = args.start
        end = args.end if args.end is not None else start
    else:
        start, end = 1, args.pages if args.pages is not None else 10**9

    names = ["tesseract", "vision"] if args.compare else [args.backend]
    report = run(args.pdf, names, start, end)

    tag = "compare" if args.compare else args.backend
    stem = args.pdf.stem[:28].replace(" ", "_")
    out = args.out or Path("evaluation") / f"ocr_{tag}_{stem}_{start}_{end}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, ensure_ascii=False)

    print("\nSummary")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
