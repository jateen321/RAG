"""
OCR Engine — Extracts Hindi + English text from scanned PDF files.

Uses PyMuPDF to convert PDF pages to images, then EasyOCR for text extraction.
"""

import fitz  # PyMuPDF
import easyocr
import numpy as np
from PIL import Image
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.console import Console

from config import OCR_LANGUAGES, PDF_DPI

console = Console()

# Initialize EasyOCR reader (loaded once, reused)
_reader = None


def _get_reader():
    """Lazily initialize the EasyOCR reader (downloads model on first use)."""
    global _reader
    if _reader is None:
        console.print("[yellow]⏳ Loading OCR model (first time may download ~100MB)...[/yellow]")
        _reader = easyocr.Reader(OCR_LANGUAGES, gpu=False)
        console.print("[green]✅ OCR model loaded![/green]")
    return _reader


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """
    Extract text from a scanned PDF file.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dicts with keys: 'page', 'text'
        Example: [{'page': 1, 'text': 'पाठ 1: भारत का इतिहास...'}]
    """
    reader = _get_reader()
    doc = fitz.open(pdf_path)
    total_pages = len(doc)

    console.print(f"\n📄 Processing: [bold]{pdf_path}[/bold] ({total_pages} pages)")

    pages_text = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("({task.completed}/{task.total})"),
        console=console,
    ) as progress:
        task = progress.add_task("🔍 OCR in progress", total=total_pages)

        for page_num in range(total_pages):
            page = doc[page_num]

            # Convert page to image at specified DPI
            mat = fitz.Matrix(PDF_DPI / 72, PDF_DPI / 72)
            pix = page.get_pixmap(matrix=mat)

            # Convert to numpy array for EasyOCR
            img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(
                pix.height, pix.width, pix.n
            )

            # If RGBA, convert to RGB
            if pix.n == 4:
                img = img[:, :, :3]

            # Run OCR
            results = reader.readtext(img, detail=0, paragraph=True)
            text = "\n".join(results).strip()

            if text:
                pages_text.append({
                    "page": page_num + 1,
                    "text": text,
                })

            progress.update(task, advance=1)

    doc.close()

    # Summary
    pages_with_text = len(pages_text)
    total_chars = sum(len(p["text"]) for p in pages_text)
    console.print(f"\n[green]✅ OCR Complete![/green]")
    console.print(f"   📊 {pages_with_text}/{total_pages} pages had text")
    console.print(f"   📝 {total_chars:,} characters extracted")

    return pages_text
