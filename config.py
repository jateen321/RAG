"""
Configuration for the Hindi Textbook RAG Application.
Loads environment variables and defines constants.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Configuration ────────────────────────────────────────────────
# Two separately-named keys so we can hold both and switch between them:
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # Gemini Developer API (AIza..., free-tier fallback)
VERTEX_API_KEY = os.getenv("VERTEX_API_KEY")   # Agent Platform / Vertex express mode (AQ...)

# Which backend to route calls through:
#   "vertex"    → Agent Platform / Vertex (express mode, VERTEX_API_KEY)  [primary]
#   "developer" → Gemini Developer API (GEMINI_API_KEY)                   [fallback]
LLM_BACKEND = os.getenv("LLM_BACKEND", "developer").strip().lower()
GCP_LOCATION = os.getenv("GCP_LOCATION", "global")   # region for Vertex/express

if LLM_BACKEND == "vertex":
    if not VERTEX_API_KEY:
        print("❌ Error: LLM_BACKEND=vertex but VERTEX_API_KEY is not set in .env!")
        sys.exit(1)
elif LLM_BACKEND == "developer":
    if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
        print("❌ Error: LLM_BACKEND=developer but GEMINI_API_KEY is not set in .env!")
        sys.exit(1)
else:
    print(f"❌ Error: LLM_BACKEND must be 'vertex' or 'developer' (got '{LLM_BACKEND}').")
    sys.exit(1)

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"  # bare name (no "models/"): works on Developer API AND Vertex/express
LLM_MODEL = "gemini-2.5-flash"            # bare name; free-tier GA (2.x-flash-lite is closed to new keys)

# ── Chunking Configuration ────────────────────────────────────────────
CHUNK_SIZE = 800          # Target characters per chunk (soft cap: a chunk may
                          # slightly overflow to avoid splitting a sentence)
CHUNK_OVERLAP = 100       # Minimum overlap between chunks (chars). Filled with
                          # WHOLE trailing sentences until this budget is met, so
                          # the overlap is boundary-clean AND a real bridge.
MAX_CHUNK_OVERLAP = 250   # Maximum overlap (chars). Without a ceiling, a single
                          # long trailing sentence satisfies CHUNK_OVERLAP on the
                          # first carry-back and gets copied whole — observed
                          # producing chunks 95% identical to their neighbour,
                          # which then occupy two top-k slots with one idea.
                          # The ceiling WINS over CHUNK_OVERLAP: a short bridge
                          # is cheaper than a duplicate chunk.
MIN_CHUNK_LENGTH = 50     # Skip chunks shorter than this

# ── YouTube transcript chunking ──────────────────────────────────────
# Spoken content has a temporal shape that PDF text does not. A character-only
# limit can make a fast speaker's chunk too long in time and a slow speaker's
# chunk too short in meaning, so YouTube chunks stop at whichever soft target
# is reached first while retaining hard ceilings for both dimensions.
YOUTUBE_CHUNK_TARGET_CHARS = int(os.getenv("YOUTUBE_CHUNK_TARGET_CHARS", "800"))
YOUTUBE_CHUNK_MAX_CHARS = int(os.getenv("YOUTUBE_CHUNK_MAX_CHARS", "1200"))
YOUTUBE_CHUNK_TARGET_SECONDS = float(
    os.getenv("YOUTUBE_CHUNK_TARGET_SECONDS", "75")
)
YOUTUBE_CHUNK_MAX_SECONDS = float(os.getenv("YOUTUBE_CHUNK_MAX_SECONDS", "120"))
YOUTUBE_CHUNK_OVERLAP_SECONDS = float(
    os.getenv("YOUTUBE_CHUNK_OVERLAP_SECONDS", "12")
)

# ── Embedding request pacing ──────────────────────────────────────────
# The binding constraint on a bulk index is requests-per-MINUTE, not total
# volume: a 1877-page corpus is ~72 embedding calls, and firing them
# back-to-back trips the quota long before the day's allowance is touched.
EMBED_BATCH_SIZE = 20       # Chunks per embed_content call
EMBED_BATCH_DELAY_S = 1.0   # Pause between successive batches (proactive pacing)
EMBED_MAX_ATTEMPTS = 5      # Total tries per batch before giving up
EMBED_BACKOFF_BASE_S = 10   # First retry waits this; each retry doubles it
                            # (10 → 20 → 40 → 80s). A single fixed 30s wait was
                            # not enough to clear a per-minute window that had
                            # already been saturated.

# ── Retrieval Configuration ───────────────────────────────────────────
TOP_K = 5                 # Number of chunks to retrieve per query

# ── ChromaDB Configuration ────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hindi_textbook"

# ── OCR Configuration ────────────────────────────────────────────────
OCR_BACKEND = os.getenv("OCR_BACKEND", "google_vision").strip().lower()
# Three backends, measured 2026-08-25 over 90 pages (see OCR_NOTES.md):
#   "google_vision" — hosted dense-document OCR. Best accuracy/cost balance and
#                     the only one that never failed a page. ~$1.50/1000 pages.
#   "tesseract"     — fully local, $0, no network. Lower Devanagari accuracy and
#                     it CRASHED on 3 of 90 pages, returning nothing at all.
#   "gemini"        — multimodal LLM. Accuracy comparable to Vision, ~1.3x its
#                     cost, but it SILENTLY OMITS regions it cannot read rather
#                     than flagging them. Quiet data loss; not a safe default.
OCR_BACKENDS = ("google_vision", "tesseract", "gemini")
if OCR_BACKEND not in OCR_BACKENDS:
    raise ValueError(
        f"OCR_BACKEND must be one of {OCR_BACKENDS} (got {OCR_BACKEND!r})."
    )
# ── OCR request retry ─────────────────────────────────────────────────
# Hosted OCR fails transiently: a 503 "service is currently unavailable" was
# observed mid-run on 2026-08-25, and Vertex returns 429 RESOURCE_EXHAUSTED
# under per-minute quota. Without retry ONE blip aborts a whole 1877-page
# index, so mirror indexer.py's embedding pacing.
# ── OCR concurrency ───────────────────────────────────────────────────
# Only the NETWORK call is parallelised; rasterization stays on the main thread
# because PyMuPDF's Document is not thread-safe. That split works because the
# network dominates: ~0.25s to rasterize vs ~2.6s for a Vision round-trip.
#
# Worker counts are PER BACKEND, not one global number, because their limits
# differ by an order of magnitude:
#   google_vision  quota is generous; 8 workers ≈ 185 req/min, well inside it.
#   gemini         Vertex express 429'd at SEQUENTIAL rates in benchmarking, so
#                  parallelism mostly buys retries. Keep it near 1.
#   tesseract      pytesseract shells out, so threads do help, but past core
#                  count they just thrash.
_DEFAULT_OCR_WORKERS = {
    "google_vision": 8,
    "gemini": 2,
    "tesseract": min(4, os.cpu_count() or 1),
}
# 0 / unset → use the per-backend default above.
OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "0")) or _DEFAULT_OCR_WORKERS[OCR_BACKEND]

OCR_MAX_ATTEMPTS = 4        # total tries per page
OCR_BACKOFF_BASE_S = 2      # first retry waits this; each retry doubles (2→4→8)

GOOGLE_VISION_LANGUAGE_HINTS = [
    language.strip()
    for language in os.getenv("GOOGLE_VISION_LANGUAGE_HINTS", "hi,sa,en").split(",")
    if language.strip()
]
OCR_LANGUAGES = ["hi", "en"]        # (legacy: EasyOCR format)
TESSERACT_LANG = "eng+hin+san"      # Tesseract format: English + Hindi + Sanskrit
PDF_DPI = 300                       # Resolution for PDF to image conversion
                                    # 300 is Tesseract's recommended minimum for
                                    # small/complex glyphs (Devanagari matras).
                                    # Costs ~2.25x the pixels of 200 → slower OCR.

# ── Text-layer spot check (language-agnostic corruption defense) ──────
# Some PDFs have a broken font→Unicode map: the page LOOKS fine but get_text()
# returns scrambled codepoints. We OCR a few 'direct' pages and compare; if the
# text layer disagrees strongly with OCR, we distrust it and OCR the document.
LAYER_CHECK_SAMPLE = 3              # pages to spot-check per document
LAYER_CHECK_MIN_SIMILARITY = 0.4   # median OCR-vs-layer similarity below this → distrust layer

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

# Server-local folders that POST /index/folder may read. The CLI does not need
# an allowlist because it already runs with the invoking user's permissions.
# Use the operating system path separator (`:` on macOS/Linux, `;` on Windows)
# to configure more than one root.
INDEX_FOLDER_ROOTS = [
    os.path.realpath(os.path.expanduser(root.strip()))
    for root in os.getenv("INDEX_FOLDER_ROOTS", DATA_DIR).split(os.pathsep)
    if root.strip()
]
