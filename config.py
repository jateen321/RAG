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
MIN_CHUNK_LENGTH = 50     # Skip chunks shorter than this

# ── Retrieval Configuration ───────────────────────────────────────────
TOP_K = 5                 # Number of chunks to retrieve per query

# ── ChromaDB Configuration ────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hindi_textbook"

# ── OCR Configuration ────────────────────────────────────────────────
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
