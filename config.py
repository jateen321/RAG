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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("❌ Error: GEMINI_API_KEY not set!")
    print("   1. Copy .env.example to .env")
    print("   2. Add your free API key from https://aistudio.google.com/")
    sys.exit(1)

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "models/gemini-embedding-001"  # Free Gemini embedding model
LLM_MODEL = "gemini-2.0-flash-lite"             # Free Gemini Flash Lite (best for free tier)

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
PDF_DPI = 200                       # Resolution for PDF to image conversion

# ── Text-layer spot check (language-agnostic corruption defense) ──────
# Some PDFs have a broken font→Unicode map: the page LOOKS fine but get_text()
# returns scrambled codepoints. We OCR a few 'direct' pages and compare; if the
# text layer disagrees strongly with OCR, we distrust it and OCR the document.
LAYER_CHECK_SAMPLE = 3              # pages to spot-check per document
LAYER_CHECK_MIN_SIMILARITY = 0.4   # median OCR-vs-layer similarity below this → distrust layer

# ── Paths ─────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
