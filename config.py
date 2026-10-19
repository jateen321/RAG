"""
Configuration for the Gyaan Sarthi application.
Loads environment variables and defines constants.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# ── API Configuration ────────────────────────────────────────────────
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")   # Gemini Developer API (AIza...)

if not GEMINI_API_KEY or GEMINI_API_KEY == "your_api_key_here":
    print("❌ Error: GEMINI_API_KEY is not set in .env!")
    sys.exit(1)

# ── Model Configuration ──────────────────────────────────────────────
EMBEDDING_MODEL = "gemini-embedding-001"  # bare name (no "models/")
LLM_MODEL = "gemini-3.5-flash-lite"        # answer generation and query planning
RERANK_MODEL = os.getenv("RERANK_MODEL", "gemini-3.5-flash-lite").strip()
IMAGE_MODEL = os.getenv("IMAGE_MODEL", "gemini-3.1-flash-image").strip()

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
# Batch embedding reduces request overhead. Adaptive pacing responds to quota
# errors; batching does not eliminate token-based usage limits.
EMBED_BATCH_SIZE = 20       # Chunks per embed_content call (API max is 250)
EMBED_BATCH_DELAY_S = 1.0   # Pause between successive batches (proactive pacing)
EMBED_MAX_ATTEMPTS = 5      # Total tries per batch before giving up
EMBED_BACKOFF_BASE_S = 10   # First retry waits this; each retry doubles it
                            # (10 → 20 → 40 → 80s). A single fixed 30s wait was
                            # not enough to clear a per-minute window that had
                            # already been saturated.

# Adaptive inter-batch pacing. EMBED_BATCH_DELAY_S is the FLOOR; being throttled
# raises the working delay and sustained success decays it back toward the floor.
# A fixed delay cannot adapt: it is either too slow when quota is free, or too
# fast once the window is saturated -- and the old code reset to the floor after
# every batch, so a throttle taught it nothing.
EMBED_PACE_MAX_S = 30.0     # Ceiling for the adaptive delay
EMBED_PACE_DECAY_AFTER = 5  # Clean batches needed before easing the delay back

# ── Retrieval Configuration ───────────────────────────────────────────
TOP_K = 10                # Fixed cutoff used by explicit top_k/evaluation calls

# ── Adaptive retrieval routing ───────────────────────────────────────
# The router starts with one inexpensive direct vector search. These defaults
# are deliberately conservative and should be calibrated on evaluation data;
# environment variables make threshold sweeps reproducible without code edits.
ADAPTIVE_ROUTER_ENABLED = os.getenv("ADAPTIVE_ROUTER_ENABLED", "1").strip().lower() not in {
    "0", "false", "no"
}
ADAPTIVE_MAX_DISTANCE = float(os.getenv("ADAPTIVE_MAX_DISTANCE", "0.45"))
ADAPTIVE_MIN_MARGIN = float(os.getenv("ADAPTIVE_MIN_MARGIN", "0.05"))
ADAPTIVE_MIN_SOURCE_CONCENTRATION = float(
    os.getenv("ADAPTIVE_MIN_SOURCE_CONCENTRATION", "0.60")
)

# Query rewrites widen recall before a separate lightweight model reranks the
# fused candidates. Embeddings are batched, so rewrites still use one embedding
# request. Every model-backed stage has a deterministic fallback.
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no"
}
QUERY_REWRITE_MAX_QUERIES = min(
    10, max(1, int(os.getenv("QUERY_REWRITE_MAX_QUERIES", "10")))
)
QUERY_RETRIEVAL_TOP_K = max(1, int(os.getenv("QUERY_RETRIEVAL_TOP_K", "10")))
RRF_RANK_CONSTANT = max(1, int(os.getenv("RRF_RANK_CONSTANT", "60")))
RERANK_CANDIDATE_LIMIT = max(1, int(os.getenv("RERANK_CANDIDATE_LIMIT", "15")))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").strip().lower() not in {
    "0", "false", "no"
}
MIN_CONTEXT_CHUNKS = max(1, int(os.getenv("MIN_CONTEXT_CHUNKS", "5")))
MAX_CONTEXT_CHUNKS = max(
    MIN_CONTEXT_CHUNKS, int(os.getenv("MAX_CONTEXT_CHUNKS", "15"))
)
NEAR_DUPLICATE_OVERLAP = min(
    1.0, max(0.5, float(os.getenv("NEAR_DUPLICATE_OVERLAP", "0.85")))
)

# ── ChromaDB Configuration ────────────────────────────────────────────
CHROMA_DB_PATH = os.path.join(os.path.dirname(__file__), "chroma_db")
COLLECTION_NAME = "hindi_textbook"

# ── OCR Configuration ────────────────────────────────────────────────
OCR_BACKEND = os.getenv("OCR_BACKEND", "tesseract").strip().lower()
# Tesseract is local and free. Gemini OCR remains available for pages where the
# configured Gemini Developer API is worth its token cost.
OCR_BACKENDS = ("tesseract", "gemini")
if OCR_BACKEND not in OCR_BACKENDS:
    raise ValueError(
        f"OCR_BACKEND must be one of {OCR_BACKENDS} (got {OCR_BACKEND!r})."
    )
# ── OCR request retry ─────────────────────────────────────────────────
# OCR providers can fail transiently. Without retry ONE blip can abort a large
# document index, so mirror indexer.py's embedding pacing.
# ── OCR concurrency ───────────────────────────────────────────────────
# Only the OCR call is parallelised; rasterization stays on the main thread
# because PyMuPDF's Document is not thread-safe.
#
# Worker counts are PER BACKEND: Gemini requests are quota-limited, while
# Tesseract shells out locally and benefits from a few CPU-bound workers.
_DEFAULT_OCR_WORKERS = {
    "gemini": 1,
    "tesseract": min(4, os.cpu_count() or 1),
}
# 0 / unset → use the per-backend default above.
OCR_MAX_WORKERS = int(os.getenv("OCR_MAX_WORKERS", "0")) or _DEFAULT_OCR_WORKERS[OCR_BACKEND]

OCR_MAX_ATTEMPTS = 4        # total tries per page
OCR_BACKOFF_BASE_S = 2      # first retry waits this; each retry doubles (2→4→8)

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

# ── OCR result cache ──────────────────────────────────────────────────
# OCR is the expensive, slow, BILLED half of ingestion; embedding is the half
# that fails on quota. Without a cache between them, one 429 at embed time
# discards every page already OCR'd for that document — for Arthasastra that is
# ~14 minutes and ~$1.35 of Vision units, re-spent on every retry.
OCR_CACHE_DIR = os.path.join(os.path.dirname(__file__), "evaluation", "ocr_cache")
OCR_CACHE_ENABLED = os.getenv("OCR_CACHE_ENABLED", "1").strip().lower() not in {"0", "false", "no"}
CONVERSATION_DB_PATH = os.getenv(
    "CONVERSATION_DB_PATH",
    os.path.join(os.path.dirname(__file__), "data", "conversations.sqlite3"),
)
GENERATED_IMAGE_DIR = os.getenv(
    "GENERATED_IMAGE_DIR",
    os.path.join(DATA_DIR, "generated_images"),
)

# One local account owns conversations and privileged routes. Restrict network
# access at the VM firewall because the app does not perform hosted sign-in.
SINGLE_USER_ID = os.getenv("SINGLE_USER_ID", "single-user").strip()
if not SINGLE_USER_ID:
    raise ValueError("SINGLE_USER_ID cannot be empty.")
SHARED_CORPUS_OWNER_ID = os.getenv(
    "SHARED_CORPUS_OWNER_ID", "__shared_corpus__"
).strip()
if not SHARED_CORPUS_OWNER_ID:
    raise ValueError("SHARED_CORPUS_OWNER_ID cannot be empty.")

# Server-local folders that POST /index/folder may read. The CLI does not need
# an allowlist because it already runs with the invoking user's permissions.
# Use the operating system path separator (`:` on macOS/Linux, `;` on Windows)
# to configure more than one root.
INDEX_FOLDER_ROOTS = [
    os.path.realpath(os.path.expanduser(root.strip()))
    for root in os.getenv("INDEX_FOLDER_ROOTS", DATA_DIR).split(os.pathsep)
    if root.strip()
]
