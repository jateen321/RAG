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
# MEASURED, not assumed. The binding constraint is INPUT TOKENS per minute:
#   aiplatform.googleapis.com/embed_content_input_tokens_per_minute_per_base_model
# An earlier comment here claimed requests-per-minute. It was wrong, and the
# difference matters: under a request limit, bigger batches give proportional
# relief; under a TOKEN limit they give NONE, because 250 chunks carry exactly
# the tokens of 20 chunks x 12.5. Proof (Vertex express, gemini-embedding-001):
# a 250 x 1010-char request succeeded, and the NEXT request at 200 x 1010 --
# strictly smaller -- was rejected, because the first had drained the window.
#
# The only levers on this quota are throughput over time (see the adaptive pace
# below) and raising the quota itself. Batch size is now purely a throughput
# and overhead choice: 250 instances is the hard API ceiling (n=400 -> HTTP 400
# "too many instances"), and larger batches measured FASTER per chunk
# (23.8 chunks/s at n=250 vs 13.0 at n=50).
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

# Vertex embedding requests can rotate through independent regional quotas.
# This is active only with LLM_BACKEND=vertex and gemini-embedding-001.
# LLM_BACKEND=developer never sends generation or embeddings to Vertex.
EMBEDDING_REGION_ROTATION_ENABLED = (
    os.getenv("EMBEDDING_REGION_ROTATION_ENABLED", "1").strip().lower()
    not in {"0", "false", "no"}
)
VERTEX_EMBEDDING_PROJECT_ID = os.getenv(
    "VERTEX_EMBEDDING_PROJECT_ID", "cloudexplore-502215"
).strip()
VERTEX_EMBEDDING_REGIONS = tuple(
    region.strip()
    for region in os.getenv(
        "VERTEX_EMBEDDING_REGIONS",
        "asia-south1,asia-southeast1,asia-east1,asia-northeast1,"
        "us-west1,us-central1,us-east4,europe-west4,europe-west1",
    ).split(",")
    if region.strip()
)
if EMBEDDING_REGION_ROTATION_ENABLED and not VERTEX_EMBEDDING_REGIONS:
    raise ValueError("VERTEX_EMBEDDING_REGIONS must contain at least one region.")
VERTEX_EMBEDDING_TIMEOUT_S = float(
    os.getenv("VERTEX_EMBEDDING_TIMEOUT_S", "20")
)

# ── Retrieval Configuration ───────────────────────────────────────────
TOP_K = 5                 # Fixed cutoff used by explicit top_k/evaluation calls

# Query rewrites widen recall before a separate lightweight model reranks the
# fused candidates. Embeddings are batched, so rewrites still use one embedding
# request. Every model-backed stage has a deterministic fallback.
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "1").strip().lower() not in {
    "0", "false", "no"
}
QUERY_REWRITE_MAX_QUERIES = min(
    10, max(1, int(os.getenv("QUERY_REWRITE_MAX_QUERIES", "10")))
)
QUERY_RETRIEVAL_TOP_K = max(1, int(os.getenv("QUERY_RETRIEVAL_TOP_K", "5")))
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

# Firebase Authentication. The browser exchanges a Firebase ID token for an
# HttpOnly session cookie. Conversations remain keyed by verified Firebase UID,
# while every user and guest retrieves from this dedicated shared corpus.
FIREBASE_PROJECT_ID = os.getenv("FIREBASE_PROJECT_ID", "").strip()
SHARED_CORPUS_OWNER_ID = os.getenv(
    "SHARED_CORPUS_OWNER_ID", "__shared_corpus__"
).strip()
if not SHARED_CORPUS_OWNER_ID:
    raise ValueError("SHARED_CORPUS_OWNER_ID cannot be empty.")
SESSION_COOKIE_NAME = os.getenv("SESSION_COOKIE_NAME", "sarthi_session").strip()
SESSION_COOKIE_MAX_AGE_S = min(
    14 * 24 * 60 * 60,
    max(300, int(os.getenv("SESSION_COOKIE_MAX_AGE_S", str(5 * 24 * 60 * 60)))),
)
SESSION_COOKIE_SECURE = os.getenv("SESSION_COOKIE_SECURE", "1").strip().lower() not in {
    "0", "false", "no",
}
SESSION_COOKIE_SAMESITE = os.getenv("SESSION_COOKIE_SAMESITE", "lax").strip().lower()
if SESSION_COOKIE_SAMESITE not in {"lax", "strict", "none"}:
    raise ValueError("SESSION_COOKIE_SAMESITE must be lax, strict, or none.")
if SESSION_COOKIE_SAMESITE == "none" and not SESSION_COOKIE_SECURE:
    raise ValueError("SESSION_COOKIE_SECURE must be enabled when SameSite is none.")
AUTH_RECENT_SIGN_IN_MAX_AGE_S = max(
    60, int(os.getenv("AUTH_RECENT_SIGN_IN_MAX_AGE_S", "300"))
)
AUTH_CHECK_REVOKED = os.getenv("AUTH_CHECK_REVOKED", "0").strip().lower() not in {
    "0", "false", "no",
}
# Retained for conversation-row migration only. Pre-tenancy Chroma rows now
# belong to SHARED_CORPUS_OWNER_ID so guests can query the existing corpus.
LEGACY_ADMIN_UID = os.getenv("LEGACY_ADMIN_UID", "").strip()

# Server-local folders that POST /index/folder may read. The CLI does not need
# an allowlist because it already runs with the invoking user's permissions.
# Use the operating system path separator (`:` on macOS/Linux, `;` on Windows)
# to configure more than one root.
INDEX_FOLDER_ROOTS = [
    os.path.realpath(os.path.expanduser(root.strip()))
    for root in os.getenv("INDEX_FOLDER_ROOTS", DATA_DIR).split(os.pathsep)
    if root.strip()
]
