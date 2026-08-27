
# 📚 Sarthi AI

Query documents and YouTube transcripts in Hindi or English with an adaptive
extraction and vector-search pipeline, then use Gemini to generate grounded answers.

Uses **Google Gemini** for query planning, embeddings, reranking, and answers.
PDF pages with clean text layers are read directly; scanned or garbled pages use
the configured OCR backend: **Google Cloud Vision** (default), **Tesseract**, or
**Gemini multimodal OCR**. No LangChain — the pipeline is hand-rolled.

ChromaDB runs locally. OCR and Gemini cost and availability depend on the selected
backends, models, billing status, and project-specific quotas; do not assume every
configuration is free.

## 🚀 Quick Start

### 1. Choose an OCR backend

The default is Google Cloud Vision. Authenticate it with Application Default
Credentials after installing and initializing the Google Cloud CLI:

```bash
gcloud auth application-default login
```

For fully local OCR, set `OCR_BACKEND=tesseract` in `.env`. Tesseract is a native
binary, so install it separately with the Hindi/Sanskrit language data:

```bash
brew install tesseract tesseract-lang
```

> 🐧 On Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-san`
>
> Gemini OCR uses the API backend and key selected by `LLM_BACKEND`; it needs no
> separate OCR binary or Google Cloud Vision credentials.

### 2. Create a Virtual Environment

Keep this project's packages isolated from your system Python:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

> 🪟 On Windows: `.venv\Scripts\activate`

Built with **Python 3.14**. Your shell prompt shows `(.venv)` once it's active — the
remaining steps assume it is. Run `deactivate` to leave.

> 💡 `.venv/` is gitignored. It's disposable: delete it and rebuild from
> `requirements.txt` any time something gets tangled.

### 3. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 4. Set Up Your API Key

This project talks to Gemini through **one of two backends** — pick with
`LLM_BACKEND`. The selected backend is shared by **both embedding calls and
answer generation**:

| `LLM_BACKEND` | Backend | Key used | Key looks like |
|---|---|---|---|
| `developer` *(default)* | Gemini Developer API — free tier | `GEMINI_API_KEY` | `AIza…` |
| `vertex` | Agent Platform / Vertex, express mode | `VERTEX_API_KEY` | `AQ.…` |

For Developer API experimentation, get a key from
[Google AI Studio](https://aistudio.google.com/apikey), then:

```bash
cp .env.example .env
```

Edit `.env`:

```
LLM_BACKEND=developer
GEMINI_API_KEY=your_actual_key_here
```

> 💡 You only need the key for the backend you selected. Switching the API backend is
> a one-line change in `.env`, but model availability, quotas, and billing can differ
> between backends.

### 5. Index a PDF

Drop your Hindi textbook PDF into the `data/` folder. Run the interactive picker:

```bash
python app.py index
```

Or provide a specific PDF:

```bash
python app.py index data/your_textbook.pdf
```

This will:
- Decide **per page** whether to read the embedded text layer or run OCR
- Split into boundary-aware, searchable chunks
- Create embeddings and store them in ChromaDB

### 5b. Index a YouTube video or playlist

No YouTube API key is required, and the application does not download media:

```bash
python app.py index-youtube "https://www.youtube.com/watch?v=VIDEO_ID"
python app.py index-youtube "https://www.youtube.com/playlist?list=PLAYLIST_ID"
```

For playlists, accessible videos are indexed independently. Videos that are
private, unavailable, or have no transcript are skipped and reported without
discarding successful videos. Transcript selection prefers manually created
captions over auto-generated captions, with Hindi then English preferred within
each category.

Each transcript chunk stores its start/end timestamps, video ID and title,
channel, source URL, transcript language/type, and playlist identity/position
when applicable.

### 6. Ask Questions!

**One-shot question** — ask in English or Hindi, about whatever you indexed:

```bash
# English
python app.py ask "Where is Coal India Limited's corporate headquarters?"

# Hindi
python app.py ask "भारत के कुल कोयला उत्पादन में CIL का लगभग कितना योगदान है?"
```

The application attempts to ground each answer in retrieved chunks. The console
prints their source files and pages so you can verify the answer against the
original PDF.

**Interactive chat:**
```bash
python app.py chat
```

## 📋 CLI Commands

| Command | Description |
|---|---|
| `python app.py index` | Pick a PDF from `data/` and index it |
| `python app.py index <pdf>` | Index a PDF for searching |
| `python app.py index-folder <folder>` | Recursively index PDF, TXT, and Markdown files |
| `python app.py index-youtube <url>` | Index a YouTube video or playlist transcript |
| `python app.py ask "question"` | Ask a one-shot question |
| `python app.py chat` | Start interactive chat |
| `python app.py status` | Show indexed documents, pages, and chunk counts |
| `python app.py remove "<source>"` | Delete all chunks belonging to one indexed PDF |
| `python app.py reset` | Clear all indexed data |

Use the exact source name shown by `status` when removing one document:

```bash
python app.py status
python app.py remove "CIL.pdf"
```

`remove` preserves other documents. `reset` deletes the complete local
`chroma_db/` directory. Neither command deletes the original documents, but
deleted vectors must be regenerated by indexing the source files again.

## 🌐 Web API

The same pipeline is exposed over HTTP with FastAPI:

```bash
uvicorn api:app --reload
```

Then open **http://127.0.0.1:8000/docs** for interactive Swagger docs.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Status + index statistics |
| `POST` | `/ask` | Ask a grounded question — returns the answer **and its sources** |
| `POST` | `/index` | Index a PDF, TXT, or Markdown file already in `data/` |
| `POST` | `/index/folder` | Recursively index an allowlisted server-local folder |
| `POST` | `/upload` | Upload and index a PDF, TXT, or Markdown file |
| `POST` | `/index/youtube` | Index a YouTube video or playlist URL |

```bash
curl -X POST http://127.0.0.1:8000/index/youtube \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.youtube.com/watch?v=VIDEO_ID"}'
```

Folder indexing accepts server-local paths and is recursive by default:

```bash
curl -X POST http://127.0.0.1:8000/index/folder \
  -H "Content-Type: application/json" \
  -d '{"folder_path":"/path/to/project/data","recursive":true}'
```

For security, the API rejects folders outside `INDEX_FOLDER_ROOTS`. Configure
one or more roots in `.env`, separated by `:` on macOS/Linux or `;` on Windows.
The CLI command uses the invoking user's filesystem permissions and does not
need this allowlist:

```bash
python app.py index-folder "/path/to/local/documents"
```

```bash
curl -X POST http://127.0.0.1:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "Where is Coal India Limited headquartered?"}'
```

Responses carry the answer plus the chunks used as grounding context (page,
source, distance, and preview).

## 🖥️ React Web Interface

The `frontend/` application provides a student-friendly interface for chat,
source citations, document uploads, YouTube indexing, and library status.

### Why does the application use two ports?

During development, the frontend and backend run as separate servers:

- **Port 3000** runs the React frontend, which renders the user interface and
  provides development features such as fast refresh.
- **Port 8000** runs the Python FastAPI backend (not Flask), which handles the
  RAG pipeline, Gemini requests, ChromaDB, uploads, and API endpoints.

Keeping the servers separate makes it possible to develop and restart either
part independently. The browser opens the frontend on
`http://localhost:3000`, and the frontend sends API requests to
`http://127.0.0.1:8000`.

A production deployment can expose both parts through one public port or
domain. Common approaches are to serve the built frontend from FastAPI or use
a reverse proxy that routes `/` to the frontend and `/api/*` to FastAPI.

Start the API in one terminal:

```bash
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000
```

Start the React interface in another terminal:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open **http://localhost:3000**. The frontend connects to
`http://127.0.0.1:8000` by default. Copy `frontend/.env.example` to
`frontend/.env.local` to configure a different API or site URL.

PDF, UTF-8 TXT, and UTF-8 Markdown uploads are limited to 500 MB, saved inside
`data/`. A document already indexed in ChromaDB is rejected; when the file
already exists in `data/` but is not indexed, that local copy is indexed without
being overwritten. Set
`RAG_ALLOWED_ORIGINS` in `.env` if
the frontend runs on a different origin.

**Status codes:** `400` for a bad request, `503` when nothing is indexed yet or the Gemini
quota is exhausted — a rate limit upstream becomes a "try again later" downstream, never a 500.

## 🏗️ How It Works

```
                    ┌─ clean text layer ──→ use it directly (fast, no OCR)
PDF page ──routing ─┤
                    └─ scanned / garbled ─→ rasterize → configured OCR backend
                                                     ↓
                        Boundary-aware chunks → Gemini Embeddings* → ChromaDB
                                                                        ↓
Your Question → Gemini query rewrites → batched vector search → RRF fusion
                                                                        ↓
                         Gemini reranking → Top 5 → Gemini + Context → Answer!
```

YouTube follows a parallel ingestion path: `yt-dlp` reads video/playlist
metadata, `youtube-transcript-api` retrieves timestamped captions, and the
resulting chunks enter the same Gemini embedding and ChromaDB pipeline. Answers
cite transcript timestamps instead of PDF page numbers.

`*` Query planning, embeddings, reranking, and answer generation use the backend
selected by `LLM_BACKEND`.

**Two defenses worth knowing about:**
- **Per-page routing** — pages with a usable text layer skip OCR entirely, so indexing is much faster on mixed PDFs.
- **Corrupt-layer detection** — some PDFs render fine but return scrambled characters from a broken font map. The indexer OCRs a few sample pages and compares them to the text layer; if they disagree, it distrusts the layer and OCRs the whole document.

### Retrieval behavior and current limitations

- Gemini keeps the original question and can generate up to nine distinct rewrites.
- All query embeddings are sent in one batch. ChromaDB retrieves five candidates
  per query from the shared `hindi_textbook` collection.
- Stable chunk IDs deduplicate candidates; reciprocal rank fusion (RRF) combines
  their rankings into a deterministic shortlist.
- Gemini listwise-reranks up to 15 candidates and returns the final `TOP_K=5`.
  If planning or reranking fails, the pipeline falls back to the original query
  or the RRF order respectively.
- The five complete chunk texts are inserted into the answer prompt; the terminal
  displays only short previews.
- There is no minimum relevance threshold or source/PDF filter. A weak or general
  question can therefore still retrieve unrelated chunks.
- All indexed PDFs share one collection. Use `remove` to delete one source or
  `reset` to rebuild the complete index.

## 🧪 Evaluation

Retrieval quality is measured separately from answer quality — an end-to-end score
cannot tell you whether a bad answer came from bad retrieval or bad generation.

```bash
python evaluate.py                      # retrieval only; --top-k defaults to 5
python evaluate.py --top-k 10           # sweep k to see where recall saturates
python evaluate.py --output evaluation/results_myrun.json
python evaluate.py --generate           # ALSO generate answers (costs LLM calls)
```

> ⚠️ `evaluate.py` does **not** read `TOP_K` from `config.py` — its `--top-k` default
> is hardcoded to 5. Changing `config.TOP_K` alters the app's behaviour but not the
> harness's, so pass `--top-k` explicitly when comparing the two.

| Metric | Meaning |
|---|---|
| `retrieval_hit_rate` | Fraction of questions where **some** retrieved chunk was the expected source *and* page |
| `mean_reciprocal_rank` | Average of `1 / rank_of_first_correct_chunk` — rewards putting the answer at rank 1, not just somewhere in the top k |
| `mean_source_precision` | Fraction of the top-k drawn from the **expected document** — catches wrong-book contamination that hit rate and MRR structurally cannot see, because both stop counting at the first correct chunk |
| `hit_rate_easy` / `hit_rate_hard` | Same metric split by question tier — see the dataset note below |
| `mrr_*`, `source_precision_*` | Per-tier and per-language variants of the two metrics above |
| `citation_accuracy`, `average_keyword_recall` | Generation metrics, only with `--generate` |
| `refusal_rate_on_unanswerable`, `false_refusal_rate` | Generation metrics. Read them as a **pair**: a prompt that refuses everything scores well on the first alone |

**Dataset:** `evaluation/questions_v2.json` — 52 questions across the whole corpus.

| tier | n | what it is for |
|---|---|---|
| easy | 19 | Authored from a distinctive passage. Names its own document and often repeats its answer keywords — an **upper bound**, not a grade |
| hard | 19 | The *same* ground truth with that leakage removed; only the wording differs, so the easy→hard delta is attributable to phrasing |
| unanswerable | 14 | No answer exists in the corpus. Tests refusal, and the retriever's distance signal |

Every answerable question carries an `evidence` snippet that must be present in a chunk
at the stated source and page, so ground truth is auditable rather than written from
memory of the book. Two scripts enforce this:

```bash
python evaluation/verify_questions.py    # evidence really is at that source/page;
                                         # unanswerable questions really have no support
python evaluation/check_leakage.py       # does a question give away its own answer?
```

`check_leakage.py` scores two leak channels — a token shared with the question's own
`answer_keywords`, and naming the source document. It currently reports **14 of 19**
easy questions leaking and **0 of 19** hard. Run it after adding any question; a new
question that leaks will inflate the hit rate without improving the system.

**Latest retrieval run** (k=5, 52 questions, 2026-08-27 — retrieval only, no generation):

| metric | easy | hard |
|---|---|---|
| `retrieval_hit_rate` | 1.0 | 0.947 |
| `mean_reciprocal_rank` | 0.961 | 0.825 |
| `mean_source_precision` | 0.842 | 0.779 |

Read the easy column as a ceiling and the gap as the real signal. `source_precision`
falls further than hit rate does, because dropping the document name from a question is
exactly what stops the retriever telling the books apart. The single hard-tier miss
retrieved the correct document at the wrong page, so hit rate is 1.0 at *source*
granularity and 0.947 at *page* granularity.

> ⚠️ **Answer quality is unmeasured.** `--generate` has never been run, so there are no
> citation, keyword-recall, or refusal numbers. Retrieval being sound says nothing about
> whether the model fabricates — and this corpus is mostly famous texts the model already
> knows, which is precisely the risk `--generate` would test.

Measured results, and which of them are reproducible from this repository versus
carried over from a separate session, are tracked in **[FINDINGS.md](FINDINGS.md)**.
Text-extraction investigations live in **[OCR_NOTES.md](OCR_NOTES.md)**.

> 💡 `results_*.json` files are written into `evaluation/` locally and are **not**
> committed, so a fresh clone reproduces numbers by re-running the harness — which
> requires an indexed corpus and a working API key.

## 📁 Project Structure

```
RAG/
├── app.py              # CLI interface (main entry point)
├── api.py              # FastAPI web interface (/ask, /index, /health)
├── config.py           # Configuration & constants
├── llm_client.py       # Gemini client factory (Developer API vs Vertex)
├── ocr_engine.py       # PDF → text, per-page routing + selectable OCR
├── text_quality.py     # Scores the text layer to pick direct vs OCR
├── indexer.py          # Text → chunks → embeddings → ChromaDB
├── youtube_ingester.py # YouTube metadata + transcript → timestamped chunks
├── retriever.py        # Semantic search in ChromaDB
├── rag_engine.py       # Retrieve + Generate answers
├── evaluate.py         # Retrieval evaluation harness (hit rate, MRR, source precision)
├── requirements.txt    # Python dependencies (direct only)
├── .env                # Your API key (private, not in git)
├── .env.example        # Template for .env
├── FINDINGS.md         # Measured results + provenance tags (verified / cloud / hypothesis)
├── OCR_NOTES.md        # Text-extraction issues log: routing, legacy fonts, corrupt layers
├── CLAUDE.md           # Working conventions for AI-assisted sessions on this repo
├── .claude/
│   └── settings.json   # Repo-local hooks (uncommitted-file + README-staleness reminders)
├── evaluation/
│   ├── questions_v2.json      # Eval dataset (easy / hard / unanswerable tiers)
│   ├── verify_questions.py    # Audits ground truth against what is actually indexed
│   └── check_leakage.py       # Flags questions that give away their own answer
│                              # results_*.json are written locally, not committed
├── data/               # Drop your PDFs here
└── chroma_db/          # Vector database (auto-created)
```

## ⚙️ Configuration

Tune these settings in `config.py`; environment-backed options can be overridden
in `.env` as shown in `.env.example`:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Target characters per chunk (soft cap — respects boundaries) |
| `CHUNK_OVERLAP` | 100 | **Minimum** overlap between chunks (floor) |
| `MAX_CHUNK_OVERLAP` | 250 | **Maximum** overlap between chunks (ceiling). Takes precedence over the floor — prevents one long sentence being copied whole into the next chunk |
| `MIN_CHUNK_LENGTH` | 50 | Skip chunks shorter than this |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `QUERY_REWRITE_MAX_QUERIES` | 10 | Maximum total queries, including the original |
| `QUERY_RETRIEVAL_TOP_K` | 5 | Candidates retrieved per query before fusion |
| `RERANK_CANDIDATE_LIMIT` | 15 | Maximum RRF candidates sent to Gemini reranking |
| `OCR_BACKEND` | `google_vision` | OCR implementation: `google_vision`, `tesseract`, or `gemini` |
| `OCR_MAX_WORKERS` | backend-specific | Concurrent OCR calls/processes; defaults to Vision 8, Gemini 2, Tesseract up to 4 |
| `PDF_DPI` | 300 | Resolution used when rasterizing pages for any OCR backend |
| `GOOGLE_VISION_LANGUAGE_HINTS` | `hi,sa,en` | Soft language hints for Google Cloud Vision |
| `TESSERACT_LANG` | `eng+hin+san` | OCR languages (English + Hindi + Sanskrit) |
| `LAYER_CHECK_SAMPLE` | 3 | Pages OCR'd per document to spot-check the text layer |
| `LAYER_CHECK_MIN_SIMILARITY` | 0.4 | Median OCR-vs-layer similarity below this → distrust the layer and OCR the whole document |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `LLM_MODEL` | `gemini-2.5-flash` | Generation model |

> **Embedding model lifecycle:** This application currently uses
> `gemini-embedding-001`, which remains available for text-only workloads.
> Google's live deprecation schedule lists May 14, 2028 as its earliest shutdown
> date and recommends `gemini-embedding-2` as the replacement. Changing embedding
> models requires rebuilding the ChromaDB index because vectors produced by
> different models are not interchangeable. See Google's
> [deprecation schedule](https://ai.google.dev/gemini-api/docs/deprecations?hl=en)
> and [embeddings guide](https://ai.google.dev/gemini-api/docs/embeddings?hl=en).

## 💡 Tips

- **OCR resolution trade-off**: `PDF_DPI` defaults to 300. Pixel count scales with the
  *square* of DPI, so 300 costs ~2.25× the OCR time of 200. Drop to 200 if your
  scans are clean and indexing is too slow; validate quality before changing it.
- **Changing OCR settings requires a re-index**: current position-based chunk IDs
  and `upsert` replace existing positions safely, and stale trailing positions are
  removed only after the new document is stored successfully.
- **Index multiple books**: Run `index` on multiple PDFs — they all go into the same database.
- **Ask in any language**: Questions can be in Hindi, English, or mixed.
- **Rate limits vary**: Limits depend on the backend, model, region, billing tier,
  and Google Cloud project. Check the quota assigned to your credential instead
  of assuming a fixed requests-per-minute value.
- **Embedding quota handling**: indexing batches embeddings, spaces calls with an
  adaptive shared delay, honors server retry hints, and retries transient quota
  failures with exponential backoff. Daily quota exhaustion fails fast because
  waiting seconds cannot clear a daily window. Query rewrites are embedded in one
  batch to avoid multiplying request pressure.
- **Model migration**: Do not query existing vectors with a different embedding
  model, even when both models output the same number of dimensions. Create a
  new collection or reset and reindex every PDF.

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `LLM_BACKEND=developer but GEMINI_API_KEY is not set` | Add the matching key to `.env` for the backend you chose |
| Google Vision authentication fails | Run `gcloud auth application-default login` and confirm the credential's project has Vision enabled and billing configured |
| `tesseract is not installed` / empty OCR output | When `OCR_BACKEND=tesseract`, run `brew install tesseract tesseract-lang` (see step 1) |
| Garbled Hindi from a PDF that looks fine | Expected — corrupt-layer detection should force OCR automatically |
| OCR gives poor results | Confirm the configured `OCR_BACKEND`, its language settings, and whether the page was routed to `direct` or `ocr`; compare backends with `benchmark_ocr.py` before changing DPI |
| `429` / rate limit errors | Check the quota for the selected backend/model/region, reduce request frequency, and retry with backoff |
| `429` partway through `evaluate.py` | The retriever is unpaced (see Tips). Re-run in smaller batches, or wait for the per-minute window to reset — a mid-run failure is a burst-rate limit, not an exhausted daily allowance |
| `No indexed documents` (HTTP 503) | Run `python app.py index <pdf>` first |
