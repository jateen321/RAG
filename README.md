# 📚 Gyaan Sarthi

**Ask your books. Trace every answer.**

![Gyaan Sarthi — Ask your books. Trace every answer.](frontend/public/og.png)

Gyaan Sarthi is a bilingual study assistant for PDFs, text files, Markdown, and
YouTube transcripts. It retrieves evidence before answering, links citations
back to their exact page or timestamp, and can turn the same retrieved evidence
into a generated study visual.

Uses **Google Gemini** for query planning, embeddings, reranking, and answers.
PDF pages with clean text layers are read directly; scanned or garbled pages use
the configured OCR backend: **Google Cloud Vision** (default), **Tesseract**, or
**Gemini multimodal OCR**. No LangChain — the pipeline is hand-rolled.

ChromaDB runs locally. OCR and Gemini cost and availability depend on the selected
backends, models, billing status, and project-specific quotas; do not assume every
configuration is free.

## ✨ What You Can Demonstrate

- **Evidence-grounded chat** in English, Hindi, or mixed language, with source
  citations that open the original page, passage, or YouTube timestamp.
- **Adaptive ingestion** for PDF, TXT, Markdown, individual YouTube videos, and
  playlists, with OCR only where a PDF's text layer is missing or unreliable.
- **Multimodal questions** that combine a written prompt with PNG, JPEG, or WebP
  input (up to 10 MB).
- **Grounded study visuals**: Image mode returns the normal answer and sends the
  original question plus the same retrieved passages—not the model-written
  answer—to the image model.
- **Deliberate web fallback**: document retrieval remains the default; eligible
  answers can be replaced with a Google Search-grounded response on request.
- **Persistent, editable conversations** with regeneration from the edited turn
  and automatic truncation of the obsolete conversation branch.

### Three-minute demo

1. Upload a PDF or index a YouTube URL from the library panel.
2. Ask a factual question and open a citation to verify the supporting passage.
3. Ask a follow-up to demonstrate conversation memory.
4. Enable **Image** and ask for a diagram; compare the text answer and generated
   visual to the retrieved citations.
5. Edit an earlier prompt or use the offered web-search action to show controlled
   branching rather than hidden tool use.

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

```dotenv
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

Folder ingestion, direct uploads, `/index`, and the PDF CLI share content-based
duplicate detection. Identical files reuse the indexed document regardless of
path or filename; different files with the same filename in different folders
are not skipped just because their names match. The first indexed source path
is retained for citations. Interrupted indexing can resume through either route.
Legacy documents are compared by their complete indexed passage sequences after
extraction; their citation IDs are preserved. This does not automatically remove
duplicate documents already present in an older index or reorganize source files.

Use the exact source name shown by `status` when removing one document:

```bash
python app.py status
python app.py remove "CIL.pdf"
```

`remove` preserves other documents. `reset` deletes the complete local
`chroma_db/` directory. Neither command deletes the original documents, but
deleted vectors must be regenerated by indexing the source files again.

## 🌐 Web API and Swagger

The React interface uses this FastAPI API for chat, indexing, source navigation,
library status, and conversation history. When you run the unified development
launcher, the API starts automatically on port 8000:

```bash
.venv/bin/python dev.py
```

For backend-only development, start FastAPI without the frontend:

```bash
.venv/bin/python -m uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

Open **http://127.0.0.1:8000/docs** for interactive Swagger documentation and
request testing. FastAPI generates that page from the live route definitions, so
it is the authoritative interface when this summary and the code ever disagree.

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Service info |
| `GET` | `/health` | Status + index statistics |
| `GET` | `/documents/{source_path}` | Open a supported local document stored inside `data/` |
| `GET` | `/passages/{chunk_id}` | Fetch the complete indexed passage for an exact source-scoped citation |
| `GET` | `/passages/resolve-legacy` | Resolve an older citation that does not contain a chunk ID |
| `POST` | `/ask` | Ask a grounded question, persist the exchange, and return its answer and sources |
| `POST` | `/ask/image` | Ask with a PNG, JPEG, or WebP attachment (maximum 10 MB) |
| `GET` | `/conversations` | List saved conversations |
| `GET` | `/conversations/{conversation_id}` | Load one conversation and its exchanges |
| `PUT` | `/conversations/{conversation_id}/exchanges/{exchange_id}` | Edit a prompt, regenerate its answer, and truncate later turns |
| `POST` | `/conversations/{conversation_id}/exchanges/{exchange_id}/search-web` | Replace the latest eligible answer with a web-grounded answer |
| `POST` | `/conversations/{conversation_id}/exchanges/{exchange_id}/generate-image` | Generate a visual for an eligible saved exchange |
| `DELETE` | `/conversations/{conversation_id}` | Delete one saved conversation (`204 No Content`) |
| `GET` | `/generated-images/{image_id}` | Return a generated image referenced by a conversation |
| `POST` | `/index` | Index or deduplicate a PDF, TXT, or Markdown file already in `data/` |
| `POST` | `/index/folder` | Recursively index an allowlisted server-local folder |
| `POST` | `/upload` | Upload and index a PDF, TXT, or Markdown file (`201 Created`) |
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

### How Image mode stays grounded

For document questions, retrieval runs once. The selected, source-labelled
passages feed two outputs: the normal text answer and an internal image prompt.
The image prompt has this shape:

```text
Create one clear educational visual that answers the student's request.
Use only the retrieved evidence below as the factual grounding.
...
Student request: <original question>

Retrieved evidence:
<source-labelled passages selected by the retriever and reranker>
```

The internal prompt is stored only when needed for later image generation and is
not exposed in the `/ask` response. When Web mode and Image mode are combined,
the generated visual is grounded in the web-grounded answer because document
retrieval is not used in that path.

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

For production authentication, enable Google sign-in in Firebase Authentication
and configure the backend `FIREBASE_PROJECT_ID` plus the frontend
`NEXT_PUBLIC_FIREBASE_*` values. The browser exchanges a recent Firebase ID
token for a Secure, HttpOnly session cookie; conversations, source files,
generated images, Chroma document IDs, catalogs, and vector queries are scoped
to that verified Firebase UID. Set `LEGACY_ADMIN_UID` before the first production
start so pre-tenancy SQLite and Chroma rows are assigned to the administrator.

`LEGACY_ADMIN_UID` only decides who inherits those legacy rows; it does not grant
the administrator role. `POST /index/folder` requires an `admin` custom claim,
and Firebase custom claims can only be written by a trusted server through the
Admin SDK. Grant it once, after the account has signed in at least so it exists:

```bash
.venv/bin/python grant_admin.py you@example.com --grant
```

Run without a flag to inspect the current claim. The account must sign out and
in again before a session carries it. Revoking also revokes refresh tokens, which
the API only enforces when `AUTH_CHECK_REVOKED=1`.

Deploy the frontend and API on the same site (for example `app.example.com` and
`api.example.com`) so `SameSite=Lax` session cookies work for API requests and
direct PDF/image links. Truly cross-site domains require
`SESSION_COOKIE_SAMESITE=none`, HTTPS, `SESSION_COOKIE_SECURE=1`, credentialed
CORS, and browser third-party-cookie support; a same-site reverse proxy is more
reliable.

### Recommended: run both development servers together

Stop any backend or frontend processes that you previously started by pressing
`Ctrl+C` in their terminals. Then run this single command from the repository root:

```bash
.venv/bin/python dev.py
```

The launcher owns both child processes and keeps them aligned:

| Change | Development behavior |
|---|---|
| Python source | Uvicorn reloads the backend |
| React/CSS source | Vinext/Vite updates the frontend through HMR |
| Git branch switch | Both servers restart so mixed branch versions cannot remain live |
| Root `.env` | Backend restarts |
| Frontend environment file | Frontend restarts |
| `frontend/package.json` or `package-lock.json` | Frontend restarts; run `npm install` separately when dependencies changed |

Press `Ctrl+C` once to stop both servers. The launcher never kills an unrelated
process: if port 3000 or 8000 is occupied when it starts, it reports the conflict
and exits.

### Manual two-terminal alternative

Start the API in one terminal:

```bash
source .venv/bin/activate
uvicorn api:app --host 127.0.0.1 --port 8000 --reload
```

`--reload` is intended for local development: it restarts the backend when Python
files change. Without it, Uvicorn keeps the already-imported code in memory until
the process is restarted.

Start the React interface in another terminal:

```bash
cd frontend
npm ci
npm run dev -- --host 127.0.0.1
```

Open **http://localhost:3000**. The frontend connects to
`http://127.0.0.1:8000` by default. Copy `frontend/.env.example` to
`frontend/.env.local` to configure a different API or site URL.

### Restarting after code or branch changes

The manual commands above each start a long-running process. Do not launch a second
copy on the same port. If startup reports `Address already in use`, or `dev.py`
reports an occupied port, check which process is already listening:

```bash
lsof -nP -iTCP:3000 -sTCP:LISTEN
lsof -nP -iTCP:8000 -sTCP:LISTEN
```

Return to the terminal running that server and press `Ctrl+C`. Start `dev.py` after
the ports are free. When using the manual commands, restart both servers after
switching branches, applying a stash, or making dependency and configuration changes.
File watchers can miss large atomic working-tree changes, and a backend started without
`--reload` never reloads Python modules automatically.

If the servers are current but an existing browser tab still displays the previous
interface, use a hard refresh (`Cmd+Shift+R` on macOS or `Ctrl+Shift+R` on
Windows/Linux), or close and reopen **http://localhost:3000**.

Remember that the two servers have different responsibilities: restarting Uvicorn on
port 8000 updates the API only; it does not rebuild or restart the frontend on port 3000.

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
                  Gemini reranking → selected evidence → grounded answer
                                                        └→ study visual (optional)
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
  their rankings, and a text-overlap filter removes near-copy passages.
- Gemini 3.5 Flash-Lite receives up to 15 distinct candidates and selects only the
  answer-worthy passages. For normal app questions, it chooses a variable 5–15
  passages; explicit evaluation cutoffs remain fixed for comparable measurements.
  Every selected passage is passed to answer generation; omitted candidates are not
  silently added back.
  If planning or reranking fails, the pipeline falls back to the original query
  or the RRF order respectively.
- The selected complete chunk texts are inserted into the answer prompt; the
  terminal displays only short previews.
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
python evaluate.py --retrieval-mode pipeline --output evaluation/results_pipeline.json
python evaluate.py --generate           # ALSO generate answers (costs LLM calls)
```

> ⚠️ `evaluate.py` does **not** read `TOP_K` from `config.py` — its `--top-k` default
> is hardcoded to 5. Changing `config.TOP_K` alters the app's behaviour but not the
> harness's direct mode, so pass `--top-k` explicitly when comparing direct runs.
> Pipeline mode uses the app's adaptive 5–15 selection and ignores `--top-k`.

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

**Adaptive pipeline comparison** (52 matching questions, 2026-08-30):

| metric | direct top-5 | planner + RRF + Gemini 3.5 reranker |
|---|---:|---:|
| `retrieval_hit_rate` | 0.9737 | 0.9737 |
| `mean_reciprocal_rank` | 0.8925 | 0.8728 |
| `mean_source_precision` | 0.8105 | 0.8298 |
| hard `mean_source_precision` | 0.7789 | 0.8333 |
| average retrieval latency | 1.116s | 5.066s |
| average selected chunks | 5.0 | 5.654 |

The adaptive workflow preserves recall and improves source purity, especially on hard
questions, but it does not dominate the simpler baseline: overall rank quality is lower
and retrieval is about 4.54× slower. Treat it as a precision/context tradeoff pending
answer-quality evaluation, not as an unconditional robustness improvement.

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

## ✅ Verification Before a Demo

The unit tests mock external model calls, so they validate application behavior
without spending Gemini or OCR quota:

```bash
.venv/bin/python -m unittest discover
cd frontend
npm run lint
npm run build
```

Run the live application separately before presenting because unit tests cannot
verify your current credentials, cloud quota, indexed corpus, or network access.

## 📁 Project Structure

```
RAG/
├── app.py              # CLI interface (main entry point)
├── dev.py              # Unified backend/frontend development supervisor
├── api.py              # FastAPI web interface (/ask, /index, /health)
├── config.py           # Configuration & constants
├── llm_client.py       # Gemini client factory (Developer API vs Vertex)
├── ocr_engine.py       # PDF → text, per-page routing + selectable OCR
├── text_quality.py     # Scores the text layer to pick direct vs OCR
├── indexer.py          # Text → chunks → embeddings → ChromaDB
├── youtube_ingester.py # YouTube metadata + transcript → timestamped chunks
├── retriever.py        # Semantic search in ChromaDB
├── retrieval_pipeline.py # Query rewrites → fusion → reranking
├── rag_engine.py       # Grounded text, web, and image generation
├── conversation_store.py # Persistent conversation and generated-image metadata
├── evaluate.py         # Retrieval evaluation harness (hit rate, MRR, source precision)
├── requirements.txt    # Python dependencies (direct only)
├── .env                # Your API key (private, not in git)
├── .env.example        # Template for .env
├── FINDINGS.md         # Measured results + provenance tags (verified / cloud / hypothesis)
├── OCR_NOTES.md        # Text-extraction issues log: routing, legacy fonts, corrupt layers
├── AGENTS.md           # Working conventions for AI-assisted sessions on this repo
├── frontend/           # Vinext/React interface and contributor README
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
| `TOP_K` | 5 | Fixed cutoff for explicit `top_k` calls and evaluation |
| `QUERY_REWRITE_MAX_QUERIES` | 10 | Maximum total queries, including the original |
| `QUERY_RETRIEVAL_TOP_K` | 5 | Candidates retrieved per query before fusion |
| `RERANK_CANDIDATE_LIMIT` | 15 | Maximum RRF candidates sent to Gemini reranking |
| `RERANK_MODEL` | `gemini-3.5-flash-lite` | Lightweight structured-output model used only for reranking |
| `IMAGE_MODEL` | `gemini-3.1-flash-image` | Model used for optional generated study visuals |
| `MIN_CONTEXT_CHUNKS` | 5 | Minimum passages selected for a normal app question |
| `MAX_CONTEXT_CHUNKS` | 15 | Maximum passages selected for a normal app question |
| `NEAR_DUPLICATE_OVERLAP` | 0.85 | Three-word-shingle containment threshold for near-copy removal |
| `OCR_BACKEND` | `google_vision` | OCR implementation: `google_vision`, `tesseract`, or `gemini` |
| `OCR_MAX_WORKERS` | backend-specific | Concurrent OCR calls/processes; defaults to Vision 8, Gemini 2, Tesseract up to 4 |
| `PDF_DPI` | 300 | Resolution used when rasterizing pages for any OCR backend |
| `GOOGLE_VISION_LANGUAGE_HINTS` | `hi,sa,en` | Soft language hints for Google Cloud Vision |
| `TESSERACT_LANG` | `eng+hin+san` | OCR languages (English + Hindi + Sanskrit) |
| `LAYER_CHECK_SAMPLE` | 3 | Pages OCR'd per document to spot-check the text layer |
| `LAYER_CHECK_MIN_SIMILARITY` | 0.4 | Median OCR-vs-layer similarity below this → distrust the layer and OCR the whole document |
| `EMBEDDING_MODEL` | `gemini-embedding-001` | Embedding model |
| `LLM_MODEL` | `gemini-3.5-flash-lite` | Generation and query-planning model |
| `RAG_RATE_LIMIT_ENABLED` | `0` | Enable Redis-backed admission control; set to `1` for public deployment |
| `REDIS_URL` | unset | Managed Redis connection URL, required when admission control is enabled |
| `RAG_RATE_LIMIT_ASK_PER_MINUTE` / `ASK_BURST` | `10` / `3` | Sustained and burst allowance per authenticated user |
| `RAG_RATE_LIMIT_WEB_PER_MINUTE` / `WEB_BURST` | `5` / `2` | Web-search allowance per authenticated user |
| `RAG_RATE_LIMIT_IMAGE_PER_MINUTE` / `IMAGE_BURST` | `2` / `1` | Image-generation allowance per authenticated user |
| `RAG_RATE_LIMIT_INGEST_PER_HOUR` / `INGEST_BURST` | `5` / `5` | Ingestion starts per authenticated user; server-folder indexing also requires an administrator |
| `RAG_CONCURRENCY_INTERACTIVE` | `4` | Shared active document-answer limit across all replicas |
| `RAG_CONCURRENCY_WEB` | `2` | Shared active web-search limit across all replicas |
| `RAG_CONCURRENCY_IMAGE` | `1` | Shared active image-generation limit across all replicas |
| `RAG_CONCURRENCY_INGEST` | `1` | Shared active ingestion limit across all replicas |

### Public API admission control

Rate limits use the verified Firebase user ID, hashed before it becomes part of
a Redis key. Token-bucket updates and multi-bucket charges are atomic Lua
operations, so Web + Image requests cannot bypass a more expensive bucket.
Concurrency slots are global renewable Redis leases: they coordinate multiple
Uvicorn workers and deployment replicas and expire after a crashed worker.

When enabled, admission **fails closed** if Redis cannot be reached: costly work
returns HTTP 503 instead of reaching paid providers without protection. An
exhausted bucket or busy concurrency pool returns HTTP 429 with an integer
`Retry-After` header. Work that was admitted before a temporary Redis outage is
allowed to finish; its lease eventually expires even if renewal and cleanup
cannot reach Redis. Use a TLS `rediss://` URL, keep Redis credentials in the
deployment secret manager, and run a Redis connectivity smoke test before
serving public traffic.

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
| `Address already in use` on port 3000 or 8000 | A frontend or backend process is already listening. Find it with `lsof` and stop it with `Ctrl+C` in its original terminal before restarting |
| Browser still shows old UI after code/branch changes | Restart the frontend dev server, then hard-refresh the browser; restarting Uvicorn affects only the backend |
| Backend still uses old Python code | Start Uvicorn with `--reload` during development, or manually restart the existing backend process |
| `429` during evaluation or repeated questions | Wait for the relevant quota window, reduce request frequency, and inspect the reported quota metric; multi-query embeddings are batched, but query planning, reranking, and answer generation still make model calls |
| `No indexed documents` (HTTP 503) | Run `python app.py index <pdf>` first |
