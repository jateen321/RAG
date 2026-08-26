<img width="1318" height="641" alt="image" src="https://github.com/user-attachments/assets/f4ff5b58-b10c-4700-99d8-a9b6acfa7b51" />

# 📚 Hindi Textbook RAG

Query scanned Hindi textbook PDFs with a local OCR and vector-search pipeline,
then use Gemini to generate answers from the retrieved text.

Uses **Google Gemini** for embeddings and answers, and **Tesseract OCR** for Hindi + Sanskrit + English text extraction. No LangChain — the whole pipeline is hand-rolled.

Tesseract and ChromaDB run locally. Gemini cost and availability depend on the
selected backend, model, billing status, and project-specific quotas; do not
assume every configuration is free.

## 🚀 Quick Start

### 1. Install Tesseract (the OCR engine)

Tesseract is a **native binary**, not a Python package — install it first, plus the Hindi/Sanskrit language data:

```bash
brew install tesseract tesseract-lang
```

> 🐧 On Debian/Ubuntu: `sudo apt install tesseract-ocr tesseract-ocr-hin tesseract-ocr-san`

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
                    └─ scanned / garbled ─→ rasterize → Tesseract (hin+san+eng)
                                                     ↓
                        Boundary-aware chunks → Gemini Embeddings* → ChromaDB
                                                                        ↓
Your Question → Gemini Embedding → Similarity Search → Top 5 Chunks
                                                                        ↓
                                            Gemini Flash + Context → Answer!
```

YouTube follows a parallel ingestion path: `yt-dlp` reads video/playlist
metadata, `youtube-transcript-api` retrieves timestamped captions, and the
resulting chunks enter the same Gemini embedding and ChromaDB pipeline. Answers
cite transcript timestamps instead of PDF page numbers.

`*` Embeddings and answer generation use the backend selected by `LLM_BACKEND`.

**Two defenses worth knowing about:**
- **Per-page routing** — pages with a usable text layer skip OCR entirely, so indexing is much faster on mixed PDFs.
- **Corrupt-layer detection** — some PDFs render fine but return scrambled characters from a broken font map. The indexer OCRs a few sample pages and compares them to the text layer; if they disagree, it distrusts the layer and OCRs the whole document.

### Retrieval behavior and current limitations

- The question is embedded as one query vector.
- ChromaDB searches the shared `hindi_textbook` collection and returns `TOP_K=5`
  chunks ordered by vector distance.
- The five complete chunk texts are inserted into the Gemini prompt; the
  terminal displays only short previews.
- There is currently no minimum relevance threshold, reranker, or source/PDF
  filter. A weak or general question can therefore retrieve unrelated chunks.
- All indexed PDFs share one collection. Use `remove` to delete one source or
  `reset` to rebuild the complete index.
- **Retrieval is not rate-paced.** `indexer.py` batches embedding calls and sleeps
  between them, but `retriever.py` sends one unpaced embedding request per query.
  A scripted batch of questions can therefore trip a burst-rate quota even though
  indexing the same corpus succeeds.

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
| `citation_accuracy`, `average_keyword_recall` | Generation metrics, only with `--generate` |

**Dataset:** `evaluation/questions.json` — 20 questions, 10 English and 10 Hindi.

> ⚠️ **Known limit of the current eval set:** every question expects `CIL.pdf`.
> Because CIL is also the majority of the indexed corpus, `source_precision` scores
> near 1.00 for structural reasons rather than because retrieval is well-scoped. This
> set therefore cannot measure cross-document contamination, and it contains no
> *negative* questions (ones with no answer in the corpus) — so it cannot detect that
> retrieval always returns `TOP_K` chunks whether or not any are relevant.

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
├── ocr_engine.py       # PDF → text, per-page routing + Tesseract OCR
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
│   └── questions.json  # Eval dataset; results_*.json are written locally, not committed
├── data/               # Drop your PDFs here
└── chroma_db/          # Vector database (auto-created)
```

## ⚙️ Configuration

Edit `config.py` to tune these settings:

| Setting | Default | Description |
|---|---|---|
| `CHUNK_SIZE` | 800 | Target characters per chunk (soft cap — respects boundaries) |
| `CHUNK_OVERLAP` | 100 | **Minimum** overlap between chunks (floor) |
| `MAX_CHUNK_OVERLAP` | 250 | **Maximum** overlap between chunks (ceiling). Takes precedence over the floor — prevents one long sentence being copied whole into the next chunk |
| `MIN_CHUNK_LENGTH` | 50 | Skip chunks shorter than this |
| `TOP_K` | 5 | Number of chunks to retrieve |
| `PDF_DPI` | 300 | OCR scan resolution (Tesseract's recommended minimum) |
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

- **OCR resolution trade-off**: `PDF_DPI` defaults to 300, Tesseract's recommended
  minimum for small glyphs such as Devanagari matras. Pixel count scales with the
  *square* of DPI, so 300 costs ~2.25× the OCR time of 200. Drop to 200 if your
  scans are clean and indexing is too slow; going above 300 rarely helps.
- **Changing `PDF_DPI` requires a re-index — and `remove` first**: a new DPI
  produces different OCR text, which produces different chunk IDs. Re-indexing
  without removing first *adds* a second copy of every page instead of replacing
  it. Run `python app.py remove "<source>"`, then index again.
- **Index multiple books**: Run `index` on multiple PDFs — they all go into the same database.
- **Ask in any language**: Questions can be in Hindi, English, or mixed.
- **Rate limits vary**: Limits depend on the backend, model, region, billing tier,
  and Google Cloud project. Check the quota assigned to your credential instead
  of assuming a fixed requests-per-minute value.
- **Embedding pacing is weaker than it looks**: The *indexer* embeds up to 20 chunks
  per call and retries a quota-related failure once after 30 seconds. Note that
  `_embed_texts` contains a one-second inter-batch delay that **never executes** in
  practice: `index_document` pre-slices into batches of 20 and then calls
  `_embed_texts(batch, batch_size=len(batch))`, so the inner loop always runs exactly
  one iteration and its delay guard is never true. The *retriever* has no batching,
  pacing, or retry at all. Both paths are therefore effectively unpaced, and only the
  indexer recovers from a `429`.
- **Model migration**: Do not query existing vectors with a different embedding
  model, even when both models output the same number of dimensions. Create a
  new collection or reset and reindex every PDF.

## 🔧 Troubleshooting

| Issue | Solution |
|---|---|
| `LLM_BACKEND=developer but GEMINI_API_KEY is not set` | Add the matching key to `.env` for the backend you chose |
| `tesseract is not installed` / empty OCR output | `brew install tesseract tesseract-lang` (see step 1) |
| Garbled Hindi from a PDF that looks fine | Expected — corrupt-layer detection should force OCR automatically |
| OCR gives poor results | `PDF_DPI` is already 300; raising it further rarely helps. Confirm `TESSERACT_LANG` covers the script, and check whether the page was routed to `direct` rather than `ocr` |
| `429` / rate limit errors | Check the quota for the selected backend/model/region, reduce request frequency, and retry with backoff |
| `429` partway through `evaluate.py` | The retriever is unpaced (see Tips). Re-run in smaller batches, or wait for the per-minute window to reset — a mid-run failure is a burst-rate limit, not an exhausted daily allowance |
| `No indexed documents` (HTTP 503) | Run `python app.py index <pdf>` first |
