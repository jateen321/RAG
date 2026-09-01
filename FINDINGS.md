# Project Findings

Consolidated findings for the Hindi/English RAG app. Each result is tagged with its
**provenance** so every number is traceable:

- 🟢 **Repo-verified** — reproducible today from code + saved results in this repository.
- 🟣 **Cloud session** — measured in a separate cloud work session; the code is not yet
  ported into this repo. *Action: port the script/results back to make it reproducible here.*
- ⚪ **Hypothesis** — believed/expected, not yet measured.

Last updated: 2026-08-31.

---

## 12. Google Cloud VM deployment

- 🟢 **Cloud session (2026-09-01):** `gyaan-sarthi` is running on an
  `e2-medium` VM in `asia-south1-a` at `8.234.120.96`. Caddy obtained valid
  Let's Encrypt certificates for both DuckDNS hostnames, and the public API
  returned `total_chunks: 21499` after the ChromaDB/data transfer.

- 🟢 **Cloud session (2026-09-01):** Before the privacy hardening, anonymous
  `/health` exposed the shared document inventory and internal ChromaDB path.

---

## 11. Google Cloud deployment boundary

- 🟢 **Repo-verified (2026-08-31):** The current runtime stores approximately
  1.4 GB in local ChromaDB/data directories and uses SQLite for conversations.
  A Compute Engine VM with persistent disk preserves this design; Cloud Run's
  writable filesystem is disposable, so Cloud Run requires a datastore/storage
  migration before it can safely host this workload.

---

## 10. Shared-corpus ownership boundary

- 🟢 **Repo-verified (2026-08-31):** The live Chroma collection contained 21,499
  chunks with no `owner_id`, so authenticated UID-filtered retrieval could not see
  the pre-authentication corpus. All 21,499 rows are now labeled
  `__shared_corpus__`; guests and signed-in users retrieve only that owner, while
  conversation persistence remains Firebase-UID scoped. Chroma rejected the first
  21,499-row metadata update because its maximum batch was 5,461, so ownership
  migration is now bounded to 5,000 rows per update.

---

## 9. Development session persistence

- 🟢 **Repo-verified (2026-08-31):** Local Google login appeared successful because
  `AuthGate` immediately stored the `/auth/session` response in React state, even when
  the browser could not reuse the session cookie after refresh. The development setup
  combined a `Secure` cookie over HTTP with `localhost` frontend and `127.0.0.1` API
  hostnames. The unified launcher now uses a non-Secure cookie only in development and
  one `localhost` site for both ports; production retains Secure-by-default cookies.

---

## 10. Firebase Email/Password sign-in gap

- 🟢 **Repo- and deployment-verified (2026-09-01):** Firebase Email/Password may be
  enabled in the Firebase console, but the deployed interface only invokes
  `GoogleAuthProvider`, and the API rejects every Firebase provider except `google.com`.
  Email/Password users therefore cannot create an application session.

---

## 11. VM Firebase Admin credential scope

- 🟢 **Deployment-verified (2026-09-01):** The production VM's attached Compute
  service account lacked the `cloud-platform` OAuth scope while completed Google
  sign-ins received 401 at `/auth/session`. The scope was corrected and both public
  frontend and API health checks recovered; a real sign-in retry remains required.

---

## 12. VM OS Login and IAP hardening

- 🟢 **Deployment-verified (2026-09-01):** `gyaan-sarthi` now uses VM-level OS
  Login with 2FA. SSH is allowed from IAP only (priority 900), while a VM-tagged
  deny rule blocks public SSH and RDP (priority 1000); HTTPS remains healthy.
  The GitHub deployer has project Viewer, IAP Tunnel User, OS Admin Login, and
  Service Account User only on the VM's attached Compute service account.

---

## 13. OS Login 2FA challenge completion

- 🟢 **Deployment-verified (2026-09-01):** IAM and IAP policy checks succeed for
  `b22cs026@gmail.com`. OS Login offers a ready security-key OTP and proposed
  `AUTHZEN` (Google phone prompt); the reported failures end after selecting the
  alternate challenge while it remains pending, before a response is recorded.

---

## 8. Production identity boundary

- 🟢 **Repo-verified (2026-08-31):** The API verifies Firebase ID tokens against
  `FIREBASE_PROJECT_ID`, and the browser independently initializes Firebase from
  `NEXT_PUBLIC_FIREBASE_*`. These values must name the same project. Changing the
  Firebase/GCP project establishes a new UID namespace, so existing private-library
  ownership cannot be carried across through configuration alone.

---

## 7. Citation rendering contract

- 🟢 **Repo-verified (2026-08-30):** Citation rendering depends on an exact
  delimiter contract between `rag_engine.SYSTEM_PROMPT` and
  `frontend/app/answer-markdown.tsx`. The renderer converts `⟦source, locator⟧` and
  legacy `[[source, locator]]` forms into numbered source buttons only when the source
  name and locator also match a returned source. The renderer now also recovers a
  single-bracket `[source, locator]` form only after that exact source-and-locator match;
  unrelated bracketed prose remains literal. The focused static-render check covers all
  three wrappers, including the single-bracket delimiter drift seen in a saved answer.

---

## Codex tooling

- ⚪ **Hypothesis (2026-08-27):** The Codex TUI bootstrap error `thread/resume failed …
  already has an active writer (code -32600)` indicates that a second client/process tried
  to resume a thread currently owned for writing by another live or stale session. The
  exact trigger and recovery procedure are not documented in the official OpenAI docs.

---

## 1. Retrieval quality

### 🟢 Repo-verified (Gemini retriever)
Source: `evaluate.py` + `evaluation/results_baseline.json`. Dataset: **20 questions**
(all from CIL, bilingual `cil-01-en … cil-20-hi`), `top_k = 15`.

| Metric | Value |
|---|---|
| Retrieval hit rate | **1.0 (100%)** |
| Mean Reciprocal Rank | **0.8042** |
| Avg retrieval latency | 0.603 s |

Note: `evaluate.py` computes **hit-rate**, **MRR** and (since `b077950`) **source
precision** over one retriever (`retriever.retrieve()` = Gemini embeddings). It does
**not** compute Hit@1/@3/@5, does **not** record per-chunk `distance`, and does **not**
compare retrievers. The `results_k1 … k15` files are a **top-k sweep of the same Gemini
retriever**, not a retriever comparison.

⚠️ **These numbers are superseded.** They were measured against an index later shown to
be corrupt (see §6.1) and at `top_k = 15` while the app ships `TOP_K = 5`. Treat them as
historical until re-run against a clean corpus. Note also that `evaluate.py --top-k`
defaults to a hardcoded `5` and does *not* read `config.TOP_K`, so the harness and the
app can silently disagree.

### 🟣 Cloud session (retriever comparison — code not yet in this repo)
Measured on a larger eval set (~100 queries) in a separate cloud session:

| Retriever | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| Local MiniLM | 30% | 46% | 59% | 0.399 |
| Gemini embedding | 91% | 100% | 100% | 0.950 |

Gemini handled Hindi, English, paraphrasing and OCR noise far better than MiniLM.
*To reproduce here: port the MiniLM retriever + the Hit@k evaluator and the ~100-query set.*

---

## 2. Reranking

### 🟣 Cloud session (MiniLM → local BGE reranker — code not yet in this repo)

| Pipeline | Hit@1 | Hit@3 | Hit@5 | MRR |
|---|---:|---:|---:|---:|
| MiniLM only | 30% | 46% | 59% | 0.399 |
| MiniLM + BGE reranker | 65% | 76% | 77% | 0.704 |

- BGE substantially reorders MiniLM candidates, but **cannot recover an answer MiniLM never
  retrieved** (it only re-scores the candidate set).
- CPU reranking ≈ **4 s/query** — expensive.
- The reranker reads the raw *(query, chunk)* text together; it does **not** reuse the
  Gemini/MiniLM vectors.
- Because Gemini retrieval already has high Hit@1, **reranking stays optional** (accuracy-over-
  latency toggle).

---

### 🟢 Repo-verified (query latency audit, 2026-08-27)

- `retrieve_context()` rebuilds the source catalog on every question via
  `get_stats()`, which fetches and aggregates metadata for every indexed chunk.
  This happens even when query rewriting is disabled. The `query_planning_s`
  timer includes this scan, so it does not isolate planner model latency.
- Query planning, embedding/vector search, optional Gemini reranking, and answer
  generation run sequentially. `vector_retrieval_s` combines embedding network
  time with local Chroma work; `reranking_s` also includes rank fusion.
- The frontend waits for the complete `/ask` response before displaying answer
  text. Engine `total_s` excludes subsequent conversation persistence and browser
  delivery, so it is not a measurement of the full user-visible wait.
- ⚪ **Hypothesis:** catalog caching and streaming could reduce repeated local
  work and time to first visible answer respectively. No speedup or dominant
  bottleneck has been measured for the current multi-query pipeline.

---

## 3. TF-IDF (local lexical fallback) — 🟣 Cloud session
- Word TF-IDF works when query and document share literal words.
- Character TF-IDF (3–5 char n-grams) is more tolerant of spelling / OCR errors.
- Word + char combined = a useful **offline** fallback, but semantically weaker than Gemini.

---

## 4. OCR & ingestion

### 🟢 Repo-verified
- **Cross-route identity (2026-08-27):** folder ingestion supplied an absolute
  path as `document_key`, while direct ingestion hashed the displayed filename.
  Deterministic chunk IDs therefore did not prevent copies across routes.
  `test_document_identity.py` verifies content-based identity in temporary Chroma
  databases for PDF/TXT/MD in both upload orders, including renamed copies,
  distinct same-name documents, partial retries, and legacy citation-ID preservation.
  These tests do not constitute cleanup of pre-existing duplicates in the live index.
- **OCR cache collision (2026-08-27):** same-name files can share size and mtime
  yet contain different bytes. The new regression fixture reproduces those
  conditions; a file-content fingerprint now prevents their OCR cache reuse.
- **Resolved 2026-08-27:** the interactive CLI banner advertised the removed EasyOCR
  integration. It now describes Gemini-powered retrieval and adaptive text-layer/OCR
  routing without hardcoding one of the configurable OCR backends (Google Vision,
  Tesseract, or Gemini).
- **Per-page routing is now wired into `ocr_engine.py`**: each page's text layer goes through
  `text_quality.choose_method()` → `direct` (trust layer) or `ocr` (rasterize + Tesseract).
  No more OCR-ing clean digital pages. CIL now reads `COAL INDIA LIMITED` (was `Coall IIndia`).
  Verified: CIL 15/15 direct (OCR model never loads); Dharma-Sastra 0/638 direct (pure scan).
- **OCR engine is now Tesseract** (`pytesseract`, `lang="eng+hin+san"`), replacing EasyOCR.
  On Dharma-Sastra (old English scan), Tesseract beat EasyOCR on `engword%` on all 3 sampled
  pages: 56.7→65.1, 71.2→75.2, 77.9→84.6. Common English words much cleaner; residual errors
  concentrate on IAST-diacritic Sanskrit names. See `OCR_NOTES.md`.
- `text_quality.py` routes per page using deva% / latin% / junk% / real-word%. Validated on
  CIL, essence, 027 (legacy-font gibberish caught), bhagya (scanned), Dharma-Sastra (pure scan).
- **Language-agnostic corrupt-layer defense wired** (`ocr_engine._verify_text_layer`): OCR a few
  `direct` pages and compare to the text layer (difflib similarity). Manusmriti's broken Unicode
  layer (median 0.02) is caught and forced to OCR; clean layers (CIL 0.93, essence 0.999) stay
  `direct`. Threshold 0.4. Catches corrupt layers `deva%`/`junk%`/`engword%` cannot.

### 🟣 Cloud session
- Tesseract ≈ **5× faster** than EasyOCR per page (directionally consistent with repo runs:
  no 100MB model preload, visibly faster — not yet rigorously timed here).

### ⚪ Hypothesis / conventions to keep
- ✅ *Done 2026-08-22 (§6.4):* preserve **page number, source filename, extraction method,
  chunk id** as metadata — plus `source_type`, `document_id` and `content_hash`.
- Character-based chunks with overlap are resilient to imperfect OCR — but see §6.2:
  overlap needs a **ceiling** as well as a floor, or it degenerates into duplicates.
- ~~Deterministic **SHA-256 chunk ids** prevent duplicate ingestion.~~ **Corrected by
  §6.1.** Determinism alone is not sufficient and was actively harmful here: the ids were
  MD5, derived from the chunk *text*, and omitted the source. Re-extracting a page changed
  the text, changed the id, and therefore *added* a duplicate rather than replacing it.
  Ids must be derived from **document identity + position**, and writes must use `upsert`.

---

## 5. Model & pipeline decisions

```text
PDF → OCR/route → cleaning → chunking → Gemini embeddings → ChromaDB
Query → Gemini embedding → top-k chunks → optional BGE rerank → answer + page citations
```

- **Primary embedding:** Gemini.
- **Local fallback:** word + char TF-IDF, or MiniLM.
- **Reranker:** BGE, enabled only when accuracy > latency.
- **EmbeddingGemma:** deprioritized (official model gated; Gemini already strong).

---

## 6. Indexing integrity & chunking (2026-08-22)

### 6.1 🟢 The index held two code generations of the same pages (keystone finding)
`CIL.pdf` had **61 of 91 chunks containing a literal `\n`** — impossible under the current
`_split_units`, which runs `re.sub(r'\s+', ' ', sent)` on every unit before joining. So
those chunks were written by an **older indexer**. `Bhagya` (indexed later) had 0/30.
Per-page breakdown: **13 of 15 CIL pages carried chunks from both generations**. The
corpus contained both `Coall` / `CMAHARATNA` (pre-routing EasyOCR garbling) *and*
`COAL INDIA LIMITED` (clean post-routing text) — the same content at two extraction
qualities, competing for the same top-k slots.

**Root cause:** `chunk_id` was `md5(f"p{page}_c{i}_{chunk[:50]}")` — derived from the
chunk **text** and omitting the source. Improving extraction changed the text, which
changed the id, so Chroma **added** new chunks and kept the old ones. Nothing replaced
anything. This also explains the earlier "✅ Indexed 30 chunks!" that left the total at
121: identical text → identical ids → silent no-op.

**Fixed:** ids are now `{document_id}_p{page:04d}_c{index:03d}` (document + position, not
text) and writes use `upsert`. The database was reset on 2026-08-22.

### 6.2 🟢 Chunk overlap had a floor but no ceiling
`CHUNK_OVERLAP` was enforced as "carry whole trailing sentences until ≥ 100 chars" with no
upper bound, so **one long trailing sentence satisfied the floor on the first carry-back
and was copied whole**. Measured on varied prose with a 760-char trailing sentence:

| | before | after |
|---|---|---|
| worst neighbour overlap | **95%** | **23.8%** |
| fully-nested chunk pairs | yes | 0 |
| duplicate `content_hash` | — | 0 |

**Why it mattered:** two near-identical chunks embed to near-identical vectors, score
almost the same, and occupy **two of five top-k slots with one idea** — silently halving
effective `TOP_K`.

**Fixed:** `MAX_CHUNK_OVERLAP = 250`, checked *before* accepting each sentence; the
ceiling wins over the floor. Carry-back skips the first unit so consecutive chunks can
never be fully nested. Confirmed on real OCR data (`bhagya` p6: 115 and 113 chars carried,
inside the 100–250 band).

**Method note:** the first verification pass produced a false positive — the synthetic
text repeated one identical sentence, so every chunk was trivially a substring of another.
Containment checks need *distinct* content to mean anything.

### 6.3 🟢 The indexer's rate-limit pacing never executed
`index_document` pre-slices into batches of 20 and then calls
`_embed_texts(batch, batch_size=len(batch))`, so `_embed_texts`'s internal loop always
runs exactly one iteration and its guard `if i + batch_size < len(texts)` is never true.
Measured sleeps: **0 at every corpus size** (20/45/91/200 chunks). The 30-second 429 retry
still works; the proactive 1 s spacing does not.
`retriever.retrieve` has no batching, pacing *or* retry — one unpaced call per query,
which is why a 20-question `evaluate.py` run trips a burst-rate quota while indexing the
same corpus succeeds. The limit hit is requests-per-minute, not volume.

**Confirmed in production 2026-08-24, then fixed.** Indexing `essence-of-hinduism.pdf`
(566 chunks, 29 batches) died with an uncaught
`429 RESOURCE_EXHAUSTED … online_prediction_requests_per_base_model`. Two separate
defects, not one:
1. The pacing sleep never ran (double-batching, above).
2. The single retry was **not itself inside a `try`**, so a second 429 escaped as a raw
   traceback and abandoned the document. One fixed 30 s wait cannot reliably clear a
   per-minute window that has already been saturated.

Fix: `_embed_batch()` retries with exponential backoff (10 → 20 → 40 → 80 s,
`EMBED_MAX_ATTEMPTS=5`) and raises a clear RuntimeError if still exhausted;
`_embed_texts()` now receives the full list and does its own batching, so
`EMBED_BATCH_DELAY_S` genuinely fires between batches. Non-quota errors are re-raised
immediately rather than retried. Verified with a fake client: pacing sleeps now equal
batches−1 at every size (20→0, 45→2, 91→4, 200→9, 1459→72), backoff doubles, a transient
429 recovers, and a `ValueError` raises without retrying.

**Cost of the pacing:** 72 s across the entire 1877-page corpus — negligible against one
429 that discards a whole document's embeddings.

**Method note:** the first verification reported exactly 2× the expected sleeps. The cause
was a leftover `time.sleep(1)` from the original implementation sitting directly below the
new pacing block — the edit added the fix without removing what it replaced. Found by
patching `time.sleep` with a stack-capturing stub and printing call sites, which named
both lines immediately. Counting an effect is not the same as knowing what produced it.

### 6.4 🟢 Chunk metadata rebuilt
Was `{page, source}`. Now: `source_type`, `document_id`, `source_name`, `page_number`,
`chunk_index`, `extraction_method`, `content_hash`. `extraction_method` was **already
returned per page by `ocr_engine` and silently discarded** — so until now there was no way
to ask whether retrieval failures concentrate in OCR'd pages versus text-layer pages.
`document_id` hashes the **casefolded** name, which neutralises the macOS
case-insensitivity bug (`bhagya-…` typed, `Bhagya-…` stored, two documents from one file).

### 6.5 🟢 The evaluation set cannot measure what we most need to measure
All 20 questions in `evaluation/questions.json` expect `CIL.pdf`, and CIL was ~75% of the
indexed corpus (91/121 chunks). `source_precision` therefore scored ~1.00 for **structural
reasons** — the majority document wins by default. The set contains **no negative
questions** (no answer in the corpus), so it also cannot detect that retrieval returns
`TOP_K` chunks whether or not any are relevant. *The dataset, not the metric, is the
bottleneck.*

### 6.6 🟢 Generation-side contamination: the prompt omits the source
`rag_engine` builds context as `[पृष्ठ {page} / Page {page}]:\n{text}` — page only, **no
source name**, in both `ask` and `ask_with_sources`. With several books indexed the model
receives five "Page N" blocks from different documents with no way to attribute them,
while rule 4 instructs it to cite page numbers. Citations are therefore ambiguous across
books, and facts from unrelated documents can be blended as one source. The console
*display* was fixed in `b077950`; the **prompt** was not.

**Fixed 2026-08-22.** Context lines are now `[{source} · पृष्ठ {page} / Page {page}]`,
built by a single `_build_user_message()` shared by both callers — the duplication was
what let the two paths drift apart in the first place. Also in the same pass: `ask` now
**raises** instead of returning failures as answer text (it previously caught bare
`Exception` and returned `"❌ Error generating answer: …"`, which the evaluation harness
would have scored as a real answer, and which gave the two sibling functions opposite
error contracts); `response.text` is checked for None/empty via `_answer_text()`, since a
safety block or early stop otherwise propagates `None` as an answer; and the dead
`ask_simple()` plus unused `Panel`/`Markdown` imports were removed. The two CLI call sites
in `app.py` now render a raised error as a message — `cmd_chat` continues the session and
skips the failed turn rather than writing it into `chat_history`.

**Correction to an earlier note in this session:** `chat_history` is *not* unbounded.
`app.py` trims it to the last 6 messages and stores only the bare question/answer text,
not the ~4 KB context block. The context is rebuilt fresh each turn.

### 6.7 🟢 OCR interleaves page artwork into prose
`bhagya-bada-ya-karm.pdf` p6, seen in the new `inspect` viewer: structurally intact Hindi
with garbage spliced **inside** sentences — `ऐसी घटनाएं हो जाती हैं KS जिनसे…`,
`कई बार ऐसा भी ॥ (6 "1 Lhe कि…`. Almost certainly decorative illustrations being read as
text. This is not edge noise that could be trimmed; it sits inside the text that gets
embedded, degrading every vector on the page — and it is invisible to hit-rate/MRR.

### 6.8 🟢 Answer-language mismatch is a weak-instruction problem, not a bug
English questions were answered in Hindi. Three compounding causes: the system prompt's
**first line** (strongest position) said "Hindi textbooks"; the language rule was #3 of 7;
and Hindi answers re-entered `chat_history`, reinforcing themselves. Diagnostic tell: the
answer language *flipped* between two same-language turns — a hard rule never flips.
Rule ordering has since been changed so the language rule is #1. ⚪ Untested whether that
alone is sufficient; flipping the opening line's language would likely invert the bug
rather than remove it.

### 6.9 ⚪ No relevance threshold / abstain path
`retrieve()` always returns `TOP_K` regardless of distance, and nothing inspects
`distance`. There is no "I don't know" path. Intended approach is to calibrate a distance
cutoff the way `text_quality.py` calibrated `engword%`/`junk%` — find the gap between
clusters — **but it is not yet known that a gap exists**, because `evaluate.py` does not
persist `distance` into its result rows. Instrument first, look at the distributions, and
only then choose the mechanism; if they overlap, prompt-side abstention is the right tool
instead.

### 6.10 Tooling added
`python app.py inspect [source]` — chunk viewer: picks from indexed documents, prints each
chunk with page, index, length, extraction method and content hash, and **highlights the
region carried over from the previous chunk**. Built because 6.1 went undetected for weeks
purely because nothing rendered the stored text.

## 7. OCR backend benchmarking (2026-08-22)

### 7.1 🟢 Local Tesseract baseline
`benchmark_ocr.py`, `bhagya-bada-ya-karm.pdf` (12 pages, fully scanned), 300 DPI, M-series
Mac. Saved to `evaluation/ocr_bench_tesseract.json`.

| Metric | Value |
|---|---|
| Wall total | **32.8 s** for 12 pages |
| OCR | 28.3 s (**86.4%**) |
| Rasterization | 4.4 s (13.6%) |
| Median OCR / page | 2.57 s |
| Avg page | 7.11 MP, **4.0 MB PNG** |

### 7.2 🟢 The page image size, not the compute, is the constraint on a hosted OCR API
At 300 DPI a page renders to **~4 MB of PNG**. Sending 12 pages to a hosted API means
uploading ~48 MB. At a 10 Mbps uplink that is ≈38 s of upload alone — **longer than the
entire local run (32.8 s)**. Any "move OCR to the cloud for speed" argument has to clear
that bar first, and neither compute location changes it. Implications before deploying
anything: send JPEG rather than PNG, or hand the API the PDF directly and skip local
rasterization entirely. Choosing the transport matters more here than choosing the host.

### 7.3 🟢 Tesseract time tracks text density, not pixel count
Every page is ~7 MP, but OCR time ranges **0.59 s → 3.38 s**, correlating with character
count (83 chars → 2032 chars), not with page area. Rasterization is the pixel-bound half
and is roughly constant. Consequence: per-page timings cannot be extrapolated across
documents of different text density, so a benchmark must state which document it used.

### 7.4 🟢 Local Tesseract has no cold-start penalty
Page 1 (2.21 s) was *faster* than the median (2.96 s) — it is a sparse cover page.
`pytesseract` shells out to the binary with no model preload, unlike EasyOCR's ~100 MB
load. This matters for the comparison: a scale-to-zero Cloud Run service **would** pay
container start-up, so cold and warm requests must be reported separately rather than
averaged.

### 7.5 ⚪ Not yet measured
Cloud Vision `DOCUMENT_TEXT_DETECTION` and Tesseract-on-Cloud-Run. Both need APIs enabled
on the GCP project (`cloudexplore-502215`) and will incur charges. Only 86.4% of local
wall time is OCR, so relocating that compute cannot beat a ~1.16× ceiling on this document
before network and cold start are added.

## 8. Upload and folder ingestion (2026-08-26)

### 🟢 Repo-verified
- A file on disk is **not proof that it is indexed**. Upload checks the vector index first;
  an existing but unindexed file can now be indexed instead of returning `409`.
- Folder upload keeps each file's relative path, accepts top-level PDF/TXT/MD files, and
  skips nested, unsupported, or over-500 MB files.
- Folder selection depends on browser-provided `webkitRelativePath`. A native label linked
  to the directory input is more reliable than programmatically clicking a hidden input.

## 9. Application state and source navigation (2026-08-26)

### 🟢 Repo-verified
- React memory alone loses conversation history on reload. SQLite now stores threads,
  answers, citations, timings, and timestamps across sessions.
- A working click handler can still look broken when its destination scrolls off-screen.
  The evidence panel now stays viewport-fixed while the conversation scrolls.
- Document citations need a safe file-serving route, not only a source name. Nested local
  PDF/TXT/MD sources now open without allowing paths outside `data/`.

## 10. Embedding quota: two metrics, and which one binds is your choice (2026-08-26)

There are **two** enforced limits, and batch size decides which one you hit:

| metric | triggered by | seen at |
|---|---|---|
| `online_prediction_requests_per_base_model` | many small requests | `EMBED_BATCH_SIZE = 20` |
| `embed_content_input_tokens_per_minute_per_base_model` | few large requests | 250 x 1010 chars |

This reconciles two readings that looked contradictory during the session. The
token limit is the true ceiling; the request limit is one a small batch size
walks into needlessly. An earlier version of this section asserted the quota was
purely token-bound -- that was measured on oversized batches only.

### 🟢 Measured

**The metric, named by the API itself:**
`aiplatform.googleapis.com/embed_content_input_tokens_per_minute_per_base_model`
(Vertex express mode, `gemini-embedding-001`, `LLM_BACKEND=vertex`.)

**The value: ~100,000 input tokens/minute.** Not published anywhere reachable --
`serviceusage.googleapis.com` returns 403 and the 429 body carries NO numeric
limit. Measured by bisection instead: a 250 x 1010-char request (~99,400 tokens
at the measured 2.54 chars/token for Devanagari) succeeds, and the very next
request fails regardless of size -- so one such request consumes essentially the
whole window.

**It is NOT an express-mode artifact.** Tested directly: a second client built
with ADC + explicit `project`/`location` (standard Vertex, `us-central1`, not
express) hits the *same* metric at the *same* threshold -- one 99,400-token
request accepted, the next rejected. Switching auth from `VERTEX_API_KEY` to a
service account would gain nothing, so don't spend time on it. The limit is the
project's real regional quota.

**Correction on an earlier reading.** The `serviceusage` 403 was first attributed
here to "no billing account" (`PreconditionFailure` subject 110002). That was
wrong: billing IS enabled on the project. The actual cause is that
`cloudbilling.googleapis.com` and `serviceusage.googleapis.com` are not enabled
for this principal, so the quota simply cannot be read via API -- an
API-activation issue, not a billing one. Read it in the Console instead:
IAM & Admin -> Quotas, filtered to `aiplatform.googleapis.com`.

**Caveat on the 429 body (Vertex express):** it contains only
`{code, message, status}` -- no `google.rpc.QuotaFailure`, no `RetryInfo`. The
RetryInfo parsing added in `3290e16` therefore never fires on this backend
(it degrades to exponential backoff, and does work on the Developer API).

Google does not publish per-model embedding limits for express mode, and three
doc pages fetched during this session were JS nav shells with no quota tables.
The 429 payload is the only reliable source: `google.rpc.QuotaFailure` names the
metric, `google.rpc.RetryInfo` gives the wait the server actually wants.

**The proof that it is tokens, not requests** — three consecutive probes:

| probe | instances x chars | total chars | result |
|---|---|---|---|
| 1 | 250 x 1010 | 252,500 | OK (21.9s) |
| 2 | 200 x 1010 | 202,000 | **429 tokens_per_minute** |
| 3 | 250 x 795 | 198,750 | **429 tokens_per_minute** |

Probes 2 and 3 were *strictly smaller* than probe 1 and still failed, because
probe 1 had drained the per-minute token window. Under a request-count limit the
smaller requests would have been safer. They were not.

**Why this matters more than it sounds.** `config.py` carried a comment asserting
"the binding constraint is requests-per-MINUTE". That is wrong, and it points at
the opposite fix: under an RPM limit, raising `EMBED_BATCH_SIZE` gives
proportional relief; under a TPM limit it gives **none**, since 250 chunks carry
exactly the tokens of 20 chunks x 12.5. Comment corrected in place.

**Batch size is still worth tuning — for throughput, not for quota.**
- API ceiling is **250 instances**; `n=400` returns HTTP 400 "too many instances".
- Larger batches are *faster per chunk*: 23.8 chunks/s at n=250 vs 13.0 at n=50.
- 250 x 1010 chars (252,500) is proven to fit in one request.

### 🟢 The cost of a non-resumable embed stage, observed

`मनुस्मृति-सम्पूर्ण.pdf` (509 pages, 1030 chunks) failed at **54%** of embedding
during the folder run. `index_document` computes every batch before storing any,
so ~556 successfully embedded chunks were discarded — token quota spent, nothing
kept — and a rerun must spend it again. Quota exhaustion causing work loss
causing more quota pressure.

Note the OCR cache did not help here: the router sent only **4 of 509** pages to
Vision (`505 direct · 4 ocr`), so the loss was entirely in the stage that is not
yet resumable. Making the embed loop checkpoint per batch makes *hitting* the
limit cheap, which is more robust than trying never to hit it.

### 🟢 Fixed, and verified on the file that failed

`index_chunks` now stores each batch as soon as it is embedded, and a rerun
re-embeds only what is missing. Chunk ids are already deterministic, so Chroma
serves as its own checkpoint -- no second on-disk format to keep in sync.

**Verification:** the same `मनुस्मृति-सम्पूर्ण.pdf` that died at 54% completed
cleanly on the next run -- 1030 chunks, from cached OCR, at $0 Vision cost.
Progress is now observable mid-document: during a 4842-chunk file the collection
count rises continuously instead of staying frozen until the end.

**One trap worth remembering.** The natural way to write the resume check --
`already.get(chunk_id) != chunk.get("content_hash")` -- looks correct and is not.
A missing id yields `None`, an absent `content_hash` yields `None`, and
`None == None` marks an unstored chunk as already embedded, dropping it
silently and forever. Resume must require POSITIVE evidence: id present, both
hashes present, hashes equal.

Partial documents also had to become detectable. Chunks now carry `chunk_total`
and `indexed_file_names()` counts a document as indexed only when complete --
otherwise a document killed partway would look done and never be finished.

### 🟢 Multi-region round-robin DOES multiply throughput -- tested

Embedding quota is enforced per project, per region, per base model, so
spreading requests across regions multiplies aggregate throughput. Verified
directly rather than assumed (project `cloudexplore-502215`, ADC auth):

| check | result |
|---|---|
| 5 regions reachable | us-central1, us-east4, europe-west1, asia-east1, asia-southeast1 -- all OK, dim=3072 |
| **quota independent per region** | us-central1 saturated to a confirmed 429, then **all four other regions served immediately** |
| **vectors identical across regions** | `exact=True`, `max_delta=0.00e+00`, `cosine=1.00000000` |

**The vector-identity check is the one that decides usability.** Round-robin is
only safe because the same input yields a byte-identical vector in every region.
Had they differed even in the last decimal, mixing regions would have quietly
degraded every future search -- worse than being rate limited, and invisible.
Re-verify this before adopting round-robin for any NEW embedding model.

**Unexpected: `us-central1` is among the slowest regions from here.**

| region | single-content latency |
|---|---|
| asia-southeast1 | 1216 ms |
| asia-east1 | 1416 ms |
| us-central1 | **2692 ms** (what the pipeline used) |
| us-east4 | 2725 ms |
| europe-west1 | 2984 ms |

So region choice is worth ~2.2x on latency independently of the quota gain.

**Two flaws in the round-robin snippet as usually written:** it builds a fresh
`genai.Client` per request, re-paying credential lookup and TLS setup every time
-- the same bug already fixed for the Vision client (`ocr_engine._get_vision_client`)
-- and it shuffles randomly, which wastes calls on regions already known to be
limited. Cache one client per region and rotate deterministically, skipping
regions with a live cooldown.

**No ADC migration needed -- the API key works regionally.** An earlier version
of this note claimed round-robin required moving off `VERTEX_API_KEY` to ADC.
Tested and wrong on both halves:

* The project API key authenticates fine against regional endpoints
  (`{region}-aiplatform.googleapis.com/v1/projects/{p}/locations/{region}/...`),
  returning 200 from us-central1, asia-southeast1, europe-west1 and asia-east1.
  Google's express-mode docs say endpoints "use the global endpoint and don't
  include projects or locations" -- that describes what express mode *provides*,
  not a restriction on where the key is accepted.
* Quota is independent per region under key auth too: us-central1 was saturated
  to a 429 and the other three regions served immediately.

**The SDK accepts a key together with a region**, so no hand-rolled HTTP is
needed either:

```python
genai.Client(vertexai=True, project=PROJECT, location=REGION, api_key=VERTEX_API_KEY)  # works
genai.Client(enterprise=True, location=REGION, api_key=VERTEX_API_KEY)                 # 403
```

The comment in `llm_client.get_client()` -- "Passing project/location alongside
api_key is rejected by the SDK" -- is true only for `enterprise=True`. With
`vertexai=True` it works, which is what makes regional rotation a small change
rather than an auth migration.

**Never pass the key as `?key=...` in the URL.** Round-robin snippets usually do.
Query strings land in server logs, proxy logs and history; use the
`x-goog-api-key` header (or the SDK, which handles it).

### 🔴 `gemini-embedding-2` is not an escape from the quota -- tested

Proposed on the theory that it runs on global rather than regional per-model
limits and would bypass the bottleneck. **It does not.** Probed directly
(script kept out of the repo; project `cloudexplore-502215`, `us-central1`):

| claim | result |
|---|---|
| `gemini-embedding-2` is available | **404** -- not found on this project. Only `gemini-embedding-2-preview` resolves. |
| bypasses the request bottleneck | **No.** 429 after **4 requests in 4.6 s**, naming the SAME metric `online_prediction_requests_per_base_model`. |
| higher throughput | **No** -- it cannot batch, so a corpus needs ~1 request per chunk. |

**The batching failure is silent, which makes it dangerous.** Sent 5 *distinct*
texts, it returns **1 embedding** -- no error, no warning, four inputs dropped.
Confirmed with distinct inputs specifically to rule out deduplication of
identical strings. Wired naively into `index_chunks`, `upsert()` would receive
20 ids and 1 vector: a length mismatch at best, silently misaligned vectors at
worst.

The SDK's `t_is_vertex_embed_content_model()` predicate flags the model as
one-content-per-request and is *correct*, but its `ValueError` guard does not
fire, because `t_contents()` first collapses a list into a single Content with
multiple parts. `gemini-embedding-001` is explicitly exempted from that
predicate and takes the PREDICT path, which is why 20-per-call works today.

Two quota metrics exist and which one binds depends on batch size:
`online_prediction_requests_per_base_model` (hit by many small requests, e.g.
`EMBED_BATCH_SIZE = 20`) and `embed_content_input_tokens_per_minute_per_base_model`
(hit by few large ones, e.g. 250 x 1010 chars). Tuning batch size trades one
for the other; the token limit is the real ceiling.

---

## 11. Latency reference (measured to 2026-08-26)

Compact index of every timing measured so far, so future tuning starts from data
rather than from re-running the same experiments. Detail and method live in
`OCR_NOTES.md`; this is the summary sheet.

### Where the time actually goes

Two independent bottlenecks, with different fixes:

| Stage | Bound by | Fix that works | Fix that does NOT |
|---|---|---|---|
| OCR | **upload bandwidth** (page image size) | grayscale encoding, parallel workers | more CPU |
| Embedding | **input tokens/minute quota** | pacing, resumability, raising quota | bigger batches |

### OCR — per page (Google Vision, 300 DPI)

Encoding dominates, because latency tracks *bytes uploaded*, not page content:

| Encoding | Size | Latency | Similarity vs RGB |
|---|---|---|---|
| RGB PNG (original) | 6437 KB | ~9.0 s | 1.0000 |
| **Grayscale PNG (adopted)** | 1626 KB | **2.64 s** | **0.9994** |
| JPEG q85 (rejected, lossy) | 618 KB | 1.33 s | 0.9967 |

### OCR — parallelism (8 dense pages, interleaved across 3 reps)

| workers | median | speedup |
|---|---|---|
| 1 | 32.2 s | 1.00x |
| 2 | 12.2 s | 2.63x |
| 4 | 11.6 s | 2.78x |
| **8 (default)** | **9.8 s** | **3.27x** |

Effective throughput at 8 workers: **~1.2 s/page**, so a 428-page scan OCRs in
roughly 9 minutes. Sequential would be ~19 minutes.

### OCR — backend medians (90 pages, warm)

| backend | Gita | Arthasastra | History | $/page | failures/90 |
|---|---|---|---|---|---|
| tesseract | 3.31 s | 3.76 s | 5.74 s | $0 | 3 |
| **vision** | 8.57 s* | 2.12 s | 1.84 s | $0.00150 | **0** |
| gemini | 6.62 s | 3.96 s | 5.32 s | ~$0.0016 | 21† |

\* pre-grayscale figure.  † harness lacked backoff; not a model verdict.

### Embedding — quota-bound, not latency-bound

Raw speed is fast; the quota is what costs time.

| batch size | throughput |
|---|---|
| 50 | 13.0 chunks/s |
| 100 | 17.2 chunks/s |
| **250 (API max)** | **23.8 chunks/s** |

But the effective ceiling is **~100,000 input tokens/minute** (§10). Devanagari
measures **2.54 chars/token**, so:

| | value |
|---|---|
| Effective embedding rate | **~254,000 chars/minute** |
| One 250 x 1010-char request | ~99,400 tokens -- nearly the whole window |
| Chars per page (measured, 1178 pages) | ~1,292 |
| **Embedding floor** | **~0.24 s per 1000 chars = ~0.31 s/page** |

A 500-page book is ~646,000 chars = ~254,000 tokens = **~2.5 minutes of
quota-limited waiting minimum**, no matter how fast the network is.

### Whole-file observations (folder run, 2026-08-26)

| file | pages | routing | chunks | outcome |
|---|---|---|---|---|
| Jateen_Resume | 1 | 1 direct | 6 | 7.06 s cold / **2.56 s cached** |
| essence-of-hinduism | 241 | 239 direct, 2 ocr | 809 | indexed |
| मनुस्मृति-सम्पूर्ण | 509 | 505 direct, 4 ocr | 1030 | **failed at 54% embedding** |
| SRIMAD-BHAGAVAD-GITA | 428 | **0 direct, 428 ocr** | 857 | indexed, ~$0.64 |

**The router is the single biggest cost lever, and it is invisible in latency
terms.** essence-of-hinduism and Manusmriti are 750 pages combined but sent only
**6 pages** to Vision -- $0.009 instead of $1.13. Any change that bypasses the
router (e.g. Vision async batch on whole PDFs) forfeits that.

### Cache effects

| operation | cold | cached |
|---|---|---|
| 12-page PDF extract | 23.3 s | **0.0 s** |
| 1-page PDF, full index | 7.06 s | 2.56 s |

---

## 12. Vision failures can originate in the execution environment (2026-08-27)

### 🟢 Repo-verified
- The restricted indexing run reported repeated `ServiceUnavailable` errors. A controlled
  DNS check could not resolve `vision.googleapis.com` there but succeeded with network
  permission. ADC refresh, one Vision call, and eight concurrent blank-image calls then
  succeeded; this supports a local network restriction, not a confirmed Google outage.
- OCR logging prints only the exception class, then replaces the underlying error with
  generic credentials/billing/quota advice. Preserve the cause to distinguish DNS failures
  from service errors. Blank-image success does not establish full-page OCR performance.
- Interrupted Upanishad extraction left no OCR cache or indexed document. Dharma-Sastra
  remained at 1,500 stored chunks: an attempted resume is not evidence of completion.

## 13. The frontend degrades clearly when the API is offline (2026-08-27)

### 🟢 Repo-verified
- With the Next.js server listening on port 3000 and nothing listening on port 8000, the
  workspace loaded without browser console errors, labelled the library `Server unavailable`,
  and converted a submitted suggestion into a retryable inline error. Sidebar toggling and
  conversation reset still worked, but library data, retrieval, uploads, and source links
  could not be exercised without the FastAPI server.
- Older folder-indexed sources retain a repository-relative `data/` prefix. Retrieval still
  works because ChromaDB uses that stored source verbatim, but the document endpoint previously
  resolved it as `data/data/...` and returned 404. Accepting both prefixed and canonical paths
  restored original-document links; a live evidence link opened the cited PDF at Page 82.

## 14. Citation UI consistency depends on durable chunk identity (2026-08-27)

### 🟢 Repo-verified
- Some saved citations opened the original document while others opened the evidence dialog.
  The difference was not source type: `rag_engine` had omitted Chroma's stable `chunk_id`, so
  rows without an ID fell back to direct links. The retriever and response serializer now
  preserve that ID. Legacy rows resolve only when `source + page + stored preview` identifies
  exactly one indexed chunk; a live pre-ID Mahabharata citation recovered its complete passage.

## 15. Website conversations are stored but not sent back to the model (2026-08-27)

### 🟢 Repo-verified
- The `/ask` endpoint accepts a `conversation_id` and records each exchange, but calls
  `ask_with_sources(question)` with only the latest question. That function sends one user
  message containing the retrieved passages and current question; it does not load or include
  earlier exchanges. Conversation history therefore supports display and persistence, not
  contextual follow-ups. A question such as “What happened after that?” cannot rely on the
  previous answer unless its missing subject is independently recoverable from the new query.

## 16. Source language can overpower an underspecified response-language rule (2026-08-27)

### 🟢 Live-verified
- With predominantly Hindi passages, the English question `Hi, what is "Bhagya"?` produced
  Hindi when the prompt merely said to identify the user's language. Selecting language from
  the question's grammatical structure *before* considering passages fixed that observed case.
  Four live Vertex/Gemini trials then behaved as specified: English framing → English,
  Romanized-Hindi framing → Hindi, and explicit English/Hindi requests each overrode the
  surrounding sentence language. This is a four-case regression check, not proof over every
  possible code-switched question.

## 17. Configured Gemini model limits (verified 2026-08-27)

### 🟢 Repo + API verified
- The effective runtime configuration is Vertex AI with `gemini-2.5-flash`; retrieval uses
  `TOP_K=5`. A live Gemini Developer `models.get` call for the same model returned
  `input_token_limit=1048576` and `output_token_limit=65536`, matching Google's model page.
  These are model capacity limits, not request, rate, or embedding quotas. Vertex Express
  prediction accepts its API key, but its `GetPublisherModel` metadata endpoint rejected that
  key with HTTP 401 and requires principal-bearing OAuth credentials; the Developer Models API
  therefore supplied the direct metadata verification.

## Main conclusion
*(Revised 2026-08-22.)* The earlier conclusion — "retrieval experimentation is no longer
the bottleneck, Gemini retrieval is reliable" — **does not survive §6**. It rested on
numbers measured against a corpus that held two generations of the same pages (§6.1), with
chunks that could be 95% duplicates of each other (§6.2), scored by a question set
structurally incapable of detecting the failure mode in question (§6.5).

None of that means Gemini retrieval is bad. It means **we do not currently know how good it
is.** The real bottleneck was never the retriever — it was the absence of anything that
could look at the stored data. That is now addressed (§6.10), the storage layer is fixed,
and the index has been reset.

Next focus, in order: re-index at 300 DPI on the fixed chunker → instrument `distance` in
`evaluate.py` → write negative and multi-document eval questions → re-baseline → only then
decide about thresholds, rerankers or embedding-model comparisons.

## Reproducibility backlog (port from cloud session → this repo)
- [ ] MiniLM retriever + Hit@1/@3/@5 evaluator + the ~100-query eval set.
- [ ] BGE reranker stage (+ latency measurement).
- [ ] Word/char TF-IDF fallback retriever.
- [ ] Tesseract-vs-EasyOCR speed benchmark script.

## Open items (this repo, from §6)
- [ ] Re-index every PDF at `PDF_DPI = 300` on the fixed chunker (config says 300; the old
      corpus was built at 200, so committed config and measured results disagreed).
- [ ] Record per-chunk `distance` in `evaluate.py` rows — blocks §6.9.
- [x] ~~Negative eval questions (no answer in corpus) + questions targeting non-CIL
      documents~~ — done: 14 unanswerable questions (§8.2) and a corpus-wide set with a
      hard tier (§12). The v1 `questions.json` has been deleted; it is recoverable from
      git history if an old result ever needs reproducing.
- [x] ~~Put `source_name` into the generation prompt~~ — done, §6.6.
- [x] ~~Collapse the duplicated context/prompt construction in `rag_engine`; make `ask`
      raise like `ask_with_sources`~~ — done, §6.6.
- [ ] Pace / retry `retriever.retrieve`, or batch it in `evaluate.py` — §6.3.
- [x] ~~Make the embedding loop resumable per batch~~ — done, §10. Verified on the
      file that had failed at 54%.
- [ ] Raise `EMBED_BATCH_SIZE` from 20 — §10. At 20 the run hits the *request*
      metric needlessly; a larger batch shifts it onto the token ceiling, which is
      the real limit. API max is 250 instances.
- [ ] Round-robin embeddings across regions — §10. Quota is per-region and vectors
      are byte-identical across regions, so this is ~5x aggregate throughput.
      Needs ADC with an explicit `location`; keep generation on its own client
      rather than switching `llm_client.get_client()` wholesale.
- [ ] Switch the embedding region away from `us-central1` — §11. It is among the
      slowest from here (2692 ms vs 1216 ms for `asia-southeast1`).
- [ ] `app.py` still stores the *typed* filename casing; `document_id` makes this harmless
      for identity, but `status` can display a name that differs from the file on disk.
- [ ] Derive `COLLECTION_NAME` from the embedding model before any cross-model
      comparison, or two models' vectors mix in one collection. Note the
      embedding-001 vs embedding-2 comparison that originally motivated this is a
      dead end (§10): `gemini-embedding-2` 404s on this project and cannot batch.
      Multi-region round-robin is NOT affected — same model, byte-identical vectors.

## 8. RAG evaluation on the current corpus (2026-08-27)

Corpus at time of run: **21,360 chunks** — 17,472 PDF, 3,241 plain text, 407 YouTube;
13,903 of them OCR'd. State this number with any result: §8 is not comparable to the
Jul 18 `results_*.json` files, which ran against a ~50x smaller corpus with the same
questions. Reporting those side by side would attribute corpus growth to retrieval quality.

### 8.1 🔴 The old question set could no longer measure anything
All 20 questions in `questions.json` target `CIL.pdf` — **64 of 21,360 chunks (0.3%)**.
Written 2026-08-14, when that PDF was most of the corpus. A sample retrieval for its first
question returned five chunks of ancient Indian history at distances 0.4815–0.4880, i.e.
no discrimination at all. The set measures needle-in-a-haystack retrieval of a recruitment
notice, not the study-assistant use case.

### 8.2 🟢 `questions_v2.json` — 25 questions, weighted to the corpus
18 factual, 6 unanswerable, 1 interpretive; 16 hi / 9 en. Every question carries an
`evidence` field: a verbatim snippet from the source chunk, authored **from** text pulled
out of Chroma rather than from knowledge of the books. `verify_questions.py` re-checks all
25 against the live index (evidence present for answerable; no lexical support for
unanswerable). 25/25 pass.

### 8.3 🔴 `page_number` only locates content in PDFs
| source_type | chunks | distinct page_number |
|---|---|---|
| pdf | 17,472 | 889 |
| text | 3,241 | **1 (always 1)** |
| youtube | 407 | **all None** |

The old `_rank_expected_source` required `chunk["page"] in expected_pages`, so it matched
plain text *trivially* and YouTube *never* — the Mahabharata and every transcript were
invisible to scoring. v2 adds `match: page | source | none`.

### 8.4 🟢 Results at k=5 (retrieval only)
| Metric | Value |
|---|---|
| retrieval_hit_rate | **1.0** (19/19) |
| mean_reciprocal_rank | 0.9605 |
| mean_source_precision | 0.8421 |
| hit_rate_hi / hit_rate_en | 1.0 / 1.0 |
| avg retrieval latency | 1.193 s |

MRR < 1.0 comes from one question: `arth-01-hi` ("how should the king divide day and
night?") found the Arthasastra only at **rank 4** — the history textbook outranked it for a
question naming a specific text.

**Read the hit rate as an upper bound, not a grade.** The questions were authored from
distinctive passages, which is what makes ground truth auditable but also makes them easier
than organic student questions. It shows retrieval is not broken; it does not show it is
good on hard queries.

### 8.5 🟢 Distance separates answerable from unanswerable perfectly — no threshold exists yet
| | min | median | max |
|---|---|---|---|
| answerable (19) | 0.127 | 0.227 | **0.307** |
| unanswerable (6) | **0.330** | 0.397 | 0.482 |

**Zero overlap.** A gate at ~0.32 would have rejected all six unanswerable questions and
none of the nineteen answerable ones. Nothing in the pipeline does this today: retrieval
always returns k chunks, so an unanswerable question still reaches the LLM with five
confident-looking passages, and refusal depends entirely on `SYSTEM_PROMPT` rule 2. This is
the cheapest available accuracy win — it costs one comparison and saves a generation call.
Six points is a small sample; widen before hard-coding the constant.

### 8.6 🔴 Duplicate indexing is spreading, and it understates the metric
Documents indexed under **both** `X.pdf` and `data/X.pdf`, 2026-08-27:

| document | copy A | copy B |
|---|---|---|
| SRIMAD-BHAGAVAD-GITA.pdf | 857 | 240 (partial) |
| CIL.pdf | 64 | 64 |
| 027. Bhagya Likhne Ki Kalam - Karm.pdf | 79 | 81 |
| bhagya-bada-ya-karm.pdf | 30 | 30 |
| Jateen_Resume.pdf | 6 | 6 |

Three documents at 11:30, five by 12:10 — this is ongoing, not historical. Cost is
measurable: listing both spellings in `expected_sources` moved `mean_source_precision`
**0.8105 → 0.8421**, so the bug was understating retrieval quality by 3.2 points. The two
copies also carry *different* `content_hash` values, meaning the same PDF OCR'd twice
produced different text.

### 8.7 ⚪ Not yet measured
`--generate`: citation accuracy, keyword recall, and `refusal_rate_on_unanswerable` — the
first real test of rule 2. The harness now passes `top_k` through to `ask_with_sources`
(it previously retrieved a second time at the config default, so citation scores described
a *different* retrieval than the ranks beside them). Not run here to avoid generation spend.

## 9. OCR concurrency: first execution, and two defects (2026-08-27)

The parallel OCR path had been committed, merged to `main` and pushed without ever
running. `test_ocr_concurrency.py` stubs the backend and `_prepare_page`, so it exercises
the real control flow offline at zero API cost.

### 9.1 🟢 The reassembly was already correct
4 of the 7 tests passed against the unfixed code. `as_completed()` returns futures in
completion order, and the `futures[fut] -> page number` mapping restored page order
correctly for contiguous ranges, non-contiguous page lists, and batch sizes that are not a
multiple of `workers * 2`. Sequential and parallel produce identical dicts. The concurrency
design was sound; only its error handling was not.

### 9.2 🔴 One corrupt page aborted the whole document — but only at workers > 1
`prepared = [(n, _prepare_page(doc[n])) for n in batch]` sat OUTSIDE any `try`, while the
sequential path had the same call INSIDE one. So a page that PyMuPDF could not render was
tolerated at `OCR_MAX_WORKERS=1` and fatal at `OCR_MAX_WORKERS=8` — the docstring claimed
the two paths matched. Confirmed by test: `ValueError: corrupt page` escaped `_ocr_pages`
and would have discarded the other 19 pages. Each page is now prepared in its own `try`.

### 9.3 🔴 Systemic failure produced a complete-looking index built from nothing
A blanket per-page `except Exception` meant rejected credentials or a disabled Vision API
yielded `""` for every page, printed 1877 red lines, and returned normally — so indexing
proceeded and stored empty text. Before concurrency, the first page's exception aborted the
run with one clear error.

Fixed with a per-run tally: once the first `min(5, pages)` pages have all failed and NO page
has succeeded, raise. Threshold rather than "every page failed" because waiting for a
1877-page document to fail costs the full retry schedule (4 attempts, 2-8s backoff) on each
one. Trade-off: a document whose first five pages genuinely fail but whose sixth would have
succeeded now raises. That is the right default — five consecutive failures with nothing
working is far more often a config problem than five bad scans.

### 9.4 ⚪ Still unmeasured
Real end-to-end throughput at `OCR_MAX_WORKERS=8`. The 3.3x figure in commit `7813768`
came from a run whose provenance this session did not verify.

## 10. Distance gate: separation holds at 33 questions (2026-08-27)

Widened §8.5 from 6 unanswerable questions to 14 (terms scanned for absence first, not
guessed).

| | n | min | max |
|---|---|---|---|
| answerable | 19 | 0.127 | **0.307** |
| unanswerable | 14 | **0.330** | 0.498 |

**Gap +0.023.** Any threshold in (0.307, 0.330) separates all 33; midpoint **0.318**.

> ⚠️ **Superseded by §12.** Measured against the hard question tier the gap becomes
> **−0.010** and the bands overlap. The 0.318 threshold below would wrongly refuse 4
> answerable questions. The warning in this section turned out to be correct; read §12
> before using any number from here.

🟢 **Adjacency does not fool it.** "Who killed Gandhi?" asked against a corpus containing
Gandhi's own writings scored 0.405 — mid-band, not borderline. The gate keys on whether the
answer is present, not on topic overlap.

🔴 **Do not hard-code 0.318 yet.** The 19 answerable questions were authored from
distinctive passages, so that side is optimistically low; a vaguer real question would score
higher and could cross the line. The gap is 0.023 wide — roughly one question's noise.
Before shipping a hard refusal, add answerable questions written *without* looking at the
source text and re-measure. Until then it is a good soft signal (warn/log), not a gate.

⚪ Method note: a naive substring scan for absence gives false positives on short tokens —
"gst" matched 340 times inside unrelated words. Absence terms are now >= 6 characters.

## 11. System prompt rewritten against measured corpus properties (2026-08-27)

Corpus the prompt actually serves: 21,360 chunks, 45 documents. **68.1% OCR**, 15.2% text,
14.8% direct PDF, 1.9% transcript.

| Defect in the old prompt | Evidence | Fix |
|---|---|---|
| Never forbade parametric knowledge | corpus is famous texts (Gita, Gandhi, Mahabharata) | rule 1: no prior knowledge, *even when certain* |
| "Answer from context" invited using *all* context | `mean_source_precision` 0.842 → ~16% of retrieved chunks are off-document | rule 5: ignore irrelevant passages; flag disagreement |
| Refusal was untemplated | `_is_refusal` is a heuristic that can miss | rule 2: fixed opening sentence, EN + HI |
| No citation format at all | 3 label shapes exist in `_build_user_message` | rule 4: `(source, locator)`, copied verbatim |
| Silent about OCR noise | 68.1% of chunks | rule 6: read through damage, quote as-is, admit corruption |

🟢 **Rule numbers are load-bearing.** Five references across `evaluate.py`, `rag_engine.py`
and this file cite "rule 2" (refusal) and "rule 4" (citation). Both kept at their numbers;
a comment above `SYSTEM_PROMPT` now says so.

🔴 **`_has_expected_citation` cannot score text-file documents.** Its regex is
`(?:page|पृष्ठ)\s*[:#-]?\s*N`, but text chunks are labelled "Document section N". Verified:
`(mahabharata.txt, Document section 301)` → `False`. A *correct* citation on 15.2% of the
corpus scores as a miss. Harness bug, not a prompt bug — fix before trusting citation
accuracy.

⚪ **Unmeasured, and deliberately so.** `--generate` has never run, so there is no before
baseline; this change rests on reasoning about the numbers above, not on a measured
improvement. Baseline prompt is preserved at `git show 018e208:rag_engine.py` for a later
A/B. Success is a *pair* of metrics: `refusal_rate_on_unanswerable` up **and**
`false_refusal_rate` still 0 — a prompt told to distrust famous texts can start refusing
answerable ones. Cheapest next test: `--generate` on the 14 unanswerable questions plus 4-5
answerable Gita/Gandhi questions (~19 calls, not 33).

⚪ Caveat: rule 2 now mandates wording that `_is_refusal` matches, so that metric partly
measures instruction-following rather than honesty. Read it alongside answer text.

## 12. Hard question tier: the easy set was measuring phrasing (2026-08-27)

§8.4's hit rate of 1.0 was an upper bound, as flagged. Confirmed by building a **hard twin
for each of the 19 answerable questions** — same `evidence`, same page, same match mode,
only the question wording changed. One variable, so the delta is attributable.

🟢 **The old questions leaked their own answers.** `evaluation/check_leakage.py` scores two
channels — a token shared with `answer_keywords`, and naming the source document:

| tier | leaking |
|---|---|
| easy | **14 / 19** |
| hard | 0 / 19 |

`arth-04-hi` asked "अर्थशास्त्र के अनुसार 'त्रिपौरुषी' छाया किसे कहते हैं?" — the document
name *and* both answer keywords, in the question.

🟢 **Retrieval degrades, but does not collapse.**

| metric | easy | hard |
|---|---|---|
| hit_rate (page granularity) | 1.0 | 0.9474 |
| hit_rate (source granularity) | 1.0 | **1.0** |
| MRR | 0.9605 | 0.8246 |
| source_precision | 0.8421 | 0.7789 |

Rank worsened on 4/19, `source_precision` on 7/19. **Contamination falls faster than hit
rate** — removing the document name is precisely what stopped the retriever discriminating
between books, so this is the expected shape, not a regression.

🔴 **The distance gate from §10 is dead as specified.** §10 warned "a vaguer real question
would score higher and could cross the line." It does:

| | easy | hard | unanswerable |
|---|---|---|---|
| max | 0.307 | **0.340** | — |
| min | — | — | **0.330** |

**Gap +0.023 → −0.010.** The bands now overlap. The 0.318 midpoint would wrongly refuse
4 answerable questions (`ess-03-en-hard` 0.340, `hist-03-en-hard` 0.338, `yt-01-hi-hard`
0.334, `arth-03-en-hard` 0.320).

**The finding is that the gate does not survive harder phrasing — not that the threshold
needs retuning.** No re-fitted constant should be quoted from here: any value that looks
good on these 52 points was chosen using those same 52 points. Distance stays a soft
warning signal (log/warn), never a refusal gate.

⚪ The one hard miss, `arth-03-en-hard`, is a *page* miss, not a source miss
(`source_precision` 1.00): it found the right book, wrong page. With `match: "page"` that
scores identically to retrieving nothing, which overstates the failure.

⚪ Method bug worth remembering: the first version of `check_leakage.py` reported 0 leaks in
Hindi because `[^\W\d_]+` drops Devanagari matras (Unicode Mn/Mc), shattering every Hindi
word into single consonants below the length floor. A checker that silently passes
everything looks exactly like a clean result.

## 13. Development reload needs two layers (2026-08-27)

🟢 **Repo-verified mechanics:** `dev.py` delegates ordinary Python and React/CSS edits
to Uvicorn reload and Vinext/Vite HMR, while its own small fingerprints restart processes
for Git branch changes, backend/frontend environment files, and frontend dependency
manifests. Seven unit tests cover restart classification, Git worktree pointer resolution,
and fixed command construction. The launcher refuses occupied ports instead of killing an
unrelated process and owns both child process groups for one-step shutdown.

⚪ **Not live-verified in this session:** the complete dual-server lifecycle was not started
because ports 3000 and 8000 were already occupied by the user's running application. The
process-level restart and `Ctrl+C` behavior should be smoke-tested after those servers stop.

## 14. Saved history reaches generation, but not retrieval (2026-08-29)

🟢 **Code-path verification:** `/ask` loads the selected conversation's recent exchanges
from SQLite and passes them to `ask_with_sources`. That function bounds the history and
includes it in Gemini's generation contents, so persistence and prompt delivery are wired.

🟢 **Fixed:** retrieval previously called `retrieve(question, ...)` with only the latest
user text. A conservative bilingual gate now detects likely references and continuations;
one constrained model call rewrites those into a standalone search query before embedding.
Self-contained questions skip the extra call, and API/malformed-output failures fall back to
the original question rather than blocking an answer. Generation still receives the user's
original wording plus bounded history and retrieved evidence.

🟡 **The unmerged multi-query branch did not close this gap.** Commit `c8655de`
plans searches from the current question plus the document catalog, but its planner also
receives no conversation history. The new contextualizer is therefore a separate prerequisite
that can later feed that branch's multi-query planner.

🟢 **Runtime-verified:** 96 offline unit tests pass in the project `.venv`, including
English, Hindi, and Romanized-Hindi follow-ups, self-contained-query bypass, malformed rewrite
fallback, conversation isolation, and preservation of history in the generation request.

## 15. Generated answers use multi-locator citations (2026-08-29)

🟢 **Live UI evidence:** the model does not always emit one page per citation. A saved answer
contained `(panchatantra.pdf, पृष्ठ 82, 87)`, so a renderer limited to one numeric locator
left disruptive citation text in the prose. The citation renderer now resolves each listed
page against that answer's retrieved-source metadata and renders one compact, hover/focusable
marker per matching passage. At a 1280×717 viewport, the literal citation count was 0, both
source markers were present, and the retrieved-preview tooltip became visible on hover.

## 16. One citation can contain several unsupported locators (2026-08-29)

🟢 **Live UI evidence:** a generated answer combined three documents and several page numbers
inside one parenthetical citation; some pages were absent from the five retrieved passage
labels. The renderer correctly left that unmatched text visible instead of presenting it as
verified evidence. Citation rule 4 now requires one exact retrieved source/locator per pair of
parentheses, separate citations for separate passages, and forbids invented locators.

## 17. Persisted prompt editing needs the exchange identity (2026-08-29)

🟢 **Repo- and live-verified:** saved exchanges already had stable UUIDs, but `/ask` returned
only the conversation UUID while the frontend displayed a temporary client UUID. A newly sent
prompt therefore could not be edited reliably until the conversation was reloaded. `/ask` now
returns the persisted exchange UUID. Editing regenerates with history strictly before that
exchange, preserves its UUID, and deletes later exchanges as the abandoned conversation branch.

## 18. Citation prose and citation transport need different syntax (2026-08-29)

🟢 **Repo-verified:** parsing ordinary parenthetical citations made document references
indistinguishable from normal prose and encouraged multi-source lists inside one parenthesis.
Rule 4 and the renderer now share an explicit `⟦source, locator⟧` transport format. Only a
delimiter pair matching retrieved metadata becomes a numbered hover marker; unmatched markup
remains visible so unsupported citations are not silently presented as verified evidence.

## 19. Prompt images should be ephemeral, not conversation memory (2026-08-29)

🟢 **Repo-verified design:** an optional PNG, JPEG, or WebP attachment is passed directly to
Gemini alongside the grounded prompt for one generation. SQLite stores only the typed question,
answer, and retrieved sources, so later turns cannot silently reuse an image the interface no
longer displays. Clipboard and file-picker inputs converge on the same client/server validation.

## 20. Stored answers make citation syntax a compatibility contract (2026-08-29)

🟢 **Repo-verified design:** conversation answers persist as rendered model text, so changing the
prompt's citation delimiters does not migrate earlier answers. The renderer must recognize legacy
syntax while still requiring an exact returned source-and-locator match before showing a marker.
🟢 **Live-data/code verification:** a mixed-format snake answer showed this compatibility boundary:
citations emitted with ordinary single brackets remained readable literal text, while exact
machine-delimited citations became numbered markers. The current tokenizer recognizes `⟦…⟧` and
legacy `[[…]]`, but not single-bracket `[source, locator]` text.

## 20. Web fallback needs an explicit sufficiency contract (2026-08-29)

🟢 **Repo- and API-doc-verified:** nearest-neighbor retrieval always returns top-k passages when
the collection is nonempty, so an empty result cannot represent semantic insufficiency. The
document-grounded generation now returns a typed sufficiency decision. Only an insufficient
latest exchange offers an opt-in Google Search call; grounding annotations, rather than model-
invented URLs, supply the clickable web citations. This also keeps potentially billable search
requests behind an explicit user action.

## 21. Follow-up answers can outlive their retrieval set (2026-08-30)

🟢 **Live-data/code verification:** a language-only follow-up reused exact citations from its
preceding grounded answer, while the new turn stored five unrelated retrieval results. The UI
therefore left citation transport syntax visible because none of that turn's source metadata
could validate it. Language-only transformations now trigger contextual retrieval, and answer
sources are reconciled against current plus previously validated conversation sources in exact
citation order. Stored conversations are repaired when read; their SQLite rows are not rewritten.

## 22. A top-k limit can be consumed by repeated evidence (2026-08-30)

🟢 **Repo-verified:** stable chunk-ID fusion removes the same physical passage returned by
several query rewrites, but it does not remove different chunks containing the same copied text.
Three-word-shingle containment catches exact hashes and heavily overlapping near-copies before
they occupy the reranker shortlist. The former post-rerank character packer could silently remove
model-selected evidence, so normal answers now preserve the reranker's validated 5–15 passage
selection exactly; explicit evaluation cutoffs remain fixed and comparable.

🟢 **Official-model-doc-verified:** `gemini-2.5-flash-lite` is a stable endpoint with structured
output support and is described by Google as optimized for high-frequency, lightweight work.
The reranker uses it through a dedicated setting instead of inheriting the answer model. Source:
https://ai.google.dev/gemini-api/docs/models/gemini-2.5-flash-lite

## 23. Gemini 2.5 Flash has a dated migration boundary (2026-08-30)

🟢 **Official-doc- and live-verified:** Google lists Gemini 2.5 Flash retirement as October 20,
2026, while Gemini 3.5 Flash-Lite is GA and has a retirement date of July 21, 2027 or later.
Google also recommends `google-genai` 2.0.0+ and model-managed sampling for Gemini 3.x. The
application now uses `gemini-3.5-flash-lite` for generation, query planning, and reranking;
Vertex smoke tests completed both plain-text generation and schema-constrained passage reranking
without fallback. This supersedes the model choice recorded at the end of finding 22. Source:
https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/migrate

## 24. Adaptive reranking improves source purity, not every retrieval metric (2026-08-30)

🟢 **Matched-evaluation evidence:** on the same 52 questions, direct top-5 retrieval and the
planner/RRF/Gemini 3.5 Flash-Lite pipeline both achieved 0.9737 hit rate. The pipeline raised
overall source precision from 0.8105 to 0.8298 and hard-question precision from 0.7789 to
0.8333, while overall MRR fell from 0.8925 to 0.8728 and average retrieval latency rose from
1.116s to 5.066s (4.54×). It selected 5.654 chunks on average; one unanswerable case returned
three because only three distinct candidates survived deduplication. This is evidence of a
source-purity/context improvement, but not an unconditional robustness or speed improvement.

## 25. Document image generation should share retrieval evidence (2026-08-30)

🟢 **Repo-verified design:** document Image mode previously grounded the image model only in the
user question and the already-generated text answer. It now prepares the visual prompt while the
RAG engine still holds the selected chunks, so the answer and image calls share one retrieval run
and the image model receives the original user request plus source-labelled retrieved passages.
The generated text answer remains in the response but is not used as the document image's factual
input. Web + Image mode retains its answer-based prompt because that path has no retriever context.

## 26. One API request is not one provider request (2026-08-31)

🟢 **Repo-verified design:** a conversational document question can fan out into a
contextualization call, query-planning call, batched query embedding, reranking call, and final
generation call. One ingestion request can fan out further into concurrent OCR calls and many
embedding batches. Public protection therefore combines per-user endpoint buckets with global
operation concurrency; endpoint requests-per-minute alone cannot represent or cap upstream quota
consumption. Redis-backed atomic state is required when multiple API workers or replicas share the
same paid provider credentials.

## 27. Authentication without storage filters is not tenant isolation (2026-08-31)

🟢 **Repo-verified design:** before authentication, conversation IDs, source paths, generated-image
IDs, document fingerprints, and vector queries all addressed global stores. Verifying a user only
at the HTTP boundary would therefore still permit horizontal data exposure or retrieval leakage.
Private tenancy now carries the verified Firebase UID through SQLite predicates, opaque filesystem
roots, tenant-specific Chroma document IDs and metadata, catalog aggregation, citation lookup, and
the vector query's `where` filter. Security tests exercise denial across all four storage surfaces.

## 28. A passing suite is not a verified boundary (2026-08-31)

🟢 **Repo-verified:** every pre-existing API test overrides `get_current_user` with an
`is_admin=True` stub, so 143 green tests described the happy path of a stubbed identity and said
nothing about denial. Three gaps survived that suite. The frontend had never been compiled:
`tsc --noEmit` failed on `auth-gate.tsx` because `@cloudflare/workers-types` retypes
`Response.json()` as `Promise<unknown>`, so `payload.detail` was an error rather than the usual
silent `any`. Nothing ever wrote the `admin` custom claim that `require_admin` reads — Firebase
claims are settable only through the Admin SDK — leaving `POST /index/folder` unreachable until
`grant_admin.py` was added; note that `LEGACY_ADMIN_UID` assigns legacy rows and does *not* confer
the role. And `_require_trusted_origin` guarded only the two `/auth/*` routes, which is harmless
under the default `SameSite=Lax` but not under the `SESSION_COOKIE_SAMESITE=none` deployment the
README itself documents; the origin check now runs as middleware over every state-changing method that arrives with
an `Origin` header. A request without one still passes, so this stops browser-driven CSRF
and not scripted clients, which is the whole threat it is meant to cover. Verified
empirically: an allowed origin preflights and POSTs normally, a forged origin gets a 403.

## 29. Tenant filters that fail open on `None` (2026-08-31)

🟢 **Repo-verified design, deliberate:** `conversation_store` predicates read
`(? IS NULL OR c.owner_id = ?)` and `retriever.retrieve_many` omits the Chroma `where` clause
entirely when `owner_id is None`. Passing no owner therefore returns *every* tenant's rows rather
than none. Every API call site passes the verified `user.uid`, and cross-tenant denial is tested
across conversations, generated images, document paths, and vector queries — so there is no leak
today. The default exists because CLI and maintenance tools legitimately need unscoped access. It
is recorded here because it inverts the safe default: a forgotten argument degrades silently into
a full-corpus read instead of raising `TypeError`. Making `owner_id` required on the API-facing
functions, with an explicit unscoped entry point for the CLI, is the follow-up.

## 30. The rate limiter has not run against Redis (2026-08-31)

⚪ **Not live-verified in this session:** every rate-limit test uses `enabled=False` or a stub
context manager, and `fakeredis` is not installed, so no Lua script has executed against a real or
emulated Redis. Reading them, the `KEYS`/`ARGV` index arithmetic matches each caller's argument
layout and the `{identity_tag}` / `{global}` hash tags keep multi-key scripts in one cluster slot,
but reviewed-and-plausible is not verified. Installing `fakeredis` and exercising bucket
exhaustion, refill, lease expiry, and heartbeat renewal is the outstanding work before the
concurrency claims can be stated as fact.
