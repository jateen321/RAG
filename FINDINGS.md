# Project Findings

Consolidated findings for the Hindi/English RAG app. Each result is tagged with its
**provenance** so every number is traceable:

- 🟢 **Repo-verified** — reproducible today from code + saved results in this repository.
- 🟣 **Cloud session** — measured in a separate cloud work session; the code is not yet
  ported into this repo. *Action: port the script/results back to make it reproducible here.*
- ⚪ **Hypothesis** — believed/expected, not yet measured.

Last updated: 2026-08-26.

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

## 3. TF-IDF (local lexical fallback) — 🟣 Cloud session
- Word TF-IDF works when query and document share literal words.
- Character TF-IDF (3–5 char n-grams) is more tolerant of spelling / OCR errors.
- Word + char combined = a useful **offline** fallback, but semantically weaker than Gemini.

---

## 4. OCR & ingestion

### 🟢 Repo-verified
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

## 10. Embedding quota is TOKEN-bound, not request-bound (2026-08-26)

### 🟢 Measured

**The metric, named by the API itself:**
`aiplatform.googleapis.com/embed_content_input_tokens_per_minute_per_base_model`
(Vertex express mode, `gemini-embedding-001`, `LLM_BACKEND=vertex`.)

**The value: ~100,000 input tokens/minute.** Not published anywhere reachable --
`serviceusage.googleapis.com` returns 403 (`PreconditionFailure` subject 110002,
i.e. no billing account), and the 429 body carries NO numeric limit. Measured by
bisection instead: a 250 x 1010-char request (~99,400 tokens at the measured
2.54 chars/token for Devanagari) succeeds, and the very next request fails
regardless of size -- so one such request consumes essentially the whole window.

**Raising it requires enabling billing on the project.** Express mode is the free
tier; the 403 above confirms no billing account is attached. Quota increases go
through Console -> IAM & Admin -> Quotas, filtered to `aiplatform.googleapis.com`,
and are unavailable while the project is billing-free.

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

### 🔴 Landmine

`gemini-embedding-2` matches the SDK's `t_is_vertex_embed_content_model()`
predicate and is therefore restricted to **one content per request**; batching
raises `ValueError`. `gemini-embedding-001` is explicitly exempted and uses the
PREDICT path, which is why 20-per-call works today. Switching embedding models
would silently break batching.

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
- [ ] Negative eval questions (no answer in corpus) + questions targeting non-CIL
      documents — blocks §6.5.
- [x] ~~Put `source_name` into the generation prompt~~ — done, §6.6.
- [x] ~~Collapse the duplicated context/prompt construction in `rag_engine`; make `ask`
      raise like `ask_with_sources`~~ — done, §6.6.
- [ ] Pace / retry `retriever.retrieve`, or batch it in `evaluate.py` — §6.3.
- [ ] Make the embedding loop resumable per batch — §10. A 429 at 54% currently
      discards every batch already embedded and re-spends the token quota.
- [ ] `app.py` still stores the *typed* filename casing; `document_id` makes this harmless
      for identity, but `status` can display a name that differs from the file on disk.
- [ ] Derive `COLLECTION_NAME` from the embedding model before any embedding-001 vs
      embedding-2 comparison, or the two models' vectors mix in one collection.
