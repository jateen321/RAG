# Project Findings

Consolidated findings for the Hindi/English RAG app. Each result is tagged with its
**provenance** so every number is traceable:

- 🟢 **Repo-verified** — reproducible today from code + saved results in this repository.
- 🟣 **Cloud session** — measured in a separate cloud work session; the code is not yet
  ported into this repo. *Action: port the script/results back to make it reproducible here.*
- ⚪ **Hypothesis** — believed/expected, not yet measured.

Last updated: 2026-08-08.

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

Note: `evaluate.py` currently computes only **hit-rate** and **MRR** over one retriever
(`retriever.retrieve()` = Gemini embeddings). It does **not** compute Hit@1/@3/@5, and it
does **not** compare retrievers. The `results_k1 … k15` files are a **top-k sweep of the
same Gemini retriever**, not a retriever comparison.

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
- Preserve **page number, source filename, OCR engine, chunk id** as metadata.
- Character-based chunks with overlap are resilient to imperfect OCR.
- Deterministic **SHA-256 chunk ids** prevent duplicate ingestion.

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

## Main conclusion
Retrieval experimentation is **no longer the bottleneck** — Gemini retrieval is reliable on the
current test set. Next focus is the **end-to-end app**: robust ingestion (wire in `text_quality`),
config, answer generation, citations, error handling, CLI/API, tests, docs.

## Reproducibility backlog (port from cloud session → this repo)
- [ ] MiniLM retriever + Hit@1/@3/@5 evaluator + the ~100-query eval set.
- [ ] BGE reranker stage (+ latency measurement).
- [ ] Word/char TF-IDF fallback retriever.
- [ ] Tesseract-vs-EasyOCR speed benchmark script.
