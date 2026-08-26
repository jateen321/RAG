# Indexing / OCR Quality — Issues Log

Running notes on text-extraction problems found while making the app production-ready.
Goal: get **clean** text into ChromaDB, because noisy chunks poison retrieval no matter how good the re-ranker is.

## The PDFs we tested (data/)

| PDF | Type (by text layer) | ~chars/page | Notes |
|-----|----------------------|-------------|-------|
| essence-of-hinduism.pdf | Digital | 1880 | Rich text layer; page 0 empty (cover). Need to sample a content page. |
| CIL.pdf | Digital | 2411 | English text layer is CLEAN; but we OCR it anyway → manufactured noise. Hindi TBD. |
| bhagya-bada-ya-karm.pdf | Scanned (image-only) | 0 | Genuinely needs OCR. |
| 027. Bhagya Likhne Ki Kalam - Karm.pdf | "Digital" but broken | 1259 | Text layer is GIBBERISH: `^mΩ` obIZ{ H$r H$b_` |

## Problems identified

1. **Indiscriminate OCR (wrong-tool routing).**
   `ocr_engine.py` always rasterizes every page and runs EasyOCR — even for digital PDFs
   that already have a clean text layer (CIL, essence). This *creates* noise where none existed
   (e.g. `COAL INDIA LIMITED` → `Coall IIndia CMAHARATNA COMPAHY`).
   → Fix: route each PDF to the right method (direct-text vs OCR) instead of always OCR.

2. **Legacy / non-Unicode font encoding (the "ASCII code-points" problem).**
   027.pdf *has* a text layer, but it uses a legacy Devanagari font (Krutidev/DevLys family) that
   maps Hindi glyphs onto ASCII code-points. So `page.get_text()` returns junk like `^mΩ` obIZ{`.
   The page looks like Hindi to a human, but the bytes are meaningless.
   → Consequence: the stashed `DIRECT_TEXT_MIN_CHARS` heuristic (trust text if length > 50) would be
     FOOLED — 1259 chars ≫ 50, so it would extract gibberish. It checks *quantity*, not *quality*.
   → Fix: validate text quality (detect gibberish), not just length. For these, OCR may be better.

3. **OCR quality noise (EasyOCR on Hindi/English).**
   Where OCR really is needed (scanned bhagya-bada-ya-karm.pdf), EasyOCR output is noisy:
   `Eighty` → `Bighty`, `compensation` → `CONPENSATION`, mixed Devanagari/Latin numerals (`२025-26`).
   → Fix: OCR post-processing / better settings / evaluate alternative engines.

## Key insight
"Has a text layer" ≠ "has USABLE text." Extraction is a **routing problem**, not one method for all.
Three cases: clean-digital → extract directly; scanned → OCR; legacy-font → detect + OCR (or font map).

## Is OCR "the main issue"?
Not exactly. OCR is the right tool for only 1 of 4 PDFs (the scanned one). The bigger issue is
**applying OCR indiscriminately** + **no text-quality check**. OCR *quality* is a separate, real issue
only for the genuinely-scanned files.

## Text-quality check (how to decide "trust the text layer?")

First attempt — **character-class ratio** (is it mostly letters?) — **FAILED**: legacy Devanagari
fonts map glyphs to ASCII *letters*, so gibberish scores a high "letter ratio" (027 = 65%) and passes.
Lesson: a check that looks at character *class* can't tell real English from ASCII-letter gibberish.

Refined rule (verified on all 4 PDFs) uses two agreeing signals:
- **junk-symbol %** — legacy fonts sprinkle `© ° · ¢` etc. → 027 = 13.7% vs clean docs = 0.0%.
- **real-word ratio** — fraction of tokens that are real dictionary words → 027 = 16.6% vs 78–88%.
  This is the strongest signal: it checks *meaning*, not appearance.

Routing rule derived:
```
no text                                   → OCR   (scanned)
Devanagari-heavy + low junk               → trust (real Hindi Unicode)
Latin-heavy + low junk + high real-words  → trust (real English)
else                                      → OCR   (legacy-font gibberish)
```

Measured verdicts: essence ✅trust, CIL ✅trust, bhagya 🖼️OCR, 027 ⚠️OCR (gibberish caught).

### Caveats (not production-ready yet)
- Dictionary is **English-only** (`/usr/share/dict/words`); real Unicode Hindi needs deva%/junk% instead.
- The **"real Unicode Hindi" branch is designed but UNVERIFIED** — no such PDF in our test set.
- `/usr/share/dict/words` exists on macOS, maybe not in a prod Linux container → bundle a wordlist or use `langdetect`.

## Per-page routing + threshold calibration (text_quality.py)

Built `choose_method(text)` deciding PER PAGE: "direct" vs "ocr". Per-page proved its worth —
essence's cover pages route to OCR while its 237 body pages use the clean text layer.

BUT first thresholds were too strict → **false positives** (clean English wrongly sent to OCR):
- CIL p.3: clean text, engword 53% (< 55 cutoff) → wrongly OCR
- essence p.1 (title page): engword 53% — proper nouns/initials aren't dict words → wrongly OCR
- essence p.139: clean, junk 4.9% (> 3 cutoff, fancy quotes) → wrongly OCR

**Lesson:** a false positive is doubly bad — it re-manufactures OCR noise on a clean page.
Thresholds were set at the EDGE of the clean cluster. Fix: move them into the GAP between clusters:
- engword%: gibberish=16.6 ⟷ clean=53–88  → set cutoff ~40 (was 55)
- junk%:    clean=0–4.9 ⟷ gibberish=13.7    → set cutoff ~8 (was 3)
- Also handle near-empty pages (e.g. "[ xx ]") with a min-text guard.

### Retune result (engword 55→35, junk 3→8) — SUCCESS
- CIL: 14/1 → 15/0 (false positive fixed). essence: 237/4 → 239/2 (title page & p139 now trusted).
- 027 still 0/34, bhagya still 0/12 — gibberish/scanned still fully caught.
- Robustness gem: 027 p20 had engword=34% (just under 35!) but junk=18% caught it → requiring
  BOTH signals = defense in depth. Independent signals back each other up.
- Remaining OCR pages in essence are legit: p0 (image cover) + p20 ("[ xx ]" sparse page).

## Dharma-Sastra Vol.1 (old scanned English book) — OCR quality probe

- 638 pages, **0 direct / 638 ocr** — pure scan, no text layer. Router's OCR branch validated.
- `junk ≈ 0%` on sampled pages → genuine OCR, NOT legacy-font gibberish (contrast 027). Our
  metric correctly separates the two failure modes.
- OCR is **readable but noisy**: `engword%` = 56 / 71 / 78 on pages 25 / 100 / 300 (~1 in 4
  English tokens misread). Examples: `Introdwctaont`→Introduction, `Sam2ita`→Samhita,
  `iour`→four, `day$`→days.
- **RAG risk:** the *proper nouns* (Yajnavalkya, Apastamba, Samhita, Parasara) are the MOST
  mangled — OCR has no language model for Sanskrit names — yet they're exactly the query terms.
- **`engword%` caveat:** it rises with prose vs. name-heavy pages, so it partly measures
  proper-noun density, not just OCR quality. Rough signal, not a clean score.
- **Lead to test:** stray Devanagari/digit intrusions (`णf`, `Sam2ita`, `day$`) come from OCR
  running `["hi","en"]` on an English book. Try **`en`-only** OCR → should reduce them.

## Tesseract vs EasyOCR on Dharma-Sastra (repo-verified, same 3 pages)

Engine swapped to **Tesseract** (`pytesseract`, `lang="eng+hin+san"`, via Homebrew).
Same pages, same rasterization (DPI 200). `engword%` measured on the OCR output:

| Page | EasyOCR | Tesseract | Δ |
|------|---------|-----------|---|
| 25   | 56.68   | **65.10** | +8.4 |
| 100  | 71.19   | **75.21** | +4.0 |
| 300  | 77.90   | **84.59** | +6.7 |

- Tesseract beats EasyOCR on all 3 pages; common English words are much cleaner
  (`Introdwctaont`→`Introduction`, `iound`→`found`, `णf`→`of`).
- Residual errors concentrate on **IAST-diacritic Sanskrit names** (`Y4jiawalkya`,
  `Kdtydyana`) — the accent marks (ā, ñ, ś) fool classical OCR. Hard for any non-LLM engine.
- One Devanagari intrusion (`गला`, p100) from running `hin+san` on an English page;
  kept because the Devanagari books (Manusmriti) need those packs.
- Speed: no 100MB model preload; visibly faster (not rigorously timed).

## 🔴 Manusmriti — corrupt Devanagari text layer fools the router (keystone finding)

- 509 pages, routing 505 direct / 4 ocr. The 4 OCR pages are the **TOC** (dot-leaders dilute
  deva% below 30 → misroute; minor).
- **The real problem:** the 505 "direct" pages have a **corrupt Unicode text layer**. The PDF's
  Hindi font has a broken `ToUnicode` map, so `get_text()` returns WRONG Devanagari codepoints
  (systematic `द/व → ि` swaps): `िूध` instead of `दूध`, `यति िूसरे` instead of `यदि दूसरे`.
- **Tesseract OCR of the rendered page is CORRECT** where the text layer is garbage:
  text-layer `स्वायोंभुिो मनुधीमातनिों` → Tesseract `स्वायंभुवो मनुर्धीमानिदं`. Confirmed on pp.30, 200.
- **Router blind spot:** corrupt page scores `deva 97.7%, junk 0.0%` → passes as "clean Hindi" →
  routes `direct` → extracts scrambled text. `deva%`/`junk%` check APPEARANCE, not MEANING.
  We have `engword%` (dictionary) for English but **no Devanagari real-word check** — so this
  slips through. This is the cousin of 027's legacy-font issue, but mapped to valid Devanagari.
- **Implications:** (a) for this book the correct action is to OCR, not trust the layer;
  (b) the router needs a Hindi/Sanskrit wordlist signal to detect corrupt-but-valid text layers;
  (c) reinforces the Tesseract choice — OCR can beat a broken text layer.

## ✅ Implemented: language-agnostic corrupt-layer defense (layer-vs-OCR spot check)

- `ocr_engine._verify_text_layer(doc)`: sample up to `LAYER_CHECK_SAMPLE=3` 'direct' pages,
  OCR them, compare to the text layer via `difflib.SequenceMatcher` on NFC + whitespace-
  normalized strings. Take the **median** similarity.
- Calibration (strongly bimodal): **Manusmriti 0.02** (corrupt) vs **CIL 0.93 / essence 0.999**
  (clean). Threshold `LAYER_CHECK_MIN_SIMILARITY=0.4`, set low to avoid false alarms on merely-
  noisy OCR.
- Wired into `extract_text_from_pdf`: if median < threshold → distrust layer → force OCR for the
  whole document (`direct` pages get overridden to `ocr`).
- Verified end-to-end: Manusmriti → "corrupt, forcing OCR" (trust=False); CIL → "OK 0.93",
  15/15 direct.
- Why this matters: it's **language-agnostic** — needs no per-language wordlist, so it also
  covers the corrupt-**English**-layer case, and makes the Devanagari-wordlist idea (defense A)
  a cheap optimization rather than a necessity.
- Trade-offs: costs ~3 OCR pages per document (insurance, tunable); it's a whole-document
  decision (coarse — a mixed doc with only some corrupt pages is forced entirely to OCR).

## Open / next steps
- [ ] Add a min-text guard for sparse pages (e.g. essence "[ xx ]").
- [ ] Get/confirm a clean Unicode-Hindi PDF to test the untested Hindi-trust branch.
- [ ] Decide handling for legacy-font PDFs (027): OCR, or map the legacy font to Unicode?
- [ ] Wire `choose_method` into `ocr_engine.py` (compare with stashed version); re-index; re-check chunk quality.

---

# 2026-08-25 — Three-backend OCR bake-off (tesseract vs google_vision vs gemini)

Harness: `benchmark_ocr.py --compare` / `--backend gemini`. Each page is rasterized
**once** at `PDF_DPI=300` and the identical image handed to every backend, so
rasterization is never charged twice. Raw per-page text + timings are saved in
`evaluation/ocr_compare_*.json` and `evaluation/ocr_gemini_*.json`.

Corpus: 90 pages — Gita 250–299 (50), Arthasastra 300–319 (20), History 300–319 (20).

## 🔴 KEYSTONE CORRECTION: the Manusmriti "corrupt text layer" finding was an artifact

The earlier §"Manusmriti — corrupt Devanagari text layer fools the router" is **wrong**,
and so was the calibration derived from it.

`difflib.SequenceMatcher` defaults to `autojunk=True`: for any sequence of ≥200 elements
it treats every element occurring in >1% of the sequence as *junk* and excludes it from
matching. That heuristic is tuned for source-code diffs. **Devanagari has a small
effective alphabet**, so on a page of Hindi roughly 23 characters cover ~83% of the text —
and all of them get discarded.

Measured on the same three sample pages `_verify_text_layer()` picks:

| Document | shipped (`autojunk=True`) | correct (`autojunk=False`) |
|---|---|---|
| मनुस्मृति-सम्पूर्ण.pdf | **0.0235** → "corrupt, force OCR" | **0.8571** → clean, trust layer |
| essence-of-hinduism.pdf | 0.9981 | 0.9981 |
| CIL.pdf | 0.9173 | 0.9377 |

Manusmriti's text layer is **not corrupt** — it agrees with OCR at 0.86. The English
documents are barely affected, which is exactly why the old `0.02 vs 0.93` looked
"strongly bimodal": that spread was measuring **script**, not corruption.

Consequences that were live in production:
- `LAYER_CHECK_MIN_SIMILARITY = 0.4` was calibrated against a false bimodality.
- **Every clean Unicode-Devanagari PDF was being force-OCR'd** — for Manusmriti, 509
  needless OCR calls that *replace a clean text layer with noisier OCR output*.
- The open item "test the untested Hindi-trust branch" would have failed for this reason.

**Fixed** in `ocr_engine._verify_text_layer()` (`autojunk=False`). The same bug was
present in the new benchmark's agreement metric and is fixed there too — it had reported
two near-identical pages as 0.008 similar when the true value is 0.979.
→ **Action: `LAYER_CHECK_MIN_SIMILARITY` needs re-calibrating now that the scale changed.**

## 🟢 Vision latency is upload-bound, not OCR-bound

Vision's per-page time on the Gita was bimodal: 23 pages at ~1.7 s, 27 pages at ~9.0 s,
for the *same* amount of text (982 vs 1018 mean chars). The split tracks PNG size, not
page content — the scan mixes two encodings (981 KB vs 6474 KB).

Measured on Gita p.250/254/255 (baseline = the 6.4 MB RGB PNG the pipeline was sending):

| Encoding | Size | Vision latency | Text similarity vs baseline |
|---|---|---|---|
| RGB PNG (was shipping) | 6437 KB | ~9.0 s | 1.0000 |
| **Grayscale PNG** | 1626 KB | **2.64 s** | **0.9994** |
| JPEG q95 | 1114 KB | 2.09 s | 0.9982 |
| JPEG q85 | 618 KB | 1.33 s | 0.9967 |

Colour carries no information an OCR engine uses. **Adopted grayscale PNG** in both
`ocr_engine._ocr_with_google_vision()` and `benchmark_ocr._ocr_vision()`: ~3.4× faster for
0.9994 similarity, and lossless in the ways that matter. JPEG is faster still but lossy,
and the fidelity cost is not worth it for a corpus built once.

Also fixed: `_ocr_with_google_vision()` constructed a new `ImageAnnotatorClient()` **per
page**, so every page paid credential lookup + TLS setup. Now built once and reused.

## 🟢 Latency / cost / quality across 90 pages

| Document | backend | warm median | mean | chars | script_runs | failed pages | $/page |
|---|---|---|---|---|---|---|---|
| Gita 250–299 | tesseract | 3.31 s | 3.53 s | 49924 | 10.6 | **2** | $0 |
| | vision | 8.57 s* | 5.79 s | 50082 | 4.9 | 0 | $0.00150 |
| | gemini | 6.62 s | 11.73 s | 48945 | 4.3 | 1† | $0.00181 |
| Arthasastra 300–319 | tesseract | 3.76 s | 3.48 s | 33303 | 11.1 | 0 | $0 |
| | vision | 2.12 s | 2.01 s | 33624 | 1.6 | 0 | $0.00150 |
| | gemini | 3.96 s | 4.13 s | 18081 | 0.5 | **10†** | $0.00127 |
| History 300–319 | tesseract | 5.74 s | 5.31 s | 67812 | 10.4 | 1 | $0 |
| | vision | 1.84 s | 2.35 s | 72129 | 5.7 | 0 | $0.00150 |
| | gemini | 5.32 s | 5.42 s | 33888 | 3.5 | **10†** | $0.00182 |

\* measured *before* the grayscale fix — expect ~2.6 s now.
† **All 21 Gemini failures were `429 RESOURCE_EXHAUSTED`** (Vertex express-mode quota),
in contiguous runs. The benchmark has no retry/backoff, unlike `indexer.py`'s embedding
path. This is a **harness limitation, not a model verdict** — the Arthasastra and History
Gemini aggregates are *not* comparable, since half the pages are missing.

**Reliability is the sharpest separator.** Tesseract raised `TesseractError (-5)` on
**3 of 90 pages** (Gita 261, 293; History 312), returning an empty page with no
indication anything was lost. Vision failed 0 of 90.

Note `chars` is **volume, not quality** — on show-through pages a higher count is
plausibly *worse*. Gemini's `script_runs` is not comparable to the other two: it emits
clean blocks because it *reformats*, not because it read the layout correctly.

Pairwise agreement (tesseract vs vision, `autojunk=False`): Gita median 0.918,
Arthasastra 0.949, History 0.980.

## 🟢 Accuracy on hand-transcribed ground truth (Gita p.275, verse 11.34, 126 chars)

| backend | CER | errors |
|---|---|---|
| tesseract | **3.97%** | `जेतासि`→`नेतासि`, `सपत्नान्`→`सपलान्` |
| google_vision | **0.00%** | — |
| gemini | **0.00%** | — |

The `सपत्नान्` failure is instructive: in this typeface the `त्न` conjunct is drawn as a
ligature that closely resembles `ल` (verified by zooming the scan 3×).

Whole-page behaviour on the same page — **Gemini was best overall**, beating Vision on
both of Vision's errors:

| detail | tesseract | vision | gemini |
|---|---|---|---|
| show-through bleed line | ✗ hallucinated `॥ < स Ht SEK HH 20 Yi Das` | ✓ ignored | ✓ ignored |
| `बहुत-से-मेरे` hyphens | ✓ | ✗ dropped | ✓ |
| `(34)` marker position | ✓ | ✗ relocated | ✓ |
| `काँपता` | ✗ `Higa` | ✓ | ✓ |
| Devanagari numeral `।।३४।।` | ✗ `1138` | ✓ | ✓ |

## ⚠️ Gemini reads the page — but silently drops what it cannot read

BG 11.34 is one of the most-quoted verses in Sanskrit literature, so a perfect
transcription could just be recitation from memory. Control: located `जेतासि रणे` via
Vision's word bounding boxes, whited out those two words, re-sent the identical prompt.

| run | `जेतासि रणे` in output? |
|---|---|
| original | present |
| masked | **absent** |

So it is genuinely reading pixels, and the Gita accuracy numbers are meaningful.

**But** it did *not* emit `[ILLEGIBLE]` as the prompt explicitly instructs — it just
closed the gap silently. Tesseract produces detectable garbage and Vision produces
detectable misordering; an LLM produces *fluent plausible text* and *invisible holes*.
Quiet data loss is harder to catch downstream than noise. **Do not default to `gemini`
for bulk indexing.**

## 🟢 Can Tesseract be tuned to beat Vision? Time yes, quality no.

Sweep on Gita p.275, scored by how much of the ground-truth verse is missing
(GT aligned against the whole page, so line-filtering cannot skew it):

| config | sec | GT miss | show-through line? |
|---|---|---|---|
| baseline `eng+hin+san` psm3 | 3.17 | 3.17% | ✗ present |
| `--oem 1` (LSTM only) | 3.12 | 3.17% | ✗ present |
| `--psm 6` | 4.13 | 4.76% | ✓ gone |
| `--psm 4` | 3.09 | 3.17% | ✗ present |
| `hin+eng` (drop `san`) | 2.59 | 5.56% | ✗ present |
| `hin` only | 2.00 | 5.56% | ✗ present |
| Sauvola threshold | 2.69 | 8.73% | ✓ gone |
| **grayscale + Sauvola** | **2.19** | **3.17%** | ✓ gone |
| gray + Sauvola + oem1 + `hin+eng` | **1.73** | 5.56% | ✓ gone |
| *google_vision (reference)* | *10.39* | ***0.00%*** | *✓ gone* |

- **Time: yes.** `grayscale + -c thresholding_method=2` is 1.4× faster than baseline
  (2.19 s vs 3.17 s) at identical verse accuracy, and dropping to `hin+eng` + `--oem 1`
  reaches 1.73 s. Against grayscale-optimised Vision (~2.6 s), tuned Tesseract is
  genuinely competitive on wall-clock — and it stays $0.
- **Quality: no.** No configuration reached Vision's 0.00%. The floor is ~3.17%.
- **Sauvola is a real but mixed win.** It eliminates the show-through hallucination and
  repairs `नेतासि`→`जेतासि`, but introduces new damage: `हतांस्त्वं`→`हतास्त्वं` (anusvara
  lost) and the English `Do you kill` → `Do $०प् प्ा11`. Net GT miss unchanged.
- Language stacking costs real time: `eng+hin+san` → `hin` alone is 3.17 s → 2.00 s, at
  the price of accuracy.

## Recommendation

`google_vision` as the default (now the shipped default): best accuracy per rupee, zero
failures in 90 pages, and ~2.6 s/page after the grayscale fix. `tesseract` when offline
or cost-constrained — pair it with `grayscale + Sauvola`, and **handle `TesseractError`,
because it will silently lose pages**. `gemini` for difficult layouts where hyphenation
and block order matter, but never unattended: add retry/backoff for 429 first, and
accept that it hides what it cannot read.

## Open / next steps (from this session)
- [ ] **Re-calibrate `LAYER_CHECK_MIN_SIMILARITY`** — the 0.4 threshold predates the
      `autojunk` fix and the similarity scale has changed.
- [ ] Add retry/backoff to the Gemini OCR path (mirror `indexer.py`'s embedding pacing)
      and re-run Arthasastra/History for a fair three-way comparison.
- [ ] Handle `TesseractError` in `ocr_engine` — currently a crash yields an empty page
      silently.
- [ ] Wire `grayscale + Sauvola` into `_ocr_with_tesseract()` if `tesseract` is kept.
- [ ] Ground truth is one 126-char verse on one page. Transcribe 2–3 more bounded regions
      (ideally from Arthasastra/History) before treating the CER ranking as general.

## 🟢 Addendum — validating the router after the autojunk fix

Re-ran `_verify_text_layer()` on four documents with the fix in place, to confirm the
corrupt-layer detector still catches what it should:

| Document | layer type | old (buggy) | fixed | verdict |
|---|---|---|---|---|
| मनुस्मृति-सम्पूर्ण.pdf | Unicode Devanagari | 0.0235 | **0.8571** | trust layer ✓ |
| A_History…India.pdf | Krutidev legacy | 0.9746 | 0.9759 | trust layer ✓ |
| CIL.pdf | English digital | 0.9173 | 0.9377 | trust layer ✓ |
| essence-of-hinduism.pdf | English digital | 0.9981 | 0.9981 | trust layer ✓ |

The History result looks alarming but is correct. `choose_method()` routes **743 of its
769 pages to `ocr` on its own** — it detects the Krutidev gibberish unaided. Only 26 pages
route to `direct`, and those are the **bibliography** pages: mostly English author names
and titles, genuinely readable. The layer check only ever samples `direct`-routed pages,
so 0.976 describes the bibliography, not the book.

The division of labour is therefore: `choose_method()` catches per-page junk;
`_verify_text_layer()` is the safety net for when `choose_method()` is *fooled* — which is
exactly the Manusmriti case, and the case that was broken. **The 0.4 threshold still
separates cleanly** on the evidence available (0.857 lowest clean vs nothing below it),
though we no longer have a verified whole-document corrupt-layer example to anchor the
low end, since Manusmriti was the presumed one.

## 🔴 Hosted OCR had no retry — one transient failure aborted a whole run

Hit live while running the validation above:

```
RuntimeError: Google Cloud Vision OCR failed: The service is currently unavailable.
```

A single transient 503 killed the run. `_ocr_with_google_vision()` had no retry, and the
Gemini path had none either — which is the entire reason 21 of 90 benchmark pages failed
with 429 RESOURCE_EXHAUSTED. `indexer.py` already paces and retries *embedding* calls
(`EMBED_MAX_ATTEMPTS`, `EMBED_BACKOFF_BASE_S`); the OCR calls had no equivalent.

**Fixed:** `_with_retry()` in `ocr_engine.py`, wrapping both hosted backends.
`OCR_MAX_ATTEMPTS=4`, `OCR_BACKOFF_BASE_S=2` (2→4→8 s). Neither a 503 nor a 429 means the
page is unreadable, so retrying is the correct response; without it a single blip aborts a
multi-hour 1877-page index.

Verified end-to-end afterwards: `_ocr_page()` on Gita p.275/276 via `google_vision`
returned 1055 / 1127 chars, matching the benchmark.

## 🟢 Threshold calibration — the low end is now anchored

The previous addendum flagged that `LAYER_CHECK_MIN_SIMILARITY = 0.4` had lost its
low-end anchor: Manusmriti was the presumed corrupt-layer example and turned out to be
clean. Resolved by measuring documents whose layers **are** genuinely corrupt, deliberately
bypassing `choose_method()`'s gate (they never reach the layer check in normal operation,
because per-page routing catches them first):

| Document | layer | median similarity (`autojunk=False`) |
|---|---|---|
| 027. Bhagya Likhne Ki Kalam | legacy font, **corrupt** | **0.0048** |
| A_History…India (Krutidev pages) | Krutidev, **corrupt** | **0.0096** |
| मनुस्मृति-सम्पूर्ण | Unicode, clean | **0.8571** |
| essence-of-hinduism | English, clean | **0.9981** |

Corrupt tops out at ~0.01; clean bottoms out at ~0.86. **`0.4` sits in an empty gap two
orders of magnitude wide** — it is well placed and needs no change. The separation is real
now, rather than the script artifact the original 0.02-vs-0.93 calibration was measuring.
→ The "re-calibrate the threshold" open item is **closed**.

## 🔴 Retry bug: the in-band error check was OUTSIDE the retry

The retry added earlier did not actually fire. Vision reports transient failures **on the
response object** rather than by raising, and the `response.error.message` check sat
*after* `_with_retry()` returned — so the retry saw a successful call, returned, and the
error escaped un-retried. Observed live:

```
RuntimeError: Google Cloud Vision OCR failed: The service is currently unavailable.
  at ocr_engine.py:141   # the in-band check, outside the retry
```

**Fixed** by moving the in-band check *inside* the retried callable. Verified by injecting
two synthetic in-band failures: the retry backed off (1s → 2s) and returned correct text
(1055 chars on Gita p.275).

Note this is the *general* shape of the bug, not a Vision quirk: **any API that reports
errors in-band needs its success check inside the retry boundary**, or the retry is
decorative. The same reasoning is why `_ocr_with_google_vision` checks
`response.error.message` at all.

Incidental data point: three genuine `503 The service is currently unavailable` responses
occurred during one working session, one of them mid-test. Hosted OCR is flaky enough that
retry is not optional for a 3653-page index.

## Corpus indexing projection (google_vision default, after the fixes)

| | pages |
|---|---|
| Text layer trusted (`direct`) | 786 |
| Sent to Vision (`ocr`) | **3653** |
| **Total** | 4439 |

Cost: 3653 units − 1000 free = 2653 billable → **$3.98**.
Time: ~**158 min** at ~2.6 s/page. Before the grayscale encoding fix the same run would
have taken ~548 min — the encoding change alone saves ~6.5 hours.

Of the 786 trusted pages, **505 are Manusmriti**, which the `autojunk` bug would have
force-OCR'd — so that fix saves both money and text quality, since the clean layer is
better than OCR of the same page.
