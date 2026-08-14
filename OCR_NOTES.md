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
