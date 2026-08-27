"""Audit the evaluation dataset against what is actually indexed.

Two checks, because a question set can be wrong in two different ways:

1. Answerable questions must have their `evidence` snippet present in a chunk
   at the stated source (and page, when match == "page"). This is what stops
   ground truth being written from memory of a book rather than from the OCR'd
   text the retriever will actually see.
2. Unanswerable questions must have NO lexical support in the corpus. If a
   distinctive term turns up, the question is quietly answerable and it is
   measuring the opposite of what it claims to.
"""

import json
import re
import sys
from pathlib import Path

import chromadb

# Run from anywhere: the repo root holds config.py, this file lives one level down.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

_norm = lambda s: re.sub(r"\s+", " ", s).strip()

# Distinctive terms per unanswerable question. If any appears, the premise fails.
ABSENCE_TERMS = {
    "unans-01-en": ["Reliance Industries", "market capitalisation", "market capitalization"],
    "unans-02-hi": ["इलेक्ट्रिक वाहन", "इलेक्ट्रॉनिक वाहन"],
    "unans-03-en": ["T20 World Cup", "ICC Men's"],
    "unans-04-hi": ["पायथन", "मशीन लर्निंग"],
    "unans-05-en": ["WiFi password", "wifi password"],
    "unans-06-hi": ["क्रिप्टोकरेंसी", "बिटकॉइन"],
    # Batch 2. Terms picked by scanning the corpus first, not guessed. Note a
    # naive substring scan gives false positives on short tokens ("gst" matched
    # 340 times inside other words), so these are all >= 6 characters.
    "unans-07-hi": ["नाथूराम", "गोडसे"],
    "unans-08-en": ["penicillin", "पेनिसिलिन"],
    "unans-09-hi": ["क्वांटम", "quantum"],
    "unans-10-en": ["vaccine", "टीकाकरण"],
    "unans-11-hi": ["चैटजीपीटी", "chatgpt", "openai"],
    "unans-12-en": ["insulin", "इंसुलिन"],
    "unans-13-hi": ["ब्लॉकचेन", "blockchain"],
    "unans-14-en": ["मौसम पूर्वानुमान", "imd forecast"],
}


def main(path: str) -> int:
    col = chromadb.PersistentClient(path=config.CHROMA_DB_PATH).get_collection(
        config.COLLECTION_NAME
    )
    got = col.get(include=["documents", "metadatas"])
    docs, metas = got["documents"], got["metadatas"]
    corpus = " ".join(_norm(d) for d in docs)

    data = json.load(open(path, encoding="utf-8"))
    ok = fail = 0

    for q in data:
        if q["category"] == "unanswerable":
            leaks = [t for t in ABSENCE_TERMS.get(q["id"], []) if t.lower() in corpus.lower()]
            if leaks:
                fail += 1
                print(f"[FAIL] {q['id']}: premise broken — corpus contains {leaks}")
            else:
                ok += 1
                print(f"[ OK ] {q['id']}: absent from corpus (checked {len(ABSENCE_TERMS.get(q['id'],[]))} terms)")
            continue

        hits = []
        for doc, md in zip(docs, metas):
            if md.get("source_name") not in q["expected_sources"]:
                continue
            if q["match"] == "page" and md.get("page_number") not in q["expected_pages"]:
                continue
            hits.append(_norm(doc))

        if _norm(q["evidence"]) in " ".join(hits):
            ok += 1
            print(f"[ OK ] {q['id']}: evidence found ({len(hits)} chunk(s), match={q['match']})")
        else:
            fail += 1
            print(f"[FAIL] {q['id']}: evidence NOT found ({len(hits)} candidate chunk(s))")

    print(f"\nverified {ok} ok, {fail} failed, of {len(data)}")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "evaluation/questions_v2.json"))
