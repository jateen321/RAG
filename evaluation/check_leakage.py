# -*- coding: utf-8 -*-
"""Measure how much a question gives its own answer away.

A question that repeats the words of its answer, or names the document it is
in, tests lexical overlap rather than retrieval. This scores that leakage so
"the hard questions are harder" is an assertion about the data, not a claim
about the author's intentions.

Two leak channels:
  keyword  — a token shared between the question and its own answer_keywords
  source   — the question names the document it should be retrieved from
"""
import json
import re
import sys
from pathlib import Path

# Document-name stems, both scripts. A question containing one of these is
# telling the retriever which book to look in.
SOURCE_STEMS = [
    "अर्थशास्त्र", "arthasastra", "arthashastra", "कौटिल्य", "kautilya",
    "गीता", "gita", "भगवद्गीता", "bhagavad",
    "पंचतंत्र", "panchatantra", "pancatantra",
    "महाभारत", "mahabharata", "sabha parva", "सभापर्व",
    "रामायण", "ramayana", "sangam", "संगम",
]

_DEV = re.compile(r"[ऀ-ॿ]")
# Devanagari matras and the virama are combining marks (Unicode Mn/Mc), which
# `\w` excludes -- a naive [^\W\d_]+ shatters every Hindi word into single
# consonants that fall under the length floor, so the scan silently passes
# everything. Match a run of the Devanagari block as one token instead.
_TOKEN = re.compile(r"[\u0900-\u0965\u0970-\u097F]+|[^\W\d_]+", re.UNICODE)


def _tokens(text: str) -> set:
    """Content tokens only: >=4 chars for Devanagari, >=5 for Latin."""
    out = set()
    for t in _TOKEN.findall(text.lower()):
        floor = 4 if _DEV.search(t) else 5
        if len(t) >= floor:
            out.add(t)
    return out


def leaks(q: dict) -> dict:
    qt = _tokens(q["question"])
    kw = set()
    for k in q.get("answer_keywords") or []:
        kw |= _tokens(k)
    low = q["question"].lower()
    return {
        "keyword": sorted(qt & kw),
        "source": sorted(s for s in SOURCE_STEMS if s in low),
    }


def main(path: str) -> int:
    data = json.load(open(path, encoding="utf-8"))
    tiers = {}
    for q in data:
        d = q.get("difficulty")
        if d not in ("easy", "hard"):
            continue
        tiers.setdefault(d, []).append((q, leaks(q)))

    failures = 0
    for tier in ("easy", "hard"):
        rows = tiers.get(tier, [])
        if not rows:
            continue
        leaked = [(q, l) for q, l in rows if l["keyword"] or l["source"]]
        print(f"\n=== {tier.upper()} ({len(rows)} questions) — "
              f"{len(leaked)} leaking ===")
        for q, l in leaked:
            bits = []
            if l["keyword"]:
                bits.append("keyword=" + ",".join(l["keyword"]))
            if l["source"]:
                bits.append("source=" + ",".join(l["source"]))
            print(f"  {q['id']:22} {' | '.join(bits)}")
            if tier == "hard":
                failures += 1
        if not leaked:
            print("  (none)")

    print(f"\nhard-tier leaks: {failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1
                  else str(Path(__file__).parent / "questions_v2.json")))
