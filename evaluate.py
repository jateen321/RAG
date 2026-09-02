"""Evaluate bilingual retrieval quality and optional answer citations."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from retriever import retrieve

DEFAULT_DATASET = Path(__file__).parent / "evaluation" / "questions_v2.json"
DEFAULT_OUTPUT = Path(__file__).parent / "evaluation" / "results.json"

# Heuristic, and deliberately narrow. These phrases signal an explicit refusal;
# a generic negation like "नहीं है" is excluded because it appears in ordinary
# answers and would score fabrication as refusal.
_REFUSAL_PATTERNS = [
    r"do(?:es)?\s+not\s+contain",
    r"n(?:o|ot enough)\s+(?:relevant\s+)?information",
    r"can(?:not|'t)\s+(?:be\s+)?answer",
    r"not\s+(?:available|found|mentioned|present)\s+in\s+the\s+(?:provided\s+)?(?:context|sources?|documents?)",
    r"context\s+does\s+not",
    r"पर्याप्त\s+जानकारी\s+नहीं",
    r"जानकारी\s+उपलब्ध\s+नहीं",
    r"संदर्भ\s+में\s+.{0,20}नहीं",
    r"उत्तर\s+देने\s+में\s+असमर्थ",
    r"दिए\s+गए\s+(?:संदर्भ|दस्तावेज़ों)\s+में\s+.{0,25}नहीं",
]


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")
    return dataset


def _expected_sources(question: dict) -> list[str]:
    """Accept both schemas.

    v2 uses ``expected_sources`` (a list) because three documents are indexed
    twice, under both ``X.pdf`` and ``data/X.pdf`` — a single exact-match string
    scores the duplicate copy as a miss. v1's singular key is still read so the
    old dataset keeps working.
    """
    if "expected_sources" in question:
        return question["expected_sources"]
    single = question.get("expected_source")
    return [single] if single else []


def _match_mode(question: dict) -> str:
    """page | source | none.

    Page numbers only locate content in PDFs (889 distinct values). Every
    plain-text chunk carries page_number=1 and every YouTube chunk carries
    None, so requiring a page match scores text trivially and YouTube never.
    Questions over those sources declare match="source".
    """
    if "match" in question:
        return question["match"]
    return "none" if question.get("category") == "unanswerable" else "page"


def _rank_expected_source(question: dict, chunks: list[dict]) -> int | None:
    """1-based rank of the first chunk that satisfies the question's ground truth."""
    sources = _expected_sources(question)
    mode = _match_mode(question)
    if mode == "none" or not sources:
        return None

    expected_pages = set(question.get("expected_pages") or [])
    for rank, chunk in enumerate(chunks, 1):
        if chunk["source"] not in sources:
            continue
        if mode == "page" and chunk["page"] not in expected_pages:
            continue
        return rank
    return None


def _has_expected_citation(answer: str, pages: list[int]) -> bool:
    for page in pages:
        pattern = rf"(?:page|पृष्ठ)\s*[:#-]?\s*{page}\b"
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def _is_refusal(answer: str) -> bool:
    """Did the model decline rather than fabricate?

    SYSTEM_PROMPT rule 2 asks for an honest "I don't know" when the context is
    insufficient. This is the only metric that tests it, and it is a heuristic:
    it can miss a refusal phrased unusually. Treat it as a floor on refusal
    rate, not an exact measure.
    """
    return any(re.search(p, answer, flags=re.IGNORECASE) for p in _REFUSAL_PATTERNS)


def _retrieve_result(question: str, top_k: int, retrieval_mode: str) -> dict:
    """Retrieve once through the selected strategy and retain route metadata."""
    if retrieval_mode == "direct":
        chunks = retrieve(question, top_k=top_k)
        return {
            "chunks": chunks,
            "queries": [question],
            "raw_candidate_count": len(chunks),
            "unique_candidate_count": len(chunks),
            "distinct_candidate_count": len(chunks),
            "rerank_candidate_count": 0,
            "context_character_count": sum(len(c.get("text", "")) for c in chunks),
            "route": "direct",
            "timings": {"query_planning_s": 0.0, "vector_retrieval_s": 0.0,
                        "reranking_s": 0.0, "retrieval_s": 0.0},
        }
    if retrieval_mode == "pipeline":
        from retrieval_pipeline import retrieve_context

        return retrieve_context(question)
    if retrieval_mode == "adaptive":
        from retrieval_pipeline import retrieve_adaptive_context

        return retrieve_adaptive_context(question, top_k=top_k)
    raise ValueError(f"Unknown retrieval mode: {retrieval_mode}")


def _retrieve_chunks(question: str, top_k: int, retrieval_mode: str) -> list[dict]:
    """Compatibility helper returning only selected chunks."""
    return _retrieve_result(question, top_k, retrieval_mode)["chunks"]


def evaluate(
    dataset: list[dict],
    top_k: int,
    generate: bool = False,
    retrieval_mode: str = "direct",
) -> dict:
    rows = []
    total_latency = 0.0

    for item in dataset:
        sources = _expected_sources(item)
        unanswerable = item.get("category") == "unanswerable"

        started_at = time.perf_counter()
        retrieval_result = _retrieve_result(item["question"], top_k, retrieval_mode)
        chunks = retrieval_result["chunks"]
        latency = time.perf_counter() - started_at
        total_latency += latency
        rank = _rank_expected_source(item, chunks)

        row = {
            "id": item["id"],
            "language": item["language"],
            "category": item.get("category", "factual"),
            "match": _match_mode(item),
            "difficulty": item.get("difficulty", "n/a"),
            "retrieval_hit": None if unanswerable else rank is not None,
            "first_relevant_rank": rank,
            # Contamination metric. hit_rate/MRR stop at the FIRST correct chunk,
            # so they cannot see wrong-book chunks filling the other k-1 slots.
            # This counts how much of the top-k came from an expected document.
            "source_precision": (
                None if unanswerable
                else round(sum(c["source"] in sources for c in chunks) / len(chunks), 4)
                if chunks else 0.0
            ),
            # Which books actually answered — the diagnostic, not just the score.
            "sources_returned": sorted({c["source"] for c in chunks}),
            "best_distance": round(min((c["distance"] for c in chunks), default=0.0), 4),
            "selected_chunk_count": len(chunks),
            "latency_s": round(latency, 3),
            "route": retrieval_result.get("route", retrieval_mode),
            "confidence": retrieval_result.get("confidence"),
        }

        if generate:
            from rag_engine import ask_with_sources

            # Pass top_k through. Without it this retrieves a SECOND time at the
            # config default, so citation scores would describe a different
            # retrieval than the ranks scored above — and it doubles the
            # embedding calls against the quota that binds a full run.
            generated = ask_with_sources(
                item["question"],
                **({"top_k": top_k} if retrieval_mode == "direct" else {}),
                retrieval_result=retrieval_result,
            )
            answer = generated["answer"]
            row["declined"] = _is_refusal(answer)
            row["answer_preview"] = answer[:160].replace("\n", " ")

            if unanswerable:
                # Correct behaviour here is a refusal, not an answer.
                row["correct"] = row["declined"]
            else:
                keywords = [w.casefold() for w in item.get("answer_keywords", [])]
                row["citation_correct"] = _has_expected_citation(
                    answer, item.get("expected_pages") or []
                )
                row["keyword_recall"] = (
                    sum(w in answer.casefold() for w in keywords) / len(keywords)
                    if keywords else None
                )
                row["correct"] = not row["declined"]

        rows.append(row)
        if unanswerable:
            status = "declined" if row.get("declined") else ("answered" if generate else "n/a")
            print(f"{item['id']}: [unanswerable] {status} "
                  f"(best_dist={row['best_distance']:.3f}, {latency:.3f}s)")
        else:
            print(f"{item['id']}: {'hit' if rank else 'miss'} (rank={rank}, "
                  f"src_prec={row['source_precision']:.2f}, {latency:.3f}s)")

    scored = [r for r in rows if r["category"] != "unanswerable"]
    unans = [r for r in rows if r["category"] == "unanswerable"]

    reciprocal = [
        1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0 for r in scored
    ]
    summary = {
        "questions": len(rows),
        "scored_questions": len(scored),
        "unanswerable_questions": len(unans),
        "retrieval_mode": retrieval_mode,
        "top_k": top_k if retrieval_mode == "direct" else None,
        "average_selected_chunks": round(
            sum(r["selected_chunk_count"] for r in rows) / len(rows), 3
        ),
        "retrieval_hit_rate": round(sum(r["retrieval_hit"] for r in scored) / len(scored), 4) if scored else None,
        "mean_reciprocal_rank": round(sum(reciprocal) / len(scored), 4) if scored else None,
        "mean_source_precision": round(
            sum(r["source_precision"] for r in scored) / len(scored), 4
        ) if scored else None,
        "average_retrieval_latency_s": round(total_latency / len(rows), 3),
    }

    routes = sorted({r["route"] for r in rows})
    for route in routes:
        summary[f"route_count_{route}"] = sum(r["route"] == route for r in rows)

    # Per-language breakdown: the whole point of a bilingual corpus is knowing
    # whether one language is being served worse than the other.
    for lang in sorted({r["language"] for r in scored}):
        sub = [r for r in scored if r["language"] == lang]
        summary[f"hit_rate_{lang}"] = round(sum(r["retrieval_hit"] for r in sub) / len(sub), 4)

    # Per-difficulty breakdown. The easy tier names its own document and often
    # repeats its answer keywords, so its hit rate is an upper bound. The hard
    # tier is the SAME ground truth with that leakage removed (see
    # evaluation/check_leakage.py), so the gap between the two is the cost of
    # phrasing alone, not a difference in what is being asked.
    for tier in sorted({r["difficulty"] for r in scored} - {"n/a"}):
        sub = [r for r in scored if r["difficulty"] == tier]
        rr = [1 / r["first_relevant_rank"] if r["first_relevant_rank"] else 0 for r in sub]
        summary[f"hit_rate_{tier}"] = round(sum(r["retrieval_hit"] for r in sub) / len(sub), 4)
        summary[f"mrr_{tier}"] = round(sum(rr) / len(sub), 4)
        summary[f"source_precision_{tier}"] = round(
            sum(r["source_precision"] for r in sub) / len(sub), 4
        )

    if generate:
        summary["citation_accuracy"] = round(
            sum(r.get("citation_correct", False) for r in scored) / len(scored), 4
        ) if scored else None
        kw = [r["keyword_recall"] for r in scored if r.get("keyword_recall") is not None]
        summary["average_keyword_recall"] = round(sum(kw) / len(kw), 4) if kw else None
        # The metric nothing measured before: does the model decline when it should?
        summary["refusal_rate_on_unanswerable"] = round(
            sum(r["declined"] for r in unans) / len(unans), 4
        ) if unans else None
        # And the inverse failure: refusing a question it could have answered.
        summary["false_refusal_rate"] = round(
            sum(r.get("declined", False) for r in scored) / len(scored), 4
        ) if scored else None

    return {"summary": summary, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--retrieval-mode",
        choices=("direct", "pipeline", "adaptive"),
        default="direct",
        help=(
            "Use direct vector retrieval for historical baselines or the adaptive "
            "planner/fusion/reranker pipeline used by the app."
        ),
    )
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers and score citations, keywords and refusals.",
    )
    args = parser.parse_args()

    report = evaluate(
        load_dataset(args.dataset),
        args.top_k,
        args.generate,
        retrieval_mode=args.retrieval_mode,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nSummary")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
