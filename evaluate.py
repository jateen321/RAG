"""Evaluate bilingual retrieval quality and optional answer citations."""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

from retriever import retrieve


DEFAULT_DATASET = Path(__file__).parent / "evaluation" / "questions.json"
DEFAULT_OUTPUT = Path(__file__).parent / "evaluation" / "results.json"


def load_dataset(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        dataset = json.load(handle)
    if not isinstance(dataset, list) or not dataset:
        raise ValueError("Evaluation dataset must be a non-empty JSON list.")
    return dataset


def _rank_expected_source(question: dict, chunks: list[dict]) -> int | None:
    expected_pages = set(question["expected_pages"])
    for rank, chunk in enumerate(chunks, 1):
        if (
            chunk["source"] == question["expected_source"]
            and chunk["page"] in expected_pages
        ):
            return rank
    return None


def _has_expected_citation(answer: str, pages: list[int]) -> bool:
    for page in pages:
        pattern = rf"(?:page|पृष्ठ)\s*[:#-]?\s*{page}\b"
        if re.search(pattern, answer, flags=re.IGNORECASE):
            return True
    return False


def evaluate(dataset: list[dict], top_k: int, generate: bool = False) -> dict:
    rows = []
    total_latency = 0.0

    for item in dataset:
        started_at = time.perf_counter()
        chunks = retrieve(item["question"], top_k=top_k)
        latency = time.perf_counter() - started_at
        total_latency += latency
        rank = _rank_expected_source(item, chunks)

        row = {
            "id": item["id"],
            "language": item["language"],
            "retrieval_hit": rank is not None,
            "first_relevant_rank": rank,
            "latency_s": round(latency, 3),
        }

        if generate:
            from rag_engine import ask_with_sources

            generated = ask_with_sources(item["question"])
            answer = generated["answer"]
            keywords = [word.casefold() for word in item.get("answer_keywords", [])]
            row["citation_correct"] = _has_expected_citation(
                answer, item["expected_pages"]
            )
            row["keyword_recall"] = (
                sum(word in answer.casefold() for word in keywords) / len(keywords)
                if keywords else None
            )

        rows.append(row)
        status = "hit" if rank is not None else "miss"
        print(f"{item['id']}: {status} (rank={rank}, {latency:.3f}s)")

    hit_count = sum(row["retrieval_hit"] for row in rows)
    reciprocal_ranks = [
        1 / row["first_relevant_rank"] if row["first_relevant_rank"] else 0
        for row in rows
    ]
    summary = {
        "questions": len(rows),
        "top_k": top_k,
        "retrieval_hit_rate": round(hit_count / len(rows), 4),
        "mean_reciprocal_rank": round(sum(reciprocal_ranks) / len(rows), 4),
        "average_retrieval_latency_s": round(total_latency / len(rows), 3),
    }

    if generate:
        summary["citation_accuracy"] = round(
            sum(row["citation_correct"] for row in rows) / len(rows), 4
        )
        keyword_scores = [
            row["keyword_recall"]
            for row in rows
            if row.get("keyword_recall") is not None
        ]
        summary["average_keyword_recall"] = round(
            sum(keyword_scores) / len(keyword_scores), 4
        ) if keyword_scores else None

    return {"summary": summary, "results": rows}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--generate",
        action="store_true",
        help="Also generate answers and score citations/keywords.",
    )
    args = parser.parse_args()

    report = evaluate(load_dataset(args.dataset), args.top_k, args.generate)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print("\nSummary")
    for key, value in report["summary"].items():
        print(f"  {key}: {value}")
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
