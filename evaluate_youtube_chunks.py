"""Evaluate YouTube transcript chunk sizes in isolated in-memory collections."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from uuid import uuid4

import chromadb
from google.genai import types
from pydantic import BaseModel, Field

from config import EMBEDDING_MODEL, LLM_MODEL
from indexer import _embed_texts
from llm_client import get_client
from youtube_ingester import (
    TranscriptChunkConfig,
    _clean_snippet,
    _select_transcript,
    _transcript_chunks,
    _transcript_quality,
)


DEFAULT_VIDEO_ID = "vm-hu1Iew-M"
CHAR_TARGETS = (400, 800, 1200, 1600)


class EvalQuestion(BaseModel):
    question: str = Field(min_length=5)
    evidence_seconds: float = Field(ge=0)


class EvalQuestionSet(BaseModel):
    questions: list[EvalQuestion]


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _candidate_config(target_chars: int) -> TranscriptChunkConfig:
    return TranscriptChunkConfig(
        target_chars=target_chars,
        max_chars=max(target_chars + 200, round(target_chars * 1.25)),
        target_seconds=75,
        max_seconds=120,
        overlap_seconds=12,
    )


def _structural_metrics(chunks: list[dict], transcript_chars: int) -> dict:
    lengths = [len(chunk["text"]) for chunk in chunks]
    durations = [chunk["end_seconds"] - chunk["start_seconds"] for chunk in chunks]
    stored_chars = sum(lengths)
    return {
        "chunk_count": len(chunks),
        "average_chars": round(statistics.mean(lengths), 2),
        "p95_chars": round(_percentile(lengths, 0.95), 2),
        "average_duration_seconds": round(statistics.mean(durations), 2),
        "p95_duration_seconds": round(_percentile(durations, 0.95), 2),
        "redundancy_ratio": round(
            max(stored_chars - transcript_chars, 0) / transcript_chars, 4
        ),
    }


def _evidence_excerpts(snippets, count: int = 12) -> list[dict]:
    """Sample short local contexts uniformly across the complete timeline."""
    usable = [snippet for snippet in snippets if _clean_snippet(snippet.text)]
    if len(usable) < count:
        raise RuntimeError("Transcript is too short for the requested evaluation set.")
    positions = [round(i * (len(usable) - 1) / (count - 1)) for i in range(count)]
    excerpts = []
    for position in positions:
        start = max(0, position - 3)
        end = min(len(usable), position + 4)
        excerpts.append({
            "evidence_seconds": round(float(usable[position].start), 3),
            "context": " ".join(_clean_snippet(item.text) for item in usable[start:end]),
        })
    return excerpts


def _generate_questions(excerpts: list[dict]) -> list[EvalQuestion]:
    """Create semantic questions while retaining supplied evidence timestamps."""
    client = get_client()
    prompt = """Create exactly one natural study question for each numbered transcript
excerpt below. Each question must be answerable from that excerpt, must not mention
the timestamp or excerpt, and should paraphrase rather than copy its wording. Keep
the supplied evidence_seconds unchanged. Use the same language as each excerpt.

Return the questions in the supplied JSON schema.

""" + "\n\n".join(
        f"Excerpt {index}\nevidence_seconds: {item['evidence_seconds']}\n"
        f"text: {item['context']}"
        for index, item in enumerate(excerpts, 1)
    )
    response = client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            response_schema=EvalQuestionSet,
            temperature=0.2,
        ),
    )
    parsed = EvalQuestionSet.model_validate_json(response.text)
    if len(parsed.questions) != len(excerpts):
        raise RuntimeError(
            f"Expected {len(excerpts)} evaluation questions, got "
            f"{len(parsed.questions)}."
        )
    expected_times = [item["evidence_seconds"] for item in excerpts]
    for question, expected in zip(parsed.questions, expected_times):
        # Ground truth comes from deterministic sampling, not from an LLM's
        # potentially altered numeric output.
        question.evidence_seconds = expected
    return parsed.questions


def _distance_to_interval(point: float, start: float, end: float) -> float:
    if start <= point <= end:
        return 0.0
    return min(abs(point - start), abs(point - end))


def _retrieval_metrics(chunks: list[dict], questions: list[EvalQuestion]) -> dict:
    client = chromadb.EphemeralClient()
    collection = client.create_collection(
        f"candidate-{uuid4().hex}", metadata={"hnsw:space": "cosine"}
    )
    chunk_embeddings = _embed_texts([chunk["text"] for chunk in chunks])
    collection.add(
        ids=[f"chunk-{index}" for index in range(len(chunks))],
        embeddings=chunk_embeddings,
        documents=[chunk["text"] for chunk in chunks],
        metadatas=[{
            "start_seconds": chunk["start_seconds"],
            "end_seconds": chunk["end_seconds"],
        } for chunk in chunks],
    )
    query_embeddings = _embed_texts([question.question for question in questions])

    reciprocal_ranks = []
    hits = {1: 0, 3: 0, 5: 0}
    top_errors = []
    for question, embedding in zip(questions, query_embeddings):
        result = collection.query(
            query_embeddings=[embedding],
            n_results=min(5, len(chunks)),
            include=["metadatas", "distances"],
        )
        ranked = result["metadatas"][0]
        matching_rank = None
        for rank, metadata in enumerate(ranked, 1):
            if (
                metadata["start_seconds"]
                <= question.evidence_seconds
                <= metadata["end_seconds"]
            ):
                matching_rank = rank
                break
        reciprocal_ranks.append(1 / matching_rank if matching_rank else 0.0)
        for cutoff in hits:
            if matching_rank and matching_rank <= cutoff:
                hits[cutoff] += 1
        top = ranked[0]
        top_errors.append(_distance_to_interval(
            question.evidence_seconds,
            top["start_seconds"],
            top["end_seconds"],
        ))

    total = len(questions)
    return {
        "questions": total,
        "recall_at_1": round(hits[1] / total, 4),
        "recall_at_3": round(hits[3] / total, 4),
        "recall_at_5": round(hits[5] / total, 4),
        "mean_reciprocal_rank": round(statistics.mean(reciprocal_ranks), 4),
        "mean_top1_timestamp_error_seconds": round(statistics.mean(top_errors), 2),
        "median_top1_timestamp_error_seconds": round(statistics.median(top_errors), 2),
    }


def evaluate(video_id: str, structural_only: bool = False) -> dict:
    from youtube_transcript_api import YouTubeTranscriptApi

    transcript = _select_transcript(YouTubeTranscriptApi().list(video_id))
    fetched = transcript.fetch()
    quality = _transcript_quality(fetched)
    transcript_chars = quality["transcript_character_count"]
    candidates = []

    questions = []
    if not structural_only:
        questions = _generate_questions(_evidence_excerpts(fetched))

    for target in CHAR_TARGETS:
        config = _candidate_config(target)
        chunks = _transcript_chunks(fetched, config, video_id=video_id)
        result = {
            "target_chars": target,
            "max_chars": config.max_chars,
            "target_seconds": config.target_seconds,
            "max_seconds": config.max_seconds,
            "overlap_seconds": config.overlap_seconds,
            **_structural_metrics(chunks, transcript_chars),
        }
        if questions:
            result.update(_retrieval_metrics(chunks, questions))
        candidates.append(result)

    return {
        "video_id": video_id,
        "embedding_model": EMBEDDING_MODEL,
        "transcript_language": transcript.language,
        "transcript_language_code": transcript.language_code,
        "transcript_is_generated": transcript.is_generated,
        "transcript_quality": quality,
        "evaluation_questions": [question.model_dump() for question in questions],
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-id", default=DEFAULT_VIDEO_ID)
    parser.add_argument("--structural-only", action="store_true")
    args = parser.parse_args()
    print(json.dumps(
        evaluate(args.video_id, structural_only=args.structural_only),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
