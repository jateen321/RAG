"""Unit tests for YouTube parsing, transcript choice, and timestamp chunks."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from youtube_ingester import (  # noqa: E402
    TranscriptChunkConfig,
    _select_transcript,
    _transcript_chunks,
    _transcript_quality,
    validate_youtube_url,
)
from indexer import index_chunks  # noqa: E402


def transcript(language_code: str, generated: bool):
    return SimpleNamespace(
        language_code=language_code,
        language=language_code,
        is_generated=generated,
    )


class YouTubeUrlTests(unittest.TestCase):
    def test_accepts_video_and_playlist_urls(self):
        urls = [
            "https://youtu.be/abc123",
            "https://www.youtube.com/watch?v=abc123",
            "https://youtube.com/playlist?list=PL123",
            "https://youtube.com/shorts/abc123",
        ]
        for url in urls:
            with self.subTest(url=url):
                self.assertEqual(validate_youtube_url(url), url)

    def test_rejects_non_youtube_and_channel_urls(self):
        for url in ("https://example.com/watch?v=x", "https://youtube.com/@channel"):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_youtube_url(url)


class TranscriptSelectionTests(unittest.TestCase):
    def test_manual_transcript_wins_before_generated_language_preference(self):
        chosen = _select_transcript([
            transcript("hi", True),
            transcript("fr", False),
            transcript("en", True),
        ])
        self.assertEqual(chosen.language_code, "fr")
        self.assertFalse(chosen.is_generated)

    def test_hindi_wins_within_same_transcript_kind(self):
        chosen = _select_transcript([
            transcript("en-US", False),
            transcript("hi-IN", False),
        ])
        self.assertEqual(chosen.language_code, "hi-IN")


class TranscriptChunkTests(unittest.TestCase):
    def test_chunks_keep_timestamp_boundaries(self):
        snippets = [
            SimpleNamespace(text="पहला वाक्य " * 20, start=0.0, duration=4.0),
            SimpleNamespace(text="दूसरा वाक्य " * 20, start=4.0, duration=5.0),
            SimpleNamespace(text="third sentence " * 30, start=9.0, duration=7.0),
        ]
        chunks = _transcript_chunks(
            snippets,
            TranscriptChunkConfig(
                target_chars=400,
                max_chars=600,
                target_seconds=75,
                max_seconds=120,
                overlap_seconds=12,
            ),
        )
        self.assertGreaterEqual(len(chunks), 2)
        self.assertEqual(chunks[0]["start_seconds"], 0.0)
        self.assertEqual(chunks[0]["timestamp"], "0:00")
        self.assertGreater(chunks[-1]["end_seconds"], chunks[-1]["start_seconds"])
        self.assertEqual(
            [chunk["chunk_index"] for chunk in chunks], list(range(len(chunks)))
        )

    def test_time_target_splits_sparse_transcript_and_builds_playable_urls(self):
        snippets = [
            SimpleNamespace(text=f"caption number {index}", start=index * 30.0, duration=4.0)
            for index in range(8)
        ]
        config = TranscriptChunkConfig(
            target_chars=10_000,
            max_chars=12_000,
            target_seconds=60,
            max_seconds=90,
            overlap_seconds=12,
        )
        chunks = _transcript_chunks(snippets, config, video_id="video123")

        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(
            chunk["end_seconds"] - chunk["start_seconds"] <= 90
            for chunk in chunks
        ))
        self.assertEqual(
            chunks[1]["timestamp_url"],
            f"https://www.youtube.com/watch?v=video123&t={int(chunks[1]['start_seconds'])}s",
        )

    def test_quality_metrics_are_scalar_and_measure_coverage(self):
        snippets = [
            SimpleNamespace(text="आयुर्वेद treatment", start=0.0, duration=10.0),
            SimpleNamespace(text="आयुर्वेद treatment", start=10.0, duration=10.0),
        ]
        quality = _transcript_quality(snippets, video_duration=25.0)

        self.assertEqual(quality["transcript_snippet_count"], 2)
        self.assertEqual(quality["transcript_coverage_ratio"], 0.8)
        self.assertEqual(quality["transcript_repeated_snippet_ratio"], 0.5)
        self.assertGreater(quality["transcript_devanagari_letter_ratio"], 0)
        self.assertGreater(quality["transcript_latin_letter_ratio"], 0)


class FakeCollection:
    def __init__(self):
        self.rows = {}
        self.deleted = []

    def count(self):
        return len(self.rows)

    def upsert(self, ids, embeddings, documents, metadatas):
        for row in zip(ids, embeddings, documents, metadatas):
            self.rows[row[0]] = row[1:]

    def get(self, where=None, include=None):
        ids = [
            row_id for row_id, (_, _, metadata) in self.rows.items()
            if not where or all(metadata.get(key) == value for key, value in where.items())
        ]
        return {"ids": ids}

    def delete(self, ids=None, where=None):
        for row_id in ids or []:
            self.rows.pop(row_id, None)
            self.deleted.append(row_id)


class SharedIndexerTests(unittest.TestCase):
    def test_pdf_ids_remain_unique_when_chunk_indices_restart_each_page(self):
        collection = FakeCollection()
        chunks = [
            {"text": "A" * 60, "page_number": 1, "chunk_index": 0},
            {"text": "B" * 60, "page_number": 2, "chunk_index": 0},
        ]
        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_texts", return_value=[[0.1], [0.2]]),
        ):
            self.assertEqual(index_chunks(chunks, "book.pdf", "pdf"), 2)

        self.assertEqual(len(collection.rows), 2)
        self.assertTrue(any("_p0001_c000" in row_id for row_id in collection.rows))
        self.assertTrue(any("_p0002_c000" in row_id for row_id in collection.rows))


if __name__ == "__main__":
    unittest.main()
