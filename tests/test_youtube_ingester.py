"""Unit tests for YouTube parsing, transcript choice, and timestamp chunks."""

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

os.environ.setdefault("GEMINI_API_KEY", "test-key")

from youtube_ingester import (  # noqa: E402
    CHANNEL_VIDEO_LIMIT,
    TranscriptChunkConfig,
    VideoResult,
    _index_video,
    _select_transcript,
    _transcript_chunks,
    _transcript_quality,
    ingest_youtube,
    validate_youtube_url,
)
from indexer import get_stats, index_chunks, is_keyed_document_indexed  # noqa: E402


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

    def test_channel_urls_normalize_to_videos_tab(self):
        cases = {
            "https://www.youtube.com/@channel": "https://www.youtube.com/@channel/videos",
            "https://youtube.com/@channel/featured": "https://www.youtube.com/@channel/videos",
            "https://m.youtube.com/channel/UC123/videos": "https://www.youtube.com/channel/UC123/videos",
            "https://www.youtube.com/c/Name": "https://www.youtube.com/c/Name/videos",
            "https://www.youtube.com/user/name/shorts": "https://www.youtube.com/user/name/videos",
        }
        for url, expected in cases.items():
            with self.subTest(url=url):
                self.assertEqual(validate_youtube_url(url), expected)

    def test_rejects_non_youtube_and_incomplete_channel_urls(self):
        for url in (
            "https://example.com/watch?v=x",
            "https://example.com/@channel",
            "https://youtube.com/@",
            "https://youtube.com/channel",
        ):
            with self.subTest(url=url):
                with self.assertRaises(ValueError):
                    validate_youtube_url(url)


class ChannelIngestTests(unittest.TestCase):
    def test_channel_import_requests_only_latest_uploads(self):
        captured = {}

        class FakeYoutubeDL:
            def __init__(self, options):
                captured["options"] = options

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def extract_info(self, url, download=False):
                captured["url"] = url
                return {
                    "_type": "playlist",
                    "id": "UC123",
                    "title": "Channel - Videos",
                    "entries": [{"id": "v1", "title": "One"}],
                }

            def sanitize_info(self, info):
                return info

        with patch("yt_dlp.YoutubeDL", FakeYoutubeDL), patch(
            "youtube_ingester._index_video",
            return_value=VideoResult("v1", "One", "indexed", 3),
        ):
            report = ingest_youtube("https://www.youtube.com/@channel")

        self.assertEqual(captured["url"], "https://www.youtube.com/@channel/videos")
        self.assertEqual(captured["options"]["playlist_items"], f"1:{CHANNEL_VIDEO_LIMIT}")
        self.assertEqual(report["source_type"], "channel")
        self.assertEqual(report["chunks_indexed"], 3)


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
        matched = [
            (row_id, metadata) for row_id, (_, _, metadata) in self.rows.items()
            if not where or all(metadata.get(key) == value for key, value in where.items())
        ]
        result = {"ids": [row_id for row_id, _ in matched]}
        # Chroma returns the requested columns. The fake must too: index_chunks
        # reads metadatas to decide what is already embedded, and a fake that
        # omitted them made every resume path fall into an except branch, so
        # the tests passed while exercising nothing.
        if include and "metadatas" in include:
            result["metadatas"] = [metadata for _, metadata in matched]
        return result

    def delete(self, ids=None, where=None):
        for row_id in ids or []:
            self.rows.pop(row_id, None)
            self.deleted.append(row_id)

    def update(self, ids, metadatas):
        for row_id, metadata in zip(ids, metadatas):
            embedding, document, previous = self.rows[row_id]
            self.rows[row_id] = (embedding, document, {**previous, **metadata})


class SkipIndexedVideoTests(unittest.TestCase):
    def test_keyed_document_counts_as_indexed_only_when_complete(self):
        collection = FakeCollection()
        chunks = [
            {"text": "A" * 60, "chunk_index": 0},
            {"text": "B" * 60, "chunk_index": 1},
        ]
        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_batch", side_effect=lambda batch: [[0.1]] * len(batch)),
        ):
            self.assertFalse(is_keyed_document_indexed("vid1", "youtube", "owner"))
            index_chunks(
                chunks, "YouTube: One [vid1]", "youtube",
                document_key="vid1", owner_id="owner",
            )
            self.assertTrue(is_keyed_document_indexed("vid1", "youtube", "owner"))
            self.assertFalse(is_keyed_document_indexed("vid1", "youtube", "someone-else"))
            # A half-finished earlier run must not count, so index_chunks can resume it.
            collection.rows.pop(next(iter(collection.rows)))
            self.assertFalse(is_keyed_document_indexed("vid1", "youtube", "owner"))

    def test_already_indexed_video_skips_transcript_fetch(self):
        with (
            patch("youtube_ingester.is_keyed_document_indexed", return_value=True) as check,
            patch("youtube_transcript_api.YouTubeTranscriptApi") as transcript_api,
        ):
            result = _index_video({"id": "vid1", "title": "One"}, owner_id="owner")

        check.assert_called_once_with("vid1", "youtube", "owner")
        transcript_api.assert_not_called()
        self.assertEqual(result.status, "already_indexed")

    def test_reimport_with_nothing_new_reports_instead_of_failing(self):
        ydl = MagicMock()
        ydl.__enter__.return_value = ydl
        ydl.extract_info.return_value = {"id": "vid1", "title": "One"}
        ydl.sanitize_info.side_effect = lambda info: info
        with (
            patch("yt_dlp.YoutubeDL", return_value=ydl),
            patch(
                "youtube_ingester._index_video",
                return_value=VideoResult("vid1", "One", "already_indexed"),
            ),
        ):
            report = ingest_youtube("https://www.youtube.com/watch?v=vid1")

        self.assertEqual(report["videos_indexed"], 0)
        self.assertEqual(report["videos_already_indexed"], 1)


class SharedIndexerTests(unittest.TestCase):
    def test_stats_keep_youtube_source_url_for_library_links(self):
        collection = FakeCollection()
        chunks = [{"text": "A" * 60, "chunk_index": 0}]
        video_url = "https://www.youtube.com/watch?v=video123"

        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_batch", return_value=[[0.1]]),
        ):
            index_chunks(
                chunks,
                "Example video",
                "youtube",
                source_metadata={"source_url": video_url},
            )
            stats = get_stats()

        self.assertEqual(stats["documents"][0]["source_url"], video_url)

    def test_pdf_ids_remain_unique_when_chunk_indices_restart_each_page(self):
        collection = FakeCollection()
        chunks = [
            {"text": "A" * 60, "page_number": 1, "chunk_index": 0},
            {"text": "B" * 60, "page_number": 2, "chunk_index": 0},
        ]
        with (
            patch("indexer._get_collection", return_value=collection),
            # index_chunks embeds and stores one batch at a time so a quota
            # failure costs one batch instead of the document, so the seam to
            # stub is _embed_batch. Patching _embed_texts here silently stopped
            # intercepting anything and the test began making real API calls.
            patch("indexer._embed_batch", side_effect=lambda batch: [[0.1]] * len(batch)),
        ):
            self.assertEqual(index_chunks(chunks, "book.pdf", "pdf"), 2)

        self.assertEqual(len(collection.rows), 2)
        self.assertTrue(any("_p0001_c000" in row_id for row_id in collection.rows))
        self.assertTrue(any("_p0002_c000" in row_id for row_id in collection.rows))

    @staticmethod
    def _chunks():
        from indexer import _content_hash
        return [
            {"text": "A" * 60, "page_number": 1, "chunk_index": 0,
             "content_hash": _content_hash("A" * 60)},
            {"text": "B" * 60, "page_number": 2, "chunk_index": 0,
             "content_hash": _content_hash("B" * 60)},
        ]

    def test_second_run_reembeds_nothing(self):
        """A completed document costs zero embedding calls to re-index."""
        collection = FakeCollection()
        calls = []

        def record(batch):
            calls.append(len(batch))
            return [[0.1]] * len(batch)

        for _ in range(2):
            with (
                patch("indexer._get_collection", return_value=collection),
                patch("indexer._embed_batch", side_effect=record),
            ):
                index_chunks(self._chunks(), "book.pdf", "pdf")

        self.assertEqual(sum(calls), 2, "second run must embed nothing")
        self.assertEqual(len(collection.rows), 2)

    def test_resume_after_partial_failure_embeds_only_the_remainder(self):
        """The whole point: a quota failure costs one batch, not the document."""
        collection = FakeCollection()

        # First run dies after the first chunk is stored.
        def one_then_die(batch):
            if collection.rows:
                raise RuntimeError("quota exhausted")
            return [[0.1]] * len(batch)

        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_batch", side_effect=one_then_die),
            patch("indexer.EMBED_BATCH_SIZE", 1),
        ):
            with self.assertRaises(RuntimeError):
                index_chunks(self._chunks(), "book.pdf", "pdf")

        self.assertEqual(len(collection.rows), 1, "first batch must survive")

        embedded = []
        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_batch",
                  side_effect=lambda b: embedded.extend(b) or [[0.2]] * len(b)),
            patch("indexer.EMBED_BATCH_SIZE", 1),
        ):
            index_chunks(self._chunks(), "book.pdf", "pdf")

        self.assertEqual(len(embedded), 1, "resume must re-embed only the remainder")
        self.assertEqual(len(collection.rows), 2)

    def test_chunk_without_content_hash_is_never_skipped(self):
        """Regression: `already.get(id) != chunk.get(hash)` made None == None,
        marking an unstored chunk as already embedded and dropping it."""
        collection = FakeCollection()
        chunks = [{"text": "A" * 60, "page_number": 1, "chunk_index": 0}]
        with (
            patch("indexer._get_collection", return_value=collection),
            patch("indexer._embed_batch", side_effect=lambda b: [[0.1]] * len(b)),
        ):
            index_chunks(chunks, "book.pdf", "pdf")
        self.assertEqual(len(collection.rows), 1, "hashless chunk must still be stored")


if __name__ == "__main__":
    unittest.main()
