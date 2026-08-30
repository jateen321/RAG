"""Offline tests for query planning, fusion, and Gemini reranking."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import retrieval_pipeline
import retriever


def chunk(chunk_id: str, distance: float = 0.2) -> dict:
    return {
        "chunk_id": chunk_id,
        "text": f"Text for {chunk_id}",
        "page": 1,
        "source": "book.pdf",
        "distance": distance,
        "document_id": "doc",
        "chunk_index": int(chunk_id[-1]) if chunk_id[-1].isdigit() else 0,
        "content_hash": chunk_id,
    }


class QueryPlanningTests(unittest.TestCase):
    def test_original_is_first_and_rewrites_are_unique(self):
        queries = retrieval_pipeline._unique_queries(
            "  What is Dharma? ",
            ["what is dharma?", "Dharma meaning", "Dharma   meaning", "धर्म क्या है?"],
            maximum=3,
        )
        self.assertEqual(queries, ["What is Dharma?", "Dharma meaning", "धर्म क्या है?"])

    def test_planner_failure_falls_back_to_original(self):
        with patch.object(
            retrieval_pipeline._client.models,
            "generate_content",
            return_value=SimpleNamespace(text="not json"),
        ):
            self.assertEqual(retrieval_pipeline.generate_queries("original", []), ["original"])


class RankFusionTests(unittest.TestCase):
    def test_rrf_deduplicates_and_rewards_cross_query_hits(self):
        fused = retrieval_pipeline.reciprocal_rank_fusion([
            [chunk("a1"), chunk("b2")],
            [chunk("b2"), chunk("c3")],
            [chunk("b2"), chunk("b2")],
        ])
        self.assertEqual([item["chunk_id"] for item in fused], ["b2", "a1", "c3"])
        self.assertEqual(fused[0]["query_hits"], 3)

    def test_reranker_uses_flash_lite_and_keeps_unmentioned_candidates(self):
        candidates = [chunk("a1"), chunk("b2"), chunk("c3")]
        response = SimpleNamespace(text='{"ranked_ids":["candidate_2"]}')
        with patch.object(
            retrieval_pipeline._client.models,
            "generate_content",
            return_value=response,
        ) as generate:
            ranked = retrieval_pipeline.rerank_candidates("question", candidates, 2)
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-2.5-flash-lite")
        self.assertEqual([item["chunk_id"] for item in ranked], ["c3", "a1"])

    def test_overlap_filter_removes_content_hash_and_near_copy_duplicates(self):
        original = chunk("a1")
        original["text"] = "one two three four five six seven eight nine ten"
        exact_hash = chunk("b2")
        exact_hash["content_hash"] = original["content_hash"]
        near_copy = chunk("c3")
        near_copy["text"] = "zero one two three four five six seven eight nine ten extra"
        distinct = chunk("d4")
        distinct["text"] = "a completely separate passage about another useful fact"

        filtered = retrieval_pipeline.deduplicate_overlapping(
            [original, exact_hash, near_copy, distinct]
        )
        self.assertEqual([item["chunk_id"] for item in filtered], ["a1", "d4"])

    def test_context_packer_respects_chunk_and_character_limits(self):
        candidates = [chunk(f"a{i}") for i in range(1, 5)]
        for candidate in candidates:
            candidate["text"] = "x" * 10
        packed = retrieval_pipeline.pack_context(
            candidates, maximum_chunks=3, character_budget=25
        )
        self.assertEqual([item["chunk_id"] for item in packed], ["a1", "a2"])


class RetrievalPipelineTests(unittest.TestCase):
    def test_pipeline_batches_queries_and_keeps_unique_candidates(self):
        result_lists = [
            [chunk("a1"), chunk("b2")],
            [chunk("b2"), chunk("c3")],
        ]
        with (
            patch.object(retrieval_pipeline, "_document_catalog", return_value=[]),
            patch.object(retrieval_pipeline, "generate_queries", return_value=["original", "rewrite"]),
            patch.object(retrieval_pipeline, "retrieve_many", return_value=result_lists) as retrieve_many,
            patch.object(
                retrieval_pipeline,
                "rerank_candidates",
                side_effect=lambda question, candidates, top_k: candidates[:top_k],
            ),
        ):
            result = retrieval_pipeline.retrieve_context("original", top_k=2)

        retrieve_many.assert_called_once_with(
            ["original", "rewrite"], top_k=retrieval_pipeline.QUERY_RETRIEVAL_TOP_K
        )
        self.assertEqual(result["raw_candidate_count"], 4)
        self.assertEqual(result["unique_candidate_count"], 3)
        self.assertEqual(len(result["chunks"]), 2)
        self.assertFalse(result["adaptive"])

    def test_default_pipeline_adaptively_packs_more_than_five_chunks(self):
        candidates = [chunk(f"a{i}") for i in range(10)]
        with (
            patch.object(retrieval_pipeline, "_document_catalog", return_value=[]),
            patch.object(retrieval_pipeline, "generate_queries", return_value=["original"]),
            patch.object(retrieval_pipeline, "retrieve_many", return_value=[candidates]),
            patch.object(
                retrieval_pipeline,
                "rerank_candidates",
                side_effect=lambda question, values, top_k: values[:top_k],
            ),
        ):
            result = retrieval_pipeline.retrieve_context("original")

        self.assertTrue(result["adaptive"])
        self.assertGreater(len(result["chunks"]), 5)
        self.assertLessEqual(len(result["chunks"]), retrieval_pipeline.MAX_CONTEXT_CHUNKS)


class BatchedRetrieverTests(unittest.TestCase):
    def test_retrieve_many_batches_embeddings_and_preserves_locators(self):
        embeddings = [SimpleNamespace(values=[0.1]), SimpleNamespace(values=[0.2])]
        collection = Mock()
        collection.count.return_value = 2
        collection.query.return_value = {
            "ids": [["a1"], ["b2"]],
            "documents": [["alpha"], ["beta"]],
            "metadatas": [
                [{"source_name": "a.pdf", "page_number": 1, "source_type": "pdf"}],
                [{"source_name": "video", "timestamp": "1:05", "source_type": "youtube"}],
            ],
            "distances": [[0.1], [0.2]],
        }
        persistent_client = Mock()
        persistent_client.get_collection.return_value = collection

        with (
            patch.object(
                retriever._client.models,
                "embed_content",
                return_value=SimpleNamespace(embeddings=embeddings),
            ) as embed,
            patch.object(retriever.chromadb, "PersistentClient", return_value=persistent_client),
        ):
            results = retriever.retrieve_many(["one", "two"], top_k=1)

        embed.assert_called_once_with(model=retriever.EMBEDDING_MODEL, contents=["one", "two"])
        self.assertEqual(results[0][0]["chunk_id"], "a1")
        self.assertEqual(results[1][0]["page"], "1:05")


if __name__ == "__main__":
    unittest.main()
