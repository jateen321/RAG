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
        ) as generate:
            self.assertEqual(retrieval_pipeline.generate_queries("original", []), ["original"])
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-3.5-flash-lite")


class RankFusionTests(unittest.TestCase):
    def test_reranker_schema_uses_requested_selection_bounds(self):
        schema = retrieval_pipeline._rerank_schema(2, 2)
        selected_ids = schema["properties"]["selected_ids"]
        self.assertEqual(selected_ids["minItems"], 2)
        self.assertEqual(selected_ids["maxItems"], 2)

    def test_rrf_deduplicates_and_rewards_cross_query_hits(self):
        fused = retrieval_pipeline.reciprocal_rank_fusion([
            [chunk("a1"), chunk("b2")],
            [chunk("b2"), chunk("c3")],
            [chunk("b2"), chunk("b2")],
        ])
        self.assertEqual([item["chunk_id"] for item in fused], ["b2", "a1", "c3"])
        self.assertEqual(fused[0]["query_hits"], 3)

    def test_reranker_uses_gemini_35_flash_lite_and_discards_unselected_candidates(self):
        candidates = [chunk(f"a{i}") for i in range(6)]
        response = SimpleNamespace(
            text=(
                '{"selected_ids":["candidate_5","candidate_3",'
                '"candidate_2","candidate_1","candidate_0"]}'
            )
        )
        with patch.object(
            retrieval_pipeline._client.models,
            "generate_content",
            return_value=response,
        ) as generate:
            ranked = retrieval_pipeline.rerank_candidates(
                "question", candidates, minimum_chunks=5, maximum_chunks=6
            )
        self.assertEqual(generate.call_args.kwargs["model"], "gemini-3.5-flash-lite")
        self.assertEqual(
            [item["chunk_id"] for item in ranked],
            ["a5", "a3", "a2", "a1", "a0"],
        )

    def test_reranker_falls_back_to_minimum_rrf_candidates_for_short_output(self):
        candidates = [chunk(f"a{i}") for i in range(8)]
        response = SimpleNamespace(text='{"selected_ids":["candidate_7"]}')
        with patch.object(
            retrieval_pipeline._client.models,
            "generate_content",
            return_value=response,
        ):
            ranked = retrieval_pipeline.rerank_candidates(
                "question", candidates, minimum_chunks=5, maximum_chunks=8
            )
        self.assertEqual(
            [item["chunk_id"] for item in ranked],
            ["a0", "a1", "a2", "a3", "a4"],
        )

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

class RetrievalPipelineTests(unittest.TestCase):
    def test_direct_confidence_combines_distance_margin_and_source_coherence(self):
        strong = [chunk("a1", 0.10), chunk("a2", 0.28), chunk("a3", 0.31)]
        weak = [chunk("a1", 0.44), {**chunk("b2", 0.45), "source": "other.pdf"}]

        strong_score = retrieval_pipeline.direct_confidence(strong)
        weak_score = retrieval_pipeline.direct_confidence(weak)

        self.assertTrue(strong_score["confident"])
        self.assertGreater(strong_score["score"], weak_score["score"])
        self.assertFalse(weak_score["confident"])

    def test_adaptive_router_returns_direct_results_when_confident(self):
        direct = [chunk("a1", 0.10), chunk("a2", 0.25)]
        with patch.object(retrieval_pipeline, "retrieve", return_value=direct) as retrieve:
            result = retrieval_pipeline.retrieve_adaptive_context("question", top_k=2)

        retrieve.assert_called_once_with("question", top_k=2, owner_id=None)
        self.assertEqual(result["route"], "direct")
        self.assertEqual(result["chunks"], direct)
        self.assertTrue(result["confidence"]["confident"])

    def test_adaptive_router_expands_uncertain_results(self):
        direct = [chunk("a1", 0.44), {**chunk("b2", 0.45), "source": "other.pdf"}]
        expanded = {"chunks": [chunk("a1", 0.2)], "timings": {"retrieval_s": 0.2}}
        with (
            patch.object(retrieval_pipeline, "retrieve", return_value=direct),
            patch.object(retrieval_pipeline, "retrieve_context", return_value=expanded) as pipeline,
        ):
            result = retrieval_pipeline.retrieve_adaptive_context("question", top_k=2)

        pipeline.assert_called_once_with("question", top_k=2, owner_id=None)
        self.assertEqual(result["route"], "expanded")
        self.assertIs(result["chunks"], expanded["chunks"])
        self.assertFalse(result["confidence"]["confident"])

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
                side_effect=lambda question, candidates, **kwargs: candidates[
                    :kwargs["maximum_chunks"]
                ],
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
                side_effect=lambda question, values, **kwargs: values[
                    :kwargs["maximum_chunks"]
                ],
            ),
        ):
            result = retrieval_pipeline.retrieve_context("original")

        self.assertTrue(result["adaptive"])
        self.assertGreater(len(result["chunks"]), 5)
        self.assertLessEqual(len(result["chunks"]), retrieval_pipeline.MAX_CONTEXT_CHUNKS)

    def test_default_pipeline_sends_fifteen_candidates_and_uses_variable_selection(self):
        candidates = [chunk(f"candidate-{i}") for i in range(20)]

        def select_seven(question, values, **kwargs):
            self.assertEqual(len(values), 15)
            self.assertEqual(kwargs["minimum_chunks"], 5)
            self.assertEqual(kwargs["maximum_chunks"], 15)
            return values[:7]

        with (
            patch.object(retrieval_pipeline, "_document_catalog", return_value=[]),
            patch.object(retrieval_pipeline, "generate_queries", return_value=["original"]),
            patch.object(retrieval_pipeline, "retrieve_many", return_value=[candidates]),
            patch.object(
                retrieval_pipeline,
                "rerank_candidates",
                side_effect=select_seven,
            ),
        ):
            result = retrieval_pipeline.retrieve_context("original")

        self.assertEqual(result["rerank_candidate_count"], 15)
        self.assertEqual(len(result["chunks"]), 7)


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
