"""Unit tests for Vertex regional embedding rotation (no network calls)."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import embedding_client


class _Response:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body

    def json(self):
        return self._body


def _success(*vectors):
    return _Response(
        200,
        {"predictions": [{"embeddings": {"values": v}} for v in vectors]},
    )


class RegionalEmbeddingClientTests(unittest.TestCase):
    def _client(self):
        return embedding_client.RotatingEmbeddingClient(
            Mock(),
            project_id="project-1",
            api_key="secret",
            regions=("region-a", "region-b", "region-c"),
            enabled=True,
            timeout_s=7,
        )

    def test_round_robin_advances_the_starting_region(self):
        session = Mock()
        session.post.side_effect = [_success([0.1]), _success([0.2])]
        client = self._client()

        with (
            patch.object(embedding_client, "LLM_BACKEND", "vertex"),
            patch("embedding_client._http_session", return_value=session),
        ):
            client.models.embed_content(
                model="gemini-embedding-001", contents="first"
            )
            client.models.embed_content(
                model="gemini-embedding-001", contents="second"
            )

        self.assertIn("region-a-aiplatform", session.post.call_args_list[0].args[0])
        self.assertIn("region-b-aiplatform", session.post.call_args_list[1].args[0])

    def test_429_fails_over_and_preserves_batch_order(self):
        session = Mock()
        session.post.side_effect = [
            _Response(429, {"error": {"message": "quota"}}),
            _success([1.0, 1.1], [2.0, 2.1]),
        ]
        client = self._client()

        with (
            patch.object(embedding_client, "LLM_BACKEND", "vertex"),
            patch("embedding_client._http_session", return_value=session),
        ):
            result = client.models.embed_content(
                model="gemini-embedding-001", contents=["one", "two"]
            )

        self.assertEqual([e.values for e in result.embeddings], [[1.0, 1.1], [2.0, 2.1]])
        self.assertEqual(session.post.call_count, 2)
        self.assertIn("region-b-aiplatform", session.post.call_args.args[0])

    def test_auth_error_does_not_cycle_through_every_region(self):
        session = Mock()
        session.post.return_value = _Response(
            401, {"error": {"message": "bad credentials"}}
        )
        client = self._client()

        with (
            patch.object(embedding_client, "LLM_BACKEND", "vertex"),
            patch("embedding_client._http_session", return_value=session),
        ):
            with self.assertRaises(embedding_client.RegionalEmbeddingError) as raised:
                client.models.embed_content(
                    model="gemini-embedding-001", contents="one"
                )

        self.assertEqual(raised.exception.status_code, 401)
        self.assertEqual(session.post.call_count, 1)

    def test_developer_backend_never_uses_vertex_rotation(self):
        sdk = Mock()
        session = Mock()
        expected = SimpleNamespace(embeddings=[])
        sdk.models.embed_content.return_value = expected
        client = embedding_client.RotatingEmbeddingClient(
            sdk,
            project_id="project-1",
            api_key="secret",
            regions=("region-a",),
            enabled=True,
            timeout_s=7,
        )

        with (
            patch.object(embedding_client, "LLM_BACKEND", "developer"),
            patch("embedding_client._http_session", return_value=session),
        ):
            result = client.models.embed_content(
                model="gemini-embedding-001", contents=["one"]
            )

        self.assertIs(result, expected)
        sdk.models.embed_content.assert_called_once_with(
            model="gemini-embedding-001", contents=["one"]
        )
        session.post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
