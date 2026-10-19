"""Unit tests for the Gemini Developer API embedding client."""

import unittest
from unittest.mock import Mock, patch

import embedding_client


class GeminiDeveloperEmbeddingClientTests(unittest.TestCase):
    def test_returns_shared_gemini_client(self):
        client = Mock()
        with patch("embedding_client.get_client", return_value=client) as get_client:
            result = embedding_client.get_embedding_client()

        self.assertIs(result, client)
        get_client.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
