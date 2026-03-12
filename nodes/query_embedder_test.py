import pytest
from unittest.mock import patch, MagicMock
from gen.messages_pb2 import EmbeddingRequest, EmbeddingVector
from nodes.query_embedder import query_embedder


class _NoOpLogger:
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


class _GoodSecrets:
    def get(self, name: str):
        return "sk-fake", True


def test_query_embedder_returns_empty_vector_on_missing_secret():
    result = query_embedder(_NoOpLogger(), _NoOpSecrets(), EmbeddingRequest(text="hello"))
    assert isinstance(result, EmbeddingVector)
    assert len(result.values) == 0


def test_query_embedder_with_mock_openai():
    fake_embedding = [0.1, 0.2, 0.3]

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("nodes.query_embedder.OpenAI", return_value=mock_client):
        result = query_embedder(_NoOpLogger(), _GoodSecrets(), EmbeddingRequest(text="What is attention?"))

    assert list(result.values) == pytest.approx(fake_embedding)
    assert result.input_text == "What is attention?"
    mock_client.embeddings.create.assert_called_once_with(
        input="What is attention?", model="text-embedding-3-small"
    )


def test_query_embedder_uses_explicit_model():
    fake_embedding = [0.5, 0.6]

    mock_response = MagicMock()
    mock_response.data = [MagicMock(embedding=fake_embedding)]

    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = mock_response

    with patch("nodes.query_embedder.OpenAI", return_value=mock_client):
        query_embedder(_NoOpLogger(), _GoodSecrets(),
                       EmbeddingRequest(text="hello", model="text-embedding-3-large"))

    mock_client.embeddings.create.assert_called_once_with(
        input="hello", model="text-embedding-3-large"
    )
