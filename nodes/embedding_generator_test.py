from gen.messages_pb2 import EmbeddingRequest, EmbeddingVector
from nodes.embedding_generator import embedding_generator


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_embedding_generator_missing_secret():
    """Without a secret, the node should return an empty vector."""
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    req = EmbeddingRequest(text="Hello world", model="text-embedding-3-small")
    result = embedding_generator(log, secrets, req)
    assert isinstance(result, EmbeddingVector)
    assert len(result.values) == 0
    assert result.input_text == "Hello world"
