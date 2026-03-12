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
    """Without a secret, the node should yield no frames."""
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    req = EmbeddingRequest(text="Hello world", model="text-embedding-3-small")
    results = list(embedding_generator(log, secrets, iter([req])))
    assert len(results) == 0
