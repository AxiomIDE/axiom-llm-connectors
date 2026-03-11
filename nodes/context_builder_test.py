from gen.messages_pb2 import RetrievalContext, FormattedPrompt
from nodes.context_builder import context_builder


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_context_builder_formats_chunks():
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    ctx = RetrievalContext(question="What is Axiom?", chunks=["Axiom is a platform.", "It runs nodes."])
    result = context_builder(log, secrets, ctx)
    assert isinstance(result, FormattedPrompt)
    assert "What is Axiom?" in result.user
    assert "[1]" in result.user
    assert "[2]" in result.user
    assert len(result.system) > 0


def test_context_builder_no_chunks():
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    ctx = RetrievalContext(question="What is Axiom?", chunks=[])
    result = context_builder(log, secrets, ctx)
    assert "What is Axiom?" in result.user
    assert "No context" in result.user
