from gen.messages_pb2 import FormattedPrompt, TokenChunk
from nodes.anthropic_connector import anthropic_connector


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_anthropic_connector_missing_secret():
    """Without a secret, the node should emit a single error TokenChunk."""
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    prompt = FormattedPrompt(system="You are helpful.", user="Say hello.")
    frames = list(anthropic_connector(log, secrets, iter([prompt])))
    assert len(frames) == 1
    assert frames[0].is_final is True
    assert "ANTHROPIC_API_KEY" in frames[0].text
