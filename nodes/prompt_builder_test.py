from gen.messages_pb2 import ChatMessage, FormattedPrompt
from nodes.prompt_builder import prompt_builder


class _NoOpLogger:
    """Minimal AxiomLogger implementation for unit tests."""
    def debug(self, msg: str, **attrs) -> None: pass
    def info(self, msg: str, **attrs) -> None: pass
    def warn(self, msg: str, **attrs) -> None: pass
    def error(self, msg: str, **attrs) -> None: pass


class _NoOpSecrets:
    def get(self, name: str):
        return "", False


def test_prompt_builder_uses_default_system_prompt():
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    msg = ChatMessage(message="Hello", system_prompt="")
    result = prompt_builder(log, secrets, msg)
    assert isinstance(result, FormattedPrompt)
    assert result.user == "Hello"
    assert len(result.system) > 0


def test_prompt_builder_uses_custom_system_prompt():
    log = _NoOpLogger()
    secrets = _NoOpSecrets()
    msg = ChatMessage(message="What is Axiom?", system_prompt="Be brief.")
    result = prompt_builder(log, secrets, msg)
    assert result.system == "Be brief."
    assert result.user == "What is Axiom?"
