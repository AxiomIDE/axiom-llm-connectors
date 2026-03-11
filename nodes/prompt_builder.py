from gen.messages_pb2 import ChatMessage, FormattedPrompt
from gen.axiom_logger import AxiomLogger, AxiomSecrets

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, accurate, and concise assistant. "
    "Answer the user's question directly and clearly."
)


def prompt_builder(log: AxiomLogger, secrets: AxiomSecrets, input: ChatMessage) -> FormattedPrompt:
    """Formats a user chat message into a structured prompt with a system instruction.

    If the input carries a non-empty system_prompt field it is used as the system
    instruction; otherwise a sensible default is applied. The resulting
    FormattedPrompt is consumed by downstream LLM connector nodes such as
    AnthropicConnector.
    """
    system = input.system_prompt if input.system_prompt else DEFAULT_SYSTEM_PROMPT
    log.info("prompt_builder: built prompt", system_len=len(system), user_len=len(input.message))
    return FormattedPrompt(system=system, user=input.message)
