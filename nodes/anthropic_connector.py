from typing import Iterator

from gen.messages_pb2 import FormattedPrompt, TokenChunk
from gen.axiom_logger import AxiomLogger, AxiomSecrets


# AnthropicConnector: Calls the Anthropic Messages API with streaming enabled.
# inputs: stream of FormattedPrompt frames from the previous node.
# For the start node of a pipeline flow, the iterator yields exactly one item.
def anthropic_connector(log: AxiomLogger, secrets: AxiomSecrets, inputs: Iterator[FormattedPrompt]) -> Iterator[TokenChunk]:
    """Calls the Anthropic Messages API with streaming and yields one TokenChunk per token.

    Reads ANTHROPIC_API_KEY from secrets. Uses claude-sonnet-4-5 by
    default. Each streamed text delta is emitted as a TokenChunk frame with
    is_final=False; the final frame has is_final=True and an empty text field.
    """
    import anthropic

    api_key, ok = secrets.get("ANTHROPIC_API_KEY")
    if not ok:
        log.error("anthropic_connector: ANTHROPIC_API_KEY secret not found")
        yield TokenChunk(text="Error: ANTHROPIC_API_KEY secret not registered.", is_final=True)
        return

    client = anthropic.Anthropic(api_key=api_key)

    for prompt in inputs:
        log.info("anthropic_connector: starting stream", model="claude-sonnet-4-5")
        with client.messages.stream(
            model="claude-sonnet-4-5",
            max_tokens=2048,
            system=prompt.system,
            messages=[{"role": "user", "content": prompt.user}],
        ) as stream:
            for text in stream.text_stream:
                yield TokenChunk(text=text, is_final=False)

        yield TokenChunk(text="", is_final=True)
        log.info("anthropic_connector: stream complete")
