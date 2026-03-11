from gen.messages_pb2 import RetrievalContext, FormattedPrompt
from gen.axiom_logger import AxiomLogger, AxiomSecrets

SYSTEM_PROMPT = (
    "You are a helpful assistant. Answer the user's question using only the "
    "provided context. If the context does not contain enough information to "
    "answer confidently, say so clearly."
)


def context_builder(log: AxiomLogger, secrets: AxiomSecrets, input: RetrievalContext) -> FormattedPrompt:
    """Formats retrieved document chunks and a user question into a structured prompt for an LLM.

    Concatenates all retrieved chunks into a numbered context block and
    combines it with the user's question as the user turn. The resulting
    FormattedPrompt is ready to be passed directly to AnthropicConnector or
    any other LLM connector node.
    """
    if not input.chunks:
        log.warn("context_builder: no chunks provided, building prompt without context")
        context_block = "(No context available.)"
    else:
        numbered = "\n\n".join(f"[{i + 1}] {chunk}" for i, chunk in enumerate(input.chunks))
        context_block = f"Context:\n{numbered}"

    user_turn = f"{context_block}\n\nQuestion: {input.question}"
    log.info("context_builder: built prompt", chunks=len(input.chunks), user_len=len(user_turn))
    return FormattedPrompt(system=SYSTEM_PROMPT, user=user_turn)
