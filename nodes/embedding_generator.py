from typing import Iterator
from gen.messages_pb2 import EmbeddingRequest, EmbeddingVector
from gen.axiom_logger import AxiomLogger, AxiomSecrets

DEFAULT_MODEL = "text-embedding-3-small"


def embedding_generator(log: AxiomLogger, secrets: AxiomSecrets, inputs: Iterator[EmbeddingRequest]) -> Iterator[EmbeddingVector]:
    """Calls the OpenAI embeddings API for each input frame and streams one EmbeddingVector per frame.

    Reads OPENAI_API_KEY from secrets. Uses the model specified in
    EmbeddingRequest.model, defaulting to text-embedding-3-small if empty.
    The input text is carried through in the output for downstream use.
    """
    from openai import OpenAI

    api_key, ok = secrets.get("OPENAI_API_KEY")
    if not ok:
        log.error("embedding_generator: OPENAI_API_KEY secret not found")
        return

    client = OpenAI(api_key=api_key)

    for input in inputs:
        model = input.model if input.model else DEFAULT_MODEL
        log.info("embedding_generator: embedding text", model=model, text_len=len(input.text))
        response = client.embeddings.create(input=input.text, model=model)
        values = response.data[0].embedding
        log.info("embedding_generator: done", dim=len(values))
        yield EmbeddingVector(values=values, input_text=input.text)
