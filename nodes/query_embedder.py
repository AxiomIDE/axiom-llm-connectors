from gen.messages_pb2 import EmbeddingRequest, EmbeddingVector
from gen.axiom_logger import AxiomLogger, AxiomSecrets

DEFAULT_MODEL = "text-embedding-3-small"


def query_embedder(log: AxiomLogger, secrets: AxiomSecrets, input: EmbeddingRequest) -> EmbeddingVector:
    """Embeds a user query into a dense vector for nearest-neighbour retrieval.

    Reads OPENAI_API_KEY from secrets. Uses the model specified in
    EmbeddingRequest.model, defaulting to text-embedding-3-small if empty.
    Unlike EmbeddingGenerator this is a unary node — it embeds a single query
    and returns a single EmbeddingVector, making it suitable as the first step
    in a RAG query graph ahead of PineconeRetriever.
    """
    from openai import OpenAI

    api_key, ok = secrets.get("OPENAI_API_KEY")
    if not ok:
        log.error("query_embedder: OPENAI_API_KEY secret not found")
        return EmbeddingVector()

    client = OpenAI(api_key=api_key)
    model = input.model if input.model else DEFAULT_MODEL
    log.info("query_embedder: embedding query", model=model, text_len=len(input.text))
    response = client.embeddings.create(input=input.text, model=model)
    values = response.data[0].embedding
    log.info("query_embedder: done", dim=len(values))
    return EmbeddingVector(values=values, input_text=input.text)
