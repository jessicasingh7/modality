import openai

from modality.config import settings

_client: openai.AsyncOpenAI | None = None


def _get_client() -> openai.AsyncOpenAI:
    global _client
    if _client is None:
        _client = openai.AsyncOpenAI(api_key=settings.openai_api_key)
    return _client


async def get_embedding(text: str) -> list[float]:
    """Get an embedding vector for the given text using OpenAI's embedding API."""
    client = _get_client()
    response = await client.embeddings.create(
        model=settings.router_embedding_model,
        input=text,
    )
    return response.data[0].embedding
