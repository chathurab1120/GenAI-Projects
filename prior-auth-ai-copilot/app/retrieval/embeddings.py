from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


class EmbeddingClient:
    """
    Wrapper for OpenAI text embeddings.
    Uses text-embedding-3-small by default.
    All calls include retry logic for reliability.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.embedding_model
        logger.info(f"EmbeddingClient initialised with model: {self._model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_text(self, text: str) -> list[float]:
        """
        Embed a single text string.

        Args:
            text: The text to embed.

        Returns:
            List of floats representing the embedding vector.
        """
        response = self._client.embeddings.create(
            input=text,
            model=self._model,
        )
        logger.debug(f"Embedded text ({len(text)} chars)")
        return response.data[0].embedding

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a batch of text strings in a single API call.

        Args:
            texts: List of strings to embed.

        Returns:
            List of embedding vectors in the same order as input.
        """
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        logger.info(f"Embedded batch of {len(texts)} texts")
        return [item.embedding for item in response.data]
