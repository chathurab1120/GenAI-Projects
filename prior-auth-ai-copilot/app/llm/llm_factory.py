from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from dataclasses import dataclass
from app.core.config import get_settings
from app.core.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured response from an LLM call including token usage."""
    content: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    model: str


class LLMClient:
    """
    Wrapper for OpenAI chat completions.
    All calls use temperature=0.0 for deterministic outputs.
    All calls include retry logic and token usage logging.
    """

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.llm_model
        self._max_tokens = settings.default_max_tokens
        self._temperature = settings.default_temperature
        logger.info(f"LLMClient initialised with model: {self._model}")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    def complete(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request to OpenAI.

        Args:
            system_prompt: The system instruction for the LLM.
            user_prompt: The user message containing the case details.
            temperature: Override default temperature if needed.
            max_tokens: Override default max_tokens if needed.

        Returns:
            LLMResponse with content and token usage.
        """
        response = self._client.chat.completions.create(
            model=self._model,
            temperature=temperature if temperature is not None
                else self._temperature,
            max_tokens=max_tokens if max_tokens is not None
                else self._max_tokens,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )

        usage = response.usage
        result = LLMResponse(
            content=response.choices[0].message.content or "",
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
            model=self._model,
        )

        logger.info(
            f"LLM call complete | "
            f"prompt_tokens={result.prompt_tokens} | "
            f"completion_tokens={result.completion_tokens} | "
            f"total_tokens={result.total_tokens}"
        )
        return result
