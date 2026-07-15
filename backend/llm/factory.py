"""LLM provider factory."""

from backend.config import settings
from backend.llm.providers.ollama_provider import OllamaProvider
from backend.llm.providers.openai_compatible import OpenAICompatibleProvider
from backend.llm.types import LLMProvider


def create_provider(name: str | None = None) -> LLMProvider:
    provider_name = (name or settings.llm_provider).strip().casefold()
    if provider_name == "ollama":
        return OllamaProvider(settings.ollama_host)
    if provider_name in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleProvider(
            base_url=settings.openai_base_url,
            api_key=settings.openai_api_key,
            temperature=settings.llm_temperature,
        )
    raise ValueError(f"Unsupported LLM provider: {provider_name}")
