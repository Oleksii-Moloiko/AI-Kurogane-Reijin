"""Embedding provider factory."""

from backend.config.settings import Settings
from backend.rag.embeddings.base import EmbeddingProvider
from backend.rag.embeddings.ollama import (
    OllamaEmbeddingProvider,
)


def create_embedding_provider(
    settings: Settings,
) -> EmbeddingProvider:
    provider_name = settings.embedding_provider

    if provider_name == "ollama":
        return OllamaEmbeddingProvider(
            host=settings.ollama_host,
            model=settings.embedding_model,
        )

    raise ValueError(
        f"Unsupported embedding provider: {provider_name}"
    )