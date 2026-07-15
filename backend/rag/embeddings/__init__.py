"""Embedding providers for the RAG pipeline."""

from backend.rag.embeddings.base import EmbeddingProvider
from backend.rag.embeddings.factory import (
    create_embedding_provider,
)
from backend.rag.embeddings.ollama import (
    OllamaEmbeddingProvider,
)

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbeddingProvider",
    "create_embedding_provider",
]