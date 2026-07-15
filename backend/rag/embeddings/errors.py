"""Embedding-related exceptions."""


class EmbeddingError(Exception):
    """Base exception for embedding failures."""


class EmbeddingProviderUnavailableError(EmbeddingError):
    """Raised when the embedding provider cannot be reached."""


class EmbeddingModelNotFoundError(EmbeddingError):
    """Raised when the configured embedding model is unavailable."""


class InvalidEmbeddingError(EmbeddingError):
    """Raised when the provider returns invalid vectors."""