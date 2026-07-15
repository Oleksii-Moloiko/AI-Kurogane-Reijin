"""Embedding provider abstraction."""

from abc import ABC, abstractmethod


class EmbeddingProvider(ABC):
    """Generate vector representations for text."""

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider identifier."""

    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the configured model name."""

    @abstractmethod
    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        """Generate one embedding for every input text."""

    def embed_query(
        self,
        query: str,
    ) -> list[float]:
        """Generate an embedding for a search query."""

        vectors = self.embed_texts([query])

        if len(vectors) != 1:
            raise ValueError(
                "Embedding provider returned an unexpected vector count."
            )

        return vectors[0]

    @abstractmethod
    def healthcheck(self) -> None:
        """Verify that the provider and model are available."""