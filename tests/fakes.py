from backend.rag.embeddings.base import EmbeddingProvider
from backend.rag.embeddings.errors import (
    EmbeddingProviderUnavailableError,
)


class FailingEmbeddingProvider(EmbeddingProvider):
    @property
    def provider_name(self) -> str:
        return "failing"

    @property
    def model_name(self) -> str:
        return "failing-model"

    def healthcheck(self) -> None:
        raise EmbeddingProviderUnavailableError(
            "Provider unavailable."
        )

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        raise EmbeddingProviderUnavailableError(
            "Provider unavailable."
        )

class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        dimension: int = 3,
    ) -> None:
        self.dimension = dimension
        self.calls: list[list[str]] = []

    @property
    def provider_name(self) -> str:
        return "fake"

    @property
    def model_name(self) -> str:
        return "fake-model"

    def healthcheck(self) -> None:
        return None

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        self.calls.append(texts)

        return [
            [
                float(len(text)),
                float(index),
                1.0,
            ][: self.dimension]
            for index, text in enumerate(texts)
        ]
    
class MappingEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        embeddings: dict[str, list[float]],
    ) -> None:
        self.embeddings = embeddings

    @property
    def provider_name(self) -> str:
        return "mapping-fake"

    @property
    def model_name(self) -> str:
        return "mapping-model"

    def healthcheck(self) -> None:
        return None

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        return [
            self.embeddings[text]
            for text in texts
        ]