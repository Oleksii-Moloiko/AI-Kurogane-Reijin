"""Ollama embedding provider."""

from collections.abc import Iterable

import httpx

from backend.rag.embeddings.base import EmbeddingProvider
from backend.rag.embeddings.errors import (
    EmbeddingModelNotFoundError,
    EmbeddingProviderUnavailableError,
    InvalidEmbeddingError,
)


class OllamaEmbeddingProvider(EmbeddingProvider):
    """Generate embeddings through the Ollama HTTP API."""

    def __init__(
        self,
        *,
        host: str,
        model: str,
        timeout: float = 120.0,
    ) -> None:
        self.host = host.rstrip("/")
        self.model = model
        self.timeout = timeout

    @property
    def provider_name(self) -> str:
        return "ollama"

    @property
    def model_name(self) -> str:
        return self.model

    def healthcheck(self) -> None:
        try:
            response = httpx.get(
                f"{self.host}/api/tags",
                timeout=10.0,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise EmbeddingProviderUnavailableError(
                "Не вдалося підключитися до Ollama. "
                "Перевір, чи запущено `ollama serve`."
            ) from error

        payload = response.json()
        available_models = {
            item.get("name", "")
            for item in payload.get("models", [])
        }

        if not self._model_is_available(
            self.model,
            available_models,
        ):
            raise EmbeddingModelNotFoundError(
                f"Embedding-модель '{self.model}' не знайдена. "
                f"Виконай: ollama pull {self.model}"
            )

    def embed_texts(
        self,
        texts: list[str],
    ) -> list[list[float]]:
        if not texts:
            return []

        self._validate_texts(texts)

        try:
            response = httpx.post(
                f"{self.host}/api/embed",
                json={
                    "model": self.model,
                    "input": texts,
                },
                timeout=self.timeout,
            )
            response.raise_for_status()
        except httpx.ConnectError as error:
            raise EmbeddingProviderUnavailableError(
                "Ollama недоступна. Перевір `ollama serve`."
            ) from error
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 404:
                raise EmbeddingModelNotFoundError(
                    f"Embedding-модель '{self.model}' не знайдена."
                ) from error

            raise EmbeddingProviderUnavailableError(
                f"Ollama повернула HTTP "
                f"{error.response.status_code}."
            ) from error
        except httpx.HTTPError as error:
            raise EmbeddingProviderUnavailableError(
                f"Помилка запиту embeddings: {error}"
            ) from error

        payload = response.json()
        embeddings = payload.get("embeddings")

        self._validate_embeddings(
            embeddings,
            expected_count=len(texts),
        )

        return embeddings

    @staticmethod
    def _validate_texts(
        texts: list[str],
    ) -> None:
        if any(not isinstance(text, str) for text in texts):
            raise TypeError("All embedding inputs must be strings.")

        if any(not text.strip() for text in texts):
            raise ValueError(
                "Embedding input cannot contain empty text."
            )

    @staticmethod
    def _validate_embeddings(
        embeddings: object,
        *,
        expected_count: int,
    ) -> None:
        if not isinstance(embeddings, list):
            raise InvalidEmbeddingError(
                "Ollama не повернула список embeddings."
            )

        if len(embeddings) != expected_count:
            raise InvalidEmbeddingError(
                "Кількість embeddings не відповідає "
                "кількості вхідних текстів."
            )

        dimensions: set[int] = set()

        for vector in embeddings:
            if not isinstance(vector, list) or not vector:
                raise InvalidEmbeddingError(
                    "Отримано порожній або некоректний embedding."
                )

            if not all(
                isinstance(value, int | float)
                for value in vector
            ):
                raise InvalidEmbeddingError(
                    "Embedding містить нечислові значення."
                )

            dimensions.add(len(vector))

        if len(dimensions) != 1:
            raise InvalidEmbeddingError(
                "Embeddings мають різну розмірність."
            )

    @staticmethod
    def _model_is_available(
        requested_model: str,
        available_models: Iterable[str],
    ) -> bool:
        requested = requested_model.casefold()

        for available_model in available_models:
            available = available_model.casefold()

            if available == requested:
                return True

            if available.split(":", maxsplit=1)[0] == requested:
                return True

        return False