"""Ollama LLM provider."""

from collections.abc import Iterator

import httpx
import ollama

from backend.llm.errors import ModelUnavailableError, ProviderUnavailableError
from backend.llm.types import StreamMetrics


class OllamaStream:
    def __init__(self, client: ollama.Client, messages: list[dict[str, str]], model: str) -> None:
        self.client = client
        self.messages = messages
        self.model = model
        self.metrics = StreamMetrics(model=model)

    def __iter__(self) -> Iterator[str]:
        try:
            stream = self.client.chat(model=self.model, messages=self.messages, stream=True)
            for part in stream:
                message = getattr(part, "message", None)
                content = getattr(message, "content", "") if message else ""
                if not content and isinstance(part, dict):
                    content = part.get("message", {}).get("content", "")
                if content:
                    yield content

                done = getattr(part, "done", False)
                if not done and isinstance(part, dict):
                    done = part.get("done", False)
                if done:
                    self.metrics.eval_count = getattr(part, "eval_count", 0) or 0
                    self.metrics.eval_duration = getattr(part, "eval_duration", 0) or 0
                    self.metrics.total_duration = getattr(part, "total_duration", 0) or 0
        except ollama.ResponseError as error:
            if getattr(error, "status_code", None) == 404:
                raise ModelUnavailableError(
                    f"Модель '{self.model}' не знайдена. Виконайте: ollama pull {self.model}"
                ) from error
            raise ProviderUnavailableError(f"Помилка Ollama: {error}") from error
        except (httpx.RequestError, ConnectionError, OSError) as error:
            raise ProviderUnavailableError(
                "Ollama недоступна. Переконайтеся, що сервіс запущено командою 'ollama serve'."
            ) from error


class OllamaProvider:
    name = "ollama"

    def __init__(self, host: str) -> None:
        self.client = ollama.Client(host=host)

    def stream_chat(self, messages: list[dict[str, str]], model: str) -> OllamaStream:
        return OllamaStream(self.client, messages, model)

    def list_models(self) -> list[str]:
        try:
            response = self.client.list()
        except (httpx.RequestError, ConnectionError, OSError) as error:
            raise ProviderUnavailableError(
                "Ollama недоступна. Запустіть її командою 'ollama serve'."
            ) from error

        models = getattr(response, "models", None)
        if models is None and isinstance(response, dict):
            models = response.get("models", [])

        result: list[str] = []
        for item in models or []:
            name = getattr(item, "model", None)
            if name is None and isinstance(item, dict):
                name = item.get("model") or item.get("name")
            if name:
                result.append(str(name))
        return result

    def healthcheck(self, model: str | None = None) -> None:
        models = self.list_models()
        available = {name.casefold() for name in models}
        available.update(name.split(":", 1)[0].casefold() for name in models)
        if model and model.casefold() not in available:
            raise ModelUnavailableError(
                f"Модель '{model}' не знайдена. Виконайте: ollama pull {model}"
            )
