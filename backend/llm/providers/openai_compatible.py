"""OpenAI-compatible HTTP provider."""

import json
import time
from collections.abc import Iterator

import httpx

from backend.llm.errors import ModelUnavailableError, ProviderUnavailableError
from backend.llm.types import StreamMetrics


class OpenAICompatibleStream:
    def __init__(self, client: httpx.Client, base_url: str, api_key: str, messages: list[dict[str, str]], model: str, temperature: float) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.messages = messages
        self.model = model
        self.temperature = temperature
        self.metrics = StreamMetrics(model=model)

    def __iter__(self) -> Iterator[str]:
        started = time.perf_counter()
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": self.temperature,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        try:
            with self.client.stream(
                "POST",
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
            ) as response:
                if response.status_code == 404:
                    raise ModelUnavailableError(f"Модель або endpoint для '{self.model}' не знайдено.")
                response.raise_for_status()
                for line in response.iter_lines():
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    chunk = json.loads(data)
                    choices = chunk.get("choices", [])
                    if choices:
                        content = choices[0].get("delta", {}).get("content") or ""
                        if content:
                            yield content
                    usage = chunk.get("usage") or {}
                    if usage:
                        self.metrics.eval_count = int(usage.get("completion_tokens", 0))
        except httpx.HTTPStatusError as error:
            raise ProviderUnavailableError(
                f"OpenAI-compatible API повернув HTTP {error.response.status_code}."
            ) from error
        except (httpx.RequestError, json.JSONDecodeError) as error:
            raise ProviderUnavailableError(f"OpenAI-compatible API недоступний: {error}") from error
        finally:
            elapsed = max(time.perf_counter() - started, 0)
            self.metrics.total_duration = int(elapsed * 1_000_000_000)
            self.metrics.eval_duration = self.metrics.total_duration


class OpenAICompatibleProvider:
    name = "openai"

    def __init__(self, base_url: str, api_key: str, temperature: float, timeout: float = 120.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature
        self.client = httpx.Client(timeout=timeout)

    def stream_chat(self, messages: list[dict[str, str]], model: str) -> OpenAICompatibleStream:
        return OpenAICompatibleStream(
            self.client,
            self.base_url,
            self.api_key,
            messages,
            model,
            self.temperature,
        )

    def list_models(self) -> list[str]:
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            response = self.client.get(f"{self.base_url}/models", headers=headers)
            response.raise_for_status()
            return [str(item["id"]) for item in response.json().get("data", []) if item.get("id")]
        except httpx.RequestError as error:
            raise ProviderUnavailableError(f"OpenAI-compatible API недоступний: {error}") from error
        except httpx.HTTPStatusError as error:
            raise ProviderUnavailableError(
                f"Не вдалося отримати список моделей: HTTP {error.response.status_code}."
            ) from error

    def healthcheck(self, model: str | None = None) -> None:
        models = self.list_models()
        if model and models and model not in models:
            raise ModelUnavailableError(
                f"Модель '{model}' відсутня у списку моделей OpenAI-compatible API."
            )
