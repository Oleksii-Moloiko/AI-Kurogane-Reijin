"""Client for communication with Ollama."""

from dataclasses import dataclass
from typing import Iterator

import ollama


NANOSECONDS_IN_SECOND = 1_000_000_000


@dataclass
class StreamMetrics:
    """Metrics collected from an Ollama streaming response."""

    model: str
    eval_count: int = 0
    eval_duration: int = 0
    total_duration: int = 0

    @property
    def generation_seconds(self) -> float:
        """Return token generation duration in seconds."""

        return self.eval_duration / NANOSECONDS_IN_SECOND

    @property
    def total_seconds(self) -> float:
        """Return total request duration in seconds."""

        return self.total_duration / NANOSECONDS_IN_SECOND

    @property
    def tokens_per_second(self) -> float:
        """Calculate token generation speed."""

        seconds = self.generation_seconds

        if seconds <= 0:
            return 0.0

        return self.eval_count / seconds


class OllamaStream:
    """Iterable Ollama response that also stores final metrics."""

    def __init__(
        self,
        history: list[dict[str, str]],
        model: str,
    ) -> None:
        self.history = history
        self.model = model
        self.metrics = StreamMetrics(model=model)

    def __iter__(self) -> Iterator[str]:
        stream = ollama.chat(
            model=self.model,
            messages=self.history,
            stream=True,
        )

        for part in stream:
            content = part.get("message", {}).get("content", "")

            if content:
                yield content

            if part.get("done"):
                self.metrics.eval_count = part.get("eval_count", 0)
                self.metrics.eval_duration = part.get("eval_duration", 0)
                self.metrics.total_duration = part.get("total_duration", 0)


def ask_llm(
    history: list[dict[str, str]],
    model: str = "gemma3",
) -> str:
    """Request a complete response from Ollama."""

    response = ollama.chat(
        model=model,
        messages=history,
    )

    return response["message"]["content"]


def ask_llm_stream(
    history: list[dict[str, str]],
    model: str = "gemma3",
) -> OllamaStream:
    """Create a streaming Ollama response with metrics."""

    return OllamaStream(
        history=history,
        model=model,
    )

def get_available_models() -> list[str]:
    """Return names of locally installed Ollama models."""

    response = ollama.list()
    models = getattr(response, "models", None)

    if models is None and isinstance(response, dict):
        models = response.get("models", [])

    result: list[str] = []

    for model in models or []:
        name = getattr(model, "model", None)

        if name is None and isinstance(model, dict):
            name = model.get("model") or model.get("name")

        if name:
            result.append(str(name))

    return result