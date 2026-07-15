"""Shared LLM provider types."""

from dataclasses import dataclass
from typing import Iterator, Protocol


NANOSECONDS_IN_SECOND = 1_000_000_000


@dataclass(slots=True)
class StreamMetrics:
    """Metrics collected from a streaming LLM response."""

    model: str
    eval_count: int = 0
    eval_duration: int = 0
    total_duration: int = 0

    @property
    def generation_seconds(self) -> float:
        return self.eval_duration / NANOSECONDS_IN_SECOND

    @property
    def total_seconds(self) -> float:
        return self.total_duration / NANOSECONDS_IN_SECOND

    @property
    def tokens_per_second(self) -> float:
        seconds = self.generation_seconds
        return self.eval_count / seconds if seconds > 0 else 0.0


class LLMStream(Protocol):
    """Streaming response contract."""

    metrics: StreamMetrics

    def __iter__(self) -> Iterator[str]: ...


class LLMProvider(Protocol):
    """Provider-independent LLM contract."""

    name: str

    def stream_chat(self, messages: list[dict[str, str]], model: str) -> LLMStream: ...

    def list_models(self) -> list[str]: ...

    def healthcheck(self, model: str | None = None) -> None: ...
