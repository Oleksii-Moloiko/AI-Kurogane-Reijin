from backend.rag.context import (
    RAGContextBuilder,
)
from backend.rag.runtime import RAGRuntime


class FailingSearchService:
    def search(
        self,
        query: str,
    ):
        raise RuntimeError(
            "Search failed"
        )


def test_runtime_returns_empty_context_when_search_fails() -> None:
    runtime = RAGRuntime(
        search_service=FailingSearchService(),
        context_builder=RAGContextBuilder(
            max_context_chars=1000
        ),
    )

    context = runtime.retrieve(
        "Question"
    )

    assert context.is_empty


def test_disabled_runtime_does_not_search() -> None:
    runtime = RAGRuntime(
        search_service=FailingSearchService(),
        context_builder=RAGContextBuilder(
            max_context_chars=1000
        ),
        enabled=False,
    )

    context = runtime.retrieve(
        "Question"
    )

    assert context.is_empty