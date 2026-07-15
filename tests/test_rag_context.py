from backend.rag.context import (
    RAGContextBuilder,
)
from backend.rag.models import SearchResult


def create_result(
    *,
    content: str,
    similarity: float = 0.9,
    chunk_index: int = 0,
    page_number: int | None = None,
) -> SearchResult:
    return SearchResult(
        chunk_id=chunk_index + 1,
        document_id="doc123",
        filename="report.pdf",
        chunk_index=chunk_index,
        content=content,
        similarity=similarity,
        page_number=page_number,
    )


def test_empty_results_create_empty_context() -> None:
    builder = RAGContextBuilder(
        max_context_chars=1000
    )

    context = builder.build([])

    assert context.is_empty
    assert context.text == ""
    assert context.sources == ()


def test_context_contains_document_content() -> None:
    builder = RAGContextBuilder(
        max_context_chars=1000
    )

    context = builder.build(
        [
            create_result(
                content="Revenue increased by 20%.",
                page_number=4,
            )
        ]
    )

    assert not context.is_empty
    assert "Revenue increased by 20%." in context.text
    assert "report.pdf" in context.text
    assert "Сторінка: 4" in context.text

    assert len(context.sources) == 1
    assert context.sources[0].number == 1
    assert context.sources[0].page_number == 4


def test_context_respects_character_limit() -> None:
    builder = RAGContextBuilder(
        max_context_chars=100
    )

    context = builder.build(
        [
            create_result(
                content="A" * 1000,
            )
        ]
    )

    context_body = context.text.split(
        "КОНТЕКСТ:\n",
        maxsplit=1,
    )[1]

    assert len(context_body) <= 100


def test_sources_are_numbered() -> None:
    builder = RAGContextBuilder(
        max_context_chars=2000
    )

    context = builder.build(
        [
            create_result(
                content="First",
                chunk_index=0,
            ),
            create_result(
                content="Second",
                chunk_index=1,
            ),
        ]
    )

    assert context.sources[0].number == 1
    assert context.sources[1].number == 2

    assert "[ДЖЕРЕЛО 1]" in context.text
    assert "[ДЖЕРЕЛО 2]" in context.text