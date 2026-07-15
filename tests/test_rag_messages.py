from backend.rag.messages import (
    inject_rag_context,
)
from backend.rag.models import (
    RAGContext,
    RAGSource,
)


def test_empty_rag_context_does_not_modify_messages() -> None:
    messages = [
        {
            "role": "system",
            "content": "System",
        },
        {
            "role": "user",
            "content": "Question",
        },
    ]

    result = inject_rag_context(
        messages,
        RAGContext(
            text="",
            sources=(),
        ),
    )

    assert result == messages
    assert result is not messages


def test_rag_context_is_inserted_before_latest_user() -> None:
    messages = [
        {
            "role": "system",
            "content": "System",
        },
        {
            "role": "assistant",
            "content": "Previous response",
        },
        {
            "role": "user",
            "content": "Current question",
        },
    ]

    context = RAGContext(
        text="Document context",
        sources=(
            RAGSource(
                number=1,
                document_id="doc123",
                filename="report.pdf",
                chunk_index=0,
                content="Relevant content",
                similarity=0.9,
            ),
        ),
    )

    result = inject_rag_context(
        messages,
        context,
    )

    assert result[-2] == {
        "role": "system",
        "content": "Document context",
    }

    assert result[-1] == {
        "role": "user",
        "content": "Current question",
    }


def test_original_messages_are_not_mutated() -> None:
    messages = [
        {
            "role": "user",
            "content": "Question",
        }
    ]

    context = RAGContext(
        text="Context",
        sources=(
            RAGSource(
                number=1,
                document_id="doc",
                filename="file.pdf",
                chunk_index=0,
                content="Content",
                similarity=1.0,
            ),
        ),
    )

    inject_rag_context(
        messages,
        context,
    )

    assert len(messages) == 1