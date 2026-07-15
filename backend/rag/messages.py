"""Insert retrieved context into an LLM message list."""

from collections.abc import Mapping
from typing import Any

from backend.rag.models import RAGContext


Message = dict[str, Any]


def inject_rag_context(
    messages: list[Mapping[str, Any]],
    rag_context: RAGContext,
) -> list[Message]:
    """Insert RAG context before the latest user message."""

    copied_messages = [
        dict(message)
        for message in messages
    ]

    if rag_context.is_empty:
        return copied_messages

    context_message: Message = {
        "role": "system",
        "content": rag_context.text,
    }

    latest_user_index = _find_latest_user_index(
        copied_messages
    )

    if latest_user_index is None:
        copied_messages.append(
            context_message
        )
    else:
        copied_messages.insert(
            latest_user_index,
            context_message,
        )

    return copied_messages


def _find_latest_user_index(
    messages: list[Message],
) -> int | None:
    for index in range(
        len(messages) - 1,
        -1,
        -1,
    ):
        if messages[index].get("role") == "user":
            return index

    return None