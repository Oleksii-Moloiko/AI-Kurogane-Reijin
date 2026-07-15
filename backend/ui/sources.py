"""Render document sources used for a RAG response."""

from rich.panel import Panel

from backend.rag.models import RAGContext
from backend.ui.console import console


def print_rag_sources(
    rag_context: RAGContext,
) -> None:
    if rag_context.is_empty:
        return

    lines = [
        source.label
        for source in rag_context.sources
    ]

    console.print(
        Panel(
            "\n".join(lines),
            title="Джерела",
            border_style="dim",
        )
    )