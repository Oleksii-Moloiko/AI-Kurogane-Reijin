"""Terminal rendering for RAG documents."""

from rich.table import Table

from backend.rag.models import IndexedDocument
from backend.storage.sqlite import DocumentRecord
from backend.ui.console import console


def print_indexed_document(
    document: IndexedDocument,
) -> None:
    table = Table(
        title="Документ проіндексовано",
        show_header=False,
    )

    table.add_column(
        "Поле",
        style="bold cyan",
        no_wrap=True,
    )
    table.add_column("Значення")

    table.add_row(
        "Назва",
        document.filename,
    )
    table.add_row(
        "ID",
        document.document_id,
    )
    table.add_row(
        "Chunks",
        str(document.chunk_count),
    )
    table.add_row(
        "Embedding model",
        document.embedding_model,
    )
    table.add_row(
        "Розмір",
        format_file_size(
            document.size_bytes
        ),
    )

    if document.page_count is not None:
        table.add_row(
            "Сторінки",
            str(document.page_count),
        )

    table.add_row(
        "Локальна копія",
        str(document.stored_path),
    )

    console.print(table)


def print_documents(
    documents: list[DocumentRecord],
    *,
    chunk_counts: dict[str, int],
) -> None:
    if not documents:
        console.print(
            "[yellow]Проіндексованих документів ще немає.[/yellow]"
        )
        return

    table = Table(
        title="Документи",
        show_lines=True,
    )

    table.add_column(
        "#",
        justify="right",
    )
    table.add_column(
        "Назва",
        style="bold",
    )
    table.add_column(
        "ID",
        style="cyan",
    )
    table.add_column("Формат")
    table.add_column(
        "Chunks",
        justify="right",
    )
    table.add_column("Статус")
    table.add_column(
        "Розмір",
        justify="right",
    )

    for index, document in enumerate(
        documents,
        start=1,
    ):
        table.add_row(
            str(index),
            document.filename,
            document.id,
            document.extension.lstrip(".").upper(),
            str(
                chunk_counts.get(
                    document.id,
                    0,
                )
            ),
            document.status,
            format_file_size(
                document.size_bytes
            ),
        )

    console.print(table)


def print_document_removed(
    *,
    filename: str,
    document_id: str,
) -> None:
    console.print(
        f"[green]Документ видалено:[/green] "
        f"{filename} [dim]({document_id})[/dim]"
    )


def format_file_size(
    size_bytes: int,
) -> str:
    size = float(size_bytes)

    for unit in (
        "B",
        "KB",
        "MB",
        "GB",
    ):
        if size < 1024 or unit == "GB":
            if unit == "B":
                return f"{int(size)} {unit}"

            return f"{size:.1f} {unit}"

        size /= 1024

    return f"{size:.1f} GB"