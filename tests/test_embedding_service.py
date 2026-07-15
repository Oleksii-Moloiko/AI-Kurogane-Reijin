from pathlib import Path

from backend.rag.embedding_service import (
    DocumentEmbeddingService,
)
from backend.rag.models import TextChunk
from backend.storage.sqlite import ChatRepository
from tests.fakes import FakeEmbeddingProvider


def test_document_embeddings_are_generated_and_saved(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    document = repository.create_document(
        filename="report.pdf",
        source_path="/tmp/report.pdf",
        extension=".pdf",
        size_bytes=100,
        content_hash="d" * 64,
        page_count=1,
    )

    repository.create_document_chunks(
        document.id,
        [
            TextChunk(
                index=0,
                content="First",
                character_start=0,
                character_end=5,
            ),
            TextChunk(
                index=1,
                content="Second",
                character_start=4,
                character_end=10,
            ),
        ],
    )

    provider = FakeEmbeddingProvider()

    service = DocumentEmbeddingService(
        repository=repository,
        provider=provider,
        batch_size=1,
    )

    embedded_count = service.embed_document(
        document.id
    )

    repository.update_document_status(
        document.id,
        "indexed",
    )

    stored = repository.list_embedded_chunks()

    assert embedded_count == 2
    assert len(stored) == 2
    assert stored[0].embedding == [5.0, 0.0, 1.0]
    assert stored[1].embedding == [6.0, 0.0, 1.0]
    assert len(provider.calls) == 2