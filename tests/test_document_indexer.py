from pathlib import Path

from docx import Document as DocxDocument
import pytest

from backend.rag.chunking import TextChunker
from backend.rag.embedding_service import (
    DocumentEmbeddingService,
)
from backend.rag.errors import (
    DocumentIndexingError,
    DuplicateDocumentError,
)
from backend.rag.indexer import DocumentIndexer
from backend.storage.sqlite import ChatRepository
from tests.fakes import (
    FakeEmbeddingProvider,
    FailingEmbeddingProvider,
)


def create_docx(
    path: Path,
    text: str,
) -> None:
    document = DocxDocument()
    document.add_paragraph(text)
    document.save(path)


def create_indexer(
    *,
    repository: ChatRepository,
    documents_dir: Path,
    failing: bool = False,
) -> DocumentIndexer:
    provider = (
        FailingEmbeddingProvider()
        if failing
        else FakeEmbeddingProvider()
    )

    embedding_service = DocumentEmbeddingService(
        repository=repository,
        provider=provider,
        batch_size=2,
    )

    return DocumentIndexer(
        repository=repository,
        chunker=TextChunker(
            chunk_size=50,
            chunk_overlap=10,
        ),
        embedding_service=embedding_service,
        documents_dir=documents_dir,
        max_document_size_mb=5,
    )


def test_document_is_fully_indexed(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "report.docx"

    create_docx(
        source_path,
        "Revenue increased during the quarter. "
        "The mobile application was also updated.",
    )

    repository = ChatRepository(
        tmp_path / "test.db"
    )

    indexer = create_indexer(
        repository=repository,
        documents_dir=tmp_path / "documents",
    )

    result = indexer.index(source_path)

    document = repository.get_document(
        result.document_id
    )

    chunks = repository.list_document_chunks(
        result.document_id
    )

    assert document is not None
    assert document.status == "indexed"
    assert result.chunk_count == len(chunks)
    assert result.chunk_count > 0
    assert result.stored_path.exists()
    assert result.embedding_model == "fake-model"

    assert (
        repository.count_embedded_document_chunks(
            result.document_id
        )
        == result.chunk_count
    )


def test_duplicate_document_is_rejected(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "report.docx"

    create_docx(
        source_path,
        "Same document content.",
    )

    repository = ChatRepository(
        tmp_path / "test.db"
    )

    indexer = create_indexer(
        repository=repository,
        documents_dir=tmp_path / "documents",
    )

    first_result = indexer.index(source_path)

    with pytest.raises(DuplicateDocumentError):
        indexer.index(source_path)

    documents = repository.list_documents()

    assert len(documents) == 1
    assert documents[0].id == (
        first_result.document_id
    )


def test_failed_embedding_rolls_back_database_and_file(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "report.docx"

    create_docx(
        source_path,
        "Document that will fail embedding.",
    )

    repository = ChatRepository(
        tmp_path / "test.db"
    )

    documents_dir = tmp_path / "documents"

    indexer = create_indexer(
        repository=repository,
        documents_dir=documents_dir,
        failing=True,
    )

    with pytest.raises(DocumentIndexingError):
        indexer.index(source_path)

    assert repository.list_documents() == []

    if documents_dir.exists():
        assert list(
            documents_dir.iterdir()
        ) == []


def test_source_file_remains_unchanged(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "report.docx"

    create_docx(
        source_path,
        "Original source document.",
    )

    original_bytes = source_path.read_bytes()

    repository = ChatRepository(
        tmp_path / "test.db"
    )

    indexer = create_indexer(
        repository=repository,
        documents_dir=tmp_path / "documents",
    )

    result = indexer.index(source_path)

    assert source_path.exists()
    assert source_path.read_bytes() == original_bytes
    assert result.stored_path != source_path