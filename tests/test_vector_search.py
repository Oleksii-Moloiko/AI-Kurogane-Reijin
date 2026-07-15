from pathlib import Path

from backend.rag.models import TextChunk
from backend.rag.search import VectorSearchService
from backend.storage.sqlite import ChatRepository
from tests.fakes import MappingEmbeddingProvider


def create_indexed_document(
    repository: ChatRepository,
    *,
    filename: str,
    content_hash: str,
    chunks: list[TextChunk],
    embeddings: list[list[float]],
) -> str:
    document = repository.create_document(
        filename=filename,
        source_path=f"/tmp/{filename}",
        extension=".pdf",
        size_bytes=100,
        content_hash=content_hash,
        page_count=1,
    )

    repository.create_document_chunks(
        document.id,
        chunks,
    )

    repository.update_chunk_embeddings(
        document.id,
        embeddings,
    )

    repository.update_document_status(
        document.id,
        "indexed",
    )

    return document.id


def test_search_returns_most_similar_chunks(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    create_indexed_document(
        repository,
        filename="finance.pdf",
        content_hash="f" * 64,
        chunks=[
            TextChunk(
                index=0,
                content="Revenue increased.",
                character_start=0,
                character_end=18,
                page_number=1,
            ),
            TextChunk(
                index=1,
                content="Office relocation.",
                character_start=19,
                character_end=37,
                page_number=2,
            ),
        ],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    provider = MappingEmbeddingProvider(
        {
            "What happened to revenue?": [
                0.95,
                0.05,
            ]
        }
    )

    service = VectorSearchService(
        repository=repository,
        embedding_provider=provider,
        top_k=5,
        min_similarity=0.0,
    )

    results = service.search(
        "What happened to revenue?"
    )

    assert len(results) == 2
    assert results[0].content == "Revenue increased."
    assert results[0].filename == "finance.pdf"
    assert results[0].page_number == 1
    assert results[0].similarity > (
        results[1].similarity
    )


def test_search_applies_top_k(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    create_indexed_document(
        repository,
        filename="document.pdf",
        content_hash="1" * 64,
        chunks=[
            TextChunk(
                index=index,
                content=f"Chunk {index}",
                character_start=index * 10,
                character_end=index * 10 + 7,
            )
            for index in range(3)
        ],
        embeddings=[
            [1.0, 0.0],
            [0.9, 0.1],
            [0.8, 0.2],
        ],
    )

    provider = MappingEmbeddingProvider(
        {
            "query": [1.0, 0.0],
        }
    )

    service = VectorSearchService(
        repository=repository,
        embedding_provider=provider,
        top_k=2,
        min_similarity=0.0,
    )

    results = service.search("query")

    assert len(results) == 2
    assert results[0].chunk_index == 0
    assert results[1].chunk_index == 1


def test_search_applies_similarity_threshold(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    create_indexed_document(
        repository,
        filename="document.pdf",
        content_hash="2" * 64,
        chunks=[
            TextChunk(
                index=0,
                content="Relevant",
                character_start=0,
                character_end=8,
            ),
            TextChunk(
                index=1,
                content="Irrelevant",
                character_start=9,
                character_end=19,
            ),
        ],
        embeddings=[
            [1.0, 0.0],
            [0.0, 1.0],
        ],
    )

    provider = MappingEmbeddingProvider(
        {
            "query": [1.0, 0.0],
        }
    )

    service = VectorSearchService(
        repository=repository,
        embedding_provider=provider,
        top_k=5,
        min_similarity=0.5,
    )

    results = service.search("query")

    assert len(results) == 1
    assert results[0].content == "Relevant"


def test_empty_query_returns_no_results(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    provider = MappingEmbeddingProvider({})

    service = VectorSearchService(
        repository=repository,
        embedding_provider=provider,
    )

    assert service.search("   ") == []