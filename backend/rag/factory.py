"""Factories for RAG services."""

from backend.config.settings import Settings
from backend.rag.chunking import TextChunker
from backend.rag.embedding_service import (
    DocumentEmbeddingService,
)
from backend.rag.embeddings.factory import (
    create_embedding_provider,
)
from backend.rag.context import RAGContextBuilder
from backend.rag.runtime import RAGRuntime
from backend.rag.indexer import DocumentIndexer
from backend.rag.search import VectorSearchService
from backend.storage.sqlite import ChatRepository


def create_document_indexer(
    *,
    settings: Settings,
    repository: ChatRepository,
) -> DocumentIndexer:
    embedding_provider = (
        create_embedding_provider(settings)
    )

    embedding_service = DocumentEmbeddingService(
        repository=repository,
        provider=embedding_provider,
        batch_size=settings.embedding_batch_size,
    )

    chunker = TextChunker(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )

    return DocumentIndexer(
        repository=repository,
        chunker=chunker,
        embedding_service=embedding_service,
        documents_dir=settings.documents_dir,
        max_document_size_mb=(
            settings.max_document_size_mb
        ),
    )


def create_vector_search_service(
    *,
    settings: Settings,
    repository: ChatRepository,
) -> VectorSearchService:
    embedding_provider = (
        create_embedding_provider(settings)
    )

    return VectorSearchService(
        repository=repository,
        embedding_provider=embedding_provider,
        top_k=settings.rag_top_k,
        min_similarity=settings.rag_min_similarity,
    )

def create_rag_runtime(
    *,
    settings: Settings,
    repository: ChatRepository,
) -> RAGRuntime:
    search_service = (
        create_vector_search_service(
            settings=settings,
            repository=repository,
        )
    )

    context_builder = RAGContextBuilder(
        max_context_chars=(
            settings.rag_max_context_chars
        )
    )

    return RAGRuntime(
        search_service=search_service,
        context_builder=context_builder,
        enabled=settings.rag_enabled,
    )