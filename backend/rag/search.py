"""Vector search over locally stored document chunks."""

from backend.rag.embeddings.base import (
    EmbeddingProvider,
)
from backend.rag.models import SearchResult
from backend.rag.similarity import cosine_similarity
from backend.storage.sqlite import ChatRepository


class VectorSearchService:
    """Find document chunks relevant to a text query."""

    def __init__(
        self,
        *,
        repository: ChatRepository,
        embedding_provider: EmbeddingProvider,
        top_k: int = 5,
        min_similarity: float = 0.35,
    ) -> None:
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not -1.0 <= min_similarity <= 1.0:
            raise ValueError(
                "min_similarity must be between -1 and 1."
            )

        self.repository = repository
        self.embedding_provider = embedding_provider
        self.top_k = top_k
        self.min_similarity = min_similarity

    def search(
        self,
        query: str,
        *,
        top_k: int | None = None,
        min_similarity: float | None = None,
    ) -> list[SearchResult]:
        normalized_query = query.strip()

        if not normalized_query:
            return []

        resolved_top_k = (
            top_k
            if top_k is not None
            else self.top_k
        )

        resolved_threshold = (
            min_similarity
            if min_similarity is not None
            else self.min_similarity
        )

        if resolved_top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero."
            )

        if not -1.0 <= resolved_threshold <= 1.0:
            raise ValueError(
                "min_similarity must be between -1 and 1."
            )

        query_embedding = (
            self.embedding_provider.embed_query(
                normalized_query
            )
        )

        embedded_chunks = (
            self.repository.list_embedded_chunks()
        )

        scored_results: list[SearchResult] = []

        for item in embedded_chunks:
            similarity = cosine_similarity(
                query_embedding,
                item.embedding,
            )

            if similarity < resolved_threshold:
                continue

            chunk = item.chunk

            scored_results.append(
                SearchResult(
                    chunk_id=chunk.id,
                    document_id=chunk.document_id,
                    filename=item.filename,
                    chunk_index=chunk.chunk_index,
                    content=chunk.content,
                    similarity=similarity,
                    page_number=chunk.page_number,
                )
            )

        scored_results.sort(
            key=lambda item: (
                -item.similarity,
                item.document_id,
                item.chunk_index,
            )
        )

        return scored_results[:resolved_top_k]