"""Generate and persist embeddings for document chunks."""

from backend.rag.embeddings.base import EmbeddingProvider
from backend.storage.sqlite import ChatRepository


class DocumentEmbeddingService:
    """Generate embeddings in batches and persist them."""

    def __init__(
        self,
        *,
        repository: ChatRepository,
        provider: EmbeddingProvider,
        batch_size: int = 16,
    ) -> None:
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be greater than zero."
            )

        self.repository = repository
        self.provider = provider
        self.batch_size = batch_size

    def embed_document(
        self,
        document_id: str,
    ) -> int:
        chunks = self.repository.list_document_chunks(
            document_id
        )

        if not chunks:
            raise ValueError(
                f"Document '{document_id}' has no chunks."
            )

        all_embeddings: list[list[float]] = []

        for start in range(
            0,
            len(chunks),
            self.batch_size,
        ):
            batch = chunks[
                start : start + self.batch_size
            ]

            texts = [
                chunk.content
                for chunk in batch
            ]

            embeddings = self.provider.embed_texts(texts)

            if len(embeddings) != len(batch):
                raise ValueError(
                    "Embedding provider returned an "
                    "unexpected vector count."
                )

            all_embeddings.extend(embeddings)

        self.repository.update_chunk_embeddings(
            document_id,
            all_embeddings,
        )

        return len(all_embeddings)