"""Document indexing orchestration."""

import shutil
from pathlib import Path
from uuid import uuid4

from backend.rag.chunking import TextChunker
from backend.rag.embedding_service import (
    DocumentEmbeddingService,
)
from backend.rag.errors import (
    DocumentIndexingError,
    DuplicateDocumentError,
)
from backend.rag.hashing import calculate_file_hash
from backend.rag.loaders import load_document
from backend.rag.models import IndexedDocument
from backend.storage.sqlite import ChatRepository


class DocumentIndexer:
    """Run the full local document indexing pipeline."""

    def __init__(
        self,
        *,
        repository: ChatRepository,
        chunker: TextChunker,
        embedding_service: DocumentEmbeddingService,
        documents_dir: Path,
        max_document_size_mb: int = 20,
    ) -> None:
        self.repository = repository
        self.chunker = chunker
        self.embedding_service = embedding_service
        self.documents_dir = documents_dir
        self.max_document_size_mb = max_document_size_mb

    def index(
        self,
        document_path: str | Path,
    ) -> IndexedDocument:
        source_path = Path(
            document_path
        ).expanduser().resolve()

        loaded_document = load_document(
            source_path,
            max_size_mb=self.max_document_size_mb,
        )

        content_hash = calculate_file_hash(
            loaded_document.path
        )

        existing_document = (
            self.repository.get_document_by_hash(
                content_hash
            )
        )

        if existing_document is not None:
            raise DuplicateDocumentError(
                f"Документ '{existing_document.filename}' "
                f"уже доданий. ID: {existing_document.id}"
            )

        chunks = self.chunker.split(
            loaded_document.content
        )

        if not chunks:
            raise DocumentIndexingError(
                "Документ не містить тексту для індексації."
            )

        

        document_id = uuid4().hex[:12]
        document_dir = (
            self.documents_dir / document_id
        )
        stored_path = (
            document_dir / loaded_document.filename
        )

        metadata_created = False

        try:
            self.embedding_service.provider.healthcheck()

            document_dir.mkdir(
                parents=True,
                exist_ok=False,
            )

            shutil.copy2(
                loaded_document.path,
                stored_path,
            )

            document = (
                self.repository.create_document_with_chunks(
                    document_id=document_id,
                    filename=loaded_document.filename,
                    source_path=str(stored_path),
                    extension=loaded_document.extension,
                    size_bytes=loaded_document.size_bytes,
                    content_hash=content_hash,
                    page_count=loaded_document.page_count,
                    chunks=chunks,
                )
            )

            metadata_created = True

            embedded_count = (
                self.embedding_service.embed_document(
                    document.id
                )
            )

            if embedded_count != len(chunks):
                raise DocumentIndexingError(
                    "Не всі chunks отримали embeddings."
                )

            stored_embedding_count = (
                self.repository.count_embedded_document_chunks(
                    document.id
                )
            )

            if stored_embedding_count != len(chunks):
                raise DocumentIndexingError(
                    "Не всі embeddings були збережені."
                )

            self.repository.update_document_status(
                document.id,
                "indexed",
            )

            return IndexedDocument(
                document_id=document.id,
                filename=document.filename,
                stored_path=stored_path,
                chunk_count=len(chunks),
                embedding_model=(
                    self.embedding_service.provider.model_name
                ),
                size_bytes=document.size_bytes,
                page_count=document.page_count,
            )

        except DuplicateDocumentError:
            raise

        except Exception as error:
            if metadata_created:
                self.repository.delete_document(
                    document_id
                )

            if document_dir.exists():
                shutil.rmtree(
                    document_dir,
                    ignore_errors=True,
                )

            if isinstance(
                error,
                DocumentIndexingError,
            ):
                raise

            raise DocumentIndexingError(
                f"Не вдалося проіндексувати "
                f"'{source_path.name}': {error}"
            ) from error