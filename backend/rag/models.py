"""Data models used by the RAG pipeline."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class LoadedDocument:
    """Text extracted from a local document."""

    path: Path
    filename: str
    extension: str
    content: str
    size_bytes: int
    page_count: int | None = None

    @property
    def character_count(self) -> int:
        return len(self.content)
    

@dataclass(frozen=True, slots=True)
class TextChunk:
    """A fragment of text prepared for embedding."""

    index: int
    content: str
    character_start: int
    character_end: int
    page_number: int | None = None

    @property
    def character_count(self) -> int:
        return len(self.content)
    
@dataclass(frozen=True, slots=True)
class SearchResult:
    """A document chunk returned by vector search."""

    chunk_id: int
    document_id: str
    filename: str
    chunk_index: int
    content: str
    similarity: float
    page_number: int | None = None

@dataclass(frozen=True, slots=True)
class RAGSource:
    """A source included in the generated RAG context."""

    number: int
    document_id: str
    filename: str
    chunk_index: int
    content: str
    similarity: float
    page_number: int | None = None

    @property
    def label(self) -> str:
        if self.page_number is not None:
            return (
                f"[{self.number}] {self.filename}, "
                f"сторінка {self.page_number}"
            )

        return (
            f"[{self.number}] {self.filename}, "
            f"фрагмент {self.chunk_index + 1}"
        )


@dataclass(frozen=True, slots=True)
class RAGContext:
    """Context prepared for insertion into an LLM request."""

    text: str
    sources: tuple[RAGSource, ...]

    @property
    def is_empty(self) -> bool:
        return not self.sources


@dataclass(frozen=True, slots=True)
class IndexedDocument:
    """Result of a successful document indexing operation."""

    document_id: str
    filename: str
    stored_path: Path
    chunk_count: int
    embedding_model: str
    size_bytes: int
    page_count: int | None = None