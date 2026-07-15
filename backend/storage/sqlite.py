"""SQLite persistence for chats and messages."""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from backend.rag.models import TextChunk

@dataclass(frozen=True, slots=True)
class ChatRecord:
    id: str
    title: str
    provider: str
    model: str
    created_at: str
    updated_at: str
    message_count: int = 0

@dataclass(frozen=True, slots=True)
class DocumentRecord:
    id: str
    filename: str
    source_path: str
    extension: str
    size_bytes: int
    content_hash: str
    page_count: int | None
    status: str
    created_at: str
    updated_at: str

@dataclass(frozen=True, slots=True)
class DocumentChunkRecord:
    id: int
    document_id: str
    chunk_index: int
    content: str
    character_start: int
    character_end: int
    page_number: int | None
    embedding: str | None
    created_at: str

@dataclass(frozen=True, slots=True)
class EmbeddedChunkRecord:
    chunk: DocumentChunkRecord
    filename: str
    embedding: list[float]

class ChatRepository:
    """Small synchronous repository suitable for the local CLI MVP."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.database_path)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS chats (
                    id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    provider TEXT NOT NULL,
                    model TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    chat_id TEXT NOT NULL,
                    role TEXT NOT NULL CHECK(role IN ('system', 'user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(chat_id) REFERENCES chats(id) ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_messages_chat_id_id
                ON messages(chat_id, id);

                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    source_path TEXT NOT NULL,
                    extension TEXT NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    content_hash TEXT NOT NULL UNIQUE,
                    page_count INTEGER,
                    status TEXT NOT NULL DEFAULT 'loaded'
                        CHECK(status IN ('loaded', 'indexed', 'failed')),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE INDEX IF NOT EXISTS idx_documents_updated_at
                ON documents(updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_documents_filename
                ON documents(filename);

                CREATE TABLE IF NOT EXISTS document_chunks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL,
                    content TEXT NOT NULL,
                    character_start INTEGER NOT NULL,
                    character_end INTEGER NOT NULL,
                    page_number INTEGER,
                    embedding TEXT,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(document_id)
                        REFERENCES documents(id)
                        ON DELETE CASCADE,
                    UNIQUE(document_id, chunk_index)
                );

                CREATE INDEX IF NOT EXISTS idx_document_chunks_document_id
                ON document_chunks(document_id, chunk_index);

                CREATE INDEX IF NOT EXISTS idx_document_chunks_embedding
                ON document_chunks(document_id)
                WHERE embedding IS NOT NULL;
                """
            )

    @staticmethod
    def _now() -> str:
        return datetime.now(UTC).isoformat()

    def create_chat(self, title: str, provider: str, model: str, chat_id: str | None = None) -> ChatRecord:
        identifier = chat_id or uuid4().hex[:12]
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO chats(id, title, provider, model, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (identifier, title, provider, model, now, now),
            )
        return ChatRecord(identifier, title, provider, model, now, now, 0)

    def get_chat(self, chat_id: str) -> ChatRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT c.*, COUNT(CASE WHEN m.role != 'system' THEN 1 END) AS message_count
                FROM chats c LEFT JOIN messages m ON m.chat_id = c.id
                WHERE c.id = ? GROUP BY c.id
                """,
                (chat_id,),
            ).fetchone()
        return self._record(row) if row else None

    def list_chats(self, limit: int = 20) -> list[ChatRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT c.*, COUNT(CASE WHEN m.role != 'system' THEN 1 END) AS message_count
                FROM chats c LEFT JOIN messages m ON m.chat_id = c.id
                GROUP BY c.id ORDER BY c.updated_at DESC LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return [self._record(row) for row in rows]

    def add_message(self, chat_id: str, role: str, content: str) -> None:
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO messages(chat_id, role, content, created_at) VALUES (?, ?, ?, ?)",
                (chat_id, role, content, now),
            )
            connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (now, chat_id))

    def load_messages(self, chat_id: str) -> list[dict[str, str]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT role, content FROM messages WHERE chat_id = ? ORDER BY id",
                (chat_id,),
            ).fetchall()
        return [{"role": row["role"], "content": row["content"]} for row in rows]

    def clear_messages(self, chat_id: str) -> None:
        with self._connect() as connection:
            connection.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))
            connection.execute("UPDATE chats SET updated_at = ? WHERE id = ?", (self._now(), chat_id))


    def delete_chat(self, chat_id: str) -> None:
        """Delete a chat and its messages."""
        with self._connect() as connection:
            connection.execute("DELETE FROM chats WHERE id = ?", (chat_id,))

    def update_chat(self, chat_id: str, *, title: str | None = None, provider: str | None = None, model: str | None = None) -> None:
        fields: list[str] = []
        values: list[str] = []
        for key, value in (("title", title), ("provider", provider), ("model", model)):
            if value is not None:
                fields.append(f"{key} = ?")
                values.append(value)
        if not fields:
            return
        fields.append("updated_at = ?")
        values.append(self._now())
        values.append(chat_id)
        with self._connect() as connection:
            connection.execute(f"UPDATE chats SET {', '.join(fields)} WHERE id = ?", values)
    
    def get_document(
        self,
        document_id: str,
    ) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE id = ?
                """,
                (document_id,),
            ).fetchone()

        return self._document_record(row) if row else None

    def create_document(
        self,
        *,
        filename: str,
        source_path: str,
        extension: str,
        size_bytes: int,
        content_hash: str,
        page_count: int | None,
        document_id: str | None = None,
    ) -> DocumentRecord:
        identifier = document_id or uuid4().hex[:12]
        now = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    id,
                    filename,
                    source_path,
                    extension,
                    size_bytes,
                    content_hash,
                    page_count,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'loaded', ?, ?)
                """,
                (
                    identifier,
                    filename,
                    source_path,
                    extension,
                    size_bytes,
                    content_hash,
                    page_count,
                    now,
                    now,
                ),
            )

        return DocumentRecord(
            id=identifier,
            filename=filename,
            source_path=source_path,
            extension=extension,
            size_bytes=size_bytes,
            content_hash=content_hash,
            page_count=page_count,
            status="loaded",
            created_at=now,
            updated_at=now,
        )


    def get_document_by_hash(
        self,
        content_hash: str,
    ) -> DocumentRecord | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT *
                FROM documents
                WHERE content_hash = ?
                """,
                (content_hash,),
            ).fetchone()

        return self._document_record(row) if row else None


    def list_documents(
        self,
        limit: int = 100,
    ) -> list[DocumentRecord]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM documents
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            self._document_record(row)
            for row in rows
        ]


    def delete_document(
        self,
        document_id: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE id = ?
                """,
                (document_id,),
            )

        return cursor.rowcount > 0


    def update_document_status(
        self,
        document_id: str,
        status: str,
    ) -> None:
        allowed_statuses = {
            "loaded",
            "indexed",
            "failed",
        }

        if status not in allowed_statuses:
            raise ValueError(
                f"Unsupported document status: {status}"
            )

        with self._connect() as connection:
            connection.execute(
                """
                UPDATE documents
                SET status = ?, updated_at = ?
                WHERE id = ?
                """,
                (
                    status,
                    self._now(),
                    document_id,
                ),
            )


    def create_document_chunks(
        self,
        document_id: str,
        chunks: list[TextChunk],
    ) -> list[DocumentChunkRecord]:
        """Persist all chunks for a document in one transaction."""

        if not chunks:
            return []

        now = self._now()

        rows = [
            (
                document_id,
                chunk.index,
                chunk.content,
                chunk.character_start,
                chunk.character_end,
                chunk.page_number,
                now,
            )
            for chunk in chunks
        ]

        with self._connect() as connection:
            connection.executemany(
                """
                INSERT INTO document_chunks(
                    document_id,
                    chunk_index,
                    content,
                    character_start,
                    character_end,
                    page_number,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )

            stored_rows = connection.execute(
                """
                SELECT *
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        return [
            self._document_chunk_record(row)
            for row in stored_rows
        ]


    def list_document_chunks(
        self,
        document_id: str,
    ) -> list[DocumentChunkRecord]:
        """Return chunks in their original document order."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT *
                FROM document_chunks
                WHERE document_id = ?
                ORDER BY chunk_index
                """,
                (document_id,),
            ).fetchall()

        return [
            self._document_chunk_record(row)
            for row in rows
        ]


    def count_document_chunks(
        self,
        document_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS chunk_count
                FROM document_chunks
                WHERE document_id = ?
                """,
                (document_id,),
            ).fetchone()

        return int(row["chunk_count"])


    def delete_document_chunks(
        self,
        document_id: str,
    ) -> int:
        """Delete chunks while keeping document metadata."""

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM document_chunks
                WHERE document_id = ?
                """,
                (document_id,),
            )

        return cursor.rowcount

    def update_chunk_embeddings(
        self,
        document_id: str,
        embeddings: list[list[float]],
    ) -> None:
        """Assign embeddings to document chunks by chunk order."""

        chunks = self.list_document_chunks(document_id)

        if len(chunks) != len(embeddings):
            raise ValueError(
                "Embedding count does not match chunk count."
            )

        if not chunks:
            return

        dimensions = {
            len(embedding)
            for embedding in embeddings
        }

        if len(dimensions) != 1:
            raise ValueError(
                "All embeddings must have the same dimension."
            )

        if 0 in dimensions:
            raise ValueError(
                "Embeddings cannot be empty."
            )

        rows = [
            (
                json.dumps(
                    embedding,
                    separators=(",", ":"),
                ),
                chunk.id,
                document_id,
            )
            for chunk, embedding in zip(
                chunks,
                embeddings,
                strict=True,
            )
        ]

        with self._connect() as connection:
            connection.executemany(
                """
                UPDATE document_chunks
                SET embedding = ?
                WHERE id = ? AND document_id = ?
                """,
                rows,
            )

    def list_embedded_chunks(
    self,
) -> list[EmbeddedChunkRecord]:
        """Return indexed chunks together with document metadata."""

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    document_chunks.*,
                    documents.filename
                FROM document_chunks
                JOIN documents
                    ON documents.id = document_chunks.document_id
                WHERE document_chunks.embedding IS NOT NULL
                AND documents.status = 'indexed'
                ORDER BY
                    document_chunks.document_id,
                    document_chunks.chunk_index
                """
            ).fetchall()

        result: list[EmbeddedChunkRecord] = []

        for row in rows:
            chunk = self._document_chunk_record(row)

            try:
                vector = json.loads(row["embedding"])
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"Invalid embedding JSON for chunk {chunk.id}."
                ) from error

            if not isinstance(vector, list) or not vector:
                raise ValueError(
                    f"Invalid embedding for chunk {chunk.id}."
                )

            if not all(
                isinstance(value, int | float)
                for value in vector
            ):
                raise ValueError(
                    f"Embedding for chunk {chunk.id} "
                    "contains non-numeric values."
                )

            result.append(
                EmbeddedChunkRecord(
                    chunk=chunk,
                    filename=row["filename"],
                    embedding=[
                        float(value)
                        for value in vector
                    ],
                )
            )

        return result
    
    def delete_document_by_hash(
        self,
        content_hash: str,
    ) -> bool:
        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM documents
                WHERE content_hash = ?
                """,
                (content_hash,),
            )

        return cursor.rowcount > 0

    def count_embedded_document_chunks(
        self,
        document_id: str,
    ) -> int:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT COUNT(*) AS chunk_count
                FROM document_chunks
                WHERE document_id = ?
                AND embedding IS NOT NULL
                """,
                (document_id,),
            ).fetchone()

        return int(row["chunk_count"])

    def create_document_with_chunks(
        self,
        *,
        filename: str,
        source_path: str,
        extension: str,
        size_bytes: int,
        content_hash: str,
        page_count: int | None,
        chunks: list[TextChunk],
        document_id: str | None = None,
    ) -> DocumentRecord:
        """Create document metadata and chunks in one transaction."""

        if not chunks:
            raise ValueError(
                "Cannot create a document without chunks."
            )

        identifier = document_id or uuid4().hex[:12]
        now = self._now()

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO documents(
                    id,
                    filename,
                    source_path,
                    extension,
                    size_bytes,
                    content_hash,
                    page_count,
                    status,
                    created_at,
                    updated_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, 'loaded', ?, ?)
                """,
                (
                    identifier,
                    filename,
                    source_path,
                    extension,
                    size_bytes,
                    content_hash,
                    page_count,
                    now,
                    now,
                ),
            )

            connection.executemany(
                """
                INSERT INTO document_chunks(
                    document_id,
                    chunk_index,
                    content,
                    character_start,
                    character_end,
                    page_number,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        identifier,
                        chunk.index,
                        chunk.content,
                        chunk.character_start,
                        chunk.character_end,
                        chunk.page_number,
                        now,
                    )
                    for chunk in chunks
                ],
            )

        return DocumentRecord(
            id=identifier,
            filename=filename,
            source_path=source_path,
            extension=extension,
            size_bytes=size_bytes,
            content_hash=content_hash,
            page_count=page_count,
            status="loaded",
            created_at=now,
            updated_at=now,
        )

    @staticmethod
    def _document_chunk_record(
        row: sqlite3.Row,
    ) -> DocumentChunkRecord:
        return DocumentChunkRecord(
            id=row["id"],
            document_id=row["document_id"],
            chunk_index=row["chunk_index"],
            content=row["content"],
            character_start=row["character_start"],
            character_end=row["character_end"],
            page_number=row["page_number"],
            embedding=row["embedding"],
            created_at=row["created_at"],
        )

    @staticmethod
    def _document_record(
        row: sqlite3.Row,
    ) -> DocumentRecord:
        return DocumentRecord(
            id=row["id"],
            filename=row["filename"],
            source_path=row["source_path"],
            extension=row["extension"],
            size_bytes=row["size_bytes"],
            content_hash=row["content_hash"],
            page_count=row["page_count"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> ChatRecord:
        return ChatRecord(
            id=row["id"], title=row["title"], provider=row["provider"], model=row["model"],
            created_at=row["created_at"], updated_at=row["updated_at"], message_count=row["message_count"],
        )

