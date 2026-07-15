from pathlib import Path

from backend.storage import ChatRepository
from backend.rag.models import TextChunk

def test_repository_lists_latest_chats(tmp_path) -> None:
    repository = ChatRepository(tmp_path / "test.db")
    first = repository.create_chat("First", "ollama", "a")
    second = repository.create_chat("Second", "openai", "b")
    repository.add_message(first.id, "user", "hello")

    chats = repository.list_chats()

    assert {chat.id for chat in chats} == {first.id, second.id}
    assert repository.get_chat(first.id).message_count == 1


def test_message_count_excludes_system_prompt(tmp_path: Path) -> None:
    repository = ChatRepository(tmp_path / "test.db")
    chat = repository.create_chat("Test", "ollama", "model")
    repository.add_message(chat.id, "system", "system prompt")
    repository.add_message(chat.id, "user", "hello")
    repository.add_message(chat.id, "assistant", "hi")

    assert repository.get_chat(chat.id).message_count == 2
    assert repository.list_chats()[0].message_count == 2


def test_delete_chat_cascades_messages(tmp_path: Path) -> None:
    repository = ChatRepository(tmp_path / "test.db")
    chat = repository.create_chat("Test", "ollama", "model")
    repository.add_message(chat.id, "user", "hello")

    repository.delete_chat(chat.id)

    assert repository.get_chat(chat.id) is None
    assert repository.load_messages(chat.id) == []

def test_document_metadata_is_persisted(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    created = repository.create_document(
        filename="report.pdf",
        source_path="/tmp/report.pdf",
        extension=".pdf",
        size_bytes=1024,
        content_hash="a" * 64,
        page_count=5,
    )

    documents = repository.list_documents()
    restored = repository.get_document_by_hash(
        "a" * 64
    )

    assert len(documents) == 1
    assert restored is not None
    assert restored.id == created.id
    assert restored.filename == "report.pdf"
    assert restored.page_count == 5
    assert restored.status == "loaded"

def test_document_chunks_are_persisted(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    document = repository.create_document(
        filename="report.pdf",
        source_path="/tmp/report.pdf",
        extension=".pdf",
        size_bytes=1024,
        content_hash="b" * 64,
        page_count=2,
    )

    chunks = [
        TextChunk(
            index=0,
            content="First chunk",
            character_start=0,
            character_end=11,
            page_number=1,
        ),
        TextChunk(
            index=1,
            content="Second chunk",
            character_start=8,
            character_end=20,
            page_number=1,
        ),
    ]

    created = repository.create_document_chunks(
        document.id,
        chunks,
    )

    restored = repository.list_document_chunks(
        document.id
    )

    assert len(created) == 2
    assert len(restored) == 2
    assert restored[0].content == "First chunk"
    assert restored[1].chunk_index == 1
    assert restored[0].embedding is None

    assert repository.count_document_chunks(
        document.id
    ) == 2


def test_document_delete_cascades_to_chunks(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    document = repository.create_document(
        filename="report.pdf",
        source_path="/tmp/report.pdf",
        extension=".pdf",
        size_bytes=1024,
        content_hash="c" * 64,
        page_count=1,
    )

    repository.create_document_chunks(
        document.id,
        [
            TextChunk(
                index=0,
                content="Chunk",
                character_start=0,
                character_end=5,
            )
        ],
    )

    assert repository.count_document_chunks(
        document.id
    ) == 1

    deleted = repository.delete_document(
        document.id
    )

    assert deleted is True
    assert repository.count_document_chunks(
        document.id
    ) == 0

def test_chunk_embeddings_are_persisted(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    document = repository.create_document(
        filename="manual.pdf",
        source_path="/tmp/manual.pdf",
        extension=".pdf",
        size_bytes=200,
        content_hash="e" * 64,
        page_count=1,
    )

    repository.create_document_chunks(
        document.id,
        [
            TextChunk(
                index=0,
                content="Alpha",
                character_start=0,
                character_end=5,
            ),
            TextChunk(
                index=1,
                content="Beta",
                character_start=4,
                character_end=8,
            ),
        ],
    )

    repository.update_chunk_embeddings(
        document.id,
        [
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6],
        ],
    )



    repository.update_document_status(
        document.id,
        "indexed",
    )

    embedded = repository.list_embedded_chunks()

    
    assert len(embedded) == 2
    assert embedded[0].embedding == [0.1, 0.2, 0.3]
    assert embedded[1].embedding == [0.4, 0.5, 0.6]

def test_document_and_chunks_are_created_together(
    tmp_path: Path,
) -> None:
    repository = ChatRepository(
        tmp_path / "test.db"
    )

    document = (
        repository.create_document_with_chunks(
            filename="report.pdf",
            source_path="/tmp/report.pdf",
            extension=".pdf",
            size_bytes=100,
            content_hash="9" * 64,
            page_count=1,
            chunks=[
                TextChunk(
                    index=0,
                    content="First chunk",
                    character_start=0,
                    character_end=11,
                ),
                TextChunk(
                    index=1,
                    content="Second chunk",
                    character_start=8,
                    character_end=20,
                ),
            ],
        )
    )

    stored = repository.get_document(
        document.id
    )

    chunks = repository.list_document_chunks(
        document.id
    )

    assert stored is not None
    assert stored.status == "loaded"
    assert len(chunks) == 2