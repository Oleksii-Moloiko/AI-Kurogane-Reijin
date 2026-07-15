from pathlib import Path

from backend.chat.session import ChatSession
from backend.storage import ChatRepository


def make_session(tmp_path: Path, max_history: int = 4) -> ChatSession:
    return ChatSession(
        model="test-model",
        provider="ollama",
        repository=ChatRepository(tmp_path / "test.db"),
        max_history=max_history,
    )


def test_session_is_persisted_and_restored(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.add_message("user", "Hello persistent world")
    session.add_message("assistant", "Hello")

    restored = ChatSession(
        model="ignored",
        provider="ignored",
        repository=session.repository,
        max_history=4,
        chat_id=session.chat_id,
    )

    assert restored.title == "Hello persistent world"
    assert restored.model == "test-model"
    assert restored.visible_messages[-1]["content"] == "Hello"


def test_max_history_limits_context_not_database(tmp_path: Path) -> None:
    session = make_session(tmp_path, max_history=2)
    for number in range(4):
        session.add_message("user", f"question {number}")

    assert len(session.context_messages) == 3  # system + two latest messages
    assert len(session.repository.load_messages(session.chat_id)) == 5


def test_clear_preserves_system_prompt(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.add_message("user", "temporary")
    session.reset()

    assert session.is_empty
    stored = session.repository.load_messages(session.chat_id)
    assert len(stored) == 1
    assert stored[0]["role"] == "system"


def test_clear_resets_generated_title(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.add_message("user", "A generated chat title")

    session.reset(reset_title=True)

    assert session.title == "Новий чат"
    assert session.repository.get_chat(session.chat_id).title == "Новий чат"


def test_discard_if_empty_removes_unused_chat(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    chat_id = session.chat_id

    assert session.discard_if_empty() is True
    assert session.repository.get_chat(chat_id) is None


def test_discard_if_empty_keeps_used_chat(tmp_path: Path) -> None:
    session = make_session(tmp_path)
    session.add_message("user", "keep me")

    assert session.discard_if_empty() is False
    assert session.repository.get_chat(session.chat_id) is not None
