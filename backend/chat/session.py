"""Chat session state with SQLite persistence."""

from dataclasses import dataclass, field

from backend.prompts import load_system_prompt
from backend.storage import ChatRepository

Message = dict[str, str]


@dataclass
class ChatSession:
    model: str
    provider: str
    repository: ChatRepository
    max_history: int
    chat_id: str | None = None
    title: str = "Новий чат"
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.chat_id:
            self.restore(self.chat_id)
        else:
            record = self.repository.create_chat(self.title, self.provider, self.model)
            self.chat_id = record.id
            self.reset(persist=True)

    def reset(self, persist: bool = True, reset_title: bool = False) -> None:
        if persist and self.chat_id:
            self.repository.clear_messages(self.chat_id)
        self.messages = [{"role": "system", "content": load_system_prompt()}]
        if reset_title:
            self.rename("Новий чат")
        if persist and self.chat_id:
            self.repository.add_message(self.chat_id, "system", self.messages[0]["content"])

    def add_message(self, role: str, content: str) -> None:
        message = {"role": role, "content": content}
        self.messages.append(message)
        if self.chat_id:
            self.repository.add_message(self.chat_id, role, content)
        if role == "user" and self.title == "Новий чат":
            self.rename(self._title_from(content))

    def restore(self, chat_id: str) -> None:
        record = self.repository.get_chat(chat_id)
        if record is None:
            raise ValueError(f"Chat not found: {chat_id}")
        self.chat_id = record.id
        self.title = record.title
        self.provider = record.provider
        self.model = record.model
        self.messages = self.repository.load_messages(chat_id)
        if not self.messages:
            self.reset(persist=True)

    def rename(self, title: str) -> None:
        clean_title = " ".join(title.split()).strip() or "Новий чат"
        self.title = clean_title[:60]
        if self.chat_id:
            self.repository.update_chat(self.chat_id, title=self.title)

    def update_model(self, model: str) -> None:
        self.model = model
        if self.chat_id:
            self.repository.update_chat(self.chat_id, model=model)

    def discard_if_empty(self) -> bool:
        """Remove a never-used chat so the session list stays clean."""
        if not self.is_empty or not self.chat_id:
            return False
        self.repository.delete_chat(self.chat_id)
        return True

    @property
    def context_messages(self) -> list[Message]:
        system = [message for message in self.messages if message.get("role") == "system"][:1]
        visible = [message for message in self.messages if message.get("role") != "system"]
        if self.max_history <= 0:
            return system + visible
        return system + visible[-self.max_history:]

    @property
    def visible_messages(self) -> list[Message]:
        return [message for message in self.messages if message.get("role") != "system"]

    @property
    def is_empty(self) -> bool:
        return not self.visible_messages

    @staticmethod
    def _title_from(content: str) -> str:
        compact = " ".join(content.split())
        return compact[:57] + "..." if len(compact) > 60 else compact
