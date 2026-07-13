"""Chat session state."""

from dataclasses import dataclass, field

from backend.prompts import load_system_prompt


Message = dict[str, str]


@dataclass
class ChatSession:
    """State of the current Kuro chat session."""

    model: str
    messages: list[Message] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Ensure the session always starts with a system prompt."""

        if not self.messages:
            self.reset()

    def reset(self) -> None:
        """Clear chat history while preserving the system prompt."""

        self.messages = [
            {
                "role": "system",
                "content": load_system_prompt(),
            }
        ]

    def add_message(
        self,
        role: str,
        content: str,
    ) -> None:
        """Add a message to the current session."""

        self.messages.append(
            {
                "role": role,
                "content": content,
            }
        )

    @property
    def visible_messages(self) -> list[Message]:
        """Return messages visible to the user."""

        return [
            message
            for message in self.messages
            if message.get("role") != "system"
        ]

    @property
    def is_empty(self) -> bool:
        """Return whether the visible chat history is empty."""

        return not self.visible_messages