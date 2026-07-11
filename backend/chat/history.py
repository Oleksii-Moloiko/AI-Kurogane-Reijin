from typing import Literal

Role = Literal["system", "user", "assistant"]


def add_message(history: list, role: Role, content: str) -> None:
    history.append({"role": role, "content": content})