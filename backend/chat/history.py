def add_user_message(history: list, content: str) -> None:
    history.append({"role": "user", "content": content})

def add_assistant_message(history: list, content: str) -> None:
    history.append({"role": "assistant", "content": content})