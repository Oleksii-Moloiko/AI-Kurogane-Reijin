
import sys
from llm.client import ask_llm_stream
from chat.history import add_message
from prompts import load_system_prompt
from config import MODEL_NAME
from ui.console import (
    console,
    print_user,
    print_system,
    print_error,
    stream_assistant_reply,
)


def main():
    print_system(f"Чат з моделлю '{MODEL_NAME}' через Ollama.")
    print_system("Введіть 'exit' або 'quit', щоб вийти.")

    chat_history = [
        {"role": "system", "content": load_system_prompt()}
    ]

    while True:
        try:
            user_input = console.input("[dim]> [/]").strip()
        except (EOFError, KeyboardInterrupt):
            print_system("Вихід із програми.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print_system("Вихід із програми.")
            break

        print_user(user_input)
        add_message(chat_history, "user", user_input)

        try:
            reply = stream_assistant_reply(
                ask_llm_stream(chat_history, model=MODEL_NAME),
                title=MODEL_NAME,
            )
        except Exception as e:
            print_error(f"Помилка при зверненні до Ollama: {e}")
            continue

        add_message(chat_history, "assistant", reply)

if __name__ == "__main__":
    sys.exit(main() or 0)