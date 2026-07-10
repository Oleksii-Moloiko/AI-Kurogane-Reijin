import sys
import shutil
from llm.client import ask_llm
from chat.history import add_user_message, add_assistant_message
from config import MODEL_NAME


def main():
    print(f"Чат з моделлю '{MODEL_NAME}' через Ollama.")
    print("Введіть 'exit' або 'quit', щоб вийти.\n")

    chat_history = [
        {"role": "system", "content": "Ти помічник, який відповідає коротко та лаконічно."}
    ]
    
    while True:
        try:
            user_input = input("Ти: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВихід із програми.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Вихід із програми.")
            break

        add_user_message(chat_history, user_input)

        try:
            reply = ask_llm(chat_history, model=MODEL_NAME)
        except Exception as e:
            print(f"Помилка при зверненні до Ollama: {e}")
            continue

        add_assistant_message(chat_history, reply)

        columns = shutil.get_terminal_size().columns

        print("-" * columns)
        print(f"Модель: {reply}\n")
        print("-" * columns)


if __name__ == "__main__":
    sys.exit(main() or 0)