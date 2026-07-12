"""Kuro terminal application entry point."""

import sys

from chat.session import ChatSession
from commands.handler import handle_command
from config import MEMORY_ENABLED, MODEL_NAME, WORKSPACE_PATH
from llm.client import ask_llm_stream
from ui.console import console
from ui.layout import print_banner, print_footer, print_header
from ui.panels import (
    print_error,
    print_response_statistics,
    print_system,
    print_user,
)
from ui.prompt import print_prompt_hint, read_prompt
from ui.stream import stream_assistant_reply


def render_start_screen(model_name: str) -> None:
    """Render the initial terminal interface."""

    console.clear()

    print_banner()

    print_header(
        model_name=model_name,
        memory_enabled=MEMORY_ENABLED,
        workspace=WORKSPACE_PATH,
    )

    print_footer()
    print_prompt_hint()


def main() -> int:
    """Run the Kuro terminal chat loop."""

    session = ChatSession(
        model=MODEL_NAME,
    )

    render_start_screen(session.model)

    while True:
        try:
            prompt_result = read_prompt()
        except (EOFError, KeyboardInterrupt):
            print_system("Вихід із програми.")
            return 0

        user_input = prompt_result.text

        if not user_input:
            continue

        if user_input.casefold() in {"exit", "quit"}:
            print_system("Вихід із програми.")
            return 0

        if prompt_result.is_command:
            handle_command(
                command=prompt_result.command,
                session=session,
            )
            continue

        print_user(user_input)

        session.add_message(
            role="user",
            content=user_input,
        )

        try:
            response_stream = ask_llm_stream(
                session.messages,
                model=session.model,
            )

            reply = stream_assistant_reply(
                response_stream,
                title=session.model,
            )

        except Exception as error:
            print_error(
                f"Помилка при зверненні до Ollama: {error}"
            )
            continue

        session.add_message(
            role="assistant",
            content=reply,
        )

        print_response_statistics(
            response_stream.metrics
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())