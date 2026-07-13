"""Kuro terminal application entry point."""
import sys
from backend.chat.session import ChatSession
from backend.commands.handler import handle_command
from backend.config import settings
from backend.llm.client import ask_llm_stream
from backend.ui.console import console
from backend.ui.layout import (
    print_banner,
    print_footer,
    print_header,
)
from backend.ui.panels import (
    print_error,
    print_response_statistics,
    print_system,
    print_user,
)
from backend.ui.prompt import print_prompt_hint, read_prompt
from backend.ui.stream import stream_assistant_reply
from backend.utils.logger import get_logger


logger = get_logger("backend.main")

def render_start_screen(model_name: str) -> None:
    """Render the initial terminal interface."""

    console.clear()

    print_banner()

    print_header(
        model_name=model_name,
        memory_enabled=settings.memory_enabled,
        workspace=settings.workspace_path,
    )

    print_footer()
    print_prompt_hint()


def main() -> int:
    """Run the Kuro terminal chat loop."""

    logger.info(
        "Kuro started: version=%s model=%s",
        settings.app_version,
        settings.ollama_model,
    )

    session = ChatSession(
        model=settings.ollama_model,
    )

    render_start_screen(session.model)

    while True:
        try:
            prompt_result = read_prompt()
        except (EOFError, KeyboardInterrupt):
            logger.info("Kuro stopped by user")
            print_system("Вихід із програми.")
            return 0

        user_input = prompt_result.text

        if not user_input:
            continue

        if user_input.casefold() in {"exit", "quit"}:
            logger.info(
                "Kuro stopped using command: %s",
                user_input.casefold(),
            )
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
            logger.exception(
                "Ollama request failed: model=%s",
                session.model,
            )
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

        logger.info(
            (
                "Ollama response completed: "
                "model=%s tokens=%s total_seconds=%.2f "
                "tokens_per_second=%.1f"
            ),
            response_stream.metrics.model,
            response_stream.metrics.eval_count,
            response_stream.metrics.total_seconds,
            response_stream.metrics.tokens_per_second,
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())