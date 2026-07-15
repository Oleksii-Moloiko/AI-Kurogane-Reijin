"""Kuro terminal application entry point."""

import sys
from pathlib import Path

# Support both `python backend/main.py` and `python -m backend.main`.
if __package__ in {None, ""}:
    sys.path.insert(
        0,
        str(Path(__file__).resolve().parents[1]),
    )

from backend.chat.session import ChatSession
from backend.commands.handler import handle_command
from backend.config import settings
from backend.llm.errors import LLMError
from backend.llm.factory import create_provider
from backend.rag.factory import (
    create_document_indexer,
    create_rag_runtime,
)
from backend.rag.messages import inject_rag_context
from backend.storage import ChatRepository
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
from backend.ui.prompt import (
    print_prompt_hint,
    read_prompt,
)
from backend.ui.sources import print_rag_sources
from backend.ui.stream import stream_assistant_reply
from backend.utils.logger import get_logger


logger = get_logger("backend.main")


def render_start_screen(
    session: ChatSession,
) -> None:
    """Render the initial terminal screen."""

    console.clear()
    print_banner()

    print_header(
        model_name=session.model,
        memory_enabled=settings.memory_enabled,
        workspace=settings.workspace_path,
        provider_name=session.provider,
        chat_id=session.chat_id,
        chat_title=session.title,
    )

    print_footer()
    print_prompt_hint()


def main() -> int:
    """Run the Kuro terminal application."""

    logger.info(
        "Kuro started: version=%s provider=%s model=%s",
        settings.app_version,
        settings.llm_provider,
        settings.llm_model,
    )

    repository = ChatRepository(
        settings.database_path
    )

    document_indexer = create_document_indexer(
        settings=settings,
        repository=repository,
    )

    rag_runtime = create_rag_runtime(
        settings=settings,
        repository=repository,
    )

    provider = create_provider()

    session = ChatSession(
        model=settings.llm_model,
        provider=settings.llm_provider,
        repository=repository,
        max_history=settings.max_history,
    )

    render_start_screen(session)

    try:
        provider.healthcheck(
            session.model
        )
    except Exception as error:
        logger.warning(
            "Initial provider healthcheck failed: %s",
            error,
        )

        print_error(str(error))
        print_system(
            "Команди та історія доступні, але чат "
            "не працюватиме до відновлення provider-а."
        )

    while True:
        try:
            prompt_result = read_prompt()
        except (EOFError, KeyboardInterrupt):
            session.discard_if_empty()
            print_system("Вихід із програми.")
            return 0

        user_input = prompt_result.text.strip()

        if not user_input:
            continue

        if user_input.casefold() in {
            "exit",
            "quit",
        }:
            session.discard_if_empty()
            print_system("Вихід із програми.")
            return 0

        if prompt_result.is_unknown_command:
            print_error(
                f"Невідома команда: {prompt_result.text}"
            )
            continue

        if prompt_result.is_command:
            previous_provider = session.provider

            session = handle_command(
                command=prompt_result.command,
                session=session,
                provider=provider,
                repository=repository,
                arguments=prompt_result.arguments,
                document_indexer=document_indexer,
            )

            if session.provider != previous_provider:
                provider = create_provider(
                    session.provider
                )

                try:
                    provider.healthcheck(
                        session.model
                    )
                except Exception as error:
                    logger.warning(
                        "Provider healthcheck failed "
                        "after provider change: %s",
                        error,
                    )
                    print_error(str(error))

            continue

        print_user(user_input)

        # Зберігаємо лише справжнє повідомлення користувача.
        session.add_message(
            role="user",
            content=user_input,
        )

        # Виконуємо retrieval після додавання user message,
        # але RAG context не записуємо в ChatSession або SQLite.
        rag_context = rag_runtime.retrieve(
            user_input
        )

        llm_messages = inject_rag_context(
            session.context_messages,
            rag_context,
        )

        try:
            response_stream = provider.stream_chat(
                llm_messages,
                model=session.model,
            )

            reply = stream_assistant_reply(
                response_stream,
                title=session.model,
            )

        except LLMError as error:
            logger.exception(
                "LLM request failed"
            )
            print_error(str(error))
            continue

        except Exception as error:
            logger.exception(
                "Unexpected LLM request failure"
            )
            print_error(
                "Неочікувана помилка provider-а: "
                f"{error}"
            )
            continue

        if not reply.strip():
            logger.warning(
                "Provider returned an empty response"
            )
            print_error(
                "Provider повернув порожню відповідь."
            )
            continue

        session.add_message(
            role="assistant",
            content=reply,
        )

        print_response_statistics(
            response_stream.metrics
        )

        print_rag_sources(
            rag_context
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())