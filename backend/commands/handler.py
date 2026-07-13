"""Terminal command handlers."""

from backend.chat.session import ChatSession
from backend.config import settings
from backend.llm.client import get_available_models
from backend.ui.console import console
from backend.ui.layout import (
    print_banner,
    print_footer,
    print_header,
)
from backend.ui.panels import (
    print_error,
    print_help,
    print_history,
    print_model_list,
    print_system,
)
from backend.ui.prompt import Command, read_model_choice
from backend.utils.logger import get_logger


logger = get_logger(__name__)


def select_model(current_model: str) -> str:
    """Allow the user to select a locally installed Ollama model."""

    try:
        models = get_available_models()
    except Exception as error:
        
        logger.exception(
            "Failed to retrieve Ollama models"
        )

        print_error(
            f"Не вдалося отримати список моделей Ollama: {error}"
        )
        return current_model

    if not models:
        logger.warning("No local Ollama models found")

        print_error(
            "Локальні моделі Ollama не знайдено."
        )
        return current_model
    
    if selected_model is None:
        logger.warning(
            "Invalid model selection: choice=%s",
            choice,
        )

        print_error(
            f"Не вдалося знайти модель за значенням: {choice}"
        )
        return current_model

    print_model_list(
        models=models,
        current_model=current_model,
    )

    try:
        choice = read_model_choice()
    except (EOFError, KeyboardInterrupt):
        print_system("Зміну моделі скасовано.")
        return current_model

    if not choice:
        print_system("Зміну моделі скасовано.")
        return current_model

    selected_model = resolve_model_choice(
        choice=choice,
        models=models,
    )

    if selected_model is None:
        print_error(
            f"Не вдалося знайти модель за значенням: {choice}"
        )
        return current_model

    if selected_model == current_model:
        print_system(
            f"Модель {selected_model} уже активна."
        )
        return current_model

    print_system(
        f"Активну модель змінено: "
        f"{current_model} → {selected_model}"
    )

    logger.info(
        "Active model changed: old_model=%s new_model=%s",
        current_model,
        selected_model,
    )

    return selected_model


def resolve_model_choice(
    choice: str,
    models: list[str],
) -> str | None:
    """Resolve a model by its number or exact name."""

    if choice.isdigit():
        selected_index = int(choice) - 1

        if 0 <= selected_index < len(models):
            return models[selected_index]

        return None

    normalized_choice = choice.casefold()

    return next(
        (
            model
            for model in models
            if model.casefold() == normalized_choice
        ),
        None,
    )


def handle_command(
    command: Command,
    session: ChatSession,
) -> None:
    """Execute a terminal command."""
    logger.info(
        "Command received: command=%s",
        command.value,
    )

    if command is Command.CLEAR:
        handle_clear(session)
        return

    if command is Command.HISTORY:
        handle_history(session)
        return

    if command is Command.HELP:
        handle_help()
        return

    if command is Command.MODEL:
        handle_model(session)
        return

    print_error(
        f"Команда {command.value} не підтримується."
    )

def handle_clear(session: ChatSession) -> None:
    """Clear the current conversation."""

    session.reset()

    logger.info("Chat history cleared")

    console.clear()

    print_banner()

    print_header(
        model_name=session.model,
        memory_enabled=settings.memory_enabled,
        workspace=settings.workspace_path,
    )

    print_footer()
    print_system("Історію поточного чату очищено.")


def handle_history(session: ChatSession) -> None:
    """Display the current conversation history."""

    logger.info(
        "Chat history displayed: message_count=%s",
        len(session.visible_messages),
    )

    print_history(session.messages)


def handle_help() -> None:
    """Display available commands."""
    logger.info("Help displayed")
    print_help()


def handle_model(session: ChatSession) -> None:
    """Change the active Ollama model."""

    session.model = select_model(session.model)