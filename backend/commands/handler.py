"""Terminal command handlers."""

from chat.session import ChatSession
from config import MEMORY_ENABLED, WORKSPACE_PATH
from llm.client import get_available_models
from ui.console import console
from ui.layout import print_banner, print_footer, print_header
from ui.panels import (
    print_error,
    print_help,
    print_history,
    print_model_list,
    print_system,
)
from ui.prompt import Command, read_model_choice


def select_model(current_model: str) -> str:
    """Allow the user to select a locally installed Ollama model."""

    try:
        models = get_available_models()
    except Exception as error:
        print_error(
            f"Не вдалося отримати список моделей Ollama: {error}"
        )
        return current_model

    if not models:
        print_error(
            "Локальні моделі Ollama не знайдено."
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


def clear_chat(session: ChatSession) -> None:
    """Clear the active conversation."""

    session.reset()

    console.clear()

    print_banner()

    print_header(
        model_name=session.model,
        memory_enabled=MEMORY_ENABLED,
        workspace=WORKSPACE_PATH,
    )

    print_footer()
    print_system("Історію поточного чату очищено.")

def handle_command(
    command: Command,
    session: ChatSession,
) -> None:
    """Execute a terminal command."""

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

    console.clear()

    print_banner()

    print_header(
        model_name=session.model,
        memory_enabled=MEMORY_ENABLED,
        workspace=WORKSPACE_PATH,
    )

    print_footer()
    print_system("Історію поточного чату очищено.")


def handle_history(session: ChatSession) -> None:
    """Display the current conversation history."""

    print_history(session.messages)


def handle_help() -> None:
    """Display available commands."""

    print_help()


def handle_model(session: ChatSession) -> None:
    """Change the active Ollama model."""

    session.model = select_model(session.model)