
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

from backend.storage import ChatRecord
from backend.ui.theme import BORDER, ERROR, MUTED, PRIMARY, SUCCESS, TEXT, USER_COLOR
from backend.ui.console import console



def print_user(message: str) -> None:
    console.print(
        Panel(
            Text(message),
            title=f"[bold {USER_COLOR}]👤 Ти[/]",
            title_align="left",
            border_style=USER_COLOR,
            padding=(0, 1),
        )
    )


def print_error(message: str) -> None:
    console.print(
        Panel(
            Text(message, style=f"bold {ERROR}"),
            title=f"[bold {ERROR}]❌ Помилка[/]",
            title_align="left",
            border_style=ERROR,
            padding=(0, 1),
        )
    )


def print_system(message: str) -> None:
    console.print(f"[{MUTED}]· {message}[/]")


def print_help() -> None:
    """Display available terminal commands."""

    table = Table.grid(
        padding=(0, 2),
    )

    table.add_column(
        style=f"bold {TEXT}",
        no_wrap=True,
    )

    table.add_column(
        style=MUTED,
    )

    table.add_row(
        "/clear",
        "Очистити історію поточного чату",
    )
    table.add_row(
        "/history",
        "Показати історію повідомлень",
    )
    table.add_row(
        "/model",
        "Показати або змінити модель",
    )
    table.add_row(
        "/new",
        "Створити новий чат",
    )
    table.add_row(
        "/sessions",
        "Показати збережені чати",
    )
    table.add_row(
        "/resume",
        "Відновити збережений чат",
    )

    table.add_row("", "")

    table.add_row(
        "/index <path>",
        "Проіндексувати PDF або DOCX",
    )
    table.add_row(
        "/documents [limit]",
        "Показати проіндексовані документи",
    )
    table.add_row(
        "/remove <id>",
        "Видалити документ, chunks та embeddings",
    )

    table.add_row("", "")

    table.add_row(
        "/help",
        "Показати доступні команди",
    )
    table.add_row(
        "Ctrl+C",
        "Завершити роботу",
    )

    console.print(
        Panel(
            table,
            title=f"[bold {PRIMARY}]Допомога[/]",
            title_align="left",
            border_style=PRIMARY,
            padding=(1, 1),
        )
    )

def print_history(
    chat_history: list[dict[str, str]],
) -> None:
    """Display user and assistant messages from the current session."""

    visible_messages = [
        message
        for message in chat_history
        if message.get("role") != "system"
    ]

    if not visible_messages:
        print_system("Історія чату порожня.")
        return

    console.print()
    console.rule(
        f"[bold {PRIMARY}]Історія[/]",
        style=BORDER,
    )

    for message in visible_messages:
        role = message.get("role", "")
        content = message.get("content", "")

        if role == "user":
            title = "👤 Ти"
            color = USER_COLOR
        else:
            title = "🤖 Kuro"
            color = PRIMARY

        console.print(
            Panel(
                Markdown(content),
                title=f"[bold {color}]{title}[/]",
                title_align="left",
                border_style=color,
                padding=(0, 1),
            )
        )

def print_model_list(
    models: list[str],
    current_model: str,
) -> None:
    """Display locally installed Ollama models."""

    if not models:
        print_error("Локальні моделі Ollama не знайдено.")
        return

    table = Table.grid(
        padding=(0, 2),
    )

    table.add_column(
        style=f"bold {TEXT}",
        no_wrap=True,
    )

    table.add_column(style=TEXT)

    for index, model in enumerate(models, start=1):
        marker = "●" if model == current_model else " "

        style = SUCCESS if model == current_model else MUTED

        table.add_row(
            f"[{style}]{marker} {index}[/]",
            f"[{style}]{model}[/]",
        )

    console.print(
        Panel(
            table,
            title=f"[bold {PRIMARY}]Доступні моделі[/]",
            title_align="left",
            border_style=PRIMARY,
            padding=(1, 1),
        )
    )



def print_chat_list(chats: list[ChatRecord]) -> None:
    """Display saved chat sessions."""
    if not chats:
        print_system("Збережених чатів ще немає.")
        return

    table = Table(show_header=True, header_style=f"bold {TEXT}", expand=True)
    table.add_column("#", width=4)
    table.add_column("ID", no_wrap=True)
    table.add_column("Назва", overflow="ellipsis")
    table.add_column("Provider / Model", overflow="ellipsis")
    table.add_column("Повідомлень", justify="right")

    for index, chat in enumerate(chats, start=1):
        table.add_row(
            str(index),
            chat.id,
            chat.title,
            f"{chat.provider} / {chat.model}",
            str(chat.message_count),
        )

    console.print(
        Panel(
            table,
            title=f"[bold {PRIMARY}]Збережені чати[/]",
            title_align="left",
            border_style=PRIMARY,
            padding=(1, 1),
        )
    )
