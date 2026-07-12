
from rich.panel import Panel
from rich.text import Text
from rich.table import Table
from rich.markdown import Markdown

from llm.client import StreamMetrics
from ui.theme import BORDER, ERROR, MUTED, PRIMARY, SUCCESS, TEXT, USER_COLOR
from ui.console import console



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


def print_response_statistics(metrics: StreamMetrics) -> None:
    """Display response generation statistics."""

    table = Table.grid(
        padding=(0, 1),
    )

    table.add_column(
        style=f"bold {TEXT}",
        no_wrap=True,
    )

    table.add_column(
        style=MUTED,
    )

    table.add_row(
        f"[{SUCCESS}]✔[/] Response complete",
        "",
    )

    table.add_row(
        "🧠 Model",
        metrics.model,
    )

    table.add_row(
        "⚡ Speed",
        f"{metrics.tokens_per_second:.1f} tok/sec",
    )

    table.add_row(
        "⏱ Time",
        f"{metrics.total_seconds:.2f} sec",
    )

    table.add_row(
        "📄 Tokens",
        str(metrics.eval_count),
    )

    console.print()
    console.rule(style=BORDER)
    console.print(table)
    console.print()

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

    table.add_row("/clear", "Очистити історію поточного чату")
    table.add_row("/history", "Показати історію повідомлень")
    table.add_row("/help", "Показати доступні команди")
    table.add_row("/model", "Показати або змінити модель")
    table.add_row("Ctrl+C", "Завершити роботу")

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

