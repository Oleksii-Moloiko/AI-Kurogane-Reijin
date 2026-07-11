from __future__ import annotations

from typing import Iterable, Iterator

from rich.console import Console
from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text


console = Console() # Спільна Console


# ---
# Базові рольові виводи
# ---

def print_user(message: str) -> None:
    console.print(
        Panel(
            Text(message),
            title="[bold cyan]👤Ти[/]",
            title_align="left",
            border_style="cyan",
            padding=(0, 1),
        )
    )


def print_error(message: str) -> None:
    console.print(
        Panel(
            Text(message, style="bold red"),
            title="[bold red]❌Помилка[/]",
            title_align="left",
            border_style="red",
            padding=(0, 1),
        )
    )


def print_system(message: str) -> None:
    console.print(f"[dim]· {message}[/]")


# ---
# Cтрімінг відповіді асистента
# ---

def stream_assistant_reply(
        chunk_generator: Iterable[str],
        title: str = "Модель",
        border_style: str = "green",
) -> str:
    accumulated = ""
    iterator: Iterator[str] = iter(chunk_generator)

    def render() -> Panel:
        body = Markdown(accumulated) if accumulated else Spinner(
            "dots", text="думаю..."
        )
        return Panel(
            body,
            title=f"[bold green]🤖{title}[/]",
            title_align="left",
            border_style=border_style,
            padding=(0, 1),
        )
    
    with Live(render(), console=console, refresh_per_second=15) as live:
        for chunk in iterator:
            accumulated += chunk
            live.update(render())
    
    return accumulated


# ---
# Ручний текст модуля: python -m ui.console
# ---

if __name__ == "__main__":
    import time

    print_system("Ініціалізація демо-сесії...")
    print_user("Покажи приклад коду на Python")

    def fake_stream():
        text = (
            "Ось приклад:\n\n"
            "```python\n"
            "def hello(name: str) -> str:\n"
            "    return f'Привіт, {name}!'\n"
            "```\n"
        )
        for word in text.split(" "):
            time.sleep(0.03)
            yield word + " "

    stream_assistant_reply(fake_stream())
