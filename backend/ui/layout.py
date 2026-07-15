
from pathlib import Path

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from backend.config import settings
from backend.ui.console import console
from backend.ui.theme import (
    BORDER,
    MUTED,
    PRIMARY,
    SECONDARY,
    TEXT,
)


FULL_BANNER = r"""
██╗  ██╗██╗   ██╗██████╗  ██████╗
██║ ██╔╝██║   ██║██╔══██╗██╔═══██╗
█████╔╝ ██║   ██║██████╔╝██║   ██║
██╔═██╗ ██║   ██║██╔══██╗██║   ██║
██║  ██╗╚██████╔╝██║  ██║╚██████╔╝
╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═╝ ╚═════╝
""".strip()

COMPACT_BANNER = "KURO"

MIN_FULL_BANNER_WIDTH = 55

def print_banner() -> None:

    terminal_width = console.size.width

    if terminal_width >= MIN_FULL_BANNER_WIDTH:
        banner = Text(
            FULL_BANNER,
            style=f"bold {PRIMARY}",
            no_wrap=True,
            overflow="crop",
        )

        console.print(
            Align.center(
                banner,
                vertical="middle",
            )
        )
    else:
        console.print(
            Align.center(
                Text(
                    COMPACT_BANNER,
                    style=f"bold {PRIMARY}",
                    no_wrap=True,
                )
            )
        )

    console.print(
        Align.center(
            Text(
                settings.app_full_name,
                style=f"bold {TEXT}",
                no_wrap=True,
                overflow="ellipsis",
            )
        )
    )

    console.print(
        Align.center(
            Text(
                "Local AI Assistant",
                style=SECONDARY,
                no_wrap=True,
            )
        )
    )

    console.print(
        Align.center(
            Text(
                f"Version {settings.app_version}",
                style=MUTED,
                no_wrap=True,
            )
        )
    )

    console.print(
        Align.center(
            Text(
                "Powered by Ollama",
                style=MUTED,
                no_wrap=True,
            )
        )
    )

    console.print()

def print_header(
    model_name: str,
    memory_enabled: bool = True,
    workspace: str | Path | None = None,
    provider_name: str = "ollama",
    chat_id: str | None = None,
    chat_title: str | None = None,
) -> None:
    """Display responsive information about the current session."""

    workspace_path = Path(workspace or Path.cwd()).expanduser()
    memory_status = "Enabled" if memory_enabled else "Disabled"

    table = Table.grid(
        padding=(0, 1),
        expand=True,
    )

    table.add_column(
        style=f"bold {TEXT}",
        no_wrap=True,
        width=12,
    )

    table.add_column(
        style=TEXT,
        overflow="ellipsis",
    )

    table.add_row("🤖", f"[bold {PRIMARY}]{settings.app_name}[/]")
    table.add_row("Provider", f"[{SECONDARY}]{provider_name}[/]")
    table.add_row("Model", f"[{SECONDARY}]{model_name}[/]")
    if chat_id:
        table.add_row("Chat", f"{chat_title or 'Новий чат'} ({chat_id})")
    table.add_row("Memory", memory_status)
    table.add_row("Workspace", str(workspace_path))

    console.print(
        Panel(
            table,
            border_style=BORDER,
            padding=(0, 1),
            expand=True,
        )
    )

def print_footer() -> None:

    footer = Table.grid(
        padding=(0, 2),
        expand=True,
    )

    footer.add_column(
        style=f"bold {TEXT}",
        no_wrap=True,
        width=12,
    )

    footer.add_column(
        style=MUTED,
        overflow="ellipsis",
    )

    footer.add_row("Ctrl+C", "Exit")
    footer.add_row("/clear", "Очистити чат")
    footer.add_row("/history", "Історія")
    footer.add_row("/help", "Допомога")
    footer.add_row("/model", "Змінити модель")
    footer.add_row("/new", "Новий чат")
    footer.add_row("/sessions", "Список чатів")
    footer.add_row("/resume", "Відновити чат")

    console.print()
    console.rule(style=BORDER)
    console.print(footer)
    console.print()