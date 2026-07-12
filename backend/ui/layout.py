
from pathlib import Path

from rich.align import Align
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from config import APP_FULL_NAME, APP_NAME, APP_VERSION
from ui.console import console
from ui.theme import BORDER, MUTED, PRIMARY, SECONDARY, TEXT

APP_NAME = "Kuro"
APP_FULL_NAME = "AI Kurogane Reijin"
APP_VERSION = "0.1.0"

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
                APP_FULL_NAME,
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
                f"Version {APP_VERSION}",
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

    table.add_row("🤖", f"[bold {PRIMARY}]{APP_NAME}[/]")
    table.add_row("Local LLM", f"[{SECONDARY}]{model_name}[/]")
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

    console.print()
    console.rule(style=BORDER)
    console.print(footer)
    console.print()