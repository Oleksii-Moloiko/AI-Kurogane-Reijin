"""User prompt parsing."""

from dataclasses import dataclass
from enum import Enum

from backend.ui.console import console
from backend.ui.theme import MUTED, PRIMARY


class Command(str, Enum):
    """Commands supported by Kuro."""

    CLEAR = "/clear"
    HISTORY = "/history"
    HELP = "/help"
    MODEL = "/model"


@dataclass(frozen=True)
class PromptResult:
    """Parsed terminal input."""

    text: str
    command: Command | None = None

    @property
    def is_command(self) -> bool:
        """Return whether the input is a supported command."""

        return self.command is not None


def parse_command(value: str) -> Command | None:
    """Parse a command from user input."""

    normalized = value.strip().casefold()

    for command in Command:
        if normalized == command.value:
            return command

    return None


def read_prompt() -> PromptResult:
    """Read and parse one terminal input."""

    user_input = console.input(
        f"[bold {PRIMARY}]❯[/] "
    ).strip()

    return PromptResult(
        text=user_input,
        command=parse_command(user_input),
    )


def read_model_choice() -> str:
    """Read a model number or model name."""

    return console.input(
        f"[bold {PRIMARY}]Оберіть модель ❯[/] "
    ).strip()


def print_prompt_hint() -> None:
    """Display a short prompt hint."""

    console.print(
        f"[{MUTED}]Напиши повідомлення або введи /help[/]"
    )