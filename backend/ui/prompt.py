"""User prompt parsing."""

from dataclasses import dataclass
from enum import Enum

from backend.ui.console import console
from backend.ui.theme import MUTED, PRIMARY


class Command(str, Enum):
    CLEAR = "/clear"
    HISTORY = "/history"
    HELP = "/help"
    MODEL = "/model"
    NEW = "/new"
    SESSIONS = "/sessions"
    RESUME = "/resume"

    INDEX = "/index"
    DOCUMENTS = "/documents"
    REMOVE = "/remove"


@dataclass(frozen=True)
class PromptResult:
    text: str
    command: Command | None = None
    arguments: str = ""

    @property
    def is_command(self) -> bool:
        return self.command is not None

    @property
    def is_unknown_command(self) -> bool:
        return (
            self.text.startswith("/")
            and self.command is None
        )

def parse_command(
    value: str,
) -> tuple[Command | None, str]:
    """Parse a command and preserve its raw arguments."""

    normalized = value.strip()

    if not normalized.startswith("/"):
        return None, ""

    command_value, separator, arguments = normalized.partition(" ")

    command_value = command_value.casefold()

    command = next(
        (
            candidate
            for candidate in Command
            if candidate.value == command_value
        ),
        None,
    )

    if command is None:
        return None, ""

    return (
        command,
        arguments.strip() if separator else "",
    )


def read_prompt() -> PromptResult:
    user_input = console.input(
        f"[bold {PRIMARY}]❯[/] "
    ).strip()

    command, arguments = parse_command(user_input)

    return PromptResult(
        text=user_input,
        command=command,
        arguments=arguments,
    )


def read_model_choice() -> str:
    return console.input(
        f"[bold {PRIMARY}]Оберіть модель ❯[/] "
    ).strip()


def read_chat_choice() -> str:
    return console.input(
        f"[bold {PRIMARY}]Оберіть чат за номером або ID ❯[/] "
    ).strip()


def print_prompt_hint() -> None:
    console.print(
        f"[{MUTED}]Напиши повідомлення або введи /help[/]"
    )