from backend.commands.handler import resolve_model_choice
from backend.ui.prompt import Command, parse_command


def test_resolve_model_choice_by_number() -> None:
    assert resolve_model_choice("2", ["a", "b"]) == "b"


def test_resolve_model_choice_by_name_case_insensitive() -> None:
    assert resolve_model_choice("GEMMA3", ["gemma3"]) == "gemma3"


def test_resolve_model_choice_rejects_invalid_value() -> None:
    assert resolve_model_choice("9", ["a"]) is None


def test_parse_new_commands() -> None:
    test_cases = {
        "/sessions": Command.SESSIONS,
        "/resume": Command.RESUME,
        "/index": Command.INDEX,
        "/documents": Command.DOCUMENTS,
        "/remove": Command.REMOVE,
    }

    for value, expected_command in test_cases.items():
        command, arguments = parse_command(value)

        assert command is expected_command
        assert arguments == ""

def test_parse_command_without_arguments() -> None:
    command, arguments = parse_command(
        "/documents"
    )

    assert command is Command.DOCUMENTS
    assert arguments == ""


def test_parse_command_preserves_path_with_spaces() -> None:
    command, arguments = parse_command(
        "/index /Users/me/My Documents/report.pdf"
    )

    assert command is Command.INDEX
    assert arguments == (
        "/Users/me/My Documents/report.pdf"
    )


def test_parse_remove_command() -> None:
    command, arguments = parse_command(
        "/remove abc123"
    )

    assert command is Command.REMOVE
    assert arguments == "abc123"


def test_regular_message_is_not_command() -> None:
    command, arguments = parse_command(
        "Explain this document"
    )

    assert command is None
    assert arguments == ""