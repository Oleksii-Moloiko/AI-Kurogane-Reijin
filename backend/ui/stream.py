"""Streaming rendering for assistant responses."""

from collections.abc import Iterable, Iterator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from ui.console import console
from ui.theme import ASSISTANT_COLOR, MUTED


STREAM_CURSOR = "█"


def stream_assistant_reply(
    chunk_generator: Iterable[str],
    title: str = "Kuro",
) -> str:
    """Render assistant output incrementally and return the complete reply."""

    accumulated = ""
    iterator: Iterator[str] = iter(chunk_generator)

    def render_streaming() -> Panel:
        """Render safe plain text while the response is still incomplete."""

        if not accumulated:
            body = Spinner(
                "dots",
                text=Text("Думаю...", style=MUTED),
            )
        else:
            body = Text(
                accumulated + STREAM_CURSOR,
                overflow="fold",
            )

        return Panel(
            body,
            title=f"[bold {ASSISTANT_COLOR}]🤖 {title}[/]",
            title_align="left",
            border_style=ASSISTANT_COLOR,
            padding=(0, 1),
        )

    def render_complete() -> Panel:
        """Render final response as Markdown."""

        body = Markdown(accumulated)

        return Panel(
            body,
            title=f"[bold {ASSISTANT_COLOR}]🤖 {title}[/]",
            title_align="left",
            border_style=ASSISTANT_COLOR,
            padding=(0, 1),
        )

    with Live(
        render_streaming(),
        console=console,
        refresh_per_second=20,
    ) as live:
        for chunk in iterator:
            accumulated += chunk
            live.update(render_streaming())

        live.update(render_complete())

    return accumulated