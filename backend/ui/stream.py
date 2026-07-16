"""Streaming rendering for assistant responses."""

from collections.abc import Iterable, Iterator

from rich.live import Live
from rich.markdown import Markdown
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from backend.ui.console import console
from backend.ui.theme import ASSISTANT_COLOR, MUTED


STREAM_CURSOR = "█"


def stream_assistant_reply(
    chunk_generator: Iterable[str],
    title: str = "Kuro",
) -> str:
    """Collect the full assistant reply while showing a spinner, then
    render the complete response as a single Markdown panel once
    generation has finished."""

    accumulated = ""
    iterator: Iterator[str] = iter(chunk_generator)

    def render_thinking() -> Panel:
        """Waiting indicator shown while the reply is still generating."""

        return Panel(
            Spinner(
                "dots",
                text=Text("Думаю...", style=MUTED),
            ),
            title=f"[bold {ASSISTANT_COLOR}]🤖 {title}[/]",
            title_align="left",
            border_style=ASSISTANT_COLOR,
            padding=(0, 1),
        )

    # Спінер показуємо, поки відповідь генерується, але сам текст
    # не малюємо частинами — лише крутиться індикатор очікування.
    with Live(
        render_thinking(),
        console=console,
        refresh_per_second=10,
        transient=True,
    ) as live:
        for chunk in iterator:
            accumulated += chunk
            live.refresh()

    # Після завершення сесії генерації виводимо готову відповідь
    # одним блоком у вигляді Markdown-панелі.
    console.print(
        Panel(
            Markdown(accumulated),
            title=f"[bold {ASSISTANT_COLOR}]🤖 {title}[/]",
            title_align="left",
            border_style=ASSISTANT_COLOR,
            padding=(0, 1),
        )
    )

    return accumulated