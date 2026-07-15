"""Text chunking utilities."""

from backend.rag.models import TextChunk


class TextChunker:
    """Split text into overlapping chunks with natural boundaries."""

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than zero.")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative.")

        if chunk_overlap >= chunk_size:
            raise ValueError(
                "chunk_overlap must be smaller than chunk_size."
            )

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, text: str) -> list[TextChunk]:
        """Split normalized text into ordered overlapping chunks."""

        source = text.strip()

        if not source:
            return []

        chunks: list[TextChunk] = []
        start = 0
        text_length = len(source)

        while start < text_length:
            desired_end = min(
                start + self.chunk_size,
                text_length,
            )

            end = self._find_boundary(
                source,
                start=start,
                desired_end=desired_end,
            )

            if end <= start:
                end = desired_end

            content = source[start:end].strip()

            if content:
                actual_start = self._find_content_start(
                    source,
                    start,
                    end,
                )
                actual_end = actual_start + len(content)

                chunks.append(
                    TextChunk(
                        index=len(chunks),
                        content=content,
                        character_start=actual_start,
                        character_end=actual_end,
                    )
                )

            if end >= text_length:
                break

            next_start = max(
                end - self.chunk_overlap,
                start + 1,
            )

            next_start = self._move_to_word_start(
                source,
                next_start,
            )

            start = next_start

        return chunks

    @staticmethod
    def _find_boundary(
        text: str,
        *,
        start: int,
        desired_end: int,
    ) -> int:
        """Prefer paragraph, sentence, or word boundaries."""

        if desired_end >= len(text):
            return len(text)

        minimum_boundary = start + ((desired_end - start) // 2)

        boundary_candidates = (
            "\n\n",
            "\n",
            ". ",
            "! ",
            "? ",
            "; ",
            ", ",
            " ",
        )

        for separator in boundary_candidates:
            position = text.rfind(
                separator,
                minimum_boundary,
                desired_end,
            )

            if position != -1:
                return position + len(separator)

        return desired_end

    @staticmethod
    def _move_to_word_start(
        text: str,
        position: int,
    ) -> int:
        """Avoid beginning a chunk inside a word."""

        while (
            position > 0
            and position < len(text)
            and not text[position - 1].isspace()
            and not text[position].isspace()
        ):
            position -= 1

        while (
            position < len(text)
            and text[position].isspace()
        ):
            position += 1

        return position

    @staticmethod
    def _find_content_start(
        text: str,
        start: int,
        end: int,
    ) -> int:
        """Account for whitespace removed by strip()."""

        while start < end and text[start].isspace():
            start += 1

        return start