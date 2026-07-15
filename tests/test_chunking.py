import pytest

from backend.rag.chunking import TextChunker


def test_empty_text_returns_no_chunks() -> None:
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    assert chunker.split("") == []
    assert chunker.split("   \n\n  ") == []


def test_short_text_returns_one_chunk() -> None:
    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=20,
    )

    chunks = chunker.split(
        "Kuro is a local AI assistant."
    )

    assert len(chunks) == 1
    assert chunks[0].index == 0
    assert chunks[0].content == (
        "Kuro is a local AI assistant."
    )
    assert chunks[0].character_start == 0
    assert chunks[0].character_end == len(
        chunks[0].content
    )


def test_long_text_is_split_into_multiple_chunks() -> None:
    text = " ".join(
        f"word-{index}"
        for index in range(100)
    )

    chunker = TextChunker(
        chunk_size=120,
        chunk_overlap=20,
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1
    assert [
        chunk.index
        for chunk in chunks
    ] == list(range(len(chunks)))

    assert all(
        chunk.character_count <= 120
        for chunk in chunks
    )


def test_chunks_have_overlap() -> None:
    text = " ".join(
        f"token-{index}"
        for index in range(60)
    )

    chunker = TextChunker(
        chunk_size=100,
        chunk_overlap=25,
    )

    chunks = chunker.split(text)

    assert len(chunks) > 1

    for previous, current in zip(
        chunks,
        chunks[1:],
    ):
        assert current.character_start < (
            previous.character_end
        )


def test_chunk_does_not_start_inside_word() -> None:
    text = (
        "alpha beta gamma delta epsilon zeta eta "
        "theta iota kappa lambda mu"
    )

    chunker = TextChunker(
        chunk_size=30,
        chunk_overlap=8,
    )

    chunks = chunker.split(text)

    for chunk in chunks:
        if chunk.character_start > 0:
            assert (
                text[chunk.character_start - 1].isspace()
                or text[chunk.character_start].isspace()
            )


def test_invalid_chunk_size_raises_error() -> None:
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=0,
            chunk_overlap=0,
        )


def test_overlap_must_be_smaller_than_size() -> None:
    with pytest.raises(ValueError):
        TextChunker(
            chunk_size=100,
            chunk_overlap=100,
        )