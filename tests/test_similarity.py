import pytest

from backend.rag.embeddings.errors import (
    InvalidEmbeddingError,
)
from backend.rag.similarity import cosine_similarity


def test_identical_vectors_have_similarity_one() -> None:
    result = cosine_similarity(
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0],
    )

    assert result == pytest.approx(1.0)


def test_opposite_vectors_have_negative_similarity() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [-1.0, 0.0],
    )

    assert result == pytest.approx(-1.0)


def test_orthogonal_vectors_have_zero_similarity() -> None:
    result = cosine_similarity(
        [1.0, 0.0],
        [0.0, 1.0],
    )

    assert result == pytest.approx(0.0)


def test_zero_vector_returns_zero() -> None:
    result = cosine_similarity(
        [0.0, 0.0],
        [1.0, 1.0],
    )

    assert result == 0.0


def test_different_dimensions_raise_error() -> None:
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(
            [1.0, 2.0],
            [1.0],
        )


def test_empty_embedding_raises_error() -> None:
    with pytest.raises(InvalidEmbeddingError):
        cosine_similarity(
            [],
            [],
        )