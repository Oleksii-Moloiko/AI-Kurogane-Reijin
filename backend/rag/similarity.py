"""Vector similarity functions."""

from math import sqrt

from backend.rag.embeddings.errors import (
    InvalidEmbeddingError,
)


def cosine_similarity(
    first: list[float],
    second: list[float],
) -> float:
    """Calculate cosine similarity between two vectors."""

    if not first or not second:
        raise InvalidEmbeddingError(
            "Cannot compare empty embeddings."
        )

    if len(first) != len(second):
        raise InvalidEmbeddingError(
            "Embedding dimensions do not match."
        )

    dot_product = sum(
        first_value * second_value
        for first_value, second_value in zip(
            first,
            second,
            strict=True,
        )
    )

    first_norm = sqrt(
        sum(value * value for value in first)
    )
    second_norm = sqrt(
        sum(value * value for value in second)
    )

    if first_norm == 0 or second_norm == 0:
        return 0.0

    similarity = dot_product / (
        first_norm * second_norm
    )

    return max(-1.0, min(1.0, similarity))