"""Hash helpers for document deduplication."""

from hashlib import sha256
from pathlib import Path


def calculate_file_hash(
    path: str | Path,
    *,
    block_size: int = 1024 * 1024,
) -> str:
    """Calculate a SHA-256 hash without loading the full file into memory."""

    digest = sha256()

    with Path(path).open("rb") as file:
        while block := file.read(block_size):
            digest.update(block)

    return digest.hexdigest()