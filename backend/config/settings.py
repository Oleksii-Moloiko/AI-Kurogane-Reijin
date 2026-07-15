"""Centralized application settings."""

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BASE_DIR / ".env"

load_dotenv(ENV_FILE)


def get_bool_env(
    name: str,
    default: bool,
) -> bool:
    raw_value = os.getenv(name)

    if raw_value is None:
        return default

    normalized = raw_value.strip().casefold()

    if normalized in {"1", "true", "yes", "on"}:
        return True

    if normalized in {"0", "false", "no", "off"}:
        return False

    raise ValueError(
        f"Environment variable {name} must be a boolean."
    )


def get_int_env(
    name: str,
    default: int,
) -> int:
    """Read an integer value from an environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return int(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {name} must be an integer."
        ) from error


def get_float_env(
    name: str,
    default: float,
) -> float:
    """Read a float value from an environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    try:
        return float(value)
    except ValueError as error:
        raise ValueError(
            f"Environment variable {name} must be a number."
        ) from error


def get_path_env(
    name: str,
    default: Path,
) -> Path:
    """Read and normalize a filesystem path."""

    value = os.getenv(name, "").strip()

    if not value:
        return default

    path = Path(value).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    return path.resolve()


@dataclass(frozen=True, slots=True)
class Settings:
    """Immutable application configuration."""

    app_name: str
    app_full_name: str
    app_version: str

    llm_provider: str
    llm_model: str
    llm_temperature: float
    ollama_model: str
    ollama_host: str
    ollama_temperature: float
    openai_base_url: str
    openai_api_key: str

    memory_enabled: bool
    max_history: int

    workspace_path: Path
    data_dir: Path
    history_dir: Path
    memory_dir: Path
    logs_dir: Path
    log_file: Path
    log_level: str
    database_path: Path
    documents_dir: Path
    max_document_size_mb: int
    chunk_size: int
    chunk_overlap: int
    embedding_provider: str
    embedding_model: str
    embedding_batch_size: int
    rag_top_k: int
    rag_min_similarity: float
    rag_enabled: bool
    rag_max_context_chars: int


DATA_DIR = get_path_env(
    "DATA_DIR",
    BASE_DIR / "data",
)

LOGS_DIR = get_path_env(
    "LOGS_DIR",
    DATA_DIR / "logs",
)

settings = Settings(
    app_name=os.getenv(
        "APP_NAME",
        "Kuro",
    ),
    app_full_name=os.getenv(
        "APP_FULL_NAME",
        "AI Kurogane Reijin",
    ),
    app_version=os.getenv(
        "APP_VERSION",
        "0.4.0",
    ),
    llm_provider=os.getenv("LLM_PROVIDER", "ollama").strip().lower(),
    llm_model=os.getenv("LLM_MODEL", os.getenv("OLLAMA_MODEL", "gemma3")),
    llm_temperature=get_float_env("LLM_TEMPERATURE", get_float_env("OLLAMA_TEMPERATURE", 0.7)),
    ollama_model=os.getenv(
        "OLLAMA_MODEL",
        "gemma3",
    ),
    ollama_host=os.getenv(
        "OLLAMA_HOST",
        "http://localhost:11434",
    ),
    ollama_temperature=get_float_env(
        "OLLAMA_TEMPERATURE",
        0.7,
    ),
    openai_base_url=os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip(),
    openai_api_key=os.getenv("OPENAI_API_KEY", "").strip(),
    memory_enabled=get_bool_env(
        "MEMORY_ENABLED",
        default=False,
    ),
    max_history=get_int_env(
        "MAX_HISTORY",
        20,
    ),
    workspace_path=get_path_env(
        "WORKSPACE_PATH",
        BASE_DIR,
    ),
    data_dir=DATA_DIR,
    history_dir=get_path_env(
        "HISTORY_DIR",
        DATA_DIR / "history",
    ),
    memory_dir=get_path_env(
        "MEMORY_DIR",
        DATA_DIR / "memory",
    ),
    logs_dir=LOGS_DIR,
    log_file=get_path_env(
        "LOG_FILE",
        LOGS_DIR / "kuro.log",
    ),
    log_level=os.getenv(
        "LOG_LEVEL",
        "INFO",
    ).strip().upper(),
    database_path=get_path_env("DATABASE_PATH", DATA_DIR / "kuro.db"),
    documents_dir=get_path_env(
        "DOCUMENTS_DIR",
        DATA_DIR / "documents",
    ),
    max_document_size_mb=get_int_env(
        "MAX_DOCUMENT_SIZE_MB",
        20,
    ),
    chunk_size=get_int_env(
        "CHUNK_SIZE",
        1000,
    ),
    chunk_overlap=get_int_env(
        "CHUNK_OVERLAP",
        150,
    ),
    embedding_provider=os.getenv(
        "EMBEDDING_PROVIDER",
        "ollama",
    ).strip().casefold(),

    embedding_model=os.getenv(
        "EMBEDDING_MODEL",
        "nomic-embed-text",
    ).strip(),

    embedding_batch_size=get_int_env(
        "EMBEDDING_BATCH_SIZE",
        16,
    ),
    rag_top_k=get_int_env(
        "RAG_TOP_K",
        5,
    ),
    rag_min_similarity=get_float_env(
        "RAG_MIN_SIMILARITY",
        0.35,
    ),
    rag_enabled=get_bool_env(
        "RAG_ENABLED",
        True,
    ),

    rag_max_context_chars=get_int_env(
        "RAG_MAX_CONTEXT_CHARS",
        6000,
    ),
)