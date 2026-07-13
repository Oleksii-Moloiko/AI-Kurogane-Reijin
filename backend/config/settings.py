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
    default: bool = False,
) -> bool:
    """Read a boolean value from an environment variable."""

    value = os.getenv(name)

    if value is None:
        return default

    normalized_value = value.strip().lower()

    return normalized_value in {
        "1",
        "true",
        "yes",
        "on",
    }


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

    ollama_model: str
    ollama_host: str
    ollama_temperature: float

    memory_enabled: bool
    max_history: int

    workspace_path: Path
    data_dir: Path
    history_dir: Path
    memory_dir: Path
    logs_dir: Path
    log_file: Path
    log_level: str


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
        "0.2.0",
    ),
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
)