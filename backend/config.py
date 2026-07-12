import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent

load_dotenv(BASE_DIR / ".env")

def get_bool_env(name:str, default: bool = False) -> bool:
    value = os.getenv(name)

    if value is None:
        return default
    
    return value.strip().lower() in {
        "1",
        "true",
        "yes",
        "on"
    }

APP_NAME = os.getenv("APP_NAME", "Kuro")
APP_FULL_NAME = os.getenv(
    "APP_FULL_NAME",
    "AI Kurogane Reijin",
)
APP_VERSION = os.getenv("APP_VERSION", "0.1.0")

MODEL_NAME = os.getenv("OLLAMA_MODEL", "gemma3")
OLLAMA_HOST = os.getenv(
    "OLLAMA_HOST",
    "http://localhost:11434",
)

MEMORY_ENABLED = get_bool_env(
    "MEMORY_ENABLED",
    default=True,
)

_workspace_value = os.getenv("WORKSPACE_PATH", "").strip()

WORKSPACE_PATH = (
    Path(_workspace_value).expanduser()
    if _workspace_value
    else BASE_DIR
)