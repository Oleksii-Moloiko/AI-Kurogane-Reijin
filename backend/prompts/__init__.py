from pathlib import Path
from functools import lru_cache

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_system_prompt() -> str:
    path = _PROMPTS_DIR / "system.txt"
    return path.read_text(encoding="utf-8").strip()
