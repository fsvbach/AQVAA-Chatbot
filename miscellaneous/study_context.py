from functools import lru_cache
from pathlib import Path

from config import settings
from models.schemas import ChatbotMode


@lru_cache(maxsize=None)
def get_study_context(mode: ChatbotMode) -> str | None:
    """Load handwritten, study-specific context for the given mode.

    Reads ``<study_context_path>/<mode>.md``. Returns ``None`` when no file
    exists so the prompt simply omits the study-context section. The text is
    study-specific (survey flow, topics, answerable scope) and meant to be
    edited per study rather than in code.
    """
    path = Path(settings.study_context_path) / f"{mode}.md"
    if not path.is_file():
        return None

    text = path.read_text(encoding="utf-8").strip()
    return text or None
