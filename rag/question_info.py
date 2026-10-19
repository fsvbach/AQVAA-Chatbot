"""Per-question supplementary info: term definitions, background, pros/cons.

Sourced from the smartvote popup fields and stored as JSON keyed by question id.
The platform has no field for this, so it stays with the chatbot.
Loaded lazily on first access, or eagerly via load_additional_question_info()
during index startup (see rag.indexer.build_index).
"""
import json
import os
from typing import Dict, Optional

from config import settings
from logger import get_logger

logger = get_logger(__name__)

# None = not yet loaded; {} = loaded but unavailable/empty.
_additional_info: Optional[Dict[int, dict]] = None


def load_additional_question_info() -> Dict[int, dict]:
    """Read the additional-info JSON from disk into the module cache and return it."""
    global _additional_info
    path = settings.additional_question_info_path
    if not os.path.exists(path):
        logger.warning("Additional question info not found at %s — feature disabled.", path)
        _additional_info = {}
        return _additional_info

    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    _additional_info = {int(qid): fields for qid, fields in raw.items()}
    logger.info("Loaded additional question info for %d questions.", len(_additional_info))
    return _additional_info


def get_additional_info(question_id: int) -> Optional[dict]:
    """Return the supplementary info for a question, or None if unavailable."""
    if _additional_info is None:
        load_additional_question_info()
    return _additional_info.get(question_id)  # type: ignore[union-attr]
