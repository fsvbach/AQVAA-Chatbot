import ollama
from abc import ABC

from config import settings
from logger import get_logger

logger = get_logger(__name__)


def get_ollama_client() -> ollama.AsyncClient:
    logger.info("Creating Ollama client: base_url=%s, model=%s", settings.llm_base_url, settings.llm_model)
    return ollama.AsyncClient(host=settings.llm_base_url)
