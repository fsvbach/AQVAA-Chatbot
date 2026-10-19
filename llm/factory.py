from functools import lru_cache

from llm.ollama import OllamaClient
from llm.openai import OpenAIClient
from llm.base import LLMClient
from config import settings, LLMProvider, LLMRole, ModelProfile


@lru_cache
def _build_client(profile: ModelProfile) -> LLMClient:
    """Build (and cache) one client per distinct profile. Roles that resolve to
    the same profile share a single client and connection pool."""
    match profile.provider:
        case LLMProvider.OLLAMA:
            return OllamaClient(base_url=profile.base_url, model=profile.model)

        case LLMProvider.VLLM:
            return OpenAIClient(base_url=profile.base_url, api_key="not_needed", model=profile.model)

        case LLMProvider.OPENAI:
            return OpenAIClient(base_url=profile.base_url, api_key=profile.api_key, model=profile.model)

        case _:
            raise ValueError(f"Unsupported LLM provider: {profile.provider}")


def get_client(role: LLMRole = LLMRole.GENERATION) -> LLMClient:
    """Return the client configured for the given task role."""
    return _build_client(settings.profile(role))
