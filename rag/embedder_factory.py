from config import settings, LLMProvider
from rag.embedder_base import EmbedderClient
from rag.embedder_ollama import OllamaEmbedder
from rag.embedder_openai import OpenAIEmbedder


def get_embedder() -> EmbedderClient:
    match settings.llm_provider:
        case LLMProvider.OLLAMA:
            return OllamaEmbedder(
                base_url=settings.embedding_base_url,
                model=settings.embedding_model,
            )
        case LLMProvider.VLLM:
            return OpenAIEmbedder(
                base_url=settings.embedding_base_url,
                model=settings.embedding_model,
            )
        case LLMProvider.OPENAI:
            return OpenAIEmbedder(
                base_url=settings.embedding_base_url,
                model=settings.embedding_model,
                api_key=settings.open_ai_api_key,
            )
        case _:
            raise ValueError(f"Unsupported LLM provider: {settings.llm_provider}")
