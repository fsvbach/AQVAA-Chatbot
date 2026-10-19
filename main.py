import os

# Silence gRPC C++ layer noise from Milvus Lite (recvmsg EMSGSIZE warnings,
# HTTP/2 keepalive GOAWAYs). Must be set before pymilvus is imported.
os.environ.setdefault("GRPC_VERBOSITY", "NONE")
os.environ.setdefault("GLOG_minloglevel", "2")

import httpx
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI

from models.schemas import HealthResponse
from api.router import router
from config import settings, LLMProvider
from logger import get_logger, setup_tracing
from rag.embedder import close_embedder_client
from rag.indexer import build_index

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting LLM microservice…")
    if settings.tracing_enabled:
        setup_tracing(settings.phoenix_endpoint, settings.phoenix_project_name)
    else:
        logger.info("RAG Tracing disabled. To enable tracing, set LLM_TRACING_ENABLED=true in your .env file. (Check README.md for more details)")
    await build_index()
    logger.info("LLM microservice ready.")
    yield
    await close_embedder_client()
    logger.info("LLM microservice shutting down.")


app = FastAPI(
    title="LLM / RAG Microservice",
    description="Streams contextual explanations of political questions grounded in candidate position data.",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(router)


def _llm_health_url() -> str:
    """Liveness probe URL for the configured LLM provider.

    Ollama serves ``/api/tags``; vLLM and other OpenAI-compatible servers serve
    ``/v1/models`` (``llm_base_url`` may or may not already include the ``/v1`` prefix).
    """
    base = settings.llm_base_url.rstrip("/")
    if settings.llm_provider == LLMProvider.OLLAMA:
        return f"{base}/api/tags"
    return f"{base}/models" if base.endswith("/v1") else f"{base}/v1/models"


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health():
    llm_reachable = False
    try:
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(_llm_health_url())
            llm_reachable = resp.status_code == 200
    except Exception:
        pass

    return HealthResponse(
        status="ok" if llm_reachable else "degraded",
        llm_reachable=llm_reachable,
        model=settings.llm_model,
    )


if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host=settings.host,
        port=settings.port,
        reload=False,
    )
