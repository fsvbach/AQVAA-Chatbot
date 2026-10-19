from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path
from dataclasses import dataclass
from enum import Enum

class LLMProvider(str, Enum):
    OLLAMA = "ollama"
    VLLM = "vLLM"
    OPENAI = "openAI"


class LLMRole(str, Enum):
    """Logical task a client is used for. Decoupled from the physical model/endpoint."""
    GENERATION     = "generation"
    GUARDRAILS     = "guardrails"
    QUERY_ROUTING  = "query_routing"
    REWRITING      = "rewriting"
    CLASSIFICATION = "classification"


# Roles served by the auxiliary (small/fast) model. Everything else uses the
# generation profile. Add a role here to move it onto the aux endpoint.
_AUX_ROLES = {LLMRole.GUARDRAILS, LLMRole.QUERY_ROUTING}


@dataclass(frozen=True)
class ModelProfile:
    """Fully resolved connection bundle for one role. Hashable, so the factory
    can cache clients by profile and share a pool across roles that resolve equal."""
    provider: "LLMProvider"
    base_url: str
    api_key: str
    model: str


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="LLM_",
        env_file=Path(__file__).parent / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # default values assigned if no env variable defined
    llm_model: str = "Qwen/Qwen3.6-35B-A3B-FP8"
    embedding_model: str = "Qwen/Qwen3-Embedding-8B"
    llm_base_url: str = "http://localhost:8000/v1"
    embedding_base_url:str = "http://localhost:8001/v1"
    open_ai_api_key: str = ""
    llm_provider: LLMProvider = LLMProvider.VLLM

    
    # if none, fallback to values from base model
    aux_model: str | None = None
    aux_base_url: str | None = None
    aux_provider: LLMProvider | None = None

    # LLM params
    sampling_temperature: float = 1.0
    sampling_top_p: float = 0.95
    sampling_top_k: int = 64

    # RAG tuning
    rag_top_k: int = 8
    rag_chunk_size: int = 512
    rag_chunk_overlap: int = 50
    chat_history_window_size: int = 20

    # AQVAA platform — source of questions, candidates and their reactions/comments.
    # One reaction batch is fetched at index time via the backend's export endpoint.
    platform_url: str = "http://localhost:3000"
    reaction_batch_id: int | None = None
    # Shared secret; must equal LLM_SERVICE_TOKEN in the platform's .env
    service_token: str = ""

    # Storage
    additional_question_info_path: str = "data/additional_question_info.json"
    manifestos_path: str = "data/party_manifestos"
    # Study-specific prompt context. One editable "<mode>.md" file per ChatbotMode.
    study_context_path: str = "data/study_context"

    # How many manifesto chunks to retrieve per query (separate from candidate top_k)
    rag_manifesto_top_k: int = 24

    # Unfiltered manifesto search: enlarge the candidate pool by this factor before
    # the party-balanced round-robin selection down to rag_manifesto_top_k.
    rag_manifesto_fetch_multiplier: int = 4
    # Discard manifesto chunks whose similarity score is at or below this threshold.
    rag_manifesto_score_threshold: float = 0.2

    # How many party-candidate-averages docs to retrieve per query
    rag_party_averages_top_k: int = 9

    # Milvus
    milvus_uri: str = "./data/milvus/milvus_lite.db"
    milvus_candidates_collection: str = "candidates"
    milvus_manifestos_collection: str = "manifestos"
    milvus_party_averages_collection: str = "party_candidate_averages"

    # Control
    force_reindex: bool = False

    # Tracing (Arize Phoenix)
    tracing_enabled: bool = False
    phoenix_endpoint: str = "http://localhost:6006/v1/traces"
    phoenix_project_name: str = "chatbot"

    # Server
    host: str = "0.0.0.0"
    port: int = 8090

    def profile(self, role: LLMRole) -> ModelProfile:
        """Resolve a role to its connection bundle. Aux roles use the auxiliary
        endpoint (falling back per-field to generation); all others use generation."""
        if role in _AUX_ROLES:
            return ModelProfile(
                provider=self.aux_provider or self.llm_provider,
                base_url=self.aux_base_url or self.llm_base_url,
                api_key=self.open_ai_api_key,
                model=self.aux_model or self.llm_model,
            )
        return ModelProfile(
            provider=self.llm_provider,
            base_url=self.llm_base_url,
            api_key=self.open_ai_api_key,
            model=self.llm_model,
        )


settings = Settings()
