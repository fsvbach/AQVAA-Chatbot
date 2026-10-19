from dataclasses import dataclass
from typing import List, Literal, Optional
from pydantic import BaseModel

# StanceLabel = Literal["agree", "rather_agree", "neutral", "rather_disagree", "disagree", "irrelevant"]
StanceLabel = Literal["agree", "rather_agree", "rather_disagree", "disagree", "irrelevant"]

ChatbotMode = Literal["assistant", "interview"]

InterviewStage = Literal["education", "stance_elicitation"]

class ChatHistoryMessage(BaseModel):
    role: str
    content: str
    stance: StanceLabel | None = None
    guardrail_violated: bool = False


class DatasourceQuery(BaseModel):
    index: Literal["candidates", "party_candidate_averages", "manifestos"]
    party_filter: Optional[List[str]] = None
    name_filter: Optional[List[str]] = None


class RetrievalPlan(BaseModel):
    datasources: List[DatasourceQuery]


class GuardrailResult(BaseModel):
    violated: bool
    reason: Optional[str] = None
    category: Optional[str] = None


class SurveyInfoDecision(BaseModel):
    needed: bool
    reason: Optional[str] = None


class ExplainRequest(BaseModel):
    question_index: int
    question_text_de: str
    question_text_en: Optional[str] = None
    category: Optional[str] = None
    language: str = "de"
    user_query: str
    chat_history: list[ChatHistoryMessage] = []
    user_id: Optional[int] = None


class StanceResult(BaseModel):
    stance: StanceLabel

@dataclass
class StanceClassification:
    stance: StanceLabel
    prob: float | None
    distribution: dict[str, float] | None


class UploadResponse(BaseModel):
    status: str
    document_id: str
    chunks_indexed: int


class HealthResponse(BaseModel):
    status: str
    llm_reachable: bool
    # index_loaded: bool
    model: str
