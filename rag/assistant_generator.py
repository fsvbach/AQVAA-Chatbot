import asyncio
import json
from typing import AsyncGenerator, List

from openinference.semconv.trace import SpanAttributes

from config import settings, LLMRole
from llm.factory import get_client
from logger import get_logger, get_tracer
from models.schemas import ExplainRequest
from prompts import assistant_system_prompts
from rag.context import (
    format_candidate_context,
    format_manifesto_section,
    format_party_candidate_averages,
    serialize_docs,
)
from rag.guardrail import GUARDRAIL_RESPONSES
from rag.question_info import get_additional_info
from rag.retriever import retrieve
from rag.stance_classifier import classify_stance

logger = get_logger(__name__)

_tracer = get_tracer(__name__)


async def astream_explain(request: ExplainRequest, metadata: dict) -> AsyncGenerator[str, None]:
    """Async generator that yields LLM response tokens for a given explain request.

    1. Retrieves relevant candidate positions (and optionally manifesto chunks).
    2. Builds a language-aware system prompt with the retrieved context.
    3. Prepends the system message to the conversation and streams the response.

    """
    logger.info(
        "Streaming explain for question_index=%d, language=%s",
        request.question_index, request.language,
    )
    
    with _tracer.start_as_current_span("rag.explain") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        span.set_attribute(SpanAttributes.INPUT_VALUE, request.user_query)
        if request.user_id is not None:
            span.set_attribute(SpanAttributes.USER_ID, str(request.user_id))
        
        # only return chat history <= chat_history_window_size
        request_w_adjusted_history_size = request.model_copy(
            update={
                "chat_history": request.chat_history[:settings.chat_history_window_size]
            }
        )

        context, stance_result = await asyncio.gather(
            retrieve(request_w_adjusted_history_size, mode="assistant"),
            classify_stance(
                request.user_query,
                    request.question_text_de if request.language == "de" else request.question_text_en or "",
                request.chat_history,
            ),
        )

        plan = context.get("plan")
        metadata["retrieval_log"] = {
            "guardrail_violated": context.get("guardrail_violated", False),
            "guardrail_category": context.get("guardrail_category"),
            "retrieval_plan": [ds.model_dump() for ds in plan.datasources] if plan else None,
            "candidate_docs": serialize_docs(context.get("candidate_docs", [])),
            "party_average_docs": serialize_docs(context.get("party_candidate_averages_docs", [])),
            "manifesto_docs": serialize_docs(context.get("manifesto_docs", [])),
        }

        # needed for relevant user message count in frontend
        metadata["guardrail_violated"] = bool(context.get("guardrail_violated"))

        if context.get("guardrail_violated"):
            response = GUARDRAIL_RESPONSES.get(context.get("guardrail_category") or "default", GUARDRAIL_RESPONSES["default"]).get(request.language, GUARDRAIL_RESPONSES["default"]["en"])
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
            yield response
            return

        metadata["stance"] = stance_result.stance
        if stance_result.prob is not None:
            metadata["stance_probs"] = {
                "chosen": stance_result.stance,
                "prob": round(stance_result.prob, 4),
                "distribution": stance_result.distribution,
            }
        span.set_attribute("assistant.stance", stance_result.stance)

        system_prompt = _build_system_prompt(request_w_adjusted_history_size, context)
        messages = _build_messages(request_w_adjusted_history_size, system_prompt)
        client = get_client(LLMRole.GENERATION)
        tokens: List[str] = []

        async for chunk in client.stream(messages, metadata, options={
            "temperature": settings.sampling_temperature,
            "top_p": settings.sampling_top_p,
            "top_k": settings.sampling_top_k,
        }):
            if chunk.content:
                tokens.append(chunk.content)
            yield chunk.content
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, "".join(tokens))


async def query_explain(request: ExplainRequest) -> str:
    """Non-streaming version. Used for debugging via /explanations/completions."""
    with _tracer.start_as_current_span("rag.explain") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        span.set_attribute(SpanAttributes.INPUT_VALUE, request.user_query)
        if request.user_id is not None:
            span.set_attribute(SpanAttributes.USER_ID, str(request.user_id))

        # only return chat history <= chat_history_window_size
        request_w_adjusted_history_size = request.model_copy(
            update={
                "chat_history": request.chat_history[:settings.chat_history_window_size]
            }
        )

        context = await retrieve(request_w_adjusted_history_size, mode="assistant")
        

        if context.get("guardrail_violated"):
            response = GUARDRAIL_RESPONSES.get(context.get("guardrail_category") or "default", GUARDRAIL_RESPONSES["default"]).get(request.language, GUARDRAIL_RESPONSES["default"]["en"])
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response)
            return response

        system_prompt = _build_system_prompt(request_w_adjusted_history_size, context)
        messages = _build_messages(request_w_adjusted_history_size, system_prompt)

        llm_client = get_client(LLMRole.GENERATION)
        response = await llm_client.complete(messages=messages, options={
            "temperature": settings.sampling_temperature,
            "top_p": settings.sampling_top_p,
            "top_k": settings.sampling_top_k,
        })
        span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content or "")
        return response.content


# ---------------------------------------------------------------------------
# Prompt building
# ---------------------------------------------------------------------------

def _build_system_prompt(request: ExplainRequest, context: dict) -> str:
    candidate_docs: List[dict] = context.get("candidate_docs", [])
    party_candidate_averages_docs: List[dict] = context.get("party_candidate_averages_docs", [])
    manifesto_docs: List[dict] = context.get("manifesto_docs", [])
    study_context: str | None = context.get("study_context", None)
    is_de = request.language == "de"
    party_overview = format_party_candidate_averages(party_candidate_averages_docs)
    candidate_context = format_candidate_context(candidate_docs)
    manifesto_section = format_manifesto_section(manifesto_docs, is_de)
    question_text = request.question_text_de if is_de else request.question_text_en or ""
    additional_info = get_additional_info(request.question_index)

    if is_de:
        return assistant_system_prompts.build_de(question_text, party_overview, candidate_context, manifesto_section, additional_info, study_context=study_context)
    return assistant_system_prompts.build_en(question_text, party_overview, candidate_context, manifesto_section, additional_info, study_context=study_context)


def _build_messages(request: ExplainRequest, system_prompt: str) -> list:
    system_msg = {"role": "system", "content": system_prompt}
    if request.chat_history:
        return [system_msg] + [m.model_dump() for m in request.chat_history]
    return [system_msg, {"role": "user", "content": request.user_query}]
