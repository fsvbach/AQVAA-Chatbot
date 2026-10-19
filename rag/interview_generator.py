import asyncio
from typing import AsyncGenerator, List

from openinference.semconv.trace import SpanAttributes

from rag.dialogue_state import DialogueState, count_meaningful_user_turns
from rag.question_info import get_additional_info
from config import settings, LLMRole
from llm.factory import get_client
from logger import get_logger, get_tracer
from models.schemas import ExplainRequest
from prompts import interview_prompts
from rag.context import (
    format_candidate_context,
    format_manifesto_section,
    format_party_candidate_averages,
    serialize_docs,
)
from rag.guardrail import GUARDRAIL_RESPONSES
from rag.retriever import retrieve
from rag.stance_classifier import classify_stance

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


async def astream_interview(request: ExplainRequest, metadata: dict) -> AsyncGenerator[str, None]:
    """Stream an interview response.

    Concurrently runs retrieval and stance classification, then builds a
    single LLM call whose system prompt is aware of both the retrieved
    context and the user's derived stance. The stance label is written into
    `metadata` so the router can include it in the SSE done event.
    """
    logger.info(
        "Streaming interview for question_index=%s, language=%s",
        request.question_index, request.language,
    )

    with _tracer.start_as_current_span("rag.interview") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "CHAIN")
        span.set_attribute(SpanAttributes.INPUT_VALUE, request.user_query)
        if request.user_id is not None:
            span.set_attribute(SpanAttributes.USER_ID, str(request.user_id))

        # chat_history already includes the current user message
        full_chat_history = request.chat_history

        request = request.model_copy(
            update={"chat_history": request.chat_history[:settings.chat_history_window_size]}
        )

        context, stance_result = await asyncio.gather(
            retrieve(request, mode="interview"),
            classify_stance(
                request.user_query,
                request.question_text_de if request.language == "de" else request.question_text_en or "",
                request.chat_history,
            ),
        )

        guardrail_violated = bool(context.get("guardrail_violated"))

        plan = context.get("plan")
        metadata["retrieval_log"] = {
            "guardrail_violated": guardrail_violated,
            "guardrail_category": context.get("guardrail_category"),
            "retrieval_plan": [ds.model_dump() for ds in plan.datasources] if plan else None,
            "candidate_docs": serialize_docs(context.get("candidate_docs", [])),
            "party_average_docs": serialize_docs(context.get("party_candidate_averages_docs", [])),
            "manifesto_docs": serialize_docs(context.get("manifesto_docs", [])),
        }

        meaningful_user_turns = count_meaningful_user_turns(
            full_chat_history, current_turn_violated=guardrail_violated
        )
        dialogue_state = DialogueState.from_user_turns(meaningful_user_turns)

        metadata["show_buttons"] = dialogue_state.show_buttons
        logger.info(f"MEANINGFUL USER_TURNS: {meaningful_user_turns}, buttons_first_shown: {dialogue_state.buttons_first_shown}")

        if guardrail_violated:
            response = GUARDRAIL_RESPONSES.get(
                context.get("guardrail_category") or "default",
                GUARDRAIL_RESPONSES["default"],
            ).get(request.language, GUARDRAIL_RESPONSES["default"]["en"])
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
        span.set_attribute("interview.stance", stance_result.stance)

        is_de = request.language == "de"
        question_text = request.question_text_de if is_de else request.question_text_en or ""

        system_prompt = _build_system_prompt(request, context, dialogue_state)
        if dialogue_state.buttons_first_shown:
            buttons_reminder = interview_prompts.buttons_first_shown_de(question_text) if is_de else interview_prompts.buttons_first_shown_en(question_text)
        else:
            buttons_reminder = ""

        messages = _build_messages(request, system_prompt, buttons_reminder)
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

def _build_system_prompt(request: ExplainRequest, context: dict, state: DialogueState) -> str:
    candidate_docs: List[dict] = context.get("candidate_docs", [])
    party_docs: List[dict] = context.get("party_candidate_averages_docs", [])
    manifesto_docs: List[dict] = context.get("manifesto_docs", [])
    study_context: str | None = context.get("study_context", None)
    is_de = request.language == "de"

    party_overview = format_party_candidate_averages(party_docs)
    candidate_context = format_candidate_context(candidate_docs)
    manifesto_section = format_manifesto_section(manifesto_docs, is_de)
    question_text = request.question_text_de if is_de else request.question_text_en or ""
    additional_info = get_additional_info(request.question_index)

    if is_de:
        return interview_prompts.build_de(question_text, party_overview, candidate_context, manifesto_section, state, additional_info=additional_info, study_context=study_context)

    return interview_prompts.build_en(question_text, party_overview, candidate_context, manifesto_section, state, additional_info=additional_info, study_context=study_context)


def _build_messages(request: ExplainRequest, system_prompt: str, buttons_reminder: str|None = None) -> list:
    system_msg = {"role": "system", "content": system_prompt}
    if request.chat_history:
        messages = [system_msg] + [m.model_dump() for m in request.chat_history]
    else:
        messages = [system_msg, {"role": "user", "content": request.user_query}]

    if buttons_reminder:
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                # adding button reminder at the end for recency. Adding it to user message, as only user message allowed as most recent message for generation
                messages[i] = {**messages[i], "content": messages[i]["content"] + "\n\n" + buttons_reminder}
                break

    return messages