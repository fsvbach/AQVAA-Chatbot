"""Guardrail checks for the chatbot pipeline.

Runs all registered checks concurrently. Further checks can be added by
appending a coroutine to the list passed to asyncio.gather inside
check_for_guardrail_violation().

Current checks:
  off_topic       - rejects queries unrelated to Swiss politics
  vote_intention  - rejects "who should I vote for" style requests
  critical        - rejects offensive content and opinions on sensitive topics
  llm_opinion     - rejects requests for the model's own opinion
"""
import asyncio

from openinference.semconv.trace import MessageAttributes, SpanAttributes
from opentelemetry.trace import StatusCode

from llm import factory
from config import settings, LLMRole
from logger import get_logger, get_tracer
from models.schemas import ChatHistoryMessage, GuardrailResult
from prompts.off_topic_guardrail import CURRENT as _OFF_TOPIC_SYSTEM_PROMPT
from prompts.vote_intention_guardrail import CURRENT as _VOTE_INTENTION_SYSTEM_PROMPT
from prompts.critical_guardrail import CURRENT as _CRITICAL_SYSTEM_PROMPT
from prompts.llm_opinion_guardrail import CURRENT as _LLM_OPINION_SYSTEM_PROMPT

logger = get_logger(__name__)
_tracer = get_tracer(__name__)

GUARDRAIL_RESPONSES: dict[str, dict[str, str]] = {
    "off_topic": {
        "de": (
            "Diese Frage liegt ausserhalb meines Themenbereichs. "
            "Ich helfe Ihnen gerne bei Fragen zur Schweizer Politik, "
            "zu Parteien oder zu Kandidierenden."
        ),
        "en": (
            "This question is outside my area of expertise. "
            "I'm happy to help with questions about Swiss politics, "
            "parties, or candidates."
        ),
    },
    "vote_intention": {
        "de": (
            "Als politischer Chatbot gebe ich keine Wahlempfehlungen. "
            "Ich helfe Ihnen gerne dabei, die Positionen der Parteien und Kandidierenden "
            "zu erläutern, damit Sie Ihre eigene Entscheidung treffen können."
        ),
        "en": (
            "As a political chatbot, I don't make voting recommendations. "
            "I'm happy to explain the positions of parties and candidates "
            "so you can make your own informed decision."
        ),
    },
    "critical": {
        "de": (
            "Zu diesem Thema kann ich mich nicht äussern. "
            "Als politischer Chatbot beschränke ich mich auf sachliche "
            "Informationen zur Schweizer Politik."
        ),
        "en": (
            "I'm not able to comment on this topic. "
            "As a political chatbot, I limit myself to factual "
            "information about Swiss politics."
        ),
    },
    "llm_opinion": {
        "de": (
            "Ich darf keine eigene Meinung äussern. "
            "Als politischer Chatbot beschränke ich mich auf sachliche "
            "Informationen zur Schweizer Politik."
        ),
        "en": (
            "I'm not allowed to express my personal opinion. "
            "As a political chatbot, I limit myself to factual "
            "information about Swiss politics."
        ),
    },
    "default": {
        "de": "Diese Anfrage kann ich leider nicht bearbeiten.",
        "en": "I'm unable to process this request.",
    },
}


async def check_for_guardrail_violation(question_text: str, user_query: str, chat_history: list[ChatHistoryMessage]) -> GuardrailResult:
    """Run all guardrail checks concurrently and return the combined result.

    Returns GuardrailResult(violated=False) on any error so that a failing
    guardrail never blocks a legitimate query.
    """
    results: tuple[GuardrailResult, ...] = await asyncio.gather(
        _guardrail_llm_call("off_topic", _OFF_TOPIC_SYSTEM_PROMPT, question_text, user_query, chat_history),
        _guardrail_llm_call("vote_intention", _VOTE_INTENTION_SYSTEM_PROMPT, question_text, user_query, chat_history),
        _guardrail_llm_call("critical", _CRITICAL_SYSTEM_PROMPT, question_text, user_query, chat_history),
        _guardrail_llm_call("llm_opinion", _LLM_OPINION_SYSTEM_PROMPT, question_text, user_query, chat_history)
    )
    violated = any(r.violated for r in results)
    first_violated = next((r for r in results if r.violated), None)
    reasons = [r.reason for r in results if r.violated and r.reason]
    return GuardrailResult(
        violated=violated,
        reason="; ".join(reasons) or None,
        category=first_violated.category if first_violated else None,
    )


_GUARDRAIL_TIMEOUT = 15.0


async def _guardrail_llm_call(
    category: str,
    system_prompt: str,
    question_text: str,
    user_query: str,
    chat_history: list[ChatHistoryMessage],
) -> GuardrailResult:
    """Generic LLM-based guardrail check. Falls back to not-violated on any error."""
    with _tracer.start_as_current_span(f"guardrail.{category}") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, user_query)
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, settings.profile(LLMRole.GUARDRAILS).model)

        full_system_prompt = system_prompt + f"\n\nTopic question: {question_text}"
        span.set_attribute(SpanAttributes.LLM_SYSTEM, full_system_prompt)

        messages = [{"role": "system", "content": full_system_prompt}]
        if chat_history:
            messages.extend([m.model_dump() for m in chat_history])
        messages.append({"role": "user", "content": user_query})

        try:
            client = factory.get_client(LLMRole.GUARDRAILS)
            async with asyncio.timeout(_GUARDRAIL_TIMEOUT):
                response = await client.complete(
                    messages=messages,
                    options={"temperature": 0.0, "reasoning": False},
                    format=GuardrailResult.model_json_schema(),
                )
            if not response.content:
                raise ValueError("response.message.content is None")

            result = GuardrailResult.model_validate_json(response.content)
            result = result.model_copy(update={"category": category})
            logger.info("Guardrail %s check: violated=%s", category, result.violated)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content)
            span.set_attribute("guardrail.violated", result.violated)
            return result
        except Exception as exc:
            logger.warning(
                "Guardrail %s check failed (%s: %s) — defaulting to not-violated",
                category,
                type(exc).__name__,
                exc,
            )
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, description=type(exc).__name__)
            return GuardrailResult(violated=False, category=category)
