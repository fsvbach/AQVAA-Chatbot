"""LLM-based classifier that derives the user's stance toward a VAA statement."""
import asyncio
import json
import math
from typing import Any, List

from openinference.semconv.trace import SpanAttributes

from llm.factory import get_client
from config import LLMRole
from logger import get_logger, get_tracer
from models.schemas import ChatHistoryMessage, StanceClassification, StanceResult
from prompts.stance_prompts import CURRENT as _STANCE_SYSTEM_PROMPT

logger = get_logger(__name__)
_tracer = get_tracer(__name__)

_TIMEOUT = 15.0
_FALLBACK = StanceClassification(stance="irrelevant", prob=None, distribution=None)
_HISTORY_WINDOW = 4
_STANCE_LABELS = {"agree", "rather_agree", "rather_disagree", "disagree", "irrelevant"}


def _extract_stance_probs(logprobs: list[Any]) -> tuple[float, dict[str, float]] | tuple[None, None]:
    buf: str = ""
    buf_first_entry = None

    for entry in logprobs:
        clean = entry.token.strip().strip('"')

        if buf and not any(label.startswith(buf + clean) for label in _STANCE_LABELS):
            buf = ""
            buf_first_entry = None

        if not buf:
            if not clean or not any(label.startswith(clean) for label in _STANCE_LABELS):
                continue
            buf_first_entry = entry

        buf += clean

        if buf in _STANCE_LABELS:
            prob = math.exp(buf_first_entry.logprob)
            distribution = {
                lp.token.strip().strip('"'): round(math.exp(lp.logprob), 4)
                for lp in buf_first_entry.top_logprobs
            }
            return prob, distribution

    logger.warning("Stance token not found in logprobs — check tokenization")
    return None, None


async def classify_stance(
    user_query: str,
    question_text: str,
    chat_history: List[ChatHistoryMessage],
) -> StanceClassification:
    """Classify the user's stance toward the VAA statement."""
    with _tracer.start_as_current_span("stance_classifier.classify") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.INPUT_VALUE, user_query)

        recent_history = chat_history[-_HISTORY_WINDOW:] if chat_history else []
        history_text = "\n".join(f"{m.role}: {m.content}" for m in recent_history)

        user_content = f"Statement: {question_text}\n"
        if history_text:
            user_content += f"Recent conversation:\n{history_text}\n"
        user_content += f"Latest user message: {user_query}"

        try:
            client = get_client(LLMRole.CLASSIFICATION)
            async with asyncio.timeout(_TIMEOUT):
                response = await client.complete(
                    messages=[
                        {"role": "system", "content": _STANCE_SYSTEM_PROMPT},
                        {"role": "user", "content": user_content},
                    ],
                    options={
                        "temperature": 0.0,
                        "logprobs": True,
                        "top_logprobs": 6,
                    },
                    format=StanceResult.model_json_schema(),
                )

            if not response.content:
                raise ValueError("Stance classifier returned empty response")

            result = StanceResult.model_validate_json(response.content)

            prob, distribution = _extract_stance_probs(response.logprobs) if response.logprobs else (None, None)

            if prob is not None:
                logger.info("Stance probs — chosen: %s (p=%.4f), top candidates: %s", result.stance, prob, distribution)
                span.set_attribute("stance.prob", prob)
                span.set_attribute("stance.distribution", json.dumps(distribution))

            logger.info("Stance classified: %s", result.stance)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content)
            return StanceClassification(stance=result.stance, prob=prob, distribution=distribution)

        except Exception as exc:
            logger.warning("Stance classification failed (%s: %s) — falling back to neutral", type(exc).__name__, exc)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, _FALLBACK.stance)
            span.set_attribute("error.message", str(exc))
            return _FALLBACK
