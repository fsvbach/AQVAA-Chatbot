"""LLM-based datasource router for the RAG retrieval pipeline.

The LLM returns a list of datasource queries (0–n) with optional metadata filters. 
An empty list means no retrieval is needed and the LLM should answer from its training data.
"""
import asyncio
import difflib
from typing import List, Optional

from pydantic import field_validator

from openinference.semconv.trace import SpanAttributes
from opentelemetry.trace import StatusCode

from llm.factory import get_client
from config import settings, LLMRole
from logger import get_logger, get_tracer
from models.schemas import DatasourceQuery, RetrievalPlan, SurveyInfoDecision
from prompts.classifier_prompts import CURRENT as _CLASSIFIER_SYSTEM_PROMPT
from prompts.study_context_prompt import CURRENT as _STUDY_CONTEXT_SYSTEM_PROMPT
from rag.parties import CANONICAL_PARTIES

logger = get_logger(__name__)
_tracer = get_tracer(__name__)

# Fallback: query all three datasources without filters
FALLBACK = RetrievalPlan(
    datasources=[
        DatasourceQuery(index="candidates"),
        DatasourceQuery(index="party_candidate_averages"),
        DatasourceQuery(index="manifestos"),
    ]
)


_CLASSIFIER_TIMEOUT = 60.0


async def infer_relevant_datasources(
    user_query: str,
    question_text: str,
    known_names: List[str],
    system_prompt: str | None = None,
) -> RetrievalPlan:
    """Infer which datasources to query using the LLM.

    Returns a RetrievalPlan. Falls back to FALLBACK (all datasources, no filters)
    on any error.
    """
    with _tracer.start_as_current_span("classifier.infer_datasources") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, settings.profile(LLMRole.QUERY_ROUTING).model)
        span.set_attribute(SpanAttributes.INPUT_VALUE, f"Topic: {question_text}\nQuery: {user_query}")
        try:
            client = get_client(LLMRole.QUERY_ROUTING)
            async with asyncio.timeout(_CLASSIFIER_TIMEOUT):
                response = await client.complete(
                    messages=[
                        {"role": "system", "content": system_prompt or _CLASSIFIER_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Topic question: {question_text}\nUser query: {user_query}",
                        },
                    ],
                    options={"temperature": 0.0, "reasoning": False},
                    format=RetrievalPlan.model_json_schema(),
                )

            if not response.content:
                raise ValueError("Classifier LLM returned empty response")

            plan = RetrievalPlan.model_validate_json(response.content)
            plan = _normalize_party_filters(plan)
            plan = _resolve_candidate_names(plan, known_names)

            logger.info(
                "Classifier result: %d datasources — %s",
                len(plan.datasources),
                [(d.index, d.party_filter, d.name_filter) for d in plan.datasources],
            )
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content)
            span.set_attribute("classifier.datasources", str([(d.index, d.party_filter, d.name_filter) for d in plan.datasources]))
            return plan

        except Exception as exc:
            logger.warning(
                "Classifier failed (%s: %s) — falling back to full retrieval",
                type(exc).__name__,
                exc,
            )
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, description=type(exc).__name__)
            span.set_attribute("classifier.fallback", True)
            return FALLBACK


def _normalize_party_filters(plan: RetrievalPlan) -> RetrievalPlan:
    """Drop redundant party_filters that cover the whole party index.

    The router occasionally enumerates every canonical party in a party_filter
    instead of omitting it, which is semantically "all parties" but forces a
    pointless metadata filter on retrieval. If a filter covers all canonical
    parties (after de-duping and resolving exact casing), strip it so retrieval
    runs unfiltered over the full index.
    """
    normalized = []
    for ds in plan.datasources:
        party_filter = ds.party_filter
        if party_filter:
            wanted = {p for p in party_filter if p in CANONICAL_PARTIES}
            if wanted >= CANONICAL_PARTIES:
                logger.info(
                    "Dropping all-parties party_filter on '%s' — retrieving unfiltered",
                    ds.index,
                )
                ds = DatasourceQuery(
                    index=ds.index,
                    party_filter=None,
                    name_filter=ds.name_filter,
                )
        normalized.append(ds)
    return RetrievalPlan(datasources=normalized)


def _resolve_candidate_names(plan: RetrievalPlan, known_names: List[str]) -> RetrievalPlan:
    """Resolve any name_filter values against the canonical known names list."""
    resolved = []
    for ds in plan.datasources:
        if ds.name_filter:
            canonical = []
            for raw in ds.name_filter:
                name = _resolve_candidate_name(raw, known_names)
                if name:
                    canonical.append(name)
            ds = DatasourceQuery(
                index=ds.index,
                party_filter=ds.party_filter,
                name_filter=canonical or None,
            )
        resolved.append(ds)
    return RetrievalPlan(datasources=resolved)


def _resolve_candidate_name(raw: str, known_names: List[str]) -> Optional[str]:
    """Match a raw LLM-produced name against the list of known candidate names.

    Tries substring match first (known_names is pre-sorted longest-first),
    then falls back to difflib fuzzy match.
    """
    raw_lower = raw.lower()
    known_lower = [n.lower() for n in known_names]

    for name, name_lower in zip(known_names, known_lower):
        if raw_lower in name_lower or name_lower in raw_lower:
            return name

    matches = difflib.get_close_matches(raw_lower, known_lower, n=1, cutoff=0.75)
    if matches:
        return known_names[known_lower.index(matches[0])]

    logger.warning("Classifier returned unknown candidate name '%s' — ignoring filter", raw)
    return None

# Placeholder system prompt — to be replaced with a proper prompt.


async def infer_study_context_needed(contextualized_user_query: str) -> bool:
    """Decide whether the user's query needs survey background information.

    Returns True if background info about the survey (question count, topics,
    current position, etc.) should be surfaced. Falls back to False on any error.
    """
    with _tracer.start_as_current_span("classifier.infer_survey_info_needed") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "LLM")
        span.set_attribute(SpanAttributes.LLM_MODEL_NAME, settings.profile(LLMRole.QUERY_ROUTING).model)
        span.set_attribute(SpanAttributes.INPUT_VALUE, contextualized_user_query)
        try:
            client = get_client(LLMRole.QUERY_ROUTING)
            async with asyncio.timeout(_CLASSIFIER_TIMEOUT):
                response = await client.complete(
                    messages=[
                        {"role": "system", "content": _STUDY_CONTEXT_SYSTEM_PROMPT},
                        {"role": "user", "content": contextualized_user_query},
                    ],
                    options={"temperature": 0.0, "reasoning": False},
                    format=SurveyInfoDecision.model_json_schema(),
                )

            if not response.content:
                raise ValueError("Survey-info classifier LLM returned empty response")

            decision = SurveyInfoDecision.model_validate_json(response.content)
            logger.info("Survey-info decision: needed=%s (%s)", decision.needed, decision.reason)
            span.set_attribute(SpanAttributes.OUTPUT_VALUE, response.content)
            span.set_attribute("classifier.survey_info_needed", decision.needed)
            return decision.needed

        except Exception as exc:
            logger.warning(
                "Survey-info classifier failed (%s: %s) — defaulting to no survey info",
                type(exc).__name__,
                exc,
            )
            span.record_exception(exc)
            span.set_status(StatusCode.ERROR, description=type(exc).__name__)
            span.set_attribute("classifier.fallback", True)
            return False
