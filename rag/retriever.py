"""Retrieval orchestration.

Runs guardrail check, query embedding, and datasource classification concurrently.
If the guardrail is violated, returns a dict with guardrail_violated=True (including reason).
Otherwise, executes the datasource queries specified by the RetrievalPlan in parallel.
"""
import asyncio
from typing import Dict, List, Literal, Optional

import numpy as np
from openinference.semconv.trace import SpanAttributes

from rag.rewriter import get_contextualized_user_query
from config import settings
from logger import get_logger, get_tracer
from models.schemas import  ChatbotMode, DatasourceQuery, ExplainRequest, RetrievalPlan
from rag import indexer
from rag.query_router import infer_relevant_datasources, infer_study_context_needed
from miscellaneous.study_context import get_study_context
from rag.embedder import embed_query
from rag.guardrail import check_for_guardrail_violation
from rag.ranking import bucket_by_party, select_manifestos, _merge_sorted

logger = get_logger(__name__)
_tracer = get_tracer(__name__)


async def retrieve(request: ExplainRequest, mode: ChatbotMode) -> dict:
    """Return retrieved context for a given ExplainRequest.

    Returns one of:
      {"guardrail_violated": True, "guardrail_category": "..."}                  — query failed guardrail check
      {}                                             — empty plan, no retrieval
      {"candidate_docs": [...],
       "party_candidate_averages_docs": [...],
       "manifesto_docs": [...],
       "plan": RetrievalPlan}                        — normal retrieval result
    """
    user_query = request.user_query
    # Always use German question text for better retrieval, since all documents are written in German.
    question_text = request.question_text_de 
    chat_history = request.chat_history

    with _tracer.start_as_current_span("rag.retriever") as span:
        span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, "RETRIEVER")

        contextualized_user_query = await get_contextualized_user_query(user_query, question_text, chat_history, mode)
        
        span.set_attribute(SpanAttributes.INPUT_VALUE, f"Statement: {question_text} \n\n User Query: {user_query} \n\nRewritten:{contextualized_user_query} " )

        guardrail, query_vec, plan, survey_info_needed = await asyncio.gather(
            check_for_guardrail_violation(question_text, user_query, chat_history),
            embed_query(contextualized_user_query),
            infer_relevant_datasources(contextualized_user_query, question_text, known_names=indexer.get_known_names_original()),
            infer_study_context_needed(contextualized_user_query)
        )

        if guardrail.violated:
            logger.info("Guardrail violated — skipping retrieval.")
            span.set_attribute("guardrail.violated", True)
            return {"guardrail_violated": True, "guardrail_category": guardrail.category}

        study_context = get_study_context(mode) if survey_info_needed else None
        
        if not plan.datasources:
            if study_context:
                logger.info("Study context only")
                return {"study_context": study_context}
            
            logger.info("Empty retrieval plan — LLM will answer from training data.")
            return {}

        logger.info(
            "Retrieval plan: %s",
            [(d.index, d.party_filter, d.name_filter) for d in plan.datasources],
        )

        
        context = await _execute_plan(request.question_index, query_vec, plan)
        candidate_docs = context["candidate_docs"]
        party_docs = context["party_candidate_averages_docs"]
        manifesto_docs = context["manifesto_docs"]
        all_docs = candidate_docs + party_docs + manifesto_docs
        logger.info(
            "Retrieved %d candidate docs, %d party-candidate-averages docs, %d manifesto docs.",
            len(candidate_docs),
            len(party_docs),
            len(manifesto_docs),
        )

        for i, doc in enumerate(all_docs):
            span.set_attribute(f"retrieval.documents.{i}.document.content", doc.get("text", ""))
            span.set_attribute(f"retrieval.documents.{i}.document.score", float(doc.get("score", 0.0)))

        return {**context, "study_context": study_context ,"plan": plan}


# ---------------------------------------------------------------------------
# Plan execution
# ---------------------------------------------------------------------------

def _run(loop: asyncio.AbstractEventLoop, fn, *args):
    """Wrap a synchronous retrieval call in an executor for concurrent dispatch."""
    return loop.run_in_executor(None, lambda: fn(*args))

async def _execute_plan(
    question_index: int,
    query_vec: List[float],
    plan: RetrievalPlan,
    eval_top_k: Optional[int] = None,
) -> Dict[str, list]:
    """Dispatch all datasource queries from the plan concurrently and merge results."""
    loop = asyncio.get_event_loop()
    top_k = settings.rag_top_k if eval_top_k is None else eval_top_k
    manifesto_k = settings.rag_manifesto_top_k if eval_top_k is None else eval_top_k
    party_avg_k = settings.rag_party_averages_top_k if eval_top_k is None else eval_top_k

    candidate_tasks = []
    party_avg_tasks = []
    manifesto_tasks = []

    for ds in plan.datasources:
        if ds.index == "candidates":
            for name in (ds.name_filter or [None]):
                only_with_comments = name is None
                candidate_tasks.append(
                    _run(loop, indexer.query_candidates, question_index, query_vec, name, top_k, only_with_comments)
                )
        elif ds.index == "party_candidate_averages":
            for party in (ds.party_filter or [None]):
                party_avg_tasks.append(
                    _run(loop, indexer.query_party_candidate_averages, question_index, query_vec, party, party_avg_k)
                )

        elif ds.index == "manifestos":
            if ds.party_filter:
                # Filtered: one search per named party (each result is a single-party bucket).
                for party in ds.party_filter:
                    manifesto_tasks.append(
                        _run(loop, indexer.query_manifestos, query_vec, party, manifesto_k)
                    )
            else:
                # Unfiltered: one enlarged search, split into per-party buckets at merge time.
                fetch_k = manifesto_k * settings.rag_manifesto_fetch_multiplier
                manifesto_tasks.append(
                    _run(loop, indexer.query_manifestos, query_vec, None, fetch_k)
                )

    all_tasks = (
        candidate_tasks
        + party_avg_tasks
        + manifesto_tasks
    )
    results = await asyncio.gather(*all_tasks)

    n_cf = len(candidate_tasks)
    n_pa = len(party_avg_tasks)

    offset = 0
    candidate_results = results[offset:offset + n_cf]; offset += n_cf
    party_avg_results       = results[offset:offset + n_pa]; offset += n_pa
    manifesto_results       = results[offset:]

    candidate_docs = _merge_sorted(candidate_results)[:top_k]
    party_docs     = _merge_sorted(party_avg_results)[:party_avg_k]

    manifesto_buckets = []
    for res in manifesto_results:
        manifesto_buckets += bucket_by_party(res)
    manifesto_docs = select_manifestos(
        manifesto_buckets, manifesto_k, settings.rag_manifesto_score_threshold
    )

    return {
        "candidate_docs": candidate_docs,
        "party_candidate_averages_docs": party_docs,
        "manifesto_docs": manifesto_docs,
    }


