"""Milvus-backed vector index for candidates, manifestos, and party averages.

This module owns **both** halves of the index's lifecycle:

- Offline index construction: `build_index` runs once during FastAPI lifespan
  startup to build (or load) the three Milvus collections.
- Online query entry points: `query_candidates`, `query_manifestos`, and
  `query_party_candidate_averages` are the per-request pass-throughs into
  `rag.vector_store.search_*` that the Retriever and the eval suite call into.

Vectors live in Milvus (see `rag.vector_store`); known-name caches remain
in-memory because they're cheap to rebuild from the document loaders on startup.
"""
import asyncio
from typing import Dict, List, Optional

from config import settings
from logger import get_logger
from rag import vector_store
from rag.document import Document
from rag.embedder import embed_texts
from rag.party_documents_loader import load_manifesto_documents
from rag.platform_loader import load_platform_documents
from rag.question_info import load_additional_question_info

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Module-level caches (name lookups + metadata; vectors live in Milvus)
# ---------------------------------------------------------------------------

_candidate_meta: Dict[int, List[dict]] = {}       # question_id -> metadata list

_manifesto_meta: List[dict] = []

_known_names_lower: List[str] = []
_known_names_original: List[str] = []
_candidate_party_map: Dict[str, str] = {}  # canonical name -> party string


# ---------------------------------------------------------------------------
# Public startup function
# ---------------------------------------------------------------------------

async def build_index() -> None:
    """Build or load all vector indices. Called once during FastAPI lifespan startup."""
    global _manifesto_meta

    vector_store.connect()

    # Milvus-independent static data
    load_additional_question_info()

    if not settings.force_reindex and _all_collections_populated():
        logger.info("Milvus collections already populated — hydrating caches from Milvus (skipping loaders + embeddings).")
        vector_store.load_all_collections()
        _hydrate_caches_from_milvus()
        _populate_known_names()
        logger.info(
            "Index ready (warm). %d question groups, %d manifesto chunks, %d known candidate names.",
            len(_candidate_meta),
            len(_manifesto_meta),
            len(_known_names_original),
        )
        return

    # ---- Slow path ----
    loop = asyncio.get_event_loop()
    candidate_docs, party_avg_docs = await loop.run_in_executor(None, load_platform_documents)
    manifesto_docs = await loop.run_in_executor(None, load_manifesto_documents)

    # Group candidate docs by question_id for the in-memory cache and per-question upserts.
    candidates_by_question: Dict[int, List[Document]] = {}
    for doc in candidate_docs:
        qid = doc.metadata["question_id"]
        candidates_by_question.setdefault(qid, []).append(doc)
    for qid, q_docs in candidates_by_question.items():
        _candidate_meta[qid] = [{"text": d.text, **d.metadata} for d in q_docs]

    _manifesto_meta = [{"text": d.text, **d.metadata} for d in manifesto_docs]

    # Probe the embedder once to learn the vector dimension, then ensure collections.
    probe = await embed_texts(["dim-probe"])
    dim = len(probe[0])
    if settings.force_reindex:
        logger.info("force_reindex=True — dropping existing Milvus collections.")
        vector_store.drop_all()
    vector_store.ensure_collections(dim)

    await _upsert_candidates(candidates_by_question)
    await _upsert_manifestos(manifesto_docs)
    await _upsert_party_averages(party_avg_docs)

    _populate_known_names()
    logger.info(
        "Index ready (cold). %d question groups, %d manifesto chunks, %d known candidate names.",
        len(_candidate_meta),
        len(_manifesto_meta),
        len(_known_names_original),
    )


def _all_collections_populated() -> bool:
    """True iff all three Milvus collections exist and have data."""
    return (
        vector_store.count_total_candidates() > 0
        and vector_store.count_manifestos() > 0
        and vector_store.count_party_averages() > 0
    )


def _hydrate_caches_from_milvus() -> None:
    """Rebuild _candidate_meta and _manifesto_meta from Milvus rows (no source loaders)."""
    global _manifesto_meta

    _candidate_meta.clear()
    for row in vector_store.fetch_all_candidates():
        qid = row.get("question_id")
        if qid is None:
            continue
        _candidate_meta.setdefault(int(qid), []).append(row)

    _manifesto_meta = vector_store.fetch_all_manifestos()


# ---------------------------------------------------------------------------
# Public query functions
# ---------------------------------------------------------------------------

def query_candidates(
    question_index: int,
    query_vector: List[float],
    name_filter: Optional[str],
    top_k: int,
    only_with_comments: bool = False,
) -> List[dict]:
    """Dense search via Milvus, filtered server-side by question_id (+ optional name).

    When ``only_with_comments`` is set, restrict results to candidates whose
    position carries a free-text comment.
    """
    return vector_store.search_candidates(query_vector, question_index, name_filter, top_k, only_with_comments)


def query_manifestos(
    query_vector: List[float],
    party_filter: Optional[str],
    top_k: int,
) -> List[dict]:
    """Dense search via Milvus, filtered server-side by party."""
    return vector_store.search_manifestos(query_vector, party_filter, top_k)


def query_party_candidate_averages(
    question_index: int,
    query_vector: List[float],
    party_filter: Optional[str],
    top_k: int,
) -> List[dict]:
    """Dense search via Milvus, filtered server-side by question_id (+ optional party)."""
    return vector_store.search_party_averages(query_vector, question_index, party_filter, top_k)


def get_known_names_lower() -> List[str]:
    return _known_names_lower


def get_known_names_original() -> List[str]:
    return _known_names_original


def get_party_for_candidate(name: str) -> Optional[str]:
    """Return the party string for the given canonical candidate name, or None."""
    return _candidate_party_map.get(name)


# ---------------------------------------------------------------------------
# Build helpers
# ---------------------------------------------------------------------------

async def _upsert_candidates(by_question: Dict[int, List[Document]]) -> None:
    total = len(by_question)
    logger.info(
        "Embedding %d candidate documents across %d questions…",
        sum(len(docs) for docs in by_question.values()),
        total,
    )

    built = 0
    skipped = 0
    for step, (question_id, q_docs) in enumerate(sorted(by_question.items()), start=1):
        existing = vector_store.count_candidates(question_id)
        if existing > 0:
            skipped += 1
            logger.debug(
                "[%d/%d] Skipped question_id=%d (already %d rows in Milvus).",
                step, total, question_id, existing,
            )
            continue

        texts = [d.text for d in q_docs]
        embeddings = await embed_texts(texts)
        rows = [
            {**document.metadata, "text": document.text, "embedding": emb}
            for document, emb in zip(q_docs, embeddings)
        ]
        vector_store.upsert_candidates(rows)
        built += 1
        logger.info(
            "[%d/%d] Indexed question_id=%d (%d candidates) — %.0f%% done.",
            step, total, question_id, len(q_docs), step / total * 100,
        )

    logger.info("Candidate indexing complete: %d built, %d skipped.", built, skipped)


async def _upsert_manifestos(docs: List[Document]) -> None:
    if not docs:
        logger.warning("No manifesto documents produced — manifesto collection will be empty.")
        return

    if vector_store.count_manifestos() > 0:
        logger.info("Manifestos already populated in Milvus — skipping embedding.")
        return

    texts = [d.text for d in docs]
    embeddings = await embed_texts(texts)
    rows = [
        {**d.metadata, "text": d.text, "embedding": emb}
        for d, emb in zip(docs, embeddings)
    ]
    vector_store.upsert_manifestos(rows)
    logger.info("Indexed %d manifesto chunks.", len(docs))


async def _upsert_party_averages(docs: List[Document]) -> None:
    if not docs:
        logger.warning("No party-average documents produced — collection will be empty.")
        return

    if vector_store.count_party_averages() > 0:
        logger.info("Party averages already populated in Milvus — skipping embedding.")
        return

    texts = [d.text for d in docs]
    embeddings = await embed_texts(texts)
    rows = [
        {**d.metadata, "text": d.text, "embedding": emb}
        for d, emb in zip(docs, embeddings)
    ]
    vector_store.upsert_party_averages(rows)
    logger.info("Indexed %d party-average documents.", len(docs))


def _populate_known_names() -> None:
    global _known_names_lower, _known_names_original, _candidate_party_map

    names = set()
    for meta_list in _candidate_meta.values():
        for meta in meta_list:
            name = meta.get("name", "")
            party = meta.get("party", "")
            if name and name != "unknown":
                names.add(name)
                if party:
                    _candidate_party_map[name] = party

    # Sort descending by length so longer names are matched first
    sorted_names = sorted(names, key=len, reverse=True)
    _known_names_original = sorted_names
    _known_names_lower = [n.lower() for n in sorted_names]
