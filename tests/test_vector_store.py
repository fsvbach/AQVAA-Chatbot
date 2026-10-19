"""Integration tests for the Milvus-backed vector_store.

Requires a running Milvus instance reachable at LLM_MILVUS_URI (default
http://localhost:19530). Bring it up with:

    docker compose -f tools/docker-compose/milvus.docker-compose.yml \
        --profile milvus up -d

Run with:
    pytest tests/test_vector_store.py -m integration
"""
import os
import sys
import uuid

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from config import settings  # noqa: E402
from rag import vector_store  # noqa: E402

pytestmark = pytest.mark.integration

DIM = 32


def _rand_vec(seed: int) -> list:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(DIM).astype(np.float32)
    return v.tolist()


@pytest.fixture(scope="module", autouse=True)
def _milvus_collections():
    """Connect, isolate this test run in dedicated collections, drop on exit."""
    suffix = uuid.uuid4().hex[:8]
    settings.milvus_candidates_collection = f"test_candidates_{suffix}"
    settings.milvus_manifestos_collection = f"test_manifestos_{suffix}"
    settings.milvus_party_averages_collection = f"test_party_averages_{suffix}"

    vector_store.connect()
    vector_store.drop_all()
    vector_store.ensure_collections(DIM)
    yield
    vector_store.drop_all()


def test_ensure_collections_is_idempotent():
    vector_store.ensure_collections(DIM)
    vector_store.ensure_collections(DIM)


def test_candidate_upsert_and_search_round_trip():
    rows = [
        {
            "embedding": _rand_vec(1),
            "text": "Alice on healthcare",
            "candidate_id": "c1",
            "name": "Alice",
            "party": "P1",
            "question_id": 10,
            "category": "Health",
            "value": 1.0,
            "source": "candidate_positions",
        },
        {
            "embedding": _rand_vec(2),
            "text": "Bob on healthcare",
            "candidate_id": "c2",
            "name": "Bob",
            "party": "P2",
            "question_id": 10,
            "category": "Health",
            "value": 0.0,
            "source": "candidate_positions",
        },
        {
            "embedding": _rand_vec(3),
            "text": "Alice on retirement",
            "candidate_id": "c1",
            "name": "Alice",
            "party": "P1",
            "question_id": 20,
            "category": "Pensions",
            "value": 0.5,
            "source": "candidate_positions",
        },
    ]
    vector_store.upsert_candidates(rows)

    assert vector_store.count_candidates(10) == 2
    assert vector_store.count_candidates(20) == 1

    results = vector_store.search_candidates(_rand_vec(99), question_id=10, name_filter=None, top_k=5)
    assert len(results) == 2
    assert {r["name"] for r in results} == {"Alice", "Bob"}
    assert all(-1.0001 <= r["score"] <= 1.0001 for r in results)


def test_candidate_question_id_filter_restricts_results():
    results = vector_store.search_candidates(_rand_vec(7), question_id=20, name_filter=None, top_k=5)
    assert len(results) == 1
    assert results[0]["question_id"] == 20
    assert results[0]["name"] == "Alice"


def test_candidate_name_filter_restricts_results():
    results = vector_store.search_candidates(_rand_vec(8), question_id=10, name_filter="Alice", top_k=5)
    assert len(results) == 1
    assert results[0]["name"] == "Alice"


def test_candidate_only_with_comments_filter():
    rows = [
        {
            "embedding": _rand_vec(11),
            "text": "Cara on healthcare (with comment)",
            "candidate_id": "c-cara",
            "name": "Cara",
            "party": "P1",
            "question_id": 30,
            "category": "Health",
            "value": 1.0,
            "source": "candidate_positions",
            "has_comment": True,
        },
        {
            "embedding": _rand_vec(12),
            "text": "Dan on healthcare (no comment)",
            "candidate_id": "c-dan",
            "name": "Dan",
            "party": "P2",
            "question_id": 30,
            "category": "Health",
            "value": 0.0,
            "source": "candidate_positions",
            "has_comment": False,
        },
    ]
    vector_store.upsert_candidates(rows)

    # No flag: both candidates returned, has_comment surfaced in the payload.
    res_all = vector_store.search_candidates(_rand_vec(50), question_id=30, name_filter=None, top_k=5)
    assert {r["name"] for r in res_all} == {"Cara", "Dan"}
    assert {r["name"]: r["has_comment"] for r in res_all} == {"Cara": True, "Dan": False}

    # Comments only: Dan (no comment) is excluded.
    res_comm = vector_store.search_candidates(
        _rand_vec(50), question_id=30, name_filter=None, top_k=5, only_with_comments=True
    )
    assert {r["name"] for r in res_comm} == {"Cara"}
    assert all(r["has_comment"] for r in res_comm)

    # Filters compose with AND: Dan has no comment, so name+comments yields nothing.
    res_dan = vector_store.search_candidates(
        _rand_vec(50), question_id=30, name_filter="Dan", top_k=5, only_with_comments=True
    )
    assert res_dan == []


def test_candidate_self_retrieval_scores_near_one():
    target = _rand_vec(42)
    vector_store.upsert_candidates([
        {
            "embedding": target,
            "text": "Self-retrieval target",
            "candidate_id": "self",
            "name": "SelfTarget",
            "party": "P1",
            "question_id": 999,
            "category": "Test",
            "value": 0.5,
            "source": "candidate_positions",
        }
    ])
    results = vector_store.search_candidates(target, question_id=999, name_filter=None, top_k=1)
    assert len(results) == 1
    assert results[0]["name"] == "SelfTarget"
    assert results[0]["score"] == pytest.approx(1.0, abs=1e-3)


def test_manifesto_party_filter():
    vector_store.upsert_manifestos([
        {
            "embedding": _rand_vec(11),
            "text": "FDP manifesto chunk",
            "party": "FDP",
            "source": "party_manifesto",
            "filename": "fdp.pdf",
            "chunk_index": 0,
            "page_no": 1,
            "bbox": {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.8},
        },
        {
            "embedding": _rand_vec(12),
            "text": "SVP manifesto chunk",
            "party": "SVP",
            "source": "party_manifesto",
            "filename": "svp.pdf",
            "chunk_index": 0,
            "page_no": 2,
            "bbox": None,
        },
    ])

    fdp_only = vector_store.search_manifestos(_rand_vec(13), party_filter="FDP", top_k=5)
    assert len(fdp_only) == 1
    assert fdp_only[0]["party"] == "FDP"
    assert fdp_only[0].get("bbox") == {"l": 0.1, "t": 0.2, "r": 0.9, "b": 0.8}

    unfiltered = vector_store.search_manifestos(_rand_vec(14), party_filter=None, top_k=5)
    assert len(unfiltered) == 2


def test_party_averages_filters():
    vector_store.upsert_party_averages([
        {
            "embedding": _rand_vec(21),
            "text": "FDP average on Q10",
            "party": "FDP",
            "question_id": 10,
            "category": "Health",
            "avg_value": 0.75,
            "std": 0.1,
            "n": 42,
            "source": "party_candidate_averages",
        },
        {
            "embedding": _rand_vec(22),
            "text": "SVP average on Q10",
            "party": "SVP",
            "question_id": 10,
            "category": "Health",
            "avg_value": 0.25,
            "std": 0.2,
            "n": 50,
            "source": "party_candidate_averages",
        },
        {
            "embedding": _rand_vec(23),
            "text": "FDP average on Q20",
            "party": "FDP",
            "question_id": 20,
            "category": "Pensions",
            "avg_value": 0.5,
            "std": 0.0,
            "n": 40,
            "source": "party_candidate_averages",
        },
    ])

    q10_all = vector_store.search_party_averages(_rand_vec(31), question_id=10, party_filter=None, top_k=5)
    assert len(q10_all) == 2
    assert {r["party"] for r in q10_all} == {"FDP", "SVP"}

    q10_fdp = vector_store.search_party_averages(_rand_vec(32), question_id=10, party_filter="FDP", top_k=5)
    assert len(q10_fdp) == 1
    assert q10_fdp[0]["party"] == "FDP"
    assert q10_fdp[0]["question_id"] == 10


def test_concurrent_searches_do_not_crash_channel():
    """Regression: parallel searches over the shared client must not corrupt the
    Milvus Lite gRPC channel ("Cannot invoke RPC on closed channel!").

    Mirrors the retriever's fan-out (one search per datasource via threads). On
    Lite the _serialized_if_lite lock keeps these from interleaving on the single
    channel; on a real server they run concurrently and this still passes.
    """
    from concurrent.futures import ThreadPoolExecutor

    # Relies on collections populated by the manifesto + party-average tests above.
    def _mixed_search(i: int):
        kind = i % 3
        if kind == 0:
            return vector_store.search_manifestos(_rand_vec(1000 + i), party_filter=None, top_k=5)
        if kind == 1:
            return vector_store.search_party_averages(_rand_vec(1000 + i), question_id=10, party_filter=None, top_k=5)
        return vector_store.search_candidates(_rand_vec(1000 + i), question_id=10, name_filter=None, top_k=5)

    n = 48
    with ThreadPoolExecutor(max_workers=16) as pool:
        results = list(pool.map(_mixed_search, range(n)))

    # No exception escaped, and every call returned a list of hits.
    assert len(results) == n
    assert all(isinstance(r, list) for r in results)
