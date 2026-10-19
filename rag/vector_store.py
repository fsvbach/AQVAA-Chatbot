"""Milvus-backed vector store for candidate positions, manifestos, and party averages.

Three collections, all using FLAT index with IP metric. Vectors are L2-normalised
client-side before insert and search so IP equals cosine similarity (preserves the
scoring semantics of the previous FAISS IndexFlatIP setup).

Metadata filtering (question_id, name, party) is pushed down into Milvus filter
expressions so we no longer over-fetch and post-filter in Python.
"""
import json
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Optional

import numpy as np
from pymilvus import DataType, MilvusClient

from config import settings
from logger import get_logger

logger = get_logger(__name__)

_VECTOR_FIELD = "embedding"
_TEXT_MAX_LEN = 65535
_PARTY_MAX_LEN = 128
_NAME_MAX_LEN = 256
_CANDIDATE_ID_MAX_LEN = 64
_CATEGORY_MAX_LEN = 128
_FILENAME_MAX_LEN = 512
_SOURCE_MAX_LEN = 64
_BBOX_JSON_MAX_LEN = 1024


def _fit(value: str, max_len: int, field: str, *, truncate: bool = False) -> str:
    """Enforce a reasonable VARCHAR field's max length before insert.

    Raises ValueError on overflow so over-limit data surfaces. 
    Pass truncate=True for content fields where clipping is
    acceptable
    """
    if len(value) <= max_len:
        return value
    if truncate:
        logger.warning("Truncating %s: %d > %d chars.", field, len(value), max_len)
        return value[:max_len]
    raise ValueError(f"{field} exceeds max_length {max_len}: got {len(value)} chars ({value[:80]!r}...)")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------

_client: Optional[MilvusClient] = None


def connect() -> None:
    """Open the shared Milvus client. Idempotent.

    Supports both Milvus Lite (file path URI) and standalone server (http URI).
    For Lite, ensures the parent directory exists before opening the db file.
    """
    global _client
    if _client is not None:
        return
    uri = settings.milvus_uri
    if not uri.startswith(("http://", "https://", "unix:")):
        Path(uri).expanduser().parent.mkdir(parents=True, exist_ok=True)
    logger.info("Connecting to Milvus at %s", uri)
    _client = MilvusClient(uri=uri)


def disconnect() -> None:
    """Close the shared Milvus client and release its resources. Idempotent.

    For Milvus Lite this releases the single-process lock on the db directory, so
    another process (e.g. the chatbot, or a notebook) can open it. A subsequent
    call to connect()/_get_client() transparently reopens the client.
    """
    global _client
    if _client is None:
        return
    try:
        _client.close()
    finally:
        _client = None
    logger.info("Closed Milvus client.")


def _get_client() -> MilvusClient:
    if _client is None:
        connect()
    assert _client is not None
    return _client


_lite_lock = threading.Lock()


@contextmanager
def _serialized_if_lite() -> Iterator[None]:
    """Hold the global lock for the duration of a client call when on Milvus Lite."""
    if _is_lite():
        with _lite_lock:
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# Schemas
# ---------------------------------------------------------------------------

def _candidates_schema(dim: int):
    client = _get_client()
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name=_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
    schema.add_field(field_name="candidate_id", datatype=DataType.VARCHAR, max_length=_CANDIDATE_ID_MAX_LEN)
    schema.add_field(field_name="name", datatype=DataType.VARCHAR, max_length=_NAME_MAX_LEN)
    schema.add_field(field_name="party", datatype=DataType.VARCHAR, max_length=_PARTY_MAX_LEN)
    schema.add_field(field_name="question_id", datatype=DataType.INT64)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=_CATEGORY_MAX_LEN)
    schema.add_field(field_name="value", datatype=DataType.FLOAT)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=_SOURCE_MAX_LEN)
    schema.add_field(field_name="has_comment", datatype=DataType.BOOL)
    return schema


def _manifestos_schema(dim: int):
    client = _get_client()
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name=_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
    schema.add_field(field_name="party", datatype=DataType.VARCHAR, max_length=_PARTY_MAX_LEN)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=_SOURCE_MAX_LEN)
    schema.add_field(field_name="filename", datatype=DataType.VARCHAR, max_length=_FILENAME_MAX_LEN)
    schema.add_field(field_name="chunk_index", datatype=DataType.INT64)
    schema.add_field(field_name="page_no", datatype=DataType.INT64)  # -1 sentinel when missing
    schema.add_field(field_name="bbox_json", datatype=DataType.VARCHAR, max_length=_BBOX_JSON_MAX_LEN)
    schema.add_field(field_name="char_start", datatype=DataType.INT64)  # -1 sentinel when missing
    return schema


def _party_averages_schema(dim: int):
    client = _get_client()
    schema = client.create_schema(auto_id=True, enable_dynamic_field=False)
    schema.add_field(field_name="id", datatype=DataType.INT64, is_primary=True)
    schema.add_field(field_name=_VECTOR_FIELD, datatype=DataType.FLOAT_VECTOR, dim=dim)
    schema.add_field(field_name="text", datatype=DataType.VARCHAR, max_length=_TEXT_MAX_LEN)
    schema.add_field(field_name="party", datatype=DataType.VARCHAR, max_length=_PARTY_MAX_LEN)
    schema.add_field(field_name="question_id", datatype=DataType.INT64)
    schema.add_field(field_name="category", datatype=DataType.VARCHAR, max_length=_CATEGORY_MAX_LEN)
    schema.add_field(field_name="avg_value", datatype=DataType.FLOAT)
    schema.add_field(field_name="std", datatype=DataType.FLOAT)
    schema.add_field(field_name="n", datatype=DataType.INT64)
    schema.add_field(field_name="source", datatype=DataType.VARCHAR, max_length=_SOURCE_MAX_LEN)
    return schema


# ---------------------------------------------------------------------------
# Collection lifecycle
# ---------------------------------------------------------------------------

def _is_lite() -> bool:
    """True when settings.milvus_uri points at an embedded Lite file rather than a server."""
    return not settings.milvus_uri.startswith(("http://", "https://", "unix:"))


def _build_index_params(scalar_fields: List[str]):
    client = _get_client()
    params = client.prepare_index_params()
    params.add_index(field_name=_VECTOR_FIELD, index_type="FLAT", metric_type="IP")
    # Milvus Lite rejects scalar indexes (requires metric_type on every entry).
    # Filtering still works on Lite via sequential scan — fine at our data size.
    if not _is_lite():
        for field in scalar_fields:
            params.add_index(field_name=field, index_type="INVERTED")
    return params


def _ensure_collection(name: str, schema_fn, dim: int, scalar_indexes: List[str]) -> None:
    client = _get_client()
    if client.has_collection(name):
        client.load_collection(name)
        return

    logger.info("Creating Milvus collection %s (dim=%d)", name, dim)
    client.create_collection(
        collection_name=name,
        schema=schema_fn(dim),
        index_params=_build_index_params(scalar_indexes),
    )
    client.load_collection(name)


def _candidate_collection_name(question_id: int) -> str:
    """Per-question candidate collection name (e.g. 'candidates_q_32214')"""
    return f"{settings.milvus_candidates_collection}_q_{int(question_id)}"


def _list_candidate_collections() -> List[str]:
    """Every existing per-question candidate sub-collection."""
    client = _get_client()
    prefix = f"{settings.milvus_candidates_collection}_q_"
    return [name for name in client.list_collections() if name.startswith(prefix)]


def _ensure_candidate_collection(question_id: int, dim: int) -> str:
    """Lazy-create a per-question candidate collection. Returns its name."""
    name = _candidate_collection_name(question_id)
    _ensure_collection(
        name,
        _candidates_schema,
        dim,
        scalar_indexes=["name", "party"],  # question_id is constant per collection — no need to index
    )
    return name


def ensure_collections(dim: int) -> None:
    """Create (if missing), index, and load the global collections.

    Per-question candidate sub-collections are lazy-created on first upsert
    """
    _ensure_collection(
        settings.milvus_manifestos_collection,
        _manifestos_schema,
        dim,
        scalar_indexes=["party"],
    )
    _ensure_collection(
        settings.milvus_party_averages_collection,
        _party_averages_schema,
        dim,
        scalar_indexes=["question_id", "party"],
    )


def drop_all() -> None:
    """Drop manifestos, party_averages, every candidates_q_* sub-collection"""
    client = _get_client()

    # Per-question candidate sub-collections.
    for name in _list_candidate_collections():
        logger.info("Dropping Milvus collection %s", name)
        client.drop_collection(name)

    for name in (
        settings.milvus_manifestos_collection,
        settings.milvus_party_averages_collection,
    ):
        if client.has_collection(name):
            logger.info("Dropping Milvus collection %s", name)
            client.drop_collection(name)


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------

def _normalize(vectors: np.ndarray) -> np.ndarray:
    """L2-normalise rows in-place and return the array."""
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    vectors /= norms
    return vectors


def _normalize_one(vec: List[float]) -> List[float]:
    arr = np.array([vec], dtype=np.float32)
    _normalize(arr)
    return arr[0].tolist()


# ---------------------------------------------------------------------------
# Upserts
# ---------------------------------------------------------------------------

# To ensure insert payload stays under 64MB (gRPC receive limit)
_INSERT_BATCH_SIZE = 1000


def _insert_in_batches(client: MilvusClient, collection_name: str, data: List[dict]) -> None:
    """Insert rows in bounded batches, then flush once. Keeps each RPC under the limit."""
    for start in range(0, len(data), _INSERT_BATCH_SIZE):
        client.insert(collection_name=collection_name, data=data[start:start + _INSERT_BATCH_SIZE])
    client.flush(collection_name=collection_name)


def upsert_candidates(rows: List[dict]) -> None:
    """Insert candidate rows, routing each to its per-question sub-collection.

    Rows may span multiple question_ids; we group internally and create any
    missing sub-collection on the fly using the dim from the first embedding.
    """
    if not rows:
        return

    by_question: dict[int, List[int]] = {}
    for idx, r in enumerate(rows):
        by_question.setdefault(int(r["question_id"]), []).append(idx)

    vectors = np.array([r["embedding"] for r in rows], dtype=np.float32)
    _normalize(vectors)
    dim = int(vectors.shape[1])
    client = _get_client()

    for question_id, indices in by_question.items():
        collection_name = _ensure_candidate_collection(question_id, dim)
        data = [
            {
                _VECTOR_FIELD: vectors[i].tolist(),
                "text": _fit(rows[i]["text"], _TEXT_MAX_LEN, "text", truncate=True),
                "candidate_id": _fit(str(rows[i].get("candidate_id", "")), _CANDIDATE_ID_MAX_LEN, "candidate_id"),
                "name": _fit(str(rows[i].get("name", "")), _NAME_MAX_LEN, "name"),
                "party": _fit(str(rows[i].get("party", "")), _PARTY_MAX_LEN, "party"),
                "question_id": int(rows[i]["question_id"]),
                "category": _fit(str(rows[i].get("category", "")), _CATEGORY_MAX_LEN, "category"),
                "value": float(rows[i].get("value", 0.0)),
                "source": _fit(str(rows[i].get("source", "candidate_positions")), _SOURCE_MAX_LEN, "source"),
                "has_comment": bool(rows[i].get("has_comment", False)),
            }
            for i in indices
        ]
        _insert_in_batches(client, collection_name, data)


def upsert_manifestos(rows: List[dict]) -> None:
    if not rows:
        return
    vectors = np.array([r["embedding"] for r in rows], dtype=np.float32)
    _normalize(vectors)
    data = [
        {
            _VECTOR_FIELD: vec.tolist(),
            "text": _fit(r["text"], _TEXT_MAX_LEN, "text", truncate=True),
            "party": _fit(str(r.get("party", "")), _PARTY_MAX_LEN, "party"),
            "source": _fit(str(r.get("source", "party_manifesto")), _SOURCE_MAX_LEN, "source"),
            "filename": _fit(str(r.get("filename", "")), _FILENAME_MAX_LEN, "filename"),
            "chunk_index": int(r.get("chunk_index", 0)),
            "page_no": int(r["page_no"]) if r.get("page_no") is not None else -1,
            "bbox_json": _fit(json.dumps(r["bbox"]), _BBOX_JSON_MAX_LEN, "bbox_json") if r.get("bbox") is not None else "",
            "char_start": int(r["char_start"]) if r.get("char_start") is not None else -1,
        }
        for r, vec in zip(rows, vectors)
    ]
    client = _get_client()
    _insert_in_batches(client, settings.milvus_manifestos_collection, data)


def upsert_party_averages(rows: List[dict]) -> None:
    if not rows:
        return
    vectors = np.array([r["embedding"] for r in rows], dtype=np.float32)
    _normalize(vectors)
    data = [
        {
            _VECTOR_FIELD: vec.tolist(),
            "text": _fit(r["text"], _TEXT_MAX_LEN, "text", truncate=True),
            "party": _fit(str(r.get("party", "")), _PARTY_MAX_LEN, "party"),
            "question_id": int(r["question_id"]),
            "category": _fit(str(r.get("category", "")), _CATEGORY_MAX_LEN, "category"),
            "avg_value": float(r.get("avg_value", 0.0)),
            "std": float(r.get("std", 0.0)),
            "n": int(r.get("n", 0)),
            "source": _fit(str(r.get("source", "party_candidate_averages")), _SOURCE_MAX_LEN, "source"),
        }
        for r, vec in zip(rows, vectors)
    ]
    client = _get_client()
    _insert_in_batches(client, settings.milvus_party_averages_collection, data)


# ---------------------------------------------------------------------------
# Searches
# ---------------------------------------------------------------------------

_SEARCH_PARAMS = {"metric_type": "IP", "params": {}}


def _quote(value: str) -> str:
    """Escape double-quotes for Milvus filter string literals."""
    return value.replace('"', '\\"')


def _candidate_output_fields() -> List[str]:
    return ["text", "candidate_id", "name", "party", "question_id", "category", "value", "source", "has_comment"]


def _manifesto_output_fields() -> List[str]:
    return ["text", "party", "source", "filename", "chunk_index", "page_no", "bbox_json", "char_start"]


def _party_average_output_fields() -> List[str]:
    return ["text", "party", "question_id", "category", "avg_value", "std", "n", "source"]


def search_candidates(
    query_vec: List[float],
    question_id: int,
    name_filter: Optional[str],
    top_k: int,
    only_with_comments: bool = False,
) -> List[dict]:
    collection_name = _candidate_collection_name(question_id)
    client = _get_client()

    clauses = []
    if name_filter:
        clauses.append(f'name == "{_quote(name_filter)}"')
    if only_with_comments:
        clauses.append("has_comment == true")
    filter_expr = " and ".join(clauses)

    with _serialized_if_lite():
        if not client.has_collection(collection_name):
            return []
        res = client.search(
            collection_name=collection_name,
            data=[_normalize_one(query_vec)],
            anns_field=_VECTOR_FIELD,
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            filter=filter_expr,
            output_fields=_candidate_output_fields(),
        )
    return _hits_to_dicts(res)


def search_manifestos(
    query_vec: List[float],
    party_filter: Optional[str],
    top_k: int,
) -> List[dict]:
    filter_expr = f'party == "{_quote(party_filter)}"' if party_filter else ""

    client = _get_client()
    with _serialized_if_lite():
        res = client.search(
            collection_name=settings.milvus_manifestos_collection,
            data=[_normalize_one(query_vec)],
            anns_field=_VECTOR_FIELD,
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            filter=filter_expr,
            output_fields=_manifesto_output_fields(),
        )
    rows = _hits_to_dicts(res)
    for row in rows:
        bbox_json = row.pop("bbox_json", "")
        page_no = row.get("page_no")
        char_start = row.get("char_start")
        if bbox_json:
            try:
                row["bbox"] = json.loads(bbox_json)
            except json.JSONDecodeError:
                pass
        if page_no == -1:
            row.pop("page_no", None)
        if char_start == -1:
            row.pop("char_start", None)
    return rows


def search_party_averages(
    query_vec: List[float],
    question_id: int,
    party_filter: Optional[str],
    top_k: int,
) -> List[dict]:
    filter_expr = f"question_id == {int(question_id)}"
    if party_filter:
        filter_expr += f' and party == "{_quote(party_filter)}"'

    client = _get_client()
    with _serialized_if_lite():
        res = client.search(
            collection_name=settings.milvus_party_averages_collection,
            data=[_normalize_one(query_vec)],
            anns_field=_VECTOR_FIELD,
            search_params=_SEARCH_PARAMS,
            limit=top_k,
            filter=filter_expr,
            output_fields=_party_average_output_fields(),
        )
    return _hits_to_dicts(res)


def _hits_to_dicts(search_result) -> List[dict]:
    """Convert a MilvusClient search result (single query) into a list of metadata dicts."""
    rows = []
    if not search_result:
        return rows
    for hit in search_result[0]:
        # MilvusClient returns each hit as {"id": ..., "distance": ..., "entity": {...}}
        entity = hit.get("entity", {}) if isinstance(hit, dict) else {}
        row = dict(entity)
        row["score"] = float(hit["distance"]) if isinstance(hit, dict) else 0.0
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------

def count_candidates(question_id: int) -> int:
    """Row count for a single question's candidate sub-collection (0 if missing)."""
    client = _get_client()
    name = _candidate_collection_name(question_id)
    if not client.has_collection(name):
        return 0
    stats = client.get_collection_stats(collection_name=name)
    return int(stats.get("row_count", 0))


def count_total_candidates() -> int:
    """Total candidate rows across all per-question sub-collections."""
    client = _get_client()
    total = 0
    for name in _list_candidate_collections():
        stats = client.get_collection_stats(collection_name=name)
        total += int(stats.get("row_count", 0))
    return total


def count_manifestos() -> int:
    client = _get_client()
    if not client.has_collection(settings.milvus_manifestos_collection):
        return 0
    stats = client.get_collection_stats(collection_name=settings.milvus_manifestos_collection)
    return int(stats.get("row_count", 0))


def count_party_averages() -> int:
    client = _get_client()
    if not client.has_collection(settings.milvus_party_averages_collection):
        return 0
    stats = client.get_collection_stats(collection_name=settings.milvus_party_averages_collection)
    return int(stats.get("row_count", 0))


# ---------------------------------------------------------------------------
# Bulk reads (used at warm startup to rebuild in-memory caches without re-loading sources)
# ---------------------------------------------------------------------------

def load_all_collections() -> None:
    """Load all collections so they're queryable. No-op for any that don't exist."""
    client = _get_client()
    candidate_collections = _list_candidate_collections()
    logger.info(
        "Loading collections: %d candidate sub-collections + manifestos + party_averages",
        len(candidate_collections),
    )
    for name in candidate_collections:
        client.load_collection(name)
    for name in (
        settings.milvus_manifestos_collection,
        settings.milvus_party_averages_collection,
    ):
        if client.has_collection(name):
            client.load_collection(name)


def _fetch_all(collection_name: str, output_fields: List[str], batch_size: int = 16384) -> List[dict]:
    """Cursor-based iteration over every row. Avoids offset-based O(N²) scans."""
    client = _get_client()
    if not client.has_collection(collection_name):
        return []
    rows: List[dict] = []
    iterator = client.query_iterator(
        collection_name=collection_name,
        filter="",
        output_fields=output_fields,
        batch_size=batch_size,
    )
    try:
        while True:
            chunk = iterator.next()
            if not chunk:
                break
            rows.extend(chunk)
    finally:
        iterator.close()
    return rows


def fetch_all_candidates() -> List[dict]:
    """Return every candidate row's metadata across all per-question sub-collections."""
    sub_collections = _list_candidate_collections()
    logger.info("Fetching candidates from %d sub-collections...", len(sub_collections))
    rows: List[dict] = []
    for name in sub_collections:
        rows.extend(_fetch_all(name, _candidate_output_fields()))
    return rows


def fetch_all_manifestos() -> List[dict]:
    """Return every manifesto chunk's metadata, restoring bbox from bbox_json."""
    logger.info("Fetching all manifestos...")
    rows = _fetch_all(settings.milvus_manifestos_collection, _manifesto_output_fields())
    for row in rows:
        bbox_json = row.pop("bbox_json", "")
        if bbox_json:
            try:
                row["bbox"] = json.loads(bbox_json)
            except json.JSONDecodeError:
                pass
        if row.get("page_no") == -1:
            row.pop("page_no", None)
        if row.get("char_start") == -1:
            row.pop("char_start", None)
    return rows
