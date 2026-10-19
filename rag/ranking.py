"""Pure ranking / merge helpers for retrieval result lists.

Kept dependency-light (no Milvus/LLM imports) so the merge and selection logic
can be unit-tested in isolation from the async retriever.
"""
from typing import List


def _doc_key(doc: dict) -> tuple:
    return (doc.get("name"), doc.get("party"), doc.get("question_id"), doc.get("text", "")[:80])


def _merge_sorted(result_lists) -> list:
    """Flatten a list of doc lists, deduplicating by identity, sorted by score descending."""
    seen = set()
    merged = []
    for docs in result_lists:
        for doc in docs:
            key = _doc_key(doc)
            if key not in seen:
                seen.add(key)
                merged.append(doc)
    merged.sort(key=lambda d: d.get("score", 0.0), reverse=True)
    return merged


def bucket_by_party(docs: List[dict]) -> List[List[dict]]:
    """Group a relevance-sorted list of docs into per-party lists (first-seen order)."""
    buckets: dict = {}
    order: list = []
    for doc in docs:
        party = doc.get("party")
        if party not in buckets:
            buckets[party] = []
            order.append(party)
        buckets[party].append(doc)
    return [buckets[party] for party in order]


def select_manifestos(buckets: List[List[dict]], k: int, threshold: float) -> List[dict]:
    """Round-robin across per-party buckets up to ``k`` docs total.

    Each bucket is one party's relevance-sorted chunks. Docs whose score is at or
    below ``threshold`` are dropped first. Parties are visited most-relevant first
    (by their top chunk); one chunk per party per pass until ``k`` is reached or
    all buckets are exhausted.
    """
    buckets = [[d for d in b if d.get("score", 0.0) > threshold] for b in buckets]
    buckets = [b for b in buckets if b]
    buckets.sort(key=lambda b: b[0].get("score", 0.0), reverse=True)

    selected: List[dict] = []
    rank = 0
    while len(selected) < k:
        took = False
        for bucket in buckets:
            if rank < len(bucket):
                selected.append(bucket[rank])
                took = True
                if len(selected) >= k:
                    break
        if not took:
            break
        rank += 1
    return selected
