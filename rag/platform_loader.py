"""Candidate positions from the AQVAA platform.

The platform's database is the single source of the smartvote data (questions,
candidates, their reactions and comments). At index time the chatbot fetches one
reaction batch through the backend's export endpoint and turns it into two
document sets: one per (candidate, question) and one per (party, question).
"""
import time
from typing import Dict, List, Optional, Tuple

import httpx
import pandas as pd

from config import settings
from logger import get_logger
from rag.document import Document

logger = get_logger(__name__)

_EXPORT_PATH = "/api/llm/reaction-batches/{id}/export"
_TOKEN_HEADER = "X-LLM-Service-Token"

# The platform may still be starting when the chatbot comes up.
_CONNECT_ATTEMPTS = 30
_CONNECT_RETRY_SECONDS = 2

# Map numeric reaction scores to human-readable labels
_REACTION_LABELS = {
    1.0: "Stimme zu",
    0.75: "Stimme eher zu",
    0.5: "Neutral",
    0.25: "Stimme eher nicht zu",
    0.0: "Stimme nicht zu",
}


def load_platform_documents() -> Tuple[List[Document], List[Document]]:
    """Return (candidate-position documents, party-average documents).

    Both lists are empty when no reaction batch is configured.
    """
    data = _fetch_reaction_batch()
    if data is None:
        logger.warning(
            "LLM_REACTION_BATCH_ID is not set — candidate and party-average indices stay empty."
        )
        return [], []

    reactions, candidates, questions = _to_tables(data)
    return (
        _candidate_documents(reactions, candidates, questions),
        _party_average_documents(reactions, candidates, questions),
    )


def _fetch_reaction_batch() -> Optional[dict]:
    """GET the configured reaction batch export from the platform backend."""
    if settings.reaction_batch_id is None:
        return None

    url = settings.platform_url.rstrip("/") + _EXPORT_PATH.format(id=settings.reaction_batch_id)
    headers = {_TOKEN_HEADER: settings.service_token}

    for attempt in range(1, _CONNECT_ATTEMPTS + 1):
        try:
            response = httpx.get(url, headers=headers, timeout=120.0)
            break
        except httpx.TransportError as exc:
            if attempt == _CONNECT_ATTEMPTS:
                raise
            logger.info("Platform not reachable at %s (%s) — retrying in %ds…", url, exc, _CONNECT_RETRY_SECONDS)
            time.sleep(_CONNECT_RETRY_SECONDS)

    response.raise_for_status()
    data = response.json()
    logger.info(
        "Fetched reaction batch %d from the platform: %d questions, %d candidates, %d reactions.",
        settings.reaction_batch_id,
        len(data["questions"]),
        len(data["candidates"]),
        len(data["reactions"]),
    )
    return data


def _to_tables(data: dict) -> Tuple[pd.DataFrame, Dict[str, dict], Dict[int, dict]]:
    """Split the export into a long reactions table plus candidate and question lookups."""
    questions: Dict[int, dict] = {}
    for q in data["questions"]:
        try:
            q_id = int(q["index"])
        except ValueError:
            raise ValueError(
                f"Question index {q['index']!r} is not numeric — the chatbot keys questions "
                "by their numeric (smartvote) index."
            ) from None
        questions[q_id] = {"de": q["texts"].get("de", ""), "category": q.get("category") or ""}

    candidates = {c["index"]: c for c in data["candidates"]}

    reactions = pd.DataFrame(data["reactions"], columns=["candidateIndex", "questionIndex", "value", "comment"])
    reactions = reactions.rename(columns={"candidateIndex": "candidate_id", "questionIndex": "question_id"})
    reactions = reactions.dropna(subset=["value"])  # comment without a position: nothing to index
    reactions["question_id"] = reactions["question_id"].astype(int)
    reactions["value"] = reactions["value"].astype(float)
    return reactions, candidates, questions


def _candidate_documents(
    reactions: pd.DataFrame, candidates: Dict[str, dict], questions: Dict[int, dict]
) -> List[Document]:
    """One Document per (candidate, question) pair.

    Each document text looks like:
        "Benno Beispiel (Partei: FDP, Ort: Musterstadt) zur Frage:Befürworten Sie ...?':
         Stimme zu.Kommentar: "Durch die schrittweise Erhöhung ..."."

    Metadata: candidate_id, name, party, question_id, category, value, source, has_comment.
    """
    documents: List[Document] = []
    for row in reactions.itertuples(index=False):
        cand = candidates.get(row.candidate_id, {})
        q = questions[row.question_id]

        name = (cand.get("name") or "unknown").strip()
        party = cand.get("party") or "unknown"
        city = cand.get("city") or "unknown"
        value = row.value

        comment = row.comment
        comment_str = f'Kommentar: "{comment}"' if isinstance(comment, str) and comment.strip() else ""

        reaction_label = _REACTION_LABELS.get(value, f"scored {value}")

        text = (
            f"{name} (Partei: {party}, Ort: {city}) zur Frage:"
            f"{q['de']}': "
            f"{reaction_label}.{comment_str}."
        )

        documents.append(
            Document(
                text=text,
                metadata={
                    "candidate_id": str(row.candidate_id),
                    "name": str(name),
                    "party": str(party),
                    "question_id": int(row.question_id),
                    "category": str(q["category"]),
                    "value": value,
                    "source": "candidate_positions",
                    "has_comment": len(comment_str) > 0,
                },
            )
        )

    logger.info("Produced %d candidate-position documents.", len(documents))
    return documents


def _party_average_documents(
    reactions: pd.DataFrame, candidates: Dict[str, dict], questions: Dict[int, dict]
) -> List[Document]:
    """One Document per (party, question) pair.

    Each document text looks like:
        "Die Kandidierenden der Partei FDP (42 Kandidierende) haben zur Frage 'XYZ'
         im Durchschnitt mit 'Stimme eher zu' geantwortet. Konsens: moderater Konsens)."

    Metadata: party, question_id, category, avg_value, std, n, source.
    """
    with_party = reactions.assign(
        party=reactions["candidate_id"].map(lambda cid: candidates.get(cid, {}).get("party"))
    ).dropna(subset=["party"])

    grouped = (
        with_party.groupby(["party", "question_id"])["value"]
        .agg(avg_value="mean", std="std", n="count")
        .reset_index()
    )
    grouped["std"] = grouped["std"].fillna(0.0)

    documents: List[Document] = []
    for row in grouped.itertuples(index=False):
        q = questions.get(row.question_id)
        if not q:
            continue

        party = str(row.party)
        avg = float(row.avg_value)
        std = float(row.std)
        n = int(row.n)

        reaction_label = _avg_to_reaction_label(avg)
        polarization = _std_to_descriptor(std)

        text = (
            f"Die Kandidierenden der Partei {party} ({n} Kandidierende) haben zur Frage '{q['de']}' "
            f"im Durchschnitt mit '{reaction_label}' geantwortet."
            f"Konsens: {polarization}). "
        )

        documents.append(
            Document(
                text=text,
                metadata={
                    "party": party,
                    "question_id": int(row.question_id),
                    "category": str(q["category"]),
                    "avg_value": avg,
                    "std": std,
                    "n": n,
                    "source": "party_candidate_averages",
                },
            )
        )

    logger.info("Produced %d party-candidate-averages documents.", len(documents))
    return documents


def _avg_to_reaction_label(avg: float) -> str:
    """Map a continuous average score to the nearest human-readable reaction label.
    """
    if avg >= 0.875:
        return "Stimme zu"
    if avg >= 0.625:
        return "Stimme eher zu"
    if avg >= 0.375:
        return "Neutral"
    if avg >= 0.125:
        return "Stimme eher nicht zu"
    return "Stimme nicht zu"


def _std_to_descriptor(std: float) -> str:
    """Describe the level of internal party agreement from the standard deviation."""
    if std < 0.2:
        return "breiter Konsens"
    if std < 0.4:
        return "moderater Konsens"
    if std < 0.6:
        return "moderate Uneinigkeit"
    return "starke Uneinigkeit"
