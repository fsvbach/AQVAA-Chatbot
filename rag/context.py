"""Turn retrieved documents into the RAG-context body of a system prompt.

Condition-agnostic and shared by both the assistant generator
(``assistant_generator``) and the interview generator (``interview_generator``).
Two steps:

- ``format_*`` render the ``context`` dict's doc lists into plain-text snippets,
  each capped to a character budget so the prompt stays within the context window.
- ``render_context_sections`` / ``format_additional_info`` wrap those snippets in
  the localized ``--- ... ---`` section headers that go into the prompt.

Header strings (including their spacing) are part of the prompt contract; do not
"tidy" them.
"""
from typing import List

from logger import get_logger

logger = get_logger(__name__)

# Max characters per context section in the system prompt.
MAX_PARTY_CANDIDATE_AVERAGES_CHARS = 2000
MAX_CANDIDATE_CHARS = 6000
MAX_MANIFESTO_CHARS = 16000

_PARTY_AVG_HEADER = {
    "de": "--- DURCHSCHNITTSANTWORT DER KANDIDIERENDEN  ---",
    "en": "--- AVERAGE ANSWERS OF CANDIDATES  ---",
}
_MANIFESTO_HEADER = {
    "de": "--- PARTEI-DOKUMENTE ---",
    "en": "--- PARTY DOCUMENTS ---",
}
_CANDIDATE_HEADER = {
    "de": "--- KANDIDATENPOSITIONEN ---",
    "en": "--- CANDIDATE POSITIONS ---",
}


# ---------------------------------------------------------------------------
# Docs -> text snippets
# ---------------------------------------------------------------------------

def format_party_candidate_averages(docs: List[dict]) -> str:
    """Format party-stance summaries."""
    if not docs:
        return ""

    lines = []
    total_chars = 0
    for doc in docs:
        text = doc.get("text", "")
        if not text or total_chars + len(text) > MAX_PARTY_CANDIDATE_AVERAGES_CHARS:
            break
        lines.append(f"- {text}")
        total_chars += len(text)

    return "\n".join(lines)


def format_candidate_context(docs: List[dict]) -> str:
    if not docs:
        return ""

    lines = []
    total_chars = 0
    for doc in docs:
        text = doc.get("text", "")
        if total_chars + len(text) > MAX_CANDIDATE_CHARS:
            break
        lines.append(f"- {text}")
        total_chars += len(text)

    return "\n".join(lines)


def format_manifesto_section(docs: List[dict], is_de: bool) -> str:
    if not docs:
        return ""

    label = "Partei" if is_de else "Party"
    lines = []
    total_chars = 0
    docs_sorted_by_party = sorted(docs, key=lambda doc: doc.get("party"))

    for doc in docs_sorted_by_party:
        text = doc.get("text", "")
        party = doc.get("party", "")
        entry = f"- {text}\n  [{label}: {party}]"
        if total_chars + len(entry) > MAX_MANIFESTO_CHARS:
            logger.warning(f"Total manifesto upper limit of {MAX_MANIFESTO_CHARS} characters reached. Skipped '{entry[:10]}' and all following documents")
            break
        lines.append(entry)
        total_chars += len(entry)

    return "\n".join(lines)


def serialize_docs(docs: list[dict]) -> list[dict]:
    keep = {"text", "name", "party", "question_id", "score", "filename", "chunk_index"}
    return [{k: v for k, v in doc.items() if k in keep} for doc in docs]


# ---------------------------------------------------------------------------
# Snippets -> labelled prompt sections
# ---------------------------------------------------------------------------

def format_additional_info(additional_info: dict | None) -> str:
    """Render the per-question supplementary info sections (German source text)."""
    if not additional_info:
        return ""
    section = ""
    if additional_info.get("terms"):
        section += f"\n\n--- BEGRIFFSERKLÄRUNGEN ---\n{additional_info['terms']}"
    if additional_info.get("background"):
        section += f"\n\n--- HINTERGRUNDINFORMATION ---\n{additional_info['background']}"
    if additional_info.get("pros_cons"):
        section += f"\n\n--- PRO / CONTRA ---\n{additional_info['pros_cons']}"
    return section

def format_study_context(study_context:str | None)-> str:
    if not study_context:
        return ""
    
    return f"\n\n--- Studienkontext ---\n{study_context}"


def render_context_sections(
    party_overview: str,
    manifesto_section: str,
    candidate_context: str,
    is_de: bool,
) -> str:
    """Render the party-averages / manifesto / candidate-positions sections.

    Each section is emitted only when its snippet is non-empty, in a fixed order.
    """
    lang = "de" if is_de else "en"
    sections = (
        (party_overview, _PARTY_AVG_HEADER[lang]),
        (manifesto_section, _MANIFESTO_HEADER[lang]),
        (candidate_context, _CANDIDATE_HEADER[lang]),
    )
    body = ""
    for snippet, header in sections:
        if snippet:
            body += f"\n\n{header}\n{snippet}"
    return body
