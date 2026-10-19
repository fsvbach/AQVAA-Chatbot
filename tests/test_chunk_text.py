import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from rag.party_documents_loader import _chunk_text

CHUNK_SIZE = 200
OVERLAP = 20


# --- Validation ---

def test_invalid_chunk_size_raises():
    with pytest.raises(ValueError):
        _chunk_text("hello", chunk_size=0, overlap=0)


def test_invalid_overlap_raises():
    with pytest.raises(ValueError):
        _chunk_text("hello", chunk_size=100, overlap=100)


# --- Basic section splitting ---

def test_single_section_under_limit_is_one_chunk():
    text = "## Health Policy\n\nWe want affordable healthcare for all."
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].startswith("## Health Policy")
    assert "affordable healthcare" in chunks[0]


def test_two_sections_produce_two_chunks():
    text = (
        "## Health Policy\n\nWe want affordable healthcare.\n\n"
        "## Education Policy\n\nWe want free universities."
    )
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) == 2
    assert chunks[0].startswith("## Health Policy")
    assert chunks[1].startswith("## Education Policy")


def test_header_is_included_in_each_chunk():
    text = "## Climate\n\nReduce CO2 emissions significantly."
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert all("## Climate" in c for c in chunks)


# --- Large section fallback ---

def test_large_section_produces_multiple_subchunks():
    body = "word " * 300  # ~1500 chars
    text = f"## Big Section\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) > 1


def test_large_section_subchunks_all_start_with_header():
    body = "word " * 300
    text = f"## Big Section\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert all(c.startswith("## Big Section") for c in chunks)


def test_oversized_header_body_is_not_discarded():
    # Header alone exceeds chunk_size — body must still appear in output
    header = "## " + "x" * (CHUNK_SIZE + 10)
    body = "important body content here"
    text = f"{header}\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    all_text = " ".join(chunks)
    assert "important body content" in all_text


def test_large_section_no_subchunk_exceeds_chunk_size():
    body = "word " * 300
    text = f"## Big Section\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


# --- Edge cases ---

def test_empty_section_after_header_is_skipped():
    text = "## Empty Section\n\n## Real Section\n\nSome content here."
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) == 1
    assert chunks[0].startswith("## Real Section")


def test_preamble_before_first_header_is_chunked():
    text = "Intro text before any header.\n\n## Section One\n\nContent here."
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert any("Intro text" in c for c in chunks)


def test_no_headers_falls_back_to_sliding_window():
    text = "plain text " * 100  # ~1100 chars, no ## headers
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)


def test_whitespace_only_input_returns_empty():
    chunks = _chunk_text("   \n\n   ", CHUNK_SIZE, OVERLAP)
    assert chunks == []


# --- Hierarchical header splitting ---

def test_h1_sections_produce_separate_chunks():
    text = (
        "# Health Policy\n\nAffordable healthcare for all.\n\n"
        "# Education Policy\n\nFree universities for everyone."
    )
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) == 2
    assert chunks[0].startswith("# Health Policy")
    assert chunks[1].startswith("# Education Policy")


def test_large_h1_splits_at_h2():
    # h1 section is too big but contains two h2 sub-sections that each fit
    sub_a = "## Costs\n\nReduce costs significantly for all families across the country by reforming the system."
    sub_b = "## Access\n\nEnsure equal access in both rural and urban areas for every citizen without exception."
    # Each sub-section alone fits; together with the h1 header they exceed CHUNK_SIZE
    text = f"# Health Policy\n\n{sub_a}\n\n{sub_b}"
    assert len(text) > CHUNK_SIZE, "test setup: combined text must exceed chunk_size"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) == 2
    assert all("# Health Policy" in c for c in chunks)
    assert any("## Costs" in c for c in chunks)
    assert any("## Access" in c for c in chunks)


def test_parent_header_prepended_to_child_chunks():
    body = "detail " * 30  # ~210 chars — forces split
    text = f"# Top Level\n\n## Sub A\n\n{body}\n\n## Sub B\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert all("# Top Level" in c for c in chunks)


def test_mixed_header_levels_all_content_present():
    text = (
        "# Chapter One\n\n"
        "## Section A\n\nContent A.\n\n"
        "### Subsection A1\n\nContent A1.\n\n"
        "## Section B\n\nContent B.\n\n"
        "# Chapter Two\n\n"
        "## Section C\n\nContent C."
    )
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    all_text = " ".join(chunks)
    for fragment in ["Content A", "Content A1", "Content B", "Content C"]:
        assert fragment in all_text


def test_deep_nesting_no_chunk_exceeds_chunk_size():
    # All header levels present but sections still too large — falls back to sliding window
    body = "word " * 100  # ~500 chars
    text = f"# H1\n\n## H2\n\n### H3\n\n#### H4\n\n##### H5\n\n{body}"
    chunks = _chunk_text(text, CHUNK_SIZE, OVERLAP)
    assert len(chunks) > 1
    assert all(len(chunk) <= CHUNK_SIZE for chunk in chunks)
