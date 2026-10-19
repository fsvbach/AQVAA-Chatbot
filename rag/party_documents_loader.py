import os
import re
from pathlib import Path
from typing import Iterator, List, Optional
from config import settings
from logger import get_logger
from rag.document import Document
from rag.parties import CANONICAL_PARTIES
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling_core.types.doc import DocItemLabel, DoclingDocument

import torch
torch.backends.mkldnn.enabled = False # to avoid problems on linux machine with older CPU 


logger = get_logger(__name__)

_pipeline_options = PdfPipelineOptions()
_pipeline_options.do_ocr = False
_pipeline_options.do_table_structure = False

_converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(pipeline_options=_pipeline_options)
    }
)

_HEADING_LABELS = {DocItemLabel.TITLE, DocItemLabel.SECTION_HEADER}


def iter_party_files(manifesto_dir: Path) -> Iterator[tuple[str, Path]]:
    """Yield (party_name, file_path) for every PDF/MD inside each party subfolder.

    Documents are organized as one subfolder per party under manifesto_dir; the party
    name is the subfolder's name and is expected to be a canonical party label (see
    rag.parties.CANONICAL_PARTIES) so manifesto metadata filters on the same strings as
    candidate data. A folder whose name is not canonical is still yielded but warned
    about, since its chunks would never match a party filter. Within each party folder,
    PDFs are yielded first then markdown files, both in sorted order. Loose PDF/MD files
    sitting directly in manifesto_dir (not inside a party folder) are warned about and
    skipped.
    """
    for loose in sorted(manifesto_dir.glob("*.pdf")) + sorted(manifesto_dir.glob("*.md")):
        logger.warning(
            "Ignoring loose file %s — place documents inside a party subfolder of %s.",
            loose.name,
            manifesto_dir,
        )

    for party_dir in sorted(p for p in manifesto_dir.iterdir() if p.is_dir()):
        party = party_dir.name
        if party not in CANONICAL_PARTIES:
            logger.warning(
                "Party folder %r is not a canonical party %s — its chunks will be indexed "
                "but a party filter will never match them. Rename the folder to a canonical label.",
                party,
                sorted(CANONICAL_PARTIES),
            )
        for path in sorted(party_dir.glob("*.pdf")) + sorted(party_dir.glob("*.md")):
            yield party, path


def load_manifesto_documents() -> List[Document]:
    """Load all party documents (PDF + Markdown), chunk them, and return one Document per chunk.

    Documents live in one subfolder per party under settings.manifestos_path; the party
    name is taken from the subfolder name.

    Metadata per document: party, source, filename, chunk_index.
    PDF chunks also carry page_no and bbox provenance.
    MD chunks carry char_start provenance.
    """

    manifesto_dir = Path(settings.manifestos_path)
    if not manifesto_dir.exists():
        logger.warning("Manifesto directory not found: %s — skipping manifesto indexing.", manifesto_dir)
        return []

    documents: List[Document] = []
    file_count = 0
    parties: set[str] = set()

    for party, path in iter_party_files(manifesto_dir):
        try:
            file_docs = _load_file_documents(party, path)
        except Exception as exc:
            logger.warning("Failed to parse %s: %s — skipping.", path.name, exc)
            continue

        if not file_docs:
            continue

        documents.extend(file_docs)
        file_count += 1
        parties.add(party)
        logger.info("Loaded %s (%s): %d chunks", path.name, party, len(file_docs))

    if not file_count:
        logger.warning("No PDF or markdown files found under %s — skipping manifesto indexing.", manifesto_dir)
        return []

    logger.info(
        "Produced %d manifesto chunks from %d files across %d parties.",
        len(documents),
        file_count,
        len(parties),
    )
    return documents


def _load_file_documents(party: str, path: Path) -> List[Document]:
    """Parse a single PDF or markdown file into chunked Documents with provenance metadata."""
    if path.suffix.lower() == ".pdf":
        markdown, heading_map = _load_single_pdf(path)
        if not markdown.strip():
            logger.warning("No text extracted from %s — skipping.", path.name)
            return []
        chunks = _chunk_text(markdown, settings.rag_chunk_size, settings.rag_chunk_overlap)
        documents: List[Document] = []
        for chunk_index, chunk in enumerate(chunks):
            prov = _find_provenance(chunk, markdown, heading_map)
            metadata = {
                "party": party,
                "source": "party_manifesto",
                "filename": path.name,
                "chunk_index": chunk_index,
            }
            if prov:
                metadata["page_no"] = prov["page_no"]
                metadata["bbox"] = prov["bbox"]
            documents.append(Document(text=chunk, metadata=metadata))
        return documents

    text = _load_single_md_file(path)
    if not text.strip():
        logger.warning("No text extracted from %s — skipping.", path.name)
        return []
    chunks = _chunk_text(text, settings.rag_chunk_size, settings.rag_chunk_overlap)
    documents = []
    for chunk_index, chunk in enumerate(chunks):
        char_start = _find_md_char_start(chunk, text)
        metadata = {
            "party": party,
            "source": "party_manifesto",
            "filename": path.name,
            "chunk_index": chunk_index,
        }
        if char_start is not None:
            metadata["char_start"] = char_start
        documents.append(Document(text=chunk, metadata=metadata))
    return documents


def _build_heading_map(doc, markdown: str) -> list[dict]:
    """Build an ordered list of headings with their page/bbox provenance and char offset.

    Each entry: {text, page_no, bbox, char_offset}
    Sorted by char_offset ascending.
    """
    heading_map = []
    for item, _ in doc.iterate_items():
        if item.label not in _HEADING_LABELS:
            continue
        if not item.prov:
            continue

        prov = item.prov[0]
        level = getattr(item, "level", 1)  # SectionHeaderItem has .level; TitleItem defaults to 1
        heading_text = f"{'#' * level} {item.text}"

        char_offset = markdown.find(heading_text)
        if char_offset == -1:
            logger.warning("Heading not found in markdown: %r — provenance skipped.", heading_text[:60])
            continue

        heading_map.append({
            "text": heading_text,
            "page_no": prov.page_no,
            "bbox": {
                "l": prov.bbox.l,
                "t": prov.bbox.t,
                "r": prov.bbox.r,
                "b": prov.bbox.b,
                "coord_origin": prov.bbox.coord_origin.value,
            },
            "char_offset": char_offset,
        })

    heading_map.sort(key=lambda e: e["char_offset"])
    return heading_map


def _find_provenance(chunk: str, markdown: str, heading_map: list[dict]) -> Optional[dict]:
    """Return the heading map entry whose section contains this chunk.

    The chunker prepends parent heading lines as context prefix. Strip those to
    find the body text, locate it in the markdown, then return the last heading
    whose char_offset is at or before that position.
    """
    if not heading_map:
        return None

    body = _strip_chunk_heading_prefix(chunk)
    if not body:
        return None

    chunk_start = markdown.find(body[:100])
    if chunk_start == -1:
        return None

    best = None
    for entry in heading_map:
        if entry["char_offset"] <= chunk_start:
            best = entry
        else:
            break
    return best


def _find_md_char_start(chunk: str, text: str) -> Optional[int]:
    """Return the start character offset of a chunk's body within the source text."""
    body = _strip_chunk_heading_prefix(chunk)
    if not body:
        return None
    offset = text.find(body[:100])
    return offset if offset != -1 else None

def _strip_chunk_heading_prefix(chunk: str) -> str:
    """Strip the heading prefix lines prepended by the chunker and return the body."""
    lines = chunk.split("\n")
    body_start = 0
    for i, line in enumerate(lines):
        if re.match(r"^#{1,5} ", line):
            body_start = i + 1
        else:
            break
    return "\n".join(lines[body_start:]).strip()


def _sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Character-level sliding-window chunker. Returns stripped, non-empty chunks."""
    step = chunk_size - overlap
    chunks = []
    start = 0
    while start < len(text):
        chunk = text[start : start + chunk_size].strip()
        if chunk:
            chunks.append(chunk)
        start += step
    return chunks


_HEADER_LEVELS = [1, 2, 3, 4, 5]


def _split_at_header_level(text: str, level: int) -> List[str]:
    """Split text immediately before each header of exactly `level` depth."""
    hashes = "#" * level
    return re.split(rf"(?=^{hashes}(?!#) )", text, flags=re.MULTILINE)


def _chunk_section_recursive(
    text: str,
    chunk_size: int,
    overlap: int,
    parent_prefix: str,
    level_index: int,
) -> List[str]:
    """Recursively split text by descending through header levels.

    Tries each level in _HEADER_LEVELS in order. A section that fits within
    chunk_size is emitted whole. Oversized sections are split at the next
    deeper header level, with the parent header prepended for context.
    Falls back to sliding-window chunking after all header levels are exhausted.
    """
    if level_index >= len(_HEADER_LEVELS):
        # All header levels exhausted — fall back to sliding window.
        available = chunk_size - len(parent_prefix)
        if available <= 0:
            chunks = []
            if parent_prefix.strip():
                chunks.append(parent_prefix.rstrip())
            chunks.extend(_sliding_window_chunks(text, chunk_size, overlap))
            return chunks
        safe_overlap = min(overlap, available - 1)
        return [parent_prefix + sub for sub in _sliding_window_chunks(text, available, safe_overlap)]

    level = _HEADER_LEVELS[level_index]
    hashes = "#" * level
    sections = _split_at_header_level(text, level)

    chunks: List[str] = []
    for section in sections:
        section = section.strip()
        if not section:
            continue

        if not section.startswith(hashes + " "):
            # Preamble before the first header at this level — try next level.
            chunks.extend(_chunk_section_recursive(section, chunk_size, overlap, parent_prefix, level_index + 1))
            continue

        header_line = section.split("\n", 1)[0]
        body = section[len(header_line):].strip()

        if not body:
            continue

        new_prefix = parent_prefix + header_line + "\n"
        full_chunk = new_prefix + body

        if len(full_chunk) <= chunk_size:
            chunks.append(full_chunk)
        else:
            chunks.extend(_chunk_section_recursive(body, chunk_size, overlap, new_prefix, level_index + 1))

    return chunks


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Hierarchical markdown chunker that descends through # to ##### headers.

    Splits at the shallowest header level first. Sections that fit within
    chunk_size are kept whole. Oversized sections are split at the next
    deeper header level, with the parent header prepended for context.
    Falls back to sliding-window chunking after all header levels are exhausted.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be > 0, got {chunk_size}")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(f"overlap must be in [0, chunk_size), got {overlap}")

    return _chunk_section_recursive(text.strip(), chunk_size, overlap, parent_prefix="", level_index=0)


def _load_single_pdf(path: Path) -> tuple[str, list[dict]]:
    """Extract text and heading provenance map from a PDF using Docling."""
    doc = _converter.convert(path).document
    markdown = doc.export_to_markdown(image_placeholder="")
    heading_map = _build_heading_map(doc, markdown)
    logger.debug("Built heading map with %d entries for %s.", len(heading_map), path.name)
    return markdown, heading_map


def _load_single_md_file(path: Path) -> str:
    """Return all text from a markdown file."""
    return path.read_text(encoding="utf-8")
