"""Колекція документів: читання, метадані, поділ на фрагменти."""

import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

DOCS_DIR = Path(__file__).parent.parent / "docs"
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))


@dataclass
class Chunk:
    text: str
    source: str
    metadata: dict = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict, str]:
    lines = raw.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, raw
    metadata: dict = {}
    for i, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            body = "\n".join(lines[i + 1:]).lstrip("\n")
            return metadata, body
        if ":" in line:
            key, _, value = line.partition(":")
            metadata[key.strip()] = value.strip()
    return {}, raw


def load_documents(docs_dir: Path = DOCS_DIR) -> list[tuple[str, dict, str]]:
    documents = []
    for path in sorted(docs_dir.glob("*.md")):
        if path.name.lower() == "readme.md":
            continue
        metadata, body = parse_front_matter(path.read_text(encoding="utf-8"))
        documents.append((path.name, metadata, body))
    return documents


def _chunk_by_size(text: str, size: int, overlap: int) -> list[str]:
    if not text.strip():
        return []
    if len(text) <= size:
        return [text.strip()]
    step = max(1, size - overlap)
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end >= len(text):
            break
        start += step
    return chunks


def split(text: str, source: str, metadata: dict) -> list[Chunk]:
    doc_title = metadata.get("title") or source
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = None
    current_lines: list[str] = []

    def flush_section() -> None:
        nonlocal current_heading, current_lines
        section_text = "\n".join(current_lines).strip()
        if section_text:
            sections.append((current_heading, section_text))
        current_heading = None
        current_lines = []

    for line in text.splitlines():
        heading_match = re.match(r"^(#{1,6})\s+(.*)$", line)
        if heading_match:
            flush_section()
            current_heading = heading_match.group(2).strip()
            continue
        current_lines.append(line.rstrip())
    flush_section()

    if not sections:
        sections = [(doc_title, text.strip())]

    chunks: list[Chunk] = []
    for section_index, (heading, section_text) in enumerate(sections, start=1):
        section_label = heading or doc_title
        section_meta = {
            "title": doc_title,
            "heading": section_label,
            "section_index": section_index,
        }
        chunk_text = f"{doc_title}\nРозділ: {section_label}\n\n{section_text.strip()}"
        for part_index, piece in enumerate(_chunk_by_size(chunk_text, CHUNK_SIZE, CHUNK_OVERLAP), start=1):
            final_text = piece.strip()
            if not final_text:
                continue
            meta = dict(metadata)
            meta.update(section_meta)
            meta["chunk_index"] = part_index
            chunks.append(Chunk(text=final_text, source=source, metadata=meta))
    return chunks


def load_chunks(docs_dir: Path = DOCS_DIR) -> list[Chunk]:
    chunks: list[Chunk] = []
    for source, metadata, body in load_documents(docs_dir):
        chunks.extend(split(body, source, metadata))
    return chunks
