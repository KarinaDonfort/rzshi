"""Відбір фрагментів для моделі та збирання контексту."""

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from . import embeddings
from .documents import Chunk
from .index import Hit, SearchIndex, search as semantic_search
from .keyword import KeywordIndex, search as keyword_search

load_dotenv()

CONTEXT_CHUNKS = int(os.getenv("RAG_CONTEXT_CHUNKS", "4"))
CONTEXT_BUDGET = int(os.getenv("RAG_CONTEXT_BUDGET", "1500"))


@dataclass
class Source:
    ref: int
    chunk: Chunk
    score: float


def _estimate_tokens(text: str) -> int:
    raw = (text or "").replace("\n", " ").strip()
    if not raw:
        return 0
    return max(1, len(raw) // 4)


def _merge_hits(primary: list[Hit], secondary: list[Hit]) -> list[Hit]:
    combined: dict[tuple[str, str, str], Hit] = {}

    def add_hits(hits: list[Hit], weight: float) -> None:
        if not hits:
            return
        max_score = max(hit.score for hit in hits)
        for hit in hits:
            metadata = hit.chunk.metadata or {}
            heading = str(metadata.get("heading") or "").strip()
            key = (hit.chunk.source, heading, hit.chunk.text[:200])
            norm_score = hit.score / max_score if max_score > 0 else 0.0
            if key not in combined:
                combined[key] = Hit(chunk=hit.chunk, score=0.0)
            combined[key].score += norm_score * weight

    add_hits(primary, 1.0)
    add_hits(secondary, 0.85)

    merged = list(combined.values())
    merged.sort(key=lambda item: item.score, reverse=True)
    return merged


def retrieve(
    query: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> list[Hit]:
    if not query or not query.strip():
        return []
    if index is None:
        return []

    effective = dict(filters or {})
    effective.setdefault("audience", "клієнти")
    effective.setdefault("status", "чинний")
    if effective.get("audience") != "клієнти":
        effective["audience"] = "клієнти"
    if effective.get("status") != "чинний":
        effective["status"] = "чинний"

    top_k = int(os.getenv("SEARCH_TOP_K", "8"))
    threshold = float(os.getenv("SIMILARITY_THRESHOLD", "0.0")) if os.getenv("SIMILARITY_THRESHOLD") else 0.0

    semantic_hits = semantic_search(
        index,
        embeddings.embed_query(query),
        top_k=max(6, top_k * 2),
        filters=effective,
        threshold=threshold,
    )

    keyword_hits: list[Hit] = []
    if keyword_index is not None:
        keyword_hits = keyword_search(keyword_index, query, top_k=max(6, top_k * 2), filters=effective)

    merged = _merge_hits(semantic_hits, keyword_hits)
    merged = [hit for hit in merged if hit.score > 0.0]
    return merged[:CONTEXT_CHUNKS * 2]


def build_context(hits: list[Hit], budget: int = CONTEXT_BUDGET) -> tuple[str, list[Source]]:
    used = 0
    sources: list[Source] = []
    parts: list[str] = []
    for order, hit in enumerate(hits[:CONTEXT_CHUNKS], start=1):
        text = hit.chunk.text.strip()
        metadata = hit.chunk.metadata or {}
        title = metadata.get("title") or hit.chunk.source
        heading = metadata.get("heading") or "розділ"
        updated = metadata.get("updated") or "невідомо"
        label = f"[{order}] {title} | Розділ: {heading} | Оновлено: {updated}"
        block = f"{label}\n{text}\n"
        cost = _estimate_tokens(block)
        if sources and used + cost > budget:
            break
        if not sources and cost > budget:
            parts.append(block)
            sources.append(Source(ref=order, chunk=hit.chunk, score=hit.score))
            used += cost
            break
        parts.append(block)
        sources.append(Source(ref=order, chunk=hit.chunk, score=hit.score))
        used += cost
    context_text = "\n\n".join(parts)
    return context_text, sources
