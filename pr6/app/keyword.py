"""Пошук за ключовими словами."""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit


@dataclass
class KeywordIndex:
    chunks: list[Chunk]
    bm25: BM25Okapi | None = None
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-zа-яёіїєґ0-9]+(?:-[a-zа-яёіїєґ0-9]+)*", text)


def _matches_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected is None or expected == "":
            continue
        if key not in metadata:
            raise ValueError(f"Фільтр '{key}' не підтримується для цих фрагментів.")
        actual = str(metadata.get(key, "")).strip().lower()
        if key == "product" and not actual:
            continue
        if actual != str(expected).strip().lower():
            return False
    return True


def build(chunks: list[Chunk]) -> KeywordIndex:
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return KeywordIndex(chunks=chunks, bm25=bm25, extra={"tokenized": tokenized})


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    if top_k <= 0:
        return []
    query_tokens = tokenize(query)
    if not query_tokens:
        return []
    if index.bm25 is None:
        raise ValueError("Індекс BM25 не побудовано.")
    scores = index.bm25.get_scores(query_tokens)
    ranked: list[tuple[float, int]] = []
    for idx, score in enumerate(scores):
        if not _matches_filters(index.chunks[idx].metadata, filters):
            continue
        ranked.append((float(score), idx))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return [Hit(chunk=index.chunks[idx], score=score) for score, idx in ranked[:top_k]]
