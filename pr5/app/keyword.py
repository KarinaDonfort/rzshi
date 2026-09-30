"""Пошук за ключовими словами — точка порівняння для семантичного.

Той самий набір фрагментів, той самий формат влучень (`Hit`), ті самі
фільтри — інший спосіб ранжувати. Без цього модуля не буде з чим
порівнювати семантичний пошук, а без порівняння — не буде відповіді на
питання, чи він узагалі потрібен для цієї колекції.
"""

import re
from dataclasses import dataclass, field

from rank_bm25 import BM25Okapi

from .documents import Chunk
from .index import DEFAULT_TOP_K, Hit


@dataclass
class KeywordIndex:
    """Індекс для пошуку за словами. Що саме тут зберігати — вирішуєте ви."""

    chunks: list[Chunk]
    bm25: BM25Okapi | None = None
    extra: dict = field(default_factory=dict)


def tokenize(text: str) -> list[str]:
    """Розбити текст на слова для індексування й для запиту.

    Одна й та сама функція для обох: запит і фрагмент мають розбиватися
    однаково, інакше збігів не буде.
    """
    if not text:
        return []
    text = text.lower()
    return re.findall(r"[a-zа-яё0-9]+(?:-[a-zа-яё0-9]+)*", text)


def _matches_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    for key, expected in filters.items():
        if expected is None or expected == "":
            continue
        if key not in metadata:
            raise ValueError(f"Фільтр '{key}' не підтримується для цих фрагментів.")
        if str(metadata.get(key, "")).strip().lower() != str(expected).strip().lower():
            return False
    return True


def build(chunks: list[Chunk]) -> KeywordIndex:
    """Зібрати індекс за словами з тих самих фрагментів, що й векторний."""
    tokenized = [tokenize(chunk.text) for chunk in chunks]
    bm25 = BM25Okapi(tokenized)
    return KeywordIndex(chunks=chunks, bm25=bm25, extra={"tokenized": tokenized})


def search(
    index: KeywordIndex,
    query: str,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
) -> list[Hit]:
    """Знайти фрагменти за словами запиту.

    Формат результату — той самий `Hit`, що й у векторного пошуку, з тими
    самими фільтрами за метаданими, щоб сторінка показувала обидва
    способи поруч.
    """
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
