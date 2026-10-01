"""Векторний індекс: зберігання векторів фрагментів і пошук найближчих."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from .documents import Chunk

load_dotenv()

INDEX_DIR = Path(__file__).parent.parent / "index"
DEFAULT_TOP_K = int(os.getenv("SEARCH_TOP_K", "8"))
_threshold = os.getenv("SIMILARITY_THRESHOLD", "").strip()
SIMILARITY_THRESHOLD: float | None = float(_threshold) if _threshold else None


@dataclass
class Hit:
    chunk: Chunk
    score: float


@dataclass
class SearchIndex:
    chunks: list[Chunk]
    vectors: np.ndarray
    model_name: str
    extra: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.chunks)


def _normalize_vectors(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim == 1:
        vectors = vectors.reshape(1, -1)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return vectors / norms


def _matches_filters(metadata: dict, filters: dict | None) -> bool:
    if not filters:
        return True
    allowed = set(metadata.keys())
    for key, expected in filters.items():
        if expected is None or expected == "":
            continue
        if key not in allowed:
            raise ValueError(f"Фільтр '{key}' не підтримується для цих фрагментів.")
        actual = str(metadata.get(key, "")).strip().lower()
        if key == "product" and not actual:
            continue
        if actual != str(expected).strip().lower():
            return False
    return True


def build(chunks: list[Chunk], vectors: np.ndarray, model_name: str) -> SearchIndex:
    if len(chunks) != len(vectors):
        raise ValueError(f"Кількість фрагментів ({len(chunks)}) не відповідає кількості векторів ({len(vectors)}).")
    normalized = _normalize_vectors(vectors)
    return SearchIndex(chunks=chunks, vectors=normalized, model_name=model_name, extra={"vector_dim": normalized.shape[1]})


def save(index: SearchIndex, path: Path = INDEX_DIR) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "vectors.npy", index.vectors.astype(np.float32))
    payload = [{"text": chunk.text, "source": chunk.source, "metadata": chunk.metadata} for chunk in index.chunks]
    (path / "chunks.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (path / "model_name.txt").write_text(index.model_name, encoding="utf-8")


def load(path: Path = INDEX_DIR) -> SearchIndex:
    vector_path = path / "vectors.npy"
    chunks_path = path / "chunks.json"
    model_path = path / "model_name.txt"
    if not vector_path.exists() or not chunks_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Індекс не знайдено в {path}; запустіть python ingest.py.")
    vectors = np.load(vector_path)
    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = [Chunk(text=item["text"], source=item["source"], metadata=item.get("metadata", {})) for item in payload]
    model_name = model_path.read_text(encoding="utf-8").strip() or "unknown"
    return SearchIndex(chunks=chunks, vectors=np.asarray(vectors, dtype=np.float32), model_name=model_name)


def search(
    index: SearchIndex,
    query_vector: np.ndarray,
    top_k: int = DEFAULT_TOP_K,
    filters: dict | None = None,
    threshold: float | None = SIMILARITY_THRESHOLD,
) -> list[Hit]:
    if top_k <= 0:
        return []
    query = np.asarray(query_vector, dtype=np.float32)
    if query.ndim != 1:
        query = query.reshape(-1)
    if index.vectors.shape[1] != query.shape[0]:
        raise ValueError(f"Розмірність запиту ({query.shape[0]}) не співпадає з індексом ({index.vectors.shape[1]}).")
    query = query / max(np.linalg.norm(query), 1e-12)
    scores = index.vectors @ query
    ranked: list[tuple[float, int]] = []
    for idx, score in enumerate(scores):
        if not _matches_filters(index.chunks[idx].metadata, filters):
            continue
        if threshold is not None and float(score) < threshold:
            continue
        ranked.append((float(score), idx))
    ranked.sort(key=lambda item: item[0], reverse=True)
    hits = [Hit(chunk=index.chunks[idx], score=score) for score, idx in ranked[:top_k]]
    return hits
