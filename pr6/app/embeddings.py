"""Модуль ембедінгів."""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _needs_prefix() -> bool:
    return "e5" in MODEL_NAME.lower()


def embed_passages(texts: list[str]) -> np.ndarray:
    if not texts:
        return np.empty((0, 0), dtype=np.float32)
    model = get_model()
    prefixed = [f"passage: {text}" if _needs_prefix() else text for text in texts]
    vectors = model.encode(
        prefixed,
        normalize_embeddings=True,
        convert_to_numpy=True,
        batch_size=32,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def embed_query(text: str) -> np.ndarray:
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Порожній запит не можна перетворити на вектор.")
    model = get_model()
    prompt = f"query: {clean}" if _needs_prefix() else clean
    vector = model.encode([prompt], normalize_embeddings=True, convert_to_numpy=True)[0]
    return np.asarray(vector, dtype=np.float32)
