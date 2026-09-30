"""Модуль ембедінгів: єдине місце застосунку, яке знає, яка модель
перетворює текст на вектор і як її викликати.

Індекс (`app/index.py`) і веб-рівень (`app/main.py`) отримують звідси
готові вектори й не знають, локальна це модель чи API. Замінивши модель
тут, ви не змінюєте решту коду — але маєте перебудувати індекс: вектори
різних моделей непорівнянні.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши рішення з
розділу 2 практичної роботи:

* яку модель узяти й чому саме її: мови, розмірність, довжина входу,
  розмір, швидкість на CPU;
* чи потрібні моделі префікси для запиту й для тексту (у деяких моделей
  запит і документ кодуються по-різному — дивіться картку моделі);
* чи нормалізувати вектори і де — тут чи в індексі;
* як кодувати кілька сотень фрагментів: по одному чи пакетами.

Назва моделі читається з `.env`; значення за замовчуванням — відправна
точка, а не рекомендація.
"""

import os

import numpy as np
from dotenv import load_dotenv

load_dotenv()

# Назва моделі ембедінгів. Для локальної моделі — ідентифікатор на
# Hugging Face Hub; для моделі через API — назва в провайдера.
MODEL_NAME = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
_MODEL = None


def get_model():
    """Повернути готову до роботи модель.

    Для локальної моделі створюємо її один раз і повторно використовуємо
    в усіх викликах. Для сімейства E5 потрібні префікси `query:` і
    `passage:`, і саме вони прописані тут.
    """
    global _MODEL
    if _MODEL is None:
        from sentence_transformers import SentenceTransformer

        _MODEL = SentenceTransformer(MODEL_NAME)
    return _MODEL


def _needs_prefix() -> bool:
    return "e5" in MODEL_NAME.lower()


def embed_passages(texts: list[str]) -> np.ndarray:
    """Перетворити тексти фрагментів на вектори.

    Повертає масив розміру (кількість текстів × розмірність моделі).
    Викликається під час індексування — для всієї колекції одразу.
    """
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
    """Перетворити запит користувача на вектор тієї самої розмірності.

    Окрема функція навмисно: у моделей із префіксами запит кодується не
    так, як фрагмент, і саме тут це видно.
    """
    clean = (text or "").strip()
    if not clean:
        raise ValueError("Порожній запит не можна перетворити на вектор.")
    model = get_model()
    prompt = f"query: {clean}" if _needs_prefix() else clean
    vector = model.encode([prompt], normalize_embeddings=True, convert_to_numpy=True)[0]
    return np.asarray(vector, dtype=np.float32)
