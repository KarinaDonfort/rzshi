"""Веб-рівень застосунку: сторінка помічника і JSON-ендпоінти.

Цей файл не знає ані як шукаються фрагменти, ані як збирається контекст,
ані якою моделлю й за якою інструкцією отримано відповідь — усе це
лишається в `app/retrieval.py`, `app/llm.py`, `app/schema.py` і
поєднується в `app/rag.py`. Тут вирішується інше: що застосунок приймає
від сторінки, що віддає їй і з яким HTTP-статусом.

Індекс будується заздалегідь командою `python ingest.py` (з папки pr6),
а тут лише читається при старті — як у ПР5.

Запуск із папки pr6:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import index, keyword, rag

app = FastAPI(title="Помічник за базою знань — ПР6")
app.state.index = None
app.state.keyword_index = None

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


class AskRequest(BaseModel):
    """Те, що надсилає сторінка.

    `filters` — умови на метадані, які обрав користувач, наприклад
    `{"product": "Вега S"}`; порожній словник — без фільтрів. Те, що
    користувач обирати не має (аудиторія документа), сюди не входить —
    це рішення коду, а не сторінки.
    """

    question: str
    filters: dict = {}


def hit_to_dict(hit: index.Hit) -> dict:
    return {
        "score": hit.score,
        "text": hit.chunk.text,
        "source": hit.chunk.source,
        "metadata": hit.chunk.metadata,
    }


def answer_to_dict(result: rag.Answer) -> dict:
    """Перетворити результат конвеєра на те, що піде на сторінку."""
    return {
        "answer": result.text,
        "found": result.found,
        "sources": [
            {"ref": s.ref, "score": s.score, "source": s.chunk.source,
             "metadata": s.chunk.metadata, "text": s.chunk.text}
            for s in result.sources
        ],
        "retrieved": [hit_to_dict(h) for h in result.retrieved],
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
    }


@app.on_event("startup")
def load_indexes() -> None:
    """Прочитати збудований індекс і зібрати індекс за словами з тих
    самих фрагментів — як у ПР5.

    Без індексу застосунок усе одно стартує: сторінка має відкритися й
    пояснити, що робити.
    """
    app.state.index = None
    app.state.keyword_index = None
    try:
        app.state.index = index.load()
    except Exception as exc:  # noqa: BLE001 — старт не має падати без індексу
        print(f"Індекс не завантажено: {type(exc).__name__}: {exc}")
        return
    app.state.keyword_index = keyword.build(app.state.index.chunks)


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    """Віддати сторінку помічника."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/status")
def api_status() -> dict:
    """Стан індексу: чи збудовано, скільки фрагментів, якою моделлю."""
    idx = app.state.index
    if idx is None:
        return {"ready": False, "hint": "індекс не збудовано — виконайте python ingest.py"}
    sources = {chunk.source for chunk in idx.chunks}
    return {
        "ready": True,
        "chunks": len(idx),
        "documents": len(sources),
        "model": idx.model_name,
    }


@app.post("/api/ask")
def api_ask(payload: AskRequest) -> dict:
    """Відповісти на питання й повернути відповідь із джерелами."""
    question = (payload.question or "").strip()
    if not question:
        raise HTTPException(
            status_code=400,
            detail={"error": "empty_question", "message": "Питання порожнє."},
        )
    if app.state.index is None:
        raise HTTPException(
            status_code=503,
            detail={"error": "index_missing", "message": "Індекс не збудовано — виконайте python ingest.py."},
        )

    try:
        result = rag.answer(
            question,
            app.state.index,
            app.state.keyword_index,
            filters=payload.filters or None,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=400,
            detail={"error": "bad_request", "message": str(exc)},
        ) from exc
    except Exception as exc:  # pragma: no cover - runtime integration errors are reported to API clients
        raise HTTPException(
            status_code=502,
            detail={"error": "model_error", "message": f"Помилка обробки запиту: {exc}"},
        ) from exc

    return answer_to_dict(result)
