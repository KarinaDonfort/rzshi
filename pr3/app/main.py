"""Веб-рівень застосунку: сторінка зі зверненням і JSON-ендпоінт.

Цей файл не має знати ані про провайдера моделі, ані про те, як
складається запит до неї, — усе це лишається в `app/llm.py`. Тут
вирішується інше: що застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr3:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

import logging
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР3")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"

# Технічні помилки провайдера пише в журнал, але не показує користувачеві.
logger = logging.getLogger("pr3")


class Question(BaseModel):
    """Звернення користувача."""

    question: str = Field(default="", max_length=5000)


def load_context() -> str:
    """Прочитати правила організації, на підставі яких відповідає модель.

    Контекст — це дані застосунку, а не знання моделі. Він живе окремим
    файлом і передається в запит разом зі зверненням. Заміна файлу змінює
    поведінку помічника без зміни коду.
    """
    try:
        return CONTEXT_FILE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("Не вдалося прочитати context.md: %s", exc)
        raise


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку зі зверненням."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/ask")
def api_ask(payload: Question):
    """Повернути відповідь помічника у форматі JSON.

    Успіх — 200 + JSON з полями answer, model, elapsed, tokens.
    Помилка — відповідний статус + JSON {"error": "..."}.

    Технічні деталі збою (тип винятку провайдера, його текст) пишуться
    в журнал, а користувач бачить зрозуміле повідомлення.
    """
    try:
        context = load_context()
    except OSError:
        return JSONResponse(
            status_code=500,
            content={"error": "Правила магазину тимчасово недоступні."},
        )

    try:
        result = llm.ask(payload.question, context)
    except llm.LLMError as exc:
        # Веб-рівень не знає про openai — він лише перетворює
        # предметну помилку на HTTP-відповідь.
        logger.warning("LLMError: %s", exc)
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc)},
        )
    except Exception as exc:  # несподіване — не має пропустити назовні 500 без пояснення
        logger.exception("Несподівана помилка: %s", exc)
        return JSONResponse(
            status_code=500,
            content={"error": "Внутрішня помилка застосунку."},
        )

    return result