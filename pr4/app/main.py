"""Веб-рівень застосунку: сторінка діалогу і JSON-ендпоінт."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from . import llm

app = FastAPI(title="Помічник служби підтримки — ПР4")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
CONTEXT_FILE = Path(__file__).parent.parent / "context.md"


class Turn(BaseModel):
    """Одна репліка розмови: `user` — клієнт, `assistant` — помічник."""

    role: str
    content: str


class ChatRequest(BaseModel):
    """Те, що надсилає сторінка: нове повідомлення й розмову до нього."""

    message: str
    history: list[Turn] = []


def load_context() -> str:
    """Прочитати правила організації з файлу context.md."""
    return CONTEXT_FILE.read_text(encoding="utf-8")


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку діалогу."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/chat")
def api_chat(payload: ChatRequest):
    """Повернути структуровану відповідь помічника в JSON."""
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Порожнє повідомлення клієнта.")

    history = [turn.model_dump() for turn in payload.history]
    try:
        return llm.ask(payload.message, history, load_context())
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
