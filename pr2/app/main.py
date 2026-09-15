"""Веб-рівень застосунку: сторінка із завантаженням файлу і JSON-ендпоінт.

Цей файл не має знати про `ultralytics`, ваги моделі й формат її «сирого»
виводу — усе це лишається в `app/detector.py`. Тут вирішується інше: що
застосунок віддає клієнтові та з яким HTTP-статусом.

Запуск із папки pr2:

    uvicorn app.main:app --reload

Далі відкрийте http://127.0.0.1:8000
"""

from pathlib import Path

from fastapi import FastAPI, File, Query, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

from . import detector

app = FastAPI(title="Детекція обʼєктів — ПР2")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Віддати сторінку із завантаженням зображення."""
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.post("/api/detect")
async def api_detect(
    image: UploadFile = File(...),
    confidence: float = Query(detector.DEFAULT_CONFIDENCE),
):
    """Повернути знайдені на зображенні обʼєкти у форматі JSON.

    Успіх — 200 + JSON.
    Помилка — відповідний статус + JSON {"error": "..."}.
    """
    content = await image.read()
    try:
        return detector.detect(content, confidence=confidence)
    except detector.DetectionError as exc:
        return JSONResponse(
            status_code=exc.status_code,
            content={"error": str(exc)},
        )