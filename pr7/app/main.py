"""Веб-рівень застосунку."""

from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from . import extraction
from .images import ImageError, UPLOAD_MAX_BYTES
from .llm import LLMError

app = FastAPI(title="Розбір рахунків — ПР7")

INDEX_PAGE = Path(__file__).parent / "templates" / "index.html"
SAMPLES_DIR = Path(__file__).resolve().parent.parent / "samples"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

app.mount("/samples", StaticFiles(directory=SAMPLES_DIR), name="samples")


def result_to_dict(result: extraction.Result) -> dict:
    return {
        "decision": result.decision,
        "reasons": result.reasons,
        "document": result.document,
        "issues": [asdict(issue) for issue in result.issues],
        "image": result.image,
        "model": result.model,
        "elapsed": result.elapsed,
        "usage": result.usage,
        "raw_response": result.raw_response,
        "finish_reason": result.finish_reason,
    }


@app.get("/", response_class=HTMLResponse)
def page() -> str:
    return INDEX_PAGE.read_text(encoding="utf-8")


@app.get("/api/samples")
def api_samples() -> dict:
    groups = {}
    for folder in sorted(p for p in SAMPLES_DIR.iterdir() if p.is_dir()):
        files = sorted(f.name for f in folder.iterdir() if f.suffix.lower() in IMAGE_SUFFIXES)
        if files:
            groups[folder.name] = files
    return groups


@app.post("/api/extract")
async def api_extract(image: UploadFile = File(...)) -> dict:
    try:
        content = await image.read(UPLOAD_MAX_BYTES + 1)
        if len(content) > UPLOAD_MAX_BYTES:
            raise ImageError(
                f"Файл перевищує допустимий розмір {UPLOAD_MAX_BYTES // (1024 * 1024)} МБ."
            )
        result = extraction.process(content)
        return result_to_dict(result)
    except ImageError as exc:
        return JSONResponse(status_code=400, content={"detail": str(exc)})
    except LLMError as exc:
        return JSONResponse(status_code=502, content={"detail": str(exc)})
    except ValueError as exc:
        return JSONResponse(status_code=422, content={"detail": str(exc)})
    except Exception as exc:  # pragma: no cover - fallback
        return JSONResponse(status_code=500, content={"detail": f"Невідома помилка застосунку: {exc}"})
