"""Підготовка зображення до надсилання моделі."""

import io
import os
import warnings
from dataclasses import dataclass, field

from dotenv import load_dotenv
from PIL import Image, ImageOps, UnidentifiedImageError

load_dotenv()

UPLOAD_MAX_BYTES = int(float(os.getenv("UPLOAD_MAX_MB", "10")) * 1024 * 1024)
IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "1600"))


class ImageError(Exception):
    """Файл не можна віддати моделі. Повідомлення — для користувача."""


@dataclass
class PreparedImage:
    data: bytes
    mime: str
    original: dict
    sent: dict
    quality_issues: list[str] = field(default_factory=list)


def _content_touches_edge(image: Image.Image) -> bool:
    grayscale = image.convert("L")
    edge_size = max(1, round(min(grayscale.size) * 0.005))
    edges = (
        grayscale.crop((0, 0, grayscale.width, edge_size)),
        grayscale.crop((0, grayscale.height - edge_size, grayscale.width, grayscale.height)),
        grayscale.crop((0, 0, edge_size, grayscale.height)),
        grayscale.crop((grayscale.width - edge_size, 0, grayscale.width, grayscale.height)),
    )
    for edge in edges:
        pixels = edge.tobytes()
        dark_ratio = sum(pixel < 210 for pixel in pixels) / len(pixels)
        if 0.08 <= dark_ratio < 0.9:
            return True
    return False


def prepare(content: bytes) -> PreparedImage:
    """Перевірити байти, що надійшли, і підготувати зображення для моделі."""
    if not content:
        raise ImageError("Файл порожній. Завантажте рахунок як зображення.")
    if len(content) > UPLOAD_MAX_BYTES:
        raise ImageError(f"Файл перевищує допустимий розмір {UPLOAD_MAX_BYTES // (1024 * 1024)} МБ.")

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(content)) as probe:
                probe.verify()
            image = Image.open(io.BytesIO(content))
            image.load()
            image = ImageOps.exif_transpose(image)
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
        Image.DecompressionBombWarning,
    ) as exc:
        raise ImageError("Файл не є валідним зображенням.") from exc

    if image.mode not in {"RGB", "L", "RGBA"}:
        image = image.convert("RGB")
    if image.mode == "RGBA":
        image = image.convert("RGB")
    if image.mode == "L":
        image = image.convert("RGB")

    quality_issues = []
    if _content_touches_edge(image):
        quality_issues.append("possible_crop")

    image = ImageOps.autocontrast(image)
    try:
        from PIL import ImageEnhance
        image = ImageEnhance.Contrast(image).enhance(1.25)
    except Exception:
        pass

    original = {"width": image.width, "height": image.height, "bytes": len(content)}
    if image.width < 120 or image.height < 120:
        raise ImageError("Зображення занадто маленьке для напевного читання рахунку. Попросіть перезняти документ.")

    max_side = max(image.width, image.height)
    if max_side > IMAGE_MAX_SIDE:
        scale = IMAGE_MAX_SIDE / max_side
        image = image.resize(
            (max(1, int(round(image.width * scale))), max(1, int(round(image.height * scale)))),
            Image.Resampling.LANCZOS,
        )

    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    data = buffer.getvalue()

    return PreparedImage(
        data=data,
        mime="image/png",
        original=original,
        sent={"width": image.width, "height": image.height, "bytes": len(data)},
        quality_issues=quality_issues,
    )
