"""Модуль inference: єдине місце застосунку, яке знає про модель.

Тут живуть ваги, поріг упевненості й формат «сирого» результату моделі.
Веб-рівень (`app/main.py`) отримує звідси готовий структурований список
знайдених обʼєктів і нічого не знає ані про `ultralytics`, ані про те,
у якому вигляді модель віддає рамки.

Довідка про модель: https://docs.ultralytics.com/
"""

import io
import time

from PIL import Image, UnidentifiedImageError
from ultralytics import YOLO

WEIGHTS = "yolov8n.pt"
DEFAULT_CONFIDENCE = 0.25
MIN_CONFIDENCE = 0.0
MAX_CONFIDENCE = 1.0

# Кеш моделі: завантажується один раз при першому виклику load_model(),
# далі повертається той самий обʼєкт. Економить 3–10 секунд на кожен запит.
_model: YOLO | None = None


class DetectionError(Exception):
    """Помилка детекції, зрозуміла веб-рівню.

    status_code — рекомендований HTTP-статус. Веб-рівень просто його
    відображає, не думаючи про причини.
    """

    def __init__(self, message: str, status_code: int = 500):
        super().__init__(message)
        self.status_code = status_code


def load_model() -> YOLO:
    """Повернути готову до роботи модель.

    Завантажує ваги лише при першому виклику, далі повертає з кешу.
    """
    global _model
    if _model is None:
        try:
            _model = YOLO(WEIGHTS)
        except Exception as exc:
            raise DetectionError(
                f"Не вдалося завантажити модель: {type(exc).__name__}",
                status_code=500,
            ) from exc
    return _model


def _validate_confidence(confidence: float) -> float:
    """Перевірити, що поріг у межах [0, 1]."""
    if not (MIN_CONFIDENCE <= confidence <= MAX_CONFIDENCE):
        raise DetectionError(
            f"Поріг упевненості має бути в межах "
            f"{MIN_CONFIDENCE}–{MAX_CONFIDENCE}, отримано {confidence}.",
            status_code=400,
        )
    return confidence


def detect(image_bytes: bytes, confidence: float = DEFAULT_CONFIDENCE) -> dict:
    """Знайти обʼєкти на зображенні.

    Приймає байти завантаженого файлу, повертає структурований результат:
    для кожного знайденого обʼєкта — клас, рамку й упевненість, а також
    їхню кількість і час виконання.
    """
    if not image_bytes:
        raise DetectionError("Файл порожній.", status_code=400)

    _validate_confidence(confidence)

    try:
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise DetectionError(
            "Файл не є зображенням або пошкоджений.",
            status_code=400,
        ) from exc

    model = load_model()

    started = time.perf_counter()
    try:
        results = model.predict(
            source=image,
            conf=confidence,
            verbose=False,
        )
    except Exception as exc:
        raise DetectionError(
            f"Помилка inference: {type(exc).__name__}",
            status_code=500,
        ) from exc
    elapsed_ms = (time.perf_counter() - started) * 1000

    result = results[0]
    names = result.names 

    detections = []
    for box in result.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()

        detections.append({
            "class": names[cls_id],
            "class_id": cls_id,
            "confidence": round(conf, 3),
            "box": {
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            },
        })

    return {
        "count": len(detections),
        "confidence_threshold": confidence,
        "inference_ms": round(elapsed_ms, 1),
        "image_size": {
            "width": image.width,
            "height": image.height,
        },
        "detections": detections,
    }