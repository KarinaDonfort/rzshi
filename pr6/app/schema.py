"""Контракт відповіді моделі: формальна схема JSON та її перевірка."""

import json

from jsonschema import Draft7Validator


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "minLength": 1},
            "sources": {
                "type": "array",
                "items": {"type": "integer", "minimum": 1},
            },
            "found": {"type": "boolean"},
        },
        "required": ["answer", "sources", "found"],
        "additionalProperties": False,
    }


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна довіряти structurally."""
    if not isinstance(raw, str):
        raise ValueError("Відповідь моделі не є рядком JSON.")

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Відповідь моделі не є валідним JSON: {exc.msg}.") from exc

    if not isinstance(payload, dict):
        raise ValueError("Відповідь моделі має бути JSON-об'єктом.")

    validator = Draft7Validator(output_schema())
    errors = sorted(validator.iter_errors(payload), key=lambda item: list(item.path))
    if errors:
        first = errors[0]
        location = ".".join(str(part) for part in first.path) or "root"
        raise ValueError(f"Відповідь моделі не відповідає схемі в '{location}': {first.message}")

    return payload
