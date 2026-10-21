"""Контракт відповіді моделі."""

import json

import jsonschema


def output_schema() -> dict:
    """Повернути JSON Schema відповіді моделі."""
    return {
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "InvoiceExtraction",
        "type": "object",
        "additionalProperties": True,
        "required": [
            "document_type",
            "number",
            "date",
            "supplier",
            "buyer",
            "items",
            "total_without_vat",
            "vat",
            "total",
        ],
        "properties": {
            "document_type": {"type": ["string", "null"]},
            "number": {"type": ["string", "null"]},
            "date": {"type": ["string", "null"]},
            "valid_until": {"type": ["string", "null"]},
            "supplier": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                    "iban": {"type": ["string", "null"]},
                },
                "required": ["name", "code", "iban"],
            },
            "buyer": {
                "type": "object",
                "additionalProperties": True,
                "properties": {
                    "name": {"type": ["string", "null"]},
                    "code": {"type": ["string", "null"]},
                },
                "required": ["name", "code"],
            },
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": True,
                    "required": ["name", "unit", "quantity", "price", "amount"],
                    "properties": {
                        "name": {"type": ["string", "null"]},
                        "unit": {"type": ["string", "null"]},
                        "quantity": {"type": ["number", "string", "integer", "null"]},
                        "price": {"type": ["number", "string", "null"]},
                        "amount": {"type": ["number", "string", "null"]},
                    },
                },
            },
            "total_without_vat": {"type": ["number", "string", "null"]},
            "vat": {"type": ["number", "string", "null"]},
            "total": {"type": ["number", "string", "null"]},
        },
    }


def _coerce_missing_literals(value):
    """Перетворити текстові "null"/"none" на None, щоб відсутні поля не змішувалися з реальними значеннями."""
    if isinstance(value, dict):
        return {key: _coerce_missing_literals(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_coerce_missing_literals(item) for item in value]
    if isinstance(value, str):
        text = value.strip()
        if text.lower() in {"null", "none", "n/a", "na"}:
            return None
    return value


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна довіряти структурно."""
    if raw is None:
        raise ValueError("Модель повернула порожнє значення.")
    if isinstance(raw, (bytes, bytearray)):
        raw = raw.decode("utf-8", errors="strict")
    if not isinstance(raw, str):
        raise ValueError("Модель повернула JSON, але не рядок.")

    try:
        document = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Модель повернула не JSON: {exc.msg}") from exc

    document = _coerce_missing_literals(document)
    if not isinstance(document, dict):
        raise ValueError("Модель повернула JSON, але не обʼєкт документа.")

    validator = jsonschema.Draft7Validator(output_schema())
    errors = sorted(validator.iter_errors(document), key=lambda err: list(err.path))
    if errors:
        details = "; ".join(f"{'.'.join(map(str, err.path)) or '<root>'}: {err.message}" for err in errors[:5])
        raise ValueError(f"Відповідь моделі не відповідає схемі: {details}")

    return document
