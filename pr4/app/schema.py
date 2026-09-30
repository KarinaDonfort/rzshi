"""Контракт відповіді помічника: що саме модель має повернути і як це
перевіряється.

Окремий модуль навмисно. Схема — це домовленість між моделлю і рештою
застосунку, і вона не залежить від провайдера: змінивши модель або спосіб
виклику, схему ви не змінюєте. Веб-рівень і сторінка працюють лише з тим,
що пройшло перевірку тут.

Що потрібно решті застосунку від відповіді (розділ 2 практичної роботи):

* текст відповіді для клієнта;
* тема звернення з фіксованого переліку — за нею звернення передається
  потрібному відділу;
* чи відповідь ґрунтується на правилах, чи правила про це мовчать;
* чи потрібне уточнення;
* чи передати розмову оператору;
* номер замовлення, якщо клієнт його назвав (формат — у `context.md`).

Як назвати поля, які з них обовʼязкові, які з переліком значень, які
можуть бути порожніми — ваше рішення. Як і те, чим описати схему: JSON
Schema вручну (перевірка — пакет `jsonschema`) або pydantic-модель, з
якої схема генерується (перевірка — та сама модель).

Сторінка каркаса бере текст для клієнта з поля `reply`, а решту полів
показує як картку звернення. Назвете інакше — змініть сторінку.
"""

import json

from jsonschema import Draft202012Validator, ValidationError


def output_schema() -> dict:
    """Повернути JSON Schema відповіді помічника."""
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "reply": {"type": "string", "minLength": 1, "maxLength": 2000},
            "topic": {
                "type": "string",
                "enum": ["order", "delivery", "payment", "returns", "warranty", "general", "other"],
            },
            "grounded": {"type": "boolean"},
            "needs_clarification": {"type": "boolean"},
            "escalate_to_operator": {"type": "boolean"},
            "order_number": {"anyOf": [{"type": "string", "pattern": r"^\d{6}$"}, {"type": "null"}]},
        },
        "required": [
            "reply",
            "topic",
            "grounded",
            "needs_clarification",
            "escalate_to_operator",
            "order_number",
        ],
    }


def validate(raw: str) -> dict:
    """Перевірити сиру відповідь моделі й повернути дані, яким можна довіряти структурно."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:  # pragma: no cover - handled by caller and tests
        raise ValueError(f"Відповідь не є валідним JSON: {exc.msg}") from exc

    if not isinstance(data, dict):
        raise ValueError("Відповідь JSON має бути об'єктом, а не списком або примітивом.")

    validator = Draft202012Validator(output_schema())
    try:
        validator.validate(data)
    except ValidationError as exc:
        raise ValueError(f"Відповідь не проходить схему: {exc.message}") from exc

    if data.get("order_number") is not None and not isinstance(data["order_number"], str):
        raise ValueError("Поле order_number може бути рядком із шести цифр або null.")

    return data
