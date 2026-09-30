"""Порівняння трьох конфігурацій на реальному Gemini API.

Запуск:
    $env:PYTHONPATH='.'; python compare/run_compare.py
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.llm import build_messages, estimate_tokens
from app.schema import validate

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
CONTEXT = (ROOT / "context.md").read_text(encoding="utf-8")
MODEL = os.getenv("LLM_MODEL", "gemini-3.5-flash-lite")

DATASET = [
    {
        "id": "order_cancellation",
        "message": "Добрий день. Я хочу скасувати замовлення 482913, якщо воно ще готується.",
        "history": [],
        "expected_topic": "order",
        "expected_grounded": True,
        "expected_order_number": "482913",
    },
    {
        "id": "delivery_delay",
        "message": "Чи можна дізнатися, коли прийде замовлення, якщо я оплачую післяплатою?",
        "history": [],
        "expected_topic": "delivery",
        "expected_grounded": True,
        "expected_order_number": None,
    },
    {
        "id": "return_policy",
        "message": "Можу я повернути товар через 17 днів, якщо упаковка не пошкоджена?",
        "history": [],
        "expected_topic": "returns",
        "expected_grounded": True,
        "expected_order_number": None,
    },
    {
        "id": "out_of_scope",
        "message": "Я хочу, щоб ви повернули мені гроші за часи, коли товар ще не надійшов?",
        "history": [],
        "expected_topic": "general",
        "expected_grounded": False,
        "expected_order_number": None,
    },
    {
        "id": "warranty_claim",
        "message": "Мені потрібна гарантія на навушники. Номер замовлення 482913.",
        "history": [],
        "expected_topic": "warranty",
        "expected_grounded": True,
        "expected_order_number": "482913",
    },
    {
        "id": "ambiguous_order",
        "message": "У мене проблема з товаром, але я не пам'ятаю, що саме я купив.",
        "history": [],
        "expected_topic": "general",
        "expected_grounded": False,
        "expected_order_number": None,
    },
    {
        "id": "payment_method",
        "message": "Яка у вас оплата, якщо замовлення дорожче 10 000 грн?",
        "history": [],
        "expected_topic": "payment",
        "expected_grounded": True,
        "expected_order_number": None,
    },
]

DIALOGUES = [
    {
        "id": "dialog_order_memory",
        "history": [
            {"role": "user", "content": "Добрий день. Мій номер замовлення 482913. Я хочу скасувати замовлення до доставки."},
            {"role": "assistant", "content": "Зрозумів, працюю з номером 482913."},
            {"role": "user", "content": "І ще — чи можна змінити склад замовлення?"},
        ],
        "message": "Подивіться, чи можу я тепер самостійно скасувати його, якщо воно ще готується?",
        "expected_topic": "order",
        "expected_grounded": True,
        "expected_order_number": "482913",
    },
    {
        "id": "dialog_product_memory",
        "history": [
            {"role": "user", "content": "Проблема з навушниками, які я придбав. Потрібно уточнити гарантію."},
            {"role": "assistant", "content": "Підтверджую, зберігаю контекст про навушники."},
            {"role": "user", "content": "Чи гарантія поширюється на випадок падіння?"},
        ],
        "message": "Я хочу знати, чи відпадає гарантія, якщо я самостійно ремонтував їх.",
        "expected_topic": "warranty",
        "expected_grounded": True,
        "expected_order_number": None,
    },
]


def build_fuzzy_prompt(message: str, history: list[dict]) -> str:
    prior = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    return (
        "Ти помічник служби підтримки. Відповідай українською мовою, не вигадуй правила і не став "
        "себе вище за контекст. Відповідь давай у структурі JSON з полями reply, topic, grounded, "
        "needs_clarification, escalate_to_operator, order_number.\n\n"
        f"{prior}\n\nКлієнт: {message}"
    )


def build_structured_prompt(message: str, history: list[dict]) -> str:
    prior = "\n".join(f"{turn['role']}: {turn['content']}" for turn in history)
    system = (
        "Ти — помічник служби підтримки інтернет-магазину. "
        "Поверни тільки валідний JSON об'єкт без пояснень. "
        "Поля: reply, topic, grounded, needs_clarification, escalate_to_operator, order_number. "
        "topic: order | delivery | payment | returns | warranty | general | other. "
        "order_number: рядок із 6 цифр або null. "
        "Відповідь має підкріплюватися правилами з context.md, а якщо правил немає — говори про це чітко."
    )
    return f"system: {system}\n\nhistory:\n{prior}\n\nuser: {message}"


def clean_json_text(text: str) -> str:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.I)
        cleaned = re.sub(r"\s*```$", "", cleaned, flags=re.I)
    return cleaned.strip()


def parse_payload(raw: str):
    cleaned = clean_json_text(raw)
    if not cleaned:
        return {"schema_pass": False, "parsed": None, "raw": raw, "error": "empty"}
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        return {"schema_pass": False, "parsed": None, "raw": raw, "error": f"json_decode:{exc.msg}"}

    if not isinstance(data, dict):
        return {"schema_pass": False, "parsed": None, "raw": raw, "error": "not_object"}

    try:
        validate(json.dumps(data, ensure_ascii=False))
    except Exception as exc:  # noqa: BLE001
        return {"schema_pass": False, "parsed": data, "raw": raw, "error": str(exc)}

    return {"schema_pass": True, "parsed": data, "raw": raw, "error": None}


def call_model(prompt_text: str):
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise RuntimeError("LLM_API_KEY is missing")
    client = genai.Client(api_key=api_key)
    started = time.perf_counter()
    response = client.models.generate_content(
        model=MODEL,
        contents=prompt_text,
        config=types.GenerateContentConfig(
            temperature=0.2,
            max_output_tokens=250,
            response_mime_type="application/json",
        ),
    )
    elapsed = time.perf_counter() - started
    raw = (getattr(response, "text", "") or "").strip()
    usage = getattr(response, "usage_metadata", None)
    usage_payload = {
        "prompt_tokens": int(getattr(usage, "prompt_token_count", 0) or 0),
        "completion_tokens": int(getattr(usage, "candidates_token_count", 0) or 0),
        "total_tokens": int(getattr(usage, "total_token_count", 0) or 0),
    }
    return {"raw": raw, "elapsed": round(elapsed, 3), "usage": usage_payload}


def safe_evaluate_case(mode: str, case: dict) -> dict:
    """Оцінити кейс без падіння через квоту/мережеві помилки провайдера."""
    try:
        return evaluate_case(mode, case)
    except Exception as exc:  # noqa: BLE001
        return {
            "id": case.get("id"),
            "mode": mode,
            "schema_pass": False,
            "topic_ok": False,
            "grounded_ok": False,
            "hallucination": False,
            "order_memory_ok": False,
            "response": None,
            "elapsed": 0.0,
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "error": f"provider_error: {exc}",
        }


def evaluate_case(mode: str, case: dict) -> dict:
    message = case["message"]
    history = case.get("history", [])
    if mode == "fuzzy":
        prompt = build_fuzzy_prompt(message, history)
    elif mode == "structured":
        prompt = build_structured_prompt(message, history)
    elif mode == "context":
        prompt = "\n\n".join(f"{entry['role']}: {entry['content']}" for entry in build_messages(message, history, CONTEXT))
    else:
        raise ValueError(f"Unknown mode: {mode}")

    result = call_model(prompt)
    payload = parse_payload(result["raw"])
    parsed = payload.get("parsed") if payload.get("schema_pass") else None

    expected_topic = case.get("expected_topic")
    expected_grounded = case.get("expected_grounded")
    expected_order = case.get("expected_order_number")
    topic_ok = bool(parsed and parsed.get("topic") == expected_topic)
    grounded_ok = bool(parsed and parsed.get("grounded") == expected_grounded)
    hallucination = bool(parsed and parsed.get("grounded") and not expected_grounded)
    order_memory_ok = True
    if expected_order is not None:
        order_memory_ok = bool(parsed and parsed.get("order_number") == expected_order)

    return {
        "id": case["id"],
        "mode": mode,
        "schema_pass": payload.get("schema_pass", False),
        "topic_ok": topic_ok,
        "grounded_ok": grounded_ok,
        "hallucination": hallucination,
        "order_memory_ok": order_memory_ok,
        "response": parsed if parsed is not None else payload.get("raw"),
        "elapsed": result["elapsed"],
        "usage": result["usage"],
        "error": payload.get("error"),
    }


def summary_by_mode(results: list[dict]):
    by_mode = {}
    for mode in ("fuzzy", "structured", "context"):
        rows = [row for row in results if row["mode"] == mode]
        by_mode[mode] = {
            "total": len(rows),
            "schema_pass": sum(1 for row in rows if row["schema_pass"]),
            "topic_ok": sum(1 for row in rows if row["topic_ok"]),
            "grounded_ok": sum(1 for row in rows if row["grounded_ok"]),
            "hallucination": sum(1 for row in rows if row["hallucination"]),
            "order_memory_ok": sum(1 for row in rows if row["order_memory_ok"]),
        }
    return by_mode


def main() -> None:
    if not os.getenv("LLM_API_KEY"):
        print("LLM_API_KEY відсутній; реальний виклик до моделі неможливий.")
        return

    all_results = []
    for mode in ("fuzzy", "structured", "context"):
        print(f"\n=== {mode.upper()} ===")
        for case in DATASET + DIALOGUES:
            row = safe_evaluate_case(mode, case)
            all_results.append(row)
            print(json.dumps(row, ensure_ascii=False, indent=2))

    print("\n=== SUMMARY ===")
    print(json.dumps(summary_by_mode(all_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
