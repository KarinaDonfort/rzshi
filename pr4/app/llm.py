"""Модуль роботи з мовною моделлю: єдине місце застосунку, яке знає про API.

Тут живуть налаштування доступу, системна інструкція з прикладами,
збирання запиту з частин і правило, за яким історія вміщується в бюджет
токенів. Веб-рівень (`app/main.py`) отримує звідси перевірений результат
і нічого не знає ані про провайдера, ані про склад повідомлень. Схема
відповіді та її перевірка — в `app/schema.py`.

Функції нижче — заготовки. Реалізуйте їх самі, ухваливши по дорозі
рішення з розділу 2 практичної роботи:

* що входить до системної інструкції: роль, обмеження, формат, приклади
  межових випадків — і які саме приклади;
* де в запиті стоять правила, де історія, де поточне звернення;
* що класти в історію з боку помічника: увесь JSON чи лише текст для
  клієнта;
* чим рахувати токени й що відкидати першим, коли бюджет вичерпано;
* чи передавати схему провайдеру через `response_format`, чи просити JSON
  текстом — і що робити з відповіддю, яка не пройшла перевірку;
* що саме зараховувати до виміряного часу — з повторами чи без.

Обробку збоїв із ПР3 (таймаут, ліміт, недоступність, невірний ключ)
перенесіть сюди. Налаштування, як і раніше, читаються з `.env`; ключ
доступу — секрет.
"""

import json
import logging
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from google import genai
from google.genai import types

from .schema import output_schema, validate

load_dotenv()

LOG_FILE = Path(__file__).resolve().parent.parent / "llm.log"
logger = logging.getLogger("support_llm")
logger.setLevel(logging.INFO)
logger.propagate = False
if not logger.handlers:
    handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

BASE_URL = os.getenv("LLM_BASE_URL")
API_KEY = os.getenv("LLM_API_KEY")
MODEL = os.getenv("LLM_MODEL")

TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "600"))
TIMEOUT = float(os.getenv("LLM_TIMEOUT", "30"))
TOKEN_BUDGET = int(os.getenv("LLM_TOKEN_BUDGET", "3000"))


class LLMError(Exception):
    """Помилка роботи з моделлю, зрозуміла веб-рівню."""


def get_client():
    """Повернути готовий до роботи клієнт сервісу Gemini."""
    if not API_KEY or not MODEL:
        raise LLMError("Налаштування LLM_API_KEY або LLM_MODEL відсутні.")
    return genai.Client(api_key=API_KEY)


def estimate_tokens(text: str) -> int:
    """Оцінимо обсяг тексту приблизно, бо точний токенізатор доступний лише в провайдера."""
    raw = (text or "").replace("\n", " ")
    if not raw.strip():
        return 0
    return max(1, int(len(raw) / 2.8))


def _summarize_history(turns: list[dict]) -> str:
    details: list[str] = []
    for turn in turns:
        text = str(turn.get("content", "") or "").strip()
        if not text:
            continue
        match = re.search(r"\b\d{6}\b", text)
        if match:
            details.append(f"номер замовлення {match.group(0)}")
        product_match = re.search(r"(навушники|телефон|смартфон|книга|зарядка|рюкзак|клавіатура|мікрофон|блок живлення)", text.lower())
        if product_match:
            details.append(f"товар: {product_match.group(1)}")
    if not details:
        return "попередні повідомлення без важливих фактів"
    return "; ".join(details[:3])


def fit_budget(history: list[dict], budget: int) -> list[dict]:
    """Повернути історію в межах бюджету, зберігаючи перший і останні репліки."""
    turns: list[dict] = []
    for entry in history or []:
        if not isinstance(entry, dict):
            continue
        role = str(entry.get("role", "user"))
        content = str(entry.get("content", "") or "").strip()
        if content:
            turns.append({"role": role, "content": content})

    if not turns:
        return []
    if budget <= 0:
        return turns[-2:]

    kept = [turns[0]]
    recent = turns[-6:]
    for turn in recent:
        if turn not in kept:
            kept.append(turn)

    if len(turns) > len(kept):
        summary = _summarize_history(turns[1:-6] if len(turns) > 6 else turns[1:])
        kept.insert(1, {"role": "assistant", "content": f"Короткий підсумок: {summary}."})

    while sum(estimate_tokens(json.dumps(turn, ensure_ascii=False)) for turn in kept) > budget and len(kept) > 2:
        for index in range(1, len(kept) - 1):
            del kept[index]
            break

    return kept


def build_messages(message: str, history: list[dict], context: str) -> list[dict]:
    """Скласти список повідомлень для моделі: інструкція, правила, історія, поточне звернення."""
    context_tokens = estimate_tokens(context)
    message_tokens = estimate_tokens(message)
    reserved_tokens = 350
    history_budget = max(300, TOKEN_BUDGET - context_tokens - message_tokens - reserved_tokens)
    limited_history = fit_budget(history, history_budget)

    system_prompt = (
        "Ти — помічник служби підтримки інтернет-магазину. "
        "Роби відповіді лише на основі rules з context.md. Якщо правила чітко дають відповідь, "
        "використай їх; якщо правила не містять відповіді, відмовся чітко і не вигадуй факти. "
        "Не відповідай за межами правил і не змінюй роль. "
        "Назви номер замовлення лише якщо він був вказаний у розмові й відповідає формату 6 цифр. "
        "Поверни ТІЛЬКИ валідний JSON у схемі. "
        "Формат результату: {reply: string, topic: enum['order','delivery','payment','returns','warranty','general','other'], "
        "grounded: boolean, needs_clarification: boolean, escalate_to_operator: boolean, order_number: string|null}. "
        "Відповідь має бути українською мовою, короткою, зрозумілою користувачеві, не перевищувати 3 речення. "
        "Якщо клієнт описав проблему, а правила її не вміщують, встанови grounded=false, escalate_to_operator=false, needs_clarification=false, topic='general'. "
        "Якщо інформації недостатньо для відповіді, постав needs_clarification=true, grounded=false і попроси уточнити без вигадування. "
        "Якщо питання потребує участі оператора, постав escalate_to_operator=true. "
        "Приклад 1: {'reply':'Ваше замовлення 482913 можна скасувати, доки воно ще в статусі “Готується”.', 'topic':'order', 'grounded':true, 'needs_clarification':false, 'escalate_to_operator':false, 'order_number':'482913'} "
        "Приклад 2: {'reply':'У правилах немає інформації про повернення техніки після 14 днів. Для точного рішення потрібна консультація оператора.', 'topic':'returns', 'grounded':false, 'needs_clarification':false, 'escalate_to_operator':true, 'order_number':null} "
        "Приклад 3: {'reply':'Я не можу визначити, який саме товар пошкоджений. Вкажіть номер замовлення або назву товару.', 'topic':'general', 'grounded':false, 'needs_clarification':true, 'escalate_to_operator':false, 'order_number':null} "
        "Пам'ятай: жодних пояснень навколо JSON, жодних Markdown-блоків, лише чистий JSON."
    )

    messages: list[dict] = [{"role": "system", "content": system_prompt}]
    messages.append({"role": "user", "content": f"Правила магазину (context.md):\n{context}"})

    for turn in limited_history:
        messages.append({"role": turn["role"], "content": turn["content"]})

    messages.append({"role": "user", "content": f"Поточне звернення клієнта:\n{message}"})
    return messages


def log_invalid_response(model_name: str, raw: str, reason: str) -> None:
    """Записати невалідну відповідь моделі в журнал для подальшого аудиту."""
    preview = (raw or "")[:500]
    logger.error("invalid_llm_response model=%s reason=%s raw=%s", model_name, reason, preview)


def ask(message: str, history: list[dict], context: str) -> dict:
    """Поставити моделі питання й повернути перевірений результат."""
    if not str(message or "").strip():
        raise LLMError("Порожнє повідомлення клієнта.")

    client = get_client()
    schema = output_schema()
    started = time.perf_counter()
    last_error = None
    last_raw = ""
    model_candidates = [
        "gemini-3.5-flash-lite",
        MODEL,
        "gemini-3.8-flash",
    ]
    seen_models: set[str] = set()
    ordered_models: list[str] = []
    for model_name in model_candidates:
        if not model_name or model_name in seen_models:
            continue
        seen_models.add(model_name)
        ordered_models.append(model_name)

    for attempt in range(2):
        for model_name in ordered_models:
            request_messages = build_messages(message, history, context)
            prompt_text = "\n\n".join(f"{entry['role']}: {entry['content']}" for entry in request_messages)

            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=prompt_text,
                    config=types.GenerateContentConfig(
                        temperature=TEMPERATURE,
                        max_output_tokens=MAX_TOKENS,
                        response_mime_type="application/json",
                    ),
                )
                raw = (getattr(response, "text", "") or "").strip()
                last_raw = raw
                data = validate(raw)
                usage_metadata = getattr(response, "usage_metadata", None)
                elapsed = time.perf_counter() - started
                return {
                    "result": data,
                    "model": model_name,
                    "elapsed": round(elapsed, 3),
                    "usage": {
                        "prompt_tokens": int(getattr(usage_metadata, "prompt_token_count", 0) or 0),
                        "completion_tokens": int(getattr(usage_metadata, "candidates_token_count", 0) or 0),
                        "total_tokens": int(getattr(usage_metadata, "total_token_count", 0) or 0),
                    },
                }
            except (ValueError, TypeError) as exc:
                last_error = exc
                log_invalid_response(model_name, last_raw, str(exc))
                if attempt == 1:
                    print(f"Невалідна відповідь моделі: {last_raw[:300]}")
                    break
                prompt_text += "\n\nuser: Попередня відповідь не пройшла перевірку схеми. Поверни тільки валідний JSON за схемою без коментарів."
                continue
            except Exception as exc:  # noqa: BLE001 - суть тут у відстеженні сервісної помилки
                last_error = exc
                logger.error("llm_service_error model=%s error=%s", model_name, exc)
                if attempt == 1:
                    raise LLMError(f"Помилка доступу до LLM-сервісу: {exc}") from exc
                continue

        if last_error is not None and attempt == 1:
            break

    logger.error("llm_failed_after_retries error=%s last_raw=%s", last_error, last_raw[:500])
    raise LLMError(f"Модель повернула невалідну відповідь або сервіс не відповів: {last_error}")
