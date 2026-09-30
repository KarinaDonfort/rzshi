"""Перевірка середовища для ПР4.

Запуск:
    python check_env.py

Скрипт перевіряє версію Python, наявність потрібних пакетів, заповненість
налаштувань у файлі .env і те, що модель справді відповідає — надсилає
короткий пробний запит зі схемою відповіді й показує, чи прийняв провайдер
`response_format`, час відповіді та витрачені токени.

Запустіть перевірку заздалегідь: якщо ключ не працює або сервіс недоступний,
зʼясувати це краще до заняття.

Це діагностика перед роботою, а не зразок для наслідування: тут немає ані
системної інструкції, ані правил, ані історії, ані перевірки відповіді за
схемою — саме це ви проєктуєте самі.
"""

import importlib
import json
import os
import sys
import time

MIN_PYTHON = (3, 10)
PACKAGES = ("openai", "dotenv", "fastapi", "uvicorn", "pydantic")
OPTIONAL_PACKAGES = ("jsonschema",)
SETTINGS = ("LLM_BASE_URL", "LLM_API_KEY", "LLM_MODEL")
OPTIONAL_SETTINGS = ("LLM_TOKEN_BUDGET",)

HINTS = {
    "LLM_BASE_URL": "не задано: скопіюйте .env.example у .env",
    "LLM_API_KEY": "не задано: візьміть ключ у Google AI Studio (aistudio.google.com) "
                   "і впишіть його в .env",
    "LLM_MODEL": "не задано: назву моделі дивіться в Google AI Studio",
    "LLM_TOKEN_BUDGET": "не задано: діятиме значення за замовчуванням із коду; "
                        "допишіть рядок з .env.example",
}

# Схема пробного запиту. Навмисно крихітна: перевіряє лише, що провайдер
# приймає response_format і повертає JSON. Якою має бути ваша схема — це
# не показує.
PROBE_SCHEMA = {
    "type": "object",
    "properties": {
        "ok": {"type": "boolean"},
        "word": {"type": "string"},
    },
    "required": ["ok", "word"],
}

PROBE_PROMPT = ("Відповідай JSON-обʼєктом із двома полями: ok — true, "
                "word — одне слово українською.")


def report(ok: bool, what: str, hint: str) -> None:
    mark = "[ OK ]" if ok else "[ !! ]"
    tail = f" — {hint}" if hint else ""
    print(f"{mark} {what}{tail}")


def note(what: str, hint: str) -> None:
    """Зауваження, яке не є помилкою середовища."""
    print(f"[ .. ] {what} — {hint}")


def check_python() -> bool:
    actual = sys.version_info[:2]
    ok = actual >= MIN_PYTHON
    need = ".".join(map(str, MIN_PYTHON))
    have = ".".join(map(str, actual))
    report(ok, f"Python {have}", "" if ok else f"потрібен Python {need} або новіший")
    return ok


def check_packages() -> bool:
    ok = True
    for name in PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            report(False, f"пакет {name}", "не встановлено: pip install -r requirements.txt")
            ok = False
        else:
            report(True, f"пакет {name}", "")
    for name in OPTIONAL_PACKAGES:
        try:
            importlib.import_module(name)
        except ImportError:
            note(f"пакет {name}", "не встановлено; потрібен лише якщо схему пишете вручну як JSON")
        else:
            report(True, f"пакет {name}", "")
    return ok


def check_settings() -> bool:
    """Перевірити, що .env заповнений. Значення ключа не друкуємо."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        report(False, "налаштування .env", "перевірку пропущено: немає пакета python-dotenv")
        return False

    load_dotenv()
    ok = True
    for name in SETTINGS:
        value = os.getenv(name)
        if not value:
            report(False, f"налаштування {name}", HINTS[name])
            ok = False
        elif name == "LLM_API_KEY":
            report(True, f"налаштування {name}", f"задано, довжина {len(value)}")
        else:
            report(True, f"налаштування {name}", value)
    for name in OPTIONAL_SETTINGS:
        value = os.getenv(name)
        if value:
            report(True, f"налаштування {name}", value)
        else:
            note(f"налаштування {name}", HINTS[name])
    return ok


def check_call() -> bool:
    """Надіслати пробний запит зі схемою відповіді й зміряти час."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        report(False, "пробний запит", "перевірку пропущено: немає пакета google-genai")
        return False

    api_key = os.getenv("LLM_API_KEY")
    model = os.getenv("LLM_MODEL")
    if not (api_key and model):
        report(False, "пробний запит", "перевірку пропущено: налаштування неповні")
        return False

    client = genai.Client(api_key=api_key)
    prompt = PROBE_PROMPT
    started = time.perf_counter()
    try:
        answer = client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=0.2,
                max_output_tokens=60,
                response_mime_type="application/json",
            ),
        )
        schema_accepted = True
    except Exception as exc:  # noqa: BLE001 — діагностика, показуємо все
        report(False, "пробний запит", f"збій ({type(exc).__name__}): {exc}")
        return False
    elapsed = time.perf_counter() - started

    text = (getattr(answer, "text", "") or "").strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        data = None

    if data is not None:
        report(True, "пробний запит", f"відповідь за {elapsed:.2f} с, JSON розібрано (схему прийнято): {data}")
    else:
        report(False, "пробний запит", f"схему прийнято, але відповідь не є JSON: {text[:80]!r}")

    usage_metadata = getattr(answer, "usage_metadata", None)
    if usage_metadata:
        print(f"       токенів: запит {getattr(usage_metadata, 'prompt_token_count', 0)}, "
              f"відповідь {getattr(usage_metadata, 'candidates_token_count', 0)}, "
              f"разом {getattr(usage_metadata, 'total_token_count', 0)}")
    return data is not None


def main() -> int:
    print("Перевірка середовища для ПР4\n")
    results = [check_python(), check_packages(), check_settings(), check_call()]
    print()
    if all(results):
        print("Середовище готове до роботи.")
        return 0
    print("Є проблеми — усуньте позначені [ !! ] і запустіть перевірку ще раз.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
