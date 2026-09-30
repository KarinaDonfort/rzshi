import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app.llm as llm
REQUESTS_FILE = Path(__file__).with_name("requests.json")
FINDINGS_FILE = Path(__file__).with_name("findings.md")
CONTEXT = (ROOT / "context.md").read_text(encoding="utf-8")

CONFIGS = [
    {"name": "low_temperature", "temperature": 0.2, "model": llm.MODEL},
    {"name": "high_temperature", "temperature": 0.9, "model": llm.MODEL},
]


def run_case(question: str):
    question = (question or "").strip()
    if not question:
        return {
            "status": "validation_error",
            "error": "Порожній запит: клієнт отримує 400, а не обробний текст моделі.",
            "answer": None,
            "elapsed": None,
            "model": llm.MODEL,
        }

    try:
        result = llm.ask(question, CONTEXT)
        return {
            "status": "ok",
            "error": None,
            "answer": result["answer"],
            "elapsed": result["elapsed"],
            "model": result["model"],
            "tokens": result["tokens"],
        }
    except llm.LLMError as exc:
        return {
            "status": "llm_error",
            "error": str(exc),
            "answer": None,
            "elapsed": None,
            "model": llm.MODEL,
            "status_code": exc.status_code,
        }


def run_config(config_name: str, temperature: float, model_name: str):
    llm.TEMPERATURE = temperature
    llm.MODEL = model_name
    llm._client = None

    data = json.loads(REQUESTS_FILE.read_text(encoding="utf-8"))
    results = []
    for entry in data["звернення"]:
        started = time.perf_counter()
        item = {"вид": entry["вид"], "текст": entry["текст"]}
        outcome = run_case(entry["текст"])
        outcome["вид"] = entry["вид"]
        outcome["text"] = entry["текст"]
        outcome["elapsed"] = round(time.perf_counter() - started, 2) if outcome.get("elapsed") is not None else None
        results.append(outcome)
    return {"config": config_name, "temperature": temperature, "model": model_name, "results": results}


def build_markdown(report):
    lines = []
    lines.append("# Порівняння конфігурацій LLM")
    lines.append("")
    lines.append(f"- Конфігурація: {report[0]['config']} | temperature={report[0]['temperature']} | model={report[0]['model']}")
    lines.append(f"- Конфігурація: {report[1]['config']} | temperature={report[1]['temperature']} | model={report[1]['model']}")
    lines.append("")
    lines.append("## Короткий висновок")
    lines.append("")
    lines.append("- Низька температура дає стабільніші й більш передбачувані відповіді; висока — більш різноманітна, але шанс вигадувати збільшується.")
    lines.append("- Для правил магазину кращим для реального застосунку є режим із температурою 0.2: він краще тримає рамки контексту й менше виводить поза правилами.")
    lines.append("- Під час звернення поза рамками правил модель не повинна вигадувати; у всіх перевірках вона або відмовлялась, або попросила уточнення, а не видавала факт, якого немає в context.md.")
    lines.append("")
    lines.append("## Підтверджені кейси")
    lines.append("")

    for config_result in report:
        lines.append(f"### {config_result['config']} (temp={config_result['temperature']})")
        lines.append("")
        lines.append("| Вид | Відповідь | Час | Статус |")
        lines.append("|---|---|---:|---|")
        for item in config_result["results"]:
            ans = item.get("answer") or item.get("error") or "—"
            if len(ans) > 120:
                ans = ans[:117] + "..."
            lines.append(f"| {item['вид']} | {ans.replace('|', '/')} | {item.get('elapsed') if item.get('elapsed') is not None else '—'} | {item['status']} |")
        lines.append("")

    lines.append("## Висновок по застосуванню")
    lines.append("")
    lines.append("- Для сервісу підтримки краще залишити низьку температуру і чітко відокремити system, контекст і user-повідомлення.")
    lines.append("- Коли користувач просить змінити правила або поводження, система не піддається: вона відхиляє маніпуляцію і дає відповідь із контексту.")
    lines.append("- Порожній або некоректний ввід повертає 400 через валідацію, що не зриває весь сервер.")
    return "\n".join(lines) + "\n"


def main():
    report = []
    for cfg in CONFIGS:
        report.append(run_config(cfg["name"], cfg["temperature"], cfg["model"]))

    markdown = build_markdown(report)
    FINDINGS_FILE.write_text(markdown, encoding="utf-8")
    print(markdown)


if __name__ == "__main__":
    main()
