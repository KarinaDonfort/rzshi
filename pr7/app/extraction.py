"""Конвеєр обробки документа."""

import time
from dataclasses import dataclass, field

from . import llm, rules
from .images import ImageError, prepare


@dataclass
class Result:
    decision: str
    reasons: list[str] = field(default_factory=list)
    document: dict | None = None
    issues: list[rules.Issue] = field(default_factory=list)
    image: dict = field(default_factory=dict)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None
    raw_response: str | None = None
    finish_reason: str | None = None


def decide(document: dict, issues: list[rules.Issue]) -> tuple[str, list[str]]:
    """Вирішити долю документа за вилученими полями й знайденими проблемами."""
    if not isinstance(document, dict):
        return "reject", ["Не вдалося отримати структуровані поля з документа."]

    def has_value(value) -> bool:
        if isinstance(value, dict):
            return any(has_value(item) for item in value.values())
        if isinstance(value, list):
            return any(has_value(item) for item in value)
        return value is not None and bool(str(value).strip())

    doc_type = str(document.get("document_type") or "").strip().lower()
    valid_doc_types = {"рахунок", "рахунок на оплату", "invoice", "invoice for payment"}
    if doc_type and doc_type not in valid_doc_types:
        return "reject", ["Документ не є рахунком на оплату."]
    if not doc_type and not any(
        has_value(value) for key, value in document.items() if key != "document_type"
    ):
        return "reject", ["Не вдалося визначити тип документа або вилучити поля."]

    errors = [issue for issue in issues if issue.severity == "error"]
    warnings = [issue for issue in issues if issue.severity == "warning"]

    if errors:
        reasons = [issue.message for issue in errors[:5]]
        if warnings:
            reasons.extend(issue.message for issue in warnings[:3])
        return "review", reasons

    if warnings:
        return "review", [issue.message for issue in warnings[:5]]

    return "auto", ["Вилучені поля й перевірки пройдено; документ придатний для реєстру платежів."]


def process(content: bytes) -> Result:
    """Обробити файл: підготувати → вилучити → перевірити → вирішити."""
    prepare_started = time.perf_counter()
    try:
        prepared = prepare(content)
    except ImageError as exc:
        return Result(
            decision="reject",
            reasons=[str(exc)],
            document=None,
            issues=[],
            image={"original": {}, "sent": {}},
            elapsed={"prepare": time.perf_counter() - prepare_started, "extraction": 0.0, "checks": 0.0},
        )

    prepare_elapsed = time.perf_counter() - prepare_started
    image_info = {"original": prepared.original, "sent": prepared.sent}

    started = time.perf_counter()
    extracted = llm.extract(prepared)
    extraction_elapsed = time.perf_counter() - started

    document = extracted.get("document")
    model = extracted.get("model")
    usage = extracted.get("usage")
    raw_response = extracted.get("raw_response")
    finish_reason = extracted.get("finish_reason")
    started = time.perf_counter()
    issues = rules.check(document) if isinstance(document, dict) else []
    if "possible_crop" in prepared.quality_issues:
        issues.append(
            rules.Issue(
                field="image",
                rule="possible_crop",
                message="Вміст документа торкається краю зображення; перевірте, чи рахунок не обрізано.",
                severity="warning",
            )
        )
    checks_elapsed = time.perf_counter() - started
    decision, reasons = decide(document, issues)

    return Result(
        decision=decision,
        reasons=reasons,
        document=document,
        issues=issues,
        image=image_info,
        model=model,
        elapsed={"prepare": prepare_elapsed, "extraction": extraction_elapsed, "checks": checks_elapsed},
        usage=usage,
        raw_response=raw_response,
        finish_reason=finish_reason,
    )
