"""Run and summarize the invoice extraction evaluation."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from decimal import Decimal, InvalidOperation
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.extraction import process
from app.llm import MODEL

SAMPLES_DIR = ROOT / "samples"
EXPECTED_FILE = SAMPLES_DIR / "expected.json"
RESULTS_FILE = Path(__file__).resolve().parent / "results.json"
CRITICAL_FIELDS = {"supplier.iban", "supplier.code", "total"}
CONFUSABLE_LETTERS = str.maketrans({
    "а": "a", "в": "b", "с": "c", "е": "e", "н": "h", "і": "i",
    "к": "k", "м": "m", "о": "o", "р": "p", "т": "t", "х": "x",
    "у": "y", "×": "x",
})
DOCUMENT_TYPE_ALIASES = {
    "рахунок": "invoice",
    "рахунок на оплату": "invoice",
    "invoice": "invoice",
    "invoice for payment": "invoice",
}
EXPECTED_RULES = {
    "arithmetic_line": {"item_arithmetic"},
    "iban_checksum": {"iban_checksum"},
    "iban_registry": {"supplier_iban_match"},
    "buyer": {"buyer_company"},
    "document_type": {"document_type"},
    "missing_required": {"required_field", "item_numbers"},
}


def normalize_text(value):
    if value is None:
        return None
    text = str(value).strip()
    for old, new in {
        "’": "'", "ʼ": "'", "‘": "'", "«": '"', "»": '"',
        "\u00a0": " ", "\u2011": "-", "\u2013": "-", "\u2014": "-",
    }.items():
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text)


def normalize_amount(value):
    if value is None or isinstance(value, bool):
        return None
    text = normalize_text(value).replace(" ", "")
    if not text:
        return None
    if "," in text and "." in text:
        text = text.replace(".", "").replace(",", ".") if text.rfind(",") > text.rfind(".") else text.replace(",", "")
    elif "," in text:
        text = text.replace(",", ".")
    try:
        amount = Decimal(text)
        return amount.quantize(Decimal("0.01")) if amount.is_finite() else None
    except (InvalidOperation, ValueError):
        return None


def normalize_value(path, value):
    if value is None:
        return None
    if path.endswith(".iban"):
        return re.sub(r"\s+", "", str(value)).upper()
    if path.endswith(".code"):
        return re.sub(r"\s+", "", str(value))
    if isinstance(value, (int, float, Decimal)) and not isinstance(value, bool):
        return normalize_amount(value)
    text = normalize_text(value)
    if text is None or text.lower() in {"", "null", "none"}:
        return None
    text = text.lower()
    if path == "document_type":
        return DOCUMENT_TYPE_ALIASES.get(text, text)
    text = text.translate(CONFUSABLE_LETTERS)
    text = re.sub(r"(?<=\d),(?=\d)", ".", text)
    if re.fullmatch(r"-?\d+(?:[.,]\d+)?", text.replace(" ", "")):
        return normalize_amount(text)
    return text


def flatten(value, prefix=""):
    rows = []
    if isinstance(value, dict):
        for key, item in value.items():
            path = f"{prefix}.{key}" if prefix else key
            rows.extend(flatten(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            rows.extend(flatten(item, f"{prefix}.{index}"))
    else:
        rows.append((prefix, value))
    return rows


def classify(path, expected, actual):
    exp = normalize_value(path, expected)
    act = normalize_value(path, actual)
    if exp is None and act is None:
        return "correct"
    if exp is None:
        return "invented"
    if act is None:
        return "missing"
    return "correct" if exp == act else "error"


def issue_catches_path(issue, path):
    field = issue.get("field", "")
    return field == path or field.startswith(path + ".") or path.startswith(field + ".")


def summarize_sample(path: Path, expected_entry: dict):
    result = process(path.read_bytes())
    expected_fields = expected_entry.get("fields", {})
    actual_fields = result.document or {}
    expected_rows = dict(flatten(expected_fields))
    actual_rows = dict(flatten(actual_fields))
    all_paths = list(expected_rows)
    all_paths.extend(field for field in actual_rows if field not in expected_rows)
    issues = [vars(issue) for issue in result.issues]

    field_checks = []
    for field in all_paths:
        expected = expected_rows.get(field)
        actual = actual_rows.get(field)
        status = classify(field, expected, actual)
        caught = any(issue_catches_path(issue, field) for issue in issues)
        field_checks.append({
            "field": field,
            "expected": expected,
            "actual": actual,
            "classification": status,
            "caught_by_rules": caught if status in {"error", "invented"} else None,
            "silent_error": status in {"error", "invented"} and result.decision == "auto" and not caught,
            "critical": field in CRITICAL_FIELDS,
        })

    expected_checks = set(expected_entry.get("expected_checks", []))
    actual_rules = {issue["rule"] for issue in issues}
    expected_rules_found = {
        check for check in expected_checks
        if EXPECTED_RULES.get(check, set()) & actual_rules
    }
    tokens = result.usage or {}
    return {
        "file": path.relative_to(ROOT).as_posix(),
        "variant": expected_entry.get("variant", path.parent.name),
        "status": "ok",
        "expected_decision": expected_entry.get("expected_decision"),
        "actual_decision": result.decision,
        "matches_decision": result.decision == expected_entry.get("expected_decision"),
        "expected_checks": sorted(expected_checks),
        "triggered_expected_checks": sorted(expected_rules_found),
        "missed_expected_checks": sorted(expected_checks - expected_rules_found),
        "issues": issues,
        "reasons": result.reasons,
        "elapsed": result.elapsed,
        "usage": tokens,
        "raw_response": result.raw_response,
        "finish_reason": result.finish_reason,
        "field_checks": field_checks,
    }


def summarize_results(rows):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row.get("variant", "unknown")].append(row)

    by_variant = {}
    for variant, group in sorted(grouped.items()):
        checks = [field for row in group for field in row.get("field_checks", [])]
        counts = {key: sum(item["classification"] == key for item in checks)
                  for key in ("correct", "error", "missing", "invented")}
        critical = [item for item in checks if item["critical"]]
        by_variant[variant] = {
            "files": len(group),
            **counts,
            "critical_fields": len(critical),
            "critical_correct": sum(item["classification"] == "correct" for item in critical),
            "caught_errors_and_inventions": sum(
                item["classification"] in {"error", "invented"} and item["caught_by_rules"]
                for item in checks
            ),
            "silent_errors": sum(item["silent_error"] for item in checks),
            "decision_matches": sum(row.get("matches_decision", False) for row in group),
            "prompt_tokens": sum((row.get("usage") or {}).get("prompt_tokens") or 0 for row in group),
            "completion_tokens": sum((row.get("usage") or {}).get("completion_tokens") or 0 for row in group),
        }

    matrix = defaultdict(lambda: defaultdict(int))
    for row in rows:
        matrix[row.get("expected_decision", "unknown")][row.get("actual_decision", "error")] += 1
    fields = [field for row in rows for field in row.get("field_checks", [])]
    return {
        "files": len(rows),
        "runtime_failures": sum(row.get("status") == "error" for row in rows),
        "by_variant": by_variant,
        "decision_matrix": {expected: dict(actuals) for expected, actuals in matrix.items()},
        "total_silent_errors": sum(field["silent_error"] for field in fields),
        "total_field_classifications": {
            key: sum(field["classification"] == key for field in fields)
            for key in ("correct", "error", "missing", "invented")
        },
    }


def comparison_metrics(row):
    checks = row.get("field_checks", [])
    return {
        "correct": sum(check.get("classification") == "correct" for check in checks),
        "error": sum(check.get("classification") == "error" for check in checks),
        "missing": sum(check.get("classification") == "missing" for check in checks),
        "invented": sum(check.get("classification") == "invented" for check in checks),
        "prompt_tokens": (row.get("usage") or {}).get("prompt_tokens") or 0,
        "completion_tokens": (row.get("usage") or {}).get("completion_tokens") or 0,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=RESULTS_FILE, help="JSON output path")
    parser.add_argument("--compare", type=Path, help="compare with a previous results JSON")
    args = parser.parse_args()
    expected = json.loads(EXPECTED_FILE.read_text(encoding="utf-8"))
    rows = []

    sample_paths = []
    for folder_name in ("clean", "degraded", "own"):
        folder = SAMPLES_DIR / folder_name
        if folder.exists():
            sample_paths.extend(sorted(
                path for path in folder.iterdir()
                if path.is_file() and path.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
            ))

    for path in sample_paths:
        key = path.relative_to(SAMPLES_DIR).as_posix()
        expected_entry = expected.get(key)
        if expected_entry is None:
            rows.append({
                "file": key,
                "variant": "own",
                "status": "not_in_expected",
                "error": "No matching entry in samples/expected.json.",
            })
            continue
        try:
            rows.append(summarize_sample(path, expected_entry))
        except Exception as exc:
            rows.append({
                "file": key,
                "variant": expected_entry.get("variant", path.parent.name),
                "status": "error",
                "error": str(exc),
                "expected_decision": expected_entry.get("expected_decision"),
            })

    output = {
        "experiment": {
            "model": MODEL,
            "change": "Buyer extraction prompt: read the printed buyer; do not assume the expected company.",
        },
        "summary": summarize_results(rows),
        "results": rows,
    }
    if args.compare and args.compare.exists():
        previous = json.loads(args.compare.read_text(encoding="utf-8"))
        old_rows = previous if isinstance(previous, list) else previous.get("results", [])
        old_map = {row.get("file"): row for row in old_rows}
        output["comparison"] = {
            "files": {
                row["file"]: {
                    "previous_decision": old_map[row["file"]].get("actual_decision"),
                    "current_decision": row.get("actual_decision"),
                    "decision_changed": old_map[row["file"]].get("actual_decision") != row.get("actual_decision"),
                    "previous_fields": comparison_metrics(old_map[row["file"]]),
                    "current_fields": comparison_metrics(row),
                }
                for row in rows if row.get("file") in old_map
            },
            "compared_files": sum(row.get("file") in old_map for row in rows),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))
    print(f"Saved detailed results to {args.output}")
    return 1 if output["summary"]["runtime_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
