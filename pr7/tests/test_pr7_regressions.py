import json
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app import extraction, llm, main as web, rules, schema
from app.images import ImageError, prepare
from eval.run_eval import classify, summarize_results

ROOT = Path(__file__).resolve().parents[1]


def test_all_ground_truth_documents_get_expected_rule_decision():
    expected = json.loads((ROOT / "samples" / "expected.json").read_text(encoding="utf-8"))
    for entry in expected.values():
        issues = rules.check(entry["fields"])
        decision, _ = extraction.decide(entry["fields"], issues)
        assert decision == entry["expected_decision"]


def test_other_document_is_rejected_instead_of_auto():
    document = {"document_type": "інший документ"}
    assert extraction.decide(document, [])[0] == "reject"


def test_missing_document_type_with_extracted_content_goes_to_review():
    document = {"document_type": None, "number": "A-1"}
    issues = [rules.Issue("document_type", "required_field", "Не вказано", "error")]
    assert extraction.decide(document, issues)[0] == "review"


def test_empty_extraction_is_rejected_even_when_nested_objects_exist():
    document = {
        "document_type": None,
        "supplier": {"name": None, "code": None, "iban": None},
        "buyer": {"name": None, "code": None},
        "items": [],
    }
    assert extraction.decide(document, [])[0] == "reject"


def test_prompt_extracts_buyer_instead_of_assuming_expected_company():
    image = SimpleNamespace(data=b"image", mime="image/png")
    prompt = llm.build_messages(image)[0]["content"]
    assert "Extract the buyer exactly as printed" in prompt
    assert "The buyer is the company" not in prompt


def test_quota_error_is_not_retried(monkeypatch):
    class QuotaError(Exception):
        status_code = 429

    calls = []

    def fail(*args):
        calls.append(args)
        raise QuotaError("quota exceeded")

    monkeypatch.setattr(llm, "get_client", lambda: object())
    monkeypatch.setattr(llm, "_call_model", fail)
    with pytest.raises(llm.LLMError, match="quota exceeded"):
        llm.extract(SimpleNamespace(data=b"image", mime="image/png"))
    assert len(calls) == 1


def test_truncated_model_response_is_reported_explicitly(monkeypatch):
    response = SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="length", message=SimpleNamespace(content="{}"))]
    )
    monkeypatch.setattr(llm, "get_client", lambda: object())
    def truncated_call(client, messages):
        assert client is not None and messages
        return response

    monkeypatch.setattr(llm, "_call_model", truncated_call)
    with pytest.raises(llm.LLMError, match="обрізана"):
        llm.extract(SimpleNamespace(data=b"image", mime="image/png"))


def test_model_failure_is_not_converted_into_business_rejection(monkeypatch):
    def fail_extraction(prepared):
        assert prepared.data
        raise llm.LLMError("quota")

    monkeypatch.setattr(extraction.llm, "extract", fail_extraction)
    sample = ROOT / "samples" / "clean" / "rahunok-01.png"
    with pytest.raises(llm.LLMError, match="quota"):
        extraction.process(sample.read_bytes())


def test_full_pipeline_returns_validated_document_metrics_and_raw_response(monkeypatch):
    expected = json.loads((ROOT / "samples" / "expected.json").read_text(encoding="utf-8"))
    document = expected["clean/rahunok-01.png"]["fields"]
    raw = json.dumps(document, ensure_ascii=False)
    monkeypatch.setattr(
        extraction.llm,
        "extract",
        lambda prepared: {
            "document": document,
            "raw_response": raw,
            "finish_reason": "stop",
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        } if prepared.data else pytest.fail("prepared image should contain data"),
    )
    sample = ROOT / "samples" / "clean" / "rahunok-01.png"
    result = extraction.process(sample.read_bytes())
    payload = web.result_to_dict(result)
    assert result.decision == "auto"
    assert result.issues == []
    assert payload["raw_response"] == raw
    assert payload["finish_reason"] == "stop"
    assert payload["usage"]["total_tokens"] == 30


def test_bad_and_truncated_image_inputs_are_rejected():
    with pytest.raises(ImageError):
        prepare(b"")
    with pytest.raises(ImageError):
        prepare(b"not an image")


def test_phone_photo_detects_content_cut_at_the_image_edge():
    cropped = prepare((ROOT / "samples" / "own" / "rahunok-02-phone-cropped.png").read_bytes())
    complete = prepare((ROOT / "samples" / "own" / "rahunok-02-phone.png").read_bytes())
    assert "possible_crop" in cropped.quality_issues
    assert "possible_crop" not in complete.quality_issues


def test_possible_crop_routes_otherwise_valid_invoice_to_review(monkeypatch):
    expected = json.loads((ROOT / "samples" / "expected.json").read_text(encoding="utf-8"))
    document = expected["clean/rahunok-02.png"]["fields"]
    monkeypatch.setattr(
        extraction.llm,
        "extract",
        lambda prepared: {
            "document": document,
            "raw_response": json.dumps(document, ensure_ascii=False),
            "finish_reason": "stop",
            "model": "test-model",
            "usage": None,
        },
    )
    cropped = (ROOT / "samples" / "own" / "rahunok-02-phone-cropped.png").read_bytes()
    result = extraction.process(cropped)
    assert result.decision == "review"
    assert any(issue.rule == "possible_crop" for issue in result.issues)


def test_decimal_parser_rejects_non_finite_and_boolean_values():
    assert rules.parse_decimal("NaN") is None
    assert rules.parse_decimal("Infinity") is None
    assert rules.parse_decimal(True) is None
    assert rules.parse_decimal("12 570,00") == Decimal("12570.00")


def test_ten_digit_tax_identifier_checksum_is_validated():
    assert rules.validate_company_code("3128704592")
    assert not rules.validate_company_code("3128704593")


def test_api_returns_gateway_error_for_model_failures(monkeypatch):
    def fail_extraction(content):
        assert content
        raise llm.LLMError("quota exceeded")

    monkeypatch.setattr(web.extraction, "process", fail_extraction)
    response = TestClient(web.app).post(
        "/api/extract",
        files={"image": ("invoice.png", b"image-bytes", "image/png")},
    )
    assert response.status_code == 502
    assert "quota exceeded" in response.json()["detail"]


def test_api_rejects_oversized_upload_without_processing(monkeypatch):
    monkeypatch.setattr(web, "UPLOAD_MAX_BYTES", 3)

    def should_not_process(content):
        assert content
        pytest.fail("oversized upload should not be processed")

    monkeypatch.setattr(
        web.extraction,
        "process",
        should_not_process,
    )
    response = TestClient(web.app).post(
        "/api/extract",
        files={"image": ("invoice.png", b"four", "image/png")},
    )
    assert response.status_code == 400


def test_evaluation_normalizes_amounts_iban_and_distinguishes_hallucinations():
    assert classify("total", "12 570,00", "12570.00") == "correct"
    assert classify("supplier.iban", "UA66 3510", "UA663510") == "correct"
    assert classify("valid_until", None, "2026-09-10") == "invented"


def test_evaluation_normalizes_invoice_type_and_cross_script_lookalikes():
    assert classify("document_type", "рахунок", "Рахунок на оплату") == "correct"
    assert classify("items.0.name", "Навушники X2", "Навушники Х2") == "correct"
    assert classify("items.0.name", "Коробка 30×20 см", "Коробка 30x20 см") == "correct"
    assert classify("items.0.name", "Блок живлення 1,5 А", "Блок живлення 1.5 А") == "correct"
    assert classify("items.0.name", "Навушники X2", "Навушники X3") == "error"


def test_evaluation_summary_counts_critical_and_silent_errors():
    summary = summarize_results([{
        "variant": "clean",
        "expected_decision": "auto",
        "actual_decision": "auto",
        "matches_decision": True,
        "usage": {"prompt_tokens": 20, "completion_tokens": 5},
        "field_checks": [
            {
                "classification": "error",
                "critical": True,
                "caught_by_rules": False,
                "silent_error": True,
            }
        ],
    }])
    assert summary["total_silent_errors"] == 1
    assert summary["by_variant"]["clean"]["critical_correct"] == 0
    assert summary["by_variant"]["clean"]["prompt_tokens"] == 20


def test_schema_treats_literal_nullish_strings_as_missing_values():
    doc = schema.validate(json.dumps({
        "document_type": "рахунок",
        "number": "A-1",
        "date": "2026-09-03",
        "supplier": {"name": "Vendor", "code": "38124779", "iban": "null"},
        "buyer": {"name": "Client", "code": "44172914"},
        "items": [{"name": "Item", "unit": "шт", "quantity": 1, "price": "10.00", "amount": "10.00"}],
        "total_without_vat": "10.00",
        "vat": "2.00",
        "total": "12.00",
    }))
    assert doc["supplier"]["iban"] is None

    issues = rules.check(doc)
    assert any(issue.field == "supplier.iban" and issue.severity == "error" for issue in issues)
