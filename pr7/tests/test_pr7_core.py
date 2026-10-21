import json

from app import extraction, rules, schema


def test_output_schema_has_invoice_fields():
    s = schema.output_schema()
    assert s["type"] == "object"
    assert "document_type" in s["properties"]
    assert "supplier" in s["properties"]
    assert "items" in s["properties"]


def test_validate_accepts_normalized_invoice_document():
    raw = {
        "document_type": "рахунок",
        "number": "ОА-0917",
        "date": "2026-09-03",
        "valid_until": "2026-09-10",
        "supplier": {
            "name": "ТОВ «Оріон Аудіо Дистрибуція»",
            "code": "38124779",
            "iban": "UA663510050000026004017723561",
        },
        "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
        "items": [{
            "name": "Навушники",
            "unit": "шт",
            "quantity": 20,
            "price": "2875.00",
            "amount": "57500.00",
        }],
        "total_without_vat": "57500.00",
        "vat": "11500.00",
        "total": "69000.00",
    }
    doc = schema.validate(json.dumps(raw))
    assert doc["document_type"] == "рахунок"
    assert doc["items"][0]["quantity"] == 20


def test_rules_catch_bad_iban_and_supplier_mismatch():
    doc = {
        "document_type": "рахунок",
        "number": "A-1",
        "date": "2026-09-03",
        "supplier": {"name": "ТОВ «Оріон Аудіо Дистрибуція»", "code": "38124779", "iban": "UA000000000000000000000000000"},
        "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
        "items": [{"name": "X", "unit": "шт", "quantity": 1, "price": "10.00", "amount": "10.00"}],
        "total_without_vat": "10.00",
        "vat": "2.00",
        "total": "12.00",
    }
    issues = rules.check(doc)
    rules_by_field = {issue.field: issue.rule for issue in issues}
    assert any("supplier.iban" in field for field in rules_by_field)


def test_rules_accept_realistic_document_type_and_spaced_iban():
    doc = {
        "document_type": "Рахунок на оплату",
        "number": "ОА-0917",
        "date": "2026-09-03",
        "valid_until": "2026-09-10",
        "supplier": {
            "name": "ТОВ «Оріон Аудіо Дистрибуція»",
            "code": "38124779",
            "iban": "UA66 3510 0500 0002 6004 0177 2356 1",
        },
        "buyer": {"name": "ТОВ «Сузірʼя Рітейл»", "code": "44172914"},
        "items": [{
            "name": "Навушники бездротові Оріон Х2, чорні",
            "unit": "шт",
            "quantity": 20,
            "price": "2875.00",
            "amount": "57500.00",
        }],
        "total_without_vat": "57500.00",
        "vat": "11500.00",
        "total": "69000.00",
    }
    issues = rules.check(doc)
    assert not any(issue.field == "document_type" for issue in issues)
    assert not any(issue.field == "supplier.iban" for issue in issues)


def test_decide_accepts_invoice_type_with_payment_phrase():
    document = {"document_type": "Рахунок на оплату"}
    issues = []
    decision, reasons = extraction.decide(document, issues)
    assert decision == "auto"
    assert reasons
