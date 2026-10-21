"""Програмні правила: перевірка змісту вилучених даних."""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from pathlib import Path

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"
TWO_PLACES = Decimal("0.01")


@dataclass
class Issue:
    field: str
    rule: str
    message: str
    severity: str = "error"


def load_reference() -> tuple[dict, list[dict]]:
    company = json.loads((REFERENCE_DIR / "company.json").read_text(encoding="utf-8"))
    suppliers = json.loads((REFERENCE_DIR / "suppliers.json").read_text(encoding="utf-8"))
    return company, suppliers


def normalize_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        text = value.strip()
    else:
        text = str(value).strip()
    if text.lower() in {"null", "none", "n/a", "na"}:
        return ""
    for old, new in (
        ("’", "'"),
        ("ʼ", "'"),
        ("‘", "'"),
        ("“", '"'),
        ("”", '"'),
        ("«", '"'),
        ("»", '"'),
        ("\u00A0", " "),
    ):
        text = text.replace(old, new)
    return " ".join(text.split())


def parse_decimal(value) -> Decimal | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, Decimal):
        return value if value.is_finite() else None
    if isinstance(value, (int, float)):
        try:
            number = Decimal(str(value))
        except InvalidOperation:
            return None
        return number if number.is_finite() else None
    if isinstance(value, str):
        text = normalize_text(value).replace(" ", "")
        if not text:
            return None
        if "," in text and "." in text:
            if text.rfind(",") > text.rfind("."):
                text = text.replace(".", "").replace(",", ".")
            else:
                text = text.replace(",", "")
        elif "," in text:
            text = text.replace(",", ".")
        try:
            number = Decimal(text)
            return number if number.is_finite() else None
        except InvalidOperation:
            return None
    return None


def format_money(value: Decimal) -> str:
    return str(value.quantize(TWO_PLACES, rounding=ROUND_HALF_UP))


def parse_date(value) -> datetime | None:
    if value is None:
        return None
    text = normalize_text(value)
    if not text:
        return None
    patterns = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%S.%f",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y",
        "%d/%m/%Y",
        "%d-%m-%Y",
        "%d %B %Y",
        "%d %b %Y",
        "%Y.%m.%d",
    ]
    for fmt in patterns:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    month_map = {
        "січня": "January", "лютого": "February", "березня": "March", "квітня": "April",
        "травня": "May", "червня": "June", "липня": "July", "серпня": "August",
        "вересня": "September", "жовтня": "October", "листопада": "November", "грудня": "December",
    }
    cleaned = text
    for ua, en in month_map.items():
        cleaned = cleaned.replace(ua, en)
    cleaned = re.sub(r"\s+р\.?$", "", cleaned)
    for fmt in ["%d %B %Y", "%d %b %Y"]:
        try:
            return datetime.strptime(cleaned, fmt)
        except ValueError:
            continue
    return None


def validate_iban(value) -> bool:
    text = normalize_text(value).replace(" ", "").upper()
    if not text or len(text) != 29 or not re.fullmatch(r"UA\d{2}[A-Z0-9]{25}", text):
        return False
    rearranged = text[4:] + text[:4]
    digits = ""
    for ch in rearranged:
        if ch.isdigit():
            digits += ch
        else:
            digits += str(ord(ch) - 55)
    remainder = int(digits) % 97
    return remainder == 1


def validate_company_code(value) -> bool:
    text = normalize_text(value)
    if not text or not text.isdigit():
        return False
    if text == "0" * len(text):
        return False
    if len(text) == 8:
        weight_sets = ([1, 2, 3, 4, 5, 6, 7], [3, 4, 5, 6, 7, 8, 9], [1, 2, 3, 4, 5, 7, 9])
        checksum = int(text[-1])
        for weights in weight_sets:
            total = sum(int(d) * w for d, w in zip(text[:-1], weights))
            control = total % 11
            if control == 10:
                control = 0
            if control == checksum:
                return True
        return False
    if len(text) == 10:
        weights = (-1, 5, 7, 9, 4, 6, 10, 5, 7)
        control = sum(int(digit) * weight for digit, weight in zip(text[:9], weights)) % 11
        if control == 10:
            control = 0
        return int(text[-1]) == control
    return False


def normalize_name(value) -> str:
    return normalize_text(value).lower().replace("\"", "").replace("'", "")


def check(document: dict) -> list[Issue]:
    issues: list[Issue] = []
    company, suppliers = load_reference()
    expected_company = normalize_name(company["name"])
    supplier_ref = {normalize_name(item.get("name", "")): item for item in suppliers}

    def add(field: str, rule: str, message: str, severity: str = "error"):
        issues.append(Issue(field=field, rule=rule, message=message, severity=severity))

    if not isinstance(document, dict):
        return [Issue(field="<root>", rule="document_type", message="Документ не є обʼєктом.", severity="error")]

    document_type = normalize_text(document.get("document_type")).lower()
    if not document_type:
        add("document_type", "required_field", "Тип документа не вказано.", "error")
    else:
        allowed_document_types = {
            "рахунок",
            "рахунок на оплату",
            "invoice",
            "invoice for payment",
        }
        if document_type not in allowed_document_types:
            add("document_type", "document_type", "Тип документа не є рахунком на оплату.", "error")

    for field, label in [("number", "Номер рахунку"), ("date", "Дата рахунку")]:
        if not normalize_text(document.get(field)):
            add(field, "required_field", f"Обовʼязкове поле {label} відсутнє.", "error")

    supplier = document.get("supplier") or {}
    buyer = document.get("buyer") or {}

    if not isinstance(supplier, dict):
        add("supplier", "required_field", "Блок постачальника відсутній або не є обʼєктом.", "error")
    else:
        for field, label in [("name", "Назва постачальника"), ("code", "Код постачальника"), ("iban", "IBAN постачальника")]:
            if not normalize_text(supplier.get(field)):
                add(f"supplier.{field}", "required_field", f"Обовʼязкове поле {label} відсутнє.", "error")
        supplier_name = normalize_name(supplier.get("name"))
        if supplier_name and supplier_name not in supplier_ref:
            add("supplier.name", "supplier_known", "Постачальник не знайдений у довіднику.", "warning")
        else:
            ref = supplier_ref.get(supplier_name)
            supplier_iban = normalize_text(supplier.get("iban")).replace(" ", "").upper()
            reference_iban = normalize_text(ref.get("iban") if ref else None).replace(" ", "").upper()
            if ref and supplier_iban and reference_iban:
                if supplier_iban != reference_iban:
                    add("supplier.iban", "supplier_iban_match", "IBAN постачальника не збігається з довідником.", "error")
            supplier_code = normalize_text(supplier.get("code"))
            reference_code = normalize_text(ref.get("code") if ref else None)
            if ref and supplier_code and reference_code:
                if supplier_code != reference_code:
                    add("supplier.code", "supplier_code_match", "Код постачальника не збігається з довідником.", "error")

        if supplier.get("iban") is not None and not validate_iban(supplier.get("iban")):
            add("supplier.iban", "iban_checksum", "IBAN має некоректну контрольну суму або формат.", "error")
        code = normalize_text(supplier.get("code"))
        if code and not validate_company_code(code):
            add("supplier.code", "company_code", "Код постачальника не має коректного контрольного розряду.", "error")

    if not isinstance(buyer, dict):
        add("buyer", "required_field", "Блок покупця відсутній або не є обʼєктом.", "error")
    else:
        for field, label in [("name", "Назва покупця"), ("code", "Код покупця")]:
            if not normalize_text(buyer.get(field)):
                add(f"buyer.{field}", "required_field", f"Обовʼязкове поле {label} відсутнє.", "error")
        if normalize_name(buyer.get("name")) and normalize_name(buyer.get("name")) != expected_company:
            add("buyer.name", "buyer_company", "Покупець не є компанією «Сузірʼя Рітейл».", "error")
        buyer_code = normalize_text(buyer.get("code"))
        if buyer_code and not validate_company_code(buyer_code):
            add("buyer.code", "company_code", "Код покупця не має коректного контрольного розряду.", "error")

    if document.get("date"):
        dt = parse_date(document.get("date"))
        if dt is None:
            add("date", "date_format", "Дата рахунку має некоректний формат.", "error")
        else:
            today = datetime.now(timezone.utc).date()
            if dt.date() > today:
                add("date", "date_future", "Дата рахунку не може бути у майбутньому.", "error")

    if document.get("valid_until"):
        end_dt = parse_date(document.get("valid_until"))
        if end_dt is None:
            add("valid_until", "date_format", "Строк дії має некоректний формат.", "error")
        else:
            start_dt = parse_date(document.get("date"))
            if start_dt and end_dt.date() < start_dt.date():
                add("valid_until", "valid_until", "Строк дії не може бути раніше дати рахунку.", "error")

    for field, label in [
        ("total_without_vat", "Сума без ПДВ"),
        ("vat", "ПДВ"),
        ("total", "Усього до сплати"),
    ]:
        value = document.get(field)
        if not normalize_text(value):
            add(field, "required_field", f"Обовʼязкове поле {label} відсутнє.", "error")
            continue
        if parse_decimal(value) is None:
            add(field, "numeric_format", f"Поле {label} має бути числом або рядком з числом.", "error")

    items = document.get("items") or []
    if not isinstance(items, list) or not items:
        add("items", "required_field", "У документа немає позицій.", "error")
    else:
        total_amount = Decimal("0")
        for idx, item in enumerate(items):
            if not isinstance(item, dict):
                add(f"items.{idx}", "items_shape", "Позиція не є обʼєктом.", "error")
                continue
            if not normalize_text(item.get("name")):
                add(f"items.{idx}.name", "required_field", "У позиції відсутнє найменування.", "error")
            if not normalize_text(item.get("unit")):
                add(f"items.{idx}.unit", "required_field", "У позиції відсутня одиниця виміру.", "error")
            qty = parse_decimal(item.get("quantity"))
            price = parse_decimal(item.get("price"))
            amount = parse_decimal(item.get("amount"))
            if qty is None or price is None or amount is None:
                add(f"items.{idx}", "item_numbers", "У позиції відсутні кількість, ціна або сума.", "error")
                continue
            expected = (qty * price).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            if abs(expected - amount) > Decimal("0.01"):
                add(f"items.{idx}.amount", "item_arithmetic", f"Сума позиції {idx + 1} не дорівнює кількість × ціна.", "error")
            total_amount += amount

        total_without_vat = parse_decimal(document.get("total_without_vat"))
        if total_without_vat is not None and abs(total_amount - total_without_vat) > Decimal("0.01"):
            add("total_without_vat", "total_arithmetic", "Сума позицій не дорівнює підсумку без ПДВ.", "error")

        vat = parse_decimal(document.get("vat"))
        vat_expected = Decimal("0")
        ref_supplier = supplier_ref.get(supplier_name) if supplier_name else None
        if ref_supplier is not None and ref_supplier.get("vat_payer") is True:
            base_total = total_without_vat if total_without_vat is not None else total_amount
            vat_expected = (base_total * Decimal("0.2")).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
        if vat is not None:
            if abs(vat - vat_expected) > Decimal("0.01"):
                add("vat", "vat_arithmetic", "ПДВ не відповідає 20% від суми без ПДВ або дорівнює нулю для неплатника.", "error")

        total = parse_decimal(document.get("total"))
        if total is not None:
            base_total = total_without_vat if total_without_vat is not None else total_amount
            expected_vat = vat if vat is not None else vat_expected
            expected_total = (base_total + expected_vat).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
            if abs(total - expected_total) > Decimal("0.01"):
                add("total", "total_arithmetic", "Усього до сплати не дорівнює сумі без ПДВ і ПДВ.", "error")
    return issues
