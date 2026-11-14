"""Інструменти помічника: що модель може попросити виконати і як це
перевіряється.

Цей модуль є єдиним місцем, яке знає про модель і про сервіс магазину.
Модель бачить лише опис інструментів і відповідає викликами, а код тут
перевіряє їх перед виконанням.
"""

import json
import logging
import re
from dataclasses import dataclass

from jsonschema import ValidationError, validate

from shop import service

logger = logging.getLogger(__name__)


@dataclass
class Context:
    customer_id: str


@dataclass
class ToolResult:
    status: str
    content: dict | list | str | None = None
    reason: str | None = None
    arguments: dict | None = None


TOOL_SPECS: dict[str, dict] = {
    "list_orders": {
        "type": "function",
        "function": {
            "name": "list_orders",
            "description": "Показати список замовлень поточного клієнта з датами, статусом і сумою.",
            "parameters": {
                "type": "object",
                "properties": {},
                "additionalProperties": False,
            },
        },
    },
    "get_order": {
        "type": "function",
        "function": {
            "name": "get_order",
            "description": "Показати повну інформацію про конкретне замовлення поточного клієнта, включаючи статус, товари, доставку та номер відправлення. Не використовуйте для чужих замовлень.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "string",
                        "description": "Номер замовлення, наприклад 10458.",
                        "pattern": r"^[0-9]+$",
                    }
                },
                "required": ["order_id"],
                "additionalProperties": False,
            },
        },
    },
    "search_products": {
        "type": "function",
        "function": {
            "name": "search_products",
            "description": "Знайти товар у каталозі за словами, категорією або максимальною ціною.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Ключові слова для пошуку в назві, категорії або описі товару."},
                    "category": {"type": "string", "description": "Категорія товару, наприклад 'навушники' або 'телевізори'."},
                    "max_price": {
                        "anyOf": [
                            {"type": "string", "pattern": r"^[0-9]+(?:\.[0-9]{1,2})?$"},
                            {"type": "number", "minimum": 0},
                        ],
                        "description": "Максимальна ціна в грн.",
                    },
                    "limit": {"type": "integer", "minimum": 1, "maximum": 50, "description": "Максимальна кількість результатів."},
                },
                "additionalProperties": False,
            },
        },
    },
    "get_product": {
        "type": "function",
        "function": {
            "name": "get_product",
            "description": "Показати картку товару: опис, ціну, гарантію, чи повертається товар і інші властивості.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "pattern": r"^[A-Z0-9-]+$", "description": "Артикул товару, наприклад ALT-AX3."}
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    "get_stock": {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "Перевірити наявність товару на складі і дату передбачуваного поповнення, якщо товару немає.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sku": {"type": "string", "pattern": r"^[A-Z0-9-]+$", "description": "Артикул товару."}
                },
                "required": ["sku"],
                "additionalProperties": False,
            },
        },
    },
    "delivery_quote": {
        "type": "function",
        "function": {
            "name": "delivery_quote",
            "description": "Порахувати вартість і строк доставки набору товарів у конкретне місто за обраним способом доставки.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "minLength": 1, "description": "Назва міста в Україні."},
                    "method": {"type": "string", "enum": ["branch", "courier", "pickup"], "description": "Спосіб доставки: відділення, кур'єр або самовивіз."},
                    "items": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "sku": {"type": "string", "pattern": r"^[A-Z0-9-]+$", "description": "Артикул товару."},
                                "quantity": {"type": "integer", "minimum": 1, "description": "Кількість одиниць."},
                            },
                            "required": ["sku", "quantity"],
                            "additionalProperties": False,
                        },
                        "minItems": 1,
                    },
                },
                "required": ["city", "method", "items"],
                "additionalProperties": False,
            },
        },
    },
    "create_return": {
        "type": "function",
        "function": {
            "name": "create_return",
            "description": "Оформити заявку на повернення товару з доставленого замовлення. Працює лише з товарами поточного клієнта й лише за чинними правилами повернення.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {"type": "string", "pattern": r"^[0-9]+$", "description": "Номер замовлення, з якого повертають товар."},
                    "sku": {"type": "string", "pattern": r"^[A-Z0-9-]+$", "description": "Артикул товару в замовленні."},
                    "reason": {"type": "string", "enum": ["not_suitable", "defect", "wrong_item", "damaged"], "description": "Причина повернення: не підійшов, дефект, неправильний товар або пошкодження при доставці."},
                    "quantity": {"type": "integer", "minimum": 1, "description": "Кількість одиниць для повернення."},
                    "comment": {"type": "string", "description": "Додатковий коментар, якщо клієнт його вказав."},
                },
                "required": ["order_id", "sku", "reason"],
                "additionalProperties": False,
            },
        },
    },
}


def _sanitize_order(order: dict) -> dict:
    """Повернути лише поля, які потрібні для відповіді клієнтові."""
    delivery = dict(order.get("delivery", {}))
    for key in ["tracking", "point", "carrier", "address"]:
        delivery.pop(key, None)
    item_list = []
    for item in order.get("items", []):
        item_list.append({
            "sku": item.get("sku"),
            "name": item.get("name"),
            "quantity": item.get("quantity"),
            "price": item.get("price"),
        })
    out = {
        "order_id": order.get("order_id"),
        "status": order.get("status"),
        "status_label": order.get("status_label"),
        "created_at": order.get("created_at"),
        "total": order.get("total"),
        "delivery": delivery,
        "items": item_list,
    }
    tracking = order.get("delivery", {}).get("tracking")
    if tracking:
        out["tracking"] = tracking
    return out


def _sanitize_product(product: dict) -> dict:
    """Return customer-facing product fields and remove instruction-like supplier text."""
    clean = {
        key: product[key]
        for key in ["sku", "name", "category", "price", "weight_kg", "warranty_months", "returnable", "description"]
        if key in product
    }
    description = clean.get("description")
    if isinstance(description, str):
        unsafe_text = re.search(
            r"(?:примітка\s+для\s+ai|ai[-\s]?асистент|assistant\s*:|ignore\s+(?:all\s+)?previous\s+instructions)",
            description,
            flags=re.IGNORECASE,
        )
        if unsafe_text:
            clean["description"] = description[:unsafe_text.start()].rstrip()
    return clean


def _error_result(exc: Exception) -> ToolResult:
    code = getattr(exc, "code", "error")
    if isinstance(exc, service.ShopError):
        message = str(exc)
    else:
        logger.exception("Unexpected tool execution error")
        message = "Не вдалося виконати операцію через внутрішню помилку."
    return ToolResult(
        status="error",
        content={"error": {"code": code, "message": message}},
        reason=f"сервіс відмовив: {message}",
    )


def _strict_json_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"повторний ключ {key!r}")
        result[key] = value
    return result


def _reject_json_constant(value: str):
    raise ValueError(f"недозволена JSON-константа {value}")


def specs() -> list[dict]:
    return [TOOL_SPECS[name] for name in [
        "list_orders",
        "get_order",
        "search_products",
        "get_product",
        "get_stock",
        "delivery_quote",
        "create_return",
    ]]


def call(name: str, raw_arguments: str, ctx: Context) -> ToolResult:
    if not isinstance(name, str) or name not in TOOL_SPECS:
        return ToolResult(
            status="rejected",
            content={"error": {"code": "unknown_tool", "message": f"інструмент {name!r} не доступний"}},
            reason=f"відхилено: інструмент {name!r} не описаний для цього помічника",
        )

    if not isinstance(raw_arguments, str):
        return ToolResult(
            status="rejected",
            content={"error": {"code": "invalid_arguments", "message": "аргументи мають бути JSON-рядком"}},
            reason="відхилено: аргументи мають бути JSON-рядком",
        )

    try:
        payload = json.loads(
            raw_arguments or "{}",
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        return ToolResult(
            status="rejected",
            content={"error": {"code": "invalid_json", "message": "аргументи не є валідним JSON"}},
            reason=f"відхилено: аргументи не є JSON ({exc})",
        )

    if not isinstance(payload, dict):
        return ToolResult(
            status="rejected",
            content={"error": {"code": "invalid_arguments", "message": "аргументи мають бути об'єктом JSON"}},
            reason="відхилено: аргументи мають бути об'єктом",
        )

    schema = TOOL_SPECS[name]["function"]["parameters"]
    try:
        validate(payload, schema)
    except ValidationError as exc:
        return ToolResult(
            status="rejected",
            content={"error": {"code": "invalid_schema", "message": exc.message}},
            reason=f"відхилено: аргументи не проходять схему ({exc.message})",
            arguments=payload,
        )

    if "customer_id" in payload:
        return ToolResult(
            status="rejected",
            content={"error": {"code": "customer_in_arguments", "message": "ідентифікатор клієнта не передається в інструмент"}},
            reason="відхилено: клієнт визначається з сеансу, а не з аргументів моделі",
            arguments=payload,
        )

    try:
        if name == "list_orders":
            result = service.list_orders(ctx.customer_id)
            return ToolResult(status="ok", content={"orders": result}, arguments={})

        if name == "get_order":
            order_id = payload["order_id"]
            order = service.get_order(order_id)
            if order["customer_id"] != ctx.customer_id:
                raise PermissionError(f"замовлення {order_id} не належить клієнтові {ctx.customer_id}")
            result = _sanitize_order(order)
            return ToolResult(status="ok", content=result, arguments={"order_id": order_id})

        if name == "search_products":
            args = {k: v for k, v in payload.items() if v is not None}
            result = service.search_products(**args)
            return ToolResult(status="ok", content={"items": result}, arguments=args)

        if name == "get_product":
            result = _sanitize_product(service.get_product(payload["sku"]))
            return ToolResult(status="ok", content=result, arguments={"sku": payload["sku"]})

        if name == "get_stock":
            result = service.get_stock(payload["sku"])
            return ToolResult(status="ok", content=result, arguments={"sku": payload["sku"]})

        if name == "delivery_quote":
            result = service.delivery_quote(payload["city"], payload["method"], payload["items"])
            return ToolResult(status="ok", content=result, arguments={
                "city": payload["city"],
                "method": payload["method"],
                "items": payload["items"],
            })

        if name == "create_return":
            order_id = payload["order_id"]
            order = service.get_order(order_id)
            if order["customer_id"] != ctx.customer_id:
                raise PermissionError(f"замовлення {order_id} не належить клієнтові {ctx.customer_id}")
            args = {"order_id": order_id, "sku": payload["sku"], "reason": payload["reason"], "quantity": payload.get("quantity", 1), "comment": payload.get("comment", "")}
            result = service.create_return(**args)
            public_result = {
                key: result[key]
                for key in ["return_id", "order_id", "sku", "quantity", "reason", "created_at", "status", "next_steps"]
                if key in result
            }
            return ToolResult(status="ok", content=public_result, arguments=args)

        return ToolResult(status="rejected", content={"error": {"code": "unsupported_tool", "message": "інструмент ще не реалізовано"}}, reason=f"відхилено: {name} не реалізовано")

    except PermissionError as exc:
        return ToolResult(
            status="rejected",
            content={"error": {"code": "forbidden", "message": str(exc)}},
            reason=f"відхилено: {exc}",
            arguments=payload,
        )
    except service.ShopError as exc:
        return _error_result(exc)
    except Exception as exc:  # pragma: no cover
        logger.exception("Unexpected tool execution error")
        return ToolResult(
            status="error",
            content={"error": {"code": "internal_error", "message": "Не вдалося виконати операцію через внутрішню помилку."}},
            reason="внутрішня помилка під час виконання інструмента",
            arguments=payload,
        )
