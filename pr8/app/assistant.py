"""Цикл помічника: від питання клієнта до відповіді, спертої на дані
магазину.
"""

import json
import logging
import os
import time
from dataclasses import dataclass, field

from dotenv import load_dotenv

from app import llm, tools

load_dotenv()

MAX_ROUNDS = int(os.getenv("TOOL_MAX_ROUNDS", "3"))
logger = logging.getLogger(__name__)


@dataclass
class ToolTrace:
    round: int
    name: str
    arguments: str
    status: str
    reason: str | None = None
    result: dict | list | str | None = None
    elapsed: float | None = None


@dataclass
class Answer:
    text: str
    calls: list[ToolTrace] = field(default_factory=list)
    rounds: int = 0
    stopped: str = "answer"
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def _usage_total(current: dict | None, extra: dict | None) -> dict:
    if current is None:
        current = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if extra is None:
        extra = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return {
        "prompt_tokens": (current.get("prompt_tokens", 0) + extra.get("prompt_tokens", 0)),
        "completion_tokens": (current.get("completion_tokens", 0) + extra.get("completion_tokens", 0)),
        "total_tokens": (current.get("total_tokens", 0) + extra.get("total_tokens", 0)),
    }


def _usage_from_response(response: dict) -> dict:
    usage = response.get("usage")
    if usage is None:
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    if hasattr(usage, "model_dump"):
        return usage.model_dump()
    if isinstance(usage, dict):
        return usage
    try:
        return dict(usage)
    except (TypeError, ValueError):
        logger.warning("Could not parse token usage data from the model response")
        return {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}


def answer(question: str, customer_id: str) -> Answer:
    if not question or not str(question).strip():
        return Answer(
            text="Порожнє питання. Сформулюйте запит про замовлення, товар, доставку або повернення.",
            stopped="empty",
        )

    if not customer_id or not str(customer_id).strip():
        return Answer(
            text="Клієнт не вибраний. Оновіть сторінку і виберіть клієнта.",
            stopped="error",
        )

    messages = llm.build_messages(question)
    tool_specs = tools.specs()
    total_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    total_model_time = 0.0
    total_tool_time = 0.0
    calls: list[ToolTrace] = []
    completed_mutations: set[str] = set()

    for round_index in range(1, MAX_ROUNDS + 1):
        try:
            response = llm.chat(messages, tool_specs, tool_choice="auto")
        except llm.LLMError as exc:
            total_model_time += exc.elapsed
            return Answer(
                text="Не вдалося отримати відповідь від мовної моделі. Спробуйте ще раз пізніше.",
                calls=calls,
                rounds=round_index - 1,
                stopped="error",
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )

        total_model_time += float(response.get("elapsed", 0.0) or 0.0)
        total_usage = _usage_total(total_usage, _usage_from_response(response))

        message = response.get("message")
        if not isinstance(message, dict):
            return Answer(
                text="Мовна модель повернула відповідь у неочікуваному форматі. Спробуйте ще раз.",
                calls=calls,
                rounds=round_index,
                stopped="error",
                model=response.get("model"),
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )
        tool_calls = message.get("tool_calls") or []
        content = message.get("content")

        if not isinstance(tool_calls, list):
            return Answer(
                text="Мовна модель повернула список інструментів у неочікуваному форматі. Спробуйте ще раз.",
                calls=calls,
                rounds=round_index,
                stopped="error",
                model=response.get("model"),
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )

        if not tool_calls:
            if content:
                return Answer(
                    text=str(content),
                    calls=calls,
                    rounds=round_index,
                    stopped="answer",
                    model=response.get("model"),
                    elapsed={"model": total_model_time, "tools": total_tool_time},
                    usage=total_usage,
                )
            return Answer(
                text="Не можу відповісти на це питання без додаткових даних. Будь ласка, уточніть запит.",
                calls=calls,
                rounds=round_index,
                stopped="answer",
                model=response.get("model"),
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )

        if any(
            not isinstance(call, dict)
            or not isinstance(call.get("id"), str)
            or not call["id"]
            or not isinstance(call.get("function"), dict)
            for call in tool_calls
        ):
            return Answer(
                text="Мовна модель повернула неповний виклик інструмента. Спробуйте ще раз.",
                calls=calls,
                rounds=round_index,
                stopped="error",
                model=response.get("model"),
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )

        messages.append({
            "role": "assistant",
            "content": message.get("content"),
            "tool_calls": message.get("tool_calls"),
        })

        for call in tool_calls:
            fn = (call.get("function") or {})
            fn_name = fn.get("name")
            raw_arguments = fn.get("arguments") or "{}"
            start = time.perf_counter()
            mutation_key = None
            if fn_name == "create_return":
                try:
                    parsed_arguments = json.loads(raw_arguments)
                    if isinstance(parsed_arguments, dict):
                        mutation_key = json.dumps(parsed_arguments, sort_keys=True, ensure_ascii=False)
                except (TypeError, json.JSONDecodeError):
                    mutation_key = None
            if round_index == MAX_ROUNDS and fn_name == "create_return":
                tool_result = tools.ToolResult(
                    status="rejected",
                    content={"error": {
                        "code": "round_limit",
                        "message": "Заявку не створено: ліміт звертань вичерпано до завершення відповіді.",
                    }},
                    reason="відхилено зміну, запропоновану на останньому звертанні до моделі",
                )
            elif mutation_key and mutation_key in completed_mutations:
                tool_result = tools.ToolResult(
                    status="rejected",
                    content={"error": {
                        "code": "duplicate_action",
                        "message": "Таку саму заявку вже створено під час цього запиту.",
                    }},
                    reason="відхилено повторну ідентичну заявку в межах одного запиту",
                )
            else:
                tool_result = tools.call(fn_name, raw_arguments, tools.Context(customer_id=customer_id))
                if mutation_key and tool_result.status == "ok":
                    completed_mutations.add(mutation_key)
            elapsed = time.perf_counter() - start
            total_tool_time += elapsed
            calls.append(
                ToolTrace(
                    round=round_index,
                    name=fn_name or "unknown",
                    arguments=raw_arguments,
                    status=tool_result.status,
                    reason=tool_result.reason,
                    result=tool_result.content,
                    elapsed=elapsed,
                )
            )

            payload = tool_result.content
            if payload is None:
                payload = {"error": {"code": "no_result", "message": tool_result.reason or "інструмент не повернув результат"}}
            messages.append({
                "role": "tool",
                "tool_call_id": call.get("id"),
                "content": json.dumps(payload),
            })

        if round_index >= MAX_ROUNDS:
            return Answer(
                text="Ліміт звертань до моделі вичерпано. Я не можу дати надійну відповідь без додаткових даних або уточнення запиту.",
                calls=calls,
                rounds=round_index,
                stopped="limit",
                model=response.get("model"),
                elapsed={"model": total_model_time, "tools": total_tool_time},
                usage=total_usage,
            )

    return Answer(
        text="Ліміт звертань до моделі вичерпано, і відповідь не сформована.",
        calls=calls,
        rounds=MAX_ROUNDS,
        stopped="limit",
        elapsed={"model": total_model_time, "tools": total_tool_time},
        usage=total_usage,
    )
