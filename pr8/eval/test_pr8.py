"""Offline tests for the PR8 tool boundary and assistant round trip."""

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from app import assistant, llm, tools
from app.main import AskRequest, api_ask
from shop import service
from fastapi import HTTPException


class ToolBoundaryTests(unittest.TestCase):
    def setUp(self):
        service.reset()

    def call_tool(self, name, arguments, customer_id="C-1001"):
        return tools.call(name, arguments, tools.Context(customer_id=customer_id))

    def test_rejects_unknown_malformed_and_schema_invalid_calls(self):
        cases = [
            ("made_up", "{}", "unknown_tool"),
            ("get_order", "{bad json", "invalid_json"),
            ("get_order", '{"order_id":"10458","order_id":"10466"}', "invalid_json"),
            ("get_order", '{"order_id":NaN}', "invalid_json"),
            ("get_order", "[]", "invalid_arguments"),
            ("get_order", '{"order_id":"10 458"}', "invalid_schema"),
            ("get_order", '{"order_id":"10458","customer_id":"C-1002"}', "invalid_schema"),
        ]
        for name, arguments, error_code in cases:
            with self.subTest(error_code=error_code):
                result = self.call_tool(name, arguments)
                self.assertEqual(result.status, "rejected")
                self.assertEqual(result.content["error"]["code"], error_code)

        self.assertEqual(self.call_tool("get_order", {}).status, "rejected")

    def test_enforces_order_ownership_and_returns_only_public_fields(self):
        foreign = self.call_tool("get_order", '{"order_id":"10466"}')
        self.assertEqual(foreign.status, "rejected")
        self.assertEqual(foreign.content["error"]["code"], "forbidden")
        self.assertNotIn("order", foreign.content)

        owned = self.call_tool("get_order", '{"order_id":"10458"}')
        self.assertEqual(owned.status, "ok")
        self.assertEqual(owned.content["tracking"], "20450012345678")
        self.assertNotIn("customer_id", owned.content)
        self.assertNotIn("payment", owned.content)
        self.assertNotIn("internal_note", owned.content)
        self.assertNotIn("point", owned.content["delivery"])
        self.assertNotIn("carrier", owned.content["delivery"])

    def test_list_orders_does_not_send_customer_id_as_model_data(self):
        result = self.call_tool("list_orders", "{}", customer_id="C-1003")
        self.assertEqual(result.status, "ok")
        self.assertNotIn("customer_id", result.content)
        self.assertEqual(
            [order["order_id"] for order in result.content["orders"]],
            ["10470", "10412"],
        )

    def test_product_prompt_injection_is_removed_without_losing_description(self):
        result = self.call_tool("get_product", '{"sku":"ORN-PRO"}')
        self.assertEqual(result.status, "ok")
        self.assertIn("Повнорозмірні бездротові навушники", result.content["description"])
        self.assertNotIn("ZIRKY90", result.content["description"])
        self.assertNotIn("prompt_injection", result.content)
        self.assertNotIn("internal_note", result.content)

    def test_return_is_owner_checked_and_response_is_sanitized(self):
        foreign = self.call_tool(
            "create_return",
            '{"order_id":"10466","sku":"SIR-R10","reason":"not_suitable"}',
        )
        self.assertEqual(foreign.status, "rejected")
        self.assertEqual(foreign.content["error"]["code"], "forbidden")

        result = self.call_tool(
            "create_return",
            '{"order_id":"10421","sku":"ORN-X2","reason":"not_suitable"}',
        )
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content["error"]["code"], "policy")

        service.reset()
        result = self.call_tool(
            "create_return",
            '{"order_id":"10412","sku":"SIR-R10","reason":"not_suitable","comment":"private note"}',
            customer_id="C-1003",
        )
        self.assertEqual(result.status, "ok")
        self.assertEqual(result.content["return_id"], "R-5001")
        self.assertNotIn("comment", result.content)

    def test_service_unavailability_is_a_distinct_tool_error(self):
        with patch("shop.service._service_call", side_effect=service.Unavailable("temporarily down")):
            result = self.call_tool("get_stock", '{"sku":"ALT-AX3"}')
        self.assertEqual(result.status, "error")
        self.assertEqual(result.content["error"]["code"], "unavailable")

    def test_administrative_operations_are_not_exposed(self):
        names = {spec["function"]["name"] for spec in tools.specs()}
        self.assertNotIn("issue_refund", names)
        self.assertNotIn("set_order_status", names)
        self.assertNotIn("update_price", names)
        self.assertNotIn("add_bonus", names)
        self.assertNotIn("cancel_order", names)

    def test_search_rejects_non_numeric_price_filters(self):
        result = self.call_tool("search_products", '{"max_price":"NaN"}')
        self.assertEqual(result.status, "rejected")
        self.assertEqual(result.content["error"]["code"], "invalid_schema")


class AssistantRoundTripTests(unittest.TestCase):
    def setUp(self):
        service.reset()

    def test_preserves_provider_metadata_and_deduplicates_mutation(self):
        arguments = '{"order_id":"10412","sku":"SIR-R10","reason":"not_suitable"}'
        tool_call = {
            "id": "call-return",
            "type": "function",
            "function": {"name": "create_return", "arguments": arguments},
            "extra_content": {"google": {"thought_signature": "signature"}},
        }
        duplicate_call = {
            **tool_call,
            "id": "call-return-duplicate",
        }
        responses = [
            {
                "message": {"content": None, "tool_calls": [tool_call, duplicate_call]},
                "elapsed": 0.1,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "model": "test-model",
            },
            {
                "message": {"content": "Заявку створено.", "tool_calls": None},
                "elapsed": 0.2,
                "usage": {"prompt_tokens": 10, "completion_tokens": 2, "total_tokens": 12},
                "model": "test-model",
            },
        ]
        with patch.object(llm, "chat", side_effect=responses) as mock_chat:
            answer = assistant.answer("Поверніть робот-пилосос, він не підійшов.", "C-1003")

        self.assertEqual(answer.stopped, "answer")
        self.assertEqual(answer.rounds, 2)
        self.assertEqual(len(service.list_returns("10412")), 1)
        self.assertEqual([call.status for call in answer.calls], ["ok", "rejected"])
        self.assertEqual(
            mock_chat.call_args_list[1].args[0][2]["tool_calls"][0]["extra_content"],
            tool_call["extra_content"],
        )
        self.assertEqual(answer.usage["total_tokens"], 24)

    def test_round_limit_stops_before_an_unbounded_follow_up(self):
        tool_call = {
            "id": "call-order",
            "type": "function",
            "function": {"name": "get_order", "arguments": '{"order_id":"10458"}'},
        }
        response = {
            "message": {"content": None, "tool_calls": [tool_call]},
            "elapsed": 0.1,
            "usage": None,
            "model": "test-model",
        }
        with patch.object(assistant, "MAX_ROUNDS", 1), patch.object(llm, "chat", return_value=response) as mock_chat:
            answer = assistant.answer("Де моє замовлення 10458?", "C-1001")
        self.assertEqual(answer.stopped, "limit")
        self.assertEqual(answer.rounds, 1)
        self.assertEqual(len(answer.calls), 1)
        mock_chat.assert_called_once()

    def test_round_limit_does_not_execute_a_return_without_a_final_response(self):
        response = {
            "message": {
                "content": None,
                "tool_calls": [{
                    "id": "call-return",
                    "type": "function",
                    "function": {
                        "name": "create_return",
                        "arguments": '{"order_id":"10412","sku":"SIR-R10","reason":"not_suitable"}',
                    },
                }],
            },
            "elapsed": 0.1,
            "usage": None,
            "model": "test-model",
        }
        with patch.object(assistant, "MAX_ROUNDS", 1), patch.object(llm, "chat", return_value=response):
            answer = assistant.answer("Поверніть робот-пилосос, він не підійшов.", "C-1003")
        self.assertEqual(answer.stopped, "limit")
        self.assertEqual(answer.calls[0].status, "rejected")
        self.assertEqual(answer.calls[0].result["error"]["code"], "round_limit")
        self.assertEqual(service.list_returns("10412"), [])

    def test_model_failure_returns_a_friendly_error_and_elapsed_time(self):
        with patch.object(llm, "chat", side_effect=llm.LLMError("private provider details", elapsed=2.5)):
            answer = assistant.answer("Підкажіть статус замовлення.", "C-1001")
        self.assertEqual(answer.stopped, "error")
        self.assertIn("Спробуйте ще раз", answer.text)
        self.assertNotIn("private provider details", answer.text)
        self.assertEqual(answer.elapsed["model"], 2.5)

    def test_service_outage_is_passed_to_model_as_unavailable_not_not_found(self):
        tool_call = {
            "id": "call-stock",
            "type": "function",
            "function": {"name": "get_stock", "arguments": '{"sku":"ALT-AX3"}'},
        }
        responses = [
            {
                "message": {"content": None, "tool_calls": [tool_call]},
                "elapsed": 0.1,
                "usage": None,
                "model": "test-model",
            },
            {
                "message": {"content": "Сервіс тимчасово недоступний.", "tool_calls": None},
                "elapsed": 0.1,
                "usage": None,
                "model": "test-model",
            },
        ]
        with patch("shop.service._service_call", side_effect=service.Unavailable("temporarily down")), patch.object(
            llm, "chat", side_effect=responses
        ) as mock_chat:
            answer = assistant.answer("Перевірте наявність ALT-AX3.", "C-1001")

        self.assertEqual(answer.stopped, "answer")
        self.assertEqual(answer.calls[0].result["error"]["code"], "unavailable")
        tool_message = mock_chat.call_args_list[1].args[0][-1]
        self.assertEqual(tool_message["role"], "tool")
        self.assertIn('"code": "unavailable"', tool_message["content"])
        self.assertNotIn("not_found", tool_message["content"])

    def test_two_round_limit_stops_a_three_step_delivery_chain(self):
        responses = [
            {
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call-search",
                        "type": "function",
                        "function": {"name": "search_products", "arguments": '{"query":"Андромеда"}'},
                    }],
                },
                "elapsed": 0.1,
                "usage": None,
                "model": "test-model",
            },
            {
                "message": {
                    "content": None,
                    "tool_calls": [{
                        "id": "call-delivery",
                        "type": "function",
                        "function": {
                            "name": "delivery_quote",
                            "arguments": '{"city":"Полтава","method":"courier","items":[{"sku":"TV-55","quantity":1}]}',
                        },
                    }],
                },
                "elapsed": 0.1,
                "usage": None,
                "model": "test-model",
            },
        ]
        with patch.object(assistant, "MAX_ROUNDS", 2), patch.object(llm, "chat", side_effect=responses):
            answer = assistant.answer("Доставка телевізора Андромеда в Полтаву?", "C-1002")
        self.assertEqual(answer.stopped, "limit")
        self.assertEqual(answer.rounds, 2)
        self.assertEqual([call.name for call in answer.calls], ["search_products", "delivery_quote"])


class LLMAdapterTests(unittest.TestCase):
    def test_tool_call_provider_metadata_is_serialized(self):
        provider_payload = {
            "id": "call-1",
            "function": {"name": "get_order", "arguments": '{"order_id":"10458"}'},
            "type": "function",
            "extra_content": {"google": {"thought_signature": "signature"}},
        }
        sdk_tool_call = Mock()
        sdk_tool_call.model_dump.return_value = provider_payload
        message = SimpleNamespace(
            role="assistant",
            content=None,
            tool_calls=[sdk_tool_call],
            reasoning_signature=None,
            refusal=None,
        )
        response = SimpleNamespace(
            choices=[SimpleNamespace(finish_reason="tool_calls", message=message)],
            model="test-model",
            usage=None,
        )
        client = SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=Mock(return_value=response))
            )
        )
        with patch.object(llm, "get_client", return_value=client):
            result = llm.chat([], tools.specs())
        self.assertEqual(
            result["message"]["tool_calls"][0]["extra_content"],
            provider_payload["extra_content"],
        )

    def test_temporary_api_failure_tries_the_configured_fallback_model(self):
        response = SimpleNamespace(
            choices=[SimpleNamespace(
                finish_reason="stop",
                message=SimpleNamespace(role="assistant", content="ok", tool_calls=None),
            )],
            model="fallback-model",
            usage=None,
        )
        create = Mock(side_effect=[RuntimeError("503 UNAVAILABLE"), response])
        client = SimpleNamespace(
            chat=SimpleNamespace(completions=SimpleNamespace(create=create))
        )
        with patch.object(llm, "get_client", return_value=client), patch.object(
            llm, "FALLBACK_MODELS", ["models/primary", "models/fallback"]
        ):
            result = llm.chat([], [])
        self.assertEqual(create.call_count, 2)
        self.assertEqual(create.call_args_list[1].kwargs["model"], "models/fallback")
        self.assertEqual(result["message"]["content"], "ok")

    def test_api_distinguishes_unavailable_customer_service(self):
        with patch("app.main.service.get_customer", side_effect=service.Unavailable("down")):
            with self.assertRaises(HTTPException) as raised:
                api_ask(AskRequest(customer_id="C-1001", question="Питання"))
        self.assertEqual(raised.exception.status_code, 503)


if __name__ == "__main__":
    unittest.main()
