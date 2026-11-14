"""Run the question set and persist tool traces for assignment review."""

import argparse
from collections import Counter, defaultdict
import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
QUESTIONS_PATH = Path(__file__).with_name("questions.json")
sys.path.insert(0, str(ROOT))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["baseline", "rounds2", "failure"], default="baseline")
    parser.add_argument("--ids", nargs="+", help="Run only these question IDs")
    parser.add_argument("--output", type=Path, help="Write results to this JSON path")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    if args.mode == "rounds2":
        os.environ["TOOL_MAX_ROUNDS"] = "2"
    if args.mode == "failure":
        os.environ["SHOP_FAILURE_RATE"] = "0.3"

    from app import assistant
    from shop import service

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    if args.ids:
        wanted = set(args.ids)
        questions = [question for question in questions if question["id"] in wanted]
        missing = wanted - {question["id"] for question in questions}
        if missing:
            parser.error(f"unknown question IDs: {', '.join(sorted(missing))}")

    output_path = args.output or ROOT / "eval" / "results" / f"{args.mode}.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    document = {
        "mode": args.mode,
        "tool_max_rounds": assistant.MAX_ROUNDS,
        "shop_failure_rate": os.getenv("SHOP_FAILURE_RATE", "0"),
        "requested_question_count": len(questions),
        "completed_question_count": 0,
        "complete": False,
        "results": [],
    }
    summary = defaultdict(lambda: {
        "questions": 0,
        "tool_selection_matches": 0,
        "assessable_tool_selections": 0,
        "tool_calls": 0,
        "rejected_calls": 0,
        "error_calls": 0,
        "stopped_by_limit": 0,
        "stopped_by_error": 0,
        "rounds_total": 0,
        "prompt_tokens_total": 0,
        "usage_questions": 0,
        "model_seconds_total": 0.0,
        "model_time_questions": 0,
    })

    for index, question in enumerate(questions, start=1):
        service.reset()
        answer = assistant.answer(question["question"], question["customer"])
        observed_tools = [call.name for call in answer.calls]
        selection_assessable = not (answer.stopped == "error" and answer.rounds == 0)
        selection_matches = (
            Counter(observed_tools) == Counter(question["expect_tools"])
            if selection_assessable else None
        )
        result = {
            "id": question["id"],
            "kind": question["kind"],
            "customer": question["customer"],
            "question": question["question"],
            "expected_tools": question["expect_tools"],
            "observed_tools": observed_tools,
            "tool_selection_matches": selection_matches,
            "manual_review": ["answer correctness", "hallucination", "data leakage"],
            "expectation": question["expect"],
            "answer": answer.text,
            "calls": [
                {
                    "round": call.round,
                    "name": call.name,
                    "arguments": call.arguments,
                    "status": call.status,
                    "reason": call.reason,
                    "result": call.result,
                    "elapsed": call.elapsed,
                }
                for call in answer.calls
            ],
            "rounds": answer.rounds,
            "stopped": answer.stopped,
            "model": answer.model,
            "elapsed": answer.elapsed,
            "usage": answer.usage,
        }
        document["results"].append(result)
        group = summary[question["kind"]]
        group["questions"] += 1
        if selection_assessable:
            group["assessable_tool_selections"] += 1
            group["tool_selection_matches"] += int(selection_matches)
        group["tool_calls"] += len(answer.calls)
        group["rejected_calls"] += sum(call.status == "rejected" for call in answer.calls)
        group["error_calls"] += sum(call.status == "error" for call in answer.calls)
        group["stopped_by_limit"] += int(answer.stopped == "limit")
        group["stopped_by_error"] += int(answer.stopped == "error")
        group["rounds_total"] += answer.rounds
        usage = answer.usage or {}
        if usage.get("total_tokens", 0):
            group["prompt_tokens_total"] += usage.get("prompt_tokens", 0)
            group["usage_questions"] += 1
        model_seconds = (answer.elapsed or {}).get("model", 0.0)
        if model_seconds:
            group["model_seconds_total"] += model_seconds
            group["model_time_questions"] += 1
        group["average_rounds"] = round(group["rounds_total"] / group["questions"], 2)
        group["average_prompt_tokens"] = (
            round(group["prompt_tokens_total"] / group["usage_questions"], 2)
            if group["usage_questions"] else None
        )
        group["average_model_seconds"] = (
            round(group["model_seconds_total"] / group["model_time_questions"], 2)
            if group["model_time_questions"] else None
        )
        document["completed_question_count"] = len(document["results"])
        document["complete"] = document["completed_question_count"] == document["requested_question_count"]
        document["summary"] = dict(summary)
        output_path.write_text(json.dumps(document, ensure_ascii=False, indent=2), encoding="utf-8")
        names = ",".join(f"{call.name}:{call.status}" for call in answer.calls) or "none"
        total_tokens = (answer.usage or {}).get("total_tokens", 0)
        print(f"[{index}/{len(questions)}] {question['id']} stopped={answer.stopped} rounds={answer.rounds} calls={names} tokens={total_tokens}")

    print(f"saved: {output_path.relative_to(ROOT) if output_path.is_relative_to(ROOT) else output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
