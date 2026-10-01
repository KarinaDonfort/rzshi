"""Run the knowledge-base evaluation in baseline and one-change passes."""

import argparse
import json
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env", override=True)

from app import index, keyword, llm, rag, retrieval  # noqa: E402

QUESTIONS_PATH = ROOT / "eval" / "questions.json"
DEFAULT_OUTPUT = ROOT / "eval" / "results.json"
PASSES = (("baseline", 4), ("context_5", 5))


def save_report(path: Path, report: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(records: list[dict]) -> dict:
    completed = [record for record in records if "answer" in record]
    answerable = [record for record in completed if record["expected_answer"]]
    unanswerable = [record for record in completed if not record["expected_answer"]]
    expected_doc_records = [record for record in completed if record["expected_documents"]]
    expected_section_records = [record for record in completed if record.get("expected_sections")]
    token_records = [record.get("usage") or {} for record in completed]
    return {
        "completed": len(completed),
        "failed": len(records) - len(completed),
        "expected_document_in_context": {
            "hits": sum(record["expected_document_in_context"] for record in expected_doc_records),
            "total": len(expected_doc_records),
        },
        "expected_sections_in_context": {
            "hits": sum(not record.get("missing_expected_sections") for record in expected_section_records),
            "total": len(expected_section_records),
        },
        "found_flag_agreement": {
            "matches": sum(record["found"] == record["expected_answer"] for record in completed),
            "total": len(completed),
        },
        "false_answers": sum(record["found"] for record in unanswerable),
        "false_refusals": sum(not record["found"] for record in answerable),
        "policy_violations": sum(bool(record["policy_violations"]) for record in completed),
        "prompt_tokens": sum(item.get("prompt_tokens") or 0 for item in token_records),
        "completion_tokens": sum(item.get("completion_tokens") or 0 for item in token_records),
        "total_tokens": sum(item.get("total_tokens") or 0 for item in token_records),
        "factual_review_required": len(completed),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Check retrieval and contexts without calling the LLM.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true", help="Resume successful records from an existing results file.")
    args = parser.parse_args()

    payload = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    questions = payload["питання"]
    if not 12 <= len(questions) <= 15:
        raise SystemExit(f"Expected 12-15 questions, got {len(questions)}.")

    search_index = index.load()
    keyword_index = keyword.build(search_index.chunks)
    if args.resume and args.output.exists():
        report = json.loads(args.output.read_text(encoding="utf-8"))
        if report.get("model") != llm.MODEL or report.get("question_count") != len(questions):
            raise SystemExit("Cannot resume: model or question count differs from the saved run.")
    else:
        report = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "model": llm.MODEL,
            "question_count": len(questions),
            "controlled_change": "RAG_CONTEXT_CHUNKS: 4 -> 5",
            "runs": [],
        }
    stop_after_quota = False

    for pass_name, context_chunks in PASSES:
        retrieval.CONTEXT_CHUNKS = context_chunks
        run = next((item for item in report["runs"] if item.get("name") == pass_name), None)
        if run is None:
            run = {"name": pass_name, "context_chunks": context_chunks, "records": []}
            report["runs"].append(run)
        current_items = {number: item for number, item in enumerate(questions, start=1)}
        valid_records = []
        for record in run.get("records", []):
            number = record.get("number")
            item = current_items.get(number)
            if not item or record.get("question") != item["питання"] or "answer" not in record:
                continue
            record["expected_documents"] = item.get("очікувані_документи", [])
            record["expected_facts"] = item.get("очікувана_відповідь")
            record["expected_answer"] = bool(item.get("очікується_відповідь"))
            record["expected_sections"] = item.get("очікувані_розділи", {})
            record["expected_document_positions"] = {
                document: [position for position, name in enumerate(record.get("context_sources", []), start=1) if name == document]
                for document in record["expected_documents"]
            }
            record["missing_expected_documents"] = [
                document
                for document, positions in record["expected_document_positions"].items()
                if not positions
            ]
            record["expected_document_in_context"] = bool(record["expected_documents"]) and not record["missing_expected_documents"]
            headings = re.findall(r"(?m)^\[(\d+)\].*?\| Розділ: (.*?) \| Оновлено:", record.get("context", ""))
            record["context_entries"] = [
                {
                    "ref": int(ref),
                    "document": record.get("context_sources", [])[int(ref) - 1],
                    "heading": heading,
                }
                for ref, heading in headings
                if int(ref) <= len(record.get("context_sources", []))
            ]
            record["missing_expected_sections"] = [
                {"document": document, "heading": heading}
                for document, expected_headings in record["expected_sections"].items()
                for heading in expected_headings
                if not any(
                    entry["document"] == document and entry["heading"] == heading
                    for entry in record["context_entries"]
                )
            ]
            valid_records.append(record)
        run["records"] = valid_records
        for number, item in enumerate(questions, start=1):
            question = item["питання"]
            filters = item.get("filters") or {}
            expected_documents = item.get("очікувані_документи", [])
            expected_sections = item.get("очікувані_розділи", {})
            expected_answer = bool(item.get("очікується_відповідь"))

            started = time.perf_counter()
            hits = retrieval.retrieve(question, search_index, keyword_index, filters=filters or None)
            context, context_sources = retrieval.build_context(hits)
            local_retrieval_seconds = round(time.perf_counter() - started, 3)
            source_names = [source.chunk.source for source in context_sources]
            expected_positions = {
                document: [position for position, name in enumerate(source_names, start=1) if name == document]
                for document in expected_documents
            }
            missing_expected_documents = [
                document for document, positions in expected_positions.items() if not positions
            ]
            expected_in_context = bool(expected_documents) and not missing_expected_documents
            context_entries = [
                {
                    "ref": source.ref,
                    "document": source.chunk.source,
                    "heading": str((source.chunk.metadata or {}).get("heading") or "розділ"),
                }
                for source in context_sources
            ]
            missing_expected_sections = [
                {"document": document, "heading": heading}
                for document, expected_headings in expected_sections.items()
                for heading in expected_headings
                if not any(
                    entry["document"] == document and entry["heading"] == heading
                    for entry in context_entries
                )
            ]
            violations = [
                source.chunk.source
                for source in context_sources
                if (source.chunk.metadata or {}).get("audience") != "клієнти"
                or (source.chunk.metadata or {}).get("status") != "чинний"
            ]

            record = {
                "number": number,
                "kind": item.get("вид"),
                "question": question,
                "filters": filters,
                "expected_documents": expected_documents,
                "expected_facts": item.get("очікувана_відповідь"),
                "expected_answer": expected_answer,
                "expected_sections": expected_sections,
                "context": context,
                "context_sources": source_names,
                "context_entries": context_entries,
                "expected_document_positions": expected_positions,
                "missing_expected_documents": missing_expected_documents,
                "expected_document_in_context": expected_in_context,
                "missing_expected_sections": missing_expected_sections,
                "policy_violations": violations,
                "retrieval_seconds": local_retrieval_seconds,
            }

            previous = next((saved for saved in run["records"] if saved.get("number") == number), None)
            if (
                args.resume
                and previous
                and previous.get("question") == question
                and previous.get("filters", {}) == filters
                and previous.get("context") == context
                and previous.get("model") == llm.MODEL
                and "answer" in previous
            ):
                for key in ("answer", "found", "sources", "retrieved", "model", "elapsed", "usage", "factual_review"):
                    if key in previous:
                        record[key] = previous[key]
                run["records"] = [saved for saved in run["records"] if saved.get("number") != number]
                run["records"].append(record)
                print(f"[{pass_name} {number}/{len(questions)}] context unchanged; reusing answer", flush=True)
                continue
            run["records"] = [saved for saved in run["records"] if saved.get("number") != number]

            if args.dry_run:
                run["records"].append(record)
                print(f"[{pass_name} {number}/{len(questions)}] context docs={source_names}")
                continue

            try:
                result = rag.answer(question, search_index, keyword_index, filters=filters or None)
                record.update(
                    {
                        "answer": result.text,
                        "found": result.found,
                        "sources": [
                            {"ref": source.ref, "document": source.chunk.source, "score": source.score}
                            for source in result.sources
                        ],
                        "retrieved": [
                            {"document": hit.chunk.source, "text": hit.chunk.text, "score": hit.score}
                            for hit in result.retrieved
                        ],
                        "model": result.model,
                        "elapsed": result.elapsed,
                        "usage": result.usage,
                        "factual_review": "pending",
                    }
                )
            except Exception as exc:  # Preserve failures and stop if the provider quota is exhausted.
                message = str(exc)
                record["error"] = f"{type(exc).__name__}: {message}"
                if re.search(r"429|quota|RESOURCE_EXHAUSTED|daily limit", message, re.IGNORECASE):
                    record["failure_stage"] = "generation_quota"
                    stop_after_quota = True
            run["records"].append(record)
            run["summary"] = summarize(run["records"])
            save_report(args.output, report)
            status = "ERROR" if "error" in record else ("OK" if record["found"] else "NO ANSWER")
            print(f"[{pass_name} {number}/{len(questions)}] {status}; model={llm.MODEL}", flush=True)
            if stop_after_quota:
                break
        run["summary"] = summarize(run["records"])
        if stop_after_quota:
            break

    if not args.dry_run:
        save_report(args.output, report)
        print(f"Results saved to {args.output}")
    for run in report["runs"]:
        print(f"{run['name']}: {json.dumps(run.get('summary', {}), ensure_ascii=False)}")
    return 2 if stop_after_quota else 0


if __name__ == "__main__":
    raise SystemExit(main())