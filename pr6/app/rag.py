"""Конвеєр відповіді за базою знань."""

import time
from dataclasses import dataclass, field

from . import llm
from .index import Hit, SearchIndex
from .keyword import KeywordIndex
from .retrieval import Source, build_context, retrieve


@dataclass
class Answer:
    text: str
    found: bool
    sources: list[Source] = field(default_factory=list)
    retrieved: list[Hit] = field(default_factory=list)
    model: str | None = None
    elapsed: dict = field(default_factory=dict)
    usage: dict | None = None


def answer(
    question: str,
    index: SearchIndex,
    keyword_index: KeywordIndex | None,
    filters: dict | None = None,
) -> Answer:
    started = time.perf_counter()
    question = (question or "").strip()
    if not question:
        raise ValueError("Порожнє питання.")

    retrieval_started = time.perf_counter()
    retrieval_hits = retrieve(question, index, keyword_index, filters=filters)
    retrieval_elapsed = time.perf_counter() - retrieval_started

    if not retrieval_hits:
        return Answer(
            text="У базі знань немає підтвердженої інформації для цього питання.",
            found=False,
            sources=[],
            retrieved=[],
            elapsed={"retrieval": round(retrieval_elapsed, 3), "generation": 0.0},
            usage=None,
        )

    context_text, context_sources = build_context(retrieval_hits)
    generation_started = time.perf_counter()
    llm_data = llm.ask(question, context_text)
    generation_elapsed = time.perf_counter() - generation_started

    valid_refs = {source.ref for source in context_sources}
    model_sources = list(llm_data.get("sources", []))
    clean_sources = [source for source in context_sources if source.ref in model_sources]

    if llm_data.get("found") is True and not model_sources:
        llm_data["found"] = False
        llm_data["answer"] = "У базі знань немає підтвердження для цього питання."
        llm_data["sources"] = []

    invalid = [ref for ref in model_sources if ref not in valid_refs]
    if invalid:
        llm_data["found"] = False
        llm_data["answer"] = "У базі знань немає підтвердження для цього питання."
        llm_data["sources"] = []
        clean_sources = []

    found = bool(llm_data.get("found")) and bool(clean_sources)
    final_text = llm_data.get("answer") if found else "У базі знань немає підтвердження для цього питання."

    return Answer(
        text=final_text,
        found=found,
        sources=clean_sources,
        retrieved=retrieval_hits[: len(context_sources)],
        model=llm_data.get("model"),
        elapsed={"retrieval": round(retrieval_elapsed, 3), "generation": round(generation_elapsed, 3)},
        usage=llm_data.get("usage"),
    )
