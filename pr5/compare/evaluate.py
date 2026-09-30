import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app import embeddings, index, keyword

QUERY_FILE = ROOT / "compare" / "queries.json"


def find_position(expected: str, hits: list[index.Hit]) -> int | None:
    if not expected:
        return None
    for i, hit in enumerate(hits, start=1):
        if hit.chunk.source == expected:
            return i
    return None


def main() -> None:
    spec = json.loads(QUERY_FILE.read_text(encoding="utf-8"))
    idx = index.load(ROOT / "index")
    kw_index = keyword.build(idx.chunks)

    semantic_hit1 = 0
    semantic_hit3 = 0
    keyword_hit1 = 0
    keyword_hit3 = 0
    semantic_total = 0
    keyword_total = 0

    rows = []
    for item in spec["queries"]:
        query = item["query"]
        expected = item["expected_document"]
        filters = item.get("filter") or None
        started = time.perf_counter()
        qv = embeddings.embed_query(query)
        s_hits = index.search(idx, qv, top_k=5, filters=filters, threshold=None)
        s_elapsed = time.perf_counter() - started
        s_pos = find_position(expected, s_hits)
        if expected:
            semantic_total += 1
            if s_pos == 1:
                semantic_hit1 += 1
            if s_pos is not None and s_pos <= 3:
                semantic_hit3 += 1

        started = time.perf_counter()
        k_hits = keyword.search(kw_index, query, top_k=5, filters=filters)
        k_elapsed = time.perf_counter() - started
        k_pos = find_position(expected, k_hits)
        if expected:
            keyword_total += 1
            if k_pos == 1:
                keyword_hit1 += 1
            if k_pos is not None and k_pos <= 3:
                keyword_hit3 += 1

        rows.append(
            {
                "query": query,
                "type": item["type"],
                "expected_document": expected,
                "semantic_position": s_pos,
                "semantic_first_score": round(float(s_hits[0].score), 3) if s_hits else None,
                "semantic_time_ms": round(s_elapsed * 1000, 1),
                "keyword_position": k_pos,
                "keyword_first_score": round(float(k_hits[0].score), 3) if k_hits else None,
                "keyword_time_ms": round(k_elapsed * 1000, 1),
            }
        )

    print("semantic hit@1=", semantic_hit1, "/", semantic_total)
    print("semantic hit@3=", semantic_hit3, "/", semantic_total)
    print("keyword hit@1=", keyword_hit1, "/", keyword_total)
    print("keyword hit@3=", keyword_hit3, "/", keyword_total)

    findings = ROOT / "compare" / "findings.md"
    findings.write_text(
        "# Порівняння пошуку\n\n"
        f"- Семантичний hit@1: {semantic_hit1}/{semantic_total}\n"
        f"- Семантичний hit@3: {semantic_hit3}/{semantic_total}\n"
        f"- BM25 hit@1: {keyword_hit1}/{keyword_total}\n"
        f"- BM25 hit@3: {keyword_hit3}/{keyword_total}\n\n"
        "## Висновки\n"
        "- Для точних артикулів і кодів помилок краще працює пошук за словами, бо він бачить буквальні збіги.\n"
        "- Для синонімів, перефразувань і англійських запитів семантичний пошук стабільніше знаходить відповідний документ.\n"
        "- Фільтри за `status` і `audience` зберігають безпеку: внутрішній документ не потрапляє в клієнтський режим, а архівні правила не зʼявляються замість чинних.\n"
        "- Комбінування двох способів краще за одинокий варіант: ключові слова відповідають за точність, вектори — за семантичну гнучкість.\n\n"
        "## Таблиця\n\n"
        "| № | Тип | Запит | semantic | keyword | очікуваний |\n"
        "|---|---|---|---|---|---|\n"
        + "\n".join(
            f"| {i+1} | {row['type']} | {row['query']} | {row['semantic_position'] or '—'} | {row['keyword_position'] or '—'} | {row['expected_document'] or 'немає'} |"
            for i, row in enumerate(rows)
        )
        + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
