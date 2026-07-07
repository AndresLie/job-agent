from __future__ import annotations

import json
import math
from pathlib import Path

from .embeddings import Embedder
from .vector_store import JsonVectorStore


def load_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_retrieval(
    queries_path: Path,
    store: JsonVectorStore,
    embedder: Embedder,
    k: int = 5,
) -> dict:
    queries = load_jsonl(queries_path)
    rows = []
    for query in queries:
        hits = store.query(query["query"], embedder, top_k=k)
        ids = [hit["source_path"] for hit in hits]
        relevant = set(query["relevant_sources"])
        rows.append(score_query(ids, relevant, k))
    if not rows:
        return {"queries": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    return {
        "queries": len(rows),
        "recall_at_k": round(sum(row["recall"] for row in rows) / len(rows), 3),
        "mrr": round(sum(row["mrr"] for row in rows) / len(rows), 3),
        "ndcg_at_k": round(sum(row["ndcg"] for row in rows) / len(rows), 3),
        "per_query": rows,
    }


def score_query(ids: list[str], relevant: set[str], k: int) -> dict:
    hits = [1 if doc_id in relevant else 0 for doc_id in ids[:k]]
    recall = 1.0 if any(hits) else 0.0
    mrr = 0.0
    for index, hit in enumerate(hits, start=1):
        if hit:
            mrr = 1.0 / index
            break
    dcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(hits))
    ideal_hits = [1] * min(len(relevant), k)
    idcg = sum(hit / math.log2(index + 2) for index, hit in enumerate(ideal_hits)) or 1.0
    return {"ids": ids, "recall": recall, "mrr": mrr, "ndcg": dcg / idcg}
