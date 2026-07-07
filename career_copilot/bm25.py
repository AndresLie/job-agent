from __future__ import annotations

import math
import re
from collections import Counter


TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    return TOKEN_RE.findall(text.casefold())


def bm25_search(
    query: str,
    docs: list[dict],
    k: int = 8,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[dict]:
    if not docs:
        return []
    query_terms = tokenize(query)
    doc_terms = [tokenize(doc["text"]) for doc in docs]
    doc_lens = [len(terms) for terms in doc_terms]
    avgdl = sum(doc_lens) / len(doc_lens) if doc_lens else 1.0
    df: Counter[str] = Counter()
    for terms in doc_terms:
        for term in set(terms):
            df[term] += 1

    scored: list[tuple[float, int, str]] = []
    for index, (doc, terms, doc_len) in enumerate(zip(docs, doc_terms, doc_lens)):
        tf = Counter(terms)
        score = 0.0
        for term in query_terms:
            freq = tf.get(term, 0)
            if freq == 0:
                continue
            idf = math.log((len(docs) - df[term] + 0.5) / (df[term] + 0.5) + 1)
            denom = freq + k1 * (1 - b + b * doc_len / avgdl)
            score += idf * (freq * (k1 + 1)) / denom
        scored.append((score, index, doc["id"]))
    scored.sort(key=lambda item: (-item[0], item[1]))
    return [{"id": doc_id, "score": score} for score, _, doc_id in scored[:k]]
