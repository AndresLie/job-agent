from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from .documents import Chunk
from .embeddings import Embedder, cosine


class JsonVectorStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.records: list[dict] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.records = []
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = {}
        self.records = payload.get("records", []) if isinstance(payload, dict) else []

    def reset(self) -> None:
        self.records = []
        if self.path.exists():
            self.path.unlink()

    def upsert_chunks(self, chunks: list[Chunk], embedder: Embedder) -> None:
        ids = {chunk.id for chunk in chunks}
        self.records = [record for record in self.records if record["id"] not in ids]
        for chunk in chunks:
            record = asdict(chunk)
            record["embedding"] = embedder.embed(chunk.text)
            self.records.append(record)
        self.persist(embedder.name)

    def persist(self, embedder_name: str) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"embedding_model": embedder_name, "records": self.records}
        self.path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def query(
        self,
        query: str,
        embedder: Embedder,
        top_k: int = 5,
        categories: set[str] | None = None,
        exclude_categories: set[str] | None = None,
    ) -> list[dict]:
        query_embedding = embedder.embed(query)
        query_terms = keyword_tokens(query)
        scored = []
        for record in self.records:
            category = record.get("category", "general")
            if categories is not None and category not in categories:
                continue
            if exclude_categories is not None and category in exclude_categories:
                continue
            semantic = cosine(query_embedding, record.get("embedding", []))
            lexical = lexical_overlap(query_terms, keyword_tokens(record["text"] + " " + record["filename"]))
            score = 0.8 * semantic + 0.2 * lexical
            hit = dict(record)
            hit["score"] = score
            hit["semantic_score"] = semantic
            hit["lexical_score"] = lexical
            scored.append(hit)
        scored.sort(key=lambda item: item["score"], reverse=True)
        return scored[:top_k]


def clear_storage(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_dir():
            clear_storage(child)
            try:
                child.rmdir()
            except OSError:
                continue
        else:
            child.unlink()


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


def keyword_tokens(text: str) -> set[str]:
    import re

    return {
        term
        for term in re.findall(r"[a-z0-9]+", text.casefold())
        if len(term) > 2 and term not in STOPWORDS
    }


def lexical_overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left)
