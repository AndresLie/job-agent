from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path

from .bm25 import bm25_search


@dataclass(frozen=True)
class Observation:
    id: str
    summary: str
    tags: list[str]
    created_at: str
    source: str = "cli"


class MemoryStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.items: list[Observation] = []
        self.load()

    def load(self) -> None:
        if not self.path.exists():
            self.items = []
            return
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            payload = []
        self.items = [
            Observation(**item)
            for item in payload
            if isinstance(item, dict) and "summary" in item
        ] if isinstance(payload, list) else []

    def add(self, summary: str, tags: list[str] | None = None, source: str = "cli") -> bool:
        memory_id = hashlib.sha256(summary.encode("utf-8")).hexdigest()
        if any(item.id == memory_id for item in self.items):
            return False
        self.items.append(
            Observation(
                id=memory_id,
                summary=summary,
                tags=tags or [],
                created_at=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                source=source,
            )
        )
        self.persist()
        return True

    def persist(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.items], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def retrieve(self, query: str, k: int = 5) -> list[Observation]:
        docs = [
            {"id": item.id, "text": " ".join([item.summary, *item.tags])}
            for item in self.items
        ]
        ranked = bm25_search(query, docs, k)
        by_id = {item.id: item for item in self.items}
        return [by_id[result["id"]] for result in ranked if result["score"] > 0 and result["id"] in by_id]

    def injection(self, query: str, k: int = 5, char_budget: int = 1200) -> str:
        lines = []
        used = 0
        for item in self.retrieve(query, k):
            line = f"- {item.summary}"
            if used + len(line) > char_budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(lines)
