from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from .brief import generate_brief
from .documents import chunk_document
from .embeddings import HashingEmbedder
from .memory import MemoryStore
from .vector_store import JsonVectorStore


def evaluate_scoring_cases(cases_path: Path) -> dict[str, Any]:
    cases = load_cases(cases_path)
    rows = [score_case(case) for case in cases]
    passed = sum(1 for row in rows if row["passed"])
    return {
        "cases": len(rows),
        "passed": passed,
        "pass_rate": round(passed / len(rows), 3) if rows else 0.0,
        "failures": [row for row in rows if not row["passed"]],
        "per_case": rows,
    }


def load_cases(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def score_case(case: dict[str, Any]) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        embedder = HashingEmbedder()
        store = JsonVectorStore(root / "vectors.json")
        memory = MemoryStore(root / "memory.json")
        for category in ("resume", "projects", "experience"):
            for index, text in enumerate(case.get(f"{category}_docs", []), start=1):
                path = root / category / f"{category}_{index}.md"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text, encoding="utf-8")
                store.upsert_chunks(chunk_document(path, text, root=root), embedder)
        old_key = os.environ.pop("NVIDIA_API_KEY", None)
        try:
            brief = generate_brief(
                None,
                store,
                embedder,
                memory,
                job_text=case["job_text"],
                job_title=case.get("job_title", "Evaluation Job"),
            )
        finally:
            if old_key is not None:
                os.environ["NVIDIA_API_KEY"] = old_key
    score = int(brief["fit_score"])
    verdict = brief["application_verdict"]["label"]
    minimum = int(case.get("expected_score_min", 0))
    maximum = int(case.get("expected_score_max", 100))
    expected_verdicts = set(case.get("expected_verdicts", []))
    score_ok = minimum <= score <= maximum
    verdict_ok = not expected_verdicts or verdict in expected_verdicts
    return {
        "id": case.get("id", "unnamed"),
        "score": score,
        "expected_score_min": minimum,
        "expected_score_max": maximum,
        "verdict": verdict,
        "expected_verdicts": sorted(expected_verdicts),
        "passed": score_ok and verdict_ok,
        "score_ok": score_ok,
        "verdict_ok": verdict_ok,
    }
