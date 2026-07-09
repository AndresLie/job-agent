from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .embeddings import build_embedder
from .evaluate import evaluate_retrieval
from .score_eval import evaluate_scoring_cases
from .vector_store import JsonVectorStore


def build_full_evaluation_report(
    *,
    root: Path,
    scoring_cases: Path | None = None,
    retrieval_queries: Path | None = None,
    index_path: Path | None = None,
    brief_path: Path | None = None,
    k: int = 5,
) -> dict[str, Any]:
    scoring_cases = scoring_cases or root / "benchmarks" / "scoring_cases.jsonl"
    retrieval_queries = retrieval_queries or root / "benchmarks" / "queries.jsonl"
    index_path = index_path or root / "storage" / "vector_store.json"
    brief_path = brief_path or root / "outputs" / "latest_brief.json"
    report = {
        "scoring": evaluate_scoring_section(scoring_cases),
        "retrieval": evaluate_retrieval_section(retrieval_queries, index_path, k),
        "agents": evaluate_agent_section(brief_path),
        "jd_extraction": evaluate_jd_extraction_section(root),
    }
    report["overall"] = summarize_overall(report)
    return report


def evaluate_scoring_section(cases_path: Path) -> dict[str, Any]:
    if not cases_path.exists():
        return {"status": "missing", "cases": 0, "passed": 0, "pass_rate": 0.0, "failures": []}
    try:
        return {"status": "ok", **evaluate_scoring_cases(cases_path)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "cases": 0, "passed": 0, "pass_rate": 0.0, "failures": []}


def evaluate_retrieval_section(queries_path: Path, index_path: Path, k: int) -> dict[str, Any]:
    if not queries_path.exists():
        return {"status": "missing_queries", "queries": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    if not index_path.exists():
        return {"status": "not_indexed", "queries": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    store = JsonVectorStore(index_path)
    if not store.records:
        return {"status": "not_indexed", "queries": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}
    try:
        return {"status": "ok", **evaluate_retrieval(queries_path, store, build_embedder("hashing"), k=k)}
    except Exception as exc:
        return {"status": "error", "error": str(exc), "queries": 0, "recall_at_k": 0.0, "mrr": 0.0, "ndcg_at_k": 0.0}


def evaluate_agent_section(brief_path: Path) -> dict[str, Any]:
    if not brief_path.exists():
        return {
            "status": "missing_brief",
            "brief_path": str(brief_path),
            "brief_modified_at": "",
            "agent_count": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "failure_rate": 0.0,
            "structured_output_rate": 0.0,
            "schema_error_count": 0,
            "total_latency_ms": 0.0,
            "prompt_chars": 0,
            "output_chars": 0,
            "contradictions": 0,
        }
    try:
        payload = json.loads(brief_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "status": "error",
            "error": str(exc),
            "brief_path": str(brief_path),
            "brief_modified_at": "",
            "agent_count": 0,
            "successful_agents": 0,
            "failed_agents": 0,
            "failure_rate": 0.0,
            "structured_output_rate": 0.0,
            "schema_error_count": 0,
            "total_latency_ms": 0.0,
            "prompt_chars": 0,
            "output_chars": 0,
            "contradictions": 0,
        }
    trace = payload.get("agent_trace") or []
    agent_count = len(trace)
    successful = len([item for item in trace if item.get("status") == "success"])
    failed = len([item for item in trace if item.get("status") != "success"])
    structured = len([item for item in trace if item.get("structured") is True])
    schema_error_count = sum(len(item.get("schema_errors") or []) for item in trace)
    total_latency_ms = round(sum(float(item.get("usage", {}).get("latency_ms", 0.0)) for item in trace), 2)
    prompt_chars = sum(int(item.get("usage", {}).get("prompt_chars", 0)) for item in trace)
    output_chars = sum(int(item.get("usage", {}).get("output_chars", 0)) for item in trace)
    contradictions = payload.get("agent_contradictions") or []
    return {
        "status": "ok" if agent_count else "no_agent_trace",
        "brief_path": str(brief_path),
        "brief_modified_at": modified_at(brief_path),
        "schema_version": payload.get("schema_version", "unknown"),
        "agent_count": agent_count,
        "successful_agents": successful,
        "failed_agents": failed,
        "failure_rate": round(failed / agent_count, 3) if agent_count else 0.0,
        "structured_output_rate": round(structured / successful, 3) if successful else 0.0,
        "schema_error_count": schema_error_count,
        "total_latency_ms": total_latency_ms,
        "prompt_chars": prompt_chars,
        "output_chars": output_chars,
        "contradictions": len(contradictions),
        "contradiction_findings": contradictions[:5],
        "fit_score": payload.get("fit_score"),
        "verdict": (payload.get("application_verdict") or {}).get("label"),
        "llm_status": (payload.get("diagnostics") or {}).get("llm_status"),
    }


def modified_at(path: Path) -> str:
    try:
        return datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
    except OSError:
        return ""


def evaluate_jd_extraction_section(root: Path) -> dict[str, Any]:
    directory = root / "outputs" / "runs"
    if not directory.exists():
        return {"runs": 0, "successful": 0, "success_rate": 0.0, "methods": {}}
    methods: dict[str, int] = {}
    successful = 0
    total = 0
    for path in directory.glob("review-*.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "diagnostics" not in payload:
            continue
        total += 1
        diagnostics = payload.get("diagnostics") or {}
        method = diagnostics.get("job_extraction_method") or "unknown"
        methods[method] = methods.get(method, 0) + 1
        if method != "unknown" and int(diagnostics.get("job_text_chars") or 0) > 0:
            successful += 1
    return {
        "runs": total,
        "successful": successful,
        "success_rate": round(successful / total, 3) if total else 0.0,
        "methods": methods,
    }


def summarize_overall(report: dict[str, Any]) -> dict[str, Any]:
    blocking = []
    if report["scoring"].get("status") != "ok":
        blocking.append("scoring_not_ok")
    if report["scoring"].get("failures"):
        blocking.append("scoring_failures")
    if report["agents"].get("failed_agents", 0) > 0:
        blocking.append("agent_failures")
    if report["agents"].get("contradictions", 0) > 0:
        blocking.append("agent_contradictions")
    return {
        "status": "pass" if not blocking else "needs_attention",
        "blocking_issues": blocking,
        "portfolio_ready": not blocking and report["scoring"].get("pass_rate", 0.0) >= 0.9,
    }
