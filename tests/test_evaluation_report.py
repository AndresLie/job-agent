import json

from career_copilot.evaluation_report import build_full_evaluation_report


def test_build_full_evaluation_report_includes_agent_quality(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    benchmarks = tmp_path / "benchmarks"
    outputs = tmp_path / "outputs"
    benchmarks.mkdir()
    outputs.mkdir()
    benchmarks.joinpath("scoring_cases.jsonl").write_text(
        '{"id":"weak","job_text":"AI Engineer Python RAG retrieval.",'
        '"resume_docs":["Skills: Python."],'
        '"expected_score_min":0,"expected_score_max":65,"expected_verdicts":["weak_match","not_competitive","stretch"]}\n',
        encoding="utf-8",
    )
    outputs.joinpath("latest_brief.json").write_text(
        json.dumps(
            {
                "schema_version": "1.1",
                "fit_score": 42,
                "application_verdict": {"label": "stretch"},
                "diagnostics": {"llm_status": "used"},
                "agent_trace": [
                    {
                        "agent_id": "cv_match",
                        "status": "success",
                        "structured": True,
                        "schema_errors": [],
                        "usage": {"latency_ms": 12.5, "prompt_chars": 100, "output_chars": 30},
                    },
                    {
                        "agent_id": "critic",
                        "status": "failed",
                        "structured": False,
                        "schema_errors": ["missing_output"],
                        "usage": {"latency_ms": 1.5, "prompt_chars": 80, "output_chars": 0},
                    },
                ],
                "agent_contradictions": [
                    {"type": "score_verdict_conflict", "severity": "high", "reason": "conflict"}
                ],
            }
        ),
        encoding="utf-8",
    )

    report = build_full_evaluation_report(root=tmp_path)

    assert report["scoring"]["status"] == "ok"
    assert report["agents"]["agent_count"] == 2
    assert report["agents"]["failed_agents"] == 1
    assert report["agents"]["structured_output_rate"] == 1.0
    assert report["agents"]["schema_version"] == "1.1"
    assert report["agents"]["schema_error_count"] == 1
    assert report["agents"]["total_latency_ms"] == 14.0
    assert report["agents"]["brief_modified_at"]
    assert report["agents"]["contradictions"] == 1
    assert report["overall"]["status"] == "needs_attention"
