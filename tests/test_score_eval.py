import json

from career_copilot.score_eval import evaluate_scoring_cases


def test_evaluate_scoring_cases_reports_pass_rate(tmp_path, monkeypatch):
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "id": "strong",
                        "job_text": "AI Engineer Python RAG retrieval evaluation systems.",
                        "resume_docs": [
                            "Deployed Python RAG retrieval evaluation systems in production and improved coverage by 30%."
                        ],
                        "expected_score_min": 70,
                        "expected_score_max": 100,
                        "expected_verdicts": ["strong_match", "stretch"],
                    }
                ),
                json.dumps(
                    {
                        "id": "weak",
                        "job_text": "AI Engineer Python RAG retrieval evaluation systems.",
                        "resume_docs": ["Python RAG retrieval evaluation."],
                        "expected_score_min": 0,
                        "expected_score_max": 65,
                    }
                ),
            ]
        ),
        encoding="utf-8",
    )
    result = evaluate_scoring_cases(cases)
    assert result["cases"] == 2
    assert result["passed"] == 2
    assert result["pass_rate"] == 1.0
