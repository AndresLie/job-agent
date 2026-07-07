from career_copilot.contracts import validate_brief


def test_validate_brief_accepts_valid_payload():
    payload = {
        "job_title": "AI Engineer",
        "fit_score": 80,
        "matched_evidence": [],
        "skill_gaps": [],
        "recommended_actions": [],
        "citations": [],
        "confidence": 0.8,
    }
    assert validate_brief(payload) == (True, "")


def test_validate_brief_rejects_missing_field():
    ok, reason = validate_brief({"job_title": "AI Engineer"})
    assert not ok
    assert "missing fields" in reason
