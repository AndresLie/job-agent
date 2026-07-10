from career_copilot.contracts import validate_brief


def test_validate_brief_accepts_valid_payload():
    payload = {
        "schema_version": "1.4",
        "job_title": "AI Engineer",
        "fit_score": 80,
        "role_family": "ai_engineer",
        "active_resume": {},
        "cv_rankings": [],
        "jd_requirements": {"required": ["python"], "preferred": [], "responsibilities": []},
        "company_context_review": {
            "status": "no_external_sources",
            "summary": "No external company sources were used.",
            "sources_considered": 0,
            "weight_recommendations": [],
        },
        "evidence_depth": {"by_term": {}},
        "scoring_breakdown": {
            "required_skill_coverage": 45,
            "responsibility_alignment": 20,
            "evidence_depth": 10,
            "quantified_impact": 5,
            "preferred_coverage": 0,
        },
        "score_explanations": [],
        "score_rationale_summary": {
            "primary_role": "ai_engineer",
            "secondary_roles": [],
            "active_resume": "",
            "summary": "Primary scoring role: ai engineer.",
            "matched_strengths": ["python"],
            "missing_core_stack": [],
            "weak_evidence": [],
            "hidden_cv_opportunities": [],
            "skill_gaps": [],
            "score_cap": None,
            "cap_reason": "",
            "what_would_raise_score": [],
        },
        "weak_evidence": [],
        "cv_jd_review": {
            "score": 80,
            "verdict": "strong_cv_match",
            "reason": "Strong evidence.",
            "matched_terms": ["python"],
            "missing_from_cv": [],
            "weak_evidence": [],
            "scoring_breakdown": {},
        },
        "matched_evidence": [],
        "application_verdict": {
            "label": "strong_match",
            "apply_now": True,
            "risk_level": "low",
            "reason": "Strong evidence.",
        },
        "cv_rewrite_suggestions": [],
        "cv_improvement_review": {
            "direct_suggestions": [],
            "adjacent_recommendations": [],
            "no_evidence_gaps": [],
            "summary": "No project or experience evidence changed the CV improvement plan.",
        },
        "skill_gaps": [],
        "recommended_actions": [],
        "citations": [],
        "confidence": 0.8,
        "diagnostics": {},
    }
    assert validate_brief(payload) == (True, "")


def test_validate_brief_rejects_missing_field():
    ok, reason = validate_brief({"job_title": "AI Engineer"})
    assert not ok
    assert "missing fields" in reason
