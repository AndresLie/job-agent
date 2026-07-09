from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


REQUIRED_BRIEF_FIELDS = {
    "schema_version",
    "job_title",
    "fit_score",
    "role_family",
    "jd_requirements",
    "evidence_depth",
    "scoring_breakdown",
    "score_explanations",
    "weak_evidence",
    "cv_jd_review",
    "matched_evidence",
    "application_verdict",
    "cv_rewrite_suggestions",
    "skill_gaps",
    "recommended_actions",
    "citations",
    "confidence",
    "diagnostics",
}


def result_path(default: str = "result.json") -> Path:
    return Path(os.getenv("AI_JOB_COPILOT_RESULT_PATH") or default)


def validate_brief(payload: dict[str, Any]) -> tuple[bool, str]:
    missing = sorted(REQUIRED_BRIEF_FIELDS - payload.keys())
    if missing:
        return False, f"missing fields: {', '.join(missing)}"
    if not isinstance(payload["schema_version"], str):
        return False, "schema_version must be a string"
    if not isinstance(payload["matched_evidence"], list):
        return False, "matched_evidence must be a list"
    if not isinstance(payload["role_family"], str):
        return False, "role_family must be a string"
    if not isinstance(payload["jd_requirements"], dict):
        return False, "jd_requirements must be an object"
    if not isinstance(payload["evidence_depth"], dict):
        return False, "evidence_depth must be an object"
    if not isinstance(payload["scoring_breakdown"], dict):
        return False, "scoring_breakdown must be an object"
    if not isinstance(payload["score_explanations"], list):
        return False, "score_explanations must be a list"
    if not isinstance(payload["weak_evidence"], list):
        return False, "weak_evidence must be a list"
    if not isinstance(payload["cv_jd_review"], dict):
        return False, "cv_jd_review must be an object"
    if not isinstance(payload["application_verdict"], dict):
        return False, "application_verdict must be an object"
    if payload["application_verdict"].get("label") not in {
        "strong_match",
        "stretch",
        "weak_match",
        "not_competitive",
    }:
        return False, "application_verdict label is invalid"
    if not isinstance(payload["cv_rewrite_suggestions"], list):
        return False, "cv_rewrite_suggestions must be a list"
    if not isinstance(payload["skill_gaps"], list):
        return False, "skill_gaps must be a list"
    if not isinstance(payload["recommended_actions"], list):
        return False, "recommended_actions must be a list"
    if not 0 <= float(payload["fit_score"]) <= 100:
        return False, "fit_score must be between 0 and 100"
    if not 0 <= float(payload["confidence"]) <= 1:
        return False, "confidence must be between 0 and 1"
    if not isinstance(payload["diagnostics"], dict):
        return False, "diagnostics must be an object"
    return True, ""


def write_json_contract(payload: dict[str, Any], path: Path | None = None) -> Path:
    path = path or result_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, path)
    return path
