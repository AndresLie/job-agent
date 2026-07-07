from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def export_cv_rewrite(brief_path: Path, output_path: Path) -> Path:
    payload = json.loads(brief_path.read_text(encoding="utf-8"))
    markdown = render_cv_rewrite_markdown(payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(markdown, encoding="utf-8")
    return output_path


def render_cv_rewrite_markdown(payload: dict[str, Any]) -> str:
    suggestions = payload.get("cv_rewrite_suggestions") or []
    diagnostics = payload.get("diagnostics") or {}
    lines = [
        f"# CV Rewrite Plan - {payload.get('job_title', 'Job')}",
        "",
        f"- Fit score: {payload.get('fit_score')}/100",
        f"- Verdict: {(payload.get('application_verdict') or {}).get('label', 'unknown')}",
        f"- LLM status: {diagnostics.get('llm_status', 'unknown')}",
        "",
        "## CV vs JD Feedback",
        "",
        (payload.get("cv_jd_review") or {}).get("reason", "No feedback available."),
        "",
        "## Suggested Bullets",
        "",
    ]
    if not suggestions:
        lines.append("No grounded rewrite suggestions were found.")
    for index, item in enumerate(suggestions, start=1):
        confidence = float(item.get("confidence", 0.0))
        safe_to_claim = confidence >= 0.65
        lines.extend(
            [
                f"### Bullet {index}",
                "",
                item.get("bullet", ""),
                "",
                f"- Source: `{item.get('source_path', 'unknown')}` chunk `{item.get('chunk_index', 'unknown')}`",
                f"- Target terms: {', '.join(item.get('target_terms') or []) or 'none'}",
                f"- Confidence: {confidence}",
                f"- Safe to claim now: {'yes' if safe_to_claim else 'needs verification'}",
                "",
            ]
        )
    lines.extend(["## Recommended Actions", ""])
    for action in payload.get("recommended_actions") or []:
        lines.append(f"- {action}")
    return "\n".join(lines).rstrip() + "\n"
