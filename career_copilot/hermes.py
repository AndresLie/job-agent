from __future__ import annotations

import json
import re
import time
from typing import Callable

from .answer import has_llm_config, nvidia_chat


AGENT_OUTPUT_SCHEMA = {
    "conclusion": str,
    "findings": list,
    "risks": list,
    "recommendations": list,
    "confidence": (float, int, type(None)),
}


def generate_brutal_llm_review(
    *,
    job_title: str,
    job_text: str,
    fit_score: int,
    application_verdict: dict,
    cv_rewrite_suggestions: list[dict],
    requirements: dict,
    evidence_depth: dict,
    scoring_breakdown: dict,
    weak_evidence: list[str],
    matched: list[str],
    hidden_terms: list[str],
    gaps: list[str],
    resume_hits: list[dict],
    supporting_hits: list[dict],
) -> dict | None:
    if not has_llm_config():
        return None
    resume_context = format_hits_for_prompt(resume_hits[:4])
    supporting_context = format_hits_for_prompt(supporting_hits[:5])
    rewrite_context = "\n".join(
        f"- {item['bullet']} (source: {item['source_path']} chunk {item['chunk_index']})"
        for item in cv_rewrite_suggestions
    )
    common_context = (
        f"Job title: {job_title}\n"
        f"JD excerpt:\n{job_text[:3500]}\n\n"
        f"Deterministic JD-to-CV score: {fit_score}/100\n"
        f"Deterministic application verdict: {application_verdict}\n"
        f"Role family: {requirements.get('role_family')}\n"
        f"JD requirements: {requirements}\n"
        f"Scoring breakdown: {scoring_breakdown}\n"
        f"Weak evidence: {', '.join(weak_evidence) or 'none'}\n"
        f"Evidence depth: {evidence_depth}\n"
        f"CV matched terms: {', '.join(matched) or 'none'}\n"
        f"Terms found in projects/experience but missing from CV: {', '.join(hidden_terms) or 'none'}\n"
        f"Terms with no evidence: {', '.join(gaps) or 'none'}\n"
    )
    system_prompt = (
        "You are one specialist inside Hermes, a multi-agent career review system for data scientist and AI roles. "
        "Use only the evidence supplied to your specific agent role. "
        "Do not reveal hidden chain-of-thought; return concise audit conclusions only. "
        "Do not flatter the candidate. If the evidence is insufficient, say so clearly. "
        "Keep the deterministic score as the source of truth. Do not invent metrics, employers, outcomes, or tools."
    )
    agent_trace = []
    for spec in build_agent_specs(common_context, resume_context, supporting_context, rewrite_context):
        prior = format_prior_agent_steps(agent_trace)
        result = run_hermes_agent_call(
            agent_id=spec["id"],
            title=spec["title"],
            role=spec["role"],
            input_summary=spec["input_summary"],
            system_prompt=system_prompt,
            user_prompt=(
                f"Prior agent conclusions:\n{prior or 'None yet.'}\n\n"
                f"{spec['instruction']}\n"
                "Return only a JSON object with keys: conclusion, findings, risks, recommendations, confidence. "
                "Use arrays for findings, risks, and recommendations. Keep values concise."
            ),
        )
        agent_trace.append(result)
    if not any(item["status"] == "success" for item in agent_trace):
        return None
    successful_steps = [
        {"id": item["agent_id"], "title": item["title"], "output": item["output"]}
        for item in agent_trace
        if item["status"] == "success"
    ]
    consensus = build_agent_consensus(
        trace=agent_trace,
        application_verdict=application_verdict,
        fit_score=fit_score,
        hidden_terms=hidden_terms,
        gaps=gaps,
        confidence=confidence_score(resume_hits, supporting_hits),
    )
    contradictions = detect_agent_contradictions(
        trace=agent_trace,
        fit_score=fit_score,
        application_verdict=application_verdict,
        gaps=gaps,
    )
    consensus["contradiction_count"] = len(contradictions)
    return {
        "steps": successful_steps,
        "agent_trace": agent_trace,
        "agent_consensus": consensus,
        "agent_contradictions": contradictions,
        "final_review": format_hermes_review(successful_steps),
    }


def build_agent_specs(common_context: str, resume_context: str, supporting_context: str, rewrite_context: str) -> list[dict]:
    return [
        {
            "id": "jd_analyst",
            "title": "JD Analyst Agent",
            "role": "Extract role intent and risk from the job description.",
            "input_summary": "JD text, detected requirements, deterministic scoring context.",
            "instruction": (
                f"{common_context}\n\n"
                "Agent task: identify the target role, hard requirements, soft requirements, seniority/depth expectations, "
                "and likely screening risks in this JD. Do not assess the candidate yet."
            ),
        },
        {
            "id": "cv_match",
            "title": "CV Match Agent",
            "role": "Audit only the current CV against the JD.",
            "input_summary": "JD requirements, score, and resume/CV evidence only.",
            "instruction": (
                f"{common_context}\n\n"
                f"CV evidence only:\n{resume_context or 'No CV evidence.'}\n\n"
                "Agent task: explain the current CV-to-JD fit using only the CV evidence above. "
                "Do not credit project or experience evidence that is not already in the CV."
            ),
        },
        {
            "id": "evidence_miner",
            "title": "Evidence Miner Agent",
            "role": "Find hidden project and experience evidence that could improve the CV.",
            "input_summary": "JD requirements, hidden terms, skill gaps, and project/experience evidence.",
            "instruction": (
                f"{common_context}\n\n"
                f"Project/experience evidence:\n{supporting_context or 'No project/experience evidence.'}\n\n"
                "Agent task: identify which hidden requirements can be safely moved into the CV, which are weak, "
                "and which have no supporting evidence."
            ),
        },
        {
            "id": "claim_verifier",
            "title": "Claim Verifier Agent",
            "role": "Audit generated CV bullets for unsupported or exaggerated claims.",
            "input_summary": "Grounded draft bullets and source project/experience evidence.",
            "instruction": (
                f"{common_context}\n\n"
                f"Project/experience evidence:\n{supporting_context or 'No project/experience evidence.'}\n\n"
                f"Grounded draft CV bullets:\n{rewrite_context or 'No grounded rewrite suggestions.'}\n\n"
                "Agent task: classify each draft bullet as supported, needs verification, or unsupported. "
                "Flag invented metrics, production claims, seniority claims, or deployment claims."
            ),
        },
        {
            "id": "critic",
            "title": "Critic Agent",
            "role": "Challenge inflated conclusions and weak evidence.",
            "input_summary": "All prior agent conclusions plus deterministic score and evidence summaries.",
            "instruction": (
                f"{common_context}\n\n"
                "Agent task: challenge the prior agents. Find overclaiming, weak evidence, missing requirements, "
                "and places where the candidate should not pretend to be stronger than the evidence supports."
            ),
        },
        {
            "id": "contradiction_judge",
            "title": "Contradiction Judge Agent",
            "role": "Detect conflicts between score, evidence, claims, and prior agents.",
            "input_summary": "Deterministic score, verdict, skill gaps, and prior agent conclusions.",
            "instruction": (
                f"{common_context}\n\n"
                "Agent task: identify contradictions across the deterministic score, CV Match Agent, Evidence Miner Agent, "
                "Claim Verifier Agent, and Critic Agent. Flag any final recommendation that would overstate fit, "
                "claim unsupported skills, or ignore known gaps."
            ),
        },
        {
            "id": "final_synthesizer",
            "title": "Final Synthesizer Agent",
            "role": "Produce the final brutally honest recommendation.",
            "input_summary": "All prior agent conclusions, deterministic score, and rewrite suggestions.",
            "instruction": (
                f"{common_context}\n\n"
                f"Grounded draft CV bullets:\n{rewrite_context or 'No grounded rewrite suggestions.'}\n\n"
                "Agent task: produce the final recommendation with current CV fit, hidden evidence quality, "
                "biggest gaps, apply guidance, and the top CV rewrite moves. Do not change the deterministic score."
            ),
        },
    ]


def run_hermes_agent_call(
    *,
    agent_id: str,
    title: str,
    role: str,
    input_summary: str,
    system_prompt: str,
    user_prompt: str,
    chat_fn: Callable[[str, str], str | None] | None = None,
) -> dict:
    chat = chat_fn or nvidia_chat
    started = time.perf_counter()
    raw_output = chat(system_prompt=system_prompt, user_prompt=user_prompt)
    latency_ms = round((time.perf_counter() - started) * 1000, 2)
    usage = {
        "latency_ms": latency_ms,
        "prompt_chars": len(system_prompt) + len(user_prompt),
        "output_chars": len(raw_output or ""),
    }
    if raw_output:
        structured_output, structured, validation_errors = parse_agent_output(raw_output)
        return {
            "agent_id": agent_id,
            "title": title,
            "role": role,
            "input_summary": input_summary,
            "status": "success",
            "output": render_agent_output(structured_output, raw_output),
            "raw_output": raw_output,
            "structured_output": structured_output,
            "structured": structured,
            "schema_errors": validation_errors,
            "usage": usage,
            "failure_reason": "",
        }
    return {
        "agent_id": agent_id,
        "title": title,
        "role": role,
        "input_summary": input_summary,
        "status": "failed",
        "output": "",
        "raw_output": "",
        "structured_output": {},
        "structured": False,
        "schema_errors": ["missing_output"],
        "usage": usage,
        "failure_reason": "NVIDIA chat returned no content.",
    }


def parse_agent_output(text: str) -> tuple[dict, bool, list[str]]:
    candidate = extract_json_object(text)
    if candidate:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            parsed = None
        if isinstance(parsed, dict):
            raw_errors = validate_raw_agent_output(parsed)
            normalized = normalize_agent_output(parsed)
            errors = raw_errors + validate_agent_output(normalized)
            return normalized, not errors, errors
    fallback = normalize_agent_output({"conclusion": text, "findings": [], "risks": [], "recommendations": []})
    return fallback, False, ["invalid_json"]


def extract_json_object(text: str) -> str:
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = re.sub(r"^```(?:json)?\s*", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s*```$", "", stripped)
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return ""
    return stripped[start : end + 1]


def normalize_agent_output(payload: dict) -> dict:
    return {
        "conclusion": str(payload.get("conclusion") or "").strip(),
        "findings": normalize_text_list(payload.get("findings")),
        "risks": normalize_text_list(payload.get("risks")),
        "recommendations": normalize_text_list(payload.get("recommendations")),
        "confidence": normalize_confidence(payload.get("confidence")),
    }


def validate_agent_output(payload: dict) -> list[str]:
    errors = []
    for key, expected in AGENT_OUTPUT_SCHEMA.items():
        if key not in payload:
            errors.append(f"missing_{key}")
        elif not isinstance(payload[key], expected):
            errors.append(f"invalid_{key}")
    if not payload.get("conclusion"):
        errors.append("empty_conclusion")
    return errors


def validate_raw_agent_output(payload: dict) -> list[str]:
    errors = []
    for key, expected in AGENT_OUTPUT_SCHEMA.items():
        if key not in payload:
            errors.append(f"missing_{key}")
        elif not isinstance(payload[key], expected):
            errors.append(f"invalid_{key}")
    if not str(payload.get("conclusion") or "").strip():
        errors.append("empty_conclusion")
    return errors


def normalize_text_list(value) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    return [str(value).strip()] if str(value).strip() else []


def normalize_confidence(value) -> float | None:
    if value in (None, ""):
        return None
    try:
        return round(max(0.0, min(1.0, float(value))), 2)
    except (TypeError, ValueError):
        return None


def render_agent_output(structured_output: dict, raw_output: str) -> str:
    if not structured_output:
        return raw_output
    lines = []
    conclusion = structured_output.get("conclusion")
    if conclusion:
        lines.append(str(conclusion))
    for label in ("findings", "risks", "recommendations"):
        values = structured_output.get(label) or []
        if values:
            lines.append("")
            lines.append(label.replace("_", " ").title() + ":")
            lines.extend(f"- {item}" for item in values)
    confidence = structured_output.get("confidence")
    if confidence is not None:
        lines.extend(["", f"Confidence: {confidence}"])
    return "\n".join(lines) if lines else raw_output


def build_default_agent_consensus(
    *,
    application_verdict: dict,
    fit_score: int,
    hidden_terms: list[str],
    gaps: list[str],
    confidence: float,
) -> dict:
    return {
        "current_cv_fit": f"{score_verdict(fit_score)} ({fit_score}/100)",
        "hidden_evidence_quality": "available" if hidden_terms else "none detected",
        "biggest_gaps": gaps[:5],
        "apply_recommendation": application_verdict.get("label", "unknown"),
        "confidence": round(confidence, 2),
        "contradiction_count": 0,
        "source": "deterministic_fallback",
    }


def build_agent_consensus(
    *,
    trace: list[dict],
    application_verdict: dict,
    fit_score: int,
    hidden_terms: list[str],
    gaps: list[str],
    confidence: float,
) -> dict:
    consensus = build_default_agent_consensus(
        application_verdict=application_verdict,
        fit_score=fit_score,
        hidden_terms=hidden_terms,
        gaps=gaps,
        confidence=confidence,
    )
    final = next((item for item in trace if item["agent_id"] == "final_synthesizer" and item["status"] == "success"), None)
    critic = next((item for item in trace if item["agent_id"] == "critic" and item["status"] == "success"), None)
    consensus.update(
        {
            "source": "hermes_multi_agent",
            "agent_count": len(trace),
            "successful_agents": len([item for item in trace if item["status"] == "success"]),
            "total_latency_ms": round(sum(item.get("usage", {}).get("latency_ms", 0.0) for item in trace), 2),
            "total_prompt_chars": sum(item.get("usage", {}).get("prompt_chars", 0) for item in trace),
            "total_output_chars": sum(item.get("usage", {}).get("output_chars", 0) for item in trace),
            "final_summary": final["output"] if final else "",
            "critic_summary": critic["output"] if critic else "",
        }
    )
    return consensus


def detect_agent_contradictions(
    *,
    trace: list[dict],
    fit_score: int,
    application_verdict: dict,
    gaps: list[str],
) -> list[dict]:
    contradictions = []
    joined = "\n".join(item.get("output", "") for item in trace if item.get("status") == "success").casefold()
    final = next((item for item in trace if item.get("agent_id") == "final_synthesizer"), {})
    final_text = str(final.get("output") or "").casefold()
    verifier = next((item for item in trace if item.get("agent_id") == "claim_verifier"), {})
    verifier_text = str(verifier.get("output") or "").casefold()

    if fit_score < 50 and contains_positive_strong_fit_language(final_text):
        contradictions.append(
            {
                "type": "score_verdict_conflict",
                "severity": "high",
                "reason": f"Final agent used strong-match language while deterministic score is {fit_score}/100.",
            }
        )
    if application_verdict.get("apply_now") is False and contains_positive_apply_now_language(final_text):
        contradictions.append(
            {
                "type": "apply_recommendation_conflict",
                "severity": "high",
                "reason": "Final agent recommends applying now while deterministic verdict says not to apply now.",
            }
        )
    if ("unsupported" in verifier_text or "needs verification" in verifier_text) and contains_positive_safe_claim_language(final_text):
        contradictions.append(
            {
                "type": "claim_safety_conflict",
                "severity": "medium",
                "reason": "Final agent says claims are safe while Claim Verifier reported unsupported or needs-verification claims.",
            }
        )
    for gap in gaps[:6]:
        term = str(gap).replace("_", " ").casefold()
        if term and term in final_text and any(marker in final_text for marker in ("fully covers", "clearly covers", "strong evidence")):
            contradictions.append(
                {
                    "type": "known_gap_conflict",
                    "severity": "medium",
                    "reason": f"Final agent appears to credit known missing requirement: {gap}.",
                }
            )
    if "no major gap" in joined and gaps:
        contradictions.append(
            {
                "type": "gap_summary_conflict",
                "severity": "medium",
                "reason": "An agent says there are no major gaps while deterministic scoring found missing requirements.",
            }
        )
    return contradictions


def contains_positive_strong_fit_language(text: str) -> bool:
    positive_patterns = (
        r"(?<!not a )\bstrong match\b",
        r"(?<!not an )\bexcellent match\b",
        r"(?<!not )\bhighly competitive\b",
        r"\bapply immediately\b",
    )
    negative_patterns = (
        r"\bnot a strong match\b",
        r"\bnot an excellent match\b",
        r"\bnot highly competitive\b",
        r"\bdo not apply immediately\b",
    )
    return has_positive_phrase(text, positive_patterns, negative_patterns)


def contains_positive_apply_now_language(text: str) -> bool:
    positive_patterns = (r"\bapply now\b", r"\bapply immediately\b")
    negative_patterns = (
        r"\bdo not apply now\b",
        r"\bdon't apply now\b",
        r"\bnot apply now\b",
        r"\bdo not apply immediately\b",
        r"\bdon't apply immediately\b",
        r"\bnot apply immediately\b",
    )
    return has_positive_phrase(text, positive_patterns, negative_patterns)


def contains_positive_safe_claim_language(text: str) -> bool:
    positive_patterns = (r"\bsafe to claim\b", r"\bsafe to include\b")
    negative_patterns = (r"\bnot safe to claim\b", r"\bnot safe to include\b", r"\bunsafe to claim\b")
    return has_positive_phrase(text, positive_patterns, negative_patterns)


def has_positive_phrase(text: str, positive_patterns: tuple[str, ...], negative_patterns: tuple[str, ...]) -> bool:
    if any(re.search(pattern, text) for pattern in negative_patterns):
        return False
    return any(re.search(pattern, text) for pattern in positive_patterns)


def format_prior_agent_steps(steps: list[dict]) -> str:
    lines = []
    for step in steps:
        if step.get("status") == "success":
            lines.append(f"{step['title']}:\n{step['output']}")
        else:
            lines.append(f"{step['title']}: failed ({step.get('failure_reason', 'unknown failure')})")
    return "\n\n".join(lines)


def format_hermes_review(steps: list[dict]) -> str:
    lines = ["Hermes multi-step review"]
    for index, step in enumerate(steps, start=1):
        lines.extend(["", f"{index}. {step['title']}", step["output"]])
    return "\n".join(lines)


def format_hits_for_prompt(hits: list[dict]) -> str:
    lines = []
    for index, hit in enumerate(hits, start=1):
        text = re.sub(r"\s+", " ", hit["text"]).strip()[:700]
        lines.append(f"[{index}] {hit.get('category', 'general')}/{hit['filename']} chunk {hit['chunk_index']}: {text}")
    return "\n".join(lines)


def score_verdict(score: int) -> str:
    if score >= 75:
        return "strong_cv_match"
    if score >= 50:
        return "moderate_cv_match"
    if score >= 30:
        return "weak_cv_match"
    return "poor_cv_match"


def confidence_score(resume_hits: list[dict], supporting_hits: list[dict]) -> float:
    if resume_hits and supporting_hits:
        return 0.82
    if resume_hits:
        return 0.68
    if supporting_hits:
        return 0.48
    return 0.25
