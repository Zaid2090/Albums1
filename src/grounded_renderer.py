from __future__ import annotations

import re
from typing import Any, Sequence

IP_PATTERN = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])")
PROHIBITED_PATTERN = re.compile(
    r"(?i)\b(exploit|payload|reverse shell|credential stuffing tool|malware family|attacker identity)\b"
)
ATTACK_SUBTYPE_PATTERN = re.compile(
    r"(?i)\b(ftp|ssh|brute[ -]?force|botnet|ddos|dos|infiltration|sql injection|xss|heartbleed)\b"
)


def normalize_output(text: str) -> str:
    text = text.strip().replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.strip() for line in text.split("\n") if line.strip())


def supports_detector_decision(feature: dict[str, Any], decision: str) -> bool:
    return (feature["direction"] == "supports_attack") == (decision.lower() == "attack")


def evidence_sentence(first: dict[str, Any], second: dict[str, Any], decision: str) -> str:
    first_supports = supports_detector_decision(first, decision)
    second_supports = supports_detector_decision(second, decision)
    if first_supports and second_supports:
        return (
            f"{first['name']} and {second['name']} provided the strongest local "
            "SHAP support for the detector decision."
        )
    if first_supports and not second_supports:
        return (
            f"{first['name']} supported the detector decision, while "
            f"{second['name']} opposed it according to local SHAP evidence."
        )
    if not first_supports and second_supports:
        return (
            f"{second['name']} supported the detector decision, while "
            f"{first['name']} opposed it according to local SHAP evidence."
        )
    return (
        f"{first['name']} and {second['name']} locally opposed the detector "
        "decision according to SHAP evidence."
    )


def render_grounded_fast(record: dict[str, Any]) -> str:
    first, second = record["top_features"][:2]
    decision = str(record["predicted_class"])
    confidence = float(record["confidence"])
    return (
        f"Summary: {decision} with {confidence:.3f} confidence\n"
        f"Evidence: {evidence_sentence(first, second, decision)}\n"
        "Action: Review network, authentication, and host logs, corroborate before containment."
    )


def strict_validate(
    record: dict[str, Any], text: str, known_feature_names: Sequence[str]
) -> dict[str, Any]:
    normalized = normalize_output(text)
    lines = normalized.splitlines()
    schema = (
        len(lines) == 3
        and lines[0].startswith("Summary:")
        and lines[1].startswith("Evidence:")
        and lines[2].startswith("Action:")
    )
    first, second = record["top_features"][:2]
    summary = lines[0] if len(lines) > 0 else ""
    evidence = lines[1] if len(lines) > 1 else ""
    action = lines[2] if len(lines) > 2 else ""
    decision = str(record["predicted_class"])
    exact_confidence = f"{float(record['confidence']):.3f}"
    decision_aligned = decision.lower() in summary.lower()
    confidence_aligned = exact_confidence in summary
    required_features = first["name"] in evidence and second["name"] in evidence
    allowed_names = {first["name"], second["name"]}
    unsupported = any(
        name in evidence and name not in allowed_names for name in known_feature_names
    )
    decision_support = {
        supports_detector_decision(first, decision),
        supports_detector_decision(second, decision),
    }
    evidence_lower = evidence.lower()
    if decision_support == {True}:
        direction_aligned = "support" in evidence_lower
    elif decision_support == {False}:
        direction_aligned = "oppos" in evidence_lower
    else:
        direction_aligned = "support" in evidence_lower and "oppos" in evidence_lower
    action_lower = action.lower()
    action_grounded = (
        "review" in action_lower
        and "log" in action_lower
        and "corroborat" in action_lower
        and "containment" in action_lower
    )
    prohibited = bool(
        IP_PATTERN.search(normalized)
        or PROHIBITED_PATTERN.search(normalized)
        or ATTACK_SUBTYPE_PATTERN.search(normalized)
    )
    valid = all(
        [
            schema,
            decision_aligned,
            confidence_aligned,
            required_features,
            direction_aligned,
            action_grounded,
            not unsupported,
            not prohibited,
        ]
    )
    return {
        "normalized": normalized,
        "schema_compliant": schema,
        "decision_aligned": decision_aligned,
        "confidence_aligned": confidence_aligned,
        "required_features_present": required_features,
        "direction_aligned": direction_aligned,
        "action_grounded": action_grounded,
        "unsupported_feature": unsupported,
        "prohibited_content": prohibited,
        "valid": valid,
    }
