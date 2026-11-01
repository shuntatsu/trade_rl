"""Fail-closed research assurance contract for economic research changes."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass

SCHEMA_VERSION = "research_assurance_v1"

_STAGES = (
    "discovery",
    "development",
    "unused_validation",
    "prospective_paper",
    "production",
)
_LEVELS = (
    "software_validity",
    "mechanism_validity",
    "relative_improvement",
    "development_profitability",
    "unused_validation",
    "prospective_paper",
    "production",
)
_STAGE_MAX_LEVEL = {
    "discovery": "mechanism_validity",
    "development": "development_profitability",
    "unused_validation": "unused_validation",
    "prospective_paper": "prospective_paper",
    "production": "production",
}
_CHAIN = (
    "source",
    "availability",
    "feature_state",
    "model_strategy",
    "intent",
    "order",
    "fill",
    "accounting",
    "evidence",
    "decision",
)
_AUTHORITY_FIELDS = (
    "unit",
    "time",
    "state",
    "sign",
    "risk",
    "execution",
    "accounting",
    "evidence",
)
_TOP_FIELDS = {
    "schema_version",
    "identity",
    "thesis",
    "mechanism",
    "evidence",
    "claims",
}
_IDENTITY_FIELDS = {"protocol_head", "implementation_head"}
_THESIS_FIELDS = {
    "final_decision",
    "hypothesis",
    "economic_mechanism",
    "counter_hypothesis",
    "information_gain",
    "cheapest_falsifier",
    "stop_rule",
    "stage",
    "metric_proxy_rationale",
    "limitations",
}
_MECHANISM_FIELDS = {
    "chain",
    "authorities",
    "changed_boundaries",
    "independent_oracles",
}
_EVIDENCE_FIELDS = {
    "point_in_time",
    "common_accounting",
    "realistic_costs",
    "hard_risk",
    "terminal_state",
    "fit_development_unused_separated",
    "multi_symbol",
    "multi_period",
    "controls",
    "robustness",
    "independent_reconstruction",
    "no_development_rescue",
    "evidence_level",
}
_CLAIM_FIELDS = {
    "claim_level",
    "permitted",
    "forbidden",
    "next_authorized_action",
    "production_eligible",
    "live_trading_authorized",
}
_REVIEW_FIELDS = {
    "schema_version",
    "reviewed_record_digest",
    "protocol_head",
    "implementation_head",
    "reviewer_identity",
    "reviewer_execution_identity",
    "reviewer_independence",
    "result_blind",
    "g0",
    "g1",
    "g2",
    "status",
    "disposition",
    "blocking_findings",
    "missing_evidence",
    "rationale",
    "limitations",
}
REVIEW_SCHEMA_VERSION = "research_assurance_review_v1"


@dataclass(frozen=True)
class TrustedReviewEvidence:
    """Authenticated evidence from a separately trusted external adapter.

    Constructing this object from an untrusted author-controlled file does NOT
    establish authentication. In particular the network-free CLI deliberately
    does not expose an option to deserialize this authority from JSON.
    """

    review_body_sha256: str
    reviewer_identity: str
    author_identity: str
    reviewed_commit_sha: str
    reviewer_run_identity: str
    source_kind: str
    authenticated: bool


@dataclass(frozen=True)
class AssuranceResult:
    """Machine-verifiable outcome; thesis correctness still requires review."""

    status: str
    economic_execution_authorized: bool
    record_digest: str
    errors: tuple[str, ...]


def _canonical_bytes(value: object) -> bytes:
    try:
        rendered = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("assurance record must be canonical JSON data") from error
    return rendered.encode("utf-8")


def _mapping(value: object, field: str) -> dict[str, object]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{field} must be a string-keyed object")
    return dict(value)


def _exact_fields(
    value: Mapping[str, object],
    expected: set[str],
    field: str,
) -> None:
    missing = expected - set(value)
    if missing:
        raise ValueError(f"{field}.{sorted(missing)[0]} is required")
    extra = set(value) - expected
    if extra:
        raise ValueError(f"{field}.{sorted(extra)[0]} is not allowed")


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be non-empty text")
    return value


def _sha(value: object, field: str, length: int) -> str:
    text = _text(value, field)
    if len(text) != length or any(
        character not in "0123456789abcdef" for character in text
    ):
        raise ValueError(f"{field} must be a lowercase hexadecimal digest")
    return text


def _bool(value: object, field: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field} must be boolean")
    return value


def _strings(
    value: object,
    field: str,
    *,
    allow_empty: bool = False,
) -> list[str]:
    if not isinstance(value, list):
        raise ValueError(f"{field} must be an array")
    if not allow_empty and not value:
        raise ValueError(f"{field} must not be empty")
    result: list[str] = []
    for item in value:
        result.append(_text(item, field))
    return result


def _level(value: object, field: str) -> str:
    text = _text(value, field)
    if text not in _LEVELS:
        raise ValueError(f"{field} has unsupported assurance level")
    return text


def _rank(level: str) -> int:
    return _LEVELS.index(level)


def assurance_digest(record: Mapping[str, object]) -> str:
    """Hash the complete author-owned packet, never review fields."""

    payload = _mapping(record, "record")
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _validate(record: Mapping[str, object]) -> tuple[list[str], str, str, str, str]:
    errors: list[str] = []

    def capture(function: object) -> object | None:
        try:
            assert callable(function)
            return function()
        except (AssertionError, ValueError) as error:
            errors.append(str(error))
            return None

    payload = _mapping(record, "record")
    capture(lambda: _exact_fields(payload, _TOP_FIELDS, "record"))
    if payload.get("schema_version") != SCHEMA_VERSION:
        errors.append(f"schema_version must be {SCHEMA_VERSION}")

    identity = capture(lambda: _mapping(payload.get("identity"), "identity"))
    protocol_head = ""
    implementation_head = ""
    if isinstance(identity, dict):
        capture(lambda: _exact_fields(identity, _IDENTITY_FIELDS, "identity"))
        protocol = capture(
            lambda: _sha(identity.get("protocol_head"), "identity.protocol_head", 40)
        )
        implementation = capture(
            lambda: _sha(
                identity.get("implementation_head"),
                "identity.implementation_head",
                40,
            )
        )
        if isinstance(protocol, str):
            protocol_head = protocol
        if isinstance(implementation, str):
            implementation_head = implementation

    thesis = capture(lambda: _mapping(payload.get("thesis"), "thesis"))
    stage = ""
    if isinstance(thesis, dict):
        capture(lambda: _exact_fields(thesis, _THESIS_FIELDS, "thesis"))
        for name in (
            "final_decision",
            "hypothesis",
            "economic_mechanism",
            "counter_hypothesis",
            "information_gain",
            "cheapest_falsifier",
            "stop_rule",
            "metric_proxy_rationale",
            "limitations",
        ):
            capture(lambda name=name: _text(thesis.get(name), f"thesis.{name}"))
        raw_stage = capture(lambda: _text(thesis.get("stage"), "thesis.stage"))
        if isinstance(raw_stage, str):
            if raw_stage not in _STAGES:
                errors.append("thesis.stage has unsupported research stage")
            else:
                stage = raw_stage

    mechanism = capture(lambda: _mapping(payload.get("mechanism"), "mechanism"))
    if isinstance(mechanism, dict):
        capture(lambda: _exact_fields(mechanism, _MECHANISM_FIELDS, "mechanism"))
        chain = mechanism.get("chain")
        if chain != list(_CHAIN):
            errors.append(
                "mechanism.chain must cover the canonical research-to-decision path"
            )
        authorities = capture(
            lambda: _mapping(mechanism.get("authorities"), "mechanism.authorities")
        )
        if isinstance(authorities, dict):
            capture(
                lambda: _exact_fields(
                    authorities,
                    set(_AUTHORITY_FIELDS),
                    "mechanism.authorities",
                )
            )
            for name in _AUTHORITY_FIELDS:
                capture(
                    lambda name=name: _text(
                        authorities.get(name),
                        f"mechanism.authorities.{name}",
                    )
                )
        changed = capture(
            lambda: _strings(
                mechanism.get("changed_boundaries"), "mechanism.changed_boundaries"
            )
        )
        independent = capture(
            lambda: _mapping(
                mechanism.get("independent_oracles"),
                "mechanism.independent_oracles",
            )
        )
        if isinstance(changed, list):
            if len(set(changed)) != len(changed):
                errors.append("mechanism.changed_boundaries contains duplicates")
            if set(changed) - set(_AUTHORITY_FIELDS):
                errors.append(
                    "mechanism.changed_boundaries contains an unknown boundary"
                )
        if isinstance(independent, dict) and isinstance(changed, list):
            capture(
                lambda: _exact_fields(
                    independent, set(changed), "mechanism.independent_oracles"
                )
            )
            for boundary in changed:
                oracle = capture(
                    lambda boundary=boundary: _mapping(
                        independent.get(boundary),
                        f"mechanism.independent_oracles.{boundary}",
                    )
                )
                if not isinstance(oracle, dict):
                    continue
                fields = {"statement", "counterexample", "oracle", "known_limitations"}
                capture(
                    lambda boundary=boundary, oracle=oracle: _exact_fields(
                        oracle, fields, f"mechanism.independent_oracles.{boundary}"
                    )
                )
                for name in fields:
                    capture(
                        lambda name=name, boundary=boundary, oracle=oracle: _text(
                            oracle.get(name),
                            f"mechanism.independent_oracles.{boundary}.{name}",
                        )
                    )

    evidence = capture(lambda: _mapping(payload.get("evidence"), "evidence"))
    evidence_level = ""
    if isinstance(evidence, dict):
        capture(lambda: _exact_fields(evidence, _EVIDENCE_FIELDS, "evidence"))
        for name in (
            "point_in_time",
            "common_accounting",
            "realistic_costs",
            "hard_risk",
            "terminal_state",
            "fit_development_unused_separated",
            "multi_symbol",
            "multi_period",
            "no_development_rescue",
        ):
            capture(lambda name=name: _bool(evidence.get(name), f"evidence.{name}"))
        controls = capture(
            lambda: _strings(
                evidence.get("controls"),
                "evidence.controls",
                allow_empty=True,
            )
        )
        robustness = capture(
            lambda: _strings(
                evidence.get("robustness"),
                "evidence.robustness",
                allow_empty=True,
            )
        )
        capture(
            lambda: _text(
                evidence.get("independent_reconstruction"),
                "evidence.independent_reconstruction",
            )
        )
        raw_level = capture(
            lambda: _level(evidence.get("evidence_level"), "evidence.evidence_level")
        )
        if isinstance(raw_level, str):
            evidence_level = raw_level
            if _rank(raw_level) >= _rank("relative_improvement"):
                for name in (
                    "point_in_time",
                    "common_accounting",
                    "realistic_costs",
                    "hard_risk",
                    "terminal_state",
                    "fit_development_unused_separated",
                    "no_development_rescue",
                ):
                    if evidence.get(name) is not True:
                        errors.append(
                            f"evidence.{name} must be true for economic evidence"
                        )
                if not controls:
                    errors.append(
                        "evidence.controls must not be empty for economic evidence"
                    )
            if _rank(raw_level) >= _rank("development_profitability"):
                for name in ("multi_symbol", "multi_period"):
                    if evidence.get(name) is not True:
                        errors.append(
                            f"evidence.{name} must be true for profitability evidence"
                        )
                if not robustness:
                    errors.append(
                        "evidence.robustness must not be empty for profitability evidence"
                    )

    claims = capture(lambda: _mapping(payload.get("claims"), "claims"))
    claim_level = ""
    if isinstance(claims, dict):
        capture(lambda: _exact_fields(claims, _CLAIM_FIELDS, "claims"))
        raw_claim = capture(
            lambda: _level(claims.get("claim_level"), "claims.claim_level")
        )
        if isinstance(raw_claim, str):
            claim_level = raw_claim
        capture(lambda: _strings(claims.get("permitted"), "claims.permitted"))
        capture(lambda: _strings(claims.get("forbidden"), "claims.forbidden"))
        capture(
            lambda: _text(
                claims.get("next_authorized_action"),
                "claims.next_authorized_action",
            )
        )
        production = capture(
            lambda: _bool(
                claims.get("production_eligible"),
                "claims.production_eligible",
            )
        )
        live = capture(
            lambda: _bool(
                claims.get("live_trading_authorized"),
                "claims.live_trading_authorized",
            )
        )
        if stage == "development" and (
            claim_level == "production" or production is True or live is True
        ):
            errors.append(
                "development stage cannot authorize production or live trading"
            )
        if stage and claim_level:
            maximum = _STAGE_MAX_LEVEL[stage]
            if _rank(claim_level) > _rank(maximum):
                errors.append("claim_level exceeds the research stage")
        if production is True and claim_level != "production":
            errors.append("production_eligible requires production claim_level")
        if live is True and production is not True:
            errors.append("live_trading_authorized requires production_eligible")

    if evidence_level and claim_level and _rank(claim_level) > _rank(evidence_level):
        errors.append("claim_level exceeds evidence_level")

    return errors, stage, evidence_level, protocol_head, implementation_head


def _review_claims(
    review: Mapping[str, object],
    record_digest: str,
    protocol_head: str,
    implementation_head: str,
) -> list[str]:
    errors: list[str] = []
    _exact_fields(review, _REVIEW_FIELDS, "review")
    if review.get("schema_version") != REVIEW_SCHEMA_VERSION:
        errors.append("invalid review.schema_version")
    bindings = {
        "reviewed_record_digest": (record_digest, 64),
        "protocol_head": (protocol_head, 40),
        "implementation_head": (implementation_head, 40),
    }
    for name, (expected, length) in bindings.items():
        value = _sha(review.get(name), "review." + name, length)
        if value != expected:
            errors.append("review." + name + " is stale or mismatched")
    for name in ("reviewer_identity", "reviewer_execution_identity", "rationale"):
        _text(review.get(name), "review." + name)
    if review.get("reviewer_independence") != "ESTABLISHED":
        errors.append("reviewer independence is not established")
    if review.get("result_blind") is not True:
        errors.append("review is not result blind")
    status = review.get("status")
    if status not in {"pass", "blocked"}:
        errors.append("review.status must be pass or blocked")
    if review.get("g0") not in {"PASS", "FAIL", "NOT_ESTABLISHED"}:
        errors.append("invalid review.g0")
    if review.get("g1") not in {"PASS", "FAIL", "NOT_ESTABLISHED"}:
        errors.append("invalid review.g1")
    if review.get("g2") not in {"PASS", "FAIL", "EVIDENCE_BOUND", "NOT_ESTABLISHED"}:
        errors.append("invalid review.g2")
    for name in ("blocking_findings", "missing_evidence"):
        _strings(review.get(name), "review." + name, allow_empty=True)
    _strings(review.get("limitations"), "review.limitations")
    if status == "pass":
        if review.get("disposition") != "ASSURANCE_PASS":
            errors.append("review disposition is not a bounded assurance PASS")
        if any(review.get(name) != "PASS" for name in ("g0", "g1", "g2")):
            errors.append("not all G0/G1/G2 gates are PASS")
        if review.get("blocking_findings") or review.get("missing_evidence"):
            errors.append(
                "review contains unresolved blocking findings or missing evidence"
            )
    elif review.get("disposition") != "BLOCKED":
        errors.append("review status/disposition mismatch")
    return errors


def _check_trusted_source(
    review: Mapping[str, object],
    authority: object,
    implementation_head: str,
) -> list[str]:
    if authority is None:
        return ["authenticated external review authority is required"]
    errors: list[str] = []
    if not isinstance(authority, TrustedReviewEvidence):
        return ["review authority has an invalid type"]
    if authority.authenticated is not True:
        errors.append("review authority is not authenticated")
    if authority.source_kind != "authenticated_pr_review":
        errors.append("review authority must come from an authenticated PR review")
    if not authority.reviewer_identity or not authority.author_identity:
        errors.append("review authority has no principal identity")
    elif authority.reviewer_identity == authority.author_identity:
        errors.append("reviewer independence is not established: same author principal")
    if authority.reviewer_identity != review.get("reviewer_identity"):
        errors.append("review actor differs from authenticated authority")
    if authority.reviewer_run_identity != review.get("reviewer_execution_identity"):
        errors.append("review execution identity differs from authenticated authority")
    if authority.reviewed_commit_sha != implementation_head:
        errors.append("authenticated review commit_id is stale")
    try:
        body_sha256 = hashlib.sha256(_canonical_bytes(review)).hexdigest()
    except ValueError:
        errors.append("invalid canonical review data")
    else:
        if authority.review_body_sha256 != body_sha256:
            errors.append("authenticated review body digest does not match")
    return errors


def evaluate_assurance(
    record: Mapping[str, object],
    *,
    review: Mapping[str, object] | None = None,
    authority: TrustedReviewEvidence | None = None,
) -> AssuranceResult:
    """Check author packet and authenticated review binding; never authorize G4/G5.

    PASS is a bounded G0-G2 assurance state only. Network-free CLI checks can
    report UNREVIEWED/BLOCKED; only an external authenticated reviewer adapter
    may supply TrustedReviewEvidence for PASS. This module does not authorize
    economic execution, unused/final data access, or live trading.
    """
    try:
        payload = _mapping(record, "record")
        digest = assurance_digest(payload)
        errors, _, _, protocol_head, implementation_head = _validate(payload)
    except (ValueError, TypeError) as error:
        return AssuranceResult("BLOCKED", False, "", (str(error),))
    if errors:
        return AssuranceResult("BLOCKED", False, digest, tuple(errors))
    if review is None:
        if authority is not None:
            return AssuranceResult(
                "BLOCKED",
                False,
                digest,
                ("review is required with external authority",),
            )
        return AssuranceResult("UNREVIEWED", False, digest, ())
    try:
        review_payload = _mapping(review, "review")
        issues = _review_claims(
            review_payload, digest, protocol_head, implementation_head
        )
        issues += _check_trusted_source(review_payload, authority, implementation_head)
    except (ValueError, TypeError) as error:
        return AssuranceResult("BLOCKED", False, digest, (str(error),))
    if issues:
        return AssuranceResult("BLOCKED", False, digest, tuple(issues))
    return AssuranceResult(
        "PASS" if review_payload["status"] == "pass" else "BLOCKED",
        False,
        digest,
        (),
    )


__all__ = [
    "AssuranceResult",
    "TrustedReviewEvidence",
    "SCHEMA_VERSION",
    "REVIEW_SCHEMA_VERSION",
    "assurance_digest",
    "evaluate_assurance",
]
