"""Independent final-audit oracle for Issue #610 real interpretation artifacts."""

from __future__ import annotations

import argparse
import json
import math
from collections.abc import Mapping
from pathlib import Path
from statistics import median
from typing import cast

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from tools.tmp_issue610_exact_once_orchestrator import PRECOMPUTE_RUN_ID, SEEDS, SYMBOLS
from tools.tmp_issue610_ppo_global_btc_regime_evaluation import (
    ACCEPT_CANDIDATE,
    INVALID,
    KEEP_BASELINE,
)

ISSUE_NUMBER = 610
AUDIT_SCHEMA = "issue610_interpretation_final_audit_v1"
RECOVERY_RUN_ID = 35102584101
FAILED_EXECUTION_RUN_ID = 35102237151
VALID_DECISIONS = frozenset({ACCEPT_CANDIDATE, KEEP_BASELINE, INVALID})


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str) for key in value
    ):
        raise RuntimeError(f"{field} malformed")
    return cast(Mapping[str, object], value)


def _string_list(value: object, *, field: str) -> list[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise RuntimeError(f"{field} malformed")
    return cast(list[str], value)


def _load_canonical(path: Path) -> dict[str, object]:
    raw = path.read_bytes()
    payload = json.loads(raw)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{path.name} must contain an object")
    if canonical_json_bytes(payload) != raw:
        raise RuntimeError(f"{path.name} is not canonical JSON")
    digest = payload.get("content_digest")
    if not isinstance(digest, str):
        raise RuntimeError(f"{path.name} content_digest missing")
    unsigned = dict(payload)
    unsigned.pop("content_digest")
    if content_digest(unsigned) != digest:
        raise RuntimeError(f"{path.name} content_digest mismatch")
    return cast(dict[str, object], payload)


def _assert_tree_identity(left: Path, right: Path) -> tuple[str, ...]:
    left_files = {
        path.relative_to(left).as_posix(): path.read_bytes()
        for path in left.rglob("*")
        if path.is_file()
    }
    right_files = {
        path.relative_to(right).as_posix(): path.read_bytes()
        for path in right.rglob("*")
        if path.is_file()
    }
    _require(set(left_files) == set(right_files), "result file roster mismatch")
    for relative_path, left_bytes in left_files.items():
        _require(
            right_files[relative_path] == left_bytes,
            f"result byte mismatch: {relative_path}",
        )
    return tuple(sorted(left_files))


def _validate_absolute_diagnostic(v2: Mapping[str, object]) -> None:
    diagnostic = _mapping(
        v2.get("absolute_candidate_profitability_diagnostic"),
        field="absolute candidate profitability diagnostic",
    )
    _require(
        diagnostic.get("schema_version")
        == "issue610_absolute_candidate_ppo_diagnostic_v1",
        "absolute diagnostic schema drift",
    )
    _require(
        diagnostic.get("gates_development_decision") is False,
        "absolute diagnostic unexpectedly gates development decision",
    )
    by_symbol = _mapping(diagnostic.get("by_symbol"), field="absolute by_symbol")
    _require(tuple(by_symbol) == SYMBOLS, "absolute diagnostic symbol roster drift")

    medians: list[float] = []
    for symbol in SYMBOLS:
        entry = _mapping(by_symbol.get(symbol), field=f"absolute {symbol}")
        value = entry.get("median_candidate_total_return")
        _require(
            not isinstance(value, bool) and isinstance(value, (int, float)),
            f"absolute {symbol} median malformed",
        )
        resolved = float(cast(float, value))
        _require(math.isfinite(resolved), f"absolute {symbol} median non-finite")
        medians.append(resolved)

    cross = _mapping(diagnostic.get("cross_symbol"), field="absolute cross_symbol")
    _require(cross.get("symbol_count") == len(SYMBOLS), "absolute symbol count drift")
    _require(
        cross.get("positive_symbol_count") == sum(value > 0.0 for value in medians),
        "absolute positive symbol count mismatch",
    )
    _require(
        cross.get("negative_symbol_count") == sum(value < 0.0 for value in medians),
        "absolute negative symbol count mismatch",
    )
    _require(
        cross.get("zero_symbol_count") == sum(value == 0.0 for value in medians),
        "absolute zero symbol count mismatch",
    )
    _require(
        cross.get("median_candidate_total_return") == float(median(medians)),
        "absolute cross-symbol median mismatch",
    )
    _require(
        cross.get("worst_candidate_total_return") == min(medians),
        "absolute worst-symbol return mismatch",
    )
    _require(
        cross.get("best_candidate_total_return") == max(medians),
        "absolute best-symbol return mismatch",
    )


def audit_interpretation(
    *,
    producer_root: Path,
    fresh_root: Path,
    candidate_root: Path,
    precompute_path: Path,
    output_root: Path,
    interpretation_run_id: int,
    candidate_execution_run_id: int,
    candidate_artifact_id: int,
    candidate_artifact_digest: str,
    precompute_artifact_id: int,
    precompute_artifact_digest: str,
    producer_result_artifact_id: int,
    producer_result_artifact_digest: str,
    fresh_result_artifact_id: int,
    fresh_result_artifact_digest: str,
) -> dict[str, object]:
    """Audit byte-identical producer/fresh results and all frozen authority links."""

    result_files = _assert_tree_identity(producer_root, fresh_root)
    strict = _load_canonical(producer_root / "strict-precheck.json")
    decision = _load_canonical(producer_root / "decision.json")
    candidate = _load_canonical(candidate_root / "candidate-authority.json")
    recovery_binding = _load_canonical(candidate_root / "recovery-binding.json")
    recovery_authority = _load_canonical(candidate_root / "recovery-authority.json")
    precompute = _load_canonical(precompute_path)

    _require(candidate_execution_run_id > 0, "candidate execution run id invalid")
    _require(candidate_artifact_id > 0, "candidate artifact id invalid")
    _require(precompute_artifact_id > 0, "precompute artifact id invalid")
    for label, digest in (
        ("candidate artifact", candidate_artifact_digest),
        ("precompute artifact", precompute_artifact_digest),
        ("producer result artifact", producer_result_artifact_digest),
        ("fresh result artifact", fresh_result_artifact_digest),
    ):
        _require(digest.startswith("sha256:"), f"{label} API digest malformed")

    _require(candidate.get("issue_number") == ISSUE_NUMBER, "candidate issue drift")
    _require(
        candidate.get("execution_run_id") == candidate_execution_run_id,
        "candidate execution run mismatch",
    )
    _require(
        candidate.get("precompute_run_id") == PRECOMPUTE_RUN_ID,
        "candidate precompute run mismatch",
    )
    _require(
        candidate.get("precompute_artifact_id") == precompute_artifact_id,
        "candidate precompute artifact id mismatch",
    )
    _require(
        candidate.get("precompute_artifact_api_digest") == precompute_artifact_digest,
        "candidate precompute artifact digest mismatch",
    )
    _require(
        candidate.get("precompute_authority_content_digest")
        == precompute.get("content_digest"),
        "candidate/precompute content binding mismatch",
    )
    _require(candidate.get("ppo_seeds") == list(SEEDS), "candidate seed roster drift")
    _require(candidate.get("symbols") == list(SYMBOLS), "candidate symbol roster drift")
    for field, expected in (
        ("baseline_retrained", False),
        ("candidate_training_performed", True),
        ("economic_values_interpreted", False),
        ("final_test_accessed", False),
        ("operational_eligibility_established", False),
        ("production_eligible", False),
        ("live_trading_authorized", False),
        ("merge_authorized", False),
    ):
        _require(candidate.get(field) is expected, f"candidate boundary mismatch: {field}")

    _require(
        recovery_binding.get("candidate_execution_run_id") == candidate_execution_run_id,
        "recovery binding candidate run mismatch",
    )
    _require(
        recovery_binding.get("recovery_run_id") == RECOVERY_RUN_ID,
        "recovery binding recovery run mismatch",
    )
    _require(
        recovery_binding.get("failed_execution_run_id") == FAILED_EXECUTION_RUN_ID,
        "recovery binding failed run mismatch",
    )
    _require(
        recovery_binding.get("precompute_run_id") == PRECOMPUTE_RUN_ID,
        "recovery binding precompute run mismatch",
    )
    _require(
        recovery_binding.get("precompute_artifact_id") == precompute_artifact_id,
        "recovery binding precompute artifact id mismatch",
    )
    _require(
        recovery_binding.get("precompute_artifact_api_digest")
        == precompute_artifact_digest,
        "recovery binding precompute artifact digest mismatch",
    )
    _require(
        recovery_binding.get("candidate_evidence_fingerprint")
        == candidate.get("candidate_evidence_fingerprint"),
        "recovery binding candidate fingerprint mismatch",
    )
    _require(
        recovery_binding.get("candidate_authority_content_digest")
        == candidate.get("content_digest"),
        "recovery binding candidate authority mismatch",
    )
    _require(
        recovery_binding.get("recovery_authority_content_digest")
        == recovery_authority.get("content_digest"),
        "recovery binding recovery authority mismatch",
    )
    for field, expected in (
        ("failed_run_candidate_training_performed", False),
        ("recovery_candidate_training_performed", True),
        ("economic_values_interpreted", False),
        ("final_test_accessed", False),
        ("production_eligible", False),
        ("live_trading_authorized", False),
        ("merge_authorized", False),
    ):
        _require(
            recovery_binding.get(field) is expected,
            f"recovery binding boundary mismatch: {field}",
        )
    _require(
        recovery_authority.get("recovery_candidate_execution_authorized") is True,
        "recovery authority did not authorize candidate execution",
    )
    _require(
        recovery_authority.get("failed_run_candidate_training_performed") is False,
        "recovery authority says failed run trained candidate",
    )
    _require(
        recovery_authority.get("failed_run_candidate_artifact_published") is False,
        "recovery authority says failed run published candidate",
    )
    _require(
        recovery_authority.get("correct_precompute_producer_api_digest")
        == precompute_artifact_digest,
        "recovery authority precompute digest mismatch",
    )

    for payload, label in ((strict, "strict"), (decision, "decision")):
        _require(payload.get("issue_number") == ISSUE_NUMBER, f"{label} issue drift")
        _require(
            payload.get("interpretation_run_id") == interpretation_run_id,
            f"{label} interpretation run mismatch",
        )
        _require(
            payload.get("candidate_artifact_id") == candidate_artifact_id,
            f"{label} candidate artifact id mismatch",
        )
        _require(
            payload.get("candidate_artifact_api_digest") == candidate_artifact_digest,
            f"{label} candidate artifact digest mismatch",
        )
        _require(
            payload.get("precompute_artifact_id") == precompute_artifact_id,
            f"{label} precompute artifact id mismatch",
        )
        _require(
            payload.get("precompute_artifact_api_digest") == precompute_artifact_digest,
            f"{label} precompute artifact digest mismatch",
        )
        for field in (
            "final_test_accessed",
            "production_eligible",
            "live_trading_authorized",
            "merge_authorized",
        ):
            _require(payload.get(field) is False, f"{label} boundary mismatch: {field}")

    invalid = _string_list(
        strict.get("invalid_termination_evidence"),
        field="strict invalid termination evidence",
    )
    strict_new = _string_list(
        strict.get("new_termination_violations"),
        field="strict new termination violations",
    )
    _require(
        strict.get("termination_evidence_valid") is (not invalid),
        "strict termination validity flag mismatch",
    )
    _require(
        strict.get("candidate_retrained_during_interpretation") is False,
        "strict report says candidate retrained",
    )
    _require(
        strict.get("economic_values_interpreted") is False,
        "strict precheck unexpectedly interpreted economics",
    )
    _require(
        decision.get("strict_precheck_content_digest") == strict.get("content_digest"),
        "decision/strict digest mismatch",
    )
    _require(
        decision.get("strict_termination_evidence_valid") is (not invalid),
        "decision strict validity mismatch",
    )
    _require(
        decision.get("strict_new_termination_present") is bool(strict_new),
        "decision strict new-termination flag mismatch",
    )
    _require(
        decision.get("candidate_retrained_during_interpretation") is False,
        "decision says candidate retrained",
    )
    _require(
        decision.get("operational_eligibility_established") is False,
        "decision unexpectedly establishes operational eligibility",
    )

    raw_decision = decision.get("decision")
    _require(
        isinstance(raw_decision, str) and raw_decision in VALID_DECISIONS,
        "final development decision malformed",
    )
    resolved_decision = cast(str, raw_decision)
    v2_path = producer_root / "v2" / "result.json"
    v2_digest: str | None = None

    if invalid:
        _require(resolved_decision == INVALID, "invalid strict evidence was not INVALID")
        _require(
            decision.get("economic_values_interpreted") is False,
            "invalid strict evidence interpreted economics",
        )
        _require(
            decision.get("v2_result_content_digest") is None,
            "invalid strict evidence unexpectedly has v2 digest",
        )
        _require(not v2_path.exists(), "invalid strict evidence unexpectedly has v2 result")
    else:
        _require(v2_path.is_file(), "valid strict evidence missing v2 result")
        v2 = _load_canonical(v2_path)
        v2_digest = cast(str, v2["content_digest"])
        _require(v2.get("issue_number") == ISSUE_NUMBER, "v2 issue drift")
        _require(
            v2.get("interpretation_run_id") == interpretation_run_id,
            "v2 interpretation run mismatch",
        )
        _require(
            v2.get("candidate_artifact_id") == candidate_artifact_id,
            "v2 candidate artifact id mismatch",
        )
        _require(
            v2.get("candidate_artifact_api_digest") == candidate_artifact_digest,
            "v2 candidate artifact digest mismatch",
        )
        _require(
            v2.get("precompute_artifact_id") == precompute_artifact_id,
            "v2 precompute artifact id mismatch",
        )
        _require(
            v2.get("precompute_artifact_api_digest") == precompute_artifact_digest,
            "v2 precompute artifact digest mismatch",
        )
        _require(
            v2.get("candidate_authority_content_digest") == candidate.get("content_digest"),
            "v2 candidate authority mismatch",
        )
        _require(
            v2.get("candidate_evidence_fingerprint")
            == candidate.get("candidate_evidence_fingerprint"),
            "v2 candidate evidence fingerprint mismatch",
        )
        _require(
            v2.get("precompute_authority_content_digest") == precompute.get("content_digest"),
            "v2 precompute authority mismatch",
        )
        _require(
            v2.get("termination_validity_violations") == [],
            "v2 termination validity disagrees with strict precheck",
        )
        v2_new = _string_list(
            v2.get("new_termination_violations"),
            field="v2 new termination violations",
        )
        _require(
            bool(v2_new) == bool(strict_new),
            "strict/v2 new-termination classification disagreement",
        )

        comparison = _mapping(v2.get("comparison"), field="v2 comparison")
        comparison_digest = comparison.get("analysis_digest")
        _require(isinstance(comparison_digest, str), "comparison analysis digest missing")
        comparison_unsigned = dict(comparison)
        comparison_unsigned.pop("analysis_digest", None)
        _require(
            content_digest(comparison_unsigned) == comparison_digest,
            "comparison analysis digest mismatch",
        )
        _require(
            v2.get("comparison_digest") == comparison_digest,
            "v2 comparison digest binding mismatch",
        )
        _require(
            v2.get("comparison_schema") == comparison.get("schema_version"),
            "v2 comparison schema binding mismatch",
        )

        decision_gate = _mapping(v2.get("decision_gate"), field="v2 decision gate")
        _require(
            decision_gate.get("decision") == v2.get("decision"),
            "v2 decision gate mismatch",
        )
        _require(v2.get("decision") == resolved_decision, "v2/final decision mismatch")
        _require(
            decision.get("v2_result_content_digest") == v2.get("content_digest"),
            "decision/v2 digest mismatch",
        )
        for field, expected in (
            ("candidate_retrained_during_interpretation", False),
            ("economic_values_interpreted", True),
            ("final_test_accessed", False),
            ("operational_eligibility_established", False),
            ("production_eligible", False),
            ("live_trading_authorized", False),
            ("merge_authorized", False),
            ("development_acceptance_establishes_profitability", False),
        ):
            _require(v2.get(field) is expected, f"v2 boundary mismatch: {field}")
        _require(
            decision.get("economic_values_interpreted") is True,
            "valid interpretation decision missing economics-interpreted flag",
        )
        _validate_absolute_diagnostic(v2)

    audit: dict[str, object] = {
        "schema_version": AUDIT_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "interpretation_run_id": interpretation_run_id,
        "candidate_execution_run_id": candidate_execution_run_id,
        "candidate_artifact_id": candidate_artifact_id,
        "candidate_artifact_api_digest": candidate_artifact_digest,
        "precompute_artifact_id": precompute_artifact_id,
        "precompute_artifact_api_digest": precompute_artifact_digest,
        "producer_result_artifact_id": producer_result_artifact_id,
        "producer_result_artifact_api_digest": producer_result_artifact_digest,
        "fresh_result_artifact_id": fresh_result_artifact_id,
        "fresh_result_artifact_api_digest": fresh_result_artifact_digest,
        "result_files": list(result_files),
        "result_tree_byte_identical": True,
        "strict_precheck_content_digest": strict["content_digest"],
        "decision_content_digest": decision["content_digest"],
        "v2_result_content_digest": v2_digest,
        "candidate_authority_content_digest": candidate["content_digest"],
        "recovery_binding_content_digest": recovery_binding["content_digest"],
        "recovery_authority_content_digest": recovery_authority["content_digest"],
        "precompute_authority_content_digest": precompute["content_digest"],
        "decision": resolved_decision,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
    }
    audit["content_digest"] = content_digest(audit)
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "audit.json").write_bytes(canonical_json_bytes(audit))
    return audit


def self_check() -> None:
    _require(AUDIT_SCHEMA == "issue610_interpretation_final_audit_v1", "audit schema drift")
    _require(
        VALID_DECISIONS == frozenset({ACCEPT_CANDIDATE, KEEP_BASELINE, INVALID}),
        "decision roster drift",
    )
    _require(tuple(SEEDS) == (0, 1, 2, 3, 4), "seed roster drift")
    _require(
        tuple(SYMBOLS) == ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT"),
        "symbol roster drift",
    )
    print("ISSUE610_REAL_INTERPRETATION_AUDIT_V1_SELF_CHECK=PASS")


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("self-check")
    audit = sub.add_parser("audit")
    audit.add_argument("--producer-root", type=Path, required=True)
    audit.add_argument("--fresh-root", type=Path, required=True)
    audit.add_argument("--candidate-root", type=Path, required=True)
    audit.add_argument("--precompute", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--interpretation-run-id", type=int, required=True)
    audit.add_argument("--candidate-execution-run-id", type=int, required=True)
    audit.add_argument("--candidate-artifact-id", type=int, required=True)
    audit.add_argument("--candidate-artifact-digest", required=True)
    audit.add_argument("--precompute-artifact-id", type=int, required=True)
    audit.add_argument("--precompute-artifact-digest", required=True)
    audit.add_argument("--producer-result-artifact-id", type=int, required=True)
    audit.add_argument("--producer-result-artifact-digest", required=True)
    audit.add_argument("--fresh-result-artifact-id", type=int, required=True)
    audit.add_argument("--fresh-result-artifact-digest", required=True)
    args = parser.parse_args()

    if args.command == "self-check":
        self_check()
        return
    if args.command == "audit":
        audit_interpretation(
            producer_root=args.producer_root,
            fresh_root=args.fresh_root,
            candidate_root=args.candidate_root,
            precompute_path=args.precompute,
            output_root=args.output_root,
            interpretation_run_id=args.interpretation_run_id,
            candidate_execution_run_id=args.candidate_execution_run_id,
            candidate_artifact_id=args.candidate_artifact_id,
            candidate_artifact_digest=args.candidate_artifact_digest,
            precompute_artifact_id=args.precompute_artifact_id,
            precompute_artifact_digest=args.precompute_artifact_digest,
            producer_result_artifact_id=args.producer_result_artifact_id,
            producer_result_artifact_digest=args.producer_result_artifact_digest,
            fresh_result_artifact_id=args.fresh_result_artifact_id,
            fresh_result_artifact_digest=args.fresh_result_artifact_digest,
        )
        print("ISSUE610_REAL_INTERPRETATION_AUDIT_V1=PASS")
        print("DECISION_REVEALED=false")
        print("FINAL_TEST_ACCESSED=false")
        print("PRODUCTION_ELIGIBLE=false")
        print("LIVE_TRADING_AUTHORIZED=false")
        print("MERGE_AUTHORIZED=false")
        return
    raise AssertionError(args.command)


if __name__ == "__main__":
    main()
