from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

from tools.tmp_issue610_exact_once_orchestrator import (
    BASELINE_EVIDENCE_FINGERPRINT,
    DATASET_ARTIFACT_DIGEST,
    DATASET_ID,
    EVALUATION_PROTOCOL_DIGEST,
    EXPECTED_SEMANTIC_CHANGED_FIELDS,
    HELPER_SHA,
    IMPLEMENTATION_SHA,
    IMPLEMENTATION_TREE,
    ISSUE_NUMBER,
    PRECOMPUTE_RUN_ID,
    PRECOMPUTE_SCHEMA,
    SEEDS,
    SOURCE_STUDY_DIGEST,
    SYMBOLS,
    UNAFFECTED_STRATEGIES,
    termination_violations,
    validate_exactly_once_invocation,
    validate_precompute_authority,
)


def _precompute() -> dict[str, object]:
    return {
        "schema_version": PRECOMPUTE_SCHEMA,
        "issue_number": ISSUE_NUMBER,
        "precompute_run_id": PRECOMPUTE_RUN_ID,
        "implementation_git_sha": IMPLEMENTATION_SHA,
        "implementation_tree_sha": IMPLEMENTATION_TREE,
        "helper_git_sha": HELPER_SHA,
        "dataset_id": DATASET_ID,
        "dataset_artifact_digest": DATASET_ARTIFACT_DIGEST,
        "source_study_digest": SOURCE_STUDY_DIGEST,
        "baseline_evidence_fingerprint": BASELINE_EVIDENCE_FINGERPRINT,
        "evaluation_protocol_digest": EVALUATION_PROTOCOL_DIGEST,
        "prior_issue610_economic_artifacts": 0,
        "precompute_inputs_validated": True,
        "economic_publisher_requires_run_number_one": True,
        "economic_publisher_requires_attempt_one": True,
        "baseline_retrained": False,
        "candidate_training_performed": False,
        "economic_values_interpreted": False,
        "final_test_accessed": False,
        "operational_eligibility_established": False,
        "production_eligible": False,
        "live_trading_authorized": False,
        "merge_authorized": False,
        "expected_semantic_changed_paths": [
            list(path) for path in EXPECTED_SEMANTIC_CHANGED_FIELDS
        ],
        "unaffected_strategy_names": list(UNAFFECTED_STRATEGIES),
        "implementation_digest": "a" * 64,
        "runtime_environment_digest": "b" * 64,
        "candidate_carrier_study_digest": "c" * 64,
        "candidate_resolved_config_digest": "d" * 64,
        "candidate_requested_config_digest": "e" * 64,
        "content_digest": "f" * 64,
    }


def _loaded(*, count: int, reasons: list[str]) -> SimpleNamespace:
    runs = {}
    for seed in SEEDS:
        runs[seed] = SimpleNamespace(
            summary={
                "symbols": list(SYMBOLS),
                "by_symbol": [
                    {
                        "symbol": symbol,
                        "strategies": [
                            {
                                "name": "ppo",
                                "metrics": {"termination_count": count},
                                "diagnostics": {"termination_reasons": list(reasons)},
                            }
                        ],
                    }
                    for symbol in SYMBOLS
                ],
            }
        )
    return SimpleNamespace(runs=runs)


def test_precompute_authority_accepts_only_result_blind_exact_binding() -> None:
    authority = _precompute()
    assert validate_precompute_authority(authority) == ()

    for field, bad_value in (
        ("candidate_training_performed", True),
        ("economic_values_interpreted", True),
        ("final_test_accessed", True),
        ("prior_issue610_economic_artifacts", 1),
        ("implementation_git_sha", "0" * 40),
        ("helper_git_sha", "0" * 40),
    ):
        forged = deepcopy(authority)
        forged[field] = bad_value
        assert validate_precompute_authority(forged)


def test_exactly_once_runtime_gate_requires_first_run_and_attempt() -> None:
    assert validate_exactly_once_invocation(run_number=1, run_attempt=1) == ()
    assert validate_exactly_once_invocation(run_number=2, run_attempt=1) == (
        "economic publisher must use GitHub run number 1",
    )
    assert validate_exactly_once_invocation(run_number=1, run_attempt=2) == (
        "economic publisher must use GitHub run attempt 1",
    )
    assert validate_exactly_once_invocation(run_number=2, run_attempt=3) == (
        "economic publisher must use GitHub run number 1",
        "economic publisher must use GitHub run attempt 1",
    )


def test_exactly_once_runtime_gate_rejects_non_integer_or_non_positive_values() -> None:
    assert validate_exactly_once_invocation(run_number=True, run_attempt=1)
    assert validate_exactly_once_invocation(run_number=0, run_attempt=1)
    assert validate_exactly_once_invocation(run_number=1, run_attempt=False)
    assert validate_exactly_once_invocation(run_number=1, run_attempt=0)


def test_termination_gate_delegates_to_verified_reason_oracle() -> None:
    baseline = _loaded(count=1, reasons=["risk_limit"])
    equal = _loaded(count=1, reasons=["risk_limit"])
    assert termination_violations(baseline, equal) == ()

    new_reason = _loaded(count=1, reasons=["economic_floor"])
    violations = termination_violations(baseline, new_reason)
    assert any(
        item == "new PPO termination: seed=0 BTCUSDT reason=economic_floor"
        for item in violations
    )


def test_termination_gate_fails_closed_on_count_reason_inconsistency() -> None:
    baseline = _loaded(count=0, reasons=[])
    malformed = _loaded(count=1, reasons=[])
    violations = termination_violations(baseline, malformed)
    assert any(
        item == "candidate PPO termination evidence malformed: seed=0 BTCUSDT"
        for item in violations
    )


def test_termination_gate_allows_removed_existing_termination() -> None:
    baseline = _loaded(count=1, reasons=["risk_limit"])
    candidate = _loaded(count=0, reasons=[])
    assert termination_violations(baseline, candidate) == ()
