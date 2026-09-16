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
                ]
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


def test_termination_gate_rejects_new_count_or_new_reason() -> None:
    baseline = _loaded(count=1, reasons=["risk_limit"])
    equal = _loaded(count=1, reasons=["risk_limit"])
    assert termination_violations(baseline, equal) == ()

    higher_count = _loaded(count=2, reasons=["risk_limit", "risk_limit"])
    assert any(
        "new PPO termination count" in item
        for item in termination_violations(baseline, higher_count)
    )

    new_reason = _loaded(count=1, reasons=["economic_floor"])
    assert any(
        "new PPO termination reason" in item
        for item in termination_violations(baseline, new_reason)
    )


def test_termination_gate_allows_fewer_existing_terminations() -> None:
    baseline = _loaded(count=2, reasons=["risk_limit", "risk_limit"])
    candidate = _loaded(count=1, reasons=["risk_limit"])
    assert termination_violations(baseline, candidate) == ()
