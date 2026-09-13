from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest

from trade_rl.evaluation.experiments import ExperimentDecisionKind

MODULE = "research.issue529_portable_exp004_postverify"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0004 independent postverifier is not implemented"
    return import_module(MODULE)


def test_independent_decision_accepts_only_full_gate() -> None:
    module = _module()
    assert (
        module.independent_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=5,
        )
        is ExperimentDecisionKind.ACCEPT_CANDIDATE
    )


def test_independent_decision_keep_and_inconclusive_boundaries() -> None:
    module = _module()
    assert (
        module.independent_decision(
            positive_effects=2,
            median_excess=0.1,
            candidate_positive_returns=5,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.independent_decision(
            positive_effects=5,
            median_excess=0.0,
            candidate_positive_returns=5,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.independent_decision(
            positive_effects=5,
            median_excess=0.1,
            candidate_positive_returns=3,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.independent_decision(
            positive_effects=3,
            median_excess=0.1,
            candidate_positive_returns=4,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def _valid_index() -> dict[str, object]:
    return {
        "schema_version": "canonical_m2_portable_exp004_result_index_v1",
        "issue_number": 529,
        "verification_status": "CONTROLLED",
        "changed_paths": [["feature_indices"], ["feature_names"]],
        "formal_target_strategy": "lightgbm24",
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "study_frozen": False,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }


def test_validate_result_index_rejects_rerun_or_authorization_drift() -> None:
    module = _module()
    module.validate_result_index_safety(_valid_index())
    bad = _valid_index()
    bad["candidate_execution_count"] = 2
    with pytest.raises(RuntimeError, match="execution count"):
        module.validate_result_index_safety(bad)
    bad = _valid_index()
    bad["candidate_rerun"] = True
    with pytest.raises(RuntimeError, match="rerun"):
        module.validate_result_index_safety(bad)
    bad = _valid_index()
    bad["interpretation_authorized"] = True
    with pytest.raises(RuntimeError, match="sealed"):
        module.validate_result_index_safety(bad)


def test_validate_result_index_rejects_claims_and_delta_drift() -> None:
    module = _module()
    bad = _valid_index()
    bad["changed_paths"] = [["feature_names"]]
    with pytest.raises(RuntimeError, match="changed paths"):
        module.validate_result_index_safety(bad)
    for key in (
        "profitability_claimed",
        "winner_claimed",
        "final_test_authorized",
        "production_authorized",
    ):
        bad = _valid_index()
        bad[key] = True
        with pytest.raises(RuntimeError, match="unsupported claim"):
            module.validate_result_index_safety(bad)
