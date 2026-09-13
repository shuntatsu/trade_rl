from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec

import pytest

from trade_rl.evaluation.experiments import ExperimentDecisionKind

MODULE = "research.issue522_portable_exp003_postverify"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 independent postverifier is not implemented"
    return import_module(MODULE)


def test_independent_decision_accepts_only_full_gate() -> None:
    module = _module()
    assert (
        module.independent_decision(
            positive_effects=4,
            median_excess=1e-9,
            candidate_positive_returns=5,
            candidate_median_turnover=800.0,
        )
        is ExperimentDecisionKind.ACCEPT_CANDIDATE
    )


def test_independent_decision_keep_and_inconclusive_boundaries() -> None:
    module = _module()
    assert (
        module.independent_decision(
            positive_effects=4,
            median_excess=0.1,
            candidate_positive_returns=3,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.independent_decision(
            positive_effects=4,
            median_excess=0.1,
            candidate_positive_returns=5,
            candidate_median_turnover=module.FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def test_validate_result_index_rejects_execution_count_or_authorization_drift() -> None:
    module = _module()
    base = {
        "schema_version": "canonical_m2_portable_exp003_result_index_v1",
        "issue_number": 522,
        "verification_status": "CONTROLLED",
        "changed_paths": [["signal_index"], ["signal_name"]],
        "formal_target_strategy": "trend",
        "candidate_execution_count": 1,
        "candidate_rerun": False,
        "study_frozen": False,
        "interpretation_authorized": False,
        "profitability_claimed": False,
        "winner_claimed": False,
        "final_test_authorized": False,
        "production_authorized": False,
    }
    module.validate_result_index_safety_flags(base)
    bad_count = dict(base, candidate_execution_count=2)
    with pytest.raises(RuntimeError, match="candidate_execution_count"):
        module.validate_result_index_safety_flags(bad_count)
    bad_auth = dict(base, interpretation_authorized=True)
    with pytest.raises(RuntimeError, match="interpretation_authorized"):
        module.validate_result_index_safety_flags(bad_auth)
