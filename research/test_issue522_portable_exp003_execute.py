from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import MappingProxyType, SimpleNamespace

import pytest

from trade_rl.evaluation.experiments import ExperimentDecisionKind

MODULE = "research.issue522_portable_exp003_execute"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0003 execution helper is not implemented"
    return import_module(MODULE)


def test_formal_decision_accepts_only_full_preregistered_gate() -> None:
    module = _module()
    assert (
        module.formal_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=800.0,
        )
        is ExperimentDecisionKind.ACCEPT_CANDIDATE
    )


def test_formal_decision_equal_turnover_is_inconclusive() -> None:
    module = _module()
    assert (
        module.formal_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=module.FROZEN_BASELINE_TREND_MEDIAN_TURNOVER,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def test_formal_decision_keep_baseline_boundary() -> None:
    module = _module()
    assert (
        module.formal_decision(
            positive_effects=4,
            median_excess=0.01,
            candidate_positive_returns=3,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.formal_decision(
            positive_effects=2,
            median_excess=0.01,
            candidate_positive_returns=5,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )
    assert (
        module.formal_decision(
            positive_effects=5,
            median_excess=0.0,
            candidate_positive_returns=5,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.KEEP_BASELINE
    )


def test_formal_decision_middle_region_is_inconclusive() -> None:
    module = _module()
    assert (
        module.formal_decision(
            positive_effects=3,
            median_excess=0.01,
            candidate_positive_returns=4,
            candidate_median_turnover=100.0,
        )
        is ExperimentDecisionKind.INCONCLUSIVE
    )


def _comparison(*, positive: int = 4, median_excess: float = 0.05):
    trend = MappingProxyType(
        {
            "positive_symbol_count": positive,
            "median_excess_total_return": median_excess,
        }
    )
    cross_symbol = MappingProxyType({"trend": trend})
    factor_effect = MappingProxyType({"cross_symbol": cross_symbol})
    return SimpleNamespace(factor_effect=factor_effect)


def _independent(*, positive: int = 4, median_excess: float = 0.05):
    return {
        "trend_formal_inputs": {
            "positive_factor_effect_symbol_count": positive,
            "median_excess_total_return": median_excess,
        }
    }


def test_crosscheck_accepts_recursively_frozen_mapping_proxy() -> None:
    module = _module()
    module.crosscheck_persisted_comparison(_comparison(), _independent())


def test_crosscheck_rejects_mismatch() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="positive count"):
        module.crosscheck_persisted_comparison(
            _comparison(positive=4),
            _independent(positive=3),
        )
    with pytest.raises(RuntimeError, match="median"):
        module.crosscheck_persisted_comparison(
            _comparison(median_excess=0.05),
            _independent(median_excess=0.04),
        )
