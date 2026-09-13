from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from types import MappingProxyType, SimpleNamespace

import pytest

MODULE = "research.issue519_portable_exp002_recovery"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "Experiment 0002 recovery helper is not implemented"
    return import_module(MODULE)


def _comparison(*, positive: int = 4, median_excess: float = 0.05):
    mean_reversion = MappingProxyType(
        {
            "positive_symbol_count": positive,
            "median_excess_total_return": median_excess,
        }
    )
    cross_symbol = MappingProxyType({"mean_reversion": mean_reversion})
    factor_effect = MappingProxyType({"cross_symbol": cross_symbol})
    return SimpleNamespace(factor_effect=factor_effect)


def _independent(*, positive: int = 4, median_excess: float = 0.05):
    return {
        "mean_reversion_formal_inputs": {
            "positive_factor_effect_symbol_count": positive,
            "median_excess_total_return": median_excess,
        }
    }


def _execution_raw(*, ppo_excess: float = 0.0):
    ppo = {
        str(seed): {
            symbol: {
                "baseline_total_return": 0.1,
                "candidate_total_return": 0.1 + ppo_excess,
                "excess_total_return": ppo_excess,
            }
            for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
        }
        for seed in range(5)
    }
    deterministic = {"mean_reversion": {"by_symbol": {}, "aggregate": {}}}
    formal = {
        "positive_factor_effect_symbol_count": 4,
        "median_excess_total_return": 0.05,
        "candidate_positive_total_return_symbol_count": 5,
        "candidate_median_turnover": 400.0,
        "frozen_baseline_median_turnover": 858.3114067468092,
    }
    costs = {
        "aggregate_total_cost_across_seed_runs": 10.0,
        "trading_observations": 175,
        "positive_cost_observations": 175,
        "cash_observations": 25,
    }
    return {
        "baseline_fingerprint": "baseline-fp",
        "candidate_fingerprint": "candidate-fp",
        "unaffected_raw_return_equality_checks": 150,
        "deterministic_effects": deterministic,
        "ppo_by_seed_symbol": ppo,
        "ppo_cross_symbol_metrics_used_for_decision": False,
        "mean_reversion_formal_inputs": formal,
        "candidate_cost_semantics": costs,
        "formal_decision": "ACCEPT_CANDIDATE",
    }


def test_crosscheck_accepts_recursively_frozen_mapping_proxy() -> None:
    module = _module()
    module.crosscheck_persisted_comparison(_comparison(), _independent())


def test_crosscheck_rejects_positive_symbol_count_mismatch() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="positive count"):
        module.crosscheck_persisted_comparison(
            _comparison(positive=4),
            _independent(positive=3),
        )


def test_crosscheck_rejects_median_mismatch() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="median"):
        module.crosscheck_persisted_comparison(
            _comparison(median_excess=0.05),
            _independent(median_excess=0.04),
        )


def test_normalize_independent_matches_fresh_postverify_schema() -> None:
    module = _module()
    source = _execution_raw()
    normalized = module.normalize_independent_for_postverify(source)
    assert normalized == {
        "baseline_evidence_fingerprint": "baseline-fp",
        "candidate_evidence_fingerprint": "candidate-fp",
        "unaffected_raw_return_equality_checks": 150,
        "ppo_exact_zero_effect_checks": 25,
        "deterministic_effects": source["deterministic_effects"],
        "mean_reversion_formal_inputs": source["mean_reversion_formal_inputs"],
        "candidate_cost_semantics": source["candidate_cost_semantics"],
        "formal_decision": "ACCEPT_CANDIDATE",
    }


def test_normalize_independent_rejects_nonzero_ppo_effect() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="PPO"):
        module.normalize_independent_for_postverify(_execution_raw(ppo_excess=0.01))
