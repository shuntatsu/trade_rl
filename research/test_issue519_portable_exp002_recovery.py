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
