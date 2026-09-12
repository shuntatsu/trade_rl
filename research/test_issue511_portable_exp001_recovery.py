from __future__ import annotations

from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest

MODULE = "research.issue511_portable_exp001_recovery"


def _module():
    spec = find_spec(MODULE)
    assert spec is not None, "recovery helper is not implemented"
    return import_module(MODULE)


def _comparison(*, positive: int = 4, median_excess: float = 0.25):
    cross_symbol = MappingProxyType(
        {
            "mean_reversion": MappingProxyType(
                {
                    "positive_symbol_count": positive,
                    "median_excess_total_return": median_excess,
                }
            )
        }
    )
    factor_effect = MappingProxyType({"cross_symbol": cross_symbol})
    return SimpleNamespace(factor_effect=factor_effect)


def _independent(*, positive: int = 4, median_excess: float = 0.25):
    return {
        "mean_reversion_formal_inputs": {
            "positive_factor_effect_symbol_count": positive,
            "median_excess_total_return": median_excess,
        }
    }


def test_crosscheck_accepts_recursively_frozen_mapping() -> None:
    module = _module()
    module.crosscheck_persisted_comparison(_comparison(), _independent())


def test_crosscheck_rejects_missing_cross_symbol_mapping() -> None:
    module = _module()
    comparison = SimpleNamespace(factor_effect=MappingProxyType({}))
    with pytest.raises(RuntimeError, match="cross_symbol malformed"):
        module.crosscheck_persisted_comparison(comparison, _independent())


def test_crosscheck_rejects_persisted_value_drift() -> None:
    module = _module()
    with pytest.raises(RuntimeError, match="positive count"):
        module.crosscheck_persisted_comparison(_comparison(positive=3), _independent())


def test_recovery_helper_cannot_reexecute_candidate() -> None:
    module = _module()
    source = Path(module.__file__).read_text(encoding="utf-8")
    assert "run_experiment(" not in source
    assert "execute_evidence_set(" not in source
    assert "compare_experiment(" not in source
