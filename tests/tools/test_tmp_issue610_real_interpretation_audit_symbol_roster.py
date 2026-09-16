from __future__ import annotations

from statistics import median

import pytest

from tools.tmp_issue610_real_interpretation_audit_v1 import (
    SYMBOLS,
    _validate_absolute_diagnostic,
)


def _v2_with_symbols(symbols: tuple[str, ...]) -> dict[str, object]:
    medians = {symbol: 0.01 * (index + 1) for index, symbol in enumerate(SYMBOLS)}
    ordered = {symbol: medians[symbol] for symbol in symbols if symbol in medians}
    by_symbol: dict[str, object] = {
        symbol: {"median_candidate_total_return": value}
        for symbol, value in ordered.items()
    }
    values = list(ordered.values())
    diagnostic: dict[str, object] = {
        "schema_version": "issue610_absolute_candidate_ppo_diagnostic_v1",
        "gates_development_decision": False,
        "by_symbol": by_symbol,
        "cross_symbol": {
            "symbol_count": len(values),
            "positive_symbol_count": sum(value > 0.0 for value in values),
            "negative_symbol_count": sum(value < 0.0 for value in values),
            "zero_symbol_count": sum(value == 0.0 for value in values),
            "median_candidate_total_return": float(median(values)),
            "worst_candidate_total_return": min(values),
            "best_candidate_total_return": max(values),
        },
    }
    return {"absolute_candidate_profitability_diagnostic": diagnostic}


def test_canonical_sorted_symbol_keys_are_accepted() -> None:
    canonical_key_order = tuple(sorted(SYMBOLS))
    _validate_absolute_diagnostic(_v2_with_symbols(canonical_key_order))


def test_missing_symbol_is_rejected() -> None:
    payload = _v2_with_symbols(tuple(sorted(SYMBOLS))[:-1])
    with pytest.raises(RuntimeError, match="absolute diagnostic symbol roster drift"):
        _validate_absolute_diagnostic(payload)


def test_extra_symbol_is_rejected() -> None:
    payload = _v2_with_symbols(tuple(sorted(SYMBOLS)))
    diagnostic = payload["absolute_candidate_profitability_diagnostic"]
    assert isinstance(diagnostic, dict)
    by_symbol = diagnostic["by_symbol"]
    assert isinstance(by_symbol, dict)
    by_symbol["ZZZUSDT"] = {"median_candidate_total_return": 0.0}
    with pytest.raises(RuntimeError, match="absolute diagnostic symbol roster drift"):
        _validate_absolute_diagnostic(payload)
