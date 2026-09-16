from __future__ import annotations

import json
from collections.abc import Mapping
from typing import cast

import pytest

from tools.tmp_issue610_exact_once_orchestrator import SYMBOLS
from tools.tmp_issue610_real_interpretation_audit_v1 import (
    _validate_absolute_diagnostic,
)
from trade_rl.artifacts.canonical import canonical_json_bytes


def _synthetic_v2() -> dict[str, object]:
    medians = {
        "BTCUSDT": 0.01,
        "ETHUSDT": -0.02,
        "BNBUSDT": 0.03,
        "XRPUSDT": 0.0,
        "ADAUSDT": 0.02,
    }
    values = list(medians.values())
    return {
        "absolute_candidate_profitability_diagnostic": {
            "schema_version": "issue610_absolute_candidate_ppo_diagnostic_v1",
            "gates_development_decision": False,
            "by_symbol": {
                symbol: {"median_candidate_total_return": medians[symbol]}
                for symbol in SYMBOLS
            },
            "cross_symbol": {
                "symbol_count": len(SYMBOLS),
                "positive_symbol_count": sum(value > 0.0 for value in values),
                "negative_symbol_count": sum(value < 0.0 for value in values),
                "zero_symbol_count": sum(value == 0.0 for value in values),
                "median_candidate_total_return": 0.01,
                "worst_candidate_total_return": -0.02,
                "best_candidate_total_return": 0.03,
            },
        }
    }


def _canonical_round_trip(payload: Mapping[str, object]) -> dict[str, object]:
    decoded = json.loads(canonical_json_bytes(payload))
    assert isinstance(decoded, dict)
    return cast(dict[str, object], decoded)


def test_absolute_diagnostic_accepts_exact_symbols_after_canonical_round_trip() -> None:
    canonical = _canonical_round_trip(_synthetic_v2())
    diagnostic = cast(
        Mapping[str, object], canonical["absolute_candidate_profitability_diagnostic"]
    )
    by_symbol = cast(Mapping[str, object], diagnostic["by_symbol"])
    assert tuple(by_symbol) != SYMBOLS
    assert set(by_symbol) == set(SYMBOLS)

    _validate_absolute_diagnostic(canonical)


def test_absolute_diagnostic_still_rejects_extra_symbol() -> None:
    payload = _synthetic_v2()
    diagnostic = cast(
        dict[str, object], payload["absolute_candidate_profitability_diagnostic"]
    )
    by_symbol = cast(dict[str, object], diagnostic["by_symbol"])
    by_symbol["DOGEUSDT"] = {"median_candidate_total_return": 0.04}
    canonical = _canonical_round_trip(payload)

    with pytest.raises(RuntimeError, match="absolute diagnostic symbol roster drift"):
        _validate_absolute_diagnostic(canonical)
