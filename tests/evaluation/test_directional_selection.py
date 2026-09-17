from copy import deepcopy
from typing import Any

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.directional_candidates import ARMS
from trade_rl.evaluation.directional_selection import select_development_candidates


def _result(value: float = 0.1) -> dict[str, Any]:
    result: dict[str, Any] = {
        "metrics": {"total_return": value, "turnover_total": 10.0},
        "year_returns": {"2023": value / 2, "2024": value / 2},
        "ledger_max_drawdown": 0.1,
        "terminal_flat": True,
        "termination_reasons": [],
        "returns": [0.0] * 17544,
        "start_index": 1,
        "stop_index": 17545,
        "symbol_index": None,
        "cost_multiplier": 1.0,
        "latency_bars": 0,
    }
    result["stress"] = [
        {**deepcopy(result), "cost_multiplier": 2.0},
        {**deepcopy(result), "latency_bars": 1},
    ]
    result["by_symbol"] = {
        symbol: {} for symbol in ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    }
    return result


def _roster() -> dict[str, dict[str, Any]]:
    return {arm: _result(-0.1) for arm in ARMS}


def test_controls_cannot_win_and_missing_seed_is_incomplete() -> None:
    results = _roster()
    results["constant_long"] = _result(2.0)
    assert select_development_candidates(results)["winner"] is None
    del results["ppo4"]
    with pytest.raises(ValueError, match="complete roster"):
        select_development_candidates(results)


def test_ppo_is_a_family_with_four_passing_seeds_and_five_seed_medians() -> None:
    results = _roster()
    for index in range(4):
        results[f"ppo{index}"] = _result(0.2 + index * 0.01)
    report = select_development_candidates(results)
    assert report["winner"] == "ppo"
    assert report["families"]["ppo"]["total_return"] == pytest.approx(0.21)
    results["ppo3"]["stress"][0]["metrics"]["total_return"] = -0.01
    assert select_development_candidates(results)["winner"] is None


def test_stress_and_yearly_losses_fail_even_if_arm_claims_qualified() -> None:
    results = _roster()
    results["trend"] = _result()
    results["trend"]["qualified"] = True
    results["trend"]["stress"][1]["ledger_max_drawdown"] = 0.21
    assert select_development_candidates(results)["winner"] is None
    results["trend"] = _result()
    results["trend"]["year_returns"]["2024"] = -0.01
    assert select_development_candidates(results)["winner"] is None


def test_rank_uses_return_then_turnover_then_fixed_complexity() -> None:
    results = _roster()
    results["trend"] = _result()
    results["ridge24"] = _result()
    assert select_development_candidates(results)["winner"] == "trend"
    results["ridge24"]["metrics"]["turnover_total"] = 9.0
    assert select_development_candidates(results)["winner"] == "ridge24"


def test_missing_stress_or_symbol_diagnostics_cannot_qualify() -> None:
    results = _roster()
    results["channel_breakout"] = _result()
    results["channel_breakout"]["by_symbol"].pop("ADAUSDT")
    assert select_development_candidates(results)["winner"] is None
    results["channel_breakout"] = _result()
    results["channel_breakout"]["stress"] = []
    assert select_development_candidates(results)["winner"] is None


def test_early_termination_retains_missing_year_and_can_be_published() -> None:
    results = _roster()
    del results["ppo4"]["year_returns"]["2024"]
    results["ppo4"]["termination_reasons"] = ["insolvent"]
    report = select_development_candidates(results)
    assert report["families"]["ppo"]["qualified"] is False
    assert report["families"]["ppo"]["year_returns"]["2024"] is None
    assert canonical_json_bytes(report)
