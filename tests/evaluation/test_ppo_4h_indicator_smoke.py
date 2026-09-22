from __future__ import annotations

from types import SimpleNamespace

import pytest

import trade_rl.evaluation.ppo_4h_indicator_smoke as smoke


EXPECTED_FEATURES = (
    "4h__macd_line_12_26",
    "4h__macd_signal_12_26_9",
    "4h__macd_histogram_12_26_9",
    "4h__atr_pct_14bar",
    "4h__plus_di_14bar",
    "4h__minus_di_14bar",
    "4h__ichimoku_tenkan_distance_9bar",
    "4h__ichimoku_kijun_distance_26bar",
    "4h__ichimoku_cloud_position_9_26_52",
    "4h__ichimoku_cloud_thickness_9_26_52",
)


def test_smoke_feature_roster_is_exactly_four_hour_indicator_set() -> None:
    assert smoke.FEATURE_NAMES == EXPECTED_FEATURES
    assert smoke.SEED == 0
    assert smoke.REQUESTED_TIMESTEPS == 100_000
    assert smoke.OBSERVATION_WIDTH == 32
    assert smoke.SCENARIOS == {
        "base": {"cost_multiplier": 1.0, "latency_bars": 0},
        "cost_2x": {"cost_multiplier": 2.0, "latency_bars": 0},
        "latency_1": {"cost_multiplier": 1.0, "latency_bars": 1},
    }


def test_feature_indices_require_exact_names_and_preserve_order() -> None:
    names = ("unused", *EXPECTED_FEATURES, "other")
    dataset = SimpleNamespace(feature_names=names)

    assert smoke.resolve_feature_indices(dataset) == tuple(range(1, 11))

    missing = SimpleNamespace(feature_names=names[:-2] + ("other",))
    with pytest.raises(ValueError, match="4h indicator"):
        smoke.resolve_feature_indices(missing)


def _cell(
    total_return: float,
    *,
    y2023: float = 0.01,
    y2024: float = 0.01,
    drawdown: float = 0.10,
    flat: bool = True,
    terminated: bool = False,
) -> dict[str, object]:
    return {
        "total_return": total_return,
        "year_returns": {"2023": y2023, "2024": y2024},
        "ledger_max_drawdown": drawdown,
        "terminal_flat": flat,
        "termination_reasons": ["stop"] if terminated else [],
    }


def _result(
    base: tuple[float, ...],
    *,
    cost: tuple[float, ...] | None = None,
    latency: tuple[float, ...] | None = None,
) -> dict[str, dict[str, dict[str, object]]]:
    cost = base if cost is None else cost
    latency = base if latency is None else latency
    symbols = ("BTCUSDT", "ETHUSDT", "BNBUSDT", "XRPUSDT", "ADAUSDT")
    return {
        symbol: {
            "base": _cell(base[index]),
            "cost_2x": _cell(cost[index]),
            "latency_1": _cell(latency[index]),
        }
        for index, symbol in enumerate(symbols)
    }


def test_promotion_requires_robust_positive_cross_symbol_smoke() -> None:
    passing = _result(
        (0.08, 0.05, 0.02, 0.01, -0.01),
        cost=(0.04, 0.03, 0.01, 0.005, -0.02),
        latency=(0.03, 0.02, 0.01, 0.002, -0.02),
    )
    decision = smoke.promotion_decision(passing)
    assert decision["decision"] == "PROMOTE_TO_FULL_5_SEED_STUDY"
    assert decision["positive_base_symbols"] == 4

    too_few = _result((0.08, 0.05, -0.01, -0.02, -0.03))
    assert smoke.promotion_decision(too_few)["decision"] == "STOP_AFTER_SMOKE"

    weak_stress = _result(
        (0.08, 0.05, 0.02, 0.01, -0.01),
        cost=(-0.03, -0.02, -0.01, 0.001, 0.002),
    )
    assert smoke.promotion_decision(weak_stress)["decision"] == "STOP_AFTER_SMOKE"


@pytest.mark.parametrize(
    "mutation",
    (
        {"drawdown": 0.2000001},
        {"flat": False},
        {"terminated": True},
    ),
)
def test_any_hard_guard_failure_blocks_promotion(mutation: dict[str, object]) -> None:
    result = _result((0.08, 0.05, 0.02, 0.01, -0.01))
    result["BTCUSDT"]["base"] = _cell(0.08, **mutation)

    decision = smoke.promotion_decision(result)

    assert decision["decision"] == "STOP_AFTER_SMOKE"
    assert decision["hard_guards_pass"] is False
