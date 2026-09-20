from dataclasses import replace

import numpy as np
import pytest

from tests.evaluation.test_shared_cash_replay import _market
from trade_rl.evaluation import directional
from trade_rl.strategies.controls import ConstantIntentStrategy
from trade_rl.strategies.position_intent import PositionIntent


def test_terminal_exit_is_a_real_next_open_fill_and_includes_both_fees() -> None:

    dataset = _market(np.full((8, 1), 100.0))
    dataset = replace(dataset, taker_fee_rate=np.full((8, 1), 0.001))
    assert hasattr(directional, "evaluate_directional_arm")
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.LONG),
        start_index=0,
        stop_index=7,
    )
    assert result["terminal_flat"]
    assert result["metrics"]["total_return"] < 0
    assert result["metrics"]["n_trades"] >= 2
    assert result["metrics"]["total_cost"] > 1.9


def test_delayed_orders_still_close_before_end_of_evidence() -> None:
    dataset = _market(np.full((9, 1), 100.0))
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.LONG),
        start_index=0,
        stop_index=8,
        latency_bars=1,
    )
    assert result["terminal_flat"]
    assert result["metrics"]["n_trades"] == 2


def test_cash_is_not_a_profitable_candidate_and_no_period_can_be_missing() -> None:
    dataset = _market(np.full((8, 1), 100.0))
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.FLAT),
        start_index=0,
        stop_index=7,
    )
    assert not result["qualified"]
    assert result["metrics"]["n_periods"] == 7
    assert result["returns"] == [0.0] * 7


def test_recovered_intrabar_drawdown_still_fails_twenty_percent_budget() -> None:

    values = np.full((8, 5), 100.0)
    values[2:] = 110.0
    dataset = _market(values)
    opens = dataset.open.copy()
    opens[2] = 50.0
    dataset = replace(dataset, open=opens, low=np.minimum(opens, values))
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.LONG),
        start_index=0,
        stop_index=7,
    )
    assert result["ledger_max_drawdown"] >= 0.25
    assert not result["qualified"]


def test_dataset_borrow_cost_is_not_disabled_by_zero_execution_overlay() -> None:

    dataset = _market(np.full((8, 1), 100.0))
    dataset = replace(dataset, borrow_rate=np.full((8, 1), 0.365))
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.SHORT),
        start_index=0,
        stop_index=7,
    )
    assert result["metrics"]["borrow_cost"] > 0


def test_no_liquidity_for_final_exit_retains_position_and_rejects_result() -> None:

    dataset = _market(np.linspace(100.0, 110.0, 8).reshape(-1, 1))
    volume = dataset.volume.copy()
    volume[6:] = 0
    dataset = replace(dataset, volume=volume)
    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.LONG),
        start_index=0,
        stop_index=7,
    )
    assert not result["terminal_flat"]
    assert result["terminal_quantities"][0] > 0
    assert not result["qualified"]

def test_year_returns_use_interval_end_timestamp() -> None:
    dataset = _market(np.full((5, 1), 100.0))
    timestamps = np.asarray(
        [
            "2025-12-31T22:00:00",
            "2025-12-31T23:00:00",
            "2026-01-01T00:00:00",
            "2026-01-01T01:00:00",
            "2026-01-01T02:00:00",
        ],
        dtype="datetime64[ns]",
    )
    cash_rate = np.zeros(5)
    cash_rate[1] = -0.876
    cash_rate[2] = 1.752
    dataset = replace(
        dataset,
        timestamps=timestamps,
        cash_rate=cash_rate,
        identity_payload_json=None,
    ).with_content_identity()

    result = directional.evaluate_directional_arm(
        dataset,
        lambda: ConstantIntentStrategy(PositionIntent.FLAT),
        start_index=0,
        stop_index=4,
    )

    assert result["returns"][0] == pytest.approx(-0.0001)
    assert result["returns"][1] == pytest.approx(0.0002)
    assert result["year_returns"]["2025"] == pytest.approx(-0.0001)
    assert result["year_returns"]["2026"] == pytest.approx(0.0002)

