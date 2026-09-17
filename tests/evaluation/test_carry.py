from dataclasses import replace

import numpy as np
import pytest

from tests.integrations.test_binance_carry import series
from trade_rl.evaluation import carry
from trade_rl.integrations.binance.carry import assemble_carry_dataset


def dataset():
    spot, perp = series(perpetual=False), series(perpetual=True)
    return assemble_carry_dataset(
        {"BTCUSDT": (spot, replace(perp, funding_rate=perp.funding_rate * 10))}
    )


def test_equal_price_hedge_net_pnl_funding_and_real_terminal_exit() -> None:
    result = carry.replay_carry(
        dataset(), start_index=5, stop_index=8, initial_capital=4000
    )
    assert result["total_cost"] == pytest.approx(5.0)
    assert result["funding_pnl"] == pytest.approx(1.0)
    assert result["equity"][-1] == pytest.approx(3996.0)
    assert result["terminal_flat"] and result["execution_valid"]
    assert result["qualified"] is False
    np.testing.assert_array_equal(result["quantities"], [[10, -10], [10, -10], [0, 0]])
    assert np.prod(1 + np.asarray(result["returns"])) * 4000 == pytest.approx(3996)


@pytest.mark.parametrize(("start", "stop"), [(7, 10), (4, 7)])
def test_entry_after_or_exit_before_funding_boundary_does_not_collect(
    start, stop
) -> None:
    result = carry.replay_carry(
        dataset(), start_index=start, stop_index=stop, initial_capital=4000
    )
    assert result["funding_pnl"] == 0
    assert result["terminal_flat"]


def test_matched_quantity_hold_does_not_rebalance_with_prices() -> None:
    original = dataset()
    prices = original.close.copy()
    prices[7:] = 110
    moving = replace(
        original,
        dataset_id="0" * 64,
        identity_payload_json=None,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
        volume=original.volume * 10,
    ).with_content_identity()
    result = carry.replay_carry(
        moving, start_index=5, stop_index=9, initial_capital=4000
    )
    np.testing.assert_array_equal(result["quantities"][:3], [[10, -10]] * 3)
    assert result["terminal_flat"]


def test_unmatched_fill_stops_and_preserves_unfillable_exit() -> None:
    original = dataset()
    volume = original.volume.copy()
    volume[5, 0] = 50000
    volume[6:] = 0
    thin = replace(
        original, dataset_id="0" * 64, identity_payload_json=None, volume=volume
    ).with_content_identity()
    result = carry.replay_carry(thin, start_index=5, stop_index=9, initial_capital=4000)
    assert result["stop_reason"] == "unmatched_hedge"
    assert result["terminal_flat"] is False and result["qualified"] is False
    np.testing.assert_array_equal(result["terminal_quantities"], [5, -10])


def test_collateral_excludes_spot_value_and_synthetic_short_proceeds() -> None:
    assert (
        carry.futures_collateral(
            4000, np.array([10.0, -10.0]), np.array([100.0, 100.0])
        )
        == 3000
    )
    assert (
        carry.futures_collateral(
            4000, np.array([10.0, -10.0]), np.array([300.0, 300.0])
        )
        == 1000
    )


def test_intrabar_margin_breach_cannot_be_rescued_by_later_funding() -> None:
    original = dataset()
    high = original.high.copy()
    high[7, 1] = 500
    funding = original.funding_rate.copy()
    funding[7, 1] = 9.0
    stressed = replace(
        original,
        dataset_id="0" * 64,
        identity_payload_json=None,
        high=high,
        funding_rate=funding,
    ).with_content_identity()
    result = carry.replay_carry(
        stressed, start_index=5, stop_index=9, initial_capital=4000
    )
    assert result["execution_valid"] is False and result["qualified"] is False
    assert "intrabar_margin_breach" in result["invalid_reasons"]
    assert result["collateral"][1]["intrabar_before_funding"] < 0
    assert result["collateral"][1]["close_after_funding"] > 0


def test_initial_entry_delay_leaves_terminal_exit_timing_unchanged() -> None:
    result = carry.replay_carry(
        dataset(), start_index=5, stop_index=9, initial_capital=4000, entry_delay_bars=1
    )
    np.testing.assert_array_equal(result["quantities"][0], [0, 0])
    assert result["quantities"][1] == [10, -10]
    assert result["terminal_flat"]


def test_processing_halt_blocks_spot_fill_without_foreseeing_the_hedge_failure() -> (
    None
):
    original = dataset()
    tradable = original.tradable.copy()
    tradable[6, 0] = False
    halted = replace(
        original, dataset_id="0" * 64, identity_payload_json=None, tradable=tradable
    ).with_content_identity()
    result = carry.replay_carry(
        halted, start_index=5, stop_index=10, initial_capital=4000
    )
    np.testing.assert_array_equal(result["quantities"][0], [0, -10])
    assert result["stop_reason"] == "unmatched_hedge"
    assert result["qualified"] is False


def test_canonical_lot_target_does_not_create_tiny_followup_orders() -> None:
    original = dataset()
    prices = np.full_like(original.close, 100000.0)
    expensive = replace(
        original,
        dataset_id="0" * 64,
        identity_payload_json=None,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
        volume=original.volume * 1000,
    ).with_content_identity()
    result = carry.replay_carry(
        expensive, start_index=5, stop_index=9, initial_capital=3960
    )
    assert result["terminal_flat"]
    np.testing.assert_array_equal(result["quantities"][:3], [[0.009, -0.009]] * 3)


def test_matched_partial_orders_finish_the_frozen_target_without_resubmission() -> None:
    original = dataset()
    volume = original.volume.copy()
    volume[5] = 50000
    partial = replace(
        original, dataset_id="0" * 64, identity_payload_json=None, volume=volume
    ).with_content_identity()
    result = carry.replay_carry(
        partial, start_index=5, stop_index=9, initial_capital=4000
    )
    np.testing.assert_array_equal(
        result["quantities"], [[5, -5], [10, -10], [10, -10], [0, 0]]
    )
    submitted = [
        event for event in result["order_events"] if event["event_type"] == "submitted"
    ]
    assert len(submitted) == 4  # Two entries, two exits; partials keep their order IDs.
    assert result["stop_reason"] is None and result["terminal_flat"]


def test_canonical_termination_is_invalid_and_does_not_claim_real_flattening() -> None:
    original = dataset()
    funding = original.funding_rate.copy()
    funding[7, 1] = -10
    bankrupt = replace(
        original, dataset_id="0" * 64, identity_payload_json=None, funding_rate=funding
    ).with_content_identity()
    result = carry.replay_carry(
        bankrupt, start_index=5, stop_index=9, initial_capital=4000
    )
    assert result["execution_valid"] is False and result["qualified"] is False
    assert result["terminal_flat"] is False and result["complete"] is False
    assert result["stop_reason"] == "canonical_termination"
    np.testing.assert_array_equal(result["terminal_quantities"], [10, -10])


def test_open_margin_breach_before_terminal_exit_cannot_be_hidden_by_flattening() -> (
    None
):
    original = dataset()
    funding = original.funding_rate.copy()
    funding[7, 1] = 0.02
    prices = original.close.copy()
    prices[8:] = 300
    gap = replace(
        original,
        dataset_id="0" * 64,
        identity_payload_json=None,
        open=prices,
        close=prices,
        high=prices,
        low=prices,
        mark_price=prices,
        volume=original.volume * 10,
        funding_rate=funding,
    ).with_content_identity()
    result = carry.replay_carry(gap, start_index=5, stop_index=8, initial_capital=4000)
    assert result["total_return"] > 0
    assert result["execution_valid"] is False and result["qualified"] is False
    assert "open_margin_breach" in result["invalid_reasons"]
