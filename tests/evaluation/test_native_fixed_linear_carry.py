"""Declared close-event carry on complete native fixed grids; synthetic only."""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from tests.data.test_native_execution_dataset import native_values
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.native_fixed_returns import (
    load_native_fixed_return_payload,
    native_fixed_return_payload,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import (
    ExecutionCostConfig,
    MarketExecutor,
    runtime_execution_policy_digest,
)
from trade_rl.strategies.portfolio import PortfolioTargetProposal


def _values(days=7, *, weekly=True):
    values = native_values(days, weekly=weekly)
    prices = np.full((6, 1), 100.0)
    rates = np.zeros((6, 1))
    price_rates = np.zeros((6, 1))
    counts = np.zeros((6, 1), dtype=np.int32)
    rates[[1, 2, 5], 0] = (0.01, -0.005, 0.02)
    price_rates[[1, 2, 5], 0] = (1.1, -0.45, 2.4)
    counts[[1, 2, 5], 0] = 1
    values.update(
        open=prices.copy(),
        high=prices.copy(),
        low=prices.copy(),
        close=prices.copy(),
        volume_units=(VolumeUnit.CONTRACTS,),
        contract_multipliers=np.array([2.0]),
        borrow_rate=np.full((6, 1), 0.365),
        borrow_available=np.ones((6, 1), dtype=bool),
        funding_rate=rates,
        funding_price_rate=price_rates,
        funding_due=counts > 0,
        funding_event_count=counts,
    )
    return values


def _dataset(values=None):
    dataset = MarketDataset(**(_values() if values is None else values))
    return dataset.with_content_identity(
        {"fixture": "declared-native-fixed-linear-carry"}
    )


def _cost(**changes):
    return replace(
        ExecutionCostConfig.zero(),
        **(
            {
                "allow_short": True,
                "borrow_rate_multiplier": 1.0,
                "processing_bar_volume_capacity": False,
                "partial_fill_carry": False,
                "fee_rate": 0.001,
            }
            | changes
        ),
    )


class _HoldAfterEntry:
    def __init__(self, sign=1):
        self.sign = sign
        self.calls = []

    def decide_portfolio(self, snapshot):
        self.calls.append(snapshot.index)
        if snapshot.index == 0:
            return PortfolioTargetProposal((self.sign * 0.25,))
        return PortfolioTargetProposal(
            snapshot.account.weights, hold_current_quantity_symbols=(0,)
        )


def _run(dataset, policy, **changes):
    return run_shared_portfolio_replay(
        dataset,
        policy,
        **(
            {
                "start_index": 0,
                "stop_index": 5,
                "initial_capital": 1000.0,
                "execution_cost": _cost(),
                "settle_terminal_position": True,
                "capture_ledger_evidence": True,
                "native_fixed_linear_carry": True,
            }
            | changes
        ),
    )


@pytest.mark.parametrize("days,weekly", [(3, False), (7, False), (7, True)])
@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("borrow_open_price", [100.0, 120.0])
def test_declared_fixed_close_event_carry_literal_ledger(
    tmp_path, days, weekly, sign, borrow_open_price
):
    values = _values(days, weekly=weekly)
    values["open"][3, 0] = borrow_open_price
    values["high"][3, 0] = borrow_open_price
    dataset = _dataset(values)
    write_market_dataset_files(tmp_path, dataset)
    restored = load_dataset_files(tmp_path)
    assert restored.identity_verified and restored.dataset_id == dataset.dataset_id
    policy = _HoldAfterEntry(sign)
    result = _run(restored, policy)
    assert policy.calls == [0, 1, 2, 3]
    assert result.ledger_evidence is not None
    rows = result.ledger_evidence.intervals.intervals
    assert len(rows) == 5
    cash = Fraction(1000)
    previous = Fraction(1000)
    for index, row in enumerate(rows, 1):
        assert row.next_index == index
        held = index < 5
        quantity = sign * Fraction(5, 4) if held else Fraction(0)
        fee = Fraction(1, 4) if index in (1, 5) else Fraction(0)
        if index == 1:
            cash -= sign * 250
        if index == 5:
            cash += sign * 250
        funding = {
            1: -sign * Fraction(11, 4),
            2: sign * Fraction(9, 8),
        }.get(index, Fraction(0))
        borrow = Fraction(days, 4) if held and sign < 0 else Fraction(0)
        if index == 3:
            borrow *= Fraction(int(borrow_open_price), 100)
        cash += funding - fee - borrow
        equity = cash + (sign * 250 if held else 0)
        assert tuple(Fraction(q) for q in row.exact_quantities_after) == (quantity,)
        assert row.interval_cost == pytest.approx(float(fee))
        assert row.interval_funding == pytest.approx(float(funding))
        assert row.interval_borrow_cost == pytest.approx(float(borrow))
        assert row.cash_after == pytest.approx(float(cash))
        assert row.portfolio_value_after == pytest.approx(float(equity))
        assert row.interval_net_return == pytest.approx(float(equity / previous - 1))
        if index in (1, 2, 5):
            (event,) = row.funding_events
            assert event.processing_index == index
            assert event.signed_quantities == (float(quantity),)
            assert event.funding_amount == pytest.approx(float(funding))
            assert event.funding_price_rates == tuple(restored.funding_price_rate[index])
        else:
            assert not row.funding_events
        previous = equity
    assert result.book.exact_quantities == (0,)
    assert result.book.portfolio_value == pytest.approx(float(cash))
    assert result.returns.periods_per_year == Fraction(365, days)
    payload = native_fixed_return_payload(restored, result.returns, start_index=0)
    assert load_native_fixed_return_payload(restored, payload) == result.returns


def test_mode_identity_is_explicit_and_default_stays_spot():
    dataset = _dataset()
    cost = _cost()
    with pytest.raises(ValueError, match="native fixed execution requires"):
        MarketExecutor(dataset, cost)
    executor = MarketExecutor(dataset, cost, native_fixed_linear_carry=True)
    assert executor.native_fixed_linear_carry and not executor.strict_spot_cash
    expected = runtime_execution_policy_digest(
        cost, funding_price_rate_present=True, native_fixed_linear_carry=True
    )
    assert executor.execution_policy_digest == expected
    assert expected != runtime_execution_policy_digest(
        cost, funding_price_rate_present=True
    )
    spot_cost = replace(cost, allow_short=False)
    spot = MarketExecutor(dataset, spot_cost)
    assert spot.strict_spot_cash and not spot.native_fixed_linear_carry
    assert spot.execution_policy_digest == runtime_execution_policy_digest(
        spot_cost, funding_price_rate_present=True, strict_spot_cash=True
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "multiple",
        "missing_product",
        "negative_implied_price",
        "late_terminal",
        "borrow_unavailable",
        "corporate_action",
        "negative_cash",
        "spot_units",
        "unverified",
    ],
)
def test_refuse_invalid_complete_window_before_actor(mutation):
    values = _values()
    if mutation == "multiple":
        values["funding_event_count"][5, 0] = 2
    elif mutation == "missing_product":
        values["funding_price_rate"][5, 0] = 0.0
    elif mutation == "negative_implied_price":
        values["funding_price_rate"][5, 0] *= -1
    elif mutation == "late_terminal":
        values["available_at"][5, 0] += np.timedelta64(1, "ns")
    elif mutation == "borrow_unavailable":
        values["borrow_available"][5, 0] = False
    elif mutation == "corporate_action":
        values["dividend"] = np.zeros((6, 1))
        values["dividend"][5, 0] = 1.0
    elif mutation == "negative_cash":
        values["cash_rate"] = np.zeros(6)
        values["cash_rate"][5] = -0.01
    elif mutation == "spot_units":
        values["volume_units"] = (VolumeUnit.BASE_ASSET,)
        values["contract_multipliers"] = np.ones(1)
    dataset = MarketDataset(**values)
    if mutation != "unverified":
        dataset = dataset.with_content_identity()
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError):
        _run(dataset, policy)
    assert not policy.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"order_type": "limit"},
        {"max_leverage": 2.0},
        {"margin_mode": "isolated"},
        {"processing_bar_volume_capacity": True},
        {"partial_fill_carry": True},
        {"order_latency_bars": 1},
        {"slippage_std": 0.1},
        {"tail_slippage_probability": 0.1, "tail_slippage_multiplier": 2.0},
        {"collateral_haircut": 0.9},
    ],
)
def test_refuse_unsupported_native_signed_cost_before_actor(changes):
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError, match="native fixed linear carry"):
        _run(_dataset(), policy, execution_cost=_cost(**changes))
    assert not policy.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"settle_terminal_position": False},
        {"strict_spot_cash": True},
        {"native_fixed_linear_carry": 1},
    ],
)
def test_mode_refuses_ambiguous_replay_opt_in(changes):
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError):
        _run(_dataset(), policy, **changes)
    assert not policy.calls


def test_digest_rejects_conflicting_native_mode():
    with pytest.raises(ValueError, match="native fixed linear carry"):
        runtime_execution_policy_digest(
            _cost(),
            funding_price_rate_present=True,
            strict_spot_cash=True,
            native_fixed_linear_carry=True,
        )
    with pytest.raises(ValueError, match="native fixed linear carry"):
        runtime_execution_policy_digest(
            _cost(), funding_price_rate_present=False, native_fixed_linear_carry=True
        )
