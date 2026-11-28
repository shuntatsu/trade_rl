"""Independent literal ledger for declared UTC-month close-event carry."""

from fractions import Fraction

import numpy as np
import pytest

from tests.evaluation.test_native_fixed_linear_carry import (
    _cost,
    _HoldAfterEntry,
    _values,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts.codec import load_dataset_files, write_market_dataset_files
from trade_rl.data.contracts import VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.data.native_clock import NativeBarClock, NativeClockKind
from trade_rl.evaluation.native_calendar_returns import (
    load_native_calendar_month_return_payload,
    native_calendar_month_return_payload,
)
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.simulation import (
    MarketExecutor,
    runtime_execution_policy_digest,
)
from trade_rl.simulation.execution import ExecutionRuleStress


def _month_values(year=2024):
    values = _values()
    closes = np.arange(f"{year}-01", f"{year}-07", dtype="datetime64[M]").astype(
        "datetime64[ns]"
    )
    values.update(
        timestamps=closes,
        available_at=closes[:, None].copy(),
        native_clock=NativeBarClock(NativeClockKind.UTC_MONTH),
        periods_per_year=Fraction(6 * 365, 183 if year == 2024 else 182),
    )
    return values


def _dataset(values=None):
    return MarketDataset(
        **(_month_values() if values is None else values)
    ).with_content_identity()


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
                "native_calendar_month": True,
                "native_calendar_month_linear_carry": True,
            }
            | changes
        ),
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("year,february_days", [(2023, 28), (2024, 29)])
@pytest.mark.parametrize("cash_rate", [0, Fraction(365, 10000)])
def test_month_close_event_carry_literal_calendar_ledger(
    tmp_path, sign, year, february_days, cash_rate
):
    values = _month_values(year)
    values["cash_rate"] = np.full(6, float(cash_rate))
    # Explicit zero-rate funding event, still an observed event.
    values["funding_due"][4, 0] = True
    values["funding_event_count"][4, 0] = 1
    values["open"][3, 0] = 120.0
    values["high"][3, 0] = 120.0
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
    cash = previous = Fraction(1000)
    days = (31, february_days, 31, 30, 31)
    for index, (row, elapsed_days) in enumerate(zip(rows, days, strict=True), 1):
        held = index < 5
        quantity = sign * Fraction(5, 4) if held else Fraction(0)
        fee = Fraction(1, 4) if index in (1, 5) else Fraction(0)
        if index == 1:
            cash -= sign * 250
        if index == 5:
            cash += sign * 250
        funding = {1: -sign * Fraction(11, 4), 2: sign * Fraction(9, 8)}.get(
            index, Fraction(0)
        )
        borrow = Fraction(elapsed_days, 4) if held and sign < 0 else Fraction(0)
        if index == 3:
            borrow *= Fraction(6, 5)
        cash -= fee
        interest = cash * cash_rate * elapsed_days / 365
        cash += funding + interest - borrow
        equity = cash + (sign * 250 if held else 0)
        assert row.next_index == index
        assert tuple(Fraction(q) for q in row.exact_quantities_after) == (quantity,)
        assert row.interval_cost == pytest.approx(float(fee))
        assert row.interval_funding == pytest.approx(float(funding))
        assert row.interval_borrow_cost == pytest.approx(float(borrow))
        assert row.interval_cash_interest == pytest.approx(float(interest))
        assert row.cash_after == pytest.approx(float(cash))
        assert row.portfolio_value_after == pytest.approx(float(equity))
        assert row.interval_net_return == pytest.approx(float(equity / previous - 1))
        if index in (1, 2, 4, 5):
            (event,) = row.funding_events
            assert event.signed_quantities == (float(quantity),)
            assert event.funding_amount == pytest.approx(float(funding))
        else:
            assert not row.funding_events
        previous = equity
    assert result.book.exact_quantities == (0,)
    assert result.book.portfolio_value == pytest.approx(float(cash))
    assert result.returns.exact_elapsed_years == Fraction(sum(days), 365)
    payload = native_calendar_month_return_payload(
        restored, result.returns, start_index=0
    )
    assert (
        load_native_calendar_month_return_payload(restored, payload) == result.returns
    )


def test_month_signed_mode_has_distinct_digest_and_explicit_opt_in():
    dataset = _dataset()
    with pytest.raises(ValueError):
        MarketExecutor(dataset, _cost(), native_calendar_month=True)
    executor = MarketExecutor(
        dataset,
        _cost(),
        native_calendar_month=True,
        native_calendar_month_linear_carry=True,
    )
    assert not executor.strict_spot_cash
    assert not executor.native_fixed_linear_carry
    assert executor.native_calendar_month_linear_carry
    monthly = runtime_execution_policy_digest(
        _cost(),
        funding_price_rate_present=True,
        native_calendar_month_linear_carry=True,
    )
    assert monthly == executor.execution_policy_digest
    assert monthly != runtime_execution_policy_digest(
        _cost(), funding_price_rate_present=True, native_fixed_linear_carry=True
    )
    base = runtime_execution_policy_digest(_cost(), funding_price_rate_present=True)
    assert monthly != base
    semantics = {
        "base_policy_digest": base,
        "funding_semantics": "single_declared_close_event_post_open_fill",
        "borrow_semantics": "post_open_short_mark_actual_act365f",
    }
    assert monthly == content_digest(
        semantics
        | {"schema_version": "native_calendar_month_linear_carry_execution_v1"}
    )
    assert runtime_execution_policy_digest(
        _cost(), funding_price_rate_present=True, native_fixed_linear_carry=True
    ) == content_digest(
        semantics | {"schema_version": "native_fixed_linear_carry_execution_v1"}
    )


@pytest.mark.parametrize(
    "mutation",
    [
        "multiple",
        "borrow",
        "late",
        "wrong_units",
        "negative_cash",
        "unverified",
        "missing_event_price",
        "negative_event_price",
        "dividend",
    ],
)
def test_month_signed_preflight_refuses_incomplete_contract(mutation):
    values = _month_values()
    if mutation == "multiple":
        values["funding_event_count"][5, 0] = 2
    elif mutation == "borrow":
        values["borrow_available"][5, 0] = False
    elif mutation == "late":
        values["available_at"][5, 0] += np.timedelta64(1, "ns")
    elif mutation == "wrong_units":
        values["volume_units"] = (VolumeUnit.BASE_ASSET,)
        values["contract_multipliers"] = np.ones(1)
    elif mutation == "missing_event_price":
        values["funding_price_rate"][5, 0] = 0.0
    elif mutation == "negative_event_price":
        values["funding_price_rate"][5, 0] *= -1
    elif mutation == "dividend":
        values["dividend"] = np.zeros((6, 1))
        values["dividend"][5, 0] = 1.0
    elif mutation == "negative_cash":
        values["cash_rate"] = np.zeros(6)
        values["cash_rate"][5] = -0.01
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
        {"native_calendar_month": False},
        {"native_calendar_month_linear_carry": 1},
        {"native_fixed_linear_carry": True},
        {"strict_spot_cash": True},
        {"settle_terminal_position": False},
        {"execution_cost": _cost(order_latency_bars=1)},
        {"execution_cost": _cost(max_leverage=2.0)},
        {"actor_decision_indices": (0, 2)},
    ],
)
def test_month_signed_replay_rejects_ambiguous_or_unsupported_mode(changes):
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError):
        _run(_dataset(), policy, **changes)
    assert not policy.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"strict_spot_cash": True},
        {"native_fixed_linear_carry": True},
        {"funding_price_rate_present": False},
        {"market_order_profile_digest": "a" * 64},
        {"rule_stress": ExecutionRuleStress(name="lot2", lot_size_factor=2.0)},
    ],
)
def test_month_signed_digest_refuses_unsupported_configuration(changes):
    with pytest.raises(ValueError):
        runtime_execution_policy_digest(
            _cost(),
            **(
                {
                    "funding_price_rate_present": True,
                    "native_calendar_month_linear_carry": True,
                }
                | changes
            ),
        )


def test_month_signed_executor_rechecks_mutated_configuration():
    executor = MarketExecutor(
        _dataset(),
        _cost(),
        native_calendar_month=True,
        native_calendar_month_linear_carry=True,
    )
    original = executor.execution_policy_digest
    executor.rule_stress = ExecutionRuleStress(name="lot2", lot_size_factor=2.0)
    with pytest.raises(ValueError, match="rule stress"):
        _ = executor.execution_policy_digest
    executor.rule_stress = ExecutionRuleStress()
    assert executor.execution_policy_digest == original
    executor.cost = _cost(order_latency_bars=1)
    with pytest.raises(ValueError, match="zero latency"):
        _ = executor.execution_policy_digest


def test_month_signed_mode_does_not_extend_fixed_or_learning_admission():
    from tests.evaluation.test_native_fixed_linear_carry import (
        _dataset as fixed_dataset,
    )
    from trade_rl.simulation.execution import (
        _preflight_native_learning_execution_window,
    )

    with pytest.raises(ValueError):
        MarketExecutor(
            fixed_dataset(), _cost(), native_calendar_month_linear_carry=True
        )
    with pytest.raises(ValueError):
        MarketExecutor(
            _dataset(),
            _cost(),
            native_calendar_month=True,
            native_fixed_linear_carry=True,
        )
    with pytest.raises(ValueError):
        _preflight_native_learning_execution_window(
            _dataset(), _cost(), start_index=0, stop_index=5, native_calendar_month=True
        )
