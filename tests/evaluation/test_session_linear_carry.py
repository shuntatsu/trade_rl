"""Declared signed session carry, with independent exact gap/open/close ledger."""

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
from trade_rl.data.contracts import MarketCalendarKind, VolumeUnit
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.replay import run_shared_portfolio_replay
from trade_rl.evaluation.series import SessionElapsedReturnSeries
from trade_rl.evaluation.session_returns import (
    load_session_elapsed_return_payload,
    run_session_elapsed_portfolio_replay,
    session_elapsed_return_payload,
)
from trade_rl.simulation import MarketExecutor, runtime_execution_policy_digest
from trade_rl.simulation.execution import ExecutionRuleStress


def _session_values():
    values = _values()
    times = np.datetime64("2026-01-02T16:00", "ns") + np.array(
        [0, 65, 66, 67, 89, 154]
    ) * np.timedelta64(1, "h")
    values.update(
        timestamps=times,
        available_at=times[:, None].copy(),
        calendar_kind=MarketCalendarKind.SESSION,
        native_clock=None,
        nominal_bar_hours=1.0,
        periods_per_year=1638,
    )
    return values


def _dataset(values=None):
    return MarketDataset(
        **(_session_values() if values is None else values)
    ).with_content_identity({"fixture": "declared-session-linear-carry"})


def _bounds(dataset):
    return tuple(int(x) for x in dataset.timestamps.view(np.int64))


def _run(dataset, policy, **changes):
    return run_session_elapsed_portfolio_replay(
        dataset,
        policy,
        **(
            {
                "start_index": 0,
                "stop_index": 5,
                "expected_boundaries_ns": _bounds(dataset),
                "expected_nominal_bar_seconds": 3600,
                "initial_capital": 1000.0,
                "execution_cost": _cost(),
                "native_session_linear_carry": True,
            }
            | changes
        ),
    )


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("cash_rate", [0, Fraction(365, 1000)])
def test_signed_session_literal_gap_and_processing_ledger(tmp_path, sign, cash_rate):
    values = _session_values()
    values["cash_rate"] = np.full(6, float(cash_rate))
    values["open"][3, 0] = 120.0
    values["high"][3, 0] = 120.0
    # Explicit zero-rate close event.
    values["funding_due"][4, 0] = True
    values["funding_event_count"][4, 0] = 1
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
    cash = previous_equity = Fraction(1000)
    for index, (row, hours) in enumerate(zip(rows, (65, 1, 1, 22, 65), strict=True), 1):
        held_before = index > 1
        held_after = index < 5
        quantity = sign * Fraction(5, 4) if held_after else Fraction(0)
        gap_hours = hours - 1
        gap_interest = cash * cash_rate * gap_hours / 8760
        gap_borrow = Fraction(250 * gap_hours, 24000) if held_before and sign < 0 else 0
        fee = Fraction(1, 4) if index in (1, 5) else Fraction(0)
        if index == 1:
            cash -= sign * 250
        if index == 5:
            cash += sign * 250
        cash -= fee
        processing_interest = cash * cash_rate / 8760
        open_price = 120 if index == 3 else 100
        processing_borrow = (
            Fraction(5, 2) * open_price / 24000 if held_after and sign < 0 else 0
        )
        funding = {1: -sign * Fraction(11, 4), 2: sign * Fraction(9, 8)}.get(
            index, Fraction(0)
        )
        interest = gap_interest + processing_interest
        borrow = gap_borrow + processing_borrow
        cash += interest + funding - borrow
        equity = cash + (sign * 250 if held_after else 0)
        assert row.next_index == index
        assert tuple(Fraction(q) for q in row.exact_quantities_after) == (quantity,)
        assert row.interval_cost == pytest.approx(float(fee))
        assert row.interval_funding == pytest.approx(float(funding))
        assert row.interval_cash_interest == pytest.approx(float(interest))
        assert row.interval_borrow_cost == pytest.approx(float(borrow))
        assert row.cash_after == pytest.approx(float(cash))
        assert row.portfolio_value_after == pytest.approx(float(equity))
        assert row.interval_net_return == pytest.approx(
            float(equity / previous_equity - 1)
        )
        if index in (1, 2, 4, 5):
            (event,) = row.funding_events
            assert event.signed_quantities == (float(quantity),)
            assert event.funding_amount == pytest.approx(float(funding))
        else:
            assert not row.funding_events
        previous_equity = equity
    assert result.book.exact_quantities == (0,)
    assert result.book.portfolio_value == pytest.approx(float(cash))
    assert isinstance(result.returns, SessionElapsedReturnSeries)
    assert result.returns.boundaries_ns == _bounds(restored)
    assert result.returns.exact_elapsed_years == Fraction(154, 8760)
    payload = session_elapsed_return_payload(restored, result.returns, start_index=0)
    assert load_session_elapsed_return_payload(restored, payload) == result.returns


def test_session_signed_mode_has_separate_digest_and_actual_returns():
    dataset = _dataset()
    cost = _cost()
    executor = MarketExecutor(dataset, cost, native_session_linear_carry=True)
    assert executor.native_session_linear_carry
    assert not executor.strict_spot_cash
    base = runtime_execution_policy_digest(cost, funding_price_rate_present=True)
    expected = content_digest(
        {
            "schema_version": "native_session_linear_carry_execution_v1",
            "base_policy_digest": base,
            "funding_semantics": "single_declared_close_event_post_open_fill",
            "borrow_semantics": "previous_close_gap_and_post_open_short_mark_actual_act365f",
        }
    )
    assert expected == executor.execution_policy_digest
    assert expected == runtime_execution_policy_digest(
        cost, funding_price_rate_present=True, native_session_linear_carry=True
    )
    direct = run_shared_portfolio_replay(
        dataset,
        _HoldAfterEntry(-1),
        start_index=0,
        stop_index=5,
        initial_capital=1000,
        execution_cost=cost,
        settle_terminal_position=True,
        capture_ledger_evidence=True,
        native_session_linear_carry=True,
    )
    paired = _run(dataset, _HoldAfterEntry(-1))
    assert direct.returns == paired.returns
    assert direct.ledger_evidence == paired.ledger_evidence


@pytest.mark.parametrize(
    "mutation",
    [
        "multiple",
        "borrow",
        "late",
        "wrong_units",
        "unverified",
        "negative_cash",
        "bad_nominal",
        "implicit_nominal",
    ],
)
def test_session_signed_refuses_unsupported_rows_before_actor(mutation):
    values = _session_values()
    if mutation == "multiple":
        values["funding_event_count"][5, 0] = 2
    elif mutation == "borrow":
        values["borrow_available"][5, 0] = False
    elif mutation == "late":
        values["available_at"][5, 0] += np.timedelta64(1, "ns")
    elif mutation == "wrong_units":
        values["volume_units"] = (VolumeUnit.BASE_ASSET,)
        values["contract_multipliers"] = np.ones(1)
    elif mutation == "negative_cash":
        values["cash_rate"] = np.zeros(6)
        values["cash_rate"][5] = -0.01
    elif mutation == "bad_nominal":
        values["nominal_bar_hours"] = 2.0
    elif mutation == "implicit_nominal":
        values["nominal_bar_hours"] = None
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
        {"native_session_linear_carry": 1},
        {"native_session_linear_carry": False},
        {"expected_nominal_bar_seconds": 7200},
        {"expected_boundaries_ns": (0, 1)},
        {"execution_cost": _cost(order_latency_bars=1)},
        {"execution_cost": _cost(max_leverage=2.0)},
    ],
)
def test_session_signed_refuses_ambiguous_replay_declaration(changes):
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError):
        _run(_dataset(), policy, **changes)
    assert not policy.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"strict_spot_cash": True},
        {"native_fixed_linear_carry": True},
        {"native_calendar_month_linear_carry": True},
        {"funding_price_rate_present": False},
        {"market_order_profile_digest": "a" * 64},
        {"rule_stress": ExecutionRuleStress(tick_size_factor=2.0)},
    ],
)
def test_session_signed_digest_refuses_mixed_modes(changes):
    with pytest.raises(ValueError):
        runtime_execution_policy_digest(
            _cost(),
            **(
                {
                    "funding_price_rate_present": True,
                    "native_session_linear_carry": True,
                }
                | changes
            ),
        )


def test_session_signed_executor_rechecks_mutable_configuration():
    executor = MarketExecutor(_dataset(), _cost(), native_session_linear_carry=True)
    original = executor.execution_policy_digest
    executor.cost = _cost(order_latency_bars=1)
    with pytest.raises(ValueError, match="zero latency"):
        _ = executor.execution_policy_digest
    executor.cost = _cost()
    assert executor.execution_policy_digest == original
    executor.dataset = _dataset()
    with pytest.raises(ValueError, match="fresh executor"):
        _ = executor.execution_policy_digest


@pytest.mark.parametrize(
    "changes",
    [
        {"nominal_bar_hours": 2.0},
        {"nominal_bar_hours": 0.123456},
    ],
)
def test_direct_signed_session_checks_actual_duration_before_actor(changes):
    dataset = _dataset(_session_values() | changes)
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError, match="nominal"):
        run_shared_portfolio_replay(
            dataset,
            policy,
            start_index=0,
            stop_index=5,
            initial_capital=1000,
            execution_cost=_cost(),
            settle_terminal_position=True,
            native_session_linear_carry=True,
        )
    assert not policy.calls


@pytest.mark.parametrize(
    "changes",
    [
        {"settle_terminal_position": False},
        {"actor_decision_indices": (0, 2)},
        {"strict_spot_cash": True},
        {"native_fixed_linear_carry": True},
        {"native_calendar_month_linear_carry": True},
    ],
)
def test_direct_signed_session_requires_exclusive_settled_native_mode(changes):
    policy = _HoldAfterEntry(-1)
    with pytest.raises(ValueError):
        run_shared_portfolio_replay(
            _dataset(),
            policy,
            **(
                {
                    "start_index": 0,
                    "stop_index": 5,
                    "initial_capital": 1000,
                    "execution_cost": _cost(),
                    "settle_terminal_position": True,
                    "native_session_linear_carry": True,
                }
                | changes
            ),
        )
    assert not policy.calls


def test_session_signed_primitive_refuses_continuous_grid():
    from tests.evaluation.test_native_fixed_linear_carry import _dataset as fixed

    with pytest.raises(ValueError, match="verified session"):
        MarketExecutor(fixed(), _cost(), native_session_linear_carry=True)
