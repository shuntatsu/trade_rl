"""R2-vs-null short-hour disagreement audit reads native ledger, not shadow P&L."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.evaluation.paired_short_exposure import audit_paired_short_hours


@dataclass(frozen=True)
class Interval:
    start_index: int
    next_index: int
    exact_quantities_before: tuple[str, ...]
    exact_quantities_after: tuple[str, ...]
    portfolio_value_before: float = 1000.0
    portfolio_value_after: float = 1000.0
    interval_cost: float = 0.0
    interval_funding: float = 0.0
    interval_borrow_cost: float = 0.0


def trace(states: tuple[str, ...]) -> tuple[Interval, ...]:
    return tuple(
        Interval(i, i + 1, (a,), (b,))
        for i, (a, b) in enumerate(zip(states, states[1:]))
    )


def clock(n: int) -> np.ndarray:
    return np.datetime64("2022-01-01T00:00", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )


def test_all_four_disjoint_groups_have_signed_hourly_price_labels() -> None:
    prices = np.array([100.0, 99.0, 98.0, 97.0, 98.0, 97.0])
    r2 = trace(("0", "-1", "-1", "0", "0", "0"))
    ctrl = trace(("0", "0", "-1", "-1", "0", "0"))
    result = audit_paired_short_hours(
        close=prices,
        timestamps=clock(6),
        r2_intervals=r2,
        control_intervals=ctrl,
        start_index=0,
        stop_index=5,
    )
    assert result.total_hour_count == 5
    assert result.r2_short_hour_count == result.control_short_hour_count == 2
    assert {key: g.count for key, g in result.groups.items()} == {
        "both": 1,
        "r2_only": 1,
        "control_only": 1,
        "neither": 2,
    }
    assert result.groups["r2_only"].mean_signed_market_pct == pytest.approx(
        100 * (1 - 98 / 99)
    )
    assert result.groups["control_only"].mean_signed_market_pct == pytest.approx(
        100 * (1 - 98 / 97)
    )


def test_bad_clock_gap_and_reversal_fail_closed() -> None:
    prices = np.array([100.0, 99.0, 98.0, 97.0, 98.0, 97.0])
    r2 = trace(("0", "-1", "-1", "0", "0", "0"))
    mutated = clock(6)
    mutated[2] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="regular"):
        audit_paired_short_hours(
            close=prices,
            timestamps=mutated,
            r2_intervals=r2,
            control_intervals=r2,
            start_index=0,
            stop_index=5,
        )
    bad = list(r2)
    bad[3] = Interval(5, 6, ("0",), ("0",))
    with pytest.raises(ValueError, match="interval|ledger"):
        audit_paired_short_hours(
            close=prices,
            timestamps=clock(6),
            r2_intervals=tuple(bad),
            control_intervals=r2,
            start_index=0,
            stop_index=5,
        )
    with pytest.raises(ValueError, match="reversal"):
        audit_paired_short_hours(
            close=np.array([100.0, 99.0, 98.0]),
            timestamps=clock(3),
            r2_intervals=trace(("1", "-1", "-1")),
            control_intervals=trace(("0", "0", "0")),
            start_index=0,
            stop_index=2,
        )


def test_later_prices_outside_study_cannot_change_past_results() -> None:
    prices = np.array([100.0, 99.0, 98.0, 97.0, 98.0, 97.0])
    r2 = trace(("0", "-1", "-1", "0", "0", "0"))
    ctrl = trace(("0", "0", "-1", "-1", "0", "0"))
    before = audit_paired_short_hours(
        close=prices,
        timestamps=clock(6),
        r2_intervals=r2[:3],
        control_intervals=ctrl[:3],
        start_index=0,
        stop_index=3,
    )
    after = prices.copy()
    after[4:] = float("nan")
    new = audit_paired_short_hours(
        close=after,
        timestamps=clock(6),
        r2_intervals=r2[:3],
        control_intervals=ctrl[:3],
        start_index=0,
        stop_index=3,
    )
    assert before == new


def test_empty_buckets_do_not_invent_zero_mean_alpha() -> None:
    prices = np.array([100.0, 100.0, 100.0])
    flat = trace(("0", "0", "0"))
    out = audit_paired_short_hours(
        close=prices,
        timestamps=clock(3),
        r2_intervals=flat,
        control_intervals=flat,
        start_index=0,
        stop_index=2,
    )
    assert out.groups["both"].count == 0
    assert out.groups["both"].mean_signed_market_pct is None


def test_native_account_has_real_short_masks_and_no_shadow_ledger() -> None:
    from tests.strategies.test_cloud_regime_short_gate import _context
    from tests.strategies.test_divergence_cloud_follow import _native_market, _signals
    from trade_rl.evaluation.replay import run_shared_portfolio_replay
    from trade_rl.strategies.cloud_regime_short_gate import (
        CloudRegimeShortGateConfig,
        CloudRegimeShortGatePolicy,
    )
    from trade_rl.strategies.short_null_control import (
        ShortNullControlConfig,
        ShortNullControlPolicy,
    )

    ds = _native_market()
    pack = _signals([1] * 25 + [-1] * 30)
    ctx = _context(pack, allowed=tuple(range(55)))
    r2 = run_shared_portfolio_replay(
        ds,
        CloudRegimeShortGatePolicy(
            symbol="BTCUSDT",
            signals=pack,
            context=ctx,
            config=CloudRegimeShortGateConfig(mode="momentum_and_cloud_slope"),
        ),
        start_index=0,
        stop_index=54,
        initial_capital=10000,
        capture_ledger_evidence=True,
    )
    n0 = run_shared_portfolio_replay(
        ds,
        ShortNullControlPolicy(
            symbol="BTCUSDT",
            signals=pack,
            short_permission=np.zeros(55, dtype=bool),
            config=ShortNullControlConfig(mode="long_only"),
        ),
        start_index=0,
        stop_index=54,
        initial_capital=10000,
        capture_ledger_evidence=True,
    )
    assert r2.ledger_evidence and n0.ledger_evidence
    audit = audit_paired_short_hours(
        close=ds.close[:, 0],
        timestamps=ds.timestamps,
        r2_intervals=r2.ledger_evidence.intervals.intervals,
        control_intervals=n0.ledger_evidence.intervals.intervals,
        start_index=0,
        stop_index=54,
    )
    assert audit.total_hour_count == 54
    assert audit.r2_short_hour_count > 0
    assert audit.control_short_hour_count == 0


def test_bad_exact_quantity_and_discontinuous_equity_are_rejected() -> None:
    from dataclasses import replace

    prices = np.asarray([100.0, 99.0, 98.0, 97.0])
    first = trace(("0", "-1", "-1", "0"))
    invalid = list(first)
    invalid[0] = replace(invalid[0], exact_quantities_after=("not-a-number",))
    with pytest.raises(ValueError):
        audit_paired_short_hours(
            close=prices,
            timestamps=clock(4),
            r2_intervals=tuple(invalid),
            control_intervals=first,
            start_index=0,
            stop_index=3,
        )
    inconsistent = list(first)
    inconsistent[1] = replace(inconsistent[1], portfolio_value_before=899.0)
    with pytest.raises(ValueError, match="equity"):
        audit_paired_short_hours(
            close=prices,
            timestamps=clock(4),
            r2_intervals=tuple(inconsistent),
            control_intervals=first,
            start_index=0,
            stop_index=3,
        )


def test_selected_price_window_must_be_positive_and_inside_bounds() -> None:
    prices = np.asarray([100.0, 99.0, 98.0, 97.0])
    traces = trace(("0", "-1", "-1", "0"))
    bad = prices.copy()
    bad[2] = 0.0
    with pytest.raises(ValueError, match="positive"):
        audit_paired_short_hours(
            close=bad,
            timestamps=clock(4),
            r2_intervals=traces,
            control_intervals=traces,
            start_index=0,
            stop_index=3,
        )
    with pytest.raises(ValueError, match="bounds"):
        audit_paired_short_hours(
            close=prices,
            timestamps=clock(4),
            r2_intervals=traces,
            control_intervals=traces,
            start_index=0,
            stop_index=4,
        )


def test_invalid_timestamp_dtype_and_nat_are_rejected() -> None:
    values = np.array([100.0, 101.0, 102.0])
    ledgers = trace(("0", "-1", "-1"))
    with pytest.raises(ValueError, match="datetime64"):
        audit_paired_short_hours(
            close=values,
            timestamps=np.array([0, 1, 2], dtype=np.int64),
            r2_intervals=ledgers,
            control_intervals=ledgers,
            start_index=0,
            stop_index=2,
        )
    clock_with_nat = clock(3)
    clock_with_nat[1] = np.datetime64("NaT")
    with pytest.raises(ValueError, match="datetime64"):
        audit_paired_short_hours(
            close=values,
            timestamps=clock_with_nat,
            r2_intervals=ledgers,
            control_intervals=ledgers,
            start_index=0,
            stop_index=2,
        )
