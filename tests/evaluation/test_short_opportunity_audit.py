"""As-of daily short opportunity diagnostics: labels are offline-only."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from trade_rl.data.features.cloud_regime_context import CloudRegimeContext
from trade_rl.data.features.divergence_mtf import DivergenceCloudMtfPack
from trade_rl.evaluation.short_opportunity_audit import (
    audit_24h_short_opportunities,
    summarize_short_opportunities,
)


def _inputs(n: int = 24 * 8):
    stamp = np.datetime64("2022-01-01T01:00", "ns") + np.arange(n) * np.timedelta64(
        1, "h"
    )
    close = 100 + np.arange(n, dtype=float) / 4
    zero = np.zeros(n, dtype=np.int16)
    signals = DivergenceCloudMtfPack(
        timestamps=stamp,
        bull_1h=zero,
        bear_1h=zero,
        bull_4h=zero,
        bear_4h=zero,
        cloud_state=np.full(n, -1, dtype=np.int8),
        cloud_ready=np.ones(n, dtype=np.bool_),
        events_1h=(),
        events_4h=(),
    )
    mom = np.full(n, -0.1)
    slope = np.full(n, -0.1)
    mom[23] = 0.2
    slope[47] = 0.1
    context = CloudRegimeContext(
        timestamps=stamp,
        momentum_30d=mom,
        cloud_mid_slope_7d=slope,
        distance_to_cloud_mid_pct=np.zeros(n),
        realized_vol_30d=np.ones(n) * 0.4,
        ready=np.ones(n, dtype=np.bool_),
    )
    return close, signals, context


def test_groups_are_disjoint_and_same_daily_clock_with_no_lookahead_feature() -> None:
    close, signals, context = _inputs()
    rows = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert rows[0].index == 23
    assert rows[0].group == "G0_momentum_nonnegative"
    assert rows[1].index == 47
    assert rows[1].group == "G1_momentum_negative_slope_nonnegative"
    assert rows[2].index == 71
    assert rows[2].group == "G2_both_negative"
    assert all(row.index % 24 == 23 for row in rows)
    assert rows[0].signed_24h_return_pct == pytest.approx(
        100 * (1 - close[47] / close[23])
    )
    assert all(row.normalized_24h_short_move is not None for row in rows)
    summary = summarize_short_opportunities(rows)
    assert sum(group.count for group in summary.values()) == len(rows)
    assert summary["G0_momentum_nonnegative"].count == 1
    assert summary["G1_momentum_negative_slope_nonnegative"].count == 1


def test_six_completed_cloud_decisions_required() -> None:
    close, signals, context = _inputs()
    cloud_state = signals.cloud_state.copy()
    cloud_state[19] = 1  # breaks confirmation for observation t=23
    signals = dataclasses.replace(signals, cloud_state=cloud_state)
    rows = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert 23 not in {r.index for r in rows}
    assert 47 in {r.index for r in rows}


def test_offline_label_never_exceeds_stop_boundary() -> None:
    close, signals, context = _inputs()
    rows = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=context,
        start_index=24,
        stop_index=71,
    )
    assert [r.index for r in rows] == [47]
    assert rows[0].next_index == 71


def test_later_future_price_change_cannot_rewrite_earlier_finished_labels() -> None:
    close, signals, context = _inputs()
    original = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    changed = close.copy()
    changed[120:] *= 100
    new = audit_24h_short_opportunities(
        close=changed,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert tuple(r for r in original if r.next_index < 120) == tuple(
        r for r in new if r.next_index < 120
    )


def test_unready_macro_context_excludes_daily_opportunities() -> None:
    close, signals, context = _inputs()
    ready = context.ready.copy()
    ready[47] = False
    context = dataclasses.replace(context, ready=ready)
    rows = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert 47 not in {r.index for r in rows}


def test_constant_lookback_volatility_has_missing_normalized_label() -> None:
    close, signals, context = _inputs()
    zero_vol = dataclasses.replace(context, realized_vol_30d=np.zeros(len(close)))
    rows = audit_24h_short_opportunities(
        close=close,
        signals=signals,
        context=zero_vol,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert rows and all(r.normalized_24h_short_move is None for r in rows)


def test_bad_clock_and_nonfinite_ready_momentum_fail_closed() -> None:
    close, signals, context = _inputs()
    bad = signals.timestamps.copy()
    bad[50] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="regular"):
        audit_24h_short_opportunities(
            close=close,
            signals=dataclasses.replace(signals, timestamps=bad),
            context=dataclasses.replace(context, timestamps=bad),
            start_index=0,
            stop_index=len(close) - 1,
        )
    mom = context.momentum_30d.copy()
    mom[47] = np.nan
    with pytest.raises(ValueError, match="ready"):
        audit_24h_short_opportunities(
            close=close,
            signals=signals,
            context=dataclasses.replace(context, momentum_30d=mom),
            start_index=0,
            stop_index=len(close) - 1,
        )


def test_misaligned_signal_and_macro_timestamp_rejected() -> None:
    close, signals, context = _inputs()
    moved = context.timestamps + np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="align"):
        audit_24h_short_opportunities(
            close=close,
            signals=signals,
            context=dataclasses.replace(context, timestamps=moved),
            start_index=0,
            stop_index=len(close) - 1,
        )


def test_zero_forward_movement_is_not_counted_as_favorable() -> None:
    close, signals, context = _inputs()
    flat = np.full_like(close, 100.0)
    rows = audit_24h_short_opportunities(
        close=flat,
        signals=signals,
        context=context,
        start_index=0,
        stop_index=len(flat) - 1,
    )
    result = summarize_short_opportunities(rows)
    assert sum(x.count for x in result.values()) == len(rows)
    assert all(
        group.favorable_fraction == 0.0 for group in result.values() if group.count
    )


def test_missing_cloud_ready_or_macro_ready_never_creates_candidate() -> None:
    close, signals, context = _inputs()
    cloud_ready = signals.cloud_ready.copy()
    cloud_ready[45] = False
    changed = dataclasses.replace(signals, cloud_ready=cloud_ready)
    rows = audit_24h_short_opportunities(
        close=close,
        signals=changed,
        context=context,
        start_index=0,
        stop_index=len(close) - 1,
    )
    assert 47 not in {row.index for row in rows}


def test_invalid_price_or_study_bounds_are_refused() -> None:
    close, signals, context = _inputs()
    invalid = close.copy()
    invalid[-1] = float("inf")
    with pytest.raises(ValueError, match="finite"):
        audit_24h_short_opportunities(
            close=invalid,
            signals=signals,
            context=context,
            start_index=0,
            stop_index=len(close) - 1,
        )
    with pytest.raises(ValueError, match="study bounds"):
        audit_24h_short_opportunities(
            close=close,
            signals=signals,
            context=context,
            start_index=0,
            stop_index=len(close),
        )


def test_empty_category_summary_remains_visible_without_imputed_return() -> None:
    results = summarize_short_opportunities(())
    assert len(results) == 3
    assert all(
        group.count == 0
        and group.mean_signed_24h_pct is None
        and group.favorable_fraction is None
        for group in results.values()
    )
