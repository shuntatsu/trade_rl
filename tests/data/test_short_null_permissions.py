"""Causal weekly randomized null controls are never an economic winner selector."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from tests.strategies.test_cloud_regime_short_gate import _context
from tests.strategies.test_divergence_cloud_follow import _signals
from trade_rl.data.features.short_null_permissions import weekly_short_permission


def _context_for_null(n: int = 3000):
    pack = _signals([-1] * n)
    context = _context(pack, allowed=tuple(range(0, n, 2)))
    return context


def test_deterministic_weekly_prefix_and_seed_variation() -> None:
    context = _context_for_null()
    masks = [weekly_short_permission(context, seed=s) for s in (17, 37, 97)]
    for seed, mask in zip((17, 37, 97), masks, strict=True):
        assert mask.dtype == np.bool_
        assert mask.shape == context.ready.shape
        assert not mask.flags.writeable
        for k in (170, 1200, 2401):
            truncated = replace(
                context,
                timestamps=context.timestamps[:k],
                ready=context.ready[:k],
                momentum_30d=context.momentum_30d[:k],
                cloud_mid_slope_7d=context.cloud_mid_slope_7d[:k],
                distance_to_cloud_mid_pct=context.distance_to_cloud_mid_pct[:k],
                realized_vol_30d=context.realized_vol_30d[:k],
            )
            np.testing.assert_array_equal(
                mask[:k],
                weekly_short_permission(truncated, seed=seed),
            )
    assert any(np.any(masks[0] != other) for other in masks[1:])


def test_missing_data_and_late_unavailable_context_is_fail_closed() -> None:
    context = _context_for_null()
    status = context.ready.copy()
    status[1200:1300] = False
    changed = replace(context, ready=status)
    allowed = weekly_short_permission(changed, seed=17)
    assert not allowed[1200:1300].any()
    assert allowed.shape == context.ready.shape


def test_future_mutation_never_changes_previous_null_permissions() -> None:
    context = _context_for_null()
    first = weekly_short_permission(context, seed=17)
    momentum = context.momentum_30d.copy()
    slope = context.cloud_mid_slope_7d.copy()
    momentum[2000:] *= -1
    slope[2000:] *= -1
    changed = replace(context, momentum_30d=momentum, cloud_mid_slope_7d=slope)
    again = weekly_short_permission(changed, seed=17)
    np.testing.assert_array_equal(first[:2000], again[:2000])


@pytest.mark.parametrize("invalid", [0, 1, 36, True, -17, 100])
def test_bad_or_unregistered_seed_rejected(invalid: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        weekly_short_permission(_context_for_null(200), seed=invalid)


def test_nonhourly_clock_and_bad_ready_values_rejected() -> None:
    context = _context_for_null(200)
    broken = context.timestamps.copy()
    broken[70] += np.timedelta64(1, "h")
    with pytest.raises(ValueError, match="regular"):
        weekly_short_permission(replace(context, timestamps=broken), seed=17)
    invalid = context.momentum_30d.copy()
    invalid[100] = np.nan
    with pytest.raises(ValueError, match="finite"):
        weekly_short_permission(replace(context, momentum_30d=invalid), seed=17)


def test_probability_is_frozen_after_first_available_weekly_decision() -> None:
    context = _context_for_null(1100)
    original = weekly_short_permission(context, seed=17)
    # Midweek changes to eligibility cannot alter an already sampled week's
    # random permission, while missing hourly data must still fail closed.
    changed = context.momentum_30d.copy()
    changed[1010:] = -1.0 * changed[1010:]
    altered = replace(context, momentum_30d=changed)
    later = weekly_short_permission(altered, seed=17)
    np.testing.assert_array_equal(original[:1010], later[:1010])


def test_week_start_unavailable_then_first_ready_uses_past_only() -> None:
    context = _context_for_null(500)
    status = context.ready.copy()
    # The first UTC Monday in the synthetic date stream is 2022-01-03,
    # after 48 hourly observations. At the boundary, no context is available.
    status[48] = False
    later = replace(context, ready=status)
    permission = weekly_short_permission(later, seed=17)
    assert permission[48] == np.bool_(False)
    assert permission.shape == (500,)


def test_first_available_hour_not_unavailable_week_boundary_controls_sample(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.data.features import short_null_permissions

    base = _context_for_null(240)
    ready = base.ready.copy()
    ready[48] = False  # 2022-01-03 00:00 UTC Monday weekly boundary
    momentum = np.ones(240)
    momentum[49] = -0.2
    changed = replace(base, ready=ready, momentum_30d=momentum)
    monkeypatch.setattr(short_null_permissions, "_hash_uniform", lambda **_: 0.0)
    allowed = weekly_short_permission(changed, seed=17)
    assert not allowed[48]
    assert allowed[49]  # sampling deferred until this first usable hour


def test_input_mask_cannot_change_returned_weekly_decisions() -> None:
    ctx = _context_for_null(400)
    mask = weekly_short_permission(ctx, seed=17)
    original = mask.copy()
    # Return values are immutable and not aliases of the mutable inputs.
    ready = ctx.ready
    ready[:200] = False
    np.testing.assert_array_equal(original, mask)
