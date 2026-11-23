"""Causal weekly short exposure null: hash-based control, not a return signal.

The random sampling schedule and seeds are frozen in the post-hoc exploratory
protocol. At each UTC week start, the short permission is drawn using only
the R2 short-eligibility *frequency* in a trailing 90-day as-of window.
No return, outcome, or full-year eligibility quota is used for selection.
"""

from __future__ import annotations

import hashlib

import numpy as np

from trade_rl.data.features.cloud_regime_context import CloudRegimeContext

_HOUR_NS = 3_600_000_000_000
_WEEK_NS = 7 * 24 * _HOUR_NS
_WEEK_ORIGIN_NS = int(np.datetime64("1970-01-05T00:00", "ns").astype(np.int64))
_LOOKBACK_HOURS = 90 * 24
_MIN_OBSERVED_CONTEXT_HOURS = 48
_FROZEN_SEEDS = frozenset((17, 37, 97))


def _hash_uniform(*, seed: int, utc_week: int) -> float:
    digest = hashlib.sha256(f"short-null-v1|{seed}|{utc_week}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64)


def weekly_short_permission(
    context: CloudRegimeContext,
    *,
    seed: int,
) -> np.ndarray:
    """One random weekly draw with past-only R2 occupancy as its probability.

    Weekly choices are fixed at the first native hourly decision in that UTC
    week; future R2 eligibility never changes past weekly choices. Invalid
    or unavailable data masks the current decision, without forward filling.
    """
    if type(seed) is not int or seed not in _FROZEN_SEEDS:
        raise ValueError("random-null seed must be one of the fixed protocol seeds")
    n = len(context.timestamps)
    fields = (
        context.ready,
        context.momentum_30d,
        context.cloud_mid_slope_7d,
        context.distance_to_cloud_mid_pct,
        context.realized_vol_30d,
    )
    if n == 0 or any(np.asarray(field).shape != (n,) for field in fields):
        raise ValueError("null context must be nonempty with aligned 1h fields")
    ready = np.asarray(context.ready)
    if ready.dtype != np.bool_:
        raise ValueError("context ready mask must be boolean")
    for value in fields[1:]:
        arr = np.asarray(value, dtype=np.float64)
        if not np.isfinite(arr[ready]).all():
            raise ValueError("all ready context values must be finite")

    times = context.timestamps.astype("datetime64[ns]").astype(np.int64)
    if np.any(times % _HOUR_NS != 0) or (n > 1 and np.any(np.diff(times) != _HOUR_NS)):
        raise ValueError("regular UTC 1h decision clock required")

    eligible = ready & (context.momentum_30d < 0) & (context.cloud_mid_slope_7d < 0)
    ready_cumulative = np.r_[0, np.cumsum(ready, dtype=np.int64)]
    eligible_cumulative = np.r_[0, np.cumsum(eligible, dtype=np.int64)]
    weeks = (times - _WEEK_ORIGIN_NS) // _WEEK_NS
    permissions = np.zeros(n, dtype=np.bool_)

    week_allowed = False
    sampled_this_week = False
    last_week = -(2**63)
    for i in range(n):
        week = int(weeks[i])
        if week != last_week:
            sampled_this_week = False
            week_allowed = False
            last_week = week
        if not sampled_this_week and bool(ready[i]):
            # Sample at the first usable decision only: no future backfill.
            start = max(0, i + 1 - _LOOKBACK_HOURS)
            seen = int(ready_cumulative[i + 1] - ready_cumulative[start])
            if seen >= _MIN_OBSERVED_CONTEXT_HOURS:
                positive = int(eligible_cumulative[i + 1] - eligible_cumulative[start])
                frequency = positive / seen
                week_allowed = _hash_uniform(seed=seed, utc_week=week) < frequency
            sampled_this_week = True
        permissions[i] = sampled_this_week and week_allowed and bool(ready[i])
    permissions.setflags(write=False)
    return permissions


__all__ = ["weekly_short_permission"]
