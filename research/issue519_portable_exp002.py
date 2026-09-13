"""Pre-result diagnostics and preregistration helpers for Issue #519."""

from __future__ import annotations

import math

import numpy as np


def _intent_transition_count(
    signal: np.ndarray,
    available: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> int:
    """Replay only the frozen mean-reversion intent state machine."""

    values = np.asarray(signal, dtype=np.float64).reshape(-1)
    availability = np.asarray(available, dtype=np.bool_).reshape(-1)
    if values.shape != availability.shape or values.size == 0:
        raise ValueError("signal and availability must be non-empty equal-length vectors")
    if not math.isfinite(entry_threshold) or entry_threshold <= 0.0:
        raise ValueError("entry_threshold must be finite and positive")
    if (
        not math.isfinite(exit_threshold)
        or exit_threshold < 0.0
        or exit_threshold >= entry_threshold
    ):
        raise ValueError("exit_threshold must be finite and below entry_threshold")

    # -1 = SHORT, 0 = FLAT, 1 = LONG.
    current = 0
    changes = 0
    for raw, is_available in zip(values, availability, strict=True):
        if not is_available or not math.isfinite(float(raw)):
            next_intent = 0
        else:
            value = float(raw)
            if current == 0:
                if value >= entry_threshold:
                    next_intent = -1
                elif value <= -entry_threshold:
                    next_intent = 1
                else:
                    next_intent = 0
            elif current == 1:
                if value >= entry_threshold:
                    next_intent = -1
                elif value >= -exit_threshold:
                    next_intent = 0
                else:
                    next_intent = 1
            else:
                if value <= -entry_threshold:
                    next_intent = 1
                elif value <= exit_threshold:
                    next_intent = 0
                else:
                    next_intent = -1
        if next_intent != current:
            changes += 1
        current = next_intent
    return changes


def _verify_daily_signal_refresh_alignment(
    rolling_signal: np.ndarray,
    daily_signal: np.ndarray,
    daily_available: np.ndarray,
    daily_staleness: np.ndarray,
) -> int:
    """Require exact rolling/daily 24h-return equality at daily refresh points."""

    rolling = np.asarray(rolling_signal, dtype=np.float64).reshape(-1)
    daily = np.asarray(daily_signal, dtype=np.float64).reshape(-1)
    available = np.asarray(daily_available, dtype=np.bool_).reshape(-1)
    staleness = np.asarray(daily_staleness, dtype=np.float64).reshape(-1)
    if not (
        rolling.shape == daily.shape == available.shape == staleness.shape
        and rolling.size > 0
    ):
        raise ValueError("signal alignment inputs must be non-empty equal-length vectors")
    if not np.isfinite(staleness).all() or np.any(staleness < 0.0):
        raise ValueError("daily staleness must be finite and non-negative")

    refresh = available & (staleness == 0.0)
    count = int(np.count_nonzero(refresh))
    if count == 0:
        raise RuntimeError("no daily signal refresh points found")
    rolling_refresh = rolling[refresh]
    daily_refresh = daily[refresh]
    if not np.isfinite(rolling_refresh).all() or not np.isfinite(daily_refresh).all():
        raise RuntimeError("daily/rolling 24h signal refresh contains non-finite values")
    if not np.array_equal(rolling_refresh, daily_refresh):
        raise RuntimeError("daily/rolling 24h signal mismatch")
    return count


__all__ = ["_intent_transition_count", "_verify_daily_signal_refresh_alignment"]
