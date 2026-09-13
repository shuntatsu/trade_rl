from __future__ import annotations

import numpy as np
import pytest

from research.issue519_portable_exp002 import (
    _intent_transition_count,
    _verify_daily_signal_refresh_alignment,
)


def test_mean_reversion_transition_counter_obeys_frozen_hysteresis() -> None:
    signal = np.asarray([0.0, 0.02, 0.015, 0.001, -0.02, -0.015, -0.001])
    available = np.ones(signal.shape, dtype=np.bool_)

    assert (
        _intent_transition_count(
            signal,
            available,
            entry_threshold=0.01,
            exit_threshold=0.0025,
        )
        == 4
    )


def test_mean_reversion_transition_counter_flattens_when_signal_unavailable() -> None:
    signal = np.asarray([0.02, 0.02])
    available = np.asarray([True, False], dtype=np.bool_)

    assert (
        _intent_transition_count(
            signal,
            available,
            entry_threshold=0.01,
            exit_threshold=0.0025,
        )
        == 2
    )


def test_daily_signal_refresh_alignment_requires_exact_rolling_value_match() -> None:
    rolling = np.asarray([0.01, 0.02, 0.03, 0.04], dtype=np.float64)
    daily = np.asarray([0.01, 0.01, 0.03, 0.03], dtype=np.float64)
    available = np.ones(4, dtype=np.bool_)
    staleness = np.asarray([0.0, 0.5, 0.0, 0.5], dtype=np.float64)

    assert (
        _verify_daily_signal_refresh_alignment(
            rolling,
            daily,
            available,
            staleness,
        )
        == 2
    )


def test_daily_signal_refresh_alignment_fails_closed_on_value_mismatch() -> None:
    rolling = np.asarray([0.01, 0.02, 0.031, 0.04], dtype=np.float64)
    daily = np.asarray([0.01, 0.01, 0.03, 0.03], dtype=np.float64)
    available = np.ones(4, dtype=np.bool_)
    staleness = np.asarray([0.0, 0.5, 0.0, 0.5], dtype=np.float64)

    with pytest.raises(RuntimeError, match="daily/rolling 24h signal mismatch"):
        _verify_daily_signal_refresh_alignment(
            rolling,
            daily,
            available,
            staleness,
        )
