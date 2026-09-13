"""Pre-result structural diagnostics for portable Experiment 0003.

This module deliberately contains no candidate execution or P&L computation. It
only validates the chosen immutable feature identity and replays the frozen trend
intent state machine for transition-count diagnostics.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from trade_rl.strategies.interface import StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent
from trade_rl.strategies.rules.trend import TrendIntentConfig, TrendIntentStrategy


def require_feature_index(
    feature_names: Sequence[str],
    *,
    name: str,
    expected_index: int,
) -> int:
    """Require one exact feature identity at the preselected immutable index."""

    names = tuple(feature_names)
    if not name or any(not isinstance(item, str) or not item for item in names):
        raise ValueError("feature names must be non-empty strings")
    if (
        isinstance(expected_index, bool)
        or not isinstance(expected_index, int)
        or expected_index < 0
    ):
        raise ValueError("expected_index must be a non-negative integer")
    matches = tuple(index for index, item in enumerate(names) if item == name)
    if len(matches) != 1:
        raise RuntimeError(f"feature identity missing or duplicated: {name}")
    observed = matches[0]
    if observed != expected_index:
        raise RuntimeError(
            f"feature index drift for {name}: expected={expected_index} observed={observed}"
        )
    return observed


def trend_transition_count(
    signal: np.ndarray,
    available: np.ndarray,
    *,
    entry_threshold: float,
    exit_threshold: float,
) -> int:
    """Count intent changes by invoking the frozen production trend strategy."""

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

    strategy = TrendIntentStrategy(
        TrendIntentConfig(
            signal_index=0,
            entry_threshold=entry_threshold,
            exit_threshold=exit_threshold,
        )
    )
    current = PositionIntent.FLAT
    changes = 0
    for index, (raw, is_available) in enumerate(
        zip(values, availability, strict=True)
    ):
        observation = StrategyObservation(
            index=index,
            timestamp=np.datetime64(index, "ns"),
            symbol="STRUCTURAL_DIAGNOSTIC",
            features=np.asarray([raw], dtype=np.float64),
            feature_available=np.asarray([is_available], dtype=np.bool_),
            global_features=np.asarray([0.0], dtype=np.float64),
            global_feature_available=np.asarray([True], dtype=np.bool_),
            current_intent=current,
            current_weight=0.0,
        )
        next_intent = strategy.decide(observation)
        if next_intent is not current:
            changes += 1
        current = next_intent
    return changes


__all__ = ["require_feature_index", "trend_transition_count"]
