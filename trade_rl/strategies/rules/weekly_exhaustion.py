"""Ordered forming-week band reach and later native Tenkan exhaustion."""

from __future__ import annotations

import math

import numpy as np

from trade_rl.strategies.interface import SingleSymbolStrategy, StrategyObservation
from trade_rl.strategies.position_intent import PositionIntent

_HOUR_NS = 3_600_000_000_000
_WEEK_NS = 168 * _HOUR_NS
_MONDAY_NS = int(np.datetime64("1970-01-05", "ns").astype(np.int64))


class FormingWeekExhaustionStrategy:
    """Veto a direction after a band reach and its first later native cross.

    Ratios use inclusive stored-value thresholds. Confirmation needs adjacent,
    fresh UTC four-hour events; carried values never confirm or recover. Missing
    required inputs, forward observation gaps and new weeks clear filter state.
    The base receives every accepted observation, including ordinary FLAT rows.
    Reconstruct state with a fresh base and the same authoritative prefix; there
    is no checkpoint or synthetic position-history priming interface.
    """

    def __init__(
        self,
        strategy: SingleSymbolStrategy,
        feature_indices: tuple[int, ...],
        *,
        short_term_index: int,
    ) -> None:
        indices = (*feature_indices, short_term_index)
        if (
            len(feature_indices) != 2
            or any(
                isinstance(index, bool) or not isinstance(index, int) or index < 0
                for index in indices
            )
            or len(set(indices)) != 3
        ):
            raise ValueError("context requires three distinct non-negative indices")
        self.strategy = strategy
        self.feature_indices = feature_indices
        self.short_term_index = short_term_index
        self._symbol: str | None = None
        self._timestamp: int | None = None
        self._index: int | None = None
        self._week: int | None = None
        self._clear_filter()

    @property
    def protective_exit_pending(self) -> bool:
        return bool(getattr(self.strategy, "protective_exit_pending", False))

    def _clear_filter(self) -> None:
        self._arms: dict[PositionIntent, int] = {}
        self._blocked: set[PositionIntent] = set()
        self._native_timestamp: int | None = None
        self._native_distance: float | None = None

    def _accept_clock(self, observation: StrategyObservation) -> int:
        timestamp = int(observation.timestamp.astype(np.int64))
        if np.isnat(observation.timestamp) or timestamp % _HOUR_NS:
            raise ValueError("observations require UTC hourly close timestamps")
        if self._symbol is not None and observation.symbol != self._symbol:
            raise ValueError("strategy observations must belong to one symbol")
        if self._timestamp is not None and (
            timestamp <= self._timestamp
            or self._index is None
            or observation.index <= self._index
        ):
            raise ValueError("observation timestamp and index must strictly increase")
        if any(
            index >= observation.features.size
            for index in (*self.feature_indices, self.short_term_index)
        ):
            raise ValueError("context feature index out of range")
        week = (timestamp - 1 - _MONDAY_NS) // _WEEK_NS
        if self._timestamp is not None and (
            timestamp - self._timestamp != _HOUR_NS
            or self._index is None
            or observation.index != self._index + 1
            or week != self._week
        ):
            self._clear_filter()
        self._symbol = observation.symbol
        self._timestamp = timestamp
        self._index = observation.index
        self._week = week
        return timestamp

    def _native_event(self, timestamp: int, distance: float) -> None:
        if distance >= 0:
            self._blocked.discard(PositionIntent.LONG)
        if distance <= 0:
            self._blocked.discard(PositionIntent.SHORT)
        adjacent = (
            self._native_timestamp is not None
            and timestamp - self._native_timestamp == 4 * _HOUR_NS
            and self._native_distance is not None
        )
        for side, armed_at in self._arms.items():
            if armed_at >= timestamp or not adjacent:
                continue
            previous = self._native_distance
            assert previous is not None
            crossed = (
                previous >= 0 and distance < 0
                if side is PositionIntent.LONG
                else previous <= 0 and distance > 0
            )
            if crossed:
                self._blocked.add(side)
        # This event is the first opportunity even when it cannot confirm.
        self._arms.clear()
        self._native_timestamp = timestamp
        self._native_distance = distance

    def decide(self, observation: StrategyObservation) -> PositionIntent:
        timestamp = self._accept_clock(observation)
        decision = self.strategy.decide(observation)
        indices = (*self.feature_indices, self.short_term_index)
        values = tuple(float(observation.features[index]) for index in indices)
        if (
            not observation.feature_available[list(indices)].all()
            or not all(math.isfinite(value) for value in values)
            or observation.feature_staleness is None
        ):
            self._clear_filter()
            return PositionIntent.FLAT
        if np.any(observation.feature_staleness[list(self.feature_indices)] != 0):
            self._clear_filter()
            return PositionIntent.FLAT
        high, low, distance = values
        boundary = timestamp % (4 * _HOUR_NS) == 0
        fresh = observation.feature_staleness[self.short_term_index] == 0
        if boundary and not fresh:
            self._clear_filter()
            return PositionIntent.FLAT
        if boundary:
            self._native_event(timestamp, distance)
        # Existing reaches consume the event before the current hour can arm.
        for side, reaches in (
            (PositionIntent.LONG, high >= 1),
            (PositionIntent.SHORT, low <= -1),
        ):
            if reaches and side not in self._blocked:
                self._arms.setdefault(side, timestamp)
        return PositionIntent.FLAT if decision in self._blocked else decision
