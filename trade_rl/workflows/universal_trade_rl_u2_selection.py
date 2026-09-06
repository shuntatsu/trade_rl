"""Frozen U2 Development panel reduction and segmented bootstrap contracts."""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from statistics import fmean, median
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.workflows.universal_trade_rl_u2_contract import U2_TRAINING_SEEDS

U2_DEVELOPMENT_PANEL_SCHEMA: Final = "universal_trade_rl_u2_development_panel_v1"
U2_DEVELOPMENT_BOOTSTRAP_SCHEMA: Final = (
    "universal_trade_rl_u2_development_segmented_bootstrap_v1"
)
U2_DEVELOPMENT_WINDOWS: Final = (
    "development_future_1",
    "development_future_2",
)
_U2_BOOTSTRAP_RESAMPLES: Final = 2_000
_U2_BOOTSTRAP_SEED: Final = 0
_U2_BOOTSTRAP_CONFIDENCE_LEVEL: Final = 0.95
_U2_BOOTSTRAP_QUANTILE_METHOD: Final = "linear"


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)):
        raise ValueError(f"{field} must be numeric")
    resolved = float(value)
    if not math.isfinite(resolved):
        raise ValueError(f"{field} must be finite")
    return resolved


def _non_negative_int(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{field} must be a non-negative integer")
    return value


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2PairedExcessPoint:
    """One Development leaf after same-scope candidate/cash pairing."""

    training_seed: int
    source_window: str
    concrete_symbol: str
    decision_timestamp_ns: int
    candidate_minus_cash_net_log_excess: float

    def __post_init__(self) -> None:
        if self.training_seed not in U2_TRAINING_SEEDS:
            raise ValueError(
                "U2 Development panel seed is outside the fixed seed closure"
            )
        if self.source_window not in U2_DEVELOPMENT_WINDOWS:
            raise ValueError("U2 Development panel source window is not D1/D2")
        if not isinstance(self.concrete_symbol, str) or not self.concrete_symbol:
            raise ValueError("U2 Development panel symbol must be non-empty")
        _non_negative_int(
            self.decision_timestamp_ns,
            field="U2 Development panel decision timestamp",
        )
        object.__setattr__(
            self,
            "candidate_minus_cash_net_log_excess",
            _finite(
                self.candidate_minus_cash_net_log_excess,
                field="U2 Development paired net-log excess",
            ),
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReducedBootstrapSegment:
    """One chronological D1 or D2 series after symbol/seed reduction."""

    source_window: str
    decision_timestamps_ns: tuple[int, ...]
    net_log_excess: tuple[float, ...]

    def __post_init__(self) -> None:
        if self.source_window not in U2_DEVELOPMENT_WINDOWS:
            raise ValueError("U2 bootstrap segment source window is not D1/D2")
        timestamps = tuple(self.decision_timestamps_ns)
        values = tuple(
            _finite(value, field="U2 bootstrap segment net-log excess")
            for value in self.net_log_excess
        )
        if not timestamps or len(timestamps) != len(values):
            raise ValueError("U2 bootstrap segment must be non-empty and aligned")
        if any(
            _non_negative_int(timestamp, field="U2 bootstrap segment timestamp")
            != timestamp
            for timestamp in timestamps
        ):
            raise ValueError("U2 bootstrap segment timestamps are invalid")
        if timestamps != tuple(sorted(set(timestamps))):
            raise ValueError(
                "U2 bootstrap segment timestamps must be sorted and unique"
            )
        object.__setattr__(self, "decision_timestamps_ns", timestamps)
        object.__setattr__(self, "net_log_excess", values)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "schema_version": U2_DEVELOPMENT_PANEL_SCHEMA,
                "source_window": self.source_window,
                "decision_timestamps_ns": self.decision_timestamps_ns,
                "net_log_excess": self.net_log_excess,
            }
        )


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2DevelopmentBootstrapResult:
    """Deterministic segmented moving-block bootstrap evidence for D1+D2."""

    segment_digests: tuple[str, ...]
    block_lengths: tuple[int, ...]
    observed_mean: float
    lower_ci: float
    upper_ci: float
    passed: bool
    resamples: int = _U2_BOOTSTRAP_RESAMPLES
    bootstrap_seed: int = _U2_BOOTSTRAP_SEED
    confidence_level: float = _U2_BOOTSTRAP_CONFIDENCE_LEVEL
    quantile_method: str = _U2_BOOTSTRAP_QUANTILE_METHOD
    blocks_may_cross_segment_boundary: bool = False
    schema_version: str = U2_DEVELOPMENT_BOOTSTRAP_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        if self.schema_version != U2_DEVELOPMENT_BOOTSTRAP_SCHEMA:
            raise ValueError("unsupported U2 Development bootstrap schema")
        if not self.segment_digests or len(self.segment_digests) != len(
            self.block_lengths
        ):
            raise ValueError("U2 Development bootstrap segment evidence is incomplete")
        if any(len(value) != 64 for value in self.segment_digests):
            raise ValueError("U2 Development bootstrap segment digest is invalid")
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in self.block_lengths
        ):
            raise ValueError("U2 Development bootstrap block length is invalid")
        for field_name in ("observed_mean", "lower_ci", "upper_ci"):
            object.__setattr__(
                self,
                field_name,
                _finite(getattr(self, field_name), field=f"U2 bootstrap {field_name}"),
            )
        if self.lower_ci > self.upper_ci:
            raise ValueError("U2 Development bootstrap interval is reversed")
        if self.resamples != _U2_BOOTSTRAP_RESAMPLES:
            raise ValueError("U2 Development bootstrap resample count drifted")
        if self.bootstrap_seed != _U2_BOOTSTRAP_SEED:
            raise ValueError("U2 Development bootstrap seed drifted")
        if not math.isclose(
            self.confidence_level,
            _U2_BOOTSTRAP_CONFIDENCE_LEVEL,
            rel_tol=0.0,
            abs_tol=0.0,
        ):
            raise ValueError("U2 Development bootstrap confidence level drifted")
        if self.quantile_method != _U2_BOOTSTRAP_QUANTILE_METHOD:
            raise ValueError("U2 Development bootstrap quantile method drifted")
        if self.blocks_may_cross_segment_boundary is not False:
            raise ValueError("U2 Development bootstrap blocks may not cross D1/D2")
        if self.passed is not (self.lower_ci > 0.0):
            raise ValueError("U2 Development bootstrap pass state is inconsistent")

        expected = content_digest(self.to_payload(include_digest=False))
        if self.digest and self.digest != expected:
            raise ValueError("U2 Development bootstrap digest mismatch")
        object.__setattr__(self, "digest", expected)

    def to_payload(self, *, include_digest: bool = True) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "segment_digests": self.segment_digests,
            "block_lengths": self.block_lengths,
            "observed_mean": self.observed_mean,
            "lower_ci": self.lower_ci,
            "upper_ci": self.upper_ci,
            "passed": self.passed,
            "resamples": self.resamples,
            "bootstrap_seed": self.bootstrap_seed,
            "confidence_level": self.confidence_level,
            "quantile_method": self.quantile_method,
            "blocks_may_cross_segment_boundary": self.blocks_may_cross_segment_boundary,
        }
        if include_digest:
            payload["artifact_digest"] = self.digest
        return payload


def reduce_universal_trade_rl_u2_development_panel(
    *,
    points: tuple[UniversalTradeRLU2PairedExcessPoint, ...],
    expected_symbols: tuple[str, ...],
) -> tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...]:
    """Reduce D1/D2 leaves as equal-weight symbols then median training seed."""

    resolved_points = tuple(points)
    symbols = tuple(expected_symbols)
    if not resolved_points:
        raise ValueError("U2 Development panel points must be non-empty")
    if not symbols or symbols != tuple(sorted(set(symbols))):
        raise ValueError("U2 Development expected symbols must be sorted and unique")
    if any(
        not isinstance(point, UniversalTradeRLU2PairedExcessPoint)
        for point in resolved_points
    ):
        raise TypeError("U2 Development panel contains an invalid point")

    identities = tuple(
        (
            point.source_window,
            point.decision_timestamp_ns,
            point.training_seed,
            point.concrete_symbol,
        )
        for point in resolved_points
    )
    if len(set(identities)) != len(identities):
        raise ValueError("U2 Development panel contains duplicate leaf identity")
    if {point.concrete_symbol for point in resolved_points} - set(symbols):
        raise ValueError("U2 Development panel contains an unexpected symbol")

    by_window_timestamp: dict[
        tuple[str, int],
        dict[int, dict[str, float]],
    ] = defaultdict(lambda: defaultdict(dict))
    for point in resolved_points:
        by_symbol = by_window_timestamp[
            (point.source_window, point.decision_timestamp_ns)
        ][point.training_seed]
        by_symbol[point.concrete_symbol] = point.candidate_minus_cash_net_log_excess

    segments: list[UniversalTradeRLU2ReducedBootstrapSegment] = []
    for source_window in U2_DEVELOPMENT_WINDOWS:
        timestamps = tuple(
            sorted(
                timestamp
                for window, timestamp in by_window_timestamp
                if window == source_window
            )
        )
        if not timestamps:
            raise ValueError("U2 Development panel requires both D1 and D2 segments")
        reduced: list[float] = []
        for timestamp in timestamps:
            seed_map = by_window_timestamp[(source_window, timestamp)]
            if set(seed_map) != set(U2_TRAINING_SEEDS):
                raise ValueError(
                    "U2 Development timestamp closure is incomplete across training seeds"
                )
            seed_means: list[float] = []
            for seed in U2_TRAINING_SEEDS:
                symbol_map = seed_map[seed]
                if set(symbol_map) != set(symbols):
                    raise ValueError(
                        "U2 Development timestamp closure is incomplete across symbols"
                    )
                seed_means.append(fmean(symbol_map[symbol] for symbol in symbols))
            reduced.append(float(median(seed_means)))
        segments.append(
            UniversalTradeRLU2ReducedBootstrapSegment(
                source_window=source_window,
                decision_timestamps_ns=timestamps,
                net_log_excess=tuple(reduced),
            )
        )

    return tuple(segments)


def bootstrap_universal_trade_rl_u2_development_panel(
    *,
    segments: tuple[UniversalTradeRLU2ReducedBootstrapSegment, ...],
) -> UniversalTradeRLU2DevelopmentBootstrapResult:
    """Bootstrap D1 and D2 independently so a block never crosses their boundary."""

    resolved = tuple(segments)
    if tuple(segment.source_window for segment in resolved) != U2_DEVELOPMENT_WINDOWS:
        raise ValueError("U2 Development bootstrap requires canonical D1/D2 segments")
    if any(
        not isinstance(segment, UniversalTradeRLU2ReducedBootstrapSegment)
        for segment in resolved
    ):
        raise TypeError("U2 Development bootstrap segment is invalid")

    block_lengths = tuple(
        min(
            len(segment.net_log_excess),
            math.ceil(math.sqrt(len(segment.net_log_excess))),
        )
        for segment in resolved
    )
    all_values = tuple(
        value for segment in resolved for value in segment.net_log_excess
    )
    observed_mean = float(fmean(all_values))

    rng = np.random.default_rng(_U2_BOOTSTRAP_SEED)
    means = np.empty(_U2_BOOTSTRAP_RESAMPLES, dtype=np.float64)
    for draw in range(_U2_BOOTSTRAP_RESAMPLES):
        sampled_values: list[float] = []
        for segment, block_length in zip(resolved, block_lengths, strict=True):
            values = segment.net_log_excess
            segment_length = len(values)
            sampled_indices: list[int] = []
            while len(sampled_indices) < segment_length:
                start = int(rng.integers(0, segment_length))
                sampled_indices.extend(
                    (start + offset) % segment_length for offset in range(block_length)
                )
            sampled_values.extend(
                values[index] for index in sampled_indices[:segment_length]
            )
        means[draw] = float(fmean(sampled_values))

    alpha = 1.0 - _U2_BOOTSTRAP_CONFIDENCE_LEVEL
    lower, upper = np.quantile(
        means,
        [alpha / 2.0, 1.0 - alpha / 2.0],
        method=_U2_BOOTSTRAP_QUANTILE_METHOD,
    )
    lower_ci = float(lower)
    upper_ci = float(upper)
    return UniversalTradeRLU2DevelopmentBootstrapResult(
        segment_digests=tuple(segment.digest for segment in resolved),
        block_lengths=block_lengths,
        observed_mean=observed_mean,
        lower_ci=lower_ci,
        upper_ci=upper_ci,
        passed=lower_ci > 0.0,
    )


__all__ = [
    "U2_DEVELOPMENT_BOOTSTRAP_SCHEMA",
    "U2_DEVELOPMENT_PANEL_SCHEMA",
    "U2_DEVELOPMENT_WINDOWS",
    "UniversalTradeRLU2DevelopmentBootstrapResult",
    "UniversalTradeRLU2PairedExcessPoint",
    "UniversalTradeRLU2ReducedBootstrapSegment",
    "bootstrap_universal_trade_rl_u2_development_panel",
    "reduce_universal_trade_rl_u2_development_panel",
]
