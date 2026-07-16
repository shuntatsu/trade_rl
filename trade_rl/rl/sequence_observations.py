"""Causal native-timeframe sequence observations for research policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import timeframe_hours
from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import ObservationLayout

SEQUENCE_OBSERVATION_SCHEMA = "native_timeframe_sequence_observation_v1"
_FLOAT16_MAX = float(np.finfo(np.float16).max)


class SequenceNormalizerProtocol(Protocol):
    def transform(
        self,
        timeframe: str,
        values: np.ndarray,
        available: np.ndarray,
        *,
        feature_names: tuple[str, ...],
    ) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SequenceWindowSpec:
    """One native clock and the number of completed observations to expose."""

    timeframe: str
    length: int

    def __post_init__(self) -> None:
        timeframe_hours(self.timeframe)
        if isinstance(self.length, bool) or not isinstance(self.length, int):
            raise ValueError("sequence length must be an integer")
        if self.length <= 0:
            raise ValueError("sequence length must be positive")


DEFAULT_SEQUENCE_WINDOWS = (
    SequenceWindowSpec("15m", 96),
    SequenceWindowSpec("1h", 168),
    SequenceWindowSpec("4h", 120),
    SequenceWindowSpec("1d", 60),
)


@dataclass(frozen=True, slots=True)
class SequenceObservation:
    """Immutable structured market history ending at one causal decision index."""

    values: Mapping[str, np.ndarray]
    available: Mapping[str, np.ndarray]
    staleness: Mapping[str, np.ndarray]
    source_indices: Mapping[str, np.ndarray]
    feature_names: Mapping[str, tuple[str, ...]]
    schema_digest: str


@dataclass(frozen=True, slots=True)
class SequenceObservationBuilder:
    """Build per-timeframe windows from causally aligned dataset features.

    The market dataset already aligns native feature events to the base decision
    clock using ``available_at``.  This builder samples that aligned state once
    per native period.  Every selected source index is therefore less than or
    equal to the current decision index, while delayed native observations stay
    delayed (and may legitimately repeat) rather than being backfilled.
    """

    windows: tuple[SequenceWindowSpec, ...] = DEFAULT_SEQUENCE_WINDOWS

    def __post_init__(self) -> None:
        if not self.windows:
            raise ValueError("sequence windows must not be empty")
        clocks = tuple(item.timeframe for item in self.windows)
        if len(set(clocks)) != len(clocks):
            raise ValueError("sequence timeframes must be unique")

    def _feature_indices(self, dataset: MarketDataset) -> dict[str, tuple[int, ...]]:
        result: dict[str, tuple[int, ...]] = {}
        for window in self.windows:
            prefix = f"{window.timeframe}__"
            indices = tuple(
                index
                for index, name in enumerate(dataset.feature_names)
                if name.startswith(prefix)
            )
            if not indices:
                raise ValueError(
                    f"dataset has no ordered features for timeframe {window.timeframe}"
                )
            result[window.timeframe] = indices
        return result

    def _step(self, dataset: MarketDataset, timeframe: str) -> int:
        ratio = timeframe_hours(timeframe) / dataset.bar_hours
        rounded = int(round(ratio))
        if rounded <= 0 or not math.isclose(
            ratio, float(rounded), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"timeframe {timeframe} is not an integer multiple of the base clock"
            )
        return rounded

    def minimum_index(self, dataset: MarketDataset) -> int:
        return max(
            self._step(dataset, item.timeframe) * (item.length - 1)
            for item in self.windows
        )

    def layout_payload(self, dataset: MarketDataset) -> dict[str, object]:
        """Return the reusable ordered tensor contract without dataset identity."""

        indices = self._feature_indices(dataset)
        return {
            "schema_version": SEQUENCE_OBSERVATION_SCHEMA,
            "symbols": dataset.symbols,
            "base_bar_hours": dataset.bar_hours,
            "windows": tuple(
                {
                    "timeframe": item.timeframe,
                    "length": item.length,
                    "step": self._step(dataset, item.timeframe),
                    "feature_names": tuple(
                        dataset.feature_names[index]
                        for index in indices[item.timeframe]
                    ),
                }
                for item in self.windows
            ),
            "value_dtype": "float32",
            "availability_dtype": "bool",
            "staleness_dtype": "float32",
        }

    def layout_digest(self, dataset: MarketDataset) -> str:
        return content_digest(self.layout_payload(dataset))

    def schema_payload(self, dataset: MarketDataset) -> dict[str, object]:
        return {
            **self.layout_payload(dataset),
            "dataset_id": dataset.dataset_id,
        }

    def schema_digest(self, dataset: MarketDataset) -> str:
        return content_digest(self.schema_payload(dataset))

    def build(self, dataset: MarketDataset, *, index: int) -> SequenceObservation:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("sequence index must be an integer")
        if not 0 <= index < dataset.n_bars:
            raise ValueError("sequence index is outside the dataset")
        minimum = self.minimum_index(dataset)
        if index < minimum:
            raise ValueError(
                f"sequence index {index} precedes required history {minimum}"
            )
        feature_indices = self._feature_indices(dataset)
        values: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        staleness: dict[str, np.ndarray] = {}
        source_indices: dict[str, np.ndarray] = {}
        names: dict[str, tuple[str, ...]] = {}
        for window in self.windows:
            step = self._step(dataset, window.timeframe)
            rows = index - step * np.arange(window.length - 1, -1, -1)
            if np.any(rows < 0) or np.any(rows > index):
                raise RuntimeError("sequence builder selected a non-causal source row")
            columns = np.asarray(feature_indices[window.timeframe], dtype=np.int64)
            # Dataset arrays are [time, symbol, feature].  The policy contract is
            # [symbol, native-time, feature].
            raw_values = dataset.features[rows][:, :, columns].transpose(1, 0, 2)
            raw_available = dataset.feature_available[rows][:, :, columns].transpose(
                1, 0, 2
            )
            staleness_hours = dataset.resolved_array("feature_staleness_hours")
            raw_staleness = staleness_hours[rows][:, :, columns].transpose(1, 0, 2)
            if (
                not np.isfinite(raw_values).all()
                or not np.isfinite(raw_staleness).all()
            ):
                raise ValueError("sequence observation contains non-finite values")
            if np.any(raw_staleness < 0.0):
                raise ValueError("sequence staleness must be non-negative")
            values[window.timeframe] = np.asarray(raw_values, dtype=np.float32)
            available[window.timeframe] = np.asarray(raw_available, dtype=np.bool_)
            staleness[window.timeframe] = np.asarray(raw_staleness, dtype=np.float32)
            source_indices[window.timeframe] = np.asarray(rows, dtype=np.int64)
            names[window.timeframe] = tuple(
                dataset.feature_names[column] for column in columns
            )
        return SequenceObservation(
            values=MappingProxyType(values),
            available=MappingProxyType(available),
            staleness=MappingProxyType(staleness),
            source_indices=MappingProxyType(source_indices),
            feature_names=MappingProxyType(names),
            schema_digest=self.schema_digest(dataset),
        )


def sequence_policy_values(
    *,
    timeframe: str,
    values: np.ndarray,
    available: np.ndarray,
    feature_names: tuple[str, ...],
    sequence_normalizer: SequenceNormalizerProtocol | None = None,
) -> np.ndarray:
    """Return the exact finite float16 sequence tensor consumed by the policy."""

    raw_values = np.asarray(values, dtype=np.float32)
    mask = np.asarray(available, dtype=np.bool_)
    if raw_values.shape != mask.shape:
        raise ValueError("sequence values and availability shapes differ")
    if sequence_normalizer is None:
        normalized = np.where(mask, raw_values, 0.0).astype(np.float32, copy=False)
    else:
        normalized = sequence_normalizer.transform(
            timeframe,
            raw_values,
            mask,
            feature_names=feature_names,
        )
    finite = np.nan_to_num(
        np.asarray(normalized, dtype=np.float32),
        nan=0.0,
        posinf=_FLOAT16_MAX,
        neginf=-_FLOAT16_MAX,
    )
    return np.asarray(
        np.clip(finite, -_FLOAT16_MAX, _FLOAT16_MAX),
        dtype=np.float16,
    )


def build_structured_policy_observation(
    *,
    sequence: SequenceObservation,
    current_flat: np.ndarray,
    layout: ObservationLayout,
    n_features: int,
    sequence_normalizer: SequenceNormalizerProtocol | None = None,
) -> dict[str, np.ndarray]:
    """Split the current flat state and append compact native-clock histories."""

    flat = np.asarray(current_flat, dtype=np.float32).reshape(-1)
    if flat.shape != (layout.size,):
        raise ValueError("current observation does not match the declared layout")
    if n_features <= 0 or 4 * n_features > layout.per_symbol_width:
        raise ValueError("feature width is incompatible with the observation layout")
    asset_stop = layout.n_symbols * layout.per_symbol_width
    per_asset = flat[:asset_stop].reshape(layout.n_symbols, layout.per_symbol_width)
    snapshot_width = 4 * n_features
    result: dict[str, np.ndarray] = {
        "current_snapshot": np.asarray(per_asset[:, :snapshot_width], dtype=np.float32),
        "asset_state": np.asarray(per_asset[:, snapshot_width:], dtype=np.float32),
        "global_state": np.asarray(flat[asset_stop:], dtype=np.float32),
        "active": np.asarray(per_asset[:, snapshot_width], dtype=np.float32),
    }
    for timeframe in sequence.values:
        result[f"sequence_{timeframe}_values"] = sequence_policy_values(
            timeframe=timeframe,
            values=sequence.values[timeframe],
            available=sequence.available[timeframe],
            feature_names=sequence.feature_names[timeframe],
            sequence_normalizer=sequence_normalizer,
        )
        result[f"sequence_{timeframe}_available"] = np.asarray(
            sequence.available[timeframe], dtype=np.uint8
        )
        finite_staleness = np.clip(sequence.staleness[timeframe], 0.0, _FLOAT16_MAX)
        result[f"sequence_{timeframe}_staleness"] = np.asarray(
            finite_staleness, dtype=np.float16
        )
    return result


__all__ = [
    "DEFAULT_SEQUENCE_WINDOWS",
    "SEQUENCE_OBSERVATION_SCHEMA",
    "SequenceNormalizerProtocol",
    "SequenceObservation",
    "SequenceObservationBuilder",
    "SequenceWindowSpec",
    "build_structured_policy_observation",
    "sequence_policy_values",
]
