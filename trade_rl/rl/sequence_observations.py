"""Causal native-timeframe sequence observations for research policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from types import MappingProxyType
from typing import Mapping, Protocol
from weakref import WeakValueDictionary

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import timeframe_hours
from trade_rl.data.market import MarketDataset
from trade_rl.rl.observations import ObservationLayout

SEQUENCE_OBSERVATION_SCHEMA = "native_timeframe_sequence_observation_v2"
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


@dataclass(frozen=True, slots=True)
class _CompiledSequenceLayout:
    feature_indices: Mapping[str, tuple[int, ...]]
    columns: Mapping[str, np.ndarray]
    steps: Mapping[str, int]
    feature_names: Mapping[str, tuple[str, ...]]
    minimum_index: int
    layout_digest: str
    schema_digest: str


@lru_cache(maxsize=64)
def _compile_sequence_layout(
    windows: tuple[SequenceWindowSpec, ...],
    dataset_id: str,
    symbols: tuple[str, ...],
    feature_names: tuple[str, ...],
    bar_hours: float,
) -> _CompiledSequenceLayout:
    indices: dict[str, tuple[int, ...]] = {}
    columns: dict[str, np.ndarray] = {}
    steps: dict[str, int] = {}
    names: dict[str, tuple[str, ...]] = {}
    for window in windows:
        prefix = f"{window.timeframe}__"
        selected = tuple(
            index for index, name in enumerate(feature_names) if name.startswith(prefix)
        )
        if not selected:
            raise ValueError(
                f"dataset has no ordered features for timeframe {window.timeframe}"
            )
        ratio = timeframe_hours(window.timeframe) / bar_hours
        step = int(round(ratio))
        if step <= 0 or not math.isclose(
            ratio, float(step), rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"timeframe {window.timeframe} is not an integer multiple of "
                "the base clock"
            )
        selected_columns = np.asarray(selected, dtype=np.int64)
        selected_columns.setflags(write=False)
        indices[window.timeframe] = selected
        columns[window.timeframe] = selected_columns
        steps[window.timeframe] = step
        names[window.timeframe] = tuple(feature_names[index] for index in selected)
    layout_payload = {
        "schema_version": SEQUENCE_OBSERVATION_SCHEMA,
        "symbols": symbols,
        "base_bar_hours": bar_hours,
        "windows": tuple(
            {
                "timeframe": window.timeframe,
                "length": window.length,
                "step": steps[window.timeframe],
                "feature_names": names[window.timeframe],
            }
            for window in windows
        ),
        "value_dtype": "float32",
        "availability_dtype": "bool",
        "staleness_dtype": "float32",
    }
    return _CompiledSequenceLayout(
        feature_indices=MappingProxyType(indices),
        columns=MappingProxyType(columns),
        steps=MappingProxyType(steps),
        feature_names=MappingProxyType(names),
        minimum_index=max(
            steps[window.timeframe] * (window.length - 1) for window in windows
        ),
        layout_digest=content_digest(layout_payload),
        schema_digest=content_digest({**layout_payload, "dataset_id": dataset_id}),
    )


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

    def _compiled(self, dataset: MarketDataset) -> _CompiledSequenceLayout:
        return _compile_sequence_layout(
            self.windows,
            dataset.dataset_id,
            dataset.symbols,
            dataset.feature_names,
            dataset.bar_hours,
        )

    def _feature_indices(self, dataset: MarketDataset) -> dict[str, tuple[int, ...]]:
        return dict(self._compiled(dataset).feature_indices)

    def _step(self, dataset: MarketDataset, timeframe: str) -> int:
        return self._compiled(dataset).steps[timeframe]

    def minimum_index(self, dataset: MarketDataset) -> int:
        return self._compiled(dataset).minimum_index

    def layout_payload(self, dataset: MarketDataset) -> dict[str, object]:
        """Return the reusable ordered tensor contract without dataset identity."""

        compiled = self._compiled(dataset)
        return {
            "schema_version": SEQUENCE_OBSERVATION_SCHEMA,
            "symbols": dataset.symbols,
            "base_bar_hours": dataset.bar_hours,
            "windows": tuple(
                {
                    "timeframe": item.timeframe,
                    "length": item.length,
                    "step": compiled.steps[item.timeframe],
                    "feature_names": compiled.feature_names[item.timeframe],
                }
                for item in self.windows
            ),
            "value_dtype": "float32",
            "availability_dtype": "bool",
            "staleness_dtype": "float32",
        }

    def layout_digest(self, dataset: MarketDataset) -> str:
        return self._compiled(dataset).layout_digest

    def schema_payload(self, dataset: MarketDataset) -> dict[str, object]:
        return {
            **self.layout_payload(dataset),
            "dataset_id": dataset.dataset_id,
        }

    def schema_digest(self, dataset: MarketDataset) -> str:
        return self._compiled(dataset).schema_digest

    def build(self, dataset: MarketDataset, *, index: int) -> SequenceObservation:
        if isinstance(index, bool) or not isinstance(index, int):
            raise ValueError("sequence index must be an integer")
        if not 0 <= index < dataset.n_bars:
            raise ValueError("sequence index is outside the dataset")
        compiled = self._compiled(dataset)
        minimum = compiled.minimum_index
        if index < minimum:
            raise ValueError(
                f"sequence index {index} precedes required history {minimum}"
            )
        values: dict[str, np.ndarray] = {}
        available: dict[str, np.ndarray] = {}
        staleness: dict[str, np.ndarray] = {}
        source_indices: dict[str, np.ndarray] = {}
        names: dict[str, tuple[str, ...]] = {}
        staleness_hours = dataset.resolved_array("feature_staleness_hours")
        for window in self.windows:
            step = compiled.steps[window.timeframe]
            rows = index - step * np.arange(window.length - 1, -1, -1)
            if np.any(rows < 0) or np.any(rows > index):
                raise RuntimeError("sequence builder selected a non-causal source row")
            columns = compiled.columns[window.timeframe]
            # Dataset arrays are [time, symbol, feature].  The policy contract is
            # [symbol, native-time, feature].
            raw_values = dataset.features[rows][:, :, columns].transpose(1, 0, 2)
            raw_available = dataset.feature_available[rows][:, :, columns].transpose(
                1, 0, 2
            )
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
            names[window.timeframe] = compiled.feature_names[window.timeframe]
        return SequenceObservation(
            values=MappingProxyType(values),
            available=MappingProxyType(available),
            staleness=MappingProxyType(staleness),
            source_indices=MappingProxyType(source_indices),
            feature_names=MappingProxyType(names),
            schema_digest=compiled.schema_digest,
        )


@dataclass(frozen=True, slots=True, weakref_slot=True)
class SequencePolicyPlane:
    """Read-only normalized sequence channels shared by equivalent environments."""

    dataset_id: str
    layout_digest: str
    normalizer_digest: str
    windows: tuple[SequenceWindowSpec, ...]
    steps: Mapping[str, int]
    minimum_index: int
    n_bars: int
    values: Mapping[str, np.ndarray]
    available: Mapping[str, np.ndarray]
    staleness: Mapping[str, np.ndarray]

    def components(self, decision_index: int) -> dict[str, np.ndarray]:
        if isinstance(decision_index, bool) or not isinstance(decision_index, int):
            raise ValueError("sequence index must be an integer")
        if not self.minimum_index <= decision_index < self.n_bars:
            raise ValueError("sequence index is outside causal history")
        result: dict[str, np.ndarray] = {}
        for window in self.windows:
            timeframe = window.timeframe
            rows = decision_index - self.steps[timeframe] * np.arange(
                window.length - 1, -1, -1
            )
            result[f"sequence_{timeframe}_values"] = self.values[timeframe][
                rows
            ].transpose(1, 0, 2)
            result[f"sequence_{timeframe}_available"] = self.available[timeframe][
                rows
            ].transpose(1, 0, 2)
            result[f"sequence_{timeframe}_staleness"] = self.staleness[timeframe][
                rows
            ].transpose(1, 0, 2)
        return result


_SEQUENCE_POLICY_PLANE_CACHE: WeakValueDictionary[
    tuple[str, str, str], SequencePolicyPlane
] = WeakValueDictionary()


def build_sequence_policy_plane(
    dataset: MarketDataset,
    builder: SequenceObservationBuilder,
    sequence_normalizer: SequenceNormalizerProtocol | None,
) -> SequencePolicyPlane:
    """Build or reuse elementwise-normalized channels for causal window gathers."""

    compiled = builder._compiled(dataset)
    raw_normalizer_digest = getattr(sequence_normalizer, "digest", None)
    normalizer_digest = (
        "none"
        if sequence_normalizer is None
        else str(raw_normalizer_digest or f"object:{id(sequence_normalizer)}")
    )
    cache_key = (dataset.dataset_id, compiled.layout_digest, normalizer_digest)
    cached = _SEQUENCE_POLICY_PLANE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    values: dict[str, np.ndarray] = {}
    available: dict[str, np.ndarray] = {}
    staleness: dict[str, np.ndarray] = {}
    staleness_hours = dataset.resolved_array("feature_staleness_hours")
    for window in builder.windows:
        timeframe = window.timeframe
        columns = compiled.columns[timeframe]
        channel_values = dataset.features[:, :, columns]
        channel_available = dataset.feature_available[:, :, columns]
        normalized = sequence_policy_values(
            timeframe=timeframe,
            values=channel_values,
            available=channel_available,
            feature_names=compiled.feature_names[timeframe],
            sequence_normalizer=sequence_normalizer,
        )
        availability = np.asarray(channel_available, dtype=np.uint8)
        channel_staleness = np.asarray(
            np.clip(staleness_hours[:, :, columns], 0.0, _FLOAT16_MAX),
            dtype=np.float16,
        )
        for array in (normalized, availability, channel_staleness):
            array.setflags(write=False)
        values[timeframe] = normalized
        available[timeframe] = availability
        staleness[timeframe] = channel_staleness
    plane = SequencePolicyPlane(
        dataset_id=dataset.dataset_id,
        layout_digest=compiled.layout_digest,
        normalizer_digest=normalizer_digest,
        windows=builder.windows,
        steps=compiled.steps,
        minimum_index=compiled.minimum_index,
        n_bars=dataset.n_bars,
        values=MappingProxyType(values),
        available=MappingProxyType(available),
        staleness=MappingProxyType(staleness),
    )
    _SEQUENCE_POLICY_PLANE_CACHE[cache_key] = plane
    return plane


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


def build_structured_current_observation(
    *,
    current_flat: np.ndarray,
    layout: ObservationLayout,
    n_features: int,
) -> dict[str, np.ndarray]:
    """Split the current flat state into the structured policy components."""

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
    return result


def build_structured_policy_observation(
    *,
    sequence: SequenceObservation,
    current_flat: np.ndarray,
    layout: ObservationLayout,
    n_features: int,
    sequence_normalizer: SequenceNormalizerProtocol | None = None,
) -> dict[str, np.ndarray]:
    """Split the current flat state and append compact native-clock histories."""

    result = build_structured_current_observation(
        current_flat=current_flat,
        layout=layout,
        n_features=n_features,
    )
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
    "SequencePolicyPlane",
    "SequenceWindowSpec",
    "build_sequence_policy_plane",
    "build_structured_current_observation",
    "build_structured_policy_observation",
    "sequence_policy_values",
]
