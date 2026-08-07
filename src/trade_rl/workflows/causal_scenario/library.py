"""Frozen train-only historical block library for causal scenario matching."""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from numbers import Integral, Real
from types import MappingProxyType
from typing import Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.artifacts import MarketDatasetView
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.data.market import MarketDataset
from trade_rl.domain.common import require_sha256, require_unique_non_empty
from trade_rl.evaluation.causal_scenario_values import CausalScenarioSet
from trade_rl.strategies.trend import TrendStrategy
from trade_rl.workflows.causal_scenario.conditions import (
    CausalConditionConfig,
    CausalConditionLayout,
    TrainRobustConditionNormalizer,
    build_causal_condition_layout,
    compute_raw_causal_condition,
    fit_train_condition_normalizer,
)

_LIBRARY_CONFIG_SCHEMA: Final = "causal_scenario_library_config_v1"
_BLOCK_SCHEMA: Final = "relative_scenario_block_v1"
_LIBRARY_SCHEMA: Final = "frozen_causal_scenario_library_v1"
_SELECTION_SCHEMA: Final = "causal_scenario_selection_v1"
RELATIVE_SCENARIO_PRICE_FIELDS: Final = (
    "open",
    "high",
    "low",
    "close",
    "mark_price",
    "index_price",
    "dividend",
)
RELATIVE_SCENARIO_FUTURE_FIELDS: Final = (
    "features",
    "global_features",
    "funding_rate",
    "tradable",
    "feature_available",
    "funding_event_count",
    "feature_staleness_hours",
    "feature_missing_reason",
    "global_feature_available",
    "global_feature_staleness_hours",
    "global_feature_missing_reason",
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
    "borrow_available",
    "borrow_rate",
    "funding_due",
    "asset_active",
    "buy_allowed",
    "sell_allowed",
    "split_factor",
    "delisting_recovery",
    "cash_rate",
    "information_available",
    "feature_staleness",
    "availability_delay_ns",
)
_FEATURE_SHAPED_FIELDS: Final = frozenset(
    {
        "features",
        "feature_available",
        "feature_staleness_hours",
        "feature_missing_reason",
        "feature_staleness",
    }
)
_GLOBAL_SHAPED_FIELDS: Final = frozenset(
    {
        "global_features",
        "global_feature_available",
        "global_feature_staleness_hours",
        "global_feature_missing_reason",
    }
)
_SCALAR_SHAPED_FIELDS: Final = frozenset({"cash_rate"})


def _positive_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _non_negative_int(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _finite_positive(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{name} must be finite and positive")
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be finite and positive")
    return result


def _readonly_array(
    name: str,
    value: object,
    *,
    ndim: int | None = None,
    dtype: np.dtype[np.generic] | None = None,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=dtype).copy(order="C")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{name} must be an array") from error
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{name} must have rank {ndim}")
    if array.dtype.kind in "fc" and not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    if array.dtype.kind == "f":
        array[array == 0.0] = 0.0
    array.setflags(write=False)
    return array


def _array_payload(array: np.ndarray) -> dict[str, object]:
    return {
        "dtype": array.dtype.str,
        "shape": tuple(int(size) for size in array.shape),
        "values": array.tolist(),
    }


def _readonly_mapping(
    name: str,
    value: Mapping[str, np.ndarray],
    *,
    exact_keys: tuple[str, ...],
    horizon: int,
) -> Mapping[str, np.ndarray]:
    if set(value) != set(exact_keys):
        raise ValueError(f"{name} keys do not match the version-one contract")
    result: dict[str, np.ndarray] = {}
    for key in sorted(exact_keys):
        array = _readonly_array(f"{name}.{key}", value[key])
        if array.ndim == 0 or array.shape[0] != horizon:
            raise ValueError(f"{name}.{key} must begin with the horizon dimension")
        result[key] = array
    return MappingProxyType(result)


def _normal_payload(value: object) -> object:
    if isinstance(value, Enum):
        return value.value
    if dataclasses.is_dataclass(value):
        return {
            field.name: _normal_payload(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, tuple):
        return tuple(_normal_payload(item) for item in value)
    if isinstance(value, list):
        return [_normal_payload(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normal_payload(item) for key, item in sorted(value.items())}
    return value


def _trend_payload(strategy: TrendStrategy) -> dict[str, object]:
    payload = _normal_payload(strategy.config)
    if not isinstance(payload, dict):  # pragma: no cover - dataclass contract
        raise RuntimeError("trend configuration payload is invalid")
    return payload


@dataclass(frozen=True, slots=True)
class CausalScenarioLibraryConfig:
    horizon_decisions: int = 96
    scenario_count: int = 64
    relative_floor: float = 1e-12
    condition: CausalConditionConfig = CausalConditionConfig()
    schema_version: str = _LIBRARY_CONFIG_SCHEMA

    def __post_init__(self) -> None:
        horizon = _positive_int("horizon_decisions", self.horizon_decisions)
        count = _positive_int("scenario_count", self.scenario_count)
        if count != 64:
            raise ValueError("scenario_count must be exactly 64 in version one")
        floor = _finite_positive("relative_floor", self.relative_floor)
        if not isinstance(self.condition, CausalConditionConfig):
            raise ValueError("condition must be a CausalConditionConfig")
        if self.schema_version != _LIBRARY_CONFIG_SCHEMA:
            raise ValueError("unsupported causal scenario library config schema")
        object.__setattr__(self, "horizon_decisions", horizon)
        object.__setattr__(self, "scenario_count", count)
        object.__setattr__(self, "relative_floor", floor)

    @property
    def digest(self) -> str:
        return content_digest(
            {
                "condition_digest": self.condition.digest,
                "horizon_decisions": self.horizon_decisions,
                "relative_floor": self.relative_floor,
                "scenario_count": self.scenario_count,
                "schema_version": self.schema_version,
            }
        )


@dataclass(frozen=True, slots=True)
class RelativeScenarioBlock:
    anchor_index: int
    source_start: int
    source_stop: int
    elapsed_ns: np.ndarray
    raw_condition: np.ndarray
    normalized_condition: np.ndarray
    price_relatives: Mapping[str, np.ndarray]
    volume_relative: np.ndarray
    market_notional_relative: np.ndarray
    future_arrays: Mapping[str, np.ndarray]
    block_digest: str = ""
    schema_version: str = _BLOCK_SCHEMA

    def __post_init__(self) -> None:
        anchor = _non_negative_int("anchor_index", self.anchor_index)
        start = _non_negative_int("source_start", self.source_start)
        stop = _positive_int("source_stop", self.source_stop)
        if start != anchor + 1 or stop <= start:
            raise ValueError("source range must begin immediately after the anchor")
        horizon = stop - start
        elapsed = _readonly_array(
            "elapsed_ns", self.elapsed_ns, ndim=1, dtype=np.dtype(np.int64)
        )
        raw = _readonly_array(
            "raw_condition", self.raw_condition, ndim=1, dtype=np.dtype(np.float64)
        )
        normalized = _readonly_array(
            "normalized_condition",
            self.normalized_condition,
            ndim=1,
            dtype=np.dtype(np.float64),
        )
        if (
            elapsed.shape != (horizon,)
            or np.any(elapsed <= 0)
            or np.any(np.diff(elapsed) <= 0)
        ):
            raise ValueError(
                "elapsed_ns must be positive, increasing, and match the horizon"
            )
        if raw.size == 0 or normalized.shape != raw.shape:
            raise ValueError("condition vectors must be non-empty and shape matched")
        prices = _readonly_mapping(
            "price_relatives",
            self.price_relatives,
            exact_keys=RELATIVE_SCENARIO_PRICE_FIELDS,
            horizon=horizon,
        )
        volume = _readonly_array(
            "volume_relative", self.volume_relative, ndim=2, dtype=np.dtype(np.float64)
        )
        notional = _readonly_array(
            "market_notional_relative",
            self.market_notional_relative,
            ndim=2,
            dtype=np.dtype(np.float64),
        )
        if volume.shape[0] != horizon or notional.shape != volume.shape:
            raise ValueError(
                "volume and market-notional relatives must match the horizon"
            )
        if np.any(volume < 0.0) or np.any(notional < 0.0):
            raise ValueError("liquidity relatives must be non-negative")
        if any(
            array.shape != volume.shape
            for key, array in prices.items()
            if key != "dividend"
        ):
            raise ValueError("price relatives must match the liquidity shape")
        if prices["dividend"].shape != volume.shape:
            raise ValueError("dividend relatives must match the liquidity shape")
        future = _readonly_mapping(
            "future_arrays",
            self.future_arrays,
            exact_keys=RELATIVE_SCENARIO_FUTURE_FIELDS,
            horizon=horizon,
        )
        if self.schema_version != _BLOCK_SCHEMA:
            raise ValueError("unsupported relative scenario block schema")
        metadata = {
            "anchor_index": anchor,
            "schema_version": self.schema_version,
            "source_start": start,
            "source_stop": stop,
        }
        digest_arrays: list[tuple[str, np.ndarray]] = [
            ("elapsed_ns", elapsed),
            ("market_notional_relative", notional),
            ("normalized_condition", normalized),
            ("raw_condition", raw),
            ("volume_relative", volume),
        ]
        digest_arrays.extend(
            (f"price_relatives.{key}", prices[key]) for key in sorted(prices)
        )
        digest_arrays.extend(
            (f"future_arrays.{key}", future[key]) for key in sorted(future)
        )
        expected = content_and_arrays_digest(metadata, digest_arrays)
        if (
            self.block_digest
            and require_sha256(self.block_digest, field="block_digest") != expected
        ):
            raise ValueError("block_digest does not match relative scenario block")
        object.__setattr__(self, "anchor_index", anchor)
        object.__setattr__(self, "source_start", start)
        object.__setattr__(self, "source_stop", stop)
        object.__setattr__(self, "elapsed_ns", elapsed)
        object.__setattr__(self, "raw_condition", raw)
        object.__setattr__(self, "normalized_condition", normalized)
        object.__setattr__(self, "price_relatives", prices)
        object.__setattr__(self, "volume_relative", volume)
        object.__setattr__(self, "market_notional_relative", notional)
        object.__setattr__(self, "future_arrays", future)
        object.__setattr__(self, "block_digest", expected)

    @property
    def horizon_decisions(self) -> int:
        return self.source_stop - self.source_start


def _scenario_id(
    *,
    library_digest: str,
    query_index: int,
    rank: int,
    block: RelativeScenarioBlock,
) -> str:
    return content_digest(
        {
            "anchor_index": block.anchor_index,
            "block_digest": block.block_digest,
            "library_digest": library_digest,
            "query_index": query_index,
            "rank": rank,
            "schema_version": "causal_scenario_id_v1",
        }
    )


@dataclass(frozen=True, slots=True)
class FrozenCausalScenarioLibrary:
    dataset_id: str
    train_view_digest: str
    train_start: int
    train_stop: int
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    config: CausalScenarioLibraryConfig
    trend_config_payload: Mapping[str, object]
    layout: CausalConditionLayout
    normalizer: TrainRobustConditionNormalizer
    anchor_indices: np.ndarray
    raw_conditions: np.ndarray
    normalized_conditions: np.ndarray
    library_digest: str = ""
    schema_version: str = _LIBRARY_SCHEMA

    def __post_init__(self) -> None:
        dataset_id = require_sha256(self.dataset_id, field="dataset_id")
        view_digest = require_sha256(self.train_view_digest, field="train_view_digest")
        start = _non_negative_int("train_start", self.train_start)
        stop = _positive_int("train_stop", self.train_stop)
        if stop <= start:
            raise ValueError("train range must be non-empty")
        symbols = require_unique_non_empty(tuple(self.symbols), field="symbols")
        feature_names = require_unique_non_empty(
            tuple(self.feature_names), field="feature_names"
        )
        global_names = require_unique_non_empty(
            tuple(self.global_feature_names), field="global_feature_names"
        )
        if not isinstance(self.config, CausalScenarioLibraryConfig):
            raise ValueError("config must be a CausalScenarioLibraryConfig")
        if (
            not isinstance(self.layout, CausalConditionLayout)
            or self.layout.symbols != symbols
        ):
            raise ValueError("condition layout does not match library symbols")
        if (
            not isinstance(self.normalizer, TrainRobustConditionNormalizer)
            or self.normalizer.feature_names != self.layout.feature_names
            or not np.array_equal(
                self.normalizer.continuous_mask, self.layout.continuous_mask
            )
            or self.normalizer.train_view_digest != view_digest
        ):
            raise ValueError(
                "normalizer does not match the library layout or train view"
            )
        if not isinstance(self.trend_config_payload, Mapping):
            raise ValueError("trend_config_payload must be a mapping")
        anchors = _readonly_array(
            "anchor_indices", self.anchor_indices, ndim=1, dtype=np.dtype(np.int64)
        )
        raw = _readonly_array(
            "raw_conditions", self.raw_conditions, ndim=2, dtype=np.dtype(np.float64)
        )
        normalized = _readonly_array(
            "normalized_conditions",
            self.normalized_conditions,
            ndim=2,
            dtype=np.dtype(np.float64),
        )
        count = anchors.size
        width = len(self.layout.feature_names)
        if count < self.config.scenario_count:
            raise ValueError("library must contain at least 64 scenario anchors")
        if raw.shape != (count, width) or normalized.shape != raw.shape:
            raise ValueError("scenario anchor condition arrays have invalid shapes")
        if not np.array_equal(anchors, np.sort(np.unique(anchors))):
            raise ValueError("scenario anchors must be unique and ascending")
        source_stops = anchors + 1 + self.config.horizon_decisions
        if np.any(anchors < start) or np.any(source_stops > stop):
            raise ValueError("scenario anchor is outside the train library contract")
        expected_normalized = np.vstack([self.normalizer.transform(row) for row in raw])
        if not np.array_equal(normalized, expected_normalized):
            raise ValueError("scenario normalized conditions are inconsistent")
        if self.schema_version != _LIBRARY_SCHEMA:
            raise ValueError("unsupported frozen causal scenario library schema")
        trend_payload = MappingProxyType(
            dict(sorted(self.trend_config_payload.items()))
        )
        metadata = {
            "config_digest": self.config.digest,
            "dataset_id": dataset_id,
            "feature_names": feature_names,
            "global_feature_names": global_names,
            "layout_digest": self.layout.digest,
            "normalizer_digest": self.normalizer.digest,
            "schema_version": self.schema_version,
            "symbols": symbols,
            "train_start": start,
            "train_stop": stop,
            "train_view_digest": view_digest,
            "trend_config_payload": dict(trend_payload),
        }
        expected = content_and_arrays_digest(
            metadata,
            (
                ("anchor_indices", anchors),
                ("normalized_conditions", normalized),
                ("raw_conditions", raw),
            ),
        )
        if (
            self.library_digest
            and require_sha256(self.library_digest, field="library_digest") != expected
        ):
            raise ValueError("library_digest does not match frozen scenario library")
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "train_view_digest", view_digest)
        object.__setattr__(self, "train_start", start)
        object.__setattr__(self, "train_stop", stop)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "feature_names", feature_names)
        object.__setattr__(self, "global_feature_names", global_names)
        object.__setattr__(self, "trend_config_payload", trend_payload)
        object.__setattr__(self, "anchor_indices", anchors)
        object.__setattr__(self, "raw_conditions", raw)
        object.__setattr__(self, "normalized_conditions", normalized)
        object.__setattr__(self, "library_digest", expected)

    @property
    def anchor_count(self) -> int:
        return int(self.anchor_indices.size)


@dataclass(frozen=True, slots=True)
class CausalScenarioSelection:
    library_digest: str
    query_index: int
    query_timestamp_ns: int
    raw_query_condition: np.ndarray
    normalized_query_condition: np.ndarray
    scenario_set: CausalScenarioSet
    blocks: tuple[RelativeScenarioBlock, ...]
    selection_digest: str = ""
    schema_version: str = _SELECTION_SCHEMA

    def __post_init__(self) -> None:
        library_digest = require_sha256(self.library_digest, field="library_digest")
        index = _non_negative_int("query_index", self.query_index)
        timestamp = _positive_int("query_timestamp_ns", self.query_timestamp_ns)
        raw = _readonly_array(
            "raw_query_condition",
            self.raw_query_condition,
            ndim=1,
            dtype=np.dtype(np.float64),
        )
        normalized = _readonly_array(
            "normalized_query_condition",
            self.normalized_query_condition,
            ndim=1,
            dtype=np.dtype(np.float64),
        )
        if not isinstance(self.scenario_set, CausalScenarioSet):
            raise ValueError("scenario_set must be a CausalScenarioSet")
        blocks = tuple(self.blocks)
        if any(not isinstance(block, RelativeScenarioBlock) for block in blocks):
            raise ValueError("selection blocks must be RelativeScenarioBlock values")
        if (
            normalized.shape != raw.shape
            or len(blocks) != self.scenario_set.scenario_count
        ):
            raise ValueError("selection conditions or block count are invalid")
        if self.scenario_set.library_digest != library_digest:
            raise ValueError("scenario set library digest does not match selection")
        anchors = np.asarray([block.anchor_index for block in blocks], dtype=np.int64)
        conditions = np.vstack([block.normalized_condition for block in blocks])
        if not np.array_equal(self.scenario_set.anchor_indices, anchors):
            raise ValueError("scenario set anchors do not match selected blocks")
        if not np.array_equal(self.scenario_set.anchor_conditions, conditions):
            raise ValueError(
                "scenario set anchor conditions do not match selected blocks"
            )
        if not np.array_equal(self.scenario_set.query_condition, normalized):
            raise ValueError("scenario set query condition does not match selection")
        distances = np.sum((conditions - normalized) ** 2, axis=1)
        if not np.array_equal(self.scenario_set.distances, distances):
            raise ValueError("scenario set distances do not match selected blocks")
        expected_ids = tuple(
            _scenario_id(
                library_digest=library_digest,
                query_index=index,
                rank=rank,
                block=block,
            )
            for rank, block in enumerate(blocks)
        )
        if self.scenario_set.scenario_ids != expected_ids:
            raise ValueError("scenario IDs do not match selected blocks")
        if self.schema_version != _SELECTION_SCHEMA:
            raise ValueError("unsupported causal scenario selection schema")
        expected = content_digest(
            {
                "block_digests": tuple(block.block_digest for block in blocks),
                "library_digest": library_digest,
                "normalized_query_condition": _array_payload(normalized),
                "query_index": index,
                "query_timestamp_ns": timestamp,
                "raw_query_condition": _array_payload(raw),
                "scenario_set_digest": self.scenario_set.digest,
                "schema_version": self.schema_version,
            }
        )
        if (
            self.selection_digest
            and require_sha256(self.selection_digest, field="selection_digest")
            != expected
        ):
            raise ValueError(
                "selection_digest does not match causal scenario selection"
            )
        object.__setattr__(self, "library_digest", library_digest)
        object.__setattr__(self, "query_index", index)
        object.__setattr__(self, "query_timestamp_ns", timestamp)
        object.__setattr__(self, "raw_query_condition", raw)
        object.__setattr__(self, "normalized_query_condition", normalized)
        object.__setattr__(self, "blocks", blocks)
        object.__setattr__(self, "selection_digest", expected)


def _future_array(dataset: MarketDataset, field: str, rows: slice) -> np.ndarray:
    if field == "features":
        return dataset.features[rows]
    if field == "global_features":
        return dataset.global_features[rows]
    if field == "funding_rate":
        return dataset.funding_rate[rows]
    if field == "tradable":
        return dataset.tradable[rows]
    if field == "feature_available":
        return dataset.feature_available[rows]
    if field == "availability_delay_ns":
        timestamps = dataset.timestamps[rows].astype("datetime64[ns]").astype(np.int64)
        available = (
            dataset.resolved_array("available_at")[rows]
            .astype("datetime64[ns]")
            .astype(np.int64)
        )
        return available - timestamps[:, None]
    return dataset.resolved_array(field)[rows]


def _extract_block(
    dataset: MarketDataset,
    *,
    anchor: int,
    horizon: int,
    raw_condition: np.ndarray,
    normalized_condition: np.ndarray,
    relative_floor: float,
) -> RelativeScenarioBlock:
    start = anchor + 1
    stop = start + horizon
    rows = slice(start, stop)
    timestamp_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    elapsed = timestamp_ns[rows] - timestamp_ns[anchor]
    close_anchor = np.maximum(dataset.close[anchor], relative_floor)
    mark_anchor = np.maximum(
        dataset.resolved_array("mark_price")[anchor], relative_floor
    )
    index_anchor = np.maximum(
        dataset.resolved_array("index_price")[anchor], relative_floor
    )
    price_relatives = {
        "open": dataset.open[rows] / close_anchor,
        "high": dataset.high[rows] / close_anchor,
        "low": dataset.low[rows] / close_anchor,
        "close": dataset.close[rows] / close_anchor,
        "mark_price": dataset.resolved_array("mark_price")[rows] / mark_anchor,
        "index_price": dataset.resolved_array("index_price")[rows] / index_anchor,
        "dividend": dataset.resolved_array("dividend")[rows] / close_anchor,
    }
    volume_anchor = np.maximum(dataset.volume[anchor], relative_floor)
    volume_relative = dataset.volume[rows] / volume_anchor
    notional_anchor = np.maximum(
        dataset.market_notional(anchor, prices=dataset.close[anchor]),
        relative_floor,
    )
    market_notional_relative = (
        np.vstack(
            [
                dataset.market_notional(index, prices=dataset.close[index])
                for index in range(start, stop)
            ]
        )
        / notional_anchor
    )
    future_arrays = {
        field: _future_array(dataset, field, rows)
        for field in RELATIVE_SCENARIO_FUTURE_FIELDS
    }
    return RelativeScenarioBlock(
        anchor_index=anchor,
        source_start=start,
        source_stop=stop,
        elapsed_ns=elapsed,
        raw_condition=raw_condition,
        normalized_condition=normalized_condition,
        price_relatives=price_relatives,
        volume_relative=volume_relative,
        market_notional_relative=market_notional_relative,
        future_arrays=future_arrays,
    )


def _minimum_history_index_from(
    dataset: MarketDataset,
    *,
    start: int,
    hours: float,
) -> int:
    timestamp_ns = dataset.timestamps.astype("datetime64[ns]").astype(np.int64)
    target_ns = timestamp_ns[start] + int(round(hours * 3_600_000_000_000))
    result = int(np.searchsorted(timestamp_ns, target_ns, side="left"))
    if result >= dataset.n_bars:
        raise ValueError("assigned range is too short for causal condition history")
    return result


def _first_train_anchor(
    train_view: MarketDatasetView,
    trend_strategy: TrendStrategy,
    config: CausalScenarioLibraryConfig,
) -> int:
    dataset = train_view.dataset
    trend = trend_strategy.config
    if trend.slow_lookback is not None:
        trend_minimum = train_view.start + trend.slow_lookback
    else:
        trend_minimum = _minimum_history_index_from(
            dataset, start=train_view.start, hours=trend.slow_hours
        )
    correlation_minimum = _minimum_history_index_from(
        dataset,
        start=train_view.start,
        hours=config.condition.correlation_hours,
    )
    return max(train_view.start, trend_minimum, correlation_minimum)


def build_causal_scenario_library(
    train_view: MarketDatasetView,
    trend_strategy: TrendStrategy,
    config: CausalScenarioLibraryConfig | None = None,
) -> FrozenCausalScenarioLibrary:
    resolved = config or CausalScenarioLibraryConfig()
    dataset = train_view.dataset
    first_anchor = _first_train_anchor(train_view, trend_strategy, resolved)
    last_anchor_exclusive = train_view.stop - resolved.horizon_decisions
    anchors = tuple(range(first_anchor, last_anchor_exclusive))
    if len(anchors) < resolved.scenario_count:
        raise ValueError(
            "train view does not contain at least 64 complete scenario blocks"
        )
    layout = build_causal_condition_layout(dataset.symbols)
    raw_matrix = np.vstack(
        [
            compute_raw_causal_condition(
                dataset,
                anchor,
                trend_strategy,
                resolved.condition,
                history_start=train_view.start,
            )
            for anchor in anchors
        ]
    )
    normalizer = fit_train_condition_normalizer(
        raw_matrix,
        layout,
        train_view.identity,
        scale_epsilon=resolved.condition.scale_epsilon,
    )
    normalized_matrix = np.vstack([normalizer.transform(row) for row in raw_matrix])
    return FrozenCausalScenarioLibrary(
        dataset_id=dataset.dataset_id,
        train_view_digest=train_view.identity,
        train_start=train_view.start,
        train_stop=train_view.stop,
        symbols=dataset.symbols,
        feature_names=dataset.feature_names,
        global_feature_names=dataset.global_feature_names,
        config=resolved,
        trend_config_payload=_trend_payload(trend_strategy),
        layout=layout,
        normalizer=normalizer,
        anchor_indices=np.asarray(anchors, dtype=np.int64),
        raw_conditions=raw_matrix,
        normalized_conditions=normalized_matrix,
    )


def select_causal_scenarios(
    library: FrozenCausalScenarioLibrary,
    query_view: MarketDatasetView,
    query_index: int,
    trend_strategy: TrendStrategy,
) -> CausalScenarioSelection:
    dataset = query_view.dataset
    if dataset.dataset_id != library.dataset_id or dataset.symbols != library.symbols:
        raise ValueError("query dataset identity does not match scenario library")
    if (
        dataset.feature_names != library.feature_names
        or dataset.global_feature_names != library.global_feature_names
    ):
        raise ValueError("query feature schema does not match scenario library")
    if (
        query_view.stop != query_index + 1
        or not query_view.start <= query_index < query_view.stop
    ):
        raise ValueError("query view must be a causal prefix ending at query_index")
    if _trend_payload(trend_strategy) != dict(library.trend_config_payload):
        raise ValueError("query trend configuration does not match scenario library")
    raw = compute_raw_causal_condition(
        dataset,
        query_index,
        trend_strategy,
        library.config.condition,
        history_start=query_view.start,
    )
    normalized = library.normalizer.transform(raw)
    cutoff = query_index
    if library.train_start <= query_index < library.train_stop:
        cutoff = query_index - library.config.horizon_decisions
    source_stops = library.anchor_indices + 1 + library.config.horizon_decisions
    eligible_positions = np.flatnonzero(source_stops <= cutoff)
    if eligible_positions.size < library.config.scenario_count:
        raise ValueError("query has fewer than 64 eligible past scenario blocks")
    ranked = sorted(
        (
            (
                float(
                    np.sum((library.normalized_conditions[position] - normalized) ** 2)
                ),
                int(library.anchor_indices[position]),
                int(position),
            )
            for position in eligible_positions
        ),
        key=lambda item: (item[0], item[1]),
    )[: library.config.scenario_count]
    distances = np.asarray([item[0] for item in ranked], dtype=np.float64)
    anchors = np.asarray([item[1] for item in ranked], dtype=np.int64)
    positions = tuple(item[2] for item in ranked)
    blocks = tuple(
        _extract_block(
            dataset,
            anchor=int(library.anchor_indices[position]),
            horizon=library.config.horizon_decisions,
            raw_condition=library.raw_conditions[position],
            normalized_condition=library.normalized_conditions[position],
            relative_floor=library.config.relative_floor,
        )
        for position in positions
    )
    ids = tuple(
        _scenario_id(
            library_digest=library.library_digest,
            query_index=query_index,
            rank=rank,
            block=block,
        )
        for rank, block in enumerate(blocks)
    )
    scenario_set = CausalScenarioSet(
        scenario_ids=ids,
        probabilities=np.full(len(blocks), 1.0 / len(blocks), dtype=np.float64),
        anchor_indices=anchors,
        distances=distances,
        query_condition=normalized,
        anchor_conditions=np.vstack([block.normalized_condition for block in blocks]),
        library_digest=library.library_digest,
    )
    timestamp = int(
        dataset.timestamps[query_index].astype("datetime64[ns]").astype(np.int64)
    )
    return CausalScenarioSelection(
        library_digest=library.library_digest,
        query_index=query_index,
        query_timestamp_ns=timestamp,
        raw_query_condition=raw,
        normalized_query_condition=normalized,
        scenario_set=scenario_set,
        blocks=blocks,
    )


__all__ = [
    "CausalScenarioLibraryConfig",
    "CausalScenarioSelection",
    "FrozenCausalScenarioLibrary",
    "RelativeScenarioBlock",
    "RELATIVE_SCENARIO_FUTURE_FIELDS",
    "RELATIVE_SCENARIO_PRICE_FIELDS",
    "build_causal_scenario_library",
    "select_causal_scenarios",
]
