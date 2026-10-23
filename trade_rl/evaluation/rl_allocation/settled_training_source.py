"""Freeze declared settled inputs; this envelope is not consumed-source proof.

Account processing uses every native row from start through settlement stop,
inclusive. Actor features/forecast/cost declarations cover only the actor prefix.
No learner/backend, legacy environment or terminal evidence owner is imported.
Call again before reset/step, after native returns, and before fit publication;
compare the entire receipt to the constructor's detached envelope.
"""

from __future__ import annotations

from collections.abc import Mapping
from hashlib import sha256
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.contracts import MarketCalendarKind
from trade_rl.strategies.dataset_scope import validated_training_scope

if TYPE_CHECKING:
    from trade_rl.data.market import MarketDataset
    from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
    from trade_rl.evaluation.rl_allocation.settled_binding import (
        BoundSettledAllocationClock,
    )
    from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream


class _SettledSourceContext(Protocol):
    """Read-only fields consumed by the existing source envelope projection."""

    @property
    def dataset(self) -> MarketDataset: ...

    @property
    def stream(self) -> FrozenSimpleReturnStream: ...

    @property
    def bound(self) -> BoundSettledAllocationClock: ...

    @property
    def _estimates(self) -> Mapping[int, HorizonCostEstimates]: ...

    @property
    def account_id(self) -> str: ...

    @property
    def start_index(self) -> int: ...

    @property
    def actor_stop_index(self) -> int: ...

    @property
    def settlement_stop_index(self) -> int: ...

    @property
    def symbol_index(self) -> int: ...

    @property
    def feature_indices(self) -> tuple[int, ...]: ...

    @property
    def expected_horizon_seconds(self) -> int: ...

    @property
    def recipe_digest(self) -> str: ...


_FEATURE_FIELDS = frozenset(
    {
        "features",
        "feature_available",
        "feature_staleness",
        "feature_staleness_hours",
        "feature_missing_reason",
        "global_features",
        "global_feature_available",
        "global_feature_staleness_hours",
        "global_feature_missing_reason",
    }
)
_REQUIRED_FINANCIAL_FIELDS = frozenset(
    {
        "timestamps",
        "available_at",
        "information_available",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "funding_rate",
        "funding_price_rate",
        "funding_event_count",
        "tradable",
        "symbol_active",
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
        "buy_allowed",
        "sell_allowed",
        "mark_price",
        "index_price",
        "dividend",
        "split_factor",
        "delisting_recovery",
        "cash_rate",
        "contract_multipliers",
    }
)
_ACTOR_FEATURE_FIELDS = (
    "features",
    "feature_available",
    "feature_staleness",
    "feature_staleness_hours",
    "feature_missing_reason",
)


def _time_ns(value: np.datetime64) -> int:
    if not isinstance(value, np.datetime64) or np.isnat(value):
        raise ValueError("settled source requires actual finite Dataset clocks")
    return int(value.astype("datetime64[ns]").astype(np.int64))


def _bytes_receipt(value: np.ndarray) -> dict[str, Any]:
    """Pin exact C-order bytes with dtype (including units/endian) and shape."""
    if (
        not isinstance(value, np.ndarray)
        or value.dtype.hasobject
        or value.dtype.fields is not None
        or value.dtype.subdtype is not None
    ):
        raise ValueError("settled source arrays must have a plain non-object dtype")
    raw = value.tobytes(order="C")
    return {
        "dtype": value.dtype.str,
        "shape": list(value.shape),
        "order": "C",
        "byte_count": len(raw),
        "bytes_sha256": sha256(raw).hexdigest(),
    }


def _selected_features(
    dataset: MarketDataset,
    arrays: dict[str, np.ndarray],
    rows: list[int],
    symbol_index: int,
    feature_indices: tuple[int, ...],
) -> dict[str, Any]:
    indices, _ = validated_training_scope(
        dataset, feature_indices=feature_indices, fit_symbol_indices=(symbol_index,)
    )
    selected = {}
    for name in _ACTOR_FEATURE_FIELDS:
        value = arrays[name]
        if value.shape != (dataset.n_bars, dataset.n_symbols, dataset.n_features):
            raise ValueError("settled actor feature metadata shape changed")
        selected[name] = _bytes_receipt(
            value[np.asarray(rows, dtype=np.intp), symbol_index][:, list(indices)]
        )
    return {
        "actor_indices": rows,
        "symbol_index": symbol_index,
        "feature_indices": list(indices),
        "feature_names": [dataset.feature_names[i] for i in indices],
        "arrays": selected,
    }


def settled_training_source_receipt(
    env: _SettledSourceContext,
) -> dict[str, Any]:
    """Return a fresh declaration; attach actual consumption evidence elsewhere."""
    dataset = env.dataset
    start, actor_stop, settlement_stop = (
        env.start_index,
        env.actor_stop_index,
        env.settlement_stop_index,
    )
    if (
        any(type(i) is not int for i in (start, actor_stop, settlement_stop))
        or not 0 <= start < actor_stop < settlement_stop < dataset.n_bars
        or type(env.symbol_index) is not int
        or not 0 <= env.symbol_index < dataset.n_symbols
    ):
        raise ValueError("settled source requires actor and native Dataset windows")
    actor_rows = list(range(start, actor_stop))
    physical_rows = list(range(start, settlement_stop + 1))
    arrays = dataset.identity_arrays()
    financial_names = set(arrays) - _FEATURE_FIELDS
    if not _REQUIRED_FINANCIAL_FIELDS <= financial_names:
        raise ValueError("settled Dataset identity lacks required native source arrays")
    # Do not select only the policy symbol here. All book marks/carry, market
    # activity, margins and both capacity modes can read other account symbols.
    financial = {}
    for name in sorted(financial_names):
        value = arrays[name]
        if name == "contract_multipliers":
            if value.shape != (dataset.n_symbols,):
                raise ValueError("settled contract multipliers are not row arrays")
            chosen = value
        else:
            if value.shape not in (
                (dataset.n_bars,),
                (dataset.n_bars, dataset.n_symbols),
            ):
                raise ValueError("settled native identity array has an unknown shape")
            chosen = value[start : settlement_stop + 1]
        financial[name] = _bytes_receipt(chosen)
    actor_features = _selected_features(
        dataset, arrays, actor_rows, env.symbol_index, env.feature_indices
    )
    symbol = dataset.symbols[env.symbol_index]
    times = [_time_ns(dataset.timestamps[i]) for i in actor_rows]
    if env.stream.dataset_id != dataset.dataset_id:
        raise ValueError("settled actor forecast Dataset lineage changed")
    actor_times = set(times)
    matching = [
        packet
        for packet in env.stream.packets
        if packet.symbol == symbol and _time_ns(packet.as_of) in actor_times
    ]
    by_time = {_time_ns(packet.as_of): packet for packet in matching}
    if len(matching) != len(times) or set(by_time) != actor_times:
        raise ValueError("settled forecasts must cover each actor clock exactly once")
    used = {packet.vintage_digest for packet in matching}
    owners = [vintage for vintage in env.stream.vintages if vintage.digest in used]
    by_owner = {vintage.digest: vintage for vintage in owners}
    if len(owners) != len(used) or set(by_owner) != used:
        raise ValueError("settled actor packets require unique frozen owners")
    if set(env._estimates) != actor_times:
        raise ValueError("settled cost declarations must cover only actor clocks")
    packets, forecast_features, costs = [], [], []
    for index, time in zip(actor_rows, times):
        packet = by_time[time]
        owner = by_owner[packet.vintage_digest]
        estimate = env._estimates[time]
        if (
            packet.horizon_seconds != env.expected_horizon_seconds
            or _time_ns(packet.forecast_available_at) != time
            or owner.block.inference_delay_seconds != 0
            or estimate.symbol != symbol
            or _time_ns(estimate.decision_time) != time
            or _time_ns(estimate.horizon_end) != _time_ns(packet.horizon_end)
            or _time_ns(estimate.available_at) > time
        ):
            raise ValueError("settled actor forecast/cost clocks or horizons differ")
        packets.append(
            {
                "actor_index": index,
                "decision_time_ns": time,
                "packet_digest": content_digest(packet.payload()),
                "vintage_digest": packet.vintage_digest,
            }
        )
        # Native forecast preparation reads its model's selected features;
        # they need not be the same selection as the PPO observation features.
        forecast_features.append(
            {
                "vintage_digest": packet.vintage_digest,
                **_selected_features(
                    dataset,
                    arrays,
                    [index],
                    env.symbol_index,
                    owner.model.feature_indices,
                ),
            }
        )
        costs.append(
            {
                "actor_index": index,
                "decision_time_ns": time,
                "payload_digest": content_digest(estimate.payload()),
            }
        )
    descriptor = {
        "dataset_id": dataset.dataset_id,
        "n_bars": dataset.n_bars,
        "symbols": list(dataset.symbols),
        "volume_units": [unit.value for unit in dataset.volume_units],
        "feature_names": list(dataset.feature_names),
        "global_feature_names": list(dataset.global_feature_names),
        "feature_config_digest": dataset.feature_config_digest,
        "normalization_digest": dataset.normalization_digest,
        "calendar_kind": MarketCalendarKind(dataset.calendar_kind).value,
        "bar_hours": dataset.bar_hours,
        "periods_per_year": dataset.periods_per_year,
    }
    return {
        "schema": "allocation_settled_training_source_envelope_v1",
        "evidence_kind": "declared_prefit_envelope",
        "consumption_proven": False,
        "account_id": env.account_id,
        "bound_digest": env.bound.digest,
        "recipe_digest": env.recipe_digest,
        "dataset_descriptor": descriptor,
        "actor_scope": {
            "start_index": start,
            "actor_stop_index": actor_stop,
            "symbol_index": env.symbol_index,
            "actor_indices": actor_rows,
            "decision_times_ns": times,
            "expected_horizon_seconds": env.expected_horizon_seconds,
        },
        "native_scope": {
            "inclusive_start_index": start,
            "inclusive_settlement_stop_index": settlement_stop,
            "physical_source_indices": physical_rows,
            "processing_indices": list(range(start + 1, settlement_stop + 1)),
            "symbol_indices": list(range(dataset.n_symbols)),
            "calendar_kind": descriptor["calendar_kind"],
            "bar_hours": dataset.bar_hours,
            "initial_time_ns": _time_ns(dataset.timestamps[start]),
            "actor_stop_time_ns": _time_ns(dataset.timestamps[actor_stop]),
            "settlement_stop_time_ns": _time_ns(dataset.timestamps[settlement_stop]),
        },
        "native_financial_arrays": financial,
        "actor_selected_features": actor_features,
        "actor_forecast_selected_features": forecast_features,
        "actor_forecasts": packets,
        "actor_used_vintages": [
            {
                "vintage_digest": key,
                "payload_digest": content_digest(by_owner[key].payload()),
            }
            for key in sorted(used)
        ],
        "actor_cost_declarations": costs,
    }


__all__ = ["settled_training_source_receipt"]
