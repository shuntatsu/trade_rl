"""Receipts of the declared finite episode actually consumed by allocation PPO."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.rl_allocation.env import AllocationTradingEnv
from trade_rl.strategies.dataset_scope import validated_training_scope

_EXECUTION_FIELDS = (
    "open",
    "high",
    "low",
    "close",
    "volume",
    "mark_price",
    "index_price",
    "funding_rate",
    "funding_event_count",
    "funding_due",
    "borrow_rate",
    "borrow_available",
    "dividend",
    "split_factor",
    "delisting_recovery",
    "cash_rate",
    "tradable",
    "symbol_active",
    "available_at",
    "information_available",
    "buy_allowed",
    "sell_allowed",
    "fee_rate",
    "maker_fee_rate",
    "taker_fee_rate",
    "spread_rate",
    "max_participation_rate",
    "minimum_notional",
    "lot_size",
    "tick_size",
)


def allocation_training_source(
    env: AllocationTradingEnv,
    *,
    decision_counts: Mapping[int, int] | None = None,
    observation_indices: Sequence[int] | None = None,
) -> dict[str, Any]:
    """Hash sampled per-step inputs; None is an unpublished full-envelope audit."""
    dataset, start, stop = env.dataset, env.start_index, env.stop_index
    counts = (
        dict(decision_counts)
        if decision_counts is not None
        else dict.fromkeys(range(start, stop), 1)
    )
    if not counts or any(
        isinstance(i, bool)
        or not isinstance(i, int)
        or not start <= i < stop
        or isinstance(n, bool)
        or not isinstance(n, int)
        or n <= 0
        for i, n in counts.items()
    ):
        raise ValueError("source requires valid observed decision indices and counts")
    rows = sorted(counts)
    if decision_counts is not None and observation_indices is None:
        raise ValueError("actual source requires observed policy and bootstrap rows")
    observed = (
        sorted(set(observation_indices)) if observation_indices is not None else rows
    )
    if (
        not observed
        or any(
            isinstance(i, bool) or not isinstance(i, int) or not start <= i < stop
            for i in observed
        )
        or not set(rows) <= set(observed)
    ):
        raise ValueError("source requires valid observed policy and bootstrap rows")
    processing_rows = [i + 1 for i in rows]
    indices, _ = validated_training_scope(
        dataset,
        feature_indices=env.feature_indices,
        fit_symbol_indices=(env.symbol_index,),
    )
    times = dataset.timestamps[observed].astype("datetime64[ns]")
    symbol = dataset.symbols[env.symbol_index]
    packets = tuple(
        p for p in env.stream.packets if p.symbol == symbol and p.as_of in times
    )
    if len(packets) != len(observed) or {p.as_of for p in packets} != set(times):
        raise ValueError("training forecasts must cover exactly the decision clocks")
    used_vintages = {p.vintage_digest for p in packets}
    vintages = tuple(v for v in env.stream.vintages if v.digest in used_vintages)
    execution = {}
    arrays = dataset.identity_arrays()
    for name in _EXECUTION_FIELDS:
        values = arrays[name]
        selected = (
            values[processing_rows, env.symbol_index]
            if values.ndim == 2
            else values[processing_rows]
        )
        if selected.dtype.kind == "M":
            selected = selected.astype("datetime64[ns]").astype(np.int64)
        execution[name] = selected.tolist()
    costs = [env._estimates[int(t.astype(np.int64))].payload() for t in times]
    return {
        "dataset_id": dataset.dataset_id,
        "start_index": start,
        "stop_index": stop,
        "symbol": symbol,
        "feature_names": [dataset.feature_names[i] for i in indices],
        "decision_start": str(dataset.timestamps[start].astype("datetime64[ns]")),
        "terminal_time": str(dataset.timestamps[stop].astype("datetime64[ns]")),
        "decision_indices": rows,
        "decision_counts": [counts[i] for i in rows],
        "observation_indices": observed,
        "feature_consumption_digest": content_digest(
            {
                "values": dataset.features[observed, env.symbol_index][
                    :, list(indices)
                ].tolist(),
                "available": dataset.feature_available[observed, env.symbol_index][
                    :, list(indices)
                ].tolist(),
                "source_available_at": dataset.resolved_array("available_at")[
                    observed, env.symbol_index
                ]
                .astype("datetime64[ns]")
                .astype(np.int64)
                .tolist(),
            }
        ),
        "clock_consumption_digest": content_digest(
            dataset.timestamps[np.asarray([rows, processing_rows]).T]
            .astype("datetime64[ns]")
            .astype(np.int64)
            .tolist()
        ),
        "forecast_consumption_digest": content_digest(
            {
                "packets": [p.payload() for p in packets],
                "vintages": [v.payload() for v in vintages],
            }
        ),
        "cost_consumption_digest": content_digest(costs),
        "execution_consumption_digest": content_digest(
            {
                "bars": execution,
                "capacity_reference": (
                    None
                    if env.execution_cost.processing_bar_volume_capacity
                    else {
                        "indices": rows,
                        "volume": dataset.volume[rows, env.symbol_index].tolist(),
                        "close": dataset.close[rows, env.symbol_index].tolist(),
                    }
                ),
                "contract": {
                    "symbol": symbol,
                    "volume_unit": dataset.volume_units[env.symbol_index].value,
                    "calendar_kind": dataset.calendar_kind,
                    "bar_hours": dataset.bar_hours,
                },
                "multiplier": float(
                    dataset.resolved_array("contract_multipliers")[env.symbol_index]
                ),
            }
        ),
    }
