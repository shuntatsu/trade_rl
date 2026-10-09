"""Opt-in composition for already-authorized, same-close development inputs.

The caller authorizes the entire artifact before calling: the Dataset loader
reads all arrays. Bounds below are post-load conformity, never read authority.
Declared lineage, costs and historical clocks do not authenticate construction,
calibration, compute latency, research eligibility or profitable execution.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data import load_market_dataset_artifact
from trade_rl.data.build.config import _feature
from trade_rl.data.contracts import PORTABLE_FEATURE_NUMERICS_SCHEMA
from trade_rl.data.identity import MARKET_DATASET_IDENTITY_SCHEMA, parse_identity_json
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.allocation_costs import (
    DeclaredAllocationCostRecipe,
    estimate_declared_horizon_costs,
)
from trade_rl.evaluation.forecast_allocation import HorizonCostEstimates
from trade_rl.strategies.dataset_scope import validated_training_scope
from trade_rl.strategies.forecasts.simple_prequential import (
    fit_prequential_simple_ridge,
)
from trade_rl.strategies.forecasts.simple_stream import FrozenSimpleReturnStream
from trade_rl.strategies.forecasts.simple_stream_io import (
    PublishedSimpleReturnStreamArtifact,
    publish_simple_return_stream_artifact,
)
from trade_rl.strategies.forecasts.stream import (
    ForecastBlock,
    _after,
    _from_ns,
    _validate_blocks,
)
from trade_rl.strategies.forecasts.training_trace import _timestamp


@dataclass(frozen=True, slots=True)
class PreparedAllocationMarketInputs:
    """Common inputs; cost groups follow Dataset symbols, then decision order."""

    dataset: MarketDataset
    stream: FrozenSimpleReturnStream
    artifact: PublishedSimpleReturnStreamArtifact
    costs_by_symbol: tuple[tuple[HorizonCostEstimates, ...], ...]
    cost_recipe: DeclaredAllocationCostRecipe | None = None


def _cost_from_payload(row: Mapping[str, object]) -> HorizonCostEstimates:
    clocks = ("decision_time", "available_at", "horizon_end")
    rates = (
        "buy_cost",
        "sell_cost",
        "exit_cost",
        "funding_return",
        "borrow_return",
        "cash_return",
    )
    fields = {
        "schema",
        "valuation_basis",
        "cost_basis",
        "symbol",
        "source_identity",
        *clocks,
        *rates,
    }
    if (
        not isinstance(row, Mapping)
        or set(row) != fields
        or row["schema"] != "horizon_cost_estimates_v1"
        or row["cost_basis"] != "declared_initial_notional_rates"
    ):
        raise ValueError(
            "cost row must be a complete horizon_cost_estimates_v1 payload"
        )
    arguments: dict[str, Any] = {
        k: v for k, v in row.items() if k not in ("schema", "cost_basis")
    }
    for field in clocks:
        arguments[field] = _from_ns(row[field])
    try:
        estimate = HorizonCostEstimates(**arguments)
        if canonical_json_bytes(estimate.payload()) != canonical_json_bytes(row):
            raise ValueError("cost payload changed during decoding")
    except (TypeError, ValueError) as error:
        raise ValueError("invalid horizon cost payload") from error
    return estimate


def prepare_allocation_market_inputs(
    dataset_root: str | Path,
    stream_path: str | Path,
    *,
    expected_dataset_id: str,
    development_start: np.datetime64,
    development_stop: np.datetime64,
    feature_names: tuple[str, ...],
    fit_symbols: tuple[str, ...],
    blocks: tuple[ForecastBlock, ...],
    horizon_hours: int,
    alpha: float,
    cost_rows: tuple[Mapping[str, object], ...] | None = None,
    cost_recipe: DeclaredAllocationCostRecipe | None = None,
) -> PreparedAllocationMarketInputs:
    """Admit one explicit cost path, fit the existing producer once and publish it.

    Fit symbols restrict training, not prediction. Every predicted Dataset
    symbol/decision requires exactly one available, matching-horizon cost row.
    Same-close Dataset marks are required; Book/current-context checks remain
    with existing downstream consumers. No price conversion or crop is applied.
    cost_recipe opts into the concrete declared reference-size estimator on the
    already-loaded Dataset; the returned recipe/config is shared across lanes.
    """
    if (cost_rows is None) == (cost_recipe is None):
        raise ValueError("supply exactly one of cost_rows or cost_recipe")
    if cost_recipe is not None and not isinstance(
        cost_recipe, DeclaredAllocationCostRecipe
    ):
        raise ValueError("cost_recipe must be a DeclaredAllocationCostRecipe")
    require_sha256(expected_dataset_id, field="expected_dataset_id")
    start = _timestamp(development_start, field="development_start")
    stop = _timestamp(development_stop, field="development_stop")
    if start >= stop:
        raise ValueError("development bounds must be ordered")
    _validate_blocks(blocks)
    if cost_recipe is not None:
        cost_recipe.require_available(min(block.prediction_start for block in blocks))
    if (
        isinstance(horizon_hours, bool)
        or not isinstance(horizon_hours, int)
        or horizon_hours <= 0
    ):
        raise ValueError("horizon_hours must be a positive integer")
    if any(
        b.inference_delay_seconds != 0
        or b.fit_cutoff < start
        or b.prediction_stop > stop
        for b in blocks
    ):
        raise ValueError("blocks must be inside development bounds with zero delay")
    output = Path(stream_path)
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"stream destination exists: {output}")

    dataset = load_market_dataset_artifact(dataset_root)
    if dataset.dataset_id != expected_dataset_id:
        raise ValueError("loaded Dataset differs from external expected_dataset_id")
    if dataset.identity_payload_json is None:
        raise ValueError("Dataset lacks canonical build lineage")
    identity = parse_identity_json(dataset.identity_payload_json)
    config = identity.get("config")
    if (
        identity.get("schema") != MARKET_DATASET_IDENTITY_SCHEMA
        or "source_dataset" in identity
        or not isinstance(config, Mapping)
        or config.get("schema_version") != "market_build_v3"
        or config.get("feature_numerics_schema") != PORTABLE_FEATURE_NUMERICS_SCHEMA
        or not isinstance(config.get("features"), list)
    ):
        raise ValueError(
            "Dataset requires declared direct canonical MarketBuilder lineage"
        )
    specs = tuple(
        _feature(value, index=i) for i, value in enumerate(config["features"])
    )
    if tuple(s.name for s in specs) != dataset.feature_names or any(
        canonical_json_bytes(s.canonical_payload()) != canonical_json_bytes(raw)
        for s, raw in zip(specs, config["features"], strict=True)
    ):
        raise ValueError(
            "Dataset feature declarations do not match canonical names/specs"
        )
    if dataset.timestamps[0] < start or dataset.timestamps[-1] >= stop:
        raise ValueError("loaded Dataset escapes declared development bounds")
    features = require_unique_non_empty(feature_names, field="feature_names")
    symbols = require_unique_non_empty(fit_symbols, field="fit_symbols")
    try:
        indices, fit_indices = validated_training_scope(
            dataset,
            feature_indices=tuple(
                dataset.feature_names.index(name) for name in features
            ),
            fit_symbol_indices=tuple(dataset.symbols.index(name) for name in symbols),
        )
    except ValueError as error:
        raise ValueError("invalid declared feature/fit-symbol scope") from error

    roster: dict[tuple[str, np.datetime64], np.datetime64] = {}
    decision_indices: set[int] = set()
    marks = dataset.resolved_array("mark_price")
    available = dataset.resolved_array("available_at")
    information = dataset.resolved_array("information_available")
    for block in blocks:
        rows = [
            i
            for i, t in enumerate(dataset.timestamps)
            if block.prediction_start <= t < block.prediction_stop
        ]
        if not rows:
            raise ValueError("prediction block has no Dataset decisions")
        for row in rows:
            decision_indices.add(row)
            decision = _timestamp(dataset.timestamps[row], field="decision_time")
            end = _after(decision, horizon_hours * 3600)
            if end > stop:
                raise ValueError("prediction horizon escapes development bounds")
            for symbol_index, symbol in enumerate(dataset.symbols):
                if (
                    dataset.close[row, symbol_index] != marks[row, symbol_index]
                    or available[row, symbol_index] > decision
                    or not information[row, symbol_index]
                    or not np.all(
                        dataset.feature_available[row, symbol_index, list(indices)]
                    )
                ):
                    raise ValueError(
                        "prospective packet lacks available same-close inputs"
                    )
                roster[symbol, decision] = end

    if cost_recipe is not None:
        cost_rows = tuple(
            estimate.payload()
            for estimate in estimate_declared_horizon_costs(
                dataset,
                decision_indices=tuple(sorted(decision_indices)),
                horizon_hours=horizon_hours,
                recipe=cost_recipe,
            )
        )
    assert cost_rows is not None
    costs: dict[tuple[str, np.datetime64], HorizonCostEstimates] = {}
    for raw in cost_rows:
        estimate = _cost_from_payload(raw)
        key = (estimate.symbol, estimate.decision_time)
        if key in costs or roster.get(key) != estimate.horizon_end:
            raise ValueError(
                "cost rows duplicate or disagree with prospective packet roster"
            )
        costs[key] = estimate
    if costs.keys() != roster.keys():
        raise ValueError("cost rows must cover every prospective packet exactly")
    ordered_costs = tuple(
        tuple(costs[key] for key in roster if key[0] == symbol)
        for symbol in dataset.symbols
    )
    stream = fit_prequential_simple_ridge(
        dataset,
        blocks=blocks,
        feature_indices=indices,
        fit_symbol_indices=fit_indices,
        horizon_hours=horizon_hours,
        alpha=alpha,
    )
    artifact = publish_simple_return_stream_artifact(output, stream)
    return PreparedAllocationMarketInputs(
        dataset, stream, artifact, ordered_costs, cost_recipe
    )


__all__ = ["PreparedAllocationMarketInputs", "prepare_allocation_market_inputs"]
