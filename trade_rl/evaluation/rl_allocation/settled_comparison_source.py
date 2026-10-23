"""Complete passive source projection for matched native CASH/NONRL controls."""

from __future__ import annotations

import json
from dataclasses import asdict
from hashlib import sha256
from typing import Any

import numpy as np

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.rl_allocation.settled_native_validation import (
    SettledNativeValidationContext,
)
from trade_rl.evaluation.rl_allocation.settled_training_source import (
    settled_training_source_receipt,
)
from trade_rl.strategies.rl.allocation_recipe_v4 import (
    validate_settled_allocation_recipe,
)

_ARRAY_NAMES = frozenset(
    "timestamps available_at information_available features feature_available "
    "feature_staleness feature_staleness_hours feature_missing_reason global_features "
    "global_feature_available global_feature_staleness_hours global_feature_missing_reason "
    "open high low close volume funding_rate funding_price_rate funding_event_count "
    "tradable symbol_active fee_rate maker_fee_rate taker_fee_rate spread_rate "
    "max_participation_rate minimum_notional lot_size tick_size borrow_available "
    "borrow_rate funding_due buy_allowed sell_allowed mark_price index_price dividend "
    "split_factor delisting_recovery cash_rate contract_multipliers".split()
)
_ENVELOPE_NAMES = frozenset(
    "schema evidence_kind consumption_proven account_id bound_digest recipe_digest "
    "dataset_descriptor actor_scope native_scope native_financial_arrays "
    "actor_selected_features actor_forecast_selected_features actor_forecasts "
    "actor_used_vintages actor_cost_declarations".split()
)
SETTLED_NATIVE_CONTROL_SOURCE_FIELDS = frozenset(
    "schema dataset source_envelope objective recipe execution_cost risk allocator "
    "observation_schema feature_indices expected_horizon_seconds stream estimates".split()
)


def _normalized(value: Any) -> Any:
    return json.loads(canonical_json_bytes(value))


def _dataset_material(dataset: MarketDataset) -> dict[str, Any]:
    if type(dataset) is not MarketDataset:
        raise ValueError("matched controls require an exact Dataset")
    arrays = dataset.identity_arrays()
    if set(arrays) != _ARRAY_NAMES:
        raise ValueError("matched Dataset array schema changed")
    if any(
        not isinstance(array, np.ndarray)
        or array.dtype.hasobject
        or array.dtype.fields is not None
        or array.dtype.subdtype is not None
        for array in arrays.values()
    ):
        raise ValueError("matched Dataset requires plain deterministic array dtypes")
    if not dataset.identity_verified:
        raise ValueError(
            "matched controls require a verified complete Dataset identity"
        )
    identity = json.loads(dataset.identity_payload_json or "null")
    descriptor = dataset.identity_contract_payload()
    if (
        type(identity) is not dict
        or identity.get("schema") != "market_dataset_identity_v7"
        or canonical_json_bytes(identity.get("dataset_contract"))
        != canonical_json_bytes(descriptor)
    ):
        raise ValueError("matched Dataset identity differs from its live descriptor")
    return {
        "dataset_id": dataset.dataset_id,
        "identity_payload": identity,
        "descriptor": descriptor,
        "arrays": {
            name: {
                "dtype": array.dtype.str,
                "shape": list(array.shape),
                "sha256": sha256(array.tobytes(order="C")).hexdigest(),
            }
            for name, array in sorted(arrays.items())
        },
    }


def settled_native_control_source_material(
    context: SettledNativeValidationContext,
) -> dict[str, Any]:
    """Retain all sources; remove only the declared method/account differences."""
    if type(context) is not SettledNativeValidationContext:
        raise ValueError("matched controls require the exact passive native context")
    context.validate_sources()
    envelope = settled_training_source_receipt(context)
    if (
        set(envelope) != _ENVELOPE_NAMES
        or envelope["schema"] != "allocation_settled_training_source_envelope_v1"
        or envelope["evidence_kind"] != "declared_prefit_envelope"
        or envelope["consumption_proven"] is not False
    ):
        raise ValueError("matched native source envelope schema changed")
    for name in ("account_id", "bound_digest", "recipe_digest"):
        del envelope[name]
    recipe = context.recipe
    validate_settled_allocation_recipe(recipe)
    action = recipe["action"]
    if type(action) is not dict:
        raise ValueError("matched control action requires the closed recipe mapping")
    del action["mode"]
    objective = context.bound.objective.payload()
    if (
        objective["deployment_recipe_digest"] != context.recipe_digest
        or objective["economics_digest"] != context.economics_digest
        or objective["risk_digest"] != content_digest(context.risk_config)
    ):
        raise ValueError("matched objective must retain its own complete recipe")
    del objective["deployment_recipe_digest"]
    material = _normalized(
        {
            "schema": "allocation_settled_native_control_source_v1",
            "dataset": _dataset_material(context.dataset),
            "source_envelope": envelope,
            "objective": objective,
            "recipe": recipe,
            "execution_cost": asdict(context.execution_cost),
            "risk": asdict(context.risk_config),
            "allocator": asdict(context.allocator),
            "observation_schema": asdict(context.observation_schema),
            "feature_indices": context.feature_indices,
            "expected_horizon_seconds": context.expected_horizon_seconds,
            "stream": context.stream.payload(),
            "estimates": [value.payload() for value in context.estimates],
        }
    )
    context.validate_sources()
    return material


__all__ = [
    "SETTLED_NATIVE_CONTROL_SOURCE_FIELDS",
    "settled_native_control_source_material",
]
