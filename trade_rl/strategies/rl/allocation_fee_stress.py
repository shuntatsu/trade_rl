"""Exact configured-fee-only inference view; no research or trading permission."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from hashlib import sha256
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    _predict_allocation_action,
)
from trade_rl.strategies.rl.allocation_preprocessing import _native_json
from trade_rl.strategies.rl.allocation_recipe_validation import (
    validate_allocation_recipe,
)

_KEYS = {
    "schema",
    "base_recipe",
    "stress_recipe",
    "base_cost_payload",
    "stress_cost_payload",
    "fee_factor",
    "original_policy_digest",
    "original_policy_sha256",
    "model_state_digest",
    "contract_digest",
    "scenario_digest",
}
_COST_KEYS = set(
    """
    allow_short borrow_rate_multiplier collateral_haircut fee_rate impact_rate
    limit_offset_rate lot_size maintenance_margin_rate maker_fee_rate margin_mode
    max_leverage max_participation_rate minimum_notional multiplier order_latency_bars
    order_type partial_fill_carry path_mode processing_bar_volume_capacity random_seed
    schema_version slippage_std spread_rate tail_slippage_multiplier
    tail_slippage_probability taker_fee_rate tick_size trigger_volume_fractions
""".split()
)


def _number(value: object, field: str) -> float:
    if type(value) not in (int, float):
        raise ValueError(f"{field} must be a native finite number")
    try:
        result = float(cast(int | float, value))
    except (OverflowError, ValueError) as error:
        raise ValueError(f"{field} must fit finite native arithmetic") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def retained_debt_economics_digest(cost_payload: dict[str, Any]) -> str:
    """Exact supported wrapper, independently recomputed from actual cost bytes."""
    return content_digest(
        {
            "schema_version": "insolvency_execution_policy_v1",
            "base_policy_digest": content_digest(cost_payload),
            "insolvency_valuation": "retain_debt",
        }
    )


def allocation_policy_state_digest(policy: AllocationPPOPolicy) -> str:
    """Read parameter/persistent-buffer bytes, not optimizer or transient state."""
    if type(policy) is not AllocationPPOPolicy:
        raise ValueError("fee view requires an original AllocationPPOPolicy")
    state = policy.model.policy.state_dict()
    if (
        not isinstance(state, Mapping)
        or not state
        or any(type(k) is not str for k in state)
    ):
        raise ValueError("policy requires named tensor state")
    digest = sha256()
    for name in sorted(state):
        tensor = state[name]
        # NumPy arrays support explicit fake-model software tests only. Actual
        # PPO class/protocol checking remains the existing prediction owner's job.
        values = np.asarray(
            tensor if isinstance(tensor, np.ndarray) else tensor.detach().cpu().numpy()
        )
        if values.dtype.kind not in "biuf" or not np.isfinite(values).all():
            raise ValueError("policy tensor state must be finite numeric bytes")
        raw = np.ascontiguousarray(values).tobytes()
        header = canonical_json_bytes(
            {"name": name, "dtype": values.dtype.str, "shape": list(values.shape)}
        )
        digest.update(
            len(header).to_bytes(8, "big") + header + len(raw).to_bytes(8, "big") + raw
        )
    return digest.hexdigest()


def _validate(payload: object) -> dict[str, Any]:
    _native_json(payload)
    if type(payload) is not dict or set(payload) != _KEYS:
        raise ValueError("fee binding requires exactly its declared fields")
    if payload["schema"] != "allocation_fee_stress_binding_v1":
        raise ValueError("unknown fee binding schema")
    for name in (
        "original_policy_digest",
        "original_policy_sha256",
        "model_state_digest",
        "contract_digest",
        "scenario_digest",
    ):
        require_sha256(payload[name], field=name)
    factor = _number(payload["fee_factor"], "fee factor")
    if factor <= 1:
        raise ValueError("fee factor must be finite and greater than one")
    base, stress = payload["base_cost_payload"], payload["stress_cost_payload"]
    if (
        type(base) is not dict
        or type(stress) is not dict
        or set(base) != _COST_KEYS
        or set(stress) != _COST_KEYS
    ):
        raise ValueError("fee costs require identical native payload fields")
    if (
        base.get("schema_version") != "execution_policy_v2"
        or stress.get("schema_version") != "execution_policy_v2"
    ):
        raise ValueError("fee binding requires native execution policy v2")
    fee = _number(base["fee_rate"], "base fee")
    stressed_fee = _number(stress["fee_rate"], "stress fee")
    if fee <= 0 or not math.isfinite(fee * factor) or stressed_fee != fee * factor:
        raise ValueError("fee costs differ from the positive declared transformation")
    if canonical_json_bytes(
        base | {"fee_rate": stress["fee_rate"]}
    ) != canonical_json_bytes(stress):
        raise ValueError("fee stress changed another native cost field")
    for cost in (base, stress):
        if (
            cost.get("order_type") != "market"
            or type(cost.get("order_latency_bars")) is not int
            or cost["order_latency_bars"] != 0
        ):
            raise ValueError("fee view supports MARKET with zero extra latency")
    base_recipe, stress_recipe = payload["base_recipe"], payload["stress_recipe"]
    for recipe, cost in ((base_recipe, base), (stress_recipe, stress)):
        if type(recipe) is not dict or recipe.get("schema") not in (
            "allocation_ppo_recipe_v2",
            "allocation_ppo_recipe_v3",
        ):
            raise ValueError("fee view requires recipe v2/v3")
        validate_allocation_recipe(recipe)
        if recipe["runtime_profile"][
            "economics_digest"
        ] != retained_debt_economics_digest(cost):
            raise ValueError("fee recipe differs from actual cost economics")
    equivalent = json.loads(canonical_json_bytes(base_recipe))
    equivalent["runtime_profile"]["economics_digest"] = stress_recipe[
        "runtime_profile"
    ]["economics_digest"]
    if canonical_json_bytes(equivalent) != canonical_json_bytes(stress_recipe):
        raise ValueError("fee stress changed risk, candidate or runtime invariant")
    return payload


@dataclass(frozen=True, slots=True, init=False)
class AllocationFeeStressBinding:
    """Detached exact declaration; consistency pins are not authenticity proof."""

    _raw: bytes

    def __init__(self, payload: object) -> None:
        object.__setattr__(self, "_raw", canonical_json_bytes(_validate(payload)))

    def payload(self) -> dict[str, Any]:
        return json.loads(self._raw)

    @property
    def digest(self) -> str:
        return sha256(self._raw).hexdigest()


class AllocationFeeStressPolicyView:
    """Same original model under one exact fee intervention; never a new bundle."""

    def __init__(
        self, policy: AllocationPPOPolicy, binding: AllocationFeeStressBinding
    ) -> None:
        if type(binding) is not AllocationFeeStressBinding:
            raise ValueError("fee policy view requires the exact typed binding")
        self._policy, self._binding = policy, binding
        self._model = policy.model
        self.validate_original()

    def validate_original(self) -> None:
        payload = _validate(self._binding.payload())
        manifest = self._policy.manifest
        if (
            manifest.get("schema") != "allocation_ppo_inference_bundle_v5"
            or content_digest(manifest) != payload["original_policy_digest"]
            or manifest.get("policy_sha256") != payload["original_policy_sha256"]
            or canonical_json_bytes(manifest.get("recipe"))
            != canonical_json_bytes(payload["base_recipe"])
            or self._policy.model is not self._model
        ):
            raise ValueError("fee view original policy differs from its admitted pins")
        if (
            allocation_policy_state_digest(self._policy)
            != payload["model_state_digest"]
        ):
            raise ValueError("fee view model tensor contents changed")

    def action(self, observation: np.ndarray, *, runtime_recipe_digest: str) -> int:
        if type(runtime_recipe_digest) is not str:
            raise ValueError("fee runtime recipe must be a native digest")
        require_sha256(runtime_recipe_digest, field="runtime_recipe_digest")
        if runtime_recipe_digest != content_digest(
            self._binding.payload()["stress_recipe"]
        ):
            raise ValueError("fee view actual runtime recipe differs from declaration")
        self.validate_original()
        action = _predict_allocation_action(
            self._model, self._policy.manifest, observation
        )
        self.validate_original()
        return action
