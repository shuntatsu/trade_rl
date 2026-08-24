"""Pure state/contract science helpers for the Causal Alpha V4 stage runner."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np

_VOLATILITY_FEATURE: Final = "15m__garman_klass_volatility_32bar"
_LIQUIDITY_FEATURE: Final = "15m__relative_volume_32bar"
_STRESS_FEATURE: Final = "spot_perp_basis_robust_z_7d"


@dataclass(frozen=True, slots=True)
class CausalAlphaV4StageStateInputs:
    realized_volatility: np.ndarray
    liquidity: np.ndarray
    basis_positioning_stress: np.ndarray
    state_eligible: np.ndarray
    actionable: np.ndarray

    def __post_init__(self) -> None:
        arrays = {
            "realized_volatility": np.asarray(
                self.realized_volatility, dtype=np.float64
            )
            .reshape(-1)
            .copy(order="C"),
            "liquidity": np.asarray(self.liquidity, dtype=np.float64)
            .reshape(-1)
            .copy(order="C"),
            "basis_positioning_stress": np.asarray(
                self.basis_positioning_stress, dtype=np.float64
            )
            .reshape(-1)
            .copy(order="C"),
            "state_eligible": np.asarray(self.state_eligible, dtype=np.bool_)
            .reshape(-1)
            .copy(order="C"),
            "actionable": np.asarray(self.actionable, dtype=np.bool_)
            .reshape(-1)
            .copy(order="C"),
        }
        shape = arrays["realized_volatility"].shape
        if (
            not shape
            or shape[0] == 0
            or any(value.shape != shape for value in arrays.values())
        ):
            raise ValueError("V4 stage state arrays must be non-empty and aligned")
        for name in (
            "realized_volatility",
            "liquidity",
            "basis_positioning_stress",
        ):
            mask = arrays["state_eligible"]
            if not np.isfinite(arrays[name][mask]).all():
                raise ValueError("V4 stage eligible state inputs must be finite")
        for array in arrays.values():
            array.setflags(write=False)
        for name, array in arrays.items():
            object.__setattr__(self, name, array)


def resolve_causal_alpha_v4_contract_rows(
    sample: object,
    *,
    start: int,
    stop: int,
) -> np.ndarray:
    """Require exact one-row-per-decision V4 sample coverage for an episode contract."""

    if (
        isinstance(start, bool)
        or not isinstance(start, int)
        or isinstance(stop, bool)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start + 1
    ):
        raise ValueError("V4 contract range must contain at least one decision")
    decisions = np.asarray(
        getattr(sample, "decision_indices", None), dtype=np.int64
    ).reshape(-1)
    if decisions.size == 0 or np.any(np.diff(decisions) <= 0):
        raise ValueError("V4 sample decisions must be strictly increasing")
    expected = np.arange(start, stop - 1, dtype=np.int64)
    positions = np.searchsorted(decisions, expected)
    present = positions < decisions.size
    if not np.all(present) or not np.array_equal(decisions[positions], expected):
        raise ValueError("V4 contract requires complete decision coverage")
    result = np.asarray(positions, dtype=np.int64)
    result.setflags(write=False)
    return result


def _named_column(
    *,
    names: tuple[str, ...],
    values: object,
    available: object,
    name: str,
    field: str,
) -> tuple[np.ndarray, np.ndarray]:
    if names.count(name) != 1:
        raise ValueError(f"V4 stage required {field} channel is unavailable")
    index = names.index(name)
    matrix = np.asarray(values, dtype=np.float64)
    mask = np.asarray(available, dtype=np.bool_)
    if matrix.ndim != 2 or mask.shape != matrix.shape or matrix.shape[1] != len(names):
        raise ValueError(f"V4 stage {field} arrays are misaligned")
    return matrix[:, index], mask[:, index]


def resolve_causal_alpha_v4_stage_state_inputs(
    sample: object,
) -> CausalAlphaV4StageStateInputs:
    """Resolve the frozen V4 uncertainty state channels and alpha actionability."""

    target_names = tuple(getattr(sample, "target_local_feature_names", ()))
    local = getattr(sample, "local_context", None)
    global_market = getattr(sample, "global_context", None)
    local_names = tuple(getattr(local, "feature_names", ()))
    global_names = tuple(getattr(global_market, "feature_names", ()))
    volatility, volatility_available = _named_column(
        names=target_names,
        values=getattr(sample, "target_local_features", None),
        available=getattr(sample, "target_local_available", None),
        name=_VOLATILITY_FEATURE,
        field="realized volatility",
    )
    liquidity, liquidity_available = _named_column(
        names=target_names,
        values=getattr(sample, "target_local_features", None),
        available=getattr(sample, "target_local_available", None),
        name=_LIQUIDITY_FEATURE,
        field="liquidity",
    )
    stress, stress_available = _named_column(
        names=local_names,
        values=getattr(local, "values", None),
        available=getattr(local, "available", None),
        name=_STRESS_FEATURE,
        field="basis positioning stress",
    )

    local_available = np.asarray(getattr(local, "available", None), dtype=np.bool_)
    global_available = np.asarray(
        getattr(global_market, "available", None), dtype=np.bool_
    )
    beta = np.asarray(getattr(sample, "beta", None), dtype=np.float64).reshape(-1)
    beta_available = np.asarray(
        getattr(sample, "beta_available", None), dtype=np.bool_
    ).reshape(-1)
    rows = volatility.shape[0]
    if (
        rows == 0
        or local_available.shape != (rows, len(local_names))
        or global_available.shape != (rows, len(global_names))
        or beta.shape != (rows,)
        or beta_available.shape != (rows,)
        or not np.isfinite(beta).all()
    ):
        raise ValueError("V4 stage required context arrays are misaligned")
    actionable = (
        np.all(local_available, axis=1)
        & np.all(global_available, axis=1)
        & beta_available
    )
    state_eligible = (
        actionable
        & volatility_available
        & liquidity_available
        & stress_available
        & np.isfinite(volatility)
        & np.isfinite(liquidity)
        & np.isfinite(stress)
    )
    return CausalAlphaV4StageStateInputs(
        realized_volatility=volatility,
        liquidity=liquidity,
        basis_positioning_stress=stress,
        state_eligible=state_eligible,
        actionable=actionable,
    )


__all__ = [
    "CausalAlphaV4StageStateInputs",
    "resolve_causal_alpha_v4_contract_rows",
    "resolve_causal_alpha_v4_stage_state_inputs",
]
