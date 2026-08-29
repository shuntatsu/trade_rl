"""Closed-loop hierarchical exposure policy for Causal Alpha V10."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.domain.common import require_sha256
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v6_target import causal_alpha_v6_fast_objective
from trade_rl.learning.causal_alpha_v10 import CausalAlphaV10Config

_EPSILON = 1e-12
_OBSERVATION_TOLERANCE = 1e-6


class CausalAlphaV10BoundaryMode(str, Enum):
    """Ownership treatment for a non-flat episode-start position."""

    INHERIT_CONFIRM = "inherit_confirm"
    FLATTEN_THEN_RESET = "flatten_then_reset"
    NEUTRAL_FAST_EXPIRY = "neutral_fast_expiry"
    FLATTEN_ON_RISK_BREACH = "flatten_on_risk_breach"
    FAST_ONLY_OWNERSHIP = "fast_only_ownership"


class _AttributionBoundaries(Protocol):
    liquidity: tuple[float, float, float]
    realized_volatility: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class CausalAlphaV10ExecutionContract:
    """PreTrade fields that can change one V10 replay's realized exposure."""

    max_gross: float = 1.0
    max_abs_weight: float = 1.0
    max_turnover: float | None = 2.0
    entry_threshold: float = 0.0
    exit_threshold: float = 0.0
    no_trade_band: float = 0.0
    drawdown_start: float = 0.10
    drawdown_stop: float = 0.20
    emergency_turnover_override: bool = True
    fail_closed_tolerance: float = 1e-10

    def __post_init__(self) -> None:
        finite_values = (
            self.max_gross,
            self.max_abs_weight,
            self.entry_threshold,
            self.exit_threshold,
            self.no_trade_band,
            self.drawdown_start,
            self.drawdown_stop,
            self.fail_closed_tolerance,
        )
        if any(
            isinstance(value, bool) or not np.isfinite(value) for value in finite_values
        ):
            raise ValueError("V10 execution contract values must be finite")
        if not 0.0 < self.max_gross <= 10.0:
            raise ValueError("V10 execution max_gross is invalid")
        if not 0.0 < self.max_abs_weight <= self.max_gross:
            raise ValueError("V10 execution max_abs_weight is invalid")
        if self.max_turnover is not None and (
            isinstance(self.max_turnover, bool)
            or not np.isfinite(self.max_turnover)
            or not 0.0 <= self.max_turnover <= 2.0 * self.max_gross
        ):
            raise ValueError("V10 execution max_turnover is invalid")
        if not 0.0 <= self.entry_threshold <= self.max_abs_weight:
            raise ValueError("V10 execution entry_threshold is invalid")
        if not 0.0 <= self.exit_threshold <= self.entry_threshold:
            raise ValueError("V10 execution exit_threshold is invalid")
        if not 0.0 <= self.no_trade_band <= 2.0 * self.max_abs_weight:
            raise ValueError("V10 execution no_trade_band is invalid")
        if not 0.0 <= self.drawdown_start <= self.drawdown_stop <= 1.0:
            raise ValueError("V10 execution drawdown thresholds are invalid")
        if self.fail_closed_tolerance < 0.0:
            raise ValueError("V10 execution fail_closed_tolerance is invalid")
        if not isinstance(self.emergency_turnover_override, bool):
            raise TypeError("V10 execution emergency_turnover_override must be bool")

    def to_payload(self) -> dict[str, object]:
        return {
            "drawdown_start": self.drawdown_start,
            "drawdown_stop": self.drawdown_stop,
            "emergency_turnover_override": self.emergency_turnover_override,
            "entry_threshold": self.entry_threshold,
            "exit_threshold": self.exit_threshold,
            "fail_closed_tolerance": self.fail_closed_tolerance,
            "max_abs_weight": self.max_abs_weight,
            "max_gross": self.max_gross,
            "max_turnover": self.max_turnover,
            "no_trade_band": self.no_trade_band,
            "schema_version": "causal_alpha_v10_execution_contract_v1",
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class CausalAlphaV10HierarchyPolicyInput:
    """Immutable pre-replay causal input and execution identity for V10."""

    decision_indices: np.ndarray
    fast_head_predictions: np.ndarray
    slow_head_predictions: np.ndarray
    one_way_cost_rates: np.ndarray
    liquidity_weight_caps: np.ndarray
    risk_weight_caps: np.ndarray
    realized_volatility: np.ndarray
    liquidity: np.ndarray
    actionable_mask: np.ndarray
    attribution_liquidity: tuple[float, float, float]
    attribution_realized_volatility: tuple[float, float, float]
    source_forecast_digest: str
    dual_fit_digest: str
    config: CausalAlphaV10Config
    initial_weight: float
    execution_contract: CausalAlphaV10ExecutionContract
    compiler_config_digest: str
    boundary_mode: CausalAlphaV10BoundaryMode = CausalAlphaV10BoundaryMode.INHERIT_CONFIRM
    digest: str = ""

    def __post_init__(self) -> None:
        require_sha256(self.source_forecast_digest, field="V10 source forecast digest")
        require_sha256(self.dual_fit_digest, field="V10 dual fit digest")
        require_sha256(self.compiler_config_digest, field="V10 compiler config digest")
        if not isinstance(self.config, CausalAlphaV10Config):
            raise TypeError("V10 hierarchy input config is invalid")
        if not isinstance(self.execution_contract, CausalAlphaV10ExecutionContract):
            raise TypeError("V10 hierarchy execution contract is invalid")
        boundary_mode = CausalAlphaV10BoundaryMode(self.boundary_mode)
        if not np.isfinite(self.initial_weight):
            raise ValueError("V10 hierarchy initial weight must be finite")

        decisions = np.asarray(self.decision_indices, dtype=np.int64).reshape(-1).copy()
        rows = len(decisions)
        if rows == 0 or np.any(decisions < 0) or np.any(np.diff(decisions) <= 0):
            raise ValueError("V10 decision indices must be non-empty and increasing")
        fast_heads = np.asarray(self.fast_head_predictions, dtype=np.float64).copy()
        slow_heads = np.asarray(self.slow_head_predictions, dtype=np.float64).copy()
        if (
            fast_heads.shape != (3, rows)
            or slow_heads.shape != (3, rows)
            or not np.isfinite(fast_heads).all()
            or not np.isfinite(slow_heads).all()
        ):
            raise ValueError("V10 head predictions are invalid")

        vectors: dict[str, np.ndarray] = {}
        for name, value, dtype in (
            ("one_way_cost_rates", self.one_way_cost_rates, np.float64),
            ("liquidity_weight_caps", self.liquidity_weight_caps, np.float64),
            ("risk_weight_caps", self.risk_weight_caps, np.float64),
            ("realized_volatility", self.realized_volatility, np.float64),
            ("liquidity", self.liquidity, np.float64),
            ("actionable_mask", self.actionable_mask, np.bool_),
        ):
            array = np.asarray(value, dtype=dtype).reshape(-1).copy()
            if array.shape != (rows,):
                raise ValueError(f"V10 {name} must be decision aligned")
            if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                raise ValueError(f"V10 {name} must be finite")
            vectors[name] = array
        if any(
            np.any(vectors[name] < 0.0)
            for name in (
                "one_way_cost_rates",
                "liquidity_weight_caps",
                "risk_weight_caps",
            )
        ):
            raise ValueError("V10 costs and caps must be non-negative")

        liquidity_bounds = tuple(float(value) for value in self.attribution_liquidity)
        volatility_bounds = tuple(
            float(value) for value in self.attribution_realized_volatility
        )
        for values, field in (
            (liquidity_bounds, "liquidity boundaries"),
            (volatility_bounds, "realized-volatility boundaries"),
        ):
            if (
                len(values) != 3
                or not np.isfinite(values).all()
                or not values[0] <= values[1] <= values[2]
            ):
                raise ValueError(f"V10 {field} are invalid")

        for array in (decisions, fast_heads, slow_heads, *vectors.values()):
            array.setflags(write=False)
        object.__setattr__(self, "decision_indices", decisions)
        object.__setattr__(self, "fast_head_predictions", fast_heads)
        object.__setattr__(self, "slow_head_predictions", slow_heads)
        for name, array in vectors.items():
            object.__setattr__(self, name, array)
        object.__setattr__(self, "attribution_liquidity", liquidity_bounds)
        object.__setattr__(
            self,
            "attribution_realized_volatility",
            volatility_bounds,
        )
        object.__setattr__(self, "boundary_mode", boundary_mode)

        expected = content_and_arrays_digest(
            {
                "attribution_liquidity": liquidity_bounds,
                "attribution_realized_volatility": volatility_bounds,
                "compiler_config_digest": self.compiler_config_digest,
                "dual_fit_digest": self.dual_fit_digest,
                "execution_contract_digest": self.execution_contract.digest,
                "initial_weight": float(self.initial_weight),
                "boundary_mode": boundary_mode.value,
                "schema_version": "causal_alpha_v10_hierarchy_policy_input_v1",
                "source_forecast_digest": self.source_forecast_digest,
                "v10_config_digest": self.config.digest,
            },
            (
                ("decision_indices", decisions),
                ("fast_head_predictions", fast_heads),
                ("slow_head_predictions", slow_heads),
                ("one_way_cost_rates", vectors["one_way_cost_rates"]),
                ("liquidity_weight_caps", vectors["liquidity_weight_caps"]),
                ("risk_weight_caps", vectors["risk_weight_caps"]),
                ("realized_volatility", vectors["realized_volatility"]),
                ("liquidity", vectors["liquidity"]),
                ("actionable_mask", vectors["actionable_mask"]),
            ),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V10 hierarchy policy input digest mismatch")
        object.__setattr__(self, "digest", expected)


@dataclass(frozen=True, slots=True)
class CausalAlphaV10HierarchyResult:
    v6_target_path: CausalAlphaV6TargetPath
    input_digest: str
    hierarchy_reasons: tuple[str, ...]
    hierarchy_reason_counts: tuple[tuple[str, int], ...]

    def __post_init__(self) -> None:
        if not isinstance(self.v6_target_path, CausalAlphaV6TargetPath):
            raise TypeError("V10 hierarchy result target path is invalid")
        require_sha256(self.input_digest, field="V10 hierarchy input digest")
        reasons = tuple(self.hierarchy_reasons)
        counts = tuple(
            sorted((reason, reasons.count(reason)) for reason in set(reasons))
        )
        if len(reasons) != len(self.v6_target_path.decision_indices):
            raise ValueError("V10 hierarchy reasons must cover every decision")
        if tuple(self.hierarchy_reason_counts) != counts:
            raise ValueError("V10 hierarchy reason counts are inconsistent")
        object.__setattr__(self, "hierarchy_reasons", reasons)
        object.__setattr__(self, "hierarchy_reason_counts", counts)


def _qualified(
    heads: np.ndarray,
    *,
    edge_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(heads, axis=0, dtype=np.float64)
    uncertainty = np.std(heads, axis=0, dtype=np.float64)
    signs = np.sign(heads)
    agreed = np.all(signs == signs[0], axis=0) & (signs[0] != 0.0)
    direction = np.where(
        agreed & (np.abs(mean) > uncertainty + edge_margin),
        np.sign(mean),
        0.0,
    ).astype(np.int8)
    return mean, uncertainty, direction


def _position_sign(value: float) -> int:
    if abs(value) <= _OBSERVATION_TOLERANCE:
        return 0
    return int(np.sign(value))


def _slow_state(direction: int, current: float) -> CausalAlphaV6SlowState:
    current_sign = _position_sign(current)
    if current_sign == 0:
        return CausalAlphaV6SlowState.FLAT
    if direction == current_sign:
        return CausalAlphaV6SlowState.SUPPORTIVE
    if direction == -current_sign:
        return CausalAlphaV6SlowState.OPPOSED
    return CausalAlphaV6SlowState.MIXED


def _compiler_config_digest(
    *,
    config: CausalAlphaV10Config,
    execution_contract: CausalAlphaV10ExecutionContract,
    economic_config: CausalAlphaV6TargetConfig,
    boundary_mode: CausalAlphaV10BoundaryMode,
) -> str:
    return content_digest(
        {
            "execution_contract_digest": execution_contract.digest,
            "boundary_mode": boundary_mode.value,
            "schema_version": "causal_alpha_v10_target_compiler_contract_v3",
            "v6_economic_config_digest": economic_config.digest,
            "v10_config_digest": config.digest,
        }
    )


def _policy_input(
    *,
    decision_indices: object,
    fast_head_predictions: object,
    slow_head_predictions: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object,
    realized_volatility: object,
    liquidity: object,
    attribution_boundaries: _AttributionBoundaries,
    actionable_mask: object,
    source_forecast_digest: str,
    dual_fit_digest: str,
    config: CausalAlphaV10Config,
    initial_weight: float,
    execution_contract: CausalAlphaV10ExecutionContract,
    boundary_mode: CausalAlphaV10BoundaryMode,
) -> CausalAlphaV10HierarchyPolicyInput:
    economic_config = CausalAlphaV6TargetConfig()
    return CausalAlphaV10HierarchyPolicyInput(
        decision_indices=np.asarray(decision_indices),
        fast_head_predictions=np.asarray(fast_head_predictions),
        slow_head_predictions=np.asarray(slow_head_predictions),
        one_way_cost_rates=np.asarray(one_way_cost_rates),
        liquidity_weight_caps=np.asarray(liquidity_weight_caps),
        risk_weight_caps=np.asarray(risk_weight_caps),
        realized_volatility=np.asarray(realized_volatility),
        liquidity=np.asarray(liquidity),
        actionable_mask=np.asarray(actionable_mask),
        attribution_liquidity=(
            float(attribution_boundaries.liquidity[0]),
            float(attribution_boundaries.liquidity[1]),
            float(attribution_boundaries.liquidity[2]),
        ),
        attribution_realized_volatility=(
            float(attribution_boundaries.realized_volatility[0]),
            float(attribution_boundaries.realized_volatility[1]),
            float(attribution_boundaries.realized_volatility[2]),
        ),
        source_forecast_digest=source_forecast_digest,
        dual_fit_digest=dual_fit_digest,
        config=config,
        initial_weight=float(initial_weight),
        execution_contract=execution_contract,
        boundary_mode=boundary_mode,
        compiler_config_digest=_compiler_config_digest(
            config=config,
            execution_contract=execution_contract,
            economic_config=economic_config,
            boundary_mode=boundary_mode,
        ),
    )


class CausalAlphaV10HierarchyPolicy:
    """Sequential V10 policy that reasons from simulator-realized exposure."""

    def __init__(self, policy_input: CausalAlphaV10HierarchyPolicyInput) -> None:
        if not isinstance(policy_input, CausalAlphaV10HierarchyPolicyInput):
            raise TypeError("V10 hierarchy policy input is invalid")
        self.input = policy_input
        self._boundary_mode = policy_input.boundary_mode
        self._economic_config = CausalAlphaV6TargetConfig()
        self._fast_mean, self._fast_uncertainty, self._fast_direction = _qualified(
            policy_input.fast_head_predictions,
            edge_margin=policy_input.config.edge_margin,
        )
        self._slow_mean, self._slow_uncertainty, self._slow_direction = _qualified(
            policy_input.slow_head_predictions,
            edge_margin=policy_input.config.edge_margin,
        )
        self._execution_eligible = (
            (policy_input.liquidity >= policy_input.attribution_liquidity[1])
            & (
                policy_input.realized_volatility
                >= policy_input.attribution_realized_volatility[0]
            )
            & (
                policy_input.realized_volatility
                < policy_input.attribution_realized_volatility[2]
            )
        )
        rows = len(policy_input.decision_indices)
        self._targets = np.empty(rows, dtype=np.float64)
        self._objectives = np.zeros(rows, dtype=np.float64)
        self._entry_objectives = np.zeros(rows, dtype=np.float64)
        self._confirmation_counts = np.zeros(rows, dtype=np.int64)
        self._reasons: list[str] = []
        self._hierarchy_reasons: list[str] = []
        self._slow_states: list[CausalAlphaV6SlowState] = []
        self._offset = 0
        self._last_observed: float | None = None
        self._last_requested: float | None = None
        self._risk_flatten_latched = False
        self._inherited = abs(policy_input.initial_weight) > _OBSERVATION_TOLERANCE
        self._position_origin: str | None = (
            "inherited" if self._inherited else None
        )
        self._boundary_flatten_latched = (
            self._boundary_mode is CausalAlphaV10BoundaryMode.FLATTEN_THEN_RESET
            and self._inherited
        )
        self._inherited_checks = 0
        self._inherited_matches = 0
        self._entry_intent = 0
        self._entry_count = 0
        self._fast_exit_count = 0
        self._fast_neutral_count = 0
        self._slow_exit_count = 0
        self._slow_opposite_count = 0
        self._neutral_slow_count = 0
        self._slow_regime = 0
        self._last_trace_metadata: dict[str, object] | None = None

    @property
    def input_digest(self) -> str:
        return self.input.digest

    @property
    def last_step_trace_metadata(self) -> dict[str, object]:
        """Diagnostics for the most recently emitted action, without changing it."""

        if self._last_trace_metadata is None:
            raise RuntimeError("V10 step trace metadata is unavailable before predict")
        return dict(self._last_trace_metadata)

    def _reset_flat_state(self) -> None:
        self._inherited = False
        self._inherited_checks = 0
        self._inherited_matches = 0
        self._entry_intent = 0
        self._entry_count = 0
        self._fast_exit_count = 0
        self._fast_neutral_count = 0
        self._slow_exit_count = 0
        self._slow_opposite_count = 0
        self._neutral_slow_count = 0
        self._slow_regime = 0

    def _current_weight(self, observation: object) -> float:
        if not isinstance(observation, Mapping) or "current_weights" not in observation:
            raise ValueError("V10 closed-loop observation is missing current_weights")
        values = np.asarray(observation["current_weights"], dtype=np.float64).reshape(
            -1
        )
        if values.shape != (1,) or not np.isfinite(values).all():
            raise ValueError("V10 closed-loop current_weights must be one finite value")
        return float(values[0])

    def _partial_risk_reduction_executable(self, current: float, target: float) -> bool:
        contract = self.input.execution_contract
        if abs(target) <= contract.exit_threshold:
            return False
        if abs(target) < contract.entry_threshold:
            return False
        return abs(target - current) >= contract.no_trade_band

    def _record(
        self,
        *,
        offset: int,
        observed_current: float,
        requested: float,
        reason: str,
        hierarchy_reason: str,
    ) -> tuple[np.ndarray, None]:
        self._targets[offset] = requested
        self._objectives[offset] = causal_alpha_v6_fast_objective(
            observed_current,
            requested,
            float(self._fast_mean[offset]),
            float(self._fast_uncertainty[offset]),
            float(self.input.one_way_cost_rates[offset]),
            self._economic_config,
        )
        trace_origin = self._position_origin
        if hierarchy_reason == "entry" and abs(requested) > _OBSERVATION_TOLERANCE:
            trace_origin = "native_entry"
            self._position_origin = trace_origin
        elif trace_origin is None:
            trace_origin = (
                "inherited"
                if abs(observed_current) > _OBSERVATION_TOLERANCE and self._inherited
                else "flat"
            )
        self._last_trace_metadata = {
            "active_liquidity_caps": float(
                min(
                    self.input.config.target_magnitude,
                    self.input.liquidity_weight_caps[offset],
                )
            ),
            "active_risk_caps": float(
                min(
                    self.input.config.target_magnitude,
                    self.input.risk_weight_caps[offset],
                )
            ),
            "after_cost_entry_objective": float(self._entry_objectives[offset]),
            "fast_edge_margin": float(
                abs(self._fast_mean[offset])
                - self._fast_uncertainty[offset]
                - self.input.config.edge_margin
            ),
            "fast_mean": float(self._fast_mean[offset]),
            "fast_qualified_direction": int(self._fast_direction[offset]),
            "fast_std": float(self._fast_uncertainty[offset]),
            "hierarchy_reason": hierarchy_reason,
            "position_origin": trace_origin,
            "slow_direction": int(self._slow_direction[offset]),
            "slow_mean": float(self._slow_mean[offset]),
            "slow_std": float(self._slow_uncertainty[offset]),
        }
        self._confirmation_counts[offset] = max(
            self._inherited_matches,
            self._entry_count,
            self._fast_exit_count,
            self._slow_exit_count,
            self._neutral_slow_count,
        )
        self._reasons.append(reason)
        self._hierarchy_reasons.append(hierarchy_reason)
        self._slow_states.append(_slow_state(self._slow_regime, requested))
        self._last_observed = observed_current
        self._last_requested = requested
        self._offset += 1
        if (
            abs(observed_current) <= _OBSERVATION_TOLERANCE
            and abs(requested) <= _OBSERVATION_TOLERANCE
        ):
            self._position_origin = None
        return np.asarray([requested], dtype=np.float32), None

    def predict(
        self,
        observation: object,
        deterministic: bool = True,
    ) -> tuple[np.ndarray, None]:
        if not isinstance(deterministic, bool):
            raise TypeError("V10 deterministic flag must be boolean")
        if self._offset >= len(self.input.decision_indices):
            raise RuntimeError("V10 hierarchy policy exhausted its decision rows")
        offset = self._offset
        observed_current = self._current_weight(observation)
        if offset == 0 and not np.isclose(
            observed_current,
            self.input.initial_weight,
            atol=_OBSERVATION_TOLERANCE,
            rtol=0.0,
        ):
            raise ValueError("V10 initial realized weight drifted from frozen contract")

        current_sign = _position_sign(observed_current)
        last_sign = (
            0 if self._last_observed is None else _position_sign(self._last_observed)
        )
        if last_sign != 0 and current_sign == -last_sign:
            raise RuntimeError(
                "V10 realized position flipped without an intervening flat state"
            )
        if self._boundary_flatten_latched:
            if current_sign == 0:
                self._boundary_flatten_latched = False
                self._risk_flatten_latched = False
                self._reset_flat_state()
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="hold_flat",
                    hierarchy_reason="realized_state_reset",
                )
            return self._record(
                offset=offset,
                observed_current=observed_current,
                requested=0.0,
                reason="risk_projection",
                hierarchy_reason="realized_state_reset",
            )
        external_flatten = (
            last_sign != 0
            and current_sign == 0
            and self._last_requested is not None
            and abs(self._last_requested) > _OBSERVATION_TOLERANCE
        )
        if external_flatten:
            self._risk_flatten_latched = False
            self._reset_flat_state()
            return self._record(
                offset=offset,
                observed_current=observed_current,
                requested=0.0,
                reason="hold_flat",
                hierarchy_reason="realized_state_reset",
            )

        config = self.input.config
        fast_only_ownership = (
            self._boundary_mode is CausalAlphaV10BoundaryMode.FAST_ONLY_OWNERSHIP
        )
        liquidity_cap = min(
            config.target_magnitude,
            float(self.input.liquidity_weight_caps[offset]),
        )
        risk_cap = min(
            config.target_magnitude,
            float(self.input.risk_weight_caps[offset]),
        )
        cadence = (
            int(self.input.decision_indices[offset]) % config.fast_horizon_decisions
            == 0
        )

        if self._risk_flatten_latched:
            if current_sign == 0:
                self._risk_flatten_latched = False
                self._reset_flat_state()
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="hold_flat",
                    hierarchy_reason="realized_state_reset",
                )
            if (
                self._boundary_mode
                is CausalAlphaV10BoundaryMode.FLATTEN_ON_RISK_BREACH
            ):
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="risk_projection",
                    hierarchy_reason="risk_cap_flatten",
                )
            if abs(observed_current) > risk_cap + _OBSERVATION_TOLERANCE:
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="risk_projection",
                    hierarchy_reason="risk_cap_flatten",
                )
            self._risk_flatten_latched = False

        decision_current = observed_current
        requested = observed_current
        reason = "hold_position" if current_sign != 0 else "hold_flat"
        hierarchy_reason = reason
        risk_projected = False

        if abs(observed_current) > risk_cap + _OBSERVATION_TOLERANCE:
            if (
                self._boundary_mode
                is CausalAlphaV10BoundaryMode.FLATTEN_ON_RISK_BREACH
            ):
                self._risk_flatten_latched = True
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="risk_projection",
                    hierarchy_reason="risk_cap_flatten",
                )
            partial = float(np.sign(observed_current) * risk_cap)
            if not self._partial_risk_reduction_executable(
                observed_current,
                partial,
            ):
                self._risk_flatten_latched = True
                return self._record(
                    offset=offset,
                    observed_current=observed_current,
                    requested=0.0,
                    reason="risk_projection",
                    hierarchy_reason="risk_cap_flatten",
                )
            decision_current = partial
            requested = partial
            reason = "risk_projection"
            hierarchy_reason = "risk_cap_projection"
            risk_projected = True

        decision_sign = _position_sign(decision_current)
        if (
            not risk_projected
            and abs(decision_current) > liquidity_cap + _OBSERVATION_TOLERANCE
        ):
            hierarchy_reason = "liquidity_capacity_hold"

        if cadence and bool(self.input.actionable_mask[offset]):
            fast = int(self._fast_direction[offset])
            observed_slow = int(self._slow_direction[offset])
            if self._inherited and decision_sign != 0:
                self._inherited_checks += 1
                coherent_inherited = (
                    fast == decision_sign
                    if fast_only_ownership
                    else fast == observed_slow == decision_sign
                )
                if coherent_inherited and bool(
                    self._execution_eligible[offset]
                ):
                    self._inherited_matches += 1
                if self._inherited_checks >= config.entry_confirmation_count:
                    if self._inherited_matches < config.entry_confirmation_count:
                        requested = 0.0
                        reason = hierarchy_reason = "exit"
                    elif not risk_projected:
                        if fast_only_ownership:
                            reason = "hold_position"
                            hierarchy_reason = "fast_support_hold"
                        else:
                            reason = hierarchy_reason = "slow_support_hold"
                    self._inherited = False
                    self._inherited_checks = 0
                    self._inherited_matches = 0
                elif not risk_projected:
                    reason = hierarchy_reason = "confirmation_hold"
            elif decision_sign == 0:
                if observed_slow != 0:
                    self._slow_regime = observed_slow
                coherent = (
                    fast
                    if fast != 0
                    and (
                        fast_only_ownership
                        or fast == self._slow_regime
                    )
                    and bool(self._execution_eligible[offset])
                    else 0
                )
                cap = min(liquidity_cap, risk_cap)
                entry_target = float(coherent * cap)
                entry_objective = causal_alpha_v6_fast_objective(
                    0.0,
                    entry_target,
                    float(self._fast_mean[offset]),
                    float(self._fast_uncertainty[offset]),
                    float(self.input.one_way_cost_rates[offset]),
                    self._economic_config,
                )
                self._entry_objectives[offset] = entry_objective
                if coherent == 0 or entry_objective <= _EPSILON:
                    self._entry_intent = 0
                    self._entry_count = 0
                    reason = hierarchy_reason = "cost_or_uncertainty_hold"
                elif abs(entry_target) < max(
                    self.input.execution_contract.entry_threshold,
                    self.input.execution_contract.no_trade_band,
                ):
                    self._entry_intent = 0
                    self._entry_count = 0
                    reason = "hold_flat"
                    hierarchy_reason = "entry_floor_hold"
                else:
                    if coherent == self._entry_intent:
                        self._entry_count += 1
                    else:
                        self._entry_intent = coherent
                        self._entry_count = 1
                    if self._entry_count >= config.entry_confirmation_count:
                        requested = entry_target
                        reason = hierarchy_reason = "entry"
                        self._entry_intent = 0
                        self._entry_count = 0
                        self._fast_exit_count = 0
                        self._slow_exit_count = 0
                        self._neutral_slow_count = 0
                    else:
                        reason = hierarchy_reason = "confirmation_hold"
            else:
                neutral_fast_expired = False
                if self._boundary_mode in (
                    CausalAlphaV10BoundaryMode.NEUTRAL_FAST_EXPIRY,
                    CausalAlphaV10BoundaryMode.FAST_ONLY_OWNERSHIP,
                ):
                    if risk_projected:
                        self._fast_neutral_count = 0
                    elif fast == 0:
                        self._fast_neutral_count += 1
                    else:
                        self._fast_neutral_count = 0
                    neutral_fast_expired = (
                        self._fast_neutral_count
                        >= config.slow_neutral_expiry_count
                    )
                if observed_slow == decision_sign:
                    self._slow_regime = observed_slow
                    self._slow_opposite_count = 0
                    self._neutral_slow_count = 0
                elif observed_slow == -decision_sign:
                    self._slow_opposite_count += 1
                    self._neutral_slow_count = 0
                else:
                    self._slow_opposite_count = 0
                    self._neutral_slow_count += 1
                self._fast_exit_count = (
                    self._fast_exit_count + 1 if fast == -decision_sign else 0
                )
                self._slow_exit_count = self._slow_opposite_count
                should_exit = (
                    neutral_fast_expired
                    or self._fast_exit_count >= config.exit_confirmation_count
                    or (
                        not fast_only_ownership
                        and (
                            self._slow_exit_count >= config.exit_confirmation_count
                            or self._neutral_slow_count
                            >= config.slow_neutral_expiry_count
                        )
                    )
                )
                if should_exit:
                    requested = 0.0
                    reason = "exit"
                    hierarchy_reason = (
                        "neutral_fast_expiry" if neutral_fast_expired else "exit"
                    )
                    self._fast_exit_count = 0
                    self._fast_neutral_count = 0
                    self._slow_exit_count = 0
                    self._slow_opposite_count = 0
                    self._neutral_slow_count = 0
                    self._slow_regime = 0
                elif (
                    not risk_projected and hierarchy_reason != "liquidity_capacity_hold"
                ):
                    if fast_only_ownership:
                        reason = "hold_position"
                        hierarchy_reason = "fast_support_hold"
                    else:
                        reason = hierarchy_reason = (
                            "slow_support_hold"
                            if self._slow_regime == decision_sign
                            else "confirmation_hold"
                        )
        elif cadence:
            if not risk_projected and hierarchy_reason != "liquidity_capacity_hold":
                reason = hierarchy_reason = "unactionable_hold"
        elif not risk_projected and hierarchy_reason != "liquidity_capacity_hold":
            reason = hierarchy_reason = "cadence_hold"

        return self._record(
            offset=offset,
            observed_current=observed_current,
            requested=requested,
            reason=reason,
            hierarchy_reason=hierarchy_reason,
        )

    def result(self) -> CausalAlphaV10HierarchyResult:
        if self._offset != len(self.input.decision_indices):
            raise RuntimeError("V10 hierarchy result requested before replay completed")
        targets = self._targets.copy()
        previous = np.concatenate(([self.input.initial_weight], targets[:-1]))
        forecast_digest = content_and_arrays_digest(
            {
                "dual_fit_digest": self.input.dual_fit_digest,
                "schema_version": "causal_alpha_v10_hierarchical_forecast_v2",
                "source_forecast_digest": self.input.source_forecast_digest,
            },
            (
                ("fast_head_predictions", self.input.fast_head_predictions),
                ("slow_head_predictions", self.input.slow_head_predictions),
            ),
        )
        counts = tuple(
            sorted(
                (reason, self._reasons.count(reason)) for reason in set(self._reasons)
            )
        )
        path = CausalAlphaV6TargetPath(
            candidate=CausalAlphaV6Candidate.FAST_ONLY,
            initial_weight=float(self.input.initial_weight),
            decision_indices=self.input.decision_indices,
            targets=targets,
            fast_proposals=(
                self._fast_direction.astype(np.float64)
                * self.input.config.target_magnitude
            ),
            expected_returns_4h=self._fast_mean,
            expected_returns_24h=np.zeros(len(targets)),
            expected_returns_72h=self._slow_mean,
            direction_scores_4h=self._fast_direction.astype(np.float64),
            uncertainties_4h=self._fast_uncertainty,
            one_way_cost_rates=self.input.one_way_cost_rates,
            liquidity_weight_caps=self.input.liquidity_weight_caps,
            risk_weight_caps=self.input.risk_weight_caps,
            objectives=self._objectives,
            confirmation_counts=self._confirmation_counts,
            actionable_mask=self.input.actionable_mask,
            slow_states=tuple(self._slow_states),
            reasons=tuple(self._reasons),
            reason_counts=counts,
            submitted_change_count=int(
                np.count_nonzero(np.abs(targets - previous) > _EPSILON)
            ),
            sign_flip_count=int(np.count_nonzero(targets * previous < 0.0)),
            liquidity_deleveraging_count=self._reasons.count("liquidity_deleverage"),
            risk_projection_count=self._reasons.count("risk_projection"),
            forecast_digest=forecast_digest,
            config_digest=self.input.compiler_config_digest,
        )
        hierarchy_counts = tuple(
            sorted(
                (reason, self._hierarchy_reasons.count(reason))
                for reason in set(self._hierarchy_reasons)
            )
        )
        return CausalAlphaV10HierarchyResult(
            v6_target_path=path,
            input_digest=self.input.digest,
            hierarchy_reasons=tuple(self._hierarchy_reasons),
            hierarchy_reason_counts=hierarchy_counts,
        )


def prepare_causal_alpha_v10_hierarchy_policy(
    *,
    decision_indices: object,
    fast_head_predictions: object,
    slow_head_predictions: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object,
    realized_volatility: object,
    liquidity: object,
    attribution_boundaries: _AttributionBoundaries,
    actionable_mask: object,
    source_forecast_digest: str,
    dual_fit_digest: str,
    config: CausalAlphaV10Config,
    initial_weight: float,
    execution_contract: CausalAlphaV10ExecutionContract,
    boundary_mode: CausalAlphaV10BoundaryMode = CausalAlphaV10BoundaryMode.INHERIT_CONFIRM,
) -> CausalAlphaV10HierarchyPolicy:
    """Prepare one immutable-input V10 policy for simulator-driven replay."""

    return CausalAlphaV10HierarchyPolicy(
        _policy_input(
            decision_indices=decision_indices,
            fast_head_predictions=fast_head_predictions,
            slow_head_predictions=slow_head_predictions,
            one_way_cost_rates=one_way_cost_rates,
            liquidity_weight_caps=liquidity_weight_caps,
            risk_weight_caps=risk_weight_caps,
            realized_volatility=realized_volatility,
            liquidity=liquidity,
            attribution_boundaries=attribution_boundaries,
            actionable_mask=actionable_mask,
            source_forecast_digest=source_forecast_digest,
            dual_fit_digest=dual_fit_digest,
            config=config,
            initial_weight=initial_weight,
            execution_contract=execution_contract,
            boundary_mode=boundary_mode,
        )
    )


def causal_alpha_v10_hierarchical_target_path(
    *,
    decision_indices: object,
    fast_head_predictions: object,
    slow_head_predictions: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object,
    realized_volatility: object,
    liquidity: object,
    attribution_boundaries: _AttributionBoundaries,
    actionable_mask: object,
    source_forecast_digest: str,
    dual_fit_digest: str,
    config: CausalAlphaV10Config,
    initial_weight: float,
    execution_entry_threshold: float,
    execution_no_trade_band: float,
    execution_exit_threshold: float = 0.0,
    boundary_mode: CausalAlphaV10BoundaryMode = CausalAlphaV10BoundaryMode.INHERIT_CONFIRM,
) -> CausalAlphaV6TargetPath:
    """Compatibility harness that drives the closed-loop policy open-loop."""

    contract = CausalAlphaV10ExecutionContract(
        max_gross=1.0,
        max_abs_weight=1.0,
        max_turnover=2.0,
        entry_threshold=float(execution_entry_threshold),
        exit_threshold=float(execution_exit_threshold),
        no_trade_band=float(execution_no_trade_band),
    )
    policy = prepare_causal_alpha_v10_hierarchy_policy(
        decision_indices=decision_indices,
        fast_head_predictions=fast_head_predictions,
        slow_head_predictions=slow_head_predictions,
        one_way_cost_rates=one_way_cost_rates,
        liquidity_weight_caps=liquidity_weight_caps,
        risk_weight_caps=risk_weight_caps,
        realized_volatility=realized_volatility,
        liquidity=liquidity,
        attribution_boundaries=attribution_boundaries,
        actionable_mask=actionable_mask,
        source_forecast_digest=source_forecast_digest,
        dual_fit_digest=dual_fit_digest,
        config=config,
        initial_weight=initial_weight,
        execution_contract=contract,
        boundary_mode=boundary_mode,
    )
    realized = float(initial_weight)
    for _ in range(len(policy.input.decision_indices)):
        action, _state = policy.predict(
            {"current_weights": np.asarray([realized], dtype=np.float64)},
            deterministic=True,
        )
        realized = float(policy._targets[policy._offset - 1])
    return policy.result().v6_target_path


__all__ = [
    "CausalAlphaV10BoundaryMode",
    "CausalAlphaV10ExecutionContract",
    "CausalAlphaV10HierarchyPolicy",
    "CausalAlphaV10HierarchyPolicyInput",
    "CausalAlphaV10HierarchyResult",
    "causal_alpha_v10_hierarchical_target_path",
    "prepare_causal_alpha_v10_hierarchy_policy",
]
