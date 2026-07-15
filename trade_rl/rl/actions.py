"""Baseline-anchored residual action specifications and composition."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from trade_rl.strategies.trend import TrendTargets

LEGACY_ACTION_SCHEMA = "baseline_residual_v1"
ACTION_SCHEMA = "portfolio_action_v3"


class ActionMode(str, Enum):
    """Policy output semantics bound into the environment identity."""

    RESIDUAL = "residual"
    TARGET_WEIGHT = "target_weight"


class AlphaSignalKind(str, Enum):
    """Semantic contract for one alpha-provider vector."""

    DIRECTION = "direction"
    DIRECTION_CONFIDENCE = "direction_confidence"
    EXPECTED_RETURN = "expected_return"
    TARGET_WEIGHT = "target_weight"


@dataclass(frozen=True, slots=True)
class AlphaContract:
    """Convert provider outputs into a bounded portfolio residual direction."""

    kind: AlphaSignalKind | str = AlphaSignalKind.TARGET_WEIGHT
    expected_return_scale: float = 0.01
    max_gross: float = 1.0

    def __post_init__(self) -> None:
        try:
            kind = AlphaSignalKind(self.kind)
        except ValueError as error:
            raise ValueError("alpha signal kind is not supported") from error
        if (
            not np.isfinite(self.expected_return_scale)
            or self.expected_return_scale <= 0.0
        ):
            raise ValueError("expected_return_scale must be finite and positive")
        if not np.isfinite(self.max_gross) or not 0.0 < self.max_gross <= 1.0:
            raise ValueError("alpha max_gross must be within (0, 1]")
        object.__setattr__(self, "kind", kind)

    def prepare(self, value: np.ndarray, *, n_symbols: int) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (n_symbols,) or not np.isfinite(vector).all():
            raise ValueError("alpha vector does not match symbols")
        kind = AlphaSignalKind(self.kind)
        if kind is AlphaSignalKind.DIRECTION:
            prepared = np.sign(vector)
        elif kind is AlphaSignalKind.DIRECTION_CONFIDENCE:
            prepared = np.clip(vector, -1.0, 1.0)
        elif kind is AlphaSignalKind.EXPECTED_RETURN:
            prepared = np.tanh(vector / self.expected_return_scale)
        else:
            prepared = vector.copy()
        return _normalize_gross(prepared, maximum=self.max_gross)


class ActionValidationMode(str, Enum):
    """How out-of-range policy outputs are handled."""

    CLIP = "clip"
    STRICT = "strict"
    FAIL_CLOSED = "fail_closed"


def _normalize_gross(value: np.ndarray, *, maximum: float = 1.0) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float64).reshape(-1).copy()
    if not np.isfinite(vector).all():
        raise ValueError("proposal weights must be finite")
    gross = float(np.abs(vector).sum())
    if gross > maximum and gross > 0.0:
        vector *= maximum / gross
    return vector


def _normalize_basis(value: np.ndarray, *, n_symbols: int) -> np.ndarray:
    basis = np.asarray(value, dtype=np.float64)
    if basis.size == 0:
        return np.empty((0, n_symbols), dtype=np.float64)
    if basis.ndim != 2 or basis.shape[1] != n_symbols:
        raise ValueError("factor basis must have shape (n_factors, n_symbols)")
    if not np.isfinite(basis).all():
        raise ValueError("factor basis must be finite")
    normalized = basis.copy()
    for row_index in range(normalized.shape[0]):
        gross = float(np.abs(normalized[row_index]).sum())
        if gross > 1.0:
            normalized[row_index] /= gross
    return normalized


@dataclass(frozen=True, slots=True)
class ResidualAction:
    """Legacy two-dimensional action retained for artifact migration."""

    trend_mix: float
    alpha_budget: float

    def __post_init__(self) -> None:
        for field_name, value in (
            ("trend_mix", self.trend_mix),
            ("alpha_budget", self.alpha_budget),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
            if not -1.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be within [-1, 1]")

    @classmethod
    def from_array(cls, value: np.ndarray) -> ResidualAction:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (2,):
            raise ValueError("residual action requires exactly two values")
        if not np.isfinite(vector).all():
            raise ValueError("residual action values must be finite")
        clipped = np.clip(vector, -1.0, 1.0)
        return cls(
            trend_mix=float(clipped[0]),
            alpha_budget=float(clipped[1]),
        )

    def as_array(self) -> np.ndarray:
        return np.array([self.trend_mix, self.alpha_budget], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class TargetWeightAction:
    """Direct per-symbol portfolio targets emitted by the policy."""

    weights: np.ndarray
    saturated_count: int = 0
    raw_max_abs: float = 0.0

    def __post_init__(self) -> None:
        weights = np.asarray(self.weights, dtype=np.float64).reshape(-1).copy()
        if weights.size == 0:
            raise ValueError("target weights must not be empty")
        if not np.isfinite(weights).all():
            raise ValueError("target weights must be finite")
        if np.any(np.abs(weights) > 1.0):
            raise ValueError("target weights must be within [-1, 1]")
        if self.saturated_count < 0:
            raise ValueError("saturated_count must be non-negative")
        if not np.isfinite(self.raw_max_abs) or self.raw_max_abs < 0.0:
            raise ValueError("raw_max_abs must be finite and non-negative")
        weights.setflags(write=False)
        object.__setattr__(self, "weights", weights)

    def as_array(self) -> np.ndarray:
        return self.weights.astype(np.float32, copy=True)


@dataclass(frozen=True, slots=True)
class ActionSpec:
    """Exact maintained action layout for one environment identity."""

    mode: ActionMode | str = ActionMode.RESIDUAL
    alpha_enabled: bool = False
    risk_tilt_enabled: bool = True
    n_factors: int = 0
    target_weight_count: int = 0
    validation_mode: ActionValidationMode | str = ActionValidationMode.CLIP

    def __post_init__(self) -> None:
        try:
            action_mode = ActionMode(self.mode)
        except ValueError as error:
            raise ValueError("action mode is not supported") from error
        object.__setattr__(self, "mode", action_mode)
        if not isinstance(self.alpha_enabled, bool):
            raise ValueError("alpha_enabled must be a boolean")
        if not isinstance(self.risk_tilt_enabled, bool):
            raise ValueError("risk_tilt_enabled must be a boolean")
        if (
            isinstance(self.n_factors, bool)
            or not isinstance(self.n_factors, int)
            or self.n_factors < 0
        ):
            raise ValueError("n_factors must be a non-negative integer")
        if (
            isinstance(self.target_weight_count, bool)
            or not isinstance(self.target_weight_count, int)
            or self.target_weight_count < 0
        ):
            raise ValueError("target_weight_count must be a non-negative integer")
        if action_mode is ActionMode.TARGET_WEIGHT:
            if self.alpha_enabled or self.risk_tilt_enabled or self.n_factors:
                raise ValueError("target_weight mode does not accept residual controls")
            if self.target_weight_count <= 0:
                raise ValueError("target_weight mode requires positive target_weight_count")
        elif self.target_weight_count:
            raise ValueError("residual mode does not accept target_weight_count")
        try:
            mode = ActionValidationMode(self.validation_mode)
        except ValueError as error:
            raise ValueError("action validation mode is not supported") from error
        object.__setattr__(self, "validation_mode", mode)

    @property
    def names(self) -> tuple[str, ...]:
        if self.mode is ActionMode.TARGET_WEIGHT:
            return tuple(
                f"target_weight:{index}" for index in range(self.target_weight_count)
            )
        names = ["fast_tilt", "slow_tilt"]
        if self.risk_tilt_enabled:
            names.append("risk_tilt")
        if self.alpha_enabled:
            names.append("alpha_scale")
        names.extend(f"factor_{index}" for index in range(self.n_factors))
        return tuple(names)

    def names_for_symbols(self, symbols: tuple[str, ...]) -> tuple[str, ...]:
        if self.mode is ActionMode.TARGET_WEIGHT:
            if len(symbols) != self.target_weight_count:
                raise ValueError("target weight count does not match dataset symbols")
            return tuple(f"target_weight:{symbol}" for symbol in symbols)
        return self.names

    @property
    def size(self) -> int:
        return len(self.names)

    def parse(
        self,
        value: np.ndarray,
        *,
        mode: ActionValidationMode | str | None = None,
    ) -> ResidualActionV2 | TargetWeightAction:
        vector = np.asarray(value, dtype=np.float64).reshape(-1)
        if vector.shape != (self.size,):
            raise ValueError(
                f"action requires exactly {self.size} values for {self.names}"
            )
        if not np.isfinite(vector).all():
            raise ValueError("action values must be finite")
        resolved_mode = (
            ActionValidationMode(self.validation_mode)
            if mode is None
            else ActionValidationMode(mode)
        )
        outside = np.abs(vector) > 1.0
        if np.any(outside) and resolved_mode is not ActionValidationMode.CLIP:
            label = (
                "serving action failed closed"
                if resolved_mode is ActionValidationMode.FAIL_CLOSED
                else "action is outside [-1, 1]"
            )
            raise ValueError(label)
        clipped = np.clip(vector, -1.0, 1.0)
        if self.mode is ActionMode.TARGET_WEIGHT:
            return TargetWeightAction(
                weights=clipped,
                saturated_count=int(np.count_nonzero(outside)),
                raw_max_abs=float(np.max(np.abs(vector), initial=0.0)),
            )
        cursor = 0
        fast_tilt = float(clipped[cursor])
        cursor += 1
        slow_tilt = float(clipped[cursor])
        cursor += 1
        risk_tilt = 0.0
        if self.risk_tilt_enabled:
            risk_tilt = float(clipped[cursor])
            cursor += 1
        alpha_scale = 0.0
        if self.alpha_enabled:
            alpha_scale = float(clipped[cursor])
            cursor += 1
        factors = clipped[cursor:].copy()
        return ResidualActionV2(
            fast_tilt=fast_tilt,
            slow_tilt=slow_tilt,
            risk_tilt=risk_tilt,
            alpha_scale=alpha_scale,
            factor_tilts=factors,
            saturated_count=int(np.count_nonzero(outside)),
            raw_max_abs=float(np.max(np.abs(vector), initial=0.0)),
        )


@dataclass(frozen=True, slots=True)
class ResidualActionV2:
    fast_tilt: float
    slow_tilt: float
    risk_tilt: float
    alpha_scale: float = 0.0
    factor_tilts: np.ndarray = field(
        default_factory=lambda: np.empty(0, dtype=np.float64)
    )
    saturated_count: int = 0
    raw_max_abs: float = 0.0

    def __post_init__(self) -> None:
        for field_name, value in (
            ("fast_tilt", self.fast_tilt),
            ("slow_tilt", self.slow_tilt),
            ("risk_tilt", self.risk_tilt),
            ("alpha_scale", self.alpha_scale),
            ("raw_max_abs", self.raw_max_abs),
        ):
            if not np.isfinite(value):
                raise ValueError(f"{field_name} must be finite")
        for value in (
            self.fast_tilt,
            self.slow_tilt,
            self.risk_tilt,
            self.alpha_scale,
        ):
            if not -1.0 <= value <= 1.0:
                raise ValueError("maintained action values must be within [-1, 1]")
        factors = np.asarray(self.factor_tilts, dtype=np.float64).reshape(-1).copy()
        if not np.isfinite(factors).all() or np.any(np.abs(factors) > 1.0):
            raise ValueError("factor tilts must be finite and within [-1, 1]")
        if self.saturated_count < 0:
            raise ValueError("saturated_count must be non-negative")
        factors.setflags(write=False)
        object.__setattr__(self, "factor_tilts", factors)

    def as_array(
        self,
        *,
        alpha_enabled: bool,
        risk_tilt_enabled: bool = True,
    ) -> np.ndarray:
        values = [self.fast_tilt, self.slow_tilt]
        if risk_tilt_enabled:
            values.append(self.risk_tilt)
        if alpha_enabled:
            values.append(self.alpha_scale)
        values.extend(float(value) for value in self.factor_tilts)
        return np.asarray(values, dtype=np.float32)


@dataclass(frozen=True, slots=True)
class ResidualComposition:
    action: ResidualAction | ResidualActionV2 | TargetWeightAction
    baseline: np.ndarray
    trend_component: np.ndarray
    alpha_component: np.ndarray
    proposal: np.ndarray
    factor_component: np.ndarray | None = None
    residual_component: np.ndarray | None = None
    raw_gross: float = 0.0
    target_gross: float = 0.0


class BaselineResidualComposer:
    """Compose proposals while preserving zero-action baseline identity."""

    def compose(
        self,
        action: ResidualAction | ResidualActionV2 | TargetWeightAction,
        trends: TrendTargets,
        alpha: np.ndarray,
        *,
        alpha_enabled: bool,
        factor_basis: np.ndarray | None = None,
        max_gross: float = 1.0,
    ) -> ResidualComposition:
        if isinstance(action, TargetWeightAction):
            return self._compose_target(action, trends, max_gross=max_gross)
        if isinstance(action, ResidualAction):
            return self._compose_legacy(
                action,
                trends,
                alpha,
                alpha_enabled=alpha_enabled,
            )
        return self._compose_v2(
            action,
            trends,
            alpha,
            alpha_enabled=alpha_enabled,
            factor_basis=factor_basis,
            max_gross=max_gross,
        )

    @staticmethod
    def _compose_target(
        action: TargetWeightAction,
        trends: TrendTargets,
        *,
        max_gross: float,
    ) -> ResidualComposition:
        if action.weights.shape != trends.base.shape:
            raise ValueError("target weight count does not match trend targets")
        if not np.isfinite(max_gross) or not 0.0 < max_gross <= 10.0:
            raise ValueError("max_gross must be within (0, 10]")
        raw_gross = float(np.abs(action.weights).sum())
        proposal = _normalize_gross(action.weights, maximum=max_gross)
        zeros = np.zeros_like(trends.base)
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=zeros.copy(),
            alpha_component=zeros.copy(),
            factor_component=zeros.copy(),
            residual_component=proposal - trends.base,
            proposal=proposal,
            raw_gross=raw_gross,
            target_gross=float(np.abs(proposal).sum()),
        )

    def _compose_legacy(
        self,
        action: ResidualAction,
        trends: TrendTargets,
        alpha: np.ndarray,
        *,
        alpha_enabled: bool,
    ) -> ResidualComposition:
        alpha_vector = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_vector.shape != trends.base.shape:
            raise ValueError("alpha vector shape must match trend targets")
        if not np.isfinite(alpha_vector).all():
            raise ValueError("alpha vector must be finite")
        alpha_vector = _normalize_gross(alpha_vector)
        if action.trend_mix >= 0.0:
            trend = trends.base + action.trend_mix * (trends.fast - trends.base)
        else:
            trend = trends.base + (-action.trend_mix) * (trends.slow - trends.base)
        alpha_component = (
            action.alpha_budget * alpha_vector
            if alpha_enabled
            else np.zeros_like(trends.base)
        )
        proposal = _normalize_gross(trend + alpha_component)
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=trend,
            alpha_component=alpha_component,
            factor_component=np.zeros_like(trends.base),
            residual_component=proposal - trends.base,
            proposal=proposal,
            raw_gross=float(np.abs(trend + alpha_component).sum()),
            target_gross=float(np.abs(proposal).sum()),
        )

    def _compose_v2(
        self,
        action: ResidualActionV2,
        trends: TrendTargets,
        alpha: np.ndarray,
        *,
        alpha_enabled: bool,
        factor_basis: np.ndarray | None,
        max_gross: float,
    ) -> ResidualComposition:
        if not np.isfinite(max_gross) or not 0.0 < max_gross <= 10.0:
            raise ValueError("max_gross must be within (0, 10]")
        alpha_vector = np.asarray(alpha, dtype=np.float64).reshape(-1)
        if alpha_vector.shape != trends.base.shape:
            raise ValueError("alpha vector shape must match trend targets")
        if not np.isfinite(alpha_vector).all():
            raise ValueError("alpha vector must be finite")
        alpha_vector = _normalize_gross(alpha_vector)
        basis = _normalize_basis(
            np.empty((0, trends.base.size)) if factor_basis is None else factor_basis,
            n_symbols=trends.base.size,
        )
        if basis.shape[0] != action.factor_tilts.size:
            raise ValueError("factor action count does not match factor basis")

        trend_coefficients = np.array(
            [action.fast_tilt, action.slow_tilt],
            dtype=np.float64,
        )
        coefficient_l1 = float(np.abs(trend_coefficients).sum())
        if coefficient_l1 > 1.0:
            trend_coefficients /= coefficient_l1
        trend_residual = trend_coefficients[0] * (
            trends.fast - trends.base
        ) + trend_coefficients[1] * (trends.slow - trends.base)
        trend = trends.base + trend_residual
        alpha_component = (
            action.alpha_scale * alpha_vector
            if alpha_enabled
            else np.zeros_like(trends.base)
        )
        factor_coefficients = action.factor_tilts.copy()
        factor_l1 = float(np.abs(factor_coefficients).sum())
        if factor_l1 > 1.0:
            factor_coefficients /= factor_l1
        factor_component = (
            factor_coefficients @ basis
            if basis.shape[0]
            else np.zeros_like(trends.base)
        )
        raw = trend + alpha_component + factor_component
        raw_gross = float(np.abs(raw).sum())
        target_gross = min(raw_gross, max_gross)
        if action.risk_tilt < 0.0:
            target_gross *= 1.0 + action.risk_tilt
        elif action.risk_tilt > 0.0:
            target_gross += action.risk_tilt * (max_gross - target_gross)
        if raw_gross > 0.0:
            proposal = raw * (target_gross / raw_gross)
        else:
            proposal = raw
        proposal = _normalize_gross(proposal, maximum=max_gross)
        return ResidualComposition(
            action=action,
            baseline=trends.base.copy(),
            trend_component=trend,
            alpha_component=alpha_component,
            factor_component=factor_component,
            residual_component=proposal - trends.base,
            proposal=proposal,
            raw_gross=raw_gross,
            target_gross=float(np.abs(proposal).sum()),
        )
