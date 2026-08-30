"""Independent Causal Alpha V11 target compilers and trace policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.data.identity import content_and_arrays_digest
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v9 import CausalAlphaV9Config
from trade_rl.learning.causal_alpha_v9_wave import causal_alpha_v9_wave_target_path
from trade_rl.learning.causal_alpha_v11 import (
    CausalAlphaV11Candidate,
    CausalAlphaV11Config,
    CausalAlphaV11StudyArm,
    CausalAlphaV11TargetPath,
)
from trade_rl.learning.causal_alpha_v11_calibration import (
    CausalAlphaV11SignCalibration,
)

_COMPILED_SCHEMA: Final = "causal_alpha_v11_compiled_target_v1"
_EPSILON: Final = 1e-12


def _readonly(value: object, *, dtype: Any, rows: int, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1).copy(order="C")
    if array.shape != (rows,):
        raise ValueError(f"V11 {field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"V11 {field} must be finite")
    array.setflags(write=False)
    return array


@dataclass(frozen=True, slots=True)
class CausalAlphaV11CompiledTarget:
    """One target path plus complete decision-aligned diagnostic metadata."""

    target: CausalAlphaV11TargetPath
    fast_means: np.ndarray
    fast_uncertainties: np.ndarray
    fast_qualified_directions: np.ndarray
    raw_edges: np.ndarray
    after_cost_entry_objectives: np.ndarray
    policy_reasons: tuple[str, ...]
    position_origins: tuple[str, ...]
    schema_version: str = _COMPILED_SCHEMA
    digest: str = ""

    def __post_init__(self) -> None:
        rows = len(self.target.v6_target_path.decision_indices)
        arrays = {
            "fast_means": _readonly(
                self.fast_means, dtype=np.float64, rows=rows, field="fast means"
            ),
            "fast_uncertainties": _readonly(
                self.fast_uncertainties,
                dtype=np.float64,
                rows=rows,
                field="fast uncertainties",
            ),
            "fast_qualified_directions": _readonly(
                self.fast_qualified_directions,
                dtype=np.int8,
                rows=rows,
                field="fast qualified directions",
            ),
            "raw_edges": _readonly(
                self.raw_edges, dtype=np.float64, rows=rows, field="raw edges"
            ),
            "after_cost_entry_objectives": _readonly(
                self.after_cost_entry_objectives,
                dtype=np.float64,
                rows=rows,
                field="after-cost entry objectives",
            ),
        }
        if len(self.policy_reasons) != rows or len(self.position_origins) != rows:
            raise ValueError("V11 trace metadata must align with decisions")
        if self.schema_version != _COMPILED_SCHEMA:
            raise ValueError("unsupported V11 compiled target schema")
        for name, array in arrays.items():
            object.__setattr__(self, name, array)
        expected = content_and_arrays_digest(
            {
                "policy_reasons": self.policy_reasons,
                "position_origins": self.position_origins,
                "schema_version": self.schema_version,
                "target_digest": self.target.digest,
            },
            tuple(arrays.items()),
        )
        if self.digest and self.digest != expected:
            raise ValueError("V11 compiled target digest mismatch")
        object.__setattr__(self, "digest", expected)


def _qualified(
    heads: np.ndarray, *, edge_margin: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mean = np.mean(heads, axis=0, dtype=np.float64)
    uncertainty = np.std(heads, axis=0, dtype=np.float64)
    signs = np.sign(heads)
    agreed = np.all(signs == signs[0], axis=0) & (signs[0] != 0.0)
    raw_edge = np.abs(mean) - uncertainty - edge_margin
    direction = np.where(agreed & (raw_edge > 0.0), np.sign(mean), 0.0).astype(np.int8)
    return mean, uncertainty, raw_edge, direction


def _calibrated_edge(
    calibration: CausalAlphaV11SignCalibration, direction: int, raw_edge: float
) -> float:
    value = calibration.calibrated_edge(direction=direction, raw_edge=raw_edge)
    if not np.isfinite(value):
        raise ValueError("V11 calibrated edge must be finite")
    return value


def _compile_treatment_path(
    *,
    study_arm: CausalAlphaV11StudyArm,
    decisions: np.ndarray,
    mean: np.ndarray,
    uncertainty: np.ndarray,
    raw_edge: np.ndarray,
    qualified: np.ndarray,
    costs: np.ndarray,
    liquidity: np.ndarray,
    risk: np.ndarray,
    actionable: np.ndarray,
    source_forecast_digest: str,
    v9_config: CausalAlphaV9Config,
    v11_config: CausalAlphaV11Config,
    initial_weight: float,
    sign_calibration: CausalAlphaV11SignCalibration | None,
) -> tuple[CausalAlphaV6TargetPath, tuple[str, ...], tuple[str, ...]]:
    rows = len(decisions)
    if (
        study_arm
        in (
            CausalAlphaV11StudyArm.SIGN_CALIBRATED_ENTRY,
            CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
        )
        and sign_calibration is None
    ):
        raise ValueError(f"V11 {study_arm.value} requires sign calibration")

    targets = np.empty(rows, dtype=np.float64)
    objectives = np.zeros(rows, dtype=np.float64)
    confirmation_counts = np.zeros(rows, dtype=np.int64)
    reasons: list[str] = []
    policy_reasons: list[str] = []
    position_origins: list[str] = []
    current = float(
        np.clip(initial_weight, -v9_config.target_magnitude, v9_config.target_magnitude)
    )
    inherited = abs(initial_weight) > _EPSILON
    origin = "inherited" if inherited else "flat"
    intent = 0
    count = 0
    neutral_count = 0
    for index in range(rows):
        cap = min(
            v9_config.target_magnitude,
            float(liquidity[index]),
            float(risk[index]),
        )
        if abs(current) > cap:
            current = float(np.sign(current) * cap)
        reason = "hold_position" if abs(current) > _EPSILON else "hold_flat"
        policy_reason = reason
        cadence = index % v9_config.horizon_decisions == 0
        if cadence and bool(actionable[index]):
            signal = int(qualified[index])
            current_sign = int(np.sign(current))
            if inherited and current_sign != 0:
                wanted = current_sign if signal == current_sign else 0
                if wanted == intent:
                    count += 1
                else:
                    intent, count = wanted, 1
                if count >= v9_config.confirmation_count:
                    if wanted == 0:
                        current = 0.0
                        origin = "flat"
                        reason = policy_reason = "exit"
                    inherited = False
                    intent, count = 0, 0
                else:
                    reason = policy_reason = "confirmation_hold"
            elif current_sign == 0:
                entry_signal = signal
                calibrated = raw_edge[index]
                if signal != 0 and study_arm in (
                    CausalAlphaV11StudyArm.SIGN_CALIBRATED_ENTRY,
                    CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
                ):
                    assert sign_calibration is not None
                    calibrated = _calibrated_edge(
                        sign_calibration, signal, float(raw_edge[index])
                    )
                if study_arm is CausalAlphaV11StudyArm.AFTER_COST_ENTRY:
                    entry_signal = (
                        signal if raw_edge[index] - 2.0 * costs[index] > 0 else 0
                    )
                elif study_arm in (
                    CausalAlphaV11StudyArm.SIGN_CALIBRATED_ENTRY,
                    CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING,
                ):
                    entry_signal = signal if calibrated - 2.0 * costs[index] > 0 else 0
                if entry_signal == 0:
                    intent, count = 0, 0
                    reason = policy_reason = (
                        "cost_or_uncertainty_hold" if signal != 0 else "hold_flat"
                    )
                else:
                    if entry_signal == intent:
                        count += 1
                    else:
                        intent, count = entry_signal, 1
                    if count >= v9_config.confirmation_count:
                        magnitude = cap
                        if study_arm is CausalAlphaV11StudyArm.CALIBRATED_EDGE_SIZING:
                            magnitude = min(
                                cap,
                                v9_config.target_magnitude
                                * calibrated
                                / (
                                    calibrated
                                    + uncertainty[index]
                                    + 2.0 * costs[index]
                                    + v11_config.sizing_epsilon
                                ),
                            )
                        current = float(entry_signal * magnitude)
                        origin = "native"
                        reason = policy_reason = "entry"
                        intent, count, neutral_count = 0, 0, 0
                    else:
                        reason = policy_reason = "confirmation_hold"
            elif signal == -current_sign:
                neutral_count = 0
                if signal == intent:
                    count += 1
                else:
                    intent, count = signal, 1
                if count >= v9_config.confirmation_count:
                    current = 0.0
                    origin = "flat"
                    reason = policy_reason = "exit"
                    intent, count = 0, 0
                else:
                    reason = policy_reason = "confirmation_hold"
            elif (
                study_arm is CausalAlphaV11StudyArm.NEUTRAL_EXPIRY_2
                and origin == "native"
                and signal == 0
            ):
                neutral_count += 1
                intent, count = 0, 0
                if neutral_count >= v11_config.neutral_expiry_count:
                    current = 0.0
                    origin = "flat"
                    neutral_count = 0
                    reason = "exit"
                    policy_reason = "neutral_expiry_2"
                else:
                    reason = policy_reason = "hold_position"
            else:
                neutral_count = 0
                intent, count = 0, 0
                reason = policy_reason = "hold_position"
        elif cadence:
            reason = policy_reason = "unactionable_hold"

        targets[index] = current
        reasons.append(reason)
        policy_reasons.append(policy_reason)
        position_origins.append(origin)
        confirmation_counts[index] = count
        objectives[index] = current * mean[index] - abs(current) * (
            uncertainty[index] + v9_config.edge_margin
        )

    previous = np.concatenate(([initial_weight], targets[:-1]))
    forecast_digest = content_and_arrays_digest(
        {
            "schema_version": "causal_alpha_v11_wave_forecast_v1",
            "source_forecast_digest": source_forecast_digest,
            "study_arm": study_arm.value,
        },
        (("mean", mean), ("uncertainty", uncertainty), ("qualified", qualified)),
    )
    reason_tuple = tuple(reasons)
    return (
        CausalAlphaV6TargetPath(
            candidate=CausalAlphaV6Candidate.FAST_ONLY,
            initial_weight=float(initial_weight),
            decision_indices=decisions,
            targets=targets,
            fast_proposals=targets,
            expected_returns_4h=mean,
            expected_returns_24h=np.zeros(rows),
            expected_returns_72h=np.zeros(rows),
            direction_scores_4h=qualified.astype(np.float64),
            uncertainties_4h=uncertainty,
            one_way_cost_rates=costs,
            liquidity_weight_caps=liquidity,
            risk_weight_caps=risk,
            objectives=objectives,
            confirmation_counts=confirmation_counts,
            actionable_mask=actionable,
            slow_states=tuple(CausalAlphaV6SlowState.MIXED for _ in range(rows)),
            reasons=reason_tuple,
            reason_counts=tuple(
                sorted((reason, reason_tuple.count(reason)) for reason in set(reasons))
            ),
            submitted_change_count=int(
                np.count_nonzero(np.abs(targets - previous) > _EPSILON)
            ),
            sign_flip_count=int(np.count_nonzero(targets * previous < 0.0)),
            liquidity_deleveraging_count=0,
            risk_projection_count=0,
            forecast_digest=forecast_digest,
            config_digest=v9_config.digest,
        ),
        tuple(policy_reasons),
        tuple(position_origins),
    )


def compile_causal_alpha_v11_target(
    *,
    study_arm: CausalAlphaV11StudyArm | None,
    decision_indices: object,
    head_predictions: object,
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object,
    actionable_mask: object,
    source_forecast_digest: str,
    wave_fit_digest: str,
    v9_config: CausalAlphaV9Config,
    v11_config: CausalAlphaV11Config,
    initial_weight: float,
    sign_calibration: CausalAlphaV11SignCalibration | None = None,
) -> CausalAlphaV11CompiledTarget:
    """Compile exact V9 control or one independent V11 treatment."""

    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    rows = len(decisions)
    heads = np.asarray(head_predictions, dtype=np.float64)
    if rows == 0 or heads.shape != (3, rows) or not np.isfinite(heads).all():
        raise ValueError("V11 head predictions are invalid")
    costs = _readonly(one_way_cost_rates, dtype=np.float64, rows=rows, field="costs")
    liquidity = _readonly(
        liquidity_weight_caps, dtype=np.float64, rows=rows, field="liquidity caps"
    )
    risk = _readonly(risk_weight_caps, dtype=np.float64, rows=rows, field="risk caps")
    actionable = _readonly(
        actionable_mask, dtype=np.bool_, rows=rows, field="actionable mask"
    )
    mean, uncertainty, raw_edge, qualified = _qualified(
        heads, edge_margin=v9_config.edge_margin
    )
    after_cost = raw_edge - 2.0 * costs
    arm = None if study_arm is None else CausalAlphaV11StudyArm(study_arm)
    if arm is None:
        path = causal_alpha_v9_wave_target_path(
            decision_indices=decisions,
            head_predictions=heads,
            one_way_cost_rates=costs,
            liquidity_weight_caps=liquidity,
            risk_weight_caps=risk,
            actionable_mask=actionable,
            source_forecast_digest=source_forecast_digest,
            config=v9_config,
            initial_weight=initial_weight,
        )
        policy_reasons = tuple("v9_control" for _ in range(rows))
        origin = "inherited" if abs(initial_weight) > _EPSILON else "flat"
        origins: list[str] = []
        previous = float(initial_weight)
        for target, reason in zip(path.targets, path.reasons, strict=True):
            if reason == "entry":
                origin = "native"
            elif abs(target) <= _EPSILON:
                origin = "flat"
            elif abs(previous) <= _EPSILON and abs(target) > _EPSILON:
                origin = "native"
            origins.append(origin)
            previous = float(target)
        position_origins = tuple(origins)
        candidate = CausalAlphaV11Candidate.V9_CONTROL
    else:
        path, policy_reasons, position_origins = _compile_treatment_path(
            study_arm=arm,
            decisions=decisions,
            mean=mean,
            uncertainty=uncertainty,
            raw_edge=raw_edge,
            qualified=qualified,
            costs=costs,
            liquidity=liquidity,
            risk=risk,
            actionable=actionable,
            source_forecast_digest=source_forecast_digest,
            v9_config=v9_config,
            v11_config=v11_config,
            initial_weight=initial_weight,
            sign_calibration=sign_calibration,
        )
        candidate = CausalAlphaV11Candidate.TREATMENT
    calibration_digest = None if sign_calibration is None else sign_calibration.digest
    target = CausalAlphaV11TargetPath(
        candidate=candidate,
        study_arm=arm,
        v6_target_path=path,
        source_forecast_digest=source_forecast_digest,
        wave_fit_digest=wave_fit_digest,
        v9_config_digest=v9_config.digest,
        v11_config_digest=v11_config.digest,
        calibration_digest=calibration_digest,
    )
    return CausalAlphaV11CompiledTarget(
        target=target,
        fast_means=mean,
        fast_uncertainties=uncertainty,
        fast_qualified_directions=qualified,
        raw_edges=raw_edge,
        after_cost_entry_objectives=after_cost,
        policy_reasons=policy_reasons,
        position_origins=position_origins,
    )


class CausalAlphaV11TracePolicy:
    """Sequential policy exposing metadata for a precompiled action stream."""

    def __init__(self, compiled: CausalAlphaV11CompiledTarget) -> None:
        if not isinstance(compiled, CausalAlphaV11CompiledTarget):
            raise TypeError("V11 trace policy requires a compiled target")
        self.compiled = compiled
        self._offset = 0
        self._last_metadata: dict[str, object] | None = None

    @property
    def last_step_trace_metadata(self) -> dict[str, object]:
        if self._last_metadata is None:
            raise RuntimeError("V11 trace metadata is unavailable before predict")
        return dict(self._last_metadata)

    def predict(
        self, observation: object, deterministic: bool = True
    ) -> tuple[np.ndarray, None]:
        if not isinstance(deterministic, bool):
            raise TypeError("V11 deterministic flag must be boolean")
        path = self.compiled.target.v6_target_path
        if self._offset >= len(path.targets):
            raise RuntimeError("V11 trace policy exhausted its decision rows")
        if not isinstance(observation, dict) or "current_weights" not in observation:
            raise ValueError("V11 trace policy observation is missing current_weights")
        current = np.asarray(observation["current_weights"], dtype=np.float64).reshape(
            -1
        )
        if current.shape != (1,) or not np.isfinite(current).all():
            raise ValueError("V11 current_weights must contain one finite value")
        offset = self._offset
        requested = float(path.targets[offset])
        observed = float(current[0])
        reduce_only = abs(observed) > _EPSILON and (
            abs(requested) <= _EPSILON
            or (
                np.sign(requested) == np.sign(observed)
                and abs(requested) < abs(observed)
            )
        )
        arm = self.compiled.target.study_arm
        self._last_metadata = {
            "after_cost_entry_objective": float(
                self.compiled.after_cost_entry_objectives[offset]
            ),
            "fast_mean": float(self.compiled.fast_means[offset]),
            "fast_qualified_direction": int(
                self.compiled.fast_qualified_directions[offset]
            ),
            "fast_raw_edge": float(self.compiled.raw_edges[offset]),
            "fast_uncertainty": float(self.compiled.fast_uncertainties[offset]),
            "hierarchy_reason": self.compiled.policy_reasons[offset],
            "policy_reason": self.compiled.policy_reasons[offset],
            "position_origin": self.compiled.position_origins[offset],
            "reduce_only": bool(reduce_only),
            "study_arm": None if arm is None else arm.value,
        }
        self._offset += 1
        return np.asarray([requested], dtype=np.float32), None


__all__ = [
    "CausalAlphaV11CompiledTarget",
    "CausalAlphaV11TracePolicy",
    "compile_causal_alpha_v11_target",
]
