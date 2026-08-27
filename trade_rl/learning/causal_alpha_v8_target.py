"""Robust entry, continuation, and exit compiler for Causal Alpha V8."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Final

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
    build_causal_alpha_v4_forecast,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v6_target import (
    causal_alpha_v6_fast_candidates,
    causal_alpha_v6_slow_state,
    causal_alpha_v6_target_path,
)
from trade_rl.learning.causal_alpha_v7_calibration import CausalAlphaV7CalibrationFit
from trade_rl.learning.causal_alpha_v8 import (
    CausalAlphaV8Candidate,
    CausalAlphaV8TargetConfig,
    CausalAlphaV8TargetPath,
)

_EPSILON: Final = 1e-12


@dataclass(slots=True)
class _Confirmation:
    direction: int = 0
    count: int = 0

    def update(self, direction: int) -> int:
        if direction == 0:
            self.direction = 0
            self.count = 0
        elif direction == self.direction:
            self.count += 1
        else:
            self.direction = direction
            self.count = 1
        return self.count


def _sign(value: float) -> int:
    return int(value > _EPSILON) - int(value < -_EPSILON)


def _aligned(value: object, *, rows: int, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V8 {field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"V8 {field} must be finite")
    return array


def _vectors(
    forecast: CausalAlphaV4Forecast,
    *,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    risk_weight_caps: object | None,
    actionable_mask: object,
) -> dict[str, np.ndarray]:
    if set(uncertainty) != set(CAUSAL_ALPHA_V4_HORIZONS):
        raise ValueError("V8 uncertainty horizons are invalid")
    rows = len(forecast.decision_indices)
    raw: dict[str, object] = {
        "expected_returns_4h": forecast.final_predictions["4h"],
        "expected_returns_24h": forecast.final_predictions["24h"],
        "expected_returns_72h": forecast.final_predictions["72h"],
        "direction_scores_4h": forecast.direction_scores["4h"],
        "uncertainties_4h": uncertainty["4h"],
        "one_way_cost_rates": one_way_cost_rates,
        "liquidity_weight_caps": liquidity_weight_caps,
        "risk_weight_caps": (
            np.full(rows, 0.25, dtype=np.float64)
            if risk_weight_caps is None
            else risk_weight_caps
        ),
        "actionable_mask": actionable_mask,
    }
    result = {
        name: _aligned(
            value,
            rows=rows,
            dtype=np.bool_ if name == "actionable_mask" else np.float64,
            field=name,
        )
        for name, value in raw.items()
    }
    for name in (
        "uncertainties_4h",
        "one_way_cost_rates",
        "liquidity_weight_caps",
        "risk_weight_caps",
    ):
        if np.any(result[name] < 0.0):
            raise ValueError(f"V8 {name} must be non-negative")
    return result


def causal_alpha_v8_position_utility(
    target: float,
    *,
    expected_return: float,
    uncertainty: float,
    config: CausalAlphaV8TargetConfig,
) -> float:
    """Return robust utility for carrying one target through the fast horizon."""

    if not all(math.isfinite(value) for value in (target, expected_return, uncertainty)):
        raise ValueError("V8 position utility inputs must be finite")
    if uncertainty < 0.0:
        raise ValueError("V8 position utility uncertainty must be non-negative")
    base = config.base
    return target * expected_return - abs(target) * (
        base.uncertainty_multiplier * uncertainty + base.edge_margin
    )


def causal_alpha_v8_transition_score(
    previous: float,
    target: float,
    *,
    expected_return: float,
    uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV8TargetConfig,
) -> float:
    """Score a transition after robust carrying utility and execution cost."""

    if not math.isfinite(one_way_cost_rate) or one_way_cost_rate < 0.0:
        raise ValueError("V8 transition cost must be finite non-negative")
    base = config.base
    return (
        causal_alpha_v8_position_utility(
            target,
            expected_return=expected_return,
            uncertainty=uncertainty,
            config=config,
        )
        - causal_alpha_v8_position_utility(
            previous,
            expected_return=expected_return,
            uncertainty=uncertainty,
            config=config,
        )
        - abs(target - previous)
        * base.execution_cost_multiplier
        * one_way_cost_rate
    )


def _is_risk_reduction(previous: float, target: float) -> bool:
    if abs(previous) <= _EPSILON:
        return abs(target) <= _EPSILON
    return previous * target >= -_EPSILON and abs(target) <= abs(previous) + _EPSILON


def _is_risk_increase(previous: float, target: float) -> bool:
    return not _is_risk_reduction(previous, target)


def _consensus_allows(
    previous: float,
    target: float,
    *,
    expected_return: float,
    direction_score: float,
) -> bool:
    if _is_risk_reduction(previous, target):
        return True
    return (
        _sign(expected_return) != 0
        and expected_return * direction_score > 0.0
        and target * expected_return > 0.0
    )


def _transition_reason(previous: float, target: float) -> str:
    if abs(target - previous) <= _EPSILON:
        return "hold_flat" if abs(previous) <= _EPSILON else "hold_position"
    if abs(previous) <= _EPSILON:
        return "entry"
    if abs(target) <= _EPSILON:
        return "exit"
    return "add" if abs(target) > abs(previous) else "reduce"


def _choose(
    previous: float,
    candidates: tuple[float, ...],
    scores: tuple[float, ...],
) -> tuple[float, float]:
    maximum = max(scores)
    tied = tuple(
        (target, score)
        for target, score in zip(candidates, scores, strict=True)
        if score >= maximum - 1e-15
    )
    return min(tied, key=lambda item: (abs(item[0] - previous), abs(item[0]), item[0]))


def causal_alpha_v8_exposure_path(
    forecast: CausalAlphaV4Forecast,
    *,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    actionable_mask: object,
    config: CausalAlphaV8TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> CausalAlphaV6TargetPath:
    """Compile robust exposure while preserving the maintained replay contract."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V8 exposure compiler requires a V4 forecast")
    if not isinstance(config, CausalAlphaV8TargetConfig):
        raise TypeError("V8 exposure compiler requires V8 config")
    if not math.isfinite(initial_weight):
        raise ValueError("V8 initial weight must be finite")
    vectors = _vectors(
        forecast,
        uncertainty=uncertainty,
        one_way_cost_rates=one_way_cost_rates,
        liquidity_weight_caps=liquidity_weight_caps,
        risk_weight_caps=risk_weight_caps,
        actionable_mask=actionable_mask,
    )
    rows = len(forecast.decision_indices)
    targets = np.empty(rows, dtype=np.float64)
    proposals = np.empty(rows, dtype=np.float64)
    objectives = np.empty(rows, dtype=np.float64)
    confirmation_counts = np.empty(rows, dtype=np.int64)
    slow_states: list[CausalAlphaV6SlowState] = []
    reasons: list[str] = []
    previous = float(initial_weight)
    confirmation = _Confirmation()
    cached_slow = (0.0, 0.0)
    base = config.base

    for index in range(rows):
        if index % base.slow_context_decisions == 0:
            cached_slow = (
                float(vectors["expected_returns_24h"][index]),
                float(vectors["expected_returns_72h"][index]),
            )
        slow_states.append(causal_alpha_v6_slow_state(previous, *cached_slow))
        liquidity_cap = min(
            base.maximum_absolute_target,
            float(vectors["liquidity_weight_caps"][index]),
        )
        risk_cap = min(
            base.maximum_absolute_target,
            float(vectors["risk_weight_caps"][index]),
        )
        cap = min(liquidity_cap, risk_cap)
        selected = previous
        objective = 0.0

        if abs(previous) > liquidity_cap + _EPSILON:
            selected = float(np.clip(previous, -liquidity_cap, liquidity_cap))
            reason = "liquidity_deleverage"
        elif abs(previous) > risk_cap + _EPSILON:
            selected = float(np.clip(previous, -risk_cap, risk_cap))
            reason = "risk_projection"
        elif not bool(vectors["actionable_mask"][index]):
            reason = "unactionable_hold"
        elif index % base.fast_rebalance_decisions != 0:
            reason = "cadence_hold"
        else:
            mu = float(vectors["expected_returns_4h"][index])
            sigma = float(vectors["uncertainties_4h"][index])
            cost = float(vectors["one_way_cost_rates"][index])
            direction_score = float(vectors["direction_scores_4h"][index])
            intent = _sign(mu) if mu * direction_score > 0.0 else 0
            confirmed = confirmation.update(intent) >= base.confirmation_count
            reachable = causal_alpha_v6_fast_candidates(previous, cap, base)
            no_flips = tuple(
                target for target in reachable if previous * target >= -_EPSILON
            )
            consensus = tuple(
                target
                for target in no_flips
                if _consensus_allows(
                    previous,
                    target,
                    expected_return=mu,
                    direction_score=direction_score,
                )
            )
            confirmation_blocked = tuple(
                target
                for target in consensus
                if _is_risk_increase(previous, target) and not confirmed
            )
            allowed = tuple(
                target for target in consensus if target not in confirmation_blocked
            )
            scores = tuple(
                causal_alpha_v8_transition_score(
                    previous,
                    target,
                    expected_return=mu,
                    uncertainty=sigma,
                    one_way_cost_rate=cost,
                    config=config,
                )
                for target in allowed
            )
            selected, objective = _choose(previous, allowed, scores)
            if abs(selected - previous) <= _EPSILON:
                all_scores = {
                    target: causal_alpha_v8_transition_score(
                        previous,
                        target,
                        expected_return=mu,
                        uncertainty=sigma,
                        one_way_cost_rate=cost,
                        config=config,
                    )
                    for target in no_flips
                }
                profitable = tuple(
                    target
                    for target, score in all_scores.items()
                    if score > _EPSILON and abs(target - previous) > _EPSILON
                )
                if any(target not in consensus for target in profitable):
                    reason = "direction_disagreement_hold"
                elif any(target in confirmation_blocked for target in profitable):
                    reason = "confirmation_hold"
                elif profitable:
                    reason = "cost_or_uncertainty_hold"
                else:
                    reason = _transition_reason(previous, selected)
            else:
                reason = _transition_reason(previous, selected)

        targets[index] = selected
        proposals[index] = selected
        objectives[index] = objective
        confirmation_counts[index] = confirmation.count
        reasons.append(reason)
        previous = float(selected)

    previous_targets = np.concatenate(([initial_weight], targets[:-1]))
    counts = tuple(sorted((reason, reasons.count(reason)) for reason in set(reasons)))
    return CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=float(initial_weight),
        decision_indices=forecast.decision_indices,
        targets=targets,
        fast_proposals=proposals,
        expected_returns_4h=vectors["expected_returns_4h"],
        expected_returns_24h=vectors["expected_returns_24h"],
        expected_returns_72h=vectors["expected_returns_72h"],
        direction_scores_4h=vectors["direction_scores_4h"],
        uncertainties_4h=vectors["uncertainties_4h"],
        one_way_cost_rates=vectors["one_way_cost_rates"],
        liquidity_weight_caps=vectors["liquidity_weight_caps"],
        risk_weight_caps=vectors["risk_weight_caps"],
        objectives=objectives,
        confirmation_counts=confirmation_counts,
        actionable_mask=vectors["actionable_mask"],
        slow_states=tuple(slow_states),
        reasons=tuple(reasons),
        reason_counts=counts,
        submitted_change_count=int(
            np.count_nonzero(np.abs(targets - previous_targets) > _EPSILON)
        ),
        sign_flip_count=int(np.count_nonzero(targets * previous_targets < 0.0)),
        liquidity_deleveraging_count=reasons.count("liquidity_deleverage"),
        risk_projection_count=reasons.count("risk_projection"),
        forecast_digest=forecast.digest,
        config_digest=config.digest,
    )


def _effective_forecast(
    source: CausalAlphaV4Forecast,
    *,
    expected_return_4h: np.ndarray,
    direction_score_4h: np.ndarray,
    transformation: str,
    return_model_digest: str,
    direction_model_digest: str,
) -> CausalAlphaV4Forecast:
    rows = len(source.decision_indices)
    expected = np.asarray(expected_return_4h, dtype=np.float64).reshape(-1)
    direction = np.asarray(direction_score_4h, dtype=np.float64).reshape(-1)
    if (
        expected.shape != (rows,)
        or direction.shape != (rows,)
        or not np.isfinite(expected).all()
        or not np.isfinite(direction).all()
    ):
        raise ValueError("V8 effective fast forecast is invalid")
    market = {
        horizon: source.market_predictions[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    residual = {
        horizon: source.residual_predictions[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    directions = {
        horizon: source.direction_scores[horizon]
        for horizon in CAUSAL_ALPHA_V4_HORIZONS
    }
    market["4h"] = np.zeros(rows, dtype=np.float64)
    residual["4h"] = expected
    directions["4h"] = direction
    market_digests = dict(source.market_model_digests)
    residual_digests = dict(source.residual_model_digests)
    direction_digests = dict(source.direction_model_digests)
    market_digests["4h"] = content_digest(
        {
            "component": "zero_market",
            "source_forecast_digest": source.digest,
            "transformation": transformation,
        }
    )
    residual_digests["4h"] = return_model_digest
    direction_digests["4h"] = direction_model_digest
    fit_digest = content_digest(
        {
            "direction_model_digest": direction_model_digest,
            "return_model_digest": return_model_digest,
            "source_fit_digest": source.fit_digest,
            "transformation": transformation,
        }
    )
    return build_causal_alpha_v4_forecast(
        symbol=source.symbol,
        decision_indices=source.decision_indices,
        beta=source.beta,
        beta_available=source.beta_available,
        market_predictions=market,
        residual_predictions=residual,
        direction_scores=directions,
        market_model_digests=market_digests,
        residual_model_digests=residual_digests,
        direction_model_digests=direction_digests,
        fit_digest=fit_digest,
    )


def causal_alpha_v8_target_paths(
    *,
    forecast: CausalAlphaV4Forecast,
    calibration_fit: CausalAlphaV7CalibrationFit,
    calibration_features: object,
    calibration_feature_available: object,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    actionable_mask: object,
    config: CausalAlphaV8TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> Mapping[CausalAlphaV8Candidate, CausalAlphaV8TargetPath]:
    """Compile the exact V7 control and two robust V8 candidates."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V8 target paths require a V4 forecast")
    if not isinstance(calibration_fit, CausalAlphaV7CalibrationFit):
        raise TypeError("V8 target paths require a causal calibration fit")
    if not isinstance(config, CausalAlphaV8TargetConfig):
        raise TypeError("V8 target paths require V8 config")
    calibrated_return, calibrated_direction = calibration_fit.predict(
        calibration_features,
        feature_available=calibration_feature_available,
    )
    raw_return = np.asarray(forecast.final_predictions["4h"], dtype=np.float64)
    raw_direction = np.asarray(forecast.direction_scores["4h"], dtype=np.float64)
    contrarian = _effective_forecast(
        forecast,
        expected_return_4h=-raw_return,
        direction_score_4h=-raw_direction,
        transformation="v8_robust_contrarian",
        return_model_digest=content_digest(
            {
                "source_forecast_digest": forecast.digest,
                "transformation": "negate_return",
            }
        ),
        direction_model_digest=content_digest(
            {
                "source_forecast_digest": forecast.digest,
                "transformation": "negate_direction",
            }
        ),
    )
    calibrated = _effective_forecast(
        forecast,
        expected_return_4h=calibrated_return,
        direction_score_4h=calibrated_direction,
        transformation="v8_robust_calibrated",
        return_model_digest=calibration_fit.return_model.digest,
        direction_model_digest=calibration_fit.direction_model.digest,
    )
    control_path = causal_alpha_v6_target_path(
        forecast,
        uncertainty=uncertainty,
        one_way_cost_rates=one_way_cost_rates,
        liquidity_weight_caps=liquidity_weight_caps,
        risk_weight_caps=risk_weight_caps,
        actionable_mask=actionable_mask,
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        config=config.base,
        initial_weight=initial_weight,
    )
    robust = {
        CausalAlphaV8Candidate.ROBUST_CONTRARIAN: causal_alpha_v8_exposure_path(
            contrarian,
            uncertainty=uncertainty,
            one_way_cost_rates=one_way_cost_rates,
            liquidity_weight_caps=liquidity_weight_caps,
            risk_weight_caps=risk_weight_caps,
            actionable_mask=actionable_mask,
            config=config,
            initial_weight=initial_weight,
        ),
        CausalAlphaV8Candidate.ROBUST_CALIBRATED: causal_alpha_v8_exposure_path(
            calibrated,
            uncertainty=uncertainty,
            one_way_cost_rates=one_way_cost_rates,
            liquidity_weight_caps=liquidity_weight_caps,
            risk_weight_caps=risk_weight_caps,
            actionable_mask=actionable_mask,
            config=config,
            initial_weight=initial_weight,
        ),
    }
    paths = {
        CausalAlphaV8Candidate.V7_CONTROL: CausalAlphaV8TargetPath(
            candidate=CausalAlphaV8Candidate.V7_CONTROL,
            v6_target_path=control_path,
            source_forecast_digest=forecast.digest,
            calibration_fit_digest=calibration_fit.digest,
            v8_config_digest=config.digest,
        ),
        **{
            candidate: CausalAlphaV8TargetPath(
                candidate=candidate,
                v6_target_path=path,
                source_forecast_digest=forecast.digest,
                calibration_fit_digest=calibration_fit.digest,
                v8_config_digest=config.digest,
            )
            for candidate, path in robust.items()
        },
    }
    if tuple(paths) != tuple(CausalAlphaV8Candidate):
        raise RuntimeError("V8 target candidate order drifted")
    return MappingProxyType(paths)


__all__ = [
    "causal_alpha_v8_exposure_path",
    "causal_alpha_v8_position_utility",
    "causal_alpha_v8_target_paths",
    "causal_alpha_v8_transition_score",
]
