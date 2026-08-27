"""Robust entry, continuation, and exit compiler for Causal Alpha V8."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Final

import numpy as np

from trade_rl.learning.causal_alpha_v4 import (
    CAUSAL_ALPHA_V4_HORIZONS,
    CausalAlphaV4Forecast,
)
from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6SlowState,
    CausalAlphaV6TargetPath,
)
from trade_rl.learning.causal_alpha_v6_target import (
    causal_alpha_v6_fast_candidates,
    causal_alpha_v6_slow_state,
)
from trade_rl.learning.causal_alpha_v8 import CausalAlphaV8TargetConfig

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


__all__ = [
    "causal_alpha_v8_exposure_path",
    "causal_alpha_v8_position_utility",
    "causal_alpha_v8_transition_score",
]
