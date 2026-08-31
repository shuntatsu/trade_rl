"""Deterministic fast proposal and slow retention filter for Causal Alpha V6."""

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
    CausalAlphaV6TargetConfig,
    CausalAlphaV6TargetPath,
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


def _aligned_vector(
    value: object,
    *,
    rows: int,
    dtype: Any,
    field: str,
) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V6 {field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"V6 {field} must be finite")
    return array


def _sign(value: float) -> int:
    return int(value > _EPSILON) - int(value < -_EPSILON)


def _is_risk_reduction(previous: float, target: float) -> bool:
    if abs(target - previous) <= _EPSILON:
        return True
    if abs(previous) <= _EPSILON:
        return abs(target) <= _EPSILON
    return previous * target >= -_EPSILON and abs(target) <= abs(previous) + _EPSILON


def _is_add(previous: float, proposed: float) -> bool:
    return previous * proposed > _EPSILON and abs(proposed) > abs(previous) + _EPSILON


def _is_flip(previous: float, proposed: float) -> bool:
    return previous * proposed < -_EPSILON


def causal_alpha_v6_slow_state(
    previous: float,
    prediction_24h: float,
    prediction_72h: float,
) -> CausalAlphaV6SlowState:
    """Classify cached slow context relative to one symbol's position."""

    position_sign = _sign(previous)
    if position_sign == 0:
        return CausalAlphaV6SlowState.FLAT
    slow_signs = (_sign(prediction_24h), _sign(prediction_72h))
    if slow_signs == (position_sign, position_sign):
        return CausalAlphaV6SlowState.SUPPORTIVE
    if slow_signs == (-position_sign, -position_sign):
        return CausalAlphaV6SlowState.OPPOSED
    return CausalAlphaV6SlowState.MIXED


def causal_alpha_v6_fast_candidates(
    previous: float,
    cap: float,
    config: CausalAlphaV6TargetConfig,
) -> tuple[float, ...]:
    """Return fixed target levels clipped by cap and per-decision delta."""

    lower = max(-cap, previous - config.maximum_target_delta)
    upper = min(cap, previous + config.maximum_target_delta)
    values = {float(np.clip(previous, -cap, cap)), 0.0, lower, upper}
    for magnitude in config.target_magnitudes:
        values.add(float(np.clip(magnitude, lower, upper)))
        values.add(float(np.clip(-magnitude, lower, upper)))
    return tuple(sorted(value for value in values if -cap <= value <= cap))


def causal_alpha_v6_fast_objective(
    previous: float,
    target: float,
    expected_return: float,
    uncertainty: float,
    one_way_cost_rate: float,
    config: CausalAlphaV6TargetConfig,
) -> float:
    """Score one target by causal 4h alpha after uncertainty and cost."""

    delta = target - previous
    hurdle = (
        config.uncertainty_multiplier * uncertainty
        + config.execution_cost_multiplier * one_way_cost_rate
        + config.edge_margin
    )
    return delta * expected_return - abs(delta) * hurdle


def _consensus_allows(
    previous: float,
    target: float,
    expected_return: float,
    direction_score: float,
) -> bool:
    if _is_risk_reduction(previous, target):
        return True
    if _sign(expected_return) == 0 or expected_return * direction_score <= 0.0:
        return False
    return target * expected_return > 0.0


def causal_alpha_v6_retention_allows(
    previous: float,
    proposed: float,
    state: CausalAlphaV6SlowState,
    confirmed_reversal: bool,
) -> bool:
    """Apply only the candidate-specific slow retention constraint."""

    if state is CausalAlphaV6SlowState.FLAT:
        return True
    if _is_flip(previous, proposed):
        return confirmed_reversal
    if state is CausalAlphaV6SlowState.SUPPORTIVE:
        if abs(proposed) + _EPSILON < abs(previous):
            return confirmed_reversal
        return True
    return not _is_add(previous, proposed)


def _choose_best(
    candidates: tuple[float, ...],
    scores: tuple[float, ...],
    *,
    previous: float,
) -> tuple[float, float]:
    maximum = max(scores)
    tied = (
        item
        for item in zip(candidates, scores, strict=True)
        if item[1] >= maximum - 1e-15
    )
    return min(
        tied,
        key=lambda item: (
            abs(item[0] - previous),
            abs(item[0]),
            item[0],
        ),
    )


def _transition_reason(previous: float, selected: float) -> str:
    if abs(selected - previous) <= _EPSILON:
        return "hold_flat" if abs(previous) <= _EPSILON else "hold_position"
    if abs(previous) <= _EPSILON:
        return "entry"
    if abs(selected) <= _EPSILON:
        return "exit"
    if _is_flip(previous, selected):
        return "flip"
    return "add" if abs(selected) > abs(previous) else "reduce"


def _proposal(
    *,
    previous: float,
    cap: float,
    mu: float,
    sigma: float,
    cost: float,
    direction_score: float,
    confirmed: bool,
    strong_reversal: bool,
    config: CausalAlphaV6TargetConfig,
) -> tuple[float, float, str | None, bool]:
    candidates = causal_alpha_v6_fast_candidates(previous, cap, config)
    scores = tuple(
        causal_alpha_v6_fast_objective(previous, value, mu, sigma, cost, config)
        for value in candidates
    )
    profitable = tuple(
        value
        for value, score in zip(candidates, scores, strict=True)
        if score > _EPSILON and abs(value - previous) > _EPSILON
    )
    consensus = tuple(
        value
        for value in candidates
        if _consensus_allows(previous, value, mu, direction_score)
    )
    confirmation_blocked = tuple(
        value
        for value in consensus
        if not _is_risk_reduction(previous, value)
        and not confirmed
        and not (strong_reversal and _is_flip(previous, value))
    )
    allowed = tuple(value for value in consensus if value not in confirmation_blocked)
    allowed_scores = tuple(scores[candidates.index(value)] for value in allowed)
    proposed, objective = _choose_best(allowed, allowed_scores, previous=previous)
    reason: str | None = None
    if abs(proposed - previous) <= _EPSILON:
        if any(value not in consensus for value in profitable):
            reason = "direction_disagreement_hold"
        elif any(value in confirmation_blocked for value in profitable):
            reason = "confirmation_hold"
        elif not profitable:
            reason = "cost_or_uncertainty_hold"
    return proposed, objective, reason, bool(confirmation_blocked)


def _slow_reason(
    previous: float,
    proposed: float,
    state: CausalAlphaV6SlowState,
) -> str:
    if state is CausalAlphaV6SlowState.SUPPORTIVE:
        return "slow_support_hold"
    if _is_add(previous, proposed):
        return "slow_add_suppressed"
    return "slow_support_hold"


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
        raise ValueError("V6 uncertainty horizons are invalid")
    rows = int(forecast.decision_indices.size)
    raw_vectors: dict[str, object] = {
        "expected_returns_4h": np.asarray(forecast.final_predictions["4h"]),
        "expected_returns_24h": np.asarray(forecast.final_predictions["24h"]),
        "expected_returns_72h": np.asarray(forecast.final_predictions["72h"]),
        "direction_scores_4h": np.asarray(forecast.direction_scores["4h"]),
        "uncertainties_4h": uncertainty["4h"],
        "one_way_cost_rates": one_way_cost_rates,
        "liquidity_weight_caps": liquidity_weight_caps,
        "risk_weight_caps": (
            np.full(rows, 0.25) if risk_weight_caps is None else risk_weight_caps
        ),
        "actionable_mask": actionable_mask,
    }
    vectors: dict[str, np.ndarray] = {}
    for name, value in raw_vectors.items():
        dtype = np.bool_ if name == "actionable_mask" else np.float64
        vectors[name] = _aligned_vector(value, rows=rows, dtype=dtype, field=name)
    if any(
        np.any(vectors[name] < 0.0)
        for name in (
            "uncertainties_4h",
            "one_way_cost_rates",
            "liquidity_weight_caps",
            "risk_weight_caps",
        )
    ):
        raise ValueError("V6 uncertainty, costs, and caps must be non-negative")
    return vectors


def causal_alpha_v6_target_path(
    forecast: CausalAlphaV4Forecast,
    *,
    uncertainty: Mapping[str, np.ndarray],
    one_way_cost_rates: object,
    liquidity_weight_caps: object,
    actionable_mask: object,
    candidate: CausalAlphaV6Candidate,
    config: CausalAlphaV6TargetConfig,
    initial_weight: float,
    risk_weight_caps: object | None = None,
) -> CausalAlphaV6TargetPath:
    """Compile one independent symbol path with fast entry and optional retention."""

    if not isinstance(forecast, CausalAlphaV4Forecast):
        raise TypeError("V6 target compiler requires a V4 forecast")
    if not isinstance(config, CausalAlphaV6TargetConfig):
        raise TypeError("V6 target compiler requires the fixed V6 config")
    if not math.isfinite(initial_weight):
        raise ValueError("V6 target initial weight must be finite")
    selected_candidate = CausalAlphaV6Candidate(candidate)
    vectors = _vectors(
        forecast,
        uncertainty=uncertainty,
        one_way_cost_rates=one_way_cost_rates,
        liquidity_weight_caps=liquidity_weight_caps,
        risk_weight_caps=risk_weight_caps,
        actionable_mask=actionable_mask,
    )
    rows = int(forecast.decision_indices.size)
    targets = np.empty(rows)
    proposals = np.empty(rows)
    objectives = np.empty(rows)
    confirmation_counts = np.empty(rows, dtype=np.int64)
    slow_states: list[CausalAlphaV6SlowState] = []
    reasons: list[str] = []
    previous = float(initial_weight)
    confirmation = _Confirmation()
    cached_slow = (0.0, 0.0)

    for index in range(rows):
        if index % config.slow_context_decisions == 0:
            cached_slow = (
                float(vectors["expected_returns_24h"][index]),
                float(vectors["expected_returns_72h"][index]),
            )
        slow_state = causal_alpha_v6_slow_state(previous, *cached_slow)
        slow_states.append(slow_state)
        liquidity_cap = min(
            config.maximum_absolute_target,
            float(vectors["liquidity_weight_caps"][index]),
        )
        risk_cap = min(
            config.maximum_absolute_target,
            float(vectors["risk_weight_caps"][index]),
        )
        cap = min(liquidity_cap, risk_cap)
        selected = previous
        proposal = previous
        objective = 0.0
        reason: str

        if abs(previous) > liquidity_cap + _EPSILON:
            selected = proposal = float(
                np.clip(previous, -liquidity_cap, liquidity_cap)
            )
            reason = "liquidity_deleverage"
        elif abs(previous) > risk_cap + _EPSILON:
            selected = proposal = float(np.clip(previous, -risk_cap, risk_cap))
            reason = "risk_projection"
        elif not bool(vectors["actionable_mask"][index]):
            reason = "unactionable_hold"
        elif index % config.fast_rebalance_decisions != 0:
            reason = "cadence_hold"
        else:
            mu = float(vectors["expected_returns_4h"][index])
            direction_score = float(vectors["direction_scores_4h"][index])
            intent = _sign(mu) if mu * direction_score > 0.0 else 0
            count = confirmation.update(intent)
            confirmed = count >= config.confirmation_count
            strong = (
                _sign(previous) != 0
                and intent == -_sign(previous)
                and abs(mu) >= config.strong_reversal_threshold
            )
            proposal, objective, hold_reason, _ = _proposal(
                previous=previous,
                cap=cap,
                mu=mu,
                sigma=float(vectors["uncertainties_4h"][index]),
                cost=float(vectors["one_way_cost_rates"][index]),
                direction_score=direction_score,
                confirmed=confirmed,
                strong_reversal=strong,
                config=config,
            )
            selected = proposal
            if (
                selected_candidate is CausalAlphaV6Candidate.FAST_SLOW_RETENTION
                and not causal_alpha_v6_retention_allows(
                    previous,
                    proposal,
                    slow_state,
                    confirmed_reversal=confirmed or strong,
                )
            ):
                selected = previous
                reason = _slow_reason(previous, proposal, slow_state)
            else:
                reason = hold_reason or _transition_reason(previous, selected)
            if abs(selected - proposal) > _EPSILON:
                objective = causal_alpha_v6_fast_objective(
                    previous,
                    selected,
                    mu,
                    float(vectors["uncertainties_4h"][index]),
                    float(vectors["one_way_cost_rates"][index]),
                    config,
                )

        targets[index] = selected
        proposals[index] = proposal
        objectives[index] = objective
        confirmation_counts[index] = confirmation.count
        reasons.append(reason)
        previous = float(selected)

    previous_targets = np.concatenate(([initial_weight], targets[:-1]))
    counts = tuple(sorted((reason, reasons.count(reason)) for reason in set(reasons)))
    return CausalAlphaV6TargetPath(
        candidate=selected_candidate,
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
    "causal_alpha_v6_fast_candidates",
    "causal_alpha_v6_fast_objective",
    "causal_alpha_v6_retention_allows",
    "causal_alpha_v6_slow_state",
    "causal_alpha_v6_target_path",
]
