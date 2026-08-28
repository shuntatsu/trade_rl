"""Pure two-stage exposure compiler for Causal Alpha V10."""

from __future__ import annotations

from typing import Any

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
from trade_rl.workflows.universal_causal_alpha_v7_attribution import (
    CausalAlphaV7AttributionBoundaries,
)


def _aligned(value: object, *, rows: int, dtype: Any, field: str) -> np.ndarray:
    array = np.asarray(value, dtype=dtype).reshape(-1)
    if array.shape != (rows,):
        raise ValueError(f"V10 {field} must be decision aligned")
    if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
        raise ValueError(f"V10 {field} must be finite")
    return array


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


def _slow_state(direction: int, current: float) -> CausalAlphaV6SlowState:
    current_sign = int(np.sign(current))
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
    execution_entry_threshold: float,
    execution_no_trade_band: float,
    economic_config: CausalAlphaV6TargetConfig,
) -> str:
    return content_digest(
        {
            "execution_entry_threshold": execution_entry_threshold,
            "execution_no_trade_band": execution_no_trade_band,
            "schema_version": "causal_alpha_v10_target_compiler_contract_v2",
            "v6_economic_config_digest": economic_config.digest,
            "v10_config_digest": config.digest,
        }
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
    attribution_boundaries: CausalAlphaV7AttributionBoundaries,
    actionable_mask: object,
    source_forecast_digest: str,
    dual_fit_digest: str,
    config: CausalAlphaV10Config,
    initial_weight: float,
    execution_entry_threshold: float,
    execution_no_trade_band: float,
) -> CausalAlphaV6TargetPath:
    """Combine slow wave ownership with fast entry and early reversal signals."""

    require_sha256(source_forecast_digest, field="V10 source forecast digest")
    require_sha256(dual_fit_digest, field="V10 dual fit digest")
    if not isinstance(attribution_boundaries, CausalAlphaV7AttributionBoundaries):
        raise TypeError("V10 attribution boundaries are invalid")
    for value, field in (
        (execution_entry_threshold, "entry threshold"),
        (execution_no_trade_band, "no-trade band"),
    ):
        if (
            isinstance(value, bool)
            or not np.isfinite(value)
            or float(value) < 0.0
        ):
            raise ValueError(f"V10 execution {field} must be finite and non-negative")
    entry_threshold = float(execution_entry_threshold)
    no_trade_band = float(execution_no_trade_band)
    entry_floor = max(entry_threshold, no_trade_band)
    economic_config = CausalAlphaV6TargetConfig()
    compiler_config_digest = _compiler_config_digest(
        config=config,
        execution_entry_threshold=entry_threshold,
        execution_no_trade_band=no_trade_band,
        economic_config=economic_config,
    )

    decisions = np.asarray(decision_indices, dtype=np.int64).reshape(-1)
    rows = len(decisions)
    fast_heads = np.asarray(fast_head_predictions, dtype=np.float64)
    slow_heads = np.asarray(slow_head_predictions, dtype=np.float64)
    if (
        fast_heads.shape != (3, rows)
        or slow_heads.shape != (3, rows)
        or not np.isfinite(fast_heads).all()
        or not np.isfinite(slow_heads).all()
    ):
        raise ValueError("V10 head predictions are invalid")
    costs = _aligned(one_way_cost_rates, rows=rows, dtype=np.float64, field="costs")
    liquidity_caps = _aligned(
        liquidity_weight_caps,
        rows=rows,
        dtype=np.float64,
        field="liquidity caps",
    )
    risk_caps = _aligned(
        risk_weight_caps,
        rows=rows,
        dtype=np.float64,
        field="risk caps",
    )
    volatility = _aligned(
        realized_volatility,
        rows=rows,
        dtype=np.float64,
        field="realized volatility",
    )
    liquidity_values = _aligned(
        liquidity,
        rows=rows,
        dtype=np.float64,
        field="liquidity",
    )
    actionable = _aligned(
        actionable_mask,
        rows=rows,
        dtype=np.bool_,
        field="actionable mask",
    )
    if (
        np.any(costs < 0.0)
        or np.any(liquidity_caps < 0.0)
        or np.any(risk_caps < 0.0)
    ):
        raise ValueError("V10 costs and caps must be non-negative")
    fast_mean, fast_uncertainty, fast_direction = _qualified(
        fast_heads,
        edge_margin=config.edge_margin,
    )
    slow_mean, _slow_uncertainty, slow_direction = _qualified(
        slow_heads,
        edge_margin=config.edge_margin,
    )
    execution_eligible = (
        (liquidity_values >= attribution_boundaries.liquidity[1])
        & (volatility >= attribution_boundaries.realized_volatility[0])
        & (volatility < attribution_boundaries.realized_volatility[2])
    )

    targets = np.empty(rows, dtype=np.float64)
    objectives = np.zeros(rows, dtype=np.float64)
    confirmation_counts = np.zeros(rows, dtype=np.int64)
    reasons: list[str] = []
    slow_states: list[CausalAlphaV6SlowState] = []
    current = float(
        np.clip(initial_weight, -config.target_magnitude, config.target_magnitude)
    )
    inherited = abs(initial_weight) > 1e-12
    inherited_checks = 0
    inherited_matches = 0
    entry_intent = 0
    entry_count = 0
    fast_exit_count = 0
    slow_exit_count = 0
    slow_opposite_count = 0
    neutral_slow_count = 0
    slow_regime = 0

    for index in range(rows):
        previous_current = current
        liquidity_cap = min(
            config.target_magnitude,
            float(liquidity_caps[index]),
        )
        risk_cap = min(
            config.target_magnitude,
            float(risk_caps[index]),
        )
        cap = min(liquidity_cap, risk_cap)
        cadence = index % config.fast_horizon_decisions == 0
        resize_reason: str | None = None
        reason = "hold_position" if abs(current) > 1e-12 else "hold_flat"

        hard_target = float(np.clip(current, -risk_cap, risk_cap))
        if abs(hard_target - current) > 1e-12:
            current = hard_target
            resize_reason = "risk_projection"
            reason = resize_reason

        if (
            cadence
            and resize_reason is None
            and abs(current) > liquidity_cap + 1e-12
        ):
            resize_reason = "execution_contract_hold"
            reason = resize_reason

        if cadence and bool(actionable[index]):
            fast = int(fast_direction[index])
            observed_slow = int(slow_direction[index])
            current_sign = int(np.sign(current))
            if inherited and current_sign != 0:
                inherited_checks += 1
                if (
                    fast == observed_slow == current_sign
                    and bool(execution_eligible[index])
                ):
                    inherited_matches += 1
                if inherited_checks >= config.entry_confirmation_count:
                    if inherited_matches < config.entry_confirmation_count:
                        current = 0.0
                        reason = "exit"
                    elif resize_reason is None:
                        reason = "slow_support_hold"
                    inherited = False
                    inherited_checks = 0
                    inherited_matches = 0
                elif resize_reason is None:
                    reason = "confirmation_hold"
            elif current_sign == 0:
                if observed_slow != 0:
                    slow_regime = observed_slow
                coherent = (
                    fast
                    if fast != 0
                    and fast == slow_regime
                    and bool(execution_eligible[index])
                    else 0
                )
                entry_target = float(coherent * cap)
                entry_objective = causal_alpha_v6_fast_objective(
                    0.0,
                    entry_target,
                    float(fast_mean[index]),
                    float(fast_uncertainty[index]),
                    float(costs[index]),
                    economic_config,
                )
                if coherent == 0 or entry_objective <= 1e-12:
                    entry_intent = 0
                    entry_count = 0
                    reason = "cost_or_uncertainty_hold"
                elif abs(entry_target) < entry_floor:
                    entry_intent = 0
                    entry_count = 0
                    reason = "execution_contract_hold"
                else:
                    if coherent == entry_intent:
                        entry_count += 1
                    else:
                        entry_intent = coherent
                        entry_count = 1
                    if entry_count >= config.entry_confirmation_count:
                        current = entry_target
                        reason = "entry"
                        entry_intent = 0
                        entry_count = 0
                        fast_exit_count = 0
                        slow_exit_count = 0
                        neutral_slow_count = 0
                    else:
                        reason = "confirmation_hold"
            else:
                if observed_slow == current_sign:
                    slow_regime = observed_slow
                    slow_opposite_count = 0
                    neutral_slow_count = 0
                elif observed_slow == -current_sign:
                    slow_opposite_count += 1
                    neutral_slow_count = 0
                else:
                    slow_opposite_count = 0
                    neutral_slow_count += 1
                fast_exit_count = fast_exit_count + 1 if fast == -current_sign else 0
                slow_exit_count = slow_opposite_count
                should_exit = (
                    fast_exit_count >= config.exit_confirmation_count
                    or slow_exit_count >= config.exit_confirmation_count
                    or neutral_slow_count >= config.slow_neutral_expiry_count
                )
                if should_exit:
                    current = 0.0
                    reason = "exit"
                    fast_exit_count = 0
                    slow_exit_count = 0
                    slow_opposite_count = 0
                    neutral_slow_count = 0
                    slow_regime = 0
                elif resize_reason is None:
                    reason = (
                        "slow_support_hold"
                        if slow_regime == current_sign
                        else "confirmation_hold"
                    )
        elif cadence:
            if resize_reason is None:
                reason = "unactionable_hold"
        elif resize_reason is None:
            reason = "cadence_hold"

        targets[index] = current
        objectives[index] = causal_alpha_v6_fast_objective(
            previous_current,
            current,
            float(fast_mean[index]),
            float(fast_uncertainty[index]),
            float(costs[index]),
            economic_config,
        )
        confirmation_counts[index] = max(
            inherited_matches,
            entry_count,
            fast_exit_count,
            slow_exit_count,
            neutral_slow_count,
        )
        reasons.append(reason)
        slow_states.append(_slow_state(slow_regime, current))

    previous = np.concatenate(([initial_weight], targets[:-1]))
    forecast_digest = content_and_arrays_digest(
        {
            "dual_fit_digest": dual_fit_digest,
            "schema_version": "causal_alpha_v10_hierarchical_forecast_v1",
            "source_forecast_digest": source_forecast_digest,
        },
        (
            ("fast_head_predictions", fast_heads),
            ("slow_head_predictions", slow_heads),
        ),
    )
    reason_counts = tuple(
        sorted((reason, reasons.count(reason)) for reason in set(reasons))
    )
    return CausalAlphaV6TargetPath(
        candidate=CausalAlphaV6Candidate.FAST_ONLY,
        initial_weight=float(initial_weight),
        decision_indices=decisions,
        targets=targets,
        fast_proposals=fast_direction.astype(np.float64) * config.target_magnitude,
        expected_returns_4h=fast_mean,
        expected_returns_24h=np.zeros(rows),
        expected_returns_72h=slow_mean,
        direction_scores_4h=fast_direction.astype(np.float64),
        uncertainties_4h=fast_uncertainty,
        one_way_cost_rates=costs,
        liquidity_weight_caps=liquidity_caps,
        risk_weight_caps=risk_caps,
        objectives=objectives,
        confirmation_counts=confirmation_counts,
        actionable_mask=actionable,
        slow_states=tuple(slow_states),
        reasons=tuple(reasons),
        reason_counts=reason_counts,
        submitted_change_count=int(
            np.count_nonzero(np.abs(targets - previous) > 1e-12)
        ),
        sign_flip_count=int(np.count_nonzero(targets * previous < 0.0)),
        liquidity_deleveraging_count=reasons.count("liquidity_deleverage"),
        risk_projection_count=reasons.count("risk_projection"),
        forecast_digest=forecast_digest,
        config_digest=compiler_config_digest,
    )


__all__ = ["causal_alpha_v10_hierarchical_target_path"]
