"""Synthetic-only, fixed linear joint-action value heads.

Inference sees only the current portfolio snapshot; no market replay, cost
book, or evaluation dataset is accessible from the strategy package.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from numbers import Real
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioPendingOrderSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    decode_portfolio_action_codes,
)

_SCHEMA = "synthetic_linear_decision_value_v1"
_CONSERVATIVE_SCHEMA = "conservative_linear_decision_value_v2"
_POSITIVE_EDGE = 1e-12


def _finite(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field} must be a finite real")
    try:
        result = float(value)
    except OverflowError as error:
        raise ValueError(f"{field} must be finite") from error
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def _validated_hold_relative(
    hold_relative: bool, decision_margin_multiplier: float
) -> bool:
    if not isinstance(hold_relative, bool):
        raise TypeError("hold_relative must be a bool")
    margin = _finite(decision_margin_multiplier, field="decision_margin_multiplier")
    if margin < 0.0:
        raise ValueError("decision_margin_multiplier must be non-negative")
    if margin != 0.0 and not hold_relative:
        raise ValueError("a decision margin requires HOLD-relative decisions")
    return hold_relative


def compute_pending_order_weights(
    observation: PortfolioDecisionSnapshot,
) -> tuple[float, ...]:
    """Calculate signed pending order notionals as fractions of portfolio value."""
    if not isinstance(observation, PortfolioDecisionSnapshot):
        raise TypeError("observation must be PortfolioDecisionSnapshot")
    portfolio_value = _finite(
        observation.account.portfolio_value, field="portfolio_value"
    )
    if portfolio_value <= 0.0:
        raise ValueError("portfolio_value must be positive")
    n_symbols = len(observation.symbols)
    mark_prices: list[float] = []
    multipliers: list[float] = []
    for m in range(n_symbols):
        p = _finite(observation.account.mark_prices[m], field=f"mark_prices[{m}]")
        if p <= 0.0:
            raise ValueError("mark_prices must be positive")
        mark_prices.append(p)
        q = _finite(
            observation.account.contract_multipliers[m],
            field=f"contract_multipliers[{m}]",
        )
        if q <= 0.0:
            raise ValueError("contract_multipliers must be positive")
        multipliers.append(q)

    pending_notionals = [0.0] * n_symbols
    for order in observation.active_orders:
        if not isinstance(order, PortfolioPendingOrderSnapshot):
            raise TypeError("active order must be PortfolioPendingOrderSnapshot")
        if (
            type(order.symbol_index) is not int
            or order.symbol_index < 0
            or order.symbol_index >= n_symbols
        ):
            raise ValueError("order symbol_index is invalid")
        rem = _finite(order.remaining_quantity, field="order.remaining_quantity")
        pending_notionals[order.symbol_index] += (
            rem * mark_prices[order.symbol_index] * multipliers[order.symbol_index]
        )
    return tuple(float(notional / portfolio_value) for notional in pending_notionals)


def encode_decision_value_state(
    observation: PortfolioDecisionSnapshot,
    *,
    symbols: tuple[str, ...],
    feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
    allow_active_orders: bool = False,
) -> np.ndarray:
    """Encode only decision-time features and actual portfolio weights.

    This v1 head has no orders/pending-state representation. Reject such
    snapshots rather than silently extrapolating an unsupported state, unless
    allow_active_orders is explicitly opted in by the policy runtime.
    """
    if not isinstance(observation, PortfolioDecisionSnapshot):
        raise TypeError("observation must be PortfolioDecisionSnapshot")
    if (
        observation.symbols != symbols
        or observation.feature_names != feature_names
        or observation.global_feature_names != global_feature_names
    ):
        raise ValueError("decision feature or symbol roster differs from the model")
    if observation.account.insolvent or observation.account.termination_reason:
        raise ValueError("decision account has already terminated")
    if observation.active_orders and not allow_active_orders:
        raise ValueError("pending orders are unsupported by linear value v1")
    if any(not value for row in observation.feature_available for value in row) or any(
        not value for value in observation.global_feature_available
    ):
        raise ValueError("unavailable decision feature cannot be scored")
    values = [
        1.0,
        *(feature for row in observation.features for feature in row),
        *observation.global_features,
        *observation.account.weights,
    ]
    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("decision features must be finite")
    return features


def encode_decision_value_state_v2(
    observation: PortfolioDecisionSnapshot,
    *,
    symbols: tuple[str, ...],
    feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
) -> np.ndarray:
    """Encode decision features, actual portfolio weights, and pending order weights.

    Unlike v1, non-empty active_orders are supported and encoded as explicit in-flight
    pending weights.
    """
    if not isinstance(observation, PortfolioDecisionSnapshot):
        raise TypeError("observation must be PortfolioDecisionSnapshot")
    if (
        observation.symbols != symbols
        or observation.feature_names != feature_names
        or observation.global_feature_names != global_feature_names
    ):
        raise ValueError("decision feature or symbol roster differs from the model")
    if observation.account.insolvent or observation.account.termination_reason:
        raise ValueError("decision account has already terminated")
    if any(not value for row in observation.feature_available for value in row) or any(
        not value for value in observation.global_feature_available
    ):
        raise ValueError("unavailable decision feature cannot be scored")
    pending_weights = compute_pending_order_weights(observation)
    values = [
        1.0,
        *(feature for row in observation.features for feature in row),
        *observation.global_features,
        *observation.account.weights,
        *pending_weights,
    ]
    features = np.asarray(values, dtype=np.float64)
    if not np.isfinite(features).all():
        raise ValueError("decision features must be finite")
    return features


@dataclass(frozen=True, slots=True)
class LinearDecisionValueModel:
    """Versioned, immutable synthetic action-value head with fixed action order."""

    schema_version: str
    source_scope: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    feature_config_digest: str
    normalization_digest: str
    execution_policy_digest: str
    reference_policy_digest: str
    target_abs_weights: tuple[float, ...]
    candidate_actions: tuple[tuple[int, ...], ...]
    fit_cutoff_ns: int
    evaluation_start_ns: int
    ridge_penalty: float
    training_rows: int
    overlap_groups: int
    coefficients: tuple[tuple[float, ...], ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA or self.source_scope not in (
            "synthetic_only",
            "real_development",
        ):
            raise ValueError("only the versioned synthetic linear model is supported")
        if any(
            type(value) is not tuple
            for value in (self.symbols, self.feature_names, self.global_feature_names)
        ):
            raise TypeError("model symbol and feature rosters must be tuples")
        require_unique_non_empty(self.symbols, field="symbols")
        require_unique_non_empty(self.feature_names, field="feature_names")
        require_unique_non_empty(
            self.global_feature_names, field="global_feature_names"
        )
        for name in (
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        if any(
            type(getattr(self, field)) is not int
            for field in (
                "fit_cutoff_ns",
                "evaluation_start_ns",
                "training_rows",
                "overlap_groups",
            )
        ):
            raise ValueError("model cutoffs and counts must be native integers")
        if self.fit_cutoff_ns >= self.evaluation_start_ns:
            raise ValueError("evaluation must begin after the fit cutoff")
        if self.training_rows <= 0 or self.overlap_groups <= 0:
            raise ValueError("training counts must be positive")
        penalty = _finite(self.ridge_penalty, field="ridge_penalty")
        if penalty <= 0:
            raise ValueError("ridge_penalty must be positive")
        if type(self.target_abs_weights) is not tuple or len(
            self.target_abs_weights
        ) != len(self.symbols):
            raise ValueError("target weights must match symbol roster")
        if any(_finite(w, field="target weight") < 0 for w in self.target_abs_weights):
            raise ValueError("target weights must be non-negative")
        if type(self.candidate_actions) is not tuple or len(self.candidate_actions) < 2:
            raise ValueError("candidate actions require HOLD and another action")
        seen: set[tuple[int, ...]] = set()
        for action in self.candidate_actions:
            if type(action) is not tuple:
                raise TypeError("candidate action must be a tuple")
            codes = decode_portfolio_action_codes(action, n_symbols=len(self.symbols))
            if codes in seen:
                raise ValueError("candidate actions must be unique")
            seen.add(codes)
        if self.candidate_actions[0] != (ACTION_HOLD,) * len(self.symbols):
            raise ValueError("first candidate must be joint HOLD")
        width = (
            1
            + len(self.symbols) * len(self.feature_names)
            + len(self.global_feature_names)
            + len(self.symbols)
        )
        if (
            type(self.coefficients) is not tuple
            or len(self.coefficients) != len(self.candidate_actions) - 1
        ):
            raise ValueError("coefficients must match candidate count")
        for row in self.coefficients:
            if type(row) is not tuple or len(row) != width:
                raise ValueError("coefficients have incorrect state dimension")
            for value in row:
                _finite(value, field="coefficient")


@dataclass(frozen=True, slots=True)
class LinearDecisionValueRuntimeContext:
    """Caller-declared synthetic runtime recipe; external attestation is separate."""

    source_scope: str
    feature_config_digest: str
    normalization_digest: str
    execution_policy_digest: str
    reference_policy_digest: str

    def __post_init__(self) -> None:
        if self.source_scope not in ("synthetic_only", "real_development"):
            raise ValueError("synthetic-only runtime context is required")
        for field_name in (
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            require_sha256(getattr(self, field_name), field=field_name)


def compute_effective_posture_weights(
    snapshot: PortfolioDecisionSnapshot,
    target_abs_weights: tuple[float, ...],
) -> tuple[float, ...]:
    """Calculate effective settled + genuinely in-flight portfolio weights."""
    if not isinstance(snapshot, PortfolioDecisionSnapshot):
        raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
    n = len(snapshot.symbols)
    if len(target_abs_weights) != n:
        raise ValueError("target_abs_weights length must match symbols length")
    total_pend = (
        compute_pending_order_weights(snapshot)
        if snapshot.active_orders
        else (0.0,) * n
    )
    effective: list[float] = [0.0] * n
    for i in range(n):
        sym_orders = [o for o in snapshot.active_orders if o.symbol_index == i]
        waiting = any(
            o.status in ("submitted", "latency_wait", "triggered")
            or o.last_processed_index is None
            or o.eligible_index > snapshot.index
            for o in sym_orders
        )
        meaningful_partial = abs(float(total_pend[i])) > max(
            1e-4, 0.02 * float(target_abs_weights[i])
        )
        has_inflight = bool(sym_orders) and (waiting or meaningful_partial)
        w_act = float(snapshot.account.weights[i])
        w_total_eff = w_act + float(total_pend[i])
        if has_inflight and abs(w_total_eff) <= 1e-6:
            effective[i] = 0.0
        else:
            effective[i] = w_act + (float(total_pend[i]) if has_inflight else 0.0)
    return tuple(effective)


def candidate_target_weights(
    action: tuple[int, ...],
    effective_weights: tuple[float, ...],
    target_abs_weights: tuple[float, ...],
) -> tuple[float, ...]:
    """Signed target weight each symbol would hold after ``action``.

    HOLD keeps the current effective weight, FLAT is zero, LONG/SHORT are the
    declared signed budget.
    """
    n = len(target_abs_weights)
    codes = decode_portfolio_action_codes(action, n_symbols=n)
    targets: list[float] = []
    for i, code in enumerate(codes):
        if code == ACTION_HOLD:
            targets.append(float(effective_weights[i]))
        elif code == ACTION_FLAT:
            targets.append(0.0)
        elif code == ACTION_LONG:
            targets.append(float(target_abs_weights[i]))
        else:  # ACTION_SHORT
            targets.append(-float(target_abs_weights[i]))
    return tuple(targets)


def round_trip_cost_fraction(
    target_weights: tuple[float, ...], one_way_cost: float
) -> float:
    """Capital fraction paid by entering and later closing ``target_weights``."""
    cost = _finite(one_way_cost, field="one_way_cost")
    if cost < 0.0:
        raise ValueError("one_way_cost must be non-negative")
    gross = sum(abs(_finite(w, field="target weight")) for w in target_weights)
    return float(2.0 * cost * gross)


def compute_posture_cost_adjustment(
    action: tuple[int, ...],
    effective_weights: tuple[float, ...],
    target_abs_weights: tuple[float, ...],
    *,
    eval_one_way_cost: float,
    train_one_way_cost: float,
) -> float:
    """Signed incremental turnover switching and sunk-cost adjustment relative to flat start."""
    eval_c = _finite(eval_one_way_cost, field="eval_one_way_cost")
    train_c = _finite(train_one_way_cost, field="train_one_way_cost")
    targets = candidate_target_weights(action, effective_weights, target_abs_weights)

    turnover_diff = sum(
        abs(t - e) for t, e in zip(targets, effective_weights, strict=True)
    ) - sum(abs(t) for t in targets)
    extra_train_diff = 2.0 * (eval_c - train_c) * sum(abs(t) for t in targets)
    return float(eval_c * turnover_diff + extra_train_diff)


def compute_hold_relative_cost_adjustment(
    action: tuple[int, ...],
    effective_weights: tuple[float, ...],
    target_abs_weights: tuple[float, ...],
    *,
    eval_one_way_cost: float,
    train_one_way_cost: float,
) -> float:
    """Stress-cost correction for a label that is already an increment over HOLD.

    A HOLD-relative label from a reachable state already contains the training
    cost of trading now and closing at the horizon. Only the *difference*
    between the evaluation and training one-way cost, applied to the extra
    turnover versus HOLD (``|t-e|`` now, ``|t|`` closed later, ``|e|`` saved),
    is added back. It is zero whenever evaluation and training costs agree.
    """
    eval_c = _finite(eval_one_way_cost, field="eval_one_way_cost")
    train_c = _finite(train_one_way_cost, field="train_one_way_cost")
    targets = candidate_target_weights(action, effective_weights, target_abs_weights)
    extra_turnover = (
        sum(abs(t - e) for t, e in zip(targets, effective_weights, strict=True))
        + sum(abs(t) for t in targets)
        - sum(abs(e) for e in effective_weights)
    )
    return float((eval_c - train_c) * extra_turnover)


def select_hold_relative_action(
    candidate_actions: tuple[tuple[int, ...], ...],
    snapshot: PortfolioDecisionSnapshot,
    increment_for_action: Callable[[tuple[int, ...]], float],
    *,
    target_abs_weights: tuple[float, ...],
    eval_one_way_cost: float,
    margin_multiplier: float,
) -> tuple[int, ...]:
    """Pick the action whose increment over HOLD most exceeds its margin.

    The margin of a candidate is ``margin_multiplier`` times the round-trip cost
    fraction at that candidate's target weights. HOLD is the structural origin;
    an action is chosen only when ``increment - margin`` is strictly positive
    (and above the legacy numerical floor), so ties go to HOLD. A candidate whose
    native proposal is identical to HOLD's is a no-op and never an edge.
    """
    multiplier = _finite(margin_multiplier, field="margin_multiplier")
    if multiplier < 0.0:
        raise ValueError("margin_multiplier must be non-negative")
    hold = candidate_actions[0]
    effective = compute_effective_posture_weights(snapshot, target_abs_weights)
    hold_proposal = preserve_posture_aware_proposal(hold, snapshot, target_abs_weights)
    best_action = hold
    best_excess = _POSITIVE_EDGE
    for action in candidate_actions[1:]:
        proposal = preserve_posture_aware_proposal(action, snapshot, target_abs_weights)
        if (
            proposal.target_weights == hold_proposal.target_weights
            and proposal.hold_current_quantity_symbols
            == hold_proposal.hold_current_quantity_symbols
        ):
            continue
        margin = multiplier * round_trip_cost_fraction(
            candidate_target_weights(action, effective, target_abs_weights),
            eval_one_way_cost,
        )
        excess = float(increment_for_action(action)) - margin
        if excess > best_excess:
            best_excess = excess
            best_action = action
    return best_action


def preserve_posture_aware_proposal(
    action: tuple[int, ...],
    snapshot: PortfolioDecisionSnapshot,
    target_abs_weights: tuple[float, ...],
    *,
    all_scores_non_positive: bool = False,
) -> PortfolioTargetProposal:
    """Posture-aware proposal helper preserving in-flight orders and suppressing churn."""
    if not isinstance(snapshot, PortfolioDecisionSnapshot):
        raise TypeError("snapshot must be a PortfolioDecisionSnapshot")
    if snapshot.account.insolvent or snapshot.account.termination_reason is not None:
        raise ValueError("decision account has already terminated")
    n = len(snapshot.symbols)
    if type(target_abs_weights) is not tuple or len(target_abs_weights) != n:
        raise ValueError("target_abs_weights length must match symbols length")
    for w in target_abs_weights:
        if _finite(w, field="target weight") < 0.0:
            raise ValueError("target weights must be non-negative")
    codes = decode_portfolio_action_codes(action, n_symbols=n)

    peak = float(snapshot.account.peak_value)
    dd = (
        max(0.0, 1.0 - float(snapshot.account.portfolio_value) / peak)
        if peak > 0.0
        else 0.0
    )
    start = snapshot.risk.drawdown_start
    stop = snapshot.risk.drawdown_stop
    if dd <= start:
        risk_scale = 1.0
    elif dd >= stop or stop == start:
        risk_scale = 0.0
    else:
        risk_scale = 1.0 - (dd - start) / (stop - start)

    max_abs_cap = snapshot.risk.max_abs_weight * risk_scale
    max_gross_cap = snapshot.risk.max_gross * risk_scale

    total_pend = (
        compute_pending_order_weights(snapshot)
        if snapshot.active_orders
        else (0.0,) * n
    )

    has_inflight: list[bool] = [False] * n
    for i in range(n):
        sym_orders = [o for o in snapshot.active_orders if o.symbol_index == i]
        waiting = any(
            o.status in ("submitted", "latency_wait", "triggered")
            or o.last_processed_index is None
            or o.eligible_index > snapshot.index
            for o in sym_orders
        )
        meaningful_partial = abs(float(total_pend[i])) > max(
            1e-4, 0.02 * float(target_abs_weights[i])
        )
        has_inflight[i] = bool(sym_orders) and (waiting or meaningful_partial)

    target_weights = [0.0] * n
    hold_symbols: set[int] = set()

    for i, code in enumerate(codes):
        w_act = float(snapshot.account.weights[i])
        exact_qty = Fraction(snapshot.account.exact_quantities[i])
        d_act = 1 if exact_qty > 0 else (-1 if exact_qty < 0 else 0)
        w_total_eff = w_act + float(total_pend[i])

        if has_inflight[i] and abs(w_total_eff) <= 1e-6:
            flattening_inflight = True
            d_eff = 0
            w_eff = 0.0
        else:
            flattening_inflight = False
            w_eff = w_act + (float(total_pend[i]) if has_inflight[i] else 0.0)
            d_eff = 1 if w_eff > 1e-6 else (-1 if w_eff < -1e-6 else 0)

        if all_scores_non_positive:
            d_star = 0
        elif code == ACTION_HOLD:
            d_star = d_eff
        elif code == ACTION_FLAT:
            d_star = 0
        elif code == ACTION_LONG:
            d_star = 1
        else:  # ACTION_SHORT
            d_star = -1

        if d_star == 0:
            if flattening_inflight or d_act != 0:
                target_weights[i] = 0.0
            else:
                target_weights[i] = w_act
                sym_orders = [o for o in snapshot.active_orders if o.symbol_index == i]
                if not sym_orders:
                    hold_symbols.add(i)
        elif d_star == d_eff:
            if has_inflight[i] and abs(w_eff) <= max_abs_cap + 1e-12:
                target_weights[i] = w_eff
            elif d_act == d_star:
                target_weights[i] = w_act
                hold_symbols.add(i)
            else:
                target_weights[i] = min(
                    float(target_abs_weights[i]), max_abs_cap
                ) * float(d_star)
        else:
            target_weights[i] = min(float(target_abs_weights[i]), max_abs_cap) * float(
                d_star
            )

    gross = sum(abs(w) for w in target_weights)
    if gross > max_gross_cap + 1e-12 and gross > 0.0:
        held_gross = sum(abs(target_weights[i]) for i in hold_symbols)
        free_gross = sum(
            abs(target_weights[i]) for i in range(n) if i not in hold_symbols
        )
        if held_gross <= max_gross_cap + 1e-12 and free_gross > 0.0:
            free_scale = max(0.0, max_gross_cap - held_gross) / free_gross
            for i in range(n):
                if i not in hold_symbols:
                    target_weights[i] *= free_scale
        else:
            hold_symbols.clear()
            scale = max_gross_cap / gross
            for i in range(n):
                target_weights[i] *= scale

    return PortfolioTargetProposal(
        target_weights=tuple(target_weights),
        hold_current_quantity_symbols=tuple(sorted(hold_symbols)),
    )


class LinearDecisionValuePolicy:
    """Synthetic-only PortfolioPolicy; hard risk and native execution stay downstream."""

    def __init__(
        self,
        model: LinearDecisionValueModel,
        *,
        runtime_context: LinearDecisionValueRuntimeContext,
        allow_active_orders: bool = False,
        eval_one_way_cost: float = 0.0,
        train_one_way_cost: float = 0.0,
        hold_relative: bool = False,
        decision_margin_multiplier: float = 0.0,
    ) -> None:
        if not isinstance(model, LinearDecisionValueModel):
            raise TypeError("model must be LinearDecisionValueModel")
        if not isinstance(runtime_context, LinearDecisionValueRuntimeContext):
            raise TypeError("runtime_context must be a typed synthetic runtime context")
        for field_name in (
            "source_scope",
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            if getattr(model, field_name) != getattr(runtime_context, field_name):
                raise ValueError(f"runtime {field_name} differs from fitted model")
        self.model = model
        self.allow_active_orders = allow_active_orders
        self.eval_one_way_cost = _finite(eval_one_way_cost, field="eval_one_way_cost")
        self.train_one_way_cost = _finite(
            train_one_way_cost, field="train_one_way_cost"
        )
        self.hold_relative = _validated_hold_relative(
            hold_relative, decision_margin_multiplier
        )
        self.decision_margin_multiplier = float(decision_margin_multiplier)
        self._coefs = [np.asarray(row, dtype=np.float64) for row in model.coefficients]

    def _features(self, snapshot: PortfolioDecisionSnapshot) -> np.ndarray:
        if snapshot.timestamp_ns < self.model.evaluation_start_ns:
            raise ValueError("policy decision precedes evaluation start")
        return encode_decision_value_state(
            snapshot,
            symbols=self.model.symbols,
            feature_names=self.model.feature_names,
            global_feature_names=self.model.global_feature_names,
            allow_active_orders=self.allow_active_orders,
        )

    def value_for_action(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        if action not in self.model.candidate_actions:
            raise ValueError("action is not a declared candidate")
        if self.hold_relative:
            if action == self.model.candidate_actions[0]:
                return 0.0
            return self._hold_relative_value(snapshot, action)
        n_symbols = len(self.model.symbols)
        if action == (ACTION_FLAT,) * n_symbols:
            return 0.0
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        if action == self.model.candidate_actions[0]:
            if all(abs(w) <= 1e-6 for w in w_eff):
                return 0.0
            d_eff = tuple(1 if w > 1e-6 else (-1 if w < -1e-6 else 0) for w in w_eff)
            for cand in self.model.candidate_actions[1:]:
                cand_codes = decode_portfolio_action_codes(cand, n_symbols=n_symbols)
                cand_dir = tuple(
                    1 if c == ACTION_LONG else (-1 if c == ACTION_SHORT else 0)
                    for c in cand_codes
                )
                if cand_dir == d_eff:
                    return self.value_for_action(snapshot, cand)
            return 0.0

        k = self.model.candidate_actions.index(action) - 1
        features = self._features(snapshot)
        raw_val = float(features @ self._coefs[k])
        if not math.isfinite(raw_val):
            raise ValueError("predicted decision value must be finite")
        adj_cost = compute_posture_cost_adjustment(
            action,
            w_eff,
            self.model.target_abs_weights,
            eval_one_way_cost=self.eval_one_way_cost,
            train_one_way_cost=self.train_one_way_cost,
        )
        return float(raw_val - adj_cost)

    def _hold_relative_value(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        k = self.model.candidate_actions.index(action) - 1
        raw_val = float(self._features(snapshot) @ self._coefs[k])
        if not math.isfinite(raw_val):
            raise ValueError("predicted decision value must be finite")
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        return float(
            raw_val
            - compute_hold_relative_cost_adjustment(
                action,
                w_eff,
                self.model.target_abs_weights,
                eval_one_way_cost=self.eval_one_way_cost,
                train_one_way_cost=self.train_one_way_cost,
            )
        )

    def _select_best_action(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[tuple[int, ...], bool]:
        if self.hold_relative:
            return (
                select_hold_relative_action(
                    self.model.candidate_actions,
                    snapshot,
                    lambda action: self.value_for_action(snapshot, action),
                    target_abs_weights=self.model.target_abs_weights,
                    eval_one_way_cost=self.eval_one_way_cost,
                    margin_multiplier=self.decision_margin_multiplier,
                ),
                False,
            )
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        is_invested = any(abs(w) > 1e-6 for w in w_eff) or any(
            Fraction(q) != 0 for q in snapshot.account.exact_quantities
        )
        n_symbols = len(self.model.symbols)
        flat_action = (ACTION_FLAT,) * n_symbols
        hold_action = (ACTION_HOLD,) * n_symbols

        best_index = -1
        best_score = _POSITIVE_EDGE
        for index in range(1, len(self.model.candidate_actions)):
            action = self.model.candidate_actions[index]
            if action == flat_action:
                continue
            score = self.value_for_action(snapshot, action)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index != -1:
            return self.model.candidate_actions[best_index], False

        all_non_positive = True
        if is_invested:
            if flat_action in self.model.candidate_actions:
                chosen = flat_action
            else:
                chosen = hold_action
        else:
            chosen = hold_action
        return chosen, all_non_positive

    def decide_action(self, snapshot: PortfolioDecisionSnapshot) -> tuple[int, ...]:
        chosen, _ = self._select_best_action(snapshot)
        return chosen

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        chosen, all_non_positive = self._select_best_action(snapshot)
        return preserve_posture_aware_proposal(
            chosen,
            snapshot,
            self.model.target_abs_weights,
            all_scores_non_positive=all_non_positive,
        )


def serialize_linear_decision_value(
    model: LinearDecisionValueModel,
) -> tuple[str, str]:
    """Canonical JSON string and digest, without creating a second artifact store."""
    if not isinstance(model, LinearDecisionValueModel):
        raise TypeError("model must be LinearDecisionValueModel")
    encoded = json.dumps(
        asdict(model),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_linear_decision_value(
    text: str, *, expected_sha256: str
) -> LinearDecisionValueModel:
    """Require the caller's trusted digest; reject duplicate or noncanonical JSON."""
    require_sha256(expected_sha256, field="expected digest")
    if type(text) is not str:
        raise TypeError("model artifact must be UTF-8 JSON text")
    try:
        value: object = json.loads(text)
        canonical = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("model artifact contains invalid JSON") from error
    if canonical != text:
        raise ValueError("model artifact is not canonical JSON")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256:
        raise ValueError("trusted model digest mismatch")
    if not isinstance(value, dict):
        raise ValueError("model payload must be an object")
    expected = {field.name for field in fields(LinearDecisionValueModel)}
    if set(value) != expected:
        raise ValueError("model payload fields differ from the declared schema")
    raw = cast(dict[str, Any], value)
    try:
        model = LinearDecisionValueModel(
            schema_version=raw["schema_version"],
            source_scope=raw["source_scope"],
            symbols=tuple(raw["symbols"]),
            feature_names=tuple(raw["feature_names"]),
            global_feature_names=tuple(raw["global_feature_names"]),
            feature_config_digest=raw["feature_config_digest"],
            normalization_digest=raw["normalization_digest"],
            execution_policy_digest=raw["execution_policy_digest"],
            reference_policy_digest=raw["reference_policy_digest"],
            target_abs_weights=tuple(raw["target_abs_weights"]),
            candidate_actions=tuple(tuple(row) for row in raw["candidate_actions"]),
            fit_cutoff_ns=raw["fit_cutoff_ns"],
            evaluation_start_ns=raw["evaluation_start_ns"],
            ridge_penalty=raw["ridge_penalty"],
            training_rows=raw["training_rows"],
            overlap_groups=raw["overlap_groups"],
            coefficients=tuple(tuple(row) for row in raw["coefficients"]),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model payload violates the declared schema") from error
    return model


@dataclass(frozen=True, slots=True)
class ConservativeLinearDecisionValueModel:
    """Versioned action-value head with uncertainty penalty and in-flight state."""

    schema_version: str
    source_scope: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    feature_config_digest: str
    normalization_digest: str
    execution_policy_digest: str
    reference_policy_digest: str
    target_abs_weights: tuple[float, ...]
    candidate_actions: tuple[tuple[int, ...], ...]
    fit_cutoff_ns: int
    evaluation_start_ns: int
    ridge_penalty: float
    uncertainty_penalty_weight: float
    one_way_switch_cost: float
    training_rows: int
    overlap_groups: int
    coefficients: tuple[tuple[float, ...], ...]
    covariance_matrices: tuple[tuple[tuple[float, ...], ...], ...]

    def __post_init__(self) -> None:
        if self.schema_version != _CONSERVATIVE_SCHEMA or self.source_scope not in (
            "synthetic_only",
            "real_development",
        ):
            raise ValueError("only the conservative linear model schema is supported")
        if any(
            type(value) is not tuple
            for value in (self.symbols, self.feature_names, self.global_feature_names)
        ):
            raise TypeError("model symbol and feature rosters must be tuples")
        require_unique_non_empty(self.symbols, field="symbols")
        require_unique_non_empty(self.feature_names, field="feature_names")
        require_unique_non_empty(
            self.global_feature_names, field="global_feature_names"
        )
        for name in (
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        if any(
            type(getattr(self, field)) is not int
            for field in (
                "fit_cutoff_ns",
                "evaluation_start_ns",
                "training_rows",
                "overlap_groups",
            )
        ):
            raise ValueError("model cutoffs and counts must be native integers")
        if self.fit_cutoff_ns >= self.evaluation_start_ns:
            raise ValueError("evaluation must begin after the fit cutoff")
        if self.training_rows <= 0 or self.overlap_groups <= 0:
            raise ValueError("training counts must be positive")
        penalty = _finite(self.ridge_penalty, field="ridge_penalty")
        if penalty <= 0.0:
            raise ValueError("ridge_penalty must be positive")
        unc_w = _finite(
            self.uncertainty_penalty_weight, field="uncertainty_penalty_weight"
        )
        if unc_w < 0.0:
            raise ValueError("uncertainty_penalty_weight must be non-negative")
        switch_c = _finite(self.one_way_switch_cost, field="one_way_switch_cost")
        if switch_c < 0.0:
            raise ValueError("one_way_switch_cost must be non-negative")
        if type(self.target_abs_weights) is not tuple or len(
            self.target_abs_weights
        ) != len(self.symbols):
            raise ValueError("target weights must match symbol roster")
        if any(
            _finite(w, field="target weight") < 0.0 for w in self.target_abs_weights
        ):
            raise ValueError("target weights must be non-negative")
        if type(self.candidate_actions) is not tuple or len(self.candidate_actions) < 2:
            raise ValueError("candidate actions require HOLD and another action")
        seen: set[tuple[int, ...]] = set()
        for action in self.candidate_actions:
            if type(action) is not tuple:
                raise TypeError("candidate action must be a tuple")
            codes = decode_portfolio_action_codes(action, n_symbols=len(self.symbols))
            if codes in seen:
                raise ValueError("candidate actions must be unique")
            seen.add(codes)
        if self.candidate_actions[0] != (ACTION_HOLD,) * len(self.symbols):
            raise ValueError("first candidate must be joint HOLD")
        width = (
            1
            + len(self.symbols) * len(self.feature_names)
            + len(self.global_feature_names)
            + 2 * len(self.symbols)
        )
        n_candidates = len(self.candidate_actions) - 1
        if (
            type(self.coefficients) is not tuple
            or len(self.coefficients) != n_candidates
        ):
            raise ValueError("coefficients length must match non-HOLD actions")
        for row in self.coefficients:
            if type(row) is not tuple or len(row) != width:
                raise ValueError("coefficients row length must match model width")
            for value in row:
                _finite(value, field="coefficient")
        if (
            type(self.covariance_matrices) is not tuple
            or len(self.covariance_matrices) != n_candidates
        ):
            raise ValueError("covariance_matrices length must match non-HOLD actions")
        for matrix in self.covariance_matrices:
            if type(matrix) is not tuple or len(matrix) != width:
                raise ValueError(
                    "covariance matrix shape must be square (width x width)"
                )
            for i, row in enumerate(matrix):
                if type(row) is not tuple or len(row) != width:
                    raise ValueError("covariance matrix row length must match width")
                for j, value in enumerate(row):
                    _finite(value, field="covariance element")
                if matrix[i][i] < 0.0:
                    raise ValueError(
                        "covariance diagonal elements must be non-negative"
                    )
            for i in range(width):
                for j in range(i + 1, width):
                    if not math.isclose(
                        matrix[i][j], matrix[j][i], rel_tol=1e-5, abs_tol=1e-8
                    ):
                        raise ValueError("covariance matrix must be symmetric")


class ConservativeLinearDecisionValuePolicy:
    """Conservative PortfolioPolicy with uncertainty penalty and in-flight order handling."""

    def __init__(
        self,
        model: ConservativeLinearDecisionValueModel,
        *,
        runtime_context: LinearDecisionValueRuntimeContext,
        allow_active_orders: bool = True,
        eval_one_way_cost: float | None = None,
        train_one_way_cost: float | None = None,
        hold_relative: bool = False,
        decision_margin_multiplier: float = 0.0,
    ) -> None:
        if not isinstance(model, ConservativeLinearDecisionValueModel):
            raise TypeError("model must be ConservativeLinearDecisionValueModel")
        if not isinstance(runtime_context, LinearDecisionValueRuntimeContext):
            raise TypeError("runtime_context must be a typed runtime context")
        for field_name in (
            "source_scope",
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            if getattr(model, field_name) != getattr(runtime_context, field_name):
                raise ValueError(f"runtime {field_name} differs from fitted model")
        self.model = model
        self.eval_one_way_cost = (
            model.one_way_switch_cost
            if eval_one_way_cost is None
            else _finite(eval_one_way_cost, field="eval_one_way_cost")
        )
        self.train_one_way_cost = (
            model.one_way_switch_cost
            if train_one_way_cost is None
            else _finite(train_one_way_cost, field="train_one_way_cost")
        )
        self.hold_relative = _validated_hold_relative(
            hold_relative, decision_margin_multiplier
        )
        self.decision_margin_multiplier = float(decision_margin_multiplier)
        self._coefs = [np.asarray(row, dtype=np.float64) for row in model.coefficients]
        self._covs = [
            np.asarray(m, dtype=np.float64) for m in model.covariance_matrices
        ]

    def _features(self, snapshot: PortfolioDecisionSnapshot) -> np.ndarray:
        if snapshot.timestamp_ns < self.model.evaluation_start_ns:
            raise ValueError("policy decision precedes evaluation start")
        return encode_decision_value_state_v2(
            snapshot,
            symbols=self.model.symbols,
            feature_names=self.model.feature_names,
            global_feature_names=self.model.global_feature_names,
        )

    def predictive_std_for_action(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        if action not in self.model.candidate_actions:
            raise ValueError("action is not a declared candidate")
        if self.hold_relative and action == self.model.candidate_actions[0]:
            return 0.0
        if action == self.model.candidate_actions[0]:
            w_eff = compute_effective_posture_weights(
                snapshot, self.model.target_abs_weights
            )
            if all(abs(w) <= 1e-6 for w in w_eff):
                return 0.0
            n_symbols = len(self.model.symbols)
            d_eff = tuple(1 if w > 1e-6 else (-1 if w < -1e-6 else 0) for w in w_eff)
            for cand in self.model.candidate_actions[1:]:
                cand_codes = decode_portfolio_action_codes(cand, n_symbols=n_symbols)
                cand_dir = tuple(
                    1 if c == ACTION_LONG else (-1 if c == ACTION_SHORT else 0)
                    for c in cand_codes
                )
                if cand_dir == d_eff:
                    return self.predictive_std_for_action(snapshot, cand)
            return 0.0
        k = self.model.candidate_actions.index(action) - 1
        features = self._features(snapshot)
        var = float(features @ self._covs[k] @ features)
        return math.sqrt(max(0.0, var))

    def conservative_value_for_action(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        if action not in self.model.candidate_actions:
            raise ValueError("action is not a declared candidate")
        if self.hold_relative:
            if action == self.model.candidate_actions[0]:
                return 0.0
            return self._hold_relative_lcb(snapshot, action)
        n_symbols = len(self.model.symbols)
        if action == (ACTION_FLAT,) * n_symbols:
            return 0.0
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        if action == self.model.candidate_actions[0]:
            if all(abs(w) <= 1e-6 for w in w_eff):
                return 0.0
            d_eff = tuple(1 if w > 1e-6 else (-1 if w < -1e-6 else 0) for w in w_eff)
            for cand in self.model.candidate_actions[1:]:
                cand_codes = decode_portfolio_action_codes(cand, n_symbols=n_symbols)
                cand_dir = tuple(
                    1 if c == ACTION_LONG else (-1 if c == ACTION_SHORT else 0)
                    for c in cand_codes
                )
                if cand_dir == d_eff:
                    return self.conservative_value_for_action(snapshot, cand)
            return 0.0

        k = self.model.candidate_actions.index(action) - 1
        features = self._features(snapshot)
        mean_val = float(features @ self._coefs[k])
        se = self.predictive_std_for_action(snapshot, action)
        adj_cost = compute_posture_cost_adjustment(
            action,
            w_eff,
            self.model.target_abs_weights,
            eval_one_way_cost=self.eval_one_way_cost,
            train_one_way_cost=self.train_one_way_cost,
        )
        return float(mean_val - self.model.uncertainty_penalty_weight * se - adj_cost)

    def value_for_action(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        return self.conservative_value_for_action(snapshot, action)

    def _hold_relative_lcb(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        k = self.model.candidate_actions.index(action) - 1
        features = self._features(snapshot)
        mean_val = float(features @ self._coefs[k])
        var = float(features @ self._covs[k] @ features)
        se = math.sqrt(max(0.0, var))
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        adj_cost = compute_hold_relative_cost_adjustment(
            action,
            w_eff,
            self.model.target_abs_weights,
            eval_one_way_cost=self.eval_one_way_cost,
            train_one_way_cost=self.train_one_way_cost,
        )
        return float(mean_val - self.model.uncertainty_penalty_weight * se - adj_cost)

    def _select_best_action(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> tuple[tuple[int, ...], bool]:
        if self.hold_relative:
            return (
                select_hold_relative_action(
                    self.model.candidate_actions,
                    snapshot,
                    lambda action: self.conservative_value_for_action(snapshot, action),
                    target_abs_weights=self.model.target_abs_weights,
                    eval_one_way_cost=self.eval_one_way_cost,
                    margin_multiplier=self.decision_margin_multiplier,
                ),
                False,
            )
        w_eff = compute_effective_posture_weights(
            snapshot, self.model.target_abs_weights
        )
        is_invested = any(abs(w) > 1e-6 for w in w_eff) or any(
            Fraction(q) != 0 for q in snapshot.account.exact_quantities
        )
        n_symbols = len(self.model.symbols)
        flat_action = (ACTION_FLAT,) * n_symbols
        hold_action = (ACTION_HOLD,) * n_symbols

        best_index = -1
        best_score = _POSITIVE_EDGE
        for index in range(1, len(self.model.candidate_actions)):
            action = self.model.candidate_actions[index]
            if action == flat_action:
                continue
            score = self.conservative_value_for_action(snapshot, action)
            if score > best_score:
                best_score = score
                best_index = index

        if best_index != -1:
            return self.model.candidate_actions[best_index], False

        all_non_positive = True
        if is_invested:
            if flat_action in self.model.candidate_actions:
                chosen = flat_action
            else:
                chosen = hold_action
        else:
            chosen = hold_action
        return chosen, all_non_positive

    def decide_action(self, snapshot: PortfolioDecisionSnapshot) -> tuple[int, ...]:
        chosen, _ = self._select_best_action(snapshot)
        return chosen

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        chosen, all_non_positive = self._select_best_action(snapshot)
        return preserve_posture_aware_proposal(
            chosen,
            snapshot,
            self.model.target_abs_weights,
            all_scores_non_positive=all_non_positive,
        )


def serialize_conservative_linear_decision_value(
    model: ConservativeLinearDecisionValueModel,
) -> tuple[str, str]:
    """Canonical JSON string and digest for conservative linear value model."""
    if not isinstance(model, ConservativeLinearDecisionValueModel):
        raise TypeError("model must be ConservativeLinearDecisionValueModel")
    encoded = json.dumps(
        asdict(model),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_conservative_linear_decision_value(
    text: str, *, expected_sha256: str
) -> ConservativeLinearDecisionValueModel:
    """Require caller's trusted digest; reject duplicate or noncanonical JSON."""
    require_sha256(expected_sha256, field="expected digest")
    if type(text) is not str:
        raise TypeError("model artifact must be UTF-8 JSON text")
    try:
        value: object = json.loads(text)
        canonical = json.dumps(
            value,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError("model artifact contains invalid JSON") from error
    if canonical != text:
        raise ValueError("model artifact is not canonical JSON")
    if hashlib.sha256(text.encode("utf-8")).hexdigest() != expected_sha256:
        raise ValueError("trusted model digest mismatch")
    if not isinstance(value, dict):
        raise ValueError("model payload must be an object")
    expected_fields = {
        field.name for field in fields(ConservativeLinearDecisionValueModel)
    }
    if set(value) != expected_fields:
        raise ValueError("model payload fields differ from the declared schema")
    raw = cast(dict[str, Any], value)
    if raw.get("schema_version") != _CONSERVATIVE_SCHEMA:
        raise ValueError("unsupported model artifact schema")
    try:
        model = ConservativeLinearDecisionValueModel(
            schema_version=raw["schema_version"],
            source_scope=raw["source_scope"],
            symbols=tuple(raw["symbols"]),
            feature_names=tuple(raw["feature_names"]),
            global_feature_names=tuple(raw["global_feature_names"]),
            feature_config_digest=raw["feature_config_digest"],
            normalization_digest=raw["normalization_digest"],
            execution_policy_digest=raw["execution_policy_digest"],
            reference_policy_digest=raw["reference_policy_digest"],
            target_abs_weights=tuple(raw["target_abs_weights"]),
            candidate_actions=tuple(tuple(row) for row in raw["candidate_actions"]),
            fit_cutoff_ns=raw["fit_cutoff_ns"],
            evaluation_start_ns=raw["evaluation_start_ns"],
            ridge_penalty=raw["ridge_penalty"],
            uncertainty_penalty_weight=raw["uncertainty_penalty_weight"],
            one_way_switch_cost=raw["one_way_switch_cost"],
            training_rows=raw["training_rows"],
            overlap_groups=raw["overlap_groups"],
            coefficients=tuple(tuple(row) for row in raw["coefficients"]),
            covariance_matrices=tuple(
                tuple(tuple(elem for elem in r) for r in m)
                for m in raw["covariance_matrices"]
            ),
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model payload violates the declared schema") from error
    return model


__all__ = [
    "ConservativeLinearDecisionValueModel",
    "ConservativeLinearDecisionValuePolicy",
    "LinearDecisionValueModel",
    "LinearDecisionValuePolicy",
    "LinearDecisionValueRuntimeContext",
    "compute_effective_posture_weights",
    "compute_pending_order_weights",
    "compute_posture_cost_adjustment",
    "encode_decision_value_state",
    "encode_decision_value_state_v2",
    "load_conservative_linear_decision_value",
    "load_linear_decision_value",
    "preserve_posture_aware_proposal",
    "serialize_conservative_linear_decision_value",
    "serialize_linear_decision_value",
]
