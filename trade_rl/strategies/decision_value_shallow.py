"""Synthetic-only shallow non-linear joint-action value head (B2).

B2 uses the same state encoder, candidate roster, runtime context and margin rule
as the linear head (B1). The only difference is a fixed random tanh hidden layer
whose weights are stored in the model, followed by one fixed Ridge head per
non-HOLD candidate. It exists to test representational power against B1, not to
add a tunable model family.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from fractions import Fraction
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.strategies.decision_value import (
    LinearDecisionValueRuntimeContext,
    _validated_hold_relative,
    compute_effective_posture_weights,
    compute_hold_relative_cost_adjustment,
    compute_posture_cost_adjustment,
    encode_decision_value_state,
    preserve_posture_aware_proposal,
    select_hold_relative_action,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_FLAT,
    ACTION_HOLD,
    ACTION_LONG,
    ACTION_SHORT,
    decode_portfolio_action_codes,
)

_SCHEMA = "synthetic_shallow_decision_value_v1"
_POSITIVE_EDGE = 1e-12
HIDDEN_UNITS = 8
HIDDEN_SEED = 20261029
RIDGE_PENALTY = 1e-3


def fixed_hidden_weights(state_width: int) -> tuple[tuple[float, ...], ...]:
    """Deterministic hidden layer over the non-bias state coordinates."""
    if type(state_width) is not int or state_width < 2:
        raise ValueError("state_width must include a bias and one feature")
    rng = np.random.default_rng(HIDDEN_SEED)
    raw = rng.standard_normal((HIDDEN_UNITS, state_width - 1)) / math.sqrt(
        state_width - 1
    )
    return tuple(tuple(float(v) for v in row) for row in raw)


def expand_state(
    state: np.ndarray, hidden: tuple[tuple[float, ...], ...]
) -> np.ndarray:
    """``[state, tanh(W state[1:])]``; keeps the linear part so B2 contains B1."""
    weights = np.asarray(hidden, dtype=np.float64)
    if weights.ndim != 2 or weights.shape[1] != state.shape[0] - 1:
        raise ValueError("hidden layer does not match the state width")
    return np.concatenate([state, np.tanh(weights @ state[1:])])


@dataclass(frozen=True, slots=True)
class ShallowDecisionValueModel:
    """Versioned immutable shallow action-value head with fixed action order."""

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
    hidden_weights: tuple[tuple[float, ...], ...]
    coefficients: tuple[tuple[float, ...], ...]
    input_means: tuple[float, ...] | None = None
    input_scales: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA or self.source_scope not in (
            "synthetic_only",
            "real_development",
        ):
            raise ValueError("only the versioned synthetic shallow model is supported")
        for roster in (self.symbols, self.feature_names, self.global_feature_names):
            if type(roster) is not tuple:
                raise TypeError("model rosters must be tuples")
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
        for name in (
            "fit_cutoff_ns",
            "evaluation_start_ns",
            "training_rows",
            "overlap_groups",
        ):
            if type(getattr(self, name)) is not int:
                raise ValueError("model cutoffs and counts must be native integers")
        if self.fit_cutoff_ns >= self.evaluation_start_ns:
            raise ValueError("evaluation must begin after the fit cutoff")
        if self.training_rows <= 0 or self.overlap_groups <= 0:
            raise ValueError("training counts must be positive")
        if not math.isfinite(self.ridge_penalty) or self.ridge_penalty <= 0:
            raise ValueError("ridge_penalty must be positive and finite")
        n = len(self.symbols)
        if (
            type(self.target_abs_weights) is not tuple
            or len(self.target_abs_weights) != n
        ):
            raise ValueError("target weights must match symbol roster")
        if any(not math.isfinite(w) or w < 0 for w in self.target_abs_weights):
            raise ValueError("target weights must be finite and non-negative")
        if type(self.candidate_actions) is not tuple or len(self.candidate_actions) < 2:
            raise ValueError("candidate actions require HOLD and another action")
        seen: set[tuple[int, ...]] = set()
        for action in self.candidate_actions:
            codes = decode_portfolio_action_codes(action, n_symbols=n)
            if codes in seen:
                raise ValueError("candidate actions must be unique")
            seen.add(codes)
        if self.candidate_actions[0] != (ACTION_HOLD,) * n:
            raise ValueError("first candidate must be joint HOLD")
        state_width = (
            1 + n * len(self.feature_names) + len(self.global_feature_names) + n
        )
        if self.hidden_weights != fixed_hidden_weights(state_width):
            raise ValueError("hidden layer differs from the fixed declared layer")
        width = state_width + HIDDEN_UNITS
        if len(self.coefficients) != len(self.candidate_actions) - 1:
            raise ValueError("coefficients must match candidate count")
        for row in self.coefficients:
            if type(row) is not tuple or len(row) != width:
                raise ValueError("coefficients have incorrect expanded dimension")
            if any(not math.isfinite(v) for v in row):
                raise ValueError("coefficients must be finite")
        if (self.input_means is None) != (self.input_scales is None):
            raise ValueError(
                "input_means and input_scales must both be None or both be tuples"
            )
        if self.input_means is not None and self.input_scales is not None:
            if (
                type(self.input_means) is not tuple
                or type(self.input_scales) is not tuple
            ):
                raise TypeError("input_means and input_scales must be tuples")
            if (
                len(self.input_means) != state_width - 1
                or len(self.input_scales) != state_width - 1
            ):
                raise ValueError(
                    "input_means and input_scales length must match state_width - 1"
                )
            for m in self.input_means:
                if isinstance(m, bool) or not math.isfinite(m):
                    raise ValueError("input_means elements must be finite floats")
            for s in self.input_scales:
                if isinstance(s, bool) or not math.isfinite(s) or s < 0.0:
                    raise ValueError(
                        "input_scales elements must be non-negative finite floats"
                    )


class ShallowDecisionValuePolicy:
    """Synthetic-only PortfolioPolicy; hard risk and execution stay downstream."""

    def __init__(
        self,
        model: ShallowDecisionValueModel,
        *,
        runtime_context: LinearDecisionValueRuntimeContext,
        allow_active_orders: bool = False,
        eval_one_way_cost: float = 0.0,
        train_one_way_cost: float = 0.0,
        hold_relative: bool = False,
        decision_margin_multiplier: float = 0.0,
    ) -> None:
        if not isinstance(model, ShallowDecisionValueModel):
            raise TypeError("model must be ShallowDecisionValueModel")
        if not isinstance(runtime_context, LinearDecisionValueRuntimeContext):
            raise TypeError("runtime_context must be a typed synthetic context")
        for name in (
            "source_scope",
            "feature_config_digest",
            "normalization_digest",
            "execution_policy_digest",
            "reference_policy_digest",
        ):
            if getattr(model, name) != getattr(runtime_context, name):
                raise ValueError(f"runtime {name} differs from fitted model")
        self.model = model
        self.allow_active_orders = allow_active_orders
        self.eval_one_way_cost = float(eval_one_way_cost)
        self.train_one_way_cost = float(train_one_way_cost)
        self.hold_relative = _validated_hold_relative(
            hold_relative, decision_margin_multiplier
        )
        self.decision_margin_multiplier = float(decision_margin_multiplier)
        self._coefs = [np.asarray(row, dtype=np.float64) for row in model.coefficients]
        self._hidden_w = np.asarray(model.hidden_weights, dtype=np.float64)
        self._input_means = (
            np.asarray(model.input_means, dtype=np.float64)
            if model.input_means is not None
            else None
        )
        self._input_scales = (
            np.asarray(model.input_scales, dtype=np.float64)
            if model.input_scales is not None
            else None
        )

    def _basis(self, snapshot: PortfolioDecisionSnapshot) -> np.ndarray:
        if snapshot.timestamp_ns < self.model.evaluation_start_ns:
            raise ValueError("policy decision precedes evaluation start")
        state = encode_decision_value_state(
            snapshot,
            symbols=self.model.symbols,
            feature_names=self.model.feature_names,
            global_feature_names=self.model.global_feature_names,
            allow_active_orders=self.allow_active_orders,
        )
        raw_x = state[1:]
        if self._input_means is not None and self._input_scales is not None:
            x_norm = (raw_x - self._input_means) * self._input_scales
        else:
            x_norm = raw_x
        hidden = np.tanh(self._hidden_w @ x_norm)
        return np.concatenate(([1.0], x_norm, hidden))

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
        basis = self._basis(snapshot)
        raw_val = float(basis @ self._coefs[k])
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
        raw_val = float(self._basis(snapshot) @ self._coefs[k])
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


def serialize_shallow_decision_value(
    model: ShallowDecisionValueModel,
) -> tuple[str, str]:
    if not isinstance(model, ShallowDecisionValueModel):
        raise TypeError("model must be ShallowDecisionValueModel")
    encoded = json.dumps(
        asdict(model),
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
        allow_nan=False,
    )
    return encoded, hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def load_shallow_decision_value(
    text: str, *, expected_sha256: str
) -> ShallowDecisionValueModel:
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
    expected_fields = {f.name for f in fields(ShallowDecisionValueModel)}
    base_fields = expected_fields - {"input_means", "input_scales"}
    if set(value) != expected_fields and set(value) != base_fields:
        raise ValueError("model payload fields differ from the declared schema")
    raw = cast(dict[str, Any], value)
    input_means = (
        tuple(float(v) for v in raw["input_means"])
        if raw.get("input_means") is not None
        else None
    )
    input_scales = (
        tuple(float(v) for v in raw["input_scales"])
        if raw.get("input_scales") is not None
        else None
    )
    try:
        return ShallowDecisionValueModel(
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
            candidate_actions=tuple(tuple(r) for r in raw["candidate_actions"]),
            fit_cutoff_ns=raw["fit_cutoff_ns"],
            evaluation_start_ns=raw["evaluation_start_ns"],
            ridge_penalty=raw["ridge_penalty"],
            training_rows=raw["training_rows"],
            overlap_groups=raw["overlap_groups"],
            hidden_weights=tuple(tuple(r) for r in raw["hidden_weights"]),
            coefficients=tuple(tuple(r) for r in raw["coefficients"]),
            input_means=input_means,
            input_scales=input_scales,
        )
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise ValueError("model payload violates the declared schema") from error


__all__ = [
    "ShallowDecisionValueModel",
    "ShallowDecisionValuePolicy",
    "expand_state",
    "fixed_hidden_weights",
    "load_shallow_decision_value",
    "serialize_shallow_decision_value",
]
