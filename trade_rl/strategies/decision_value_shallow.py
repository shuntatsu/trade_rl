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
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.strategies.decision_value import (
    LinearDecisionValueRuntimeContext,
    encode_decision_value_state,
)
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
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

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA or self.source_scope != "synthetic_only":
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


class ShallowDecisionValuePolicy:
    """Synthetic-only PortfolioPolicy; hard risk and execution stay downstream."""

    def __init__(
        self,
        model: ShallowDecisionValueModel,
        *,
        runtime_context: LinearDecisionValueRuntimeContext,
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

    def _scores(self, snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        if snapshot.timestamp_ns < self.model.evaluation_start_ns:
            raise ValueError("policy decision precedes evaluation start")
        state = encode_decision_value_state(
            snapshot,
            symbols=self.model.symbols,
            feature_names=self.model.feature_names,
            global_feature_names=self.model.global_feature_names,
        )
        expanded = expand_state(state, self.model.hidden_weights)
        values = [0.0]
        for row in self.model.coefficients:
            value = float(expanded @ np.asarray(row, dtype=np.float64))
            if not math.isfinite(value):
                raise ValueError("predicted decision value must be finite")
            values.append(value)
        return tuple(values)

    def decide_action(self, snapshot: PortfolioDecisionSnapshot) -> tuple[int, ...]:
        scores = self._scores(snapshot)
        best, best_edge = 0, _POSITIVE_EDGE
        for index in range(1, len(scores)):
            if scores[index] > best_edge:
                best, best_edge = index, scores[index]
        return self.model.candidate_actions[best]

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return portfolio_action_to_proposal(
            self.decide_action(snapshot), snapshot, self.model.target_abs_weights
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
    if set(value) != {f.name for f in fields(ShallowDecisionValueModel)}:
        raise ValueError("model payload fields differ from the declared schema")
    raw = cast(dict[str, Any], value)
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
