"""Synthetic-only, fixed linear joint-action value heads.

Inference sees only the current portfolio snapshot; no market replay, cost
book, or evaluation dataset is accessible from the strategy package.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields
from numbers import Real
from typing import Any, cast

import numpy as np

from trade_rl._validation import require_sha256, require_unique_non_empty
from trade_rl.strategies.portfolio import (
    PortfolioDecisionSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
)

_SCHEMA = "synthetic_linear_decision_value_v1"
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


def encode_decision_value_state(
    observation: PortfolioDecisionSnapshot,
    *,
    symbols: tuple[str, ...],
    feature_names: tuple[str, ...],
    global_feature_names: tuple[str, ...],
) -> np.ndarray:
    """Encode only decision-time features and actual portfolio weights.

    This v1 head has no orders/pending-state representation. Reject such
    snapshots rather than silently extrapolating an unsupported state.
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
    if observation.active_orders:
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
        if self.schema_version != _SCHEMA or self.source_scope != "synthetic_only":
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


class LinearDecisionValuePolicy:
    """Synthetic-only PortfolioPolicy; hard risk and native execution stay downstream."""

    def __init__(self, model: LinearDecisionValueModel) -> None:
        if not isinstance(model, LinearDecisionValueModel):
            raise TypeError("model must be LinearDecisionValueModel")
        self.model = model

    def _scores(self, snapshot: PortfolioDecisionSnapshot) -> tuple[float, ...]:
        if snapshot.timestamp_ns < self.model.evaluation_start_ns:
            raise ValueError("policy decision precedes evaluation start")
        features = encode_decision_value_state(
            snapshot,
            symbols=self.model.symbols,
            feature_names=self.model.feature_names,
            global_feature_names=self.model.global_feature_names,
        )
        values = [0.0]
        for row in self.model.coefficients:
            value = float(features @ np.asarray(row, dtype=np.float64))
            if not math.isfinite(value):
                raise ValueError("predicted decision value must be finite")
            values.append(value)
        return tuple(values)

    def value_for_action(
        self, snapshot: PortfolioDecisionSnapshot, action: tuple[int, ...]
    ) -> float:
        if action not in self.model.candidate_actions:
            raise ValueError("action is not a declared candidate")
        return self._scores(snapshot)[self.model.candidate_actions.index(action)]

    def decide_action(self, snapshot: PortfolioDecisionSnapshot) -> tuple[int, ...]:
        scores = self._scores(snapshot)
        best = 0
        best_edge = _POSITIVE_EDGE
        for index in range(1, len(scores)):
            if scores[index] > best_edge:
                best = index
                best_edge = scores[index]
        return self.model.candidate_actions[best]

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        return portfolio_action_to_proposal(
            self.decide_action(snapshot), snapshot, self.model.target_abs_weights
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


__all__ = [
    "LinearDecisionValueModel",
    "LinearDecisionValuePolicy",
    "encode_decision_value_state",
    "serialize_linear_decision_value",
    "load_linear_decision_value",
]
