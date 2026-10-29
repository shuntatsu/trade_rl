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
    PortfolioPendingOrderSnapshot,
    PortfolioTargetProposal,
)
from trade_rl.strategies.portfolio_policy_codec import (
    ACTION_HOLD,
    decode_portfolio_action_codes,
    portfolio_action_to_proposal,
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


class LinearDecisionValuePolicy:
    """Synthetic-only PortfolioPolicy; hard risk and native execution stay downstream."""

    def __init__(
        self,
        model: LinearDecisionValueModel,
        *,
        runtime_context: LinearDecisionValueRuntimeContext,
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
        if action == self.model.candidate_actions[0]:
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
        if action == self.model.candidate_actions[0]:
            return 0.0
        k = self.model.candidate_actions.index(action) - 1
        features = self._features(snapshot)
        mean_val = float(features @ self._coefs[k])
        se = self.predictive_std_for_action(snapshot, action)
        pending_weights = compute_pending_order_weights(snapshot)
        effective_weights = tuple(
            w_s + w_p
            for w_s, w_p in zip(snapshot.account.weights, pending_weights, strict=True)
        )
        proposal = portfolio_action_to_proposal(
            action, snapshot, self.model.target_abs_weights
        )
        extra_switch_turnover = max(
            0.0,
            sum(
                abs(t - e)
                for t, e in zip(proposal.target_weights, effective_weights, strict=True)
            )
            - sum(abs(t) for t in proposal.target_weights),
        )
        return (
            mean_val
            - self.model.uncertainty_penalty_weight * se
            - self.model.one_way_switch_cost * extra_switch_turnover
        )

    def decide_action(self, snapshot: PortfolioDecisionSnapshot) -> tuple[int, ...]:
        best = 0
        best_edge = _POSITIVE_EDGE
        for index in range(1, len(self.model.candidate_actions)):
            action = self.model.candidate_actions[index]
            score = self.conservative_value_for_action(snapshot, action)
            if score > best_edge:
                best = index
                best_edge = score
        return self.model.candidate_actions[best]

    def decide_portfolio(
        self, snapshot: PortfolioDecisionSnapshot
    ) -> PortfolioTargetProposal:
        action = self.decide_action(snapshot)
        proposal = portfolio_action_to_proposal(
            action, snapshot, self.model.target_abs_weights
        )
        if not snapshot.active_orders:
            return proposal
        pending_weights = compute_pending_order_weights(snapshot)
        effective_weights = tuple(
            w_s + w_p
            for w_s, w_p in zip(snapshot.account.weights, pending_weights, strict=True)
        )
        target_weights = list(proposal.target_weights)
        hold_symbols = set(proposal.hold_current_quantity_symbols)
        for i in range(len(self.model.symbols)):
            if (
                i not in hold_symbols
                and abs(proposal.target_weights[i] - effective_weights[i]) <= 1e-6
            ):
                hold_symbols.add(i)
                target_weights[i] = snapshot.account.weights[i]
        return PortfolioTargetProposal(
            target_weights=tuple(target_weights),
            hold_current_quantity_symbols=tuple(sorted(hold_symbols)),
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
    "compute_pending_order_weights",
    "encode_decision_value_state",
    "encode_decision_value_state_v2",
    "load_conservative_linear_decision_value",
    "load_linear_decision_value",
    "serialize_conservative_linear_decision_value",
    "serialize_linear_decision_value",
]
