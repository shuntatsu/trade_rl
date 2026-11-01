"""Typed, metadata-only inputs for a fixed four-family comparison roster.

This post-freeze declaration binds already-pinned inputs and comparison windows.
It does not read or verify artifact bytes, establish model eligibility, fit, run,
select, authorize, or publish a study.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Literal, cast

from trade_rl._validation import require_sha256
from trade_rl.artifacts.canonical import canonical_json_bytes, to_json_value
from trade_rl.data.market_order_rules import MarketOrderProfile
from trade_rl.evaluation.experiments.errors import ContractViolationError
from trade_rl.evaluation.runs import FourFamilyPortfolioAttemptIdentity
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.strategies.rl.portfolio_training import SharedPortfolioPPOConfig

_SCHEMA = "four_family_post_freeze_input_manifest_v1"
_DIGEST_DOMAIN = b"trade_rl.four_family_post_freeze_input_manifest_v1\0"
_FAMILY_ROSTER = ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
_WINDOW_STAGES = ("training", "selection", "evaluation")
_ATTEMPT_STAGES = ("selection", "evaluation")
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_NANOSECONDS = 1_000_000_000


def _text(value: object, field: str) -> str:
    if not isinstance(value, str) or not value or value.strip() != value:
        raise ContractViolationError(f"{field} must be a non-empty canonical string")
    return value


def _sha256(value: object, field: str) -> str:
    if not isinstance(value, str):
        raise ContractViolationError(f"{field} must be a SHA-256 digest")
    try:
        return require_sha256(value, field=field)
    except ValueError as error:
        raise ContractViolationError(str(error)) from error


def _unique_texts(value: object, field: str) -> tuple[str, ...]:
    if type(value) is not tuple or not value:
        raise ContractViolationError(f"{field} must be a non-empty tuple")
    result = tuple(_text(item, f"{field}[]") for item in value)
    if len(set(result)) != len(result):
        raise ContractViolationError(f"{field} must contain unique values")
    return result


def _datetime_ns(value: datetime) -> int:
    normalized = value.astimezone(UTC)
    delta = normalized - _EPOCH
    return (
        delta.days * 86_400 + delta.seconds
    ) * _NANOSECONDS + delta.microseconds * 1_000


def _objective_projection(bound: BoundObjectiveClock) -> dict[str, object]:
    payload = bound.objective.payload()
    payload.pop("evaluation_start")
    payload.pop("evaluation_stop_exclusive")
    return payload


def _clock_projection(bound: BoundObjectiveClock) -> dict[str, object]:
    payload = bound.clock.payload()
    payload.pop("economic_horizon_seconds")
    return payload


def _profile_declaration(profile: MarketOrderProfile | None) -> object:
    if profile is None:
        return None
    return {
        "declaration": profile.canonical_payload(),
        "digest": profile.digest,
    }


def _financial_projection(
    *,
    initial_capital: float,
    target_abs_weights: tuple[float, ...],
    execution_cost: ExecutionCostConfig,
    risk_config: PreTradeRiskConfig,
    market_order_profile: MarketOrderProfile | None,
) -> bytes:
    return canonical_json_bytes(
        {
            "execution_cost": to_json_value(execution_cost),
            "initial_capital": initial_capital,
            "market_order_profile": _profile_declaration(market_order_profile),
            "risk_config": to_json_value(risk_config),
            "target_abs_weights": list(target_abs_weights),
        }
    )


def _identity_declaration(
    identity: FourFamilyPortfolioAttemptIdentity,
) -> dict[str, object]:
    value = to_json_value(identity)
    if not isinstance(value, dict):
        raise ContractViolationError("attempt identity must serialize as an object")
    declaration = cast(dict[str, object], value)
    if identity.market_order_profile is not None:
        declaration["market_order_profile"] = _profile_declaration(
            identity.market_order_profile
        )
    return declaration


@dataclass(frozen=True, slots=True)
class DatasetInputPin:
    """Declared Dataset identity and roster; no Dataset bytes are inspected."""

    artifact_sha256: str
    dataset_id: str
    schema_version: str
    symbols: tuple[str, ...]
    feature_names: tuple[str, ...]
    global_feature_names: tuple[str, ...]
    n_bars: int

    def __post_init__(self) -> None:
        _sha256(self.artifact_sha256, "dataset.artifact_sha256")
        _sha256(self.dataset_id, "dataset.dataset_id")
        _text(self.schema_version, "dataset.schema_version")
        object.__setattr__(
            self, "symbols", _unique_texts(self.symbols, "dataset.symbols")
        )
        object.__setattr__(
            self,
            "feature_names",
            _unique_texts(self.feature_names, "dataset.feature_names"),
        )
        object.__setattr__(
            self,
            "global_feature_names",
            _unique_texts(self.global_feature_names, "dataset.global_feature_names"),
        )
        if type(self.n_bars) is not int or self.n_bars < 2:
            raise ContractViolationError("dataset.n_bars must be an integer >= 2")


@dataclass(frozen=True, slots=True)
class WindowSpec:
    """One explicit chronological index window and its full objective/clock."""

    stage: Literal["training", "selection", "evaluation"]
    start_index: int
    stop_index: int
    bound: BoundObjectiveClock

    def __post_init__(self) -> None:
        if self.stage not in _WINDOW_STAGES:
            raise ContractViolationError("window stage is not supported")
        if (
            type(self.start_index) is not int
            or type(self.stop_index) is not int
            or self.start_index < 0
            or self.stop_index <= self.start_index
        ):
            raise ContractViolationError(
                "window indexes must satisfy 0 <= start < stop"
            )
        if type(self.bound) is not BoundObjectiveClock:
            raise ContractViolationError("window bound must be BoundObjectiveClock")
        elapsed = (
            self.bound.objective.evaluation_stop_exclusive
            - self.bound.objective.evaluation_start
        )
        seconds = elapsed.days * 86_400 + elapsed.seconds
        expected = (self.stop_index - self.start_index) * (
            self.bound.clock.decision_interval_seconds
        )
        if elapsed.microseconds or seconds != expected:
            raise ContractViolationError(
                "window duration must match its indexes and decision interval"
            )


@dataclass(frozen=True, slots=True)
class PPOTrainingDeclaration:
    """Typed PPO inputs, including its declared shared financial context."""

    training_context_sha256: str
    transform_digest: str
    config: SharedPortfolioPPOConfig
    target_abs_weights: tuple[float, ...]
    execution_cost: ExecutionCostConfig
    risk_config: PreTradeRiskConfig
    market_order_profile: MarketOrderProfile | None

    def __post_init__(self) -> None:
        _sha256(self.training_context_sha256, "ppo.training_context_sha256")
        _sha256(self.transform_digest, "ppo.transform_digest")
        if type(self.config) is not SharedPortfolioPPOConfig:
            raise ContractViolationError("ppo.config must be SharedPortfolioPPOConfig")
        if type(self.target_abs_weights) is not tuple or not self.target_abs_weights:
            raise ContractViolationError(
                "ppo.target_abs_weights must be a non-empty tuple"
            )
        if type(self.execution_cost) is not ExecutionCostConfig:
            raise ContractViolationError(
                "ppo.execution_cost must be ExecutionCostConfig"
            )
        if type(self.risk_config) is not PreTradeRiskConfig:
            raise ContractViolationError("ppo.risk_config must be PreTradeRiskConfig")
        if (
            self.market_order_profile is not None
            and type(self.market_order_profile) is not MarketOrderProfile
        ):
            raise ContractViolationError(
                "ppo.market_order_profile must be MarketOrderProfile or None"
            )


@dataclass(frozen=True, slots=True)
class ComparisonAttemptSpec:
    """One four-family child attempt; the family roster is never split."""

    stage: Literal["selection", "evaluation"]
    identity: FourFamilyPortfolioAttemptIdentity
    scenario: Literal["base"] = "base"

    def __post_init__(self) -> None:
        if self.stage not in _ATTEMPT_STAGES:
            raise ContractViolationError("comparison attempt stage is not supported")
        if self.scenario != "base":
            raise ContractViolationError(
                "only the base execution scenario is supported"
            )
        if type(self.identity) is not FourFamilyPortfolioAttemptIdentity:
            raise ContractViolationError(
                "comparison attempt identity must use the existing typed identity"
            )


@dataclass(frozen=True, slots=True)
class FourFamilyPostFreezeInputManifest:
    """Exact two-attempt metadata roster; not an executable study plan."""

    dataset: DatasetInputPin
    training_window: WindowSpec
    selection_window: WindowSpec
    evaluation_window: WindowSpec
    ppo_training: PPOTrainingDeclaration
    attempts: tuple[ComparisonAttemptSpec, ComparisonAttemptSpec]

    def __post_init__(self) -> None:
        if type(self.dataset) is not DatasetInputPin:
            raise ContractViolationError("dataset must be DatasetInputPin")
        windows = (self.training_window, self.selection_window, self.evaluation_window)
        if any(type(window) is not WindowSpec for window in windows):
            raise ContractViolationError("all windows must be WindowSpec values")
        if tuple(window.stage for window in windows) != _WINDOW_STAGES:
            raise ContractViolationError(
                "windows must be training, selection, evaluation"
            )
        if type(self.ppo_training) is not PPOTrainingDeclaration:
            raise ContractViolationError("ppo_training must be PPOTrainingDeclaration")
        if type(self.attempts) is not tuple or len(self.attempts) != 2:
            raise ContractViolationError(
                "exactly two four-family attempts are required"
            )
        if any(type(item) is not ComparisonAttemptSpec for item in self.attempts):
            raise ContractViolationError(
                "attempts must be ComparisonAttemptSpec values"
            )
        if tuple(item.stage for item in self.attempts) != _ATTEMPT_STAGES:
            raise ContractViolationError(
                "one selection and one evaluation attempt are required"
            )

        training, selection, evaluation = windows
        if any(window.stop_index >= self.dataset.n_bars for window in windows):
            raise ContractViolationError("window stop index must be inside the Dataset")
        if not (
            training.stop_index <= selection.start_index
            and selection.stop_index <= evaluation.start_index
        ):
            raise ContractViolationError(
                "index windows must be chronological and disjoint"
            )
        if not (
            training.bound.objective.evaluation_stop_exclusive
            <= selection.bound.objective.evaluation_start
            and selection.bound.objective.evaluation_stop_exclusive
            <= evaluation.bound.objective.evaluation_start
        ):
            raise ContractViolationError(
                "datetime windows must be chronological and disjoint"
            )

        bound_values = tuple(window.bound for window in windows)
        objective_values = tuple(_objective_projection(bound) for bound in bound_values)
        clock_values = tuple(_clock_projection(bound) for bound in bound_values)
        if any(value != objective_values[0] for value in objective_values[1:]):
            raise ContractViolationError(
                "objective contracts may differ only in their evaluation dates"
            )
        if any(value != clock_values[0] for value in clock_values[1:]):
            raise ContractViolationError(
                "financial clocks may differ only in economic horizon"
            )

        if (
            self.ppo_training.config.n_steps != training.bound.clock.rollout_steps
            or self.ppo_training.config.gae_lambda != training.bound.clock.gae_lambda
        ):
            raise ContractViolationError("PPO config differs from the training clock")

        identities = tuple(item.identity for item in self.attempts)
        if len({identity.attempt_id for identity in identities}) != len(identities):
            raise ContractViolationError("attempt IDs must be unique")
        for item, window in zip(self.attempts, (selection, evaluation), strict=True):
            identity = item.identity
            if (
                identity.start_index != window.start_index
                or identity.stop_index != window.stop_index
                or identity.bound.digest != window.bound.digest
            ):
                raise ContractViolationError(
                    "attempt identity must exactly bind its stage window"
                )
        identity_payloads = tuple(
            _identity_declaration(identity) for identity in identities
        )
        for payload in identity_payloads:
            payload.pop("attempt_id")
            payload.pop("start_index")
            payload.pop("stop_index")
            payload.pop("bound")
        if identity_payloads[0] != identity_payloads[1]:
            raise ContractViolationError(
                "selection and evaluation attempts must share all non-window inputs"
            )

        training_capital = training.bound.objective.capital.total_initial_capital
        declared_financials = _financial_projection(
            initial_capital=training_capital,
            target_abs_weights=self.ppo_training.target_abs_weights,
            execution_cost=self.ppo_training.execution_cost,
            risk_config=self.ppo_training.risk_config,
            market_order_profile=self.ppo_training.market_order_profile,
        )
        for identity in identities:
            if (
                _financial_projection(
                    initial_capital=identity.initial_capital,
                    target_abs_weights=identity.target_abs_weights,
                    execution_cost=identity.execution_cost,
                    risk_config=identity.risk_config,
                    market_order_profile=identity.market_order_profile,
                )
                != declared_financials
            ):
                raise ContractViolationError(
                    "attempt financial declaration differs from PPO training context"
                )

        first = identities[0]
        if any(
            identity.expected_dataset_artifact_sha256 != self.dataset.artifact_sha256
            or identity.expected_symbols != self.dataset.symbols
            for identity in identities
        ):
            raise ContractViolationError("attempt Dataset pin or symbol roster differs")
        ridge = first.expected_ridge_binding
        if (
            ridge.dataset_id != self.dataset.dataset_id
            or ridge.dataset_schema_version != self.dataset.schema_version
            or ridge.dataset_artifact_sha256 != self.dataset.artifact_sha256
            or ridge.feature_names != self.dataset.feature_names
            or any(
                index >= len(self.dataset.symbols) for index in ridge.fit_symbol_indices
            )
            or tuple(self.dataset.symbols[index] for index in ridge.fit_symbol_indices)
            != ridge.fit_symbol_names
        ):
            raise ContractViolationError("Ridge binding differs from the Dataset pin")
        if ridge.fit_cutoff_ns > _datetime_ns(
            selection.bound.objective.evaluation_start
        ):
            raise ContractViolationError("Ridge fit cutoff is after selection start")
        if (
            first.expected_training_context_sha256
            != self.ppo_training.training_context_sha256
        ):
            raise ContractViolationError(
                "PPO training context pin differs from attempt identity"
            )

    @property
    def family_roster(self) -> tuple[str, ...]:
        return _FAMILY_ROSTER

    @property
    def scenario_roster(self) -> tuple[str, ...]:
        return ("base",)

    @property
    def fit_cap(self) -> int:
        return 0

    @property
    def child_attempt_cap(self) -> int:
        return 2

    @property
    def family_result_row_cap(self) -> int:
        return self.child_attempt_cap * len(self.family_roster)

    def payload(self) -> dict[str, object]:
        return {
            "schema": _SCHEMA,
            "dataset": {
                "artifact_sha256": self.dataset.artifact_sha256,
                "dataset_id": self.dataset.dataset_id,
                "schema_version": self.dataset.schema_version,
                "symbols": list(self.dataset.symbols),
                "feature_names": list(self.dataset.feature_names),
                "global_feature_names": list(self.dataset.global_feature_names),
                "n_bars": self.dataset.n_bars,
            },
            "windows": {
                window.stage: {
                    "start_index": window.start_index,
                    "stop_index": window.stop_index,
                    "objective": window.bound.objective.payload(),
                    "financial_clock": window.bound.clock.payload(),
                }
                for window in (
                    self.training_window,
                    self.selection_window,
                    self.evaluation_window,
                )
            },
            "ppo_training": {
                "training_context_sha256": self.ppo_training.training_context_sha256,
                "transform_digest": self.ppo_training.transform_digest,
                "config": to_json_value(self.ppo_training.config),
                "target_abs_weights": list(self.ppo_training.target_abs_weights),
                "execution_cost": to_json_value(self.ppo_training.execution_cost),
                "risk_config": to_json_value(self.ppo_training.risk_config),
                "market_order_profile": _profile_declaration(
                    self.ppo_training.market_order_profile
                ),
            },
            "attempts": [
                {
                    "stage": attempt.stage,
                    "scenario": attempt.scenario,
                    "identity_declaration": _identity_declaration(attempt.identity),
                }
                for attempt in self.attempts
            ],
            "family_roster": list(self.family_roster),
            "scenario_roster": list(self.scenario_roster),
            "fit_cap": self.fit_cap,
            "child_attempt_cap": self.child_attempt_cap,
            "family_result_row_cap": self.family_result_row_cap,
            "scope": "declarations_only_no_load_fit_replay_selection_or_authorization",
        }

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self.payload())

    @property
    def digest(self) -> str:
        return hashlib.sha256(_DIGEST_DOMAIN + self.to_bytes()).hexdigest()


def _parse_strict_json(raw: bytes) -> object:
    def object_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise ContractViolationError("manifest contains a duplicate JSON key")
            result[key] = value
        return result

    def reject_constant(value: str) -> object:
        raise ContractViolationError(f"manifest contains non-finite value {value}")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ContractViolationError("manifest is not valid UTF-8 JSON") from error
    try:
        encoded = canonical_json_bytes(value)
    except (TypeError, ValueError) as error:
        raise ContractViolationError(
            "manifest contains a non-canonical value"
        ) from error
    if encoded != raw:
        raise ContractViolationError("manifest bytes are not canonical JSON")
    if type(value) is not dict or value.get("schema") != _SCHEMA:
        raise ContractViolationError("manifest schema is invalid")
    return value


def verify_bytes(
    raw: bytes,
    *,
    expected: FourFamilyPostFreezeInputManifest,
) -> FourFamilyPostFreezeInputManifest:
    """Verify exact bytes against a separately retained typed declaration.

    The returned object is the caller-supplied expected value. This does not
    reconstruct or trust a manifest supplied only by the serialized bytes.
    """
    if not isinstance(raw, bytes):
        raise TypeError("raw must be bytes")
    if type(expected) is not FourFamilyPostFreezeInputManifest:
        raise TypeError("expected must be FourFamilyPostFreezeInputManifest")
    _parse_strict_json(raw)
    if raw != expected.to_bytes():
        raise ContractViolationError(
            "manifest bytes differ from the expected declaration"
        )
    return expected


__all__ = [
    "ComparisonAttemptSpec",
    "DatasetInputPin",
    "FourFamilyPostFreezeInputManifest",
    "PPOTrainingDeclaration",
    "WindowSpec",
    "verify_bytes",
]
