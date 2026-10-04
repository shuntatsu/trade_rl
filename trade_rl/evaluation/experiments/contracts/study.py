"""Immutable Study-level contracts for controlled development research."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import ClassVar, cast

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.experiments.contracts._common import (
    contract_aware_datetime,
    contract_non_negative_int,
    contract_non_negative_int_tuple,
    contract_positive_int,
    contract_sha256,
    contract_sha256_tuple,
    contract_text,
    contract_unique_enum_tuple,
    contract_unique_texts,
)
from trade_rl.evaluation.experiments.contracts.experiment import ControlledFactor
from trade_rl.evaluation.experiments.contracts.research import StudyResearchContext
from trade_rl.evaluation.experiments.contracts.run import ResolvedRunConfig
from trade_rl.evaluation.experiments.errors import ContractViolationError
from trade_rl.risk import PreTradeRiskConfig

CANDIDATE_STRATEGY_NAMES = (
    "trend",
    "mean_reversion",
    "ridge24",
    "lightgbm24",
    "ppo",
)
CONTROL_STRATEGY_NAMES = (
    "cash",
    "constant_long",
    "constant_short",
)
_STUDY_FREEZE_SCHEMA = "controlled_study_freeze_v1"


def _canonical_ns_timestamp(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        )
    try:
        instant = np.datetime64(value, "ns")
    except (TypeError, ValueError) as error:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        ) from error
    if np.isnat(instant):
        raise ContractViolationError(f"{field} must not be NaT")
    canonical = np.datetime_as_string(instant, unit="ns")
    if canonical != value:
        raise ContractViolationError(
            f"{field} must use canonical nanosecond timestamp formatting"
        )
    return value


class StudyOutcome(StrEnum):
    WINNER = "WINNER"
    NO_WINNER = "NO_WINNER"


class StudyProtocol(StrEnum):
    PPO_HOLDING_DURATION = "ppo_holding_duration_v1"
    PPO_SHARED_CASH_HOLDING_DURATION = "ppo_shared_cash_holding_duration_v2"


PPO_HOLDING_DURATION_HORIZONS = (72, 168, 336, 504)
PPO_HOLDING_DURATION_SEED_COUNT = 5
PPO_HOLDING_DURATION_MAX_DRAWDOWN = 0.20
PPO_HOLDING_DURATION_RISK_CONFIG = PreTradeRiskConfig(
    max_gross=0.5,
    max_abs_weight=0.1,
    max_turnover=None,
    drawdown_start=0.10,
    drawdown_stop=PPO_HOLDING_DURATION_MAX_DRAWDOWN,
)
PPO_HOLDING_DURATION_SELECTION_RULE = (
    "PPO holding-duration preregistered selection rule: compare a freshly trained "
    "H=0 PPO with 72, 168, 336, and 504 completed one-hour bars on the same "
    "Dataset, evaluation window, costs, initial capital, and risk settings. "
    "Primary score: median across the five registered seeds of the equal-weight "
    "mean across symbols of each independent account's after-cost total return. "
    "An arm is eligible only if every H=0 and candidate seed-symbol account "
    "completes terminal settlement flat with no active order remainder, every "
    "account's realized maximum drawdown is at most 20%, and the median across "
    "seeds of the equal-weight mean across symbols of paired total-return "
    "differences versus H=0 is positive. Select the eligible arm with the "
    "highest primary score; break exact ties toward the shorter hold. If no arm "
    "is eligible, record NO_WINNER. This four-arm development screen is not a "
    "profitability claim; do not report unadjusted p-values, and require a "
    "separate one-shot sealed unused-future evaluation for any profitability claim."
)
PPO_SHARED_CASH_HOLDING_DURATION_SELECTION_RULE = (
    "PPO shared-cash holding-duration preregistered selection rule: compare a "
    "freshly trained H=0 PPO with 72, 168, 336, and 504 completed one-hour "
    "bars on the same Dataset, evaluation window, costs, initial capital, and "
    "risk settings. Evaluate each seed as one shared-cash portfolio across the "
    "full registered symbol roster; do not average independent per-symbol "
    "accounts for selection. Primary score: median across the five registered "
    "seeds of the combined portfolio's after-cost total return. An arm is "
    "eligible only if every H=0 and candidate portfolio completes terminal "
    "settlement flat with no active order remainder, each combined portfolio's "
    "realized maximum drawdown is at most 20%, and median paired total-return "
    "improvement versus H=0 is positive. Select the eligible arm with the "
    "highest primary score; break exact ties toward the shorter hold. If no arm "
    "is eligible, record NO_WINNER. PPO is still trained on single-symbol "
    "episodes; shared-cash evaluation does not establish joint-portfolio "
    "training. This four-arm development screen is not a profitability claim; "
    "do not report unadjusted p-values, and require a separate one-shot sealed "
    "unused-future evaluation for any profitability claim."
)


@dataclass(frozen=True, slots=True)
class StudyPlan:
    """Frozen development Study contract and adaptive-iteration budget."""

    FIXED_RESOLVED_FIELDS: ClassVar[tuple[str, ...]] = (
        "fit_cutoff",
        "evaluation_start",
        "evaluation_stop_exclusive",
        "initial_capital",
        "execution_overlay",
        "schema_version",
        "ppo_observation_schema",
        "ppo_global_feature_names",
        "ppo_settle_terminal_position",
        "pretrade_risk_config",
    )

    STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
        *CONTROL_STRATEGY_NAMES,
        *CANDIDATE_STRATEGY_NAMES,
    )
    PPO_SEED_INVARIANT_STRATEGY_NAMES: ClassVar[tuple[str, ...]] = (
        CONTROL_STRATEGY_NAMES
        + tuple(name for name in CANDIDATE_STRATEGY_NAMES if name != "ppo")
    )

    research_question: str
    dataset_id: str
    dataset_artifact_schema: str
    dataset_artifact_digest: str
    symbols: tuple[str, ...]
    baseline_config: ResolvedRunConfig
    ppo_seeds: tuple[int, ...]
    allowed_factors: tuple[object, ...]
    max_experiments: int
    n_bootstrap: int
    bootstrap_seed: int
    implementation_digest: str
    runtime_environment_digest: str
    final_evaluation_start: str | None = None
    final_evaluation_stop_exclusive: str | None = None
    research_context: StudyResearchContext | None = None
    schema_version: str = "controlled_study_plan_v1"
    protocol: StudyProtocol | None = None

    def __post_init__(self) -> None:
        research_question = contract_text(
            self.research_question,
            field="research_question",
        )
        dataset_id = contract_sha256(self.dataset_id, field="dataset_id")
        dataset_artifact_schema = contract_text(
            self.dataset_artifact_schema,
            field="dataset_artifact_schema",
        )
        dataset_artifact_digest = contract_sha256(
            self.dataset_artifact_digest,
            field="dataset_artifact_digest",
        )
        symbols = contract_unique_texts(self.symbols, field="symbols")
        if not isinstance(self.baseline_config, ResolvedRunConfig):
            raise ContractViolationError("baseline_config must be a ResolvedRunConfig")
        ppo_seeds = contract_non_negative_int_tuple(
            self.ppo_seeds,
            field="ppo_seeds",
            minimum_items=2,
        )
        if self.baseline_config.ppo_seed != ppo_seeds[0]:
            raise ContractViolationError(
                "baseline_config ppo_seed must equal the first registered ppo_seeds value"
            )
        allowed = contract_unique_enum_tuple(
            self.allowed_factors,
            field="allowed_factors",
            expected_type=ControlledFactor,
        )
        max_experiments = contract_positive_int(
            self.max_experiments,
            field="max_experiments",
            maximum=9_999,
        )
        n_bootstrap = contract_positive_int(
            self.n_bootstrap,
            field="n_bootstrap",
        )
        bootstrap_seed = contract_non_negative_int(
            self.bootstrap_seed,
            field="bootstrap_seed",
        )
        implementation_digest = contract_sha256(
            self.implementation_digest,
            field="implementation_digest",
        )
        runtime_environment_digest = contract_sha256(
            self.runtime_environment_digest,
            field="runtime_environment_digest",
        )
        schema_version = contract_text(self.schema_version, field="schema_version")
        final_start = self.final_evaluation_start
        final_stop = self.final_evaluation_stop_exclusive
        research_context = self.research_context
        protocol = self.protocol

        if schema_version == "controlled_study_plan_v1":
            if final_start is not None or final_stop is not None:
                raise ContractViolationError(
                    "controlled_study_plan_v1 forbids final evaluation fields"
                )
            if research_context is not None:
                raise ContractViolationError(
                    "controlled_study_plan_v1 forbids research_context"
                )
        elif schema_version == "controlled_study_plan_v2":
            if research_context is not None:
                raise ContractViolationError(
                    "controlled_study_plan_v2 forbids research_context"
                )
            if final_start is None or final_stop is None:
                raise ContractViolationError(
                    "controlled_study_plan_v2 requires both final evaluation fields"
                )
        elif schema_version == "controlled_study_plan_v3":
            if not isinstance(research_context, StudyResearchContext):
                raise ContractViolationError(
                    "controlled_study_plan_v3 requires research_context"
                )
            if (final_start is None) != (final_stop is None):
                raise ContractViolationError(
                    "controlled_study_plan_v3 requires both final evaluation fields or neither"
                )
        elif schema_version == "controlled_study_plan_v4":
            if final_start is not None or final_stop is not None:
                raise ContractViolationError(
                    "controlled_study_plan_v4 forbids final evaluation fields"
                )
            if research_context is not None:
                raise ContractViolationError(
                    "controlled_study_plan_v4 forbids research_context"
                )
            if protocol is not StudyProtocol.PPO_HOLDING_DURATION:
                raise ContractViolationError(
                    "controlled_study_plan_v4 requires a supported Study protocol"
                )
        elif schema_version == "controlled_study_plan_v5":
            if final_start is None or final_stop is None:
                raise ContractViolationError(
                    "controlled_study_plan_v5 requires both final evaluation fields"
                )
            if not isinstance(research_context, StudyResearchContext):
                raise ContractViolationError(
                    "controlled_study_plan_v5 requires research_context"
                )
            if protocol is not StudyProtocol.PPO_HOLDING_DURATION:
                raise ContractViolationError(
                    "controlled_study_plan_v5 requires a supported Study protocol"
                )
        elif schema_version == "controlled_study_plan_v6":
            if final_start is None or final_stop is None:
                raise ContractViolationError(
                    "controlled_study_plan_v6 requires both final evaluation fields"
                )
            if not isinstance(research_context, StudyResearchContext):
                raise ContractViolationError(
                    "controlled_study_plan_v6 requires research_context"
                )
            if protocol is not StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION:
                raise ContractViolationError(
                    "controlled_study_plan_v6 requires the shared-cash PPO protocol"
                )
        else:
            raise ContractViolationError("unsupported StudyPlan schema_version")

        if (
            schema_version
            not in {
                "controlled_study_plan_v4",
                "controlled_study_plan_v5",
                "controlled_study_plan_v6",
            }
            and protocol is not None
        ):
            raise ContractViolationError(
                "Study protocol requires a versioned PPO holding StudyPlan"
            )
        if ControlledFactor.PPO_MINIMUM_HOLD in allowed and protocol not in {
            StudyProtocol.PPO_HOLDING_DURATION,
            StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION,
        }:
            raise ContractViolationError(
                "PPO_MINIMUM_HOLD requires a versioned Study protocol"
            )
        if protocol in {
            StudyProtocol.PPO_HOLDING_DURATION,
            StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION,
        }:
            if (
                allowed != (ControlledFactor.PPO_MINIMUM_HOLD,)
                or max_experiments != len(PPO_HOLDING_DURATION_HORIZONS)
                or len(ppo_seeds) != PPO_HOLDING_DURATION_SEED_COUNT
                or self.baseline_config.ppo_minimum_hold_bars != 0
                or self.baseline_config.ppo_observation_schema != "ppo_observation_v3"
                or not self.baseline_config.ppo_settle_terminal_position
                or self.baseline_config.pretrade_risk_config
                != PPO_HOLDING_DURATION_RISK_CONFIG
            ):
                raise ContractViolationError(
                    "StudyPlan violates the PPO holding-duration protocol"
                )
            selection_rule = (
                PPO_SHARED_CASH_HOLDING_DURATION_SELECTION_RULE
                if protocol is StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION
                else PPO_HOLDING_DURATION_SELECTION_RULE
            )
            if not research_question.endswith(selection_rule):
                raise ContractViolationError(
                    "PPO holding-duration StudyPlan must preregister the full selection rule"
                )

        if final_start is not None and final_stop is not None:
            final_start = _canonical_ns_timestamp(
                final_start,
                field="final_evaluation_start",
            )
            final_stop = _canonical_ns_timestamp(
                final_stop,
                field="final_evaluation_stop_exclusive",
            )
            development_stop = np.datetime64(
                self.baseline_config.evaluation_stop_exclusive,
                "ns",
            )
            if np.datetime64(final_start, "ns") < development_stop:
                raise ContractViolationError(
                    "final evaluation start must not precede development evaluation stop"
                )
            if np.datetime64(final_stop, "ns") <= np.datetime64(final_start, "ns"):
                raise ContractViolationError(
                    "final evaluation stop must be strictly later than final evaluation start"
                )
            if research_context is not None:
                final_start_ns = np.datetime64(final_start, "ns")
                for evidence in research_context.consumed_evidence:
                    evidence_stop = np.datetime64(
                        evidence.development_stop_exclusive,
                        "ns",
                    )
                    if evidence_stop > final_start_ns:
                        raise ContractViolationError(
                            "final evaluation start must not precede consumed development evidence"
                        )

        object.__setattr__(self, "research_question", research_question)
        object.__setattr__(self, "dataset_id", dataset_id)
        object.__setattr__(self, "dataset_artifact_schema", dataset_artifact_schema)
        object.__setattr__(self, "dataset_artifact_digest", dataset_artifact_digest)
        object.__setattr__(self, "symbols", symbols)
        object.__setattr__(self, "ppo_seeds", ppo_seeds)
        object.__setattr__(
            self,
            "allowed_factors",
            cast(tuple[ControlledFactor, ...], allowed),
        )
        object.__setattr__(self, "max_experiments", max_experiments)
        object.__setattr__(self, "n_bootstrap", n_bootstrap)
        object.__setattr__(self, "bootstrap_seed", bootstrap_seed)
        object.__setattr__(self, "implementation_digest", implementation_digest)
        object.__setattr__(
            self, "runtime_environment_digest", runtime_environment_digest
        )
        object.__setattr__(self, "final_evaluation_start", final_start)
        object.__setattr__(self, "final_evaluation_stop_exclusive", final_stop)
        object.__setattr__(self, "research_context", research_context)
        object.__setattr__(self, "protocol", protocol)
        object.__setattr__(self, "schema_version", schema_version)

    @property
    def candidate_strategy_names(self) -> tuple[str, ...]:
        return CANDIDATE_STRATEGY_NAMES

    @property
    def control_strategy_names(self) -> tuple[str, ...]:
        return CONTROL_STRATEGY_NAMES

    @property
    def is_ppo_holding_duration_study(self) -> bool:
        return self.protocol in {
            StudyProtocol.PPO_HOLDING_DURATION,
            StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION,
        }

    @property
    def is_ppo_shared_cash_holding_duration_study(self) -> bool:
        return self.protocol is StudyProtocol.PPO_SHARED_CASH_HOLDING_DURATION

    def to_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "schema_version": self.schema_version,
            "research_question": self.research_question,
            "dataset_id": self.dataset_id,
            "dataset_artifact_schema": self.dataset_artifact_schema,
            "dataset_artifact_digest": self.dataset_artifact_digest,
            "symbols": list(self.symbols),
            "baseline_config": self.baseline_config.to_payload(),
            "ppo_seeds": list(self.ppo_seeds),
            "allowed_factors": [
                cast(StrEnum, factor).value for factor in self.allowed_factors
            ],
            "max_experiments": self.max_experiments,
            "n_bootstrap": self.n_bootstrap,
            "bootstrap_seed": self.bootstrap_seed,
            "implementation_digest": self.implementation_digest,
            "runtime_environment_digest": self.runtime_environment_digest,
            "candidate_strategy_names": list(CANDIDATE_STRATEGY_NAMES),
            "control_strategy_names": list(CONTROL_STRATEGY_NAMES),
        }
        if self.schema_version == "controlled_study_plan_v3":
            assert self.research_context is not None
            payload["research_context"] = self.research_context.to_payload()
        if self.schema_version in {
            "controlled_study_plan_v4",
            "controlled_study_plan_v5",
            "controlled_study_plan_v6",
        }:
            assert self.protocol is not None
            payload["protocol"] = self.protocol.value
            if self.research_context is not None:
                payload["research_context"] = self.research_context.to_payload()
        if self.final_evaluation_start is not None:
            payload["final_evaluation_start"] = self.final_evaluation_start
            payload["final_evaluation_stop_exclusive"] = (
                self.final_evaluation_stop_exclusive
            )
        return payload

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


@dataclass(frozen=True, slots=True)
class StudyFreeze:
    """Terminal development-only Study outcome."""

    study_digest: str
    experiment_decision_digests: tuple[str, ...]
    outcome: StudyOutcome
    selected_evidence_digest: str | None
    selected_strategy: str | None
    rationale: str
    frozen_by: str
    frozen_at: datetime
    schema_version: str = _STUDY_FREEZE_SCHEMA

    def __post_init__(self) -> None:
        study_digest = contract_sha256(self.study_digest, field="study_digest")
        decision_digests = contract_sha256_tuple(
            self.experiment_decision_digests,
            field="experiment_decision_digests",
        )
        if not isinstance(self.outcome, StudyOutcome):
            raise ContractViolationError("outcome is unsupported")
        rationale = contract_text(self.rationale, field="rationale")
        frozen_by = contract_text(self.frozen_by, field="frozen_by")
        frozen_at = contract_aware_datetime(self.frozen_at, field="frozen_at")
        schema_version = contract_text(self.schema_version, field="schema_version")
        if schema_version != _STUDY_FREEZE_SCHEMA:
            raise ContractViolationError("unsupported StudyFreeze schema_version")

        selected_evidence_digest = self.selected_evidence_digest
        selected_strategy = self.selected_strategy
        if self.outcome is StudyOutcome.WINNER:
            if selected_evidence_digest is None:
                raise ContractViolationError("WINNER requires selected_evidence_digest")
            selected_evidence_digest = contract_sha256(
                selected_evidence_digest,
                field="selected_evidence_digest",
            )
            if selected_strategy not in CANDIDATE_STRATEGY_NAMES:
                raise ContractViolationError(
                    "WINNER selected_strategy must be a candidate strategy"
                )
        else:
            if selected_evidence_digest is not None or selected_strategy is not None:
                raise ContractViolationError(
                    "NO_WINNER forbids selected evidence and strategy fields"
                )

        object.__setattr__(self, "study_digest", study_digest)
        object.__setattr__(self, "experiment_decision_digests", decision_digests)
        object.__setattr__(self, "selected_evidence_digest", selected_evidence_digest)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "frozen_by", frozen_by)
        object.__setattr__(self, "frozen_at", frozen_at)
        object.__setattr__(self, "schema_version", schema_version)

    def to_payload(self) -> dict[str, object]:
        return {
            "schema_version": self.schema_version,
            "study_digest": self.study_digest,
            "experiment_decision_digests": list(self.experiment_decision_digests),
            "outcome": self.outcome.value,
            "selected_evidence_digest": self.selected_evidence_digest,
            "selected_strategy": self.selected_strategy,
            "rationale": self.rationale,
            "frozen_by": self.frozen_by,
            "frozen_at": self.frozen_at,
        }

    @property
    def digest(self) -> str:
        return content_digest(self.to_payload())


__all__ = [
    "CANDIDATE_STRATEGY_NAMES",
    "CONTROL_STRATEGY_NAMES",
    "PPO_HOLDING_DURATION_HORIZONS",
    "PPO_HOLDING_DURATION_MAX_DRAWDOWN",
    "PPO_HOLDING_DURATION_RISK_CONFIG",
    "PPO_HOLDING_DURATION_SEED_COUNT",
    "PPO_HOLDING_DURATION_SELECTION_RULE",
    "PPO_SHARED_CASH_HOLDING_DURATION_SELECTION_RULE",
    "StudyFreeze",
    "StudyOutcome",
    "StudyPlan",
]
