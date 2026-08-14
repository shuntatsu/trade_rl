"""Episode-aligned Oracle behavior-cloning sampling and audit helpers."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.learning.episode_behavior_cloning import BehaviorCloningSplit
from trade_rl.learning.episode_oracle_teacher import (
    EpisodeOracleBatch,
    OracleEpisodeContract,
    OracleEpisodeSamplingConfig,
)
from trade_rl.learning.evaluation import (
    ActionPathCollapseEvidence,
    PathPerformanceMetrics,
    deterministic_bootstrap_lower_bound,
    deterministic_bootstrap_upper_bound,
)
from trade_rl.learning.rollout_evaluation import (
    ActionPathEvaluation,
    evaluate_action_path,
)

EPISODE_ORACLE_BC_EVALUATION_SCHEMA = "episode_oracle_bc_evaluation_v3"

_ACTION_QUANTILES = np.asarray(
    (0.0, 0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0),
    dtype=np.float64,
)
_ACTION_HISTOGRAM_EDGES = np.asarray(
    (-1.0, -0.75, -0.5, -0.25, -0.05, 0.05, 0.25, 0.5, 0.75, 1.0),
    dtype=np.float64,
)


def _action_comparison_diagnostics(
    teacher_actions: np.ndarray,
    policy_actions: np.ndarray,
    *,
    initial_weights: np.ndarray,
    tolerance: float,
) -> dict[str, object]:
    teacher = np.asarray(teacher_actions, dtype=np.float64)
    policy = np.asarray(policy_actions, dtype=np.float64)
    initial = np.asarray(initial_weights, dtype=np.float64)
    if (
        teacher.ndim != 2
        or policy.shape != teacher.shape
        or initial.shape != (teacher.shape[1],)
        or not np.isfinite(teacher).all()
        or not np.isfinite(policy).all()
        or not np.isfinite(initial).all()
    ):
        raise ValueError("BC action diagnostics require aligned finite action paths")

    teacher_previous = np.vstack((initial[None, :], teacher[:-1]))
    policy_previous = np.vstack((initial[None, :], policy[:-1]))
    teacher_delta = np.abs(teacher - teacher_previous)
    policy_delta = np.abs(policy - policy_previous)
    teacher_flat = teacher.ravel()
    policy_flat = policy.ravel()
    direction_agreement = (
        (np.abs(teacher_flat) <= tolerance) & (np.abs(policy_flat) <= tolerance)
    ) | (teacher_flat * policy_flat > 0.0)

    def summary(
        prefix: str,
        values: np.ndarray,
        delta: np.ndarray,
    ) -> dict[str, object]:
        flat = values.ravel()
        histogram, _ = np.histogram(flat, bins=_ACTION_HISTOGRAM_EDGES)
        return {
            f"{prefix}_absolute_mean": float(np.mean(np.abs(flat))),
            f"{prefix}_absolute_target_delta_mean": float(delta.mean()),
            f"{prefix}_absolute_target_delta_total": float(delta.sum()),
            f"{prefix}_change_count": int(np.count_nonzero(delta > tolerance)),
            f"{prefix}_histogram_counts": histogram.astype(int).tolist(),
            f"{prefix}_mean": float(flat.mean()),
            f"{prefix}_near_zero_rate": float(np.mean(np.abs(flat) <= tolerance)),
            f"{prefix}_negative_rate": float(np.mean(flat < -tolerance)),
            f"{prefix}_positive_rate": float(np.mean(flat > tolerance)),
            f"{prefix}_quantiles": np.quantile(flat, _ACTION_QUANTILES).tolist(),
            f"{prefix}_saturation_rate": float(np.mean(np.abs(flat) >= 0.95)),
            f"{prefix}_sign_flip_count": int(
                np.count_nonzero(values[1:] * values[:-1] < 0.0)
            ),
            f"{prefix}_std": float(flat.std()),
        }

    teacher_std = float(teacher_flat.std())
    policy_std = float(policy_flat.std())
    correlation = (
        None
        if teacher_std == 0.0 or policy_std == 0.0
        else float(np.corrcoef(teacher_flat, policy_flat)[0, 1])
    )
    return {
        "action_count": int(teacher_flat.size),
        "action_tolerance": float(tolerance),
        "direction_agreement_rate": float(np.mean(direction_agreement)),
        "histogram_edges": _ACTION_HISTOGRAM_EDGES.tolist(),
        "mean_signed_error": float(np.mean(policy_flat - teacher_flat)),
        "pearson_correlation": correlation,
        "quantile_probabilities": _ACTION_QUANTILES.tolist(),
        **summary("teacher", teacher, teacher_delta),
        **summary("policy", policy, policy_delta),
    }


def oracle_episode_sampling_config(
    environment: Any,
    *,
    train_range: tuple[int, int],
    seed: int,
) -> OracleEpisodeSamplingConfig:
    """Derive the Oracle episode distribution from the maintained PPO environment."""

    start, stop = train_range
    if (
        isinstance(start, bool)
        or isinstance(stop, bool)
        or not isinstance(start, int)
        or not isinstance(stop, int)
        or start < 0
        or stop <= start + 1
    ):
        raise ValueError("Oracle episode train range is invalid")
    decision_bars = int(getattr(environment, "decision_bars", 0))
    if decision_bars != 1:
        raise ValueError(
            "episode-aligned Oracle currently requires one bar per decision"
        )
    episode_bars = int(getattr(environment, "episode_bars", 0))
    if episode_bars <= 0:
        raise ValueError("Oracle episode horizon must be positive")
    train_decisions = stop - start - 1
    if train_decisions < episode_bars:
        raise ValueError("training view does not contain one complete Oracle episode")
    config = getattr(environment, "config", None)
    modes = tuple(getattr(config, "initial_state_modes", ()))
    if not modes or any(mode not in {"cash", "baseline"} for mode in modes):
        raise ValueError(
            "episode-aligned Oracle supports only cash and baseline reset modes"
        )
    return OracleEpisodeSamplingConfig(
        episode_bars=episode_bars,
        episode_count=math.ceil(train_decisions / episode_bars),
        initial_state_modes=modes,
        seed=seed,
    )


def resolve_episode_initial_weights(
    environment: Any,
    mode: str,
    start: int,
) -> np.ndarray:
    """Resolve the exact deterministic reset weights used by the environment."""

    resolver = getattr(environment, "initial_weights_for_reset", None)
    if callable(resolver):
        raw = resolver(mode, start)
    else:
        private_resolver = getattr(environment, "_initial_weights", None)
        if not callable(private_resolver):
            raise TypeError(
                "training environment cannot resolve initial portfolio weights"
            )
        if mode not in {"cash", "baseline"}:
            raise ValueError("unsupported episode Oracle initial state mode")
        raw = private_resolver(mode, start)
    weights = np.asarray(raw, dtype=np.float64).copy(order="C")
    n_symbols = int(getattr(getattr(environment, "dataset", None), "n_symbols", 0))
    if weights.shape != (n_symbols,) or not np.isfinite(weights).all():
        raise ValueError("resolved episode initial weights are invalid")
    weights.setflags(write=False)
    return weights


class _InitialStateEvaluationEnvironment:
    def __init__(self, environment: Any, initial_state_mode: str) -> None:
        self._environment = environment
        self._initial_state_mode = initial_state_mode

    def __getattr__(self, name: str) -> Any:
        return getattr(self._environment, name)

    def reset(self, *, options: dict[str, object]) -> tuple[object, dict[str, object]]:
        resolved = dict(options)
        resolved["initial_state_mode"] = self._initial_state_mode
        return self._environment.reset(options=resolved)


def evaluate_episode_action_path(
    environment_factory: Any,
    contract: OracleEpisodeContract,
    *,
    actions: object | None = None,
    model: object | None = None,
) -> ActionPathEvaluation:
    environment = environment_factory()
    try:
        return evaluate_episode_action_path_on_environment(
            environment,
            contract,
            actions=actions,
            model=model,
        )
    finally:
        environment.close()


def evaluate_episode_action_path_on_environment(
    environment: Any,
    contract: OracleEpisodeContract,
    *,
    actions: object | None = None,
    model: object | None = None,
) -> ActionPathEvaluation:
    """Evaluate an episode without taking ownership of the environment lifecycle."""

    wrapped = _InitialStateEvaluationEnvironment(
        environment,
        contract.initial_state_mode,
    )
    return evaluate_action_path(
        wrapped,
        evaluation_range=(contract.start, contract.stop),
        actions=actions,
        model=model,
    )


@dataclass(frozen=True, slots=True)
class EpisodeBehaviorCloningRecord:
    episode_id: int
    start: int
    stop: int
    initial_state_mode: str
    oracle_performance: PathPerformanceMetrics
    causal_policy_performance: PathPerformanceMetrics
    causal_policy_evidence: ActionPathCollapseEvidence
    action_agreement_rate: float
    action_mae: float
    action_rmse: float
    action_diagnostics: dict[str, object]
    heldout_oracle_regret: float
    normalized_oracle_regret: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class EpisodeBehaviorCloningHoldoutEvaluation:
    records: tuple[EpisodeBehaviorCloningRecord, ...]
    causal_policy_performance: PathPerformanceMetrics
    causal_policy_evidence: ActionPathCollapseEvidence
    action_agreement_rate: float
    action_mae: float
    action_rmse: float
    heldout_oracle_regret: float
    normalized_oracle_regret: float
    causal_regret_upper_confidence_bound: float
    causal_net_return_lower_confidence_bound: float
    bootstrap_confidence_level: float
    bootstrap_resamples: int
    schema_version: str = EPISODE_ORACLE_BC_EVALUATION_SCHEMA

    def __post_init__(self) -> None:
        if not self.records:
            raise ValueError("episode BC holdout requires at least one episode")
        if self.schema_version != EPISODE_ORACLE_BC_EVALUATION_SCHEMA:
            raise ValueError("unsupported episode BC evaluation schema")
        for value in (
            self.action_agreement_rate,
            self.action_mae,
            self.action_rmse,
            self.heldout_oracle_regret,
            self.normalized_oracle_regret,
            self.causal_regret_upper_confidence_bound,
            self.bootstrap_confidence_level,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "episode BC holdout metrics must be finite and non-negative"
                )
        if not math.isfinite(self.causal_net_return_lower_confidence_bound):
            raise ValueError("causal net-return lower confidence bound must be finite")
        if self.action_agreement_rate > 1.0:
            raise ValueError("episode BC action agreement exceeds one")
        if not 0.5 < self.bootstrap_confidence_level < 1.0:
            raise ValueError("episode BC bootstrap confidence is invalid")
        if self.bootstrap_resamples < 1_000:
            raise ValueError("episode BC bootstrap resamples are insufficient")

    def to_dict(self) -> dict[str, object]:
        return {
            "action_agreement_rate": self.action_agreement_rate,
            "action_mae": self.action_mae,
            "action_rmse": self.action_rmse,
            "causal_policy_evidence": self.causal_policy_evidence.to_dict(),
            "causal_policy_performance": asdict(self.causal_policy_performance),
            "episode_count": len(self.records),
            "heldout_oracle_regret": self.heldout_oracle_regret,
            "normalized_oracle_regret": self.normalized_oracle_regret,
            "causal_regret_upper_confidence_bound": (
                self.causal_regret_upper_confidence_bound
            ),
            "causal_net_return_lower_confidence_bound": (
                self.causal_net_return_lower_confidence_bound
            ),
            "bootstrap_confidence_level": self.bootstrap_confidence_level,
            "bootstrap_resamples": self.bootstrap_resamples,
            "records": tuple(record.to_dict() for record in self.records),
            "schema_version": self.schema_version,
        }


def _aggregate_collapse_evidence(
    evaluations: tuple[ActionPathEvaluation, ...],
) -> ActionPathCollapseEvidence:
    first = evaluations[0].collapse_evidence
    evidence = tuple(item.collapse_evidence for item in evaluations)
    return ActionPathCollapseEvidence(
        decision_count=sum(item.decision_count for item in evidence),
        action_dimension_count=first.action_dimension_count,
        active_dimension_count=sum(item.active_dimension_count for item in evidence),
        inactive_dimension_count=sum(
            item.inactive_dimension_count for item in evidence
        ),
        proposal_distance_count=sum(item.proposal_distance_count for item in evidence),
        submitted_change_count=sum(item.submitted_change_count for item in evidence),
        downstream_no_trade_suppression_count=sum(
            item.downstream_no_trade_suppression_count for item in evidence
        ),
        execution_rejection_count=sum(
            item.execution_rejection_count for item in evidence
        ),
        executed_change_count=sum(item.executed_change_count for item in evidence),
        trade_count=sum(item.trade_count for item in evidence),
        constant_submitted_actions=all(
            item.constant_submitted_actions for item in evidence
        ),
        execution_rejection_reason_counts=_merge_reason_counts(
            tuple(item.execution_rejection_reason_counts for item in evidence)
        ),
        risk_projection_reason_counts=_merge_reason_counts(
            tuple(item.risk_projection_reason_counts for item in evidence)
        ),
        hard_risk_violation=any(item.hard_risk_violation for item in evidence),
    )


def _merge_reason_counts(
    values: tuple[tuple[tuple[str, int], ...], ...],
) -> tuple[tuple[str, int], ...]:
    merged: Counter[str] = Counter()
    for reason_counts in values:
        merged.update(dict(reason_counts))
    return tuple(sorted(merged.items()))


def aggregate_episode_behavior_cloning_holdouts(
    evaluations: tuple[EpisodeBehaviorCloningHoldoutEvaluation, ...],
    *,
    seed_material: str,
) -> EpisodeBehaviorCloningHoldoutEvaluation:
    """Combine per-symbol causal holdouts without losing episode support."""

    if not evaluations:
        raise ValueError("Universal BC holdout requires at least one evaluation")
    confidence = evaluations[0].bootstrap_confidence_level
    resamples = evaluations[0].bootstrap_resamples
    if any(
        evaluation.bootstrap_confidence_level != confidence
        or evaluation.bootstrap_resamples != resamples
        for evaluation in evaluations
    ):
        raise ValueError("Universal BC holdout bootstrap contracts differ")
    records = tuple(
        record for evaluation in evaluations for record in evaluation.records
    )
    evidence_values = tuple(record.causal_policy_evidence for record in records)
    first = evidence_values[0]
    if any(
        value.action_dimension_count != first.action_dimension_count
        for value in evidence_values
    ):
        raise ValueError("Universal BC holdout action dimensions differ")
    evidence = ActionPathCollapseEvidence(
        decision_count=sum(value.decision_count for value in evidence_values),
        action_dimension_count=first.action_dimension_count,
        active_dimension_count=sum(
            value.active_dimension_count for value in evidence_values
        ),
        inactive_dimension_count=sum(
            value.inactive_dimension_count for value in evidence_values
        ),
        proposal_distance_count=sum(
            value.proposal_distance_count for value in evidence_values
        ),
        submitted_change_count=sum(
            value.submitted_change_count for value in evidence_values
        ),
        downstream_no_trade_suppression_count=sum(
            value.downstream_no_trade_suppression_count for value in evidence_values
        ),
        execution_rejection_count=sum(
            value.execution_rejection_count for value in evidence_values
        ),
        executed_change_count=sum(
            value.executed_change_count for value in evidence_values
        ),
        trade_count=sum(value.trade_count for value in evidence_values),
        constant_submitted_actions=all(
            value.constant_submitted_actions for value in evidence_values
        ),
        execution_rejection_reason_counts=_merge_reason_counts(
            tuple(value.execution_rejection_reason_counts for value in evidence_values)
        ),
        risk_projection_reason_counts=_merge_reason_counts(
            tuple(value.risk_projection_reason_counts for value in evidence_values)
        ),
        hard_risk_violation=any(value.hard_risk_violation for value in evidence_values),
    )
    causal_returns = np.asarray(
        [record.causal_policy_performance.net_return for record in records],
        dtype=np.float64,
    )
    normalized_regrets = np.asarray(
        [record.normalized_oracle_regret for record in records],
        dtype=np.float64,
    )
    worst = min(records, key=lambda record: record.causal_policy_performance.net_return)
    return EpisodeBehaviorCloningHoldoutEvaluation(
        records=records,
        causal_policy_performance=worst.causal_policy_performance,
        causal_policy_evidence=evidence,
        action_agreement_rate=min(record.action_agreement_rate for record in records),
        action_mae=max(record.action_mae for record in records),
        action_rmse=max(record.action_rmse for record in records),
        heldout_oracle_regret=max(record.heldout_oracle_regret for record in records),
        normalized_oracle_regret=max(
            record.normalized_oracle_regret for record in records
        ),
        causal_regret_upper_confidence_bound=deterministic_bootstrap_upper_bound(
            normalized_regrets,
            confidence_level=confidence,
            resamples=resamples,
            seed_material=f"{seed_material}:oracle-regret",
        ),
        causal_net_return_lower_confidence_bound=deterministic_bootstrap_lower_bound(
            causal_returns,
            confidence_level=confidence,
            resamples=resamples,
            seed_material=f"{seed_material}:causal-return",
        ),
        bootstrap_confidence_level=confidence,
        bootstrap_resamples=resamples,
    )


def _write_evaluation(path: Path, payload: dict[str, object]) -> str:
    digest = content_digest(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes({**payload, "artifact_digest": digest}))
    return digest


def evaluate_episode_behavior_cloning_holdout(
    *,
    environment_factory: Any,
    model: object,
    batch: EpisodeOracleBatch,
    split: BehaviorCloningSplit,
    output_root: Path,
    action_tolerance: float = 0.05,
    bootstrap_confidence_level: float = 0.95,
    bootstrap_resamples: int = 2_000,
) -> tuple[dict[str, object], EpisodeBehaviorCloningHoldoutEvaluation | None]:
    """Evaluate complete validation episodes and retain the worst Oracle regret."""

    validation_ids = tuple(int(value) for value in split.validation_episode_ids)
    oracle_summary = {
        "episode_batch_digest": batch.digest,
        "episode_count": batch.episode_count,
        "evaluation_scope": "hindsight_episode_diagnostic",
        "schema_version": EPISODE_ORACLE_BC_EVALUATION_SCHEMA,
        "validation_episode_ids": validation_ids,
    }
    oracle_evaluation_digest = _write_evaluation(
        output_root / "oracle-evaluation.json",
        oracle_summary,
    )
    if not validation_ids:
        return (
            {
                "oracle_evaluation_digest": oracle_evaluation_digest,
                "passed": True,
                "reason": "episode holdout is disabled",
                "required": False,
                "schema_version": "oracle_bc_reproduction_gate_v2",
            },
            None,
        )
    contract_by_id = {contract.episode_index: contract for contract in batch.contracts}
    target_by_id = {
        contract.episode_index: targets
        for contract, targets in zip(batch.contracts, batch.targets, strict=True)
    }
    records: list[EpisodeBehaviorCloningRecord] = []
    policy_paths: list[ActionPathEvaluation] = []
    for episode_id in validation_ids:
        contract = contract_by_id[episode_id]
        oracle_path = evaluate_episode_action_path(
            environment_factory,
            contract,
            actions=target_by_id[episode_id],
        )
        policy_path = evaluate_episode_action_path(
            environment_factory,
            contract,
            model=model,
        )
        difference = oracle_path.actions.astype(
            np.float64
        ) - policy_path.actions.astype(np.float64)
        regret = max(
            0.0,
            oracle_path.performance.net_return - policy_path.performance.net_return,
        )
        regret_scale = max(abs(oracle_path.performance.net_return), 0.05)
        records.append(
            EpisodeBehaviorCloningRecord(
                episode_id=episode_id,
                start=contract.start,
                stop=contract.stop,
                initial_state_mode=contract.initial_state_mode,
                oracle_performance=oracle_path.performance,
                causal_policy_performance=policy_path.performance,
                causal_policy_evidence=policy_path.collapse_evidence,
                action_agreement_rate=float(
                    np.mean(np.all(np.abs(difference) <= action_tolerance, axis=1))
                ),
                action_mae=float(np.mean(np.abs(difference), dtype=np.float64)),
                action_rmse=float(
                    np.sqrt(np.mean(np.square(difference), dtype=np.float64))
                ),
                action_diagnostics=_action_comparison_diagnostics(
                    oracle_path.actions,
                    policy_path.actions,
                    initial_weights=contract.initial_weights,
                    tolerance=action_tolerance,
                ),
                heldout_oracle_regret=float(regret),
                normalized_oracle_regret=float(regret / regret_scale),
            )
        )
        policy_paths.append(policy_path)
    resolved_records = tuple(records)
    regret_upper = deterministic_bootstrap_upper_bound(
        np.asarray(
            [record.normalized_oracle_regret for record in resolved_records],
            dtype=np.float64,
        ),
        confidence_level=bootstrap_confidence_level,
        resamples=bootstrap_resamples,
        seed_material=content_digest(
            {
                "batch_digest": batch.digest,
                "validation_episode_ids": validation_ids,
            }
        ),
    )
    causal_return_lower = deterministic_bootstrap_lower_bound(
        np.asarray(
            [
                record.causal_policy_performance.net_return
                for record in resolved_records
            ],
            dtype=np.float64,
        ),
        confidence_level=bootstrap_confidence_level,
        resamples=bootstrap_resamples,
        seed_material=content_digest(
            {
                "batch_digest": batch.digest,
                "scope": "causal_policy_net_return",
                "validation_episode_ids": validation_ids,
            }
        ),
    )
    worst = max(resolved_records, key=lambda record: record.normalized_oracle_regret)
    holdout = EpisodeBehaviorCloningHoldoutEvaluation(
        records=resolved_records,
        causal_policy_performance=worst.causal_policy_performance,
        causal_policy_evidence=_aggregate_collapse_evidence(tuple(policy_paths)),
        action_agreement_rate=min(
            record.action_agreement_rate for record in resolved_records
        ),
        action_mae=max(record.action_mae for record in resolved_records),
        action_rmse=max(record.action_rmse for record in resolved_records),
        heldout_oracle_regret=max(
            record.heldout_oracle_regret for record in resolved_records
        ),
        normalized_oracle_regret=max(
            record.normalized_oracle_regret for record in resolved_records
        ),
        causal_regret_upper_confidence_bound=regret_upper,
        causal_net_return_lower_confidence_bound=causal_return_lower,
        bootstrap_confidence_level=bootstrap_confidence_level,
        bootstrap_resamples=bootstrap_resamples,
    )
    holdout_digest = _write_evaluation(
        output_root / "behavior-cloning-holdout.json",
        holdout.to_dict(),
    )
    passed = bool(
        holdout.action_agreement_rate >= 0.80
        and holdout.action_rmse <= 0.10
        and holdout.normalized_oracle_regret <= 0.25
    )
    return (
        {
            "action_agreement_minimum": 0.80,
            "action_agreement_rate": holdout.action_agreement_rate,
            "action_rmse": holdout.action_rmse,
            "action_rmse_maximum": 0.10,
            "behavior_cloning_holdout_digest": holdout_digest,
            "causal_policy_evidence": holdout.causal_policy_evidence.to_dict(),
            "heldout_episode_count": len(resolved_records),
            "heldout_oracle_regret": holdout.heldout_oracle_regret,
            "normalized_oracle_regret": holdout.normalized_oracle_regret,
            "causal_regret_upper_confidence_bound": (
                holdout.causal_regret_upper_confidence_bound
            ),
            "causal_net_return_lower_confidence_bound": (
                holdout.causal_net_return_lower_confidence_bound
            ),
            "bootstrap_confidence_level": holdout.bootstrap_confidence_level,
            "bootstrap_resamples": holdout.bootstrap_resamples,
            "normalized_oracle_regret_maximum": 0.25,
            "oracle_evaluation_digest": oracle_evaluation_digest,
            "passed": passed,
            "required": False,
            "reason": (
                "the teacher remains a hindsight upper-bound diagnostic; causal "
                "policy reproduction is reported but not required"
            ),
            "schema_version": "oracle_bc_reproduction_gate_v2",
        },
        holdout,
    )


__all__ = [
    "EPISODE_ORACLE_BC_EVALUATION_SCHEMA",
    "EpisodeBehaviorCloningHoldoutEvaluation",
    "EpisodeBehaviorCloningRecord",
    "aggregate_episode_behavior_cloning_holdouts",
    "evaluate_episode_action_path",
    "evaluate_episode_action_path_on_environment",
    "evaluate_episode_behavior_cloning_holdout",
    "oracle_episode_sampling_config",
    "resolve_episode_initial_weights",
]
