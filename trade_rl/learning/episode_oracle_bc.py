"""Episode-aligned Oracle behavior-cloning sampling and audit helpers."""

from __future__ import annotations

import math
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
)
from trade_rl.learning.rollout_evaluation import (
    ActionPathEvaluation,
    evaluate_action_path,
)

EPISODE_ORACLE_BC_EVALUATION_SCHEMA = "episode_oracle_bc_evaluation_v1"


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
    finally:
        environment.close()


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
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(
                    "episode BC holdout metrics must be finite and non-negative"
                )
        if self.action_agreement_rate > 1.0:
            raise ValueError("episode BC action agreement exceeds one")

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
                heldout_oracle_regret=float(regret),
                normalized_oracle_regret=float(regret / regret_scale),
            )
        )
        policy_paths.append(policy_path)
    resolved_records = tuple(records)
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
    "evaluate_episode_action_path",
    "evaluate_episode_behavior_cloning_holdout",
    "oracle_episode_sampling_config",
    "resolve_episode_initial_weights",
]
