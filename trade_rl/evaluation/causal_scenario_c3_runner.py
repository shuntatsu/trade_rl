"""Realized C3 comparison after a decision has been persisted."""

from __future__ import annotations

import math
from typing import Protocol

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_c3_contracts import (
    CausalScenarioC3Config,
    CausalScenarioQueryComparison,
    PerfectInformationComparison,
    PersistedScenarioDecision,
    RealizedPolicyOutcome,
)
from trade_rl.evaluation.causal_scenario_c3_decision_artifact import LoadedC3Decision
from trade_rl.evaluation.causal_scenario_values import CausalScenarioEvaluationResult


class C3RealizedReplay(Protocol):
    def run(
        self,
        raw_residual: np.ndarray,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
        policy_kind: str,
    ) -> RealizedPolicyOutcome: ...


def build_persisted_scenario_decision(
    result: CausalScenarioEvaluationResult,
) -> PersistedScenarioDecision:
    """Freeze the C1 choice without reading a realized query future."""

    payload = {
        "candidate_digests": result.candidate_digests,
        "candidate_generator_digest": result.candidate_generator_digest,
        "created_before_realized_replay": True,
        "dataset_id": result.dataset_id,
        "fold_digest": result.fold_digest,
        "projected_targets": result.projected_targets.tolist(),
        "query_index": result.query_index,
        "query_timestamp_ns": result.query_timestamp_ns,
        "raw_candidate_actions": result.raw_candidate_actions.tolist(),
        "regret": result.regret.tolist(),
        "scenario_library_digest": result.scenario_library_digest,
        "scenario_set_digest": result.scenario_set_digest,
        "schema_version": "causal_scenario_c3_decision_v1",
        "score": result.score.tolist(),
        "selected_candidate_digest": result.candidate_digests[
            result.selected_candidate_index
        ],
        "selected_candidate_index": result.selected_candidate_index,
        "state_snapshot_digest": result.state_snapshot_digest,
        "tie_candidate_indices": result.tie_candidate_indices,
        "value_result_digest": result.result_digest,
        "zero_candidate_index": result.zero_candidate_index,
    }
    return PersistedScenarioDecision(
        dataset_id=result.dataset_id,
        fold_digest=result.fold_digest,
        query_index=result.query_index,
        query_timestamp_ns=result.query_timestamp_ns,
        state_snapshot_digest=result.state_snapshot_digest,
        scenario_library_digest=result.scenario_library_digest,
        scenario_set_digest=result.scenario_set_digest,
        candidate_generator_digest=result.candidate_generator_digest,
        value_result_digest=result.result_digest,
        candidate_digests=result.candidate_digests,
        raw_candidate_actions=result.raw_candidate_actions,
        projected_targets=result.projected_targets,
        score=result.score,
        regret=result.regret,
        selected_candidate_index=result.selected_candidate_index,
        zero_candidate_index=result.zero_candidate_index,
        tie_candidate_indices=result.tie_candidate_indices,
        selected_candidate_digest=result.candidate_digests[
            result.selected_candidate_index
        ],
        created_before_realized_replay=True,
        decision_digest=content_digest(payload),
    )


def _rank_average(values: np.ndarray, *, tolerance: float) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < order.size:
        stop = start + 1
        while stop < order.size and math.isclose(
            float(values[order[stop]]),
            float(values[order[start]]),
            rel_tol=0.0,
            abs_tol=tolerance,
        ):
            stop += 1
        average_rank = 0.5 * (start + stop - 1)
        ranks[order[start:stop]] = average_rank
        start = stop
    return ranks


def _spearman(left: np.ndarray, right: np.ndarray, *, tolerance: float) -> float:
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("Spearman inputs must be matching non-trivial vectors")
    left_rank = _rank_average(left, tolerance=tolerance)
    right_rank = _rank_average(right, tolerance=tolerance)
    left_centered = left_rank - float(left_rank.mean())
    right_centered = right_rank - float(right_rank.mean())
    denominator = float(
        np.sqrt(np.dot(left_centered, left_centered))
        * np.sqrt(np.dot(right_centered, right_centered))
    )
    if denominator <= tolerance:
        return 0.0
    return float(np.dot(left_centered, right_centered) / denominator)


def _validate_ppo_action(action: np.ndarray, *, dimension: int) -> np.ndarray:
    value = np.asarray(action, dtype=np.float64).copy(order="C")
    if value.shape != (dimension,) or not np.isfinite(value).all():
        raise ValueError("ppo_mean_action has invalid shape or values")
    if np.any(np.abs(value) > 1.0):
        raise ValueError("ppo_mean_action must be within [-1, 1]")
    value[value == 0.0] = 0.0
    value.setflags(write=False)
    return value


def _random_candidate_index(
    decision: PersistedScenarioDecision, config: CausalScenarioC3Config
) -> int:
    candidates = tuple(
        index
        for index in range(len(decision.candidate_digests))
        if index != decision.zero_candidate_index
    )
    if not candidates:
        return decision.zero_candidate_index
    digest = content_digest(
        {
            "config_digest": config.digest,
            "decision_digest": decision.decision_digest,
            "schema_version": "causal_scenario_c3_random_candidate_v1",
        }
    )
    return candidates[int(digest[:16], 16) % len(candidates)]


def run_c3_query_comparison(
    loaded_decision: LoadedC3Decision,
    *,
    replay: C3RealizedReplay,
    ppo_mean_action: np.ndarray,
    config: CausalScenarioC3Config,
    perfect_information: PerfectInformationComparison | None = None,
) -> CausalScenarioQueryComparison:
    """Run realized C3 replay only from a verified persisted decision artifact."""

    if not isinstance(loaded_decision, LoadedC3Decision):
        raise TypeError("loaded_decision must be LoadedC3Decision")
    if not isinstance(config, CausalScenarioC3Config):
        raise TypeError("config must be CausalScenarioC3Config")
    decision = loaded_decision.decision
    action_dimension = int(decision.raw_candidate_actions.shape[1])
    ppo_action = _validate_ppo_action(ppo_mean_action, dimension=action_dimension)

    candidate_outcomes = tuple(
        replay.run(
            decision.raw_candidate_actions[index],
            horizon_decisions=config.horizon_decisions,
            zero_residual_after_first=True,
            policy_kind=f"candidate:{index}",
        )
        for index in range(len(decision.candidate_digests))
    )
    candidate_returns = np.asarray(
        [item.gross_log_return for item in candidate_outcomes], dtype=np.float64
    )
    zero_return = float(candidate_returns[decision.zero_candidate_index])
    realized_advantages = candidate_returns - zero_return
    realized_advantages.setflags(write=False)
    best_realized = float(realized_advantages.max())
    selected_regret = best_realized - float(
        realized_advantages[decision.selected_candidate_index]
    )
    random_index = _random_candidate_index(decision, config)
    random_regret = best_realized - float(realized_advantages[random_index])

    trend = replay.run(
        np.zeros(action_dimension, dtype=np.float64),
        horizon_decisions=config.horizon_decisions,
        zero_residual_after_first=True,
        policy_kind="trend",
    )
    scenario_oracle = replay.run(
        decision.selected_raw_residual,
        horizon_decisions=config.horizon_decisions,
        zero_residual_after_first=True,
        policy_kind="scenario_oracle",
    )
    ppo_mean = replay.run(
        ppo_action,
        horizon_decisions=config.horizon_decisions,
        zero_residual_after_first=True,
        policy_kind="ppo_mean",
    )
    random_candidate = replay.run(
        decision.raw_candidate_actions[random_index],
        horizon_decisions=config.horizon_decisions,
        zero_residual_after_first=True,
        policy_kind="random_candidate",
    )
    return CausalScenarioQueryComparison(
        decision_digest=decision.decision_digest,
        trend=trend,
        scenario_oracle=scenario_oracle,
        ppo_mean=ppo_mean,
        random_candidate=random_candidate,
        candidate_outcomes=candidate_outcomes,
        realized_candidate_advantages=realized_advantages,
        predicted_realized_spearman=_spearman(
            decision.score,
            realized_advantages,
            tolerance=config.ranking_tolerance,
        ),
        selected_realized_regret=max(selected_regret, 0.0),
        random_realized_regret=max(random_regret, 0.0),
        perfect_information=(
            PerfectInformationComparison.not_evaluated()
            if perfect_information is None
            else perfect_information
        ),
    )
