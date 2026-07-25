from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation.causal_scenario_values import (
    CausalQuerySnapshot,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
)


def sha(char: str) -> str:
    return char * 64


def query_snapshot(*, digest: str | None = None) -> CausalQuerySnapshot:
    return CausalQuerySnapshot(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        train_start=0,
        train_stop=10_000,
        query_index=10_100,
        query_timestamp_ns=1_800_000_000_000_000_000,
        source_commit="c" * 40,
        query_digest=sha("1") if digest is None else digest,
        state_snapshot_digest=sha("2"),
        observation_digest=sha("3"),
        environment_digest=sha("4"),
        action_spec_digest=sha("5"),
        execution_policy_digest=sha("6"),
        risk_digest=sha("7"),
        trend_digest=sha("8"),
        starting_equity=100_000.0,
        baseline_target=np.asarray([0.1, -0.2, 0.0]),
    )


def scenario_set(count: int = 64) -> CausalScenarioSet:
    return CausalScenarioSet(
        scenario_ids=tuple(f"scenario-{index:02d}" for index in range(count)),
        probabilities=np.full(count, 1.0 / count),
        anchor_indices=np.arange(count, dtype=np.int64),
        distances=np.arange(count, dtype=np.float64),
        query_condition=np.asarray([0.5, -0.5]),
        anchor_conditions=np.zeros((count, 2), dtype=np.float64),
        library_digest=sha("9"),
    )


def candidate_digest(target: np.ndarray, execution_digest: str) -> str:
    return content_digest(
        {
            "execution_intent_digest": execution_digest,
            "projected_target": target.tolist(),
            "schema_version": "projected_residual_candidate_v1",
        }
    )


def evidence(
    *, terminal: float, log_return: float, turnover: float, cost: float
) -> ScenarioRolloutEvidence:
    payload = {
        "feasible": True,
        "fill_ratio": 1.0,
        "filled_turnover": turnover,
        "interval_cost": cost,
        "reported_log_return": log_return,
        "schema_version": "scenario_rollout_evidence_v1",
        "terminal_equity": terminal,
        "termination_reason": "horizon",
    }
    return ScenarioRolloutEvidence(
        terminal_equity=terminal,
        reported_log_return=log_return,
        filled_turnover=turnover,
        interval_cost=cost,
        fill_ratio=1.0,
        feasible=True,
        termination_reason="horizon",
        evidence_digest=content_digest(payload),
    )


@dataclass
class ArtificialRollout:
    query: CausalQuerySnapshot
    coefficients: np.ndarray
    base_return: float
    action_cost: float

    def run(
        self,
        candidate: ProjectedResidualCandidate,
        *,
        horizon_decisions: int,
        zero_residual_after_first: bool,
    ) -> ScenarioRolloutEvidence:
        assert horizon_decisions == 96
        assert zero_residual_after_first is True
        turnover = float(np.abs(candidate.raw_action).sum())
        log_return = float(
            self.base_return
            + np.dot(self.coefficients, candidate.raw_action)
            - self.action_cost * turnover
        )
        return evidence(
            terminal=self.query.starting_equity * math.exp(log_return),
            log_return=log_return,
            turnover=turnover,
            cost=self.action_cost * turnover,
        )


class ArtificialFactory:
    def __init__(
        self,
        coefficients: np.ndarray,
        *,
        base_returns: np.ndarray | None = None,
        action_cost: float = 0.0,
        alias_projection: bool = False,
    ) -> None:
        self.coefficients = np.asarray(coefficients, dtype=np.float64)
        self.base_returns = (
            np.zeros(len(self.coefficients), dtype=np.float64)
            if base_returns is None
            else np.asarray(base_returns, dtype=np.float64)
        )
        self.action_cost = action_cost
        self.alias_projection = alias_projection
        self.created = 0

    def project_candidate(
        self, query: CausalQuerySnapshot, raw_action: np.ndarray
    ) -> ProjectedResidualCandidate:
        raw = np.asarray(raw_action, dtype=np.float64)
        if self.alias_projection:
            target = query.baseline_target.copy()
        else:
            target = np.clip(query.baseline_target + 0.25 * raw, -0.45, 0.45)
        execution_digest = content_digest(
            {
                "projected_target": target.tolist(),
                "schema_version": "artificial_execution_intent_v1",
            }
        )
        return ProjectedResidualCandidate(
            raw_action=raw,
            projected_target=target,
            execution_intent_digest=execution_digest,
            candidate_digest=candidate_digest(target, execution_digest),
            expected_turnover_hint=float(np.abs(raw).sum()),
            is_zero=bool(np.all(raw == 0.0)),
        )

    def create_rollout(
        self,
        query: CausalQuerySnapshot,
        scenario_index: int,
        scenario_id: str,
    ) -> ArtificialRollout:
        assert scenario_id == f"scenario-{scenario_index:02d}"
        self.created += 1
        return ArtificialRollout(
            query=query,
            coefficients=self.coefficients[scenario_index],
            base_return=float(self.base_returns[scenario_index]),
            action_cost=self.action_cost,
        )


def evaluate(coefficients: np.ndarray, **factory_kwargs: object):
    factory = ArtificialFactory(coefficients, **factory_kwargs)
    result = evaluate_causal_scenario_actions(
        query=query_snapshot(),
        scenarios=scenario_set(),
        config=CausalScenarioEvaluatorConfig(action_dimension=3),
        rollout_factory=factory,
    )
    return result, factory


def test_monotonic_up_scenarios_select_positive_first_asset() -> None:
    coefficients = np.tile(np.asarray([0.02, 0.0, 0.0]), (64, 1))
    result, factory = evaluate(coefficients)
    np.testing.assert_array_equal(
        result.raw_candidate_actions[result.selected_candidate_index],
        np.asarray([1.0, 0.0, 0.0]),
    )
    assert result.score[result.selected_candidate_index] > 0.0
    assert factory.created == 64 * len(result.candidate_digests)


def test_monotonic_down_scenarios_select_negative_first_asset() -> None:
    coefficients = np.tile(np.asarray([-0.02, 0.0, 0.0]), (64, 1))
    result, _ = evaluate(coefficients)
    np.testing.assert_array_equal(
        result.raw_candidate_actions[result.selected_candidate_index],
        np.asarray([-1.0, 0.0, 0.0]),
    )


def test_flat_scenarios_select_zero_by_tie_break() -> None:
    result, _ = evaluate(np.zeros((64, 3)))
    assert result.selected_candidate_index == result.zero_candidate_index
    assert len(result.tie_candidate_indices) == len(result.candidate_digests)


def test_high_cost_selects_zero() -> None:
    coefficients = np.tile(np.asarray([0.001, 0.0, 0.0]), (64, 1))
    result, _ = evaluate(coefficients, action_cost=0.01)
    assert result.selected_candidate_index == result.zero_candidate_index


def test_asymmetric_downside_prefers_safer_candidate() -> None:
    coefficients = np.zeros((64, 3), dtype=np.float64)
    coefficients[:, 0] = 0.02
    coefficients[:7, 0] = -0.20
    result, _ = evaluate(coefficients)
    selected = result.raw_candidate_actions[result.selected_candidate_index]
    assert selected[0] <= 0.5
    assert (
        result.loss_cvar[result.selected_candidate_index]
        <= result.loss_cvar[
            int(
                np.flatnonzero(
                    np.all(result.raw_candidate_actions == [1, 0, 0], axis=1)
                )[0]
            )
        ]
    )


def test_projection_aliases_deduplicate_but_keep_zero() -> None:
    result, _ = evaluate(np.zeros((64, 3)), alias_projection=True)
    assert len(result.candidate_digests) == 1
    assert result.zero_candidate_index == 0
    assert result.selected_candidate_index == 0


def test_regret_and_advantage_are_recomputed() -> None:
    coefficients = np.tile(np.asarray([0.02, -0.01, 0.0]), (64, 1))
    result, _ = evaluate(coefficients)
    np.testing.assert_allclose(result.regret, result.score.max() - result.score)
    np.testing.assert_allclose(
        result.baseline_relative_advantages,
        result.gross_log_returns
        - result.gross_log_returns[:, result.zero_candidate_index][:, None],
    )
    assert abs(result.regret[result.selected_candidate_index]) <= 1e-12


def test_bootstrap_is_deterministic_and_query_bound() -> None:
    coefficients = np.zeros((64, 3), dtype=np.float64)
    coefficients[:, 0] = np.linspace(-0.02, 0.03, 64)
    factory1 = ArtificialFactory(coefficients)
    first = evaluate_causal_scenario_actions(
        query=query_snapshot(),
        scenarios=scenario_set(),
        config=CausalScenarioEvaluatorConfig(action_dimension=3),
        rollout_factory=factory1,
    )
    factory2 = ArtificialFactory(coefficients)
    second = evaluate_causal_scenario_actions(
        query=query_snapshot(),
        scenarios=scenario_set(),
        config=CausalScenarioEvaluatorConfig(action_dimension=3),
        rollout_factory=factory2,
    )
    np.testing.assert_array_equal(first.confidence_lower, second.confidence_lower)
    assert first.result_digest == second.result_digest

    changed = evaluate_causal_scenario_actions(
        query=query_snapshot(digest=sha("d")),
        scenarios=scenario_set(),
        config=CausalScenarioEvaluatorConfig(action_dimension=3),
        rollout_factory=ArtificialFactory(coefficients),
    )
    np.testing.assert_allclose(first.mean_advantage, changed.mean_advantage)
    assert not np.array_equal(first.confidence_lower, changed.confidence_lower)
    assert first.result_digest != changed.result_digest


def test_all_positive_advantages_have_zero_downside_cvar() -> None:
    coefficients = np.tile(np.asarray([0.02, 0.0, 0.0]), (64, 1))
    result, _ = evaluate(coefficients)
    selected = result.selected_candidate_index
    assert np.all(result.baseline_relative_advantages[:, selected] > 0.0)
    assert result.loss_cvar[selected] == 0.0
