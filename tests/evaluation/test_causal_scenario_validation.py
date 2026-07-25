from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

import numpy as np
import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.evaluation import causal_scenario_values as module
from trade_rl.evaluation.causal_scenario_values import (
    CausalQuerySnapshot,
    CausalScenarioEvaluationResult,
    CausalScenarioEvaluatorConfig,
    ProjectedResidualCandidate,
    ScenarioRolloutEvidence,
    evaluate_causal_scenario_actions,
    generate_residual_candidates,
)


def sha(char: str) -> str:
    return char * 64


def query_snapshot(
    *,
    digest: str | None = None,
    baseline_target: np.ndarray | None = None,
) -> CausalQuerySnapshot:
    baseline = (
        np.asarray([0.1, -0.2, 0.0], dtype=np.float64)
        if baseline_target is None
        else np.asarray(baseline_target, dtype=np.float64)
    )
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
        baseline_target=baseline,
    )


def scenario_set(
    count: int = 64,
    *,
    condition_dimension: int = 2,
) -> module.CausalScenarioSet:
    return module.CausalScenarioSet(
        scenario_ids=tuple(f"scenario-{index:02d}" for index in range(count)),
        probabilities=np.full(count, 1.0 / count),
        anchor_indices=np.arange(count, dtype=np.int64),
        distances=np.arange(count, dtype=np.float64),
        query_condition=np.zeros(condition_dimension, dtype=np.float64),
        anchor_conditions=np.zeros(
            (count, condition_dimension),
            dtype=np.float64,
        ),
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
    *,
    terminal: float,
    log_return: float,
    turnover: float,
    cost: float,
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

    def project_candidate(
        self,
        query: CausalQuerySnapshot,
        raw_action: np.ndarray,
    ) -> ProjectedResidualCandidate:
        raw = np.asarray(raw_action, dtype=np.float64)
        target = (
            query.baseline_target.copy()
            if self.alias_projection
            else np.clip(query.baseline_target + 0.25 * raw, -0.45, 0.45)
        )
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
        return ArtificialRollout(
            query=query,
            coefficients=self.coefficients[scenario_index],
            base_return=float(self.base_returns[scenario_index]),
            action_cost=self.action_cost,
        )


def valid_result() -> CausalScenarioEvaluationResult:
    count = 64
    coefficients = np.arange(count, dtype=np.float64).reshape(count, 1) * 0.0001
    return evaluate_causal_scenario_actions(
        query=query_snapshot(baseline_target=np.asarray([0.0])),
        scenarios=scenario_set(count, condition_dimension=1),
        config=CausalScenarioEvaluatorConfig(action_dimension=1),
        rollout_factory=ArtificialFactory(coefficients),
    )


def replace_result(
    result: CausalScenarioEvaluationResult,
    **changes: object,
) -> CausalScenarioEvaluationResult:
    return dataclasses.replace(result, **changes)


def forged_candidate(**changes: object) -> ProjectedResidualCandidate:
    target = np.asarray(changes.pop("projected_target", [0.0]), dtype=np.float64)
    execution = str(changes.pop("execution_intent_digest", sha("a")))
    raw = np.asarray(changes.pop("raw_action", [0.0]), dtype=np.float64)
    values: dict[str, object] = {
        "raw_action": raw,
        "projected_target": target,
        "execution_intent_digest": execution,
        "candidate_digest": content_digest(
            {
                "execution_intent_digest": execution,
                "projected_target": target.tolist(),
                "schema_version": "projected_residual_candidate_v1",
            }
        ),
        "expected_turnover_hint": float(np.abs(raw).sum()),
        "is_zero": bool(np.all(raw == 0.0)),
    }
    values.update(changes)
    candidate = object.__new__(ProjectedResidualCandidate)
    for name, value in values.items():
        object.__setattr__(candidate, name, value)
    return candidate


@pytest.mark.parametrize("value", [True, "bad", object()])
def test_finite_float_rejects_non_real_values(value: object) -> None:
    with pytest.raises(ValueError, match="finite real"):
        module._finite_float("value", value)


def test_array_helpers_reject_conversion_rank_and_contract_errors() -> None:
    with pytest.raises(ValueError, match="numeric array"):
        module._readonly_float_array("value", object(), ndim=1)
    with pytest.raises(ValueError, match="rank 1"):
        module._readonly_float_array("value", np.zeros((1, 1)), ndim=1)
    with pytest.raises(ValueError, match="shape contract"):
        module._readonly_float_array(
            "value",
            np.zeros(1),
            ndim=1,
            shape=(1, 1),
        )
    with pytest.raises(ValueError, match="rank 1"):
        module._readonly_int_array("value", np.zeros((1, 1), dtype=np.int64), ndim=1)
    with pytest.raises(ValueError, match="boolean array"):
        module._readonly_bool_array("value", np.zeros(1), ndim=1)
    with pytest.raises(ValueError, match="rank 1"):
        module._readonly_bool_array(
            "value",
            np.zeros((1, 1), dtype=np.bool_),
            ndim=1,
        )


def test_config_query_scenario_and_candidate_reject_remaining_invalid_branches() -> (
    None
):
    with pytest.raises(ValueError, match="schema"):
        CausalScenarioEvaluatorConfig(action_dimension=1, schema_version="bad")

    base = query_snapshot()
    kwargs = {
        field.name: getattr(base, field.name) for field in dataclasses.fields(base)
    }
    with pytest.raises(ValueError, match="train_stop"):
        CausalQuerySnapshot(**{**kwargs, "train_start": 2, "train_stop": 2})
    with pytest.raises(ValueError, match="baseline_target"):
        CausalQuerySnapshot(**{**kwargs, "baseline_target": np.asarray([])})

    scenarios = scenario_set()
    scenario_kwargs = {
        field.name: getattr(scenarios, field.name)
        for field in dataclasses.fields(scenarios)
    }
    distances = scenarios.distances.copy()
    distances[0] = -1.0
    with pytest.raises(ValueError, match="distances"):
        type(scenarios)(**{**scenario_kwargs, "distances": distances})

    target = np.asarray([0.0])
    execution = sha("a")
    candidate_digest = content_digest(
        {
            "execution_intent_digest": execution,
            "projected_target": target.tolist(),
            "schema_version": "projected_residual_candidate_v1",
        }
    )
    with pytest.raises(ValueError, match="expected_turnover_hint"):
        ProjectedResidualCandidate(
            raw_action=np.asarray([0.0]),
            projected_target=target,
            execution_intent_digest=execution,
            candidate_digest=candidate_digest,
            expected_turnover_hint=-1.0,
            is_zero=True,
        )
    payload = {
        "feasible": 1,
        "fill_ratio": 1.0,
        "filled_turnover": 0.0,
        "interval_cost": 0.0,
        "reported_log_return": 0.0,
        "schema_version": "scenario_rollout_evidence_v1",
        "terminal_equity": 1.0,
        "termination_reason": "horizon",
    }
    with pytest.raises(ValueError, match="feasible"):
        ScenarioRolloutEvidence(
            terminal_equity=1.0,
            reported_log_return=0.0,
            filled_turnover=0.0,
            interval_cost=0.0,
            fill_ratio=1.0,
            feasible=1,  # type: ignore[arg-type]
            termination_reason="horizon",
            evidence_digest=content_digest(payload),
        )


def test_generator_rejects_invalid_maximum_and_empty_target() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        generate_residual_candidates(np.zeros(1), max_candidates=33)
    with pytest.raises(ValueError, match="must not be empty"):
        generate_residual_candidates(np.asarray([]))


class NonCandidateFactory(ArtificialFactory):
    def project_candidate(self, query, raw_action):
        return object()


class MismatchedRawFactory(ArtificialFactory):
    def project_candidate(self, query, raw_action):
        candidate = super().project_candidate(query, raw_action)
        return forged_candidate(
            raw_action=np.ones_like(candidate.raw_action),
            projected_target=candidate.projected_target,
            execution_intent_digest=candidate.execution_intent_digest,
        )


class WrongTargetShapeFactory(ArtificialFactory):
    def project_candidate(self, query, raw_action):
        return forged_candidate(
            raw_action=np.asarray(raw_action),
            projected_target=np.zeros(query.action_dimension + 1),
        )


class CollisionFactory(ArtificialFactory):
    def __init__(self, coefficients: np.ndarray) -> None:
        super().__init__(coefficients)
        self._first: ProjectedResidualCandidate | None = None

    def project_candidate(self, query, raw_action):
        candidate = super().project_candidate(query, raw_action)
        if self._first is None:
            self._first = candidate
            return candidate
        if np.all(raw_action == 0.0):
            return candidate
        return forged_candidate(
            raw_action=np.asarray(raw_action),
            projected_target=candidate.projected_target + 0.01,
            execution_intent_digest=candidate.execution_intent_digest,
            candidate_digest=self._first.candidate_digest,
        )


class NoZeroFactory(ArtificialFactory):
    def project_candidate(self, query, raw_action):
        candidate = super().project_candidate(query, raw_action)
        if np.all(raw_action == 0.0):
            return forged_candidate(
                raw_action=candidate.raw_action,
                projected_target=candidate.projected_target,
                execution_intent_digest=candidate.execution_intent_digest,
                is_zero=False,
            )
        return candidate


@pytest.mark.parametrize(
    ("factory_type", "message"),
    [
        (NonCandidateFactory, "ProjectedResidualCandidate"),
        (MismatchedRawFactory, "raw_action mismatch"),
        (WrongTargetShapeFactory, "target dimension"),
        (CollisionFactory, "collision"),
        (NoZeroFactory, "zero residual"),
    ],
)
def test_projection_boundary_rejects_forged_factory_results(
    factory_type: type[ArtificialFactory],
    message: str,
) -> None:
    coefficients = np.zeros((64, 3), dtype=np.float64)
    with pytest.raises(ValueError, match=message):
        evaluate_causal_scenario_actions(
            query=query_snapshot(),
            scenarios=scenario_set(),
            config=CausalScenarioEvaluatorConfig(action_dimension=3),
            rollout_factory=factory_type(coefficients),
        )


def test_projected_candidate_bound_can_reject_excess_after_private_call() -> None:
    query = query_snapshot()
    raw = generate_residual_candidates(query.baseline_target)
    factory = ArtificialFactory(np.zeros((64, 3), dtype=np.float64))
    with pytest.raises(ValueError, match="projected candidate count"):
        module._project_candidates(query, raw, factory, max_candidates=1)


def test_evaluator_rejects_dimension_count_and_non_evidence() -> None:
    coefficients = np.zeros((64, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="action dimension"):
        evaluate_causal_scenario_actions(
            query=query_snapshot(),
            scenarios=scenario_set(),
            config=CausalScenarioEvaluatorConfig(action_dimension=2),
            rollout_factory=ArtificialFactory(coefficients),
        )
    with pytest.raises(ValueError, match="scenario count"):
        evaluate_causal_scenario_actions(
            query=query_snapshot(),
            scenarios=scenario_set(),
            config=CausalScenarioEvaluatorConfig(
                action_dimension=3,
                scenario_count=63,
            ),
            rollout_factory=ArtificialFactory(coefficients),
        )

    class BadRollout:
        def run(self, candidate, *, horizon_decisions, zero_residual_after_first):
            return object()

    class BadFactory(ArtificialFactory):
        def create_rollout(self, query, scenario_index, scenario_id):
            return BadRollout()

    with pytest.raises(ValueError, match="ScenarioRolloutEvidence"):
        evaluate_causal_scenario_actions(
            query=query_snapshot(),
            scenarios=scenario_set(),
            config=CausalScenarioEvaluatorConfig(action_dimension=3),
            rollout_factory=BadFactory(coefficients),
        )


def test_result_rejects_identity_metadata_and_economic_tampering() -> None:
    result = valid_result()
    cases: list[tuple[dict[str, object], str]] = [
        ({"config": object()}, "config"),
        ({"schema_version": "bad"}, "schema"),
        ({"dataset_id": "bad"}, "dataset_id"),
        ({"train_start": 2, "train_stop": 2}, "train_stop"),
        ({"starting_equity": 0.0}, "starting_equity"),
        ({"scenario_ids": result.scenario_ids[:-1]}, "scenario metadata"),
        (
            {
                "candidate_digests": (result.candidate_digests[0],)
                * len(result.candidate_digests)
            },
            "unique",
        ),
        (
            {"execution_intent_digests": result.execution_intent_digests[:-1]},
            "candidate metadata",
        ),
        ({"termination_reasons": ("z", "a")}, "sorted"),
        ({"selected_candidate_index": len(result.candidate_digests)}, "index"),
        ({"tie_candidate_indices": (1, 0)}, "sorted and unique"),
        ({"tie_candidate_indices": (len(result.candidate_digests),)}, "invalid index"),
        ({"raw_candidate_actions": result.raw_candidate_actions * 2.0}, "within"),
        ({"zero_candidate_index": 1}, "unique zero"),
        (
            {"candidate_digests": (sha("f"),) + result.candidate_digests[1:]},
            "projected target",
        ),
        ({"scenario_probabilities": np.full(64, -1.0 / 64.0)}, "non-negative"),
        (
            {
                "scenario_probabilities": np.asarray(
                    [0.02] + [(0.98 / 63.0)] * 63,
                    dtype=np.float64,
                )
            },
            "uniform",
        ),
        ({"scenario_distances": -np.ones_like(result.scenario_distances)}, "distances"),
        ({"scenario_set_digest": sha("f")}, "scenario_set_digest"),
        ({"feasible_mask": np.zeros_like(result.feasible_mask)}, "feasible"),
        ({"terminal_equity": np.zeros_like(result.terminal_equity)}, "terminal_equity"),
        ({"filled_turnover": -np.ones_like(result.filled_turnover)}, "non-negative"),
        ({"fill_ratio": np.full_like(result.fill_ratio, 2.0)}, "fill_ratio"),
        (
            {"termination_codes": np.full_like(result.termination_codes, 99)},
            "unknown reason",
        ),
        ({"gross_log_returns": result.gross_log_returns + 0.1}, "gross_log_returns"),
        (
            {
                "baseline_relative_advantages": (
                    result.baseline_relative_advantages + 0.1
                )
            },
            "baseline_relative_advantages",
        ),
        ({"mean_advantage": result.mean_advantage + 0.1}, "mean_advantage"),
        ({"tie_candidate_indices": ()}, "tie_candidate_indices"),
        (
            {
                "selected_candidate_index": (
                    1 if result.selected_candidate_index == 0 else 0
                )
            },
            "selected_candidate_index",
        ),
        ({"result_digest": sha("f")}, "result_digest"),
    ]
    for changes, message in cases:
        with pytest.raises(ValueError, match=message):
            replace_result(result, **changes)


def test_result_digest_payload_includes_digest_by_default() -> None:
    result = valid_result()
    assert result.digest_payload()["result_digest"] == result.result_digest
    assert "result_digest" not in result.digest_payload(include_result_digest=False)
