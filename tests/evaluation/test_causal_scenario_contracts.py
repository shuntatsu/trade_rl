from __future__ import annotations

import math

import numpy as np
import pytest

from trade_rl.evaluation.causal_scenario_values import (
    CAUSAL_SCENARIO_EVALUATOR_SCHEMA,
    CausalQuerySnapshot,
    CausalScenarioEvaluatorConfig,
    CausalScenarioSet,
    generate_residual_candidates,
)


def sha(char: str) -> str:
    return char * 64


def query_snapshot() -> CausalQuerySnapshot:
    return CausalQuerySnapshot(
        dataset_id=sha("a"),
        fold_digest=sha("b"),
        train_start=0,
        train_stop=10_000,
        query_index=10_100,
        query_timestamp_ns=1_800_000_000_000_000_000,
        source_commit="c" * 40,
        query_digest=sha("1"),
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
        anchor_indices=np.full(count, -1, dtype=np.int64),
        distances=np.arange(count, dtype=np.float64),
        query_condition=np.asarray([0.5, -0.5]),
        anchor_conditions=np.zeros((count, 2), dtype=np.float64),
        library_digest=sha("9"),
    )


def test_default_config_is_digest_stable() -> None:
    config = CausalScenarioEvaluatorConfig(action_dimension=3)
    assert CAUSAL_SCENARIO_EVALUATOR_SCHEMA == "causal_scenario_action_evaluator_v1"
    assert config.scenario_count == 64
    assert config.horizon_decisions == 96
    assert config.cvar_alpha == 0.10
    assert config.cvar_penalty == 0.25
    assert config.bootstrap_resamples == 256
    assert config.confidence_level == 0.90
    assert config.score_tolerance == 1e-8
    assert config.max_candidates == 32
    assert config.digest == CausalScenarioEvaluatorConfig(action_dimension=3).digest


def test_query_and_scenario_arrays_are_read_only() -> None:
    query = query_snapshot()
    scenarios = scenario_set()
    for value in (
        query.baseline_target,
        scenarios.probabilities,
        scenarios.anchor_indices,
        scenarios.distances,
        scenarios.query_condition,
        scenarios.anchor_conditions,
    ):
        assert not value.flags.writeable
    with pytest.raises(ValueError):
        query.baseline_target[0] = 0.0


@pytest.mark.parametrize(
    "kwargs",
    [
        {"action_dimension": 0},
        {"action_dimension": True},
        {"action_dimension": 1, "scenario_count": 0},
        {"action_dimension": 1, "scenario_count": True},
        {"action_dimension": 1, "horizon_decisions": 0},
        {"action_dimension": 1, "cvar_alpha": 0.0},
        {"action_dimension": 1, "cvar_alpha": 1.1},
        {"action_dimension": 1, "cvar_penalty": -0.1},
        {"action_dimension": 1, "bootstrap_resamples": 0},
        {"action_dimension": 1, "confidence_level": 1.0},
        {"action_dimension": 1, "score_tolerance": -1.0},
        {"action_dimension": 1, "max_candidates": 33},
        {"action_dimension": 1, "replay_tolerance": 0.0},
        {"action_dimension": 1, "probability_tolerance": 0.0},
    ],
)
def test_config_rejects_invalid_values(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        CausalScenarioEvaluatorConfig(**kwargs)


def test_query_rejects_invalid_identity_and_numeric_values() -> None:
    values = (
        query_snapshot().__dict__ if hasattr(query_snapshot(), "__dict__") else None
    )
    assert values is None  # slots contract
    with pytest.raises(ValueError, match="dataset_id"):
        CausalQuerySnapshot(
            **{**_query_kwargs(), "dataset_id": "bad"},
        )
    with pytest.raises(ValueError, match="source_commit"):
        CausalQuerySnapshot(**{**_query_kwargs(), "source_commit": "bad"})
    with pytest.raises(ValueError, match="train"):
        CausalQuerySnapshot(**{**_query_kwargs(), "train_stop": 0})
    with pytest.raises(ValueError, match="query_index"):
        CausalQuerySnapshot(**{**_query_kwargs(), "query_index": -1})
    with pytest.raises(ValueError, match="query_timestamp_ns"):
        CausalQuerySnapshot(**{**_query_kwargs(), "query_timestamp_ns": 0})
    with pytest.raises(ValueError, match="starting_equity"):
        CausalQuerySnapshot(**{**_query_kwargs(), "starting_equity": 0.0})
    with pytest.raises(ValueError, match="baseline_target"):
        CausalQuerySnapshot(
            **{**_query_kwargs(), "baseline_target": np.asarray([math.nan])},
        )


def _query_kwargs() -> dict[str, object]:
    query = query_snapshot()
    return {
        "dataset_id": query.dataset_id,
        "fold_digest": query.fold_digest,
        "train_start": query.train_start,
        "train_stop": query.train_stop,
        "query_index": query.query_index,
        "query_timestamp_ns": query.query_timestamp_ns,
        "source_commit": query.source_commit,
        "query_digest": query.query_digest,
        "state_snapshot_digest": query.state_snapshot_digest,
        "observation_digest": query.observation_digest,
        "environment_digest": query.environment_digest,
        "action_spec_digest": query.action_spec_digest,
        "execution_policy_digest": query.execution_policy_digest,
        "risk_digest": query.risk_digest,
        "trend_digest": query.trend_digest,
        "starting_equity": query.starting_equity,
        "baseline_target": query.baseline_target,
    }


def test_scenario_set_rejects_malformed_values() -> None:
    base = scenario_set()
    kwargs = {
        "scenario_ids": base.scenario_ids,
        "probabilities": base.probabilities,
        "anchor_indices": base.anchor_indices,
        "distances": base.distances,
        "query_condition": base.query_condition,
        "anchor_conditions": base.anchor_conditions,
        "library_digest": base.library_digest,
    }
    with pytest.raises(ValueError, match="scenario_ids"):
        CausalScenarioSet(**{**kwargs, "scenario_ids": ("x", "x")})
    with pytest.raises(ValueError, match="probabilities"):
        CausalScenarioSet(**{**kwargs, "probabilities": np.full(64, 0.1)})
    with pytest.raises(ValueError, match="uniform"):
        probabilities = np.full(64, 1.0 / 64.0)
        probabilities[0] += 0.001
        probabilities[1] -= 0.001
        CausalScenarioSet(**{**kwargs, "probabilities": probabilities})
    with pytest.raises(ValueError, match="anchor_indices"):
        CausalScenarioSet(**{**kwargs, "anchor_indices": np.zeros(63)})
    with pytest.raises(ValueError, match="anchor_conditions"):
        CausalScenarioSet(**{**kwargs, "anchor_conditions": np.zeros((64, 3))})
    with pytest.raises(ValueError, match="library_digest"):
        CausalScenarioSet(**{**kwargs, "library_digest": "bad"})


def test_generate_residual_candidates_has_stable_mandatory_order() -> None:
    actions = generate_residual_candidates(
        np.asarray([0.3, -0.2, 0.0], dtype=np.float64),
        external_actions=(np.asarray([0.2, 0.0, -0.1]),),
        max_candidates=32,
    )
    np.testing.assert_array_equal(actions[0], np.zeros(3))
    np.testing.assert_array_equal(actions[1], np.asarray([-1.0, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[2], np.asarray([-0.5, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[3], np.asarray([0.5, 0.0, 0.0]))
    np.testing.assert_array_equal(actions[4], np.asarray([1.0, 0.0, 0.0]))
    assert any(np.array_equal(action, [-0.5, 0.5, 0.0]) for action in actions)
    assert any(np.array_equal(action, [0.2, 0.0, -0.1]) for action in actions)
    assert all(not action.flags.writeable for action in actions)


def test_generate_residual_candidates_deduplicates_and_normalizes_zero() -> None:
    actions = generate_residual_candidates(
        np.asarray([-0.0]),
        external_actions=(np.asarray([0.0]), np.asarray([-0.0])),
    )
    assert len(actions) == 5
    assert math.copysign(1.0, float(actions[0][0])) == 1.0


@pytest.mark.parametrize(
    "external",
    [
        (np.asarray([2.0]),),
        (np.asarray([math.nan]),),
        (np.asarray([0.0, 0.0]),),
        ([0.0],),
    ],
)
def test_generate_residual_candidates_rejects_invalid_external_actions(
    external: object,
) -> None:
    with pytest.raises(ValueError):
        generate_residual_candidates(np.asarray([0.0]), external_actions=external)  # type: ignore[arg-type]


def test_generate_residual_candidates_fails_instead_of_truncating() -> None:
    with pytest.raises(ValueError, match="max_candidates"):
        generate_residual_candidates(np.zeros(3), max_candidates=5)
