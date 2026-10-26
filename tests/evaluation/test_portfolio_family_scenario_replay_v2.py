from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Any

import numpy as np
import pytest

from trade_rl.artifacts.canonical import canonical_json_bytes
from trade_rl.data.artifacts.publication import publish_market_dataset_artifact
from trade_rl.simulation.objectives.binding import BoundObjectiveClock
from trade_rl.simulation.objectives.clock import FinancialClockContract
from trade_rl.simulation.objectives.contract import CapitalContract, ObjectiveContract
from trade_rl.strategies.forecasts.ridge import RidgeForecastModel
from trade_rl.strategies.forecasts.ridge_artifact import (
    RidgeForecastArtifactBinding,
    publish_ridge_forecast_artifact,
)
from trade_rl.strategies.rl.portfolio_runtime_scenario import (
    PortfolioRuntimeExecutionScenarioV2,
)
from trade_rl.strategies.rules.trend import TrendIntentConfig

pytest_plugins = ["tests.strategies.rl.test_portfolio_inference_bundle"]

_START = datetime(2026, 1, 1, tzinfo=UTC)
_FAMILY_ORDER = ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")


def _runtime_bound(start_index: int, stop_index: int) -> BoundObjectiveClock:
    return BoundObjectiveClock(
        ObjectiveContract(
            capital=CapitalContract("shared_portfolio", "USDT", (1_000.0,)),
            evaluation_start=_START + timedelta(hours=start_index),
            evaluation_stop_exclusive=_START + timedelta(hours=stop_index),
            terminal_valuation="settled",
            economics_digest="b" * 64,
            risk_digest="c" * 64,
            deployment_recipe_digest="d" * 64,
            maximum_drawdown=0.20,
        ),
        FinancialClockContract(
            decision_interval_seconds=3_600,
            execution_interval_seconds=3_600,
            reward_interval_seconds=3_600,
            economic_horizon_seconds=(stop_index - start_index) * 3_600,
            rollout_steps=4,
            gamma=1.0,
            gae_lambda=0.95,
            reward_schema="equity_delta_v1",
        ),
    )


@pytest.fixture(scope="module")
def scenario_replay_inputs(
    scenario_published_bundle: tuple[Any, Any, Any, Any, dict[str, object]],
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[dict[str, Any], Any, Any, Any, dict[str, object]]:
    result, dataset, training_env, publication, training_context = (
        scenario_published_bundle
    )
    root = tmp_path_factory.mktemp("four-family-scenario-inputs")
    dataset_artifact = publish_market_dataset_artifact(root / "dataset", dataset)
    model = RidgeForecastModel(
        feature_indices=(0,),
        feature_mean=np.asarray([0.0], dtype=np.float64),
        feature_scale=np.asarray([1.0], dtype=np.float64),
        coefficients=np.asarray([0.01], dtype=np.float64),
        intercept=0.0,
        horizon_hours=1,
        alpha=1.0,
        n_samples=2,
        fit_cutoff=dataset.timestamps[1],
    )
    binding = RidgeForecastArtifactBinding(
        dataset_id=dataset.dataset_id,
        dataset_schema_version=dataset_artifact.schema_version,
        dataset_artifact_sha256=dataset_artifact.artifact_digest,
        fit_symbol_indices=tuple(range(dataset.n_symbols)),
        fit_symbol_names=tuple(dataset.symbols),
        feature_names=tuple(dataset.feature_names),
        feature_indices=model.feature_indices,
        fit_cutoff_ns=int(
            np.asarray(model.fit_cutoff).astype("datetime64[ns]").astype(np.int64)
        ),
        horizon_hours=model.horizon_hours,
        alpha_hex=model.alpha.hex(),
        n_samples=model.n_samples,
        vintage_id="scenario-replay-software-test",
        source_commit="1" * 40,
        implementation_sha256="2" * 64,
        runtime_sha256="3" * 64,
    )
    ridge_root = root / "ridge"
    ridge_digest = publish_ridge_forecast_artifact(ridge_root, model, binding)
    start_index = 5
    stop_index = 11
    inputs: dict[str, Any] = {
        "dataset_artifact_root": dataset_artifact.root,
        "expected_dataset_artifact_sha256": dataset_artifact.artifact_digest,
        "expected_symbols": tuple(dataset.symbols),
        "ridge_artifact_root": ridge_root,
        "expected_ridge_artifact_sha256": ridge_digest,
        "expected_ridge_binding": binding,
        "bundle_root": publication.root,
        "expected_bundle_sha256": publication.bundle_sha256,
        "expected_manifest_sha256": publication.training_manifest_sha256,
        "expected_completion_sha256": publication.training_completion_sha256,
        "expected_training_context_sha256": publication.training_context_sha256,
        "expected_checkpoint_sha256": publication.checkpoint_sha256,
        "bound": _runtime_bound(start_index, stop_index),
        "start_index": start_index,
        "stop_index": stop_index,
        "initial_capital": 1_000.0,
        "target_abs_weights": tuple(training_env.target_abs_weights),
        "trend_config": TrendIntentConfig(
            signal_index=0,
            entry_threshold=0.5,
            exit_threshold=0.1,
        ),
        "ridge_entry_threshold": 0.005,
        "ridge_exit_threshold": 0.001,
        "ridge_one_way_switch_cost": None,
        "risk_config": training_env.risk_config,
        "market_order_profile": training_env.market_order_profile,
    }
    return inputs, result, dataset, training_env, training_context


def _scenario(
    result: Any,
    training_env: Any,
    training_context: dict[str, object],
    *,
    scenario_id: str,
    runtime_updates: dict[str, object],
) -> PortfolioRuntimeExecutionScenarioV2:
    base_cost = training_env.execution_cost
    runtime_cost = replace(base_cost, **runtime_updates)
    changed_fields = tuple(sorted(runtime_updates))
    return PortfolioRuntimeExecutionScenarioV2(
        scenario_id=scenario_id,
        role="base" if not changed_fields else "diagnostic_stress",
        scenario_plan_sha256="e" * 64,
        training_context_sha256=result.training_context_sha256,
        transform_digest=str(training_context["transform_digest"]),
        checkpoint_sha256=result.final_checkpoint_sha256,
        base_execution_cost=base_cost,
        runtime_execution_cost=runtime_cost,
        changed_fields=changed_fields,
    )


def _run_scenario_replay(
    inputs: dict[str, Any],
    result: Any,
    training_env: Any,
    training_context: dict[str, object],
    *,
    scenario_id: str,
    runtime_updates: dict[str, object],
    **kwargs: object,
) -> Any:
    from trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 import (
        run_four_family_shared_portfolio_scenario_replay_v2,
    )
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        capture_portfolio_scenario_inference_source_digest,
    )

    scenario = _scenario(
        result,
        training_env,
        training_context,
        scenario_id=scenario_id,
        runtime_updates=runtime_updates,
    )
    return run_four_family_shared_portfolio_scenario_replay_v2(
        **inputs,
        expected_scenario=scenario,
        expected_scenario_sha256=scenario.digest,
        expected_scenario_inference_source_sha256=(
            capture_portfolio_scenario_inference_source_digest()
        ),
        **kwargs,
    )


@pytest.mark.parametrize(
    ("scenario_id", "runtime_updates", "changed_fields"),
    [
        pytest.param("base-software-test", {}, (), id="base"),
        pytest.param(
            "fee-stress-software-test",
            {"fee_rate": 0.001},
            ("fee_rate",),
            id="fee",
        ),
        pytest.param(
            "latency-stress-software-test",
            {"order_latency_bars": 2},
            ("order_latency_bars",),
            id="latency",
        ),
        pytest.param(
            "combined-stress-software-test",
            {"fee_rate": 0.001, "order_latency_bars": 2},
            ("fee_rate", "order_latency_bars"),
            id="combined",
        ),
    ],
)
def test_scenario_replay_runs_all_four_fresh_native_arms(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
    scenario_id: str,
    runtime_updates: dict[str, object],
    changed_fields: tuple[str, ...],
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_comparison as core_module
    from trade_rl.simulation.stateful import portfolio_session as session_module
    from trade_rl.strategies.rl.portfolio_scenario_inference import (
        PortfolioScenarioInferencePolicy,
    )

    inputs, result, dataset, training_env, training_context = scenario_replay_inputs
    scenario = _scenario(
        result,
        training_env,
        training_context,
        scenario_id=scenario_id,
        runtime_updates=runtime_updates,
    )
    session_records: list[Any] = []
    replay_calls: list[tuple[object, object, dict[str, Any]]] = []
    pending_order_windows: list[tuple[int, int]] = []
    completed: list[object] = []
    ppo_decision_indexes: list[int] = []
    execution_order: list[tuple[str, str]] = []
    native_runner = core_module.run_shared_portfolio_replay
    native_session_init = session_module.SharedPortfolioSession.__init__
    native_advance = session_module.SharedPortfolioSession.advance
    inference_decide = PortfolioScenarioInferencePolicy.decide_portfolio

    def record_session_init(
        self: object, loaded_dataset: object, *args: object, **kwargs: object
    ) -> None:
        native_session_init(self, loaded_dataset, *args, **kwargs)
        session_records.append(self)

    def record_replay(
        loaded_dataset: object, policy: object, **kwargs: object
    ) -> object:
        replay_calls.append((loaded_dataset, policy, dict(kwargs)))
        execution_order.append(("replay", _FAMILY_ORDER[len(replay_calls) - 1]))
        return native_runner(loaded_dataset, policy, **kwargs)

    def record_advance(self: Any, target_weights: object, **kwargs: object) -> object:
        outcome = native_advance(self, target_weights, **kwargs)
        pending_order_windows.extend(
            (order.intent.submit_index, order.intent.eligible_index)
            for order in self.active_orders
        )
        return outcome

    def record_ppo_decision(self: object, snapshot: object) -> object:
        ppo_decision_indexes.append(snapshot.index)
        return inference_decide(self, snapshot)

    def observe_completion(outcome: Any) -> None:
        completed.append(outcome)
        execution_order.append(("callback", outcome.family))

    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "__init__", record_session_init
    )
    monkeypatch.setattr(
        session_module.SharedPortfolioSession, "advance", record_advance
    )
    monkeypatch.setattr(core_module, "run_shared_portfolio_replay", record_replay)
    monkeypatch.setattr(
        PortfolioScenarioInferencePolicy, "decide_portfolio", record_ppo_decision
    )

    replay_result = _run_scenario_replay(
        inputs,
        result,
        training_env,
        training_context,
        scenario_id=scenario_id,
        runtime_updates=runtime_updates,
        on_family_completed=observe_completion,
    )

    assert tuple(item.family for item in replay_result.common_replay.families) == (
        _FAMILY_ORDER
    )
    assert tuple(item.family for item in completed) == _FAMILY_ORDER
    assert execution_order == [
        pair
        for family in _FAMILY_ORDER
        for pair in (("replay", family), ("callback", family))
    ]
    assert all(
        observed is outcome
        for observed, outcome in zip(
            completed, replay_result.common_replay.families, strict=True
        )
    )
    assert replay_result.scenario == scenario
    assert replay_result.scenario_sha256 == scenario.digest
    assert replay_result.scenario_inference_source_sha256 == (
        replay_result.runtime_binding.scenario_inference_source_sha256
    )
    assert replay_result.training_context_sha256 == result.training_context_sha256
    assert replay_result.transform_digest == training_context["transform_digest"]
    assert replay_result.checkpoint_sha256 == result.final_checkpoint_sha256
    assert replay_result.base_execution_policy_digest == (
        scenario.base_execution_cost.execution_policy_digest
    )
    assert replay_result.runtime_execution_policy_digest == (
        scenario.runtime_execution_cost.execution_policy_digest
    )
    assert replay_result.runtime_binding.dataset_id == dataset.dataset_id
    assert replay_result.training_bound_digest == training_context["bound_digest"]
    assert replay_result.runtime_bound_digest == inputs["bound"].digest
    assert replay_result.runtime_binding.runtime_start_index == inputs["start_index"]
    assert replay_result.runtime_binding.runtime_stop_index == inputs["stop_index"]
    assert len(replay_calls) == 4
    assert len(session_records) == 4
    assert len({id(policy) for _, policy, _ in replay_calls}) == 4
    assert len({id(session) for session in session_records}) == 4
    assert len({id(kwargs["risk"]) for _, _, kwargs in replay_calls}) == 4
    assert all(
        loaded_dataset is replay_calls[0][0] for loaded_dataset, _, _ in replay_calls
    )
    assert all(
        kwargs["execution_cost"] == scenario.runtime_execution_cost
        for _, _, kwargs in replay_calls
    )
    assert len({id(kwargs["execution_cost"]) for _, _, kwargs in replay_calls}) == 1
    assert all(
        kwargs["risk"].config == inputs["risk_config"]
        and kwargs["initial_capital"] == inputs["initial_capital"]
        and kwargs["start_index"] == inputs["start_index"]
        and kwargs["stop_index"] == inputs["stop_index"]
        and kwargs["market_order_profile"] == inputs["market_order_profile"]
        and kwargs["settle_terminal_position"] is True
        for _, _, kwargs in replay_calls
    )
    assert all(
        session.agent_stop_index
        == inputs["stop_index"] - scenario.runtime_execution_cost.order_latency_bars - 1
        for session in session_records
    )
    assert isinstance(replay_calls[-1][1], PortfolioScenarioInferencePolicy)
    assert ppo_decision_indexes == [
        decision.index
        for decision in replay_result.common_replay.families[-1].replay.decisions
    ]
    assert all(item.window_complete for item in replay_result.common_replay.families)
    for outcome in replay_result.common_replay.families:
        evidence = outcome.replay.ledger_evidence
        assert evidence is not None
        assert evidence.intervals.execution_policy_digest == (
            scenario.runtime_execution_cost.execution_policy_digest
        )
        assert evidence.intervals.terminal_exact_quantities == ("0", "0")
        assert evidence.intervals.active_order_remainders == ()

    if "order_latency_bars" in changed_fields:
        trend_events = tuple(
            event
            for interval in replay_result.common_replay.families[
                1
            ].replay.ledger_evidence.intervals.intervals
            for event in interval.order_events
        )
        assert any(event.event_type == "latency_wait" for event in trend_events)
        assert any(event.event_type == "filled" for event in trend_events)
        assert any(
            eligible_index - submit_index == 2
            for submit_index, eligible_index in pending_order_windows
        )


def test_base_scenario_matches_strict_v1_deterministic_ledgers(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )

    inputs, result, _, training_env, training_context = scenario_replay_inputs
    scenario_result = _run_scenario_replay(
        inputs,
        result,
        training_env,
        training_context,
        scenario_id="base-equivalence-software-test",
        runtime_updates={},
    )
    strict_result = run_four_family_shared_portfolio_replay(
        **inputs,
        execution_cost=training_env.execution_cost,
    )

    assert tuple(item.family for item in strict_result.families) == _FAMILY_ORDER
    for strict_outcome, scenario_outcome in zip(
        strict_result.families,
        scenario_result.common_replay.families,
        strict=True,
    ):
        strict_ledger = strict_outcome.replay.ledger_evidence
        scenario_ledger = scenario_outcome.replay.ledger_evidence
        assert strict_ledger is not None
        assert scenario_ledger is not None
        assert canonical_json_bytes(strict_ledger.to_mapping()) == (
            canonical_json_bytes(scenario_ledger.to_mapping())
        )


def test_scenario_source_mismatch_precedes_checkpoint_and_any_arm(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_comparison as core_module
    from trade_rl.strategies.rl import portfolio_scenario_inference as inference_module

    inputs, result, _, training_env, training_context = scenario_replay_inputs
    scenario = _scenario(
        result,
        training_env,
        training_context,
        scenario_id="source-mismatch-software-test",
        runtime_updates={},
    )

    def forbidden_checkpoint_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("scenario source mismatch must precede checkpoint load")

    def forbidden_native_replay(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("scenario source mismatch must precede every arm")

    monkeypatch.setattr(
        inference_module, "_load_verified_bundle_policy", forbidden_checkpoint_load
    )
    monkeypatch.setattr(
        core_module, "run_shared_portfolio_replay", forbidden_native_replay
    )

    from trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 import (
        run_four_family_shared_portfolio_scenario_replay_v2,
    )

    with pytest.raises(ValueError, match="current scenario inference source differs"):
        run_four_family_shared_portfolio_scenario_replay_v2(
            **inputs,
            expected_scenario=scenario,
            expected_scenario_sha256=scenario.digest,
            expected_scenario_inference_source_sha256="0" * 64,
        )


def test_scenario_digest_mismatch_precedes_dataset_and_any_arm(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_comparison as core_module
    from trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 import (
        run_four_family_shared_portfolio_scenario_replay_v2,
    )
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        capture_portfolio_scenario_inference_source_digest,
    )

    inputs, result, _, training_env, training_context = scenario_replay_inputs
    scenario = _scenario(
        result,
        training_env,
        training_context,
        scenario_id="digest-mismatch-software-test",
        runtime_updates={},
    )

    def forbidden_dataset_load(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise AssertionError("scenario digest mismatch must precede Dataset loading")

    monkeypatch.setattr(
        core_module, "load_published_market_dataset_artifact", forbidden_dataset_load
    )
    with pytest.raises(ValueError, match="scenario digest differs"):
        run_four_family_shared_portfolio_scenario_replay_v2(
            **inputs,
            expected_scenario=scenario,
            expected_scenario_sha256="0" * 64,
            expected_scenario_inference_source_sha256=(
                capture_portfolio_scenario_inference_source_digest()
            ),
        )


def test_completion_failure_stops_before_starting_next_family(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_comparison as core_module

    inputs, result, _, training_env, training_context = scenario_replay_inputs
    native_runner = core_module.run_shared_portfolio_replay
    native_runs: list[str] = []
    callbacks: list[str] = []

    def observe_replay(
        loaded_dataset: object, policy: object, **kwargs: object
    ) -> object:
        native_runs.append(_FAMILY_ORDER[len(native_runs)])
        return native_runner(loaded_dataset, policy, **kwargs)

    def fail_second_completion(outcome: Any) -> None:
        callbacks.append(outcome.family)
        if outcome.family == "fixed_trend":
            raise RuntimeError("completion observer sentinel")

    monkeypatch.setattr(core_module, "run_shared_portfolio_replay", observe_replay)
    with pytest.raises(RuntimeError, match="completion observer sentinel"):
        _run_scenario_replay(
            inputs,
            result,
            training_env,
            training_context,
            scenario_id="callback-failure-software-test",
            runtime_updates={},
            on_family_completed=fail_second_completion,
        )
    assert callbacks == ["cash", "fixed_trend"]
    assert native_runs == ["cash", "fixed_trend"]


def test_v1_still_rejects_changed_cost_before_checkpoint_or_any_arm(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_comparison as core_module
    from trade_rl.evaluation.runs.portfolio_family_comparison import (
        run_four_family_shared_portfolio_replay,
    )
    from trade_rl.strategies.rl import portfolio_inference_bundle as bundle_module

    inputs, _, _, training_env, _ = scenario_replay_inputs
    checkpoint_loads = 0
    native_replays = 0

    def forbidden_checkpoint_load(*args: object, **kwargs: object) -> object:
        nonlocal checkpoint_loads
        checkpoint_loads += 1
        raise AssertionError("v1 cost mismatch must precede PPO deserialization")

    def forbidden_native_replay(*args: object, **kwargs: object) -> object:
        nonlocal native_replays
        native_replays += 1
        raise AssertionError("v1 cost mismatch must precede every arm")

    monkeypatch.setattr(
        bundle_module, "_load_verified_bundle_policy", forbidden_checkpoint_load
    )
    monkeypatch.setattr(
        core_module, "run_shared_portfolio_replay", forbidden_native_replay
    )

    with pytest.raises(ValueError, match="runtime financial or target-weight context"):
        run_four_family_shared_portfolio_replay(
            **inputs,
            execution_cost=replace(training_env.execution_cost, fee_rate=0.001),
        )
    assert checkpoint_loads == 0
    assert native_replays == 0
