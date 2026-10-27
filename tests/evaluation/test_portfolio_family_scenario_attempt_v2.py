from __future__ import annotations

import json
import os
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from tests.evaluation.test_portfolio_family_result_artifact import _identity

pytest_plugins = ["tests.evaluation.test_portfolio_family_scenario_replay_v2"]


def _scenario_identity():
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        FourFamilyPortfolioScenarioAttemptIdentityV2,
    )
    from trade_rl.strategies.rl.portfolio_runtime_scenario import (
        PortfolioRuntimeExecutionScenarioV2,
    )

    common = _identity()
    scenario = PortfolioRuntimeExecutionScenarioV2(
        scenario_id="base-software-test",
        role="base",
        scenario_plan_sha256="9" * 64,
        training_context_sha256=common.expected_training_context_sha256,
        transform_digest="8" * 64,
        checkpoint_sha256=common.expected_checkpoint_sha256,
        base_execution_cost=common.execution_cost,
        runtime_execution_cost=common.execution_cost,
        changed_fields=(),
    )
    return FourFamilyPortfolioScenarioAttemptIdentityV2(
        common_input_declaration=common,
        expected_scenario=scenario,
        expected_scenario_sha256=scenario.digest,
        expected_scenario_inference_source_sha256="7" * 64,
        expected_durable_source_sha256="6" * 64,
    )


def test_v2_identity_rejects_common_execution_cost_outside_scenario() -> None:
    identity = _scenario_identity()
    changed_cost = replace(
        identity.common_input_declaration.execution_cost, fee_rate=0.02
    )
    changed_common = replace(
        identity.common_input_declaration,
        execution_cost=changed_cost,
    )

    with pytest.raises(ValueError, match="common input execution cost differs"):
        replace(identity, common_input_declaration=changed_common)


def test_v2_identity_has_a_separate_canonical_digest_vector() -> None:
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        four_family_portfolio_scenario_attempt_identity_v2_digest,
        four_family_portfolio_scenario_attempt_identity_v2_payload,
    )

    identity = _scenario_identity()
    payload = four_family_portfolio_scenario_attempt_identity_v2_payload(identity)

    assert (
        identity.schema_version == "four_family_portfolio_scenario_attempt_identity_v2"
    )
    assert payload["common_input_declaration"]["schema_version"] == (
        "four_family_portfolio_attempt_identity_v1"
    )
    assert four_family_portfolio_scenario_attempt_identity_v2_digest(identity) == (
        "2a33fcbd6afea218014ac3b367d658cc1202a5e04ecaadcb6d19f9b785968735"
    )
    assert not any(key.endswith("_root") for key in payload)


@pytest.fixture(scope="module")
def actual_scenario_attempt_case(
    scenario_replay_inputs: tuple[dict[str, Any], Any, Any, Any, dict[str, object]],
    tmp_path_factory: pytest.TempPathFactory,
) -> dict[str, Any]:
    from tests.evaluation.test_portfolio_family_scenario_replay_v2 import _scenario
    from trade_rl.evaluation.runs import (
        portfolio_family_scenario_attempt_v2 as attempt_module,
    )
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        FourFamilyPortfolioScenarioAttemptContextV2,
        load_four_family_portfolio_scenario_attempt_v2,
        run_four_family_portfolio_scenario_attempt_v2,
    )
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        FourFamilyPortfolioScenarioAttemptIdentityV2,
        capture_four_family_scenario_durable_source_digest_v2,
    )
    from trade_rl.evaluation.runs.provenance import build_candidate_run_provenance
    from trade_rl.strategies.rl.portfolio_scenario_runtime import (
        capture_portfolio_scenario_inference_source_digest,
    )

    inputs, training_result, dataset, training_env, training_context = (
        scenario_replay_inputs
    )
    scenario = _scenario(
        training_result,
        training_env,
        training_context,
        scenario_id="durable-v2-fee-latency-test",
        runtime_updates={"fee_rate": 0.001, "order_latency_bars": 2},
    )
    baseline = _identity()
    provenance = build_candidate_run_provenance(
        research_context_digest=baseline.research_context_digest
    )
    common = replace(
        baseline,
        expected_implementation_digest=provenance["implementation_digest"],
        expected_runtime_environment_digest=provenance["runtime_environment_digest"],
        expected_dataset_artifact_sha256=inputs["expected_dataset_artifact_sha256"],
        expected_symbols=inputs["expected_symbols"],
        ridge_artifact_sha256=inputs["expected_ridge_artifact_sha256"],
        expected_ridge_binding=inputs["expected_ridge_binding"],
        expected_bundle_sha256=inputs["expected_bundle_sha256"],
        expected_manifest_sha256=inputs["expected_manifest_sha256"],
        expected_completion_sha256=inputs["expected_completion_sha256"],
        expected_training_context_sha256=inputs["expected_training_context_sha256"],
        expected_checkpoint_sha256=inputs["expected_checkpoint_sha256"],
        bound=inputs["bound"],
        start_index=inputs["start_index"],
        stop_index=inputs["stop_index"],
        initial_capital=inputs["initial_capital"],
        target_abs_weights=inputs["target_abs_weights"],
        trend_config=inputs["trend_config"],
        ridge_entry_threshold=inputs["ridge_entry_threshold"],
        ridge_exit_threshold=inputs["ridge_exit_threshold"],
        ridge_one_way_switch_cost=inputs["ridge_one_way_switch_cost"],
        execution_cost=scenario.runtime_execution_cost,
        risk_config=inputs["risk_config"],
        market_order_profile=inputs["market_order_profile"],
    )
    identity = FourFamilyPortfolioScenarioAttemptIdentityV2(
        common_input_declaration=common,
        expected_scenario=scenario,
        expected_scenario_sha256=scenario.digest,
        expected_scenario_inference_source_sha256=(
            capture_portfolio_scenario_inference_source_digest()
        ),
        expected_durable_source_sha256=(
            capture_four_family_scenario_durable_source_digest_v2()
        ),
    )
    context = FourFamilyPortfolioScenarioAttemptContextV2(
        identity=identity,
        dataset_artifact_root=inputs["dataset_artifact_root"],
        ridge_artifact_root=inputs["ridge_artifact_root"],
        bundle_root=inputs["bundle_root"],
    )
    attempt_root = tmp_path_factory.mktemp("durable-scenario-attempt") / "attempt"
    callbacks: list[str] = []
    replay_results: list[Any] = []
    original_runner = attempt_module.run_four_family_shared_portfolio_scenario_replay_v2

    def observe_callback(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]

        def record(outcome: object) -> None:
            observer(outcome)  # type: ignore[operator]
            family = outcome.family  # type: ignore[attr-defined]
            assert (attempt_root / "families" / family / "record.json").is_file()
            callbacks.append(family)

        kwargs["on_family_completed"] = record
        result = original_runner(**kwargs)  # type: ignore[arg-type]
        replay_results.append(result)
        return result

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            attempt_module,
            "run_four_family_shared_portfolio_scenario_replay_v2",
            observe_callback,
        )
        published = run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=attempt_root,
            context=context,
        )
    loaded = load_four_family_portfolio_scenario_attempt_v2(
        attempt_root,
        expected_identity=identity,
        expected_result_digest=published.result_artifact_digest,
    )
    return {
        "callbacks": tuple(callbacks),
        "context": context,
        "dataset": dataset,
        "identity": identity,
        "loaded": loaded,
        "root": attempt_root,
        "replay_result": replay_results[0],
        "scenario": scenario,
    }


def test_actual_four_family_scenario_attempt_persists_and_reloads_v2_identity(
    actual_scenario_attempt_case: dict[str, Any],
) -> None:
    case = actual_scenario_attempt_case
    loaded = case["loaded"]
    payload = loaded.result.payload

    assert case["callbacks"] == ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    assert payload["identity"]["schema_version"] == (
        "four_family_portfolio_scenario_attempt_identity_v2"
    )
    assert payload["identity"]["common_input_declaration"]["schema_version"] == (
        "four_family_portfolio_attempt_identity_v1"
    )
    assert payload["scenario_replay"]["scenario_sha256"] == case["scenario"].digest
    assert (
        payload["scenario_replay"]["runtime_binding"]["runtime_execution_policy_digest"]
        == case["scenario"].runtime_execution_cost.execution_policy_digest
    )
    assert tuple(row["family"] for row in payload["families"]) == (
        "cash",
        "fixed_trend",
        "ridge_forecast",
        "joint_ppo",
    )
    trend_manifest = payload["families"][1]
    trend = json.loads((case["root"] / trend_manifest["path"]).read_bytes())
    assert trend["terminal_flat"] is True
    assert trend["no_active_orders"] is True
    assert trend["ledger"]["execution_policy_digest"] == (
        case["scenario"].runtime_execution_cost.execution_policy_digest
    )
    assert trend["ledger"]["terminal_exact_quantities"] == ["0", "0"]
    assert trend["ledger"]["active_order_remainders"] == []
    order_events = tuple(
        event
        for interval in trend["ledger"]["intervals"]
        for event in interval["order_events"]
    )
    assert any(event["event_type"] == "latency_wait" for event in order_events)
    assert any(event["event_type"] == "filled" for event in order_events)
    submitted_indexes = {
        event["order_id"]: event["processing_index"]
        for event in order_events
        if event["event_type"] == "submitted"
    }
    assert any(
        submitted_indexes.get(event["order_id"]) is not None
        and event["processing_index"] - submitted_indexes[event["order_id"]] == 2
        for event in order_events
        if event["event_type"] == "eligible"
    )
    assert (case["root"] / "complete" / "record.json").is_file()
    assert not (case["root"] / "failure" / "record.json").exists()


def _emit_captured_replay(case: dict[str, Any], **kwargs: Any) -> Any:
    replay_result = case["replay_result"]
    observer = kwargs["on_family_completed"]
    for outcome in replay_result.common_replay.families:
        observer(outcome)
    return replay_result


def test_v2_identity_is_portable_across_declared_root_relocation(
    actual_scenario_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        four_family_portfolio_scenario_attempt_identity_v2_digest,
    )

    case = actual_scenario_attempt_case
    identity = case["identity"]
    original = case["context"]
    relocated = replace(
        original,
        dataset_artifact_root=Path("C:/relocated/dataset"),
        ridge_artifact_root=Path("C:/relocated/ridge"),
        bundle_root=Path("C:/relocated/bundle"),
    )

    assert relocated.identity is identity
    assert four_family_portfolio_scenario_attempt_identity_v2_digest(
        relocated.identity
    ) == four_family_portfolio_scenario_attempt_identity_v2_digest(identity)


def test_result_validation_rejects_forged_training_execution_config_digest(
    actual_scenario_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        _validate_scenario_replay_metadata,
    )

    case = actual_scenario_attempt_case
    replay = json.loads(
        canonical_json_bytes(case["loaded"].result.payload["scenario_replay"])
    )
    expected = replay["runtime_binding"]["training_execution_config_digest"]
    forged = "f" * 64
    assert forged != expected
    replay["runtime_binding"]["training_execution_config_digest"] = forged

    with pytest.raises(
        ValueError,
        match="scenario runtime binding training_execution_config_digest differs",
    ):
        _validate_scenario_replay_metadata(replay, identity=case["identity"])


@pytest.mark.parametrize(
    ("field", "message"),
    (
        (
            "training_agent_stop_index",
            "training agent stop differs from base latency",
        ),
        ("runtime_agent_stop_index", "agent stop differs from runtime latency"),
        ("runtime_start_ns", "window differs from identity"),
    ),
)
def test_result_validation_rejects_forged_derived_runtime_window_fields(
    actual_scenario_attempt_case: dict[str, Any],
    field: str,
    message: str,
) -> None:
    from trade_rl.artifacts.canonical import canonical_json_bytes
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        _validate_scenario_replay_metadata,
    )

    case = actual_scenario_attempt_case
    replay = json.loads(
        canonical_json_bytes(case["loaded"].result.payload["scenario_replay"])
    )
    replay["runtime_binding"][field] += 1

    with pytest.raises(ValueError, match=message):
        _validate_scenario_replay_metadata(replay, identity=case["identity"])


def test_v1_result_loader_rejects_the_v2_attempt_wrapper(
    actual_scenario_attempt_case: dict[str, Any],
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_result_artifact import (
        load_four_family_portfolio_result_artifact,
    )

    case = actual_scenario_attempt_case
    with pytest.raises(ValueError, match="unexpected"):
        load_four_family_portfolio_result_artifact(
            case["root"],
            expected_digest=case["loaded"].result.artifact_digest,
            expected_identity=case["identity"].common_input_declaration,
        )


def test_source_pin_mismatch_stops_before_scenario_replay(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        FourFamilyPortfolioScenarioAttemptContextV2,
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    identity = replace(
        case["identity"], expected_scenario_inference_source_sha256="5" * 64
    )
    context = FourFamilyPortfolioScenarioAttemptContextV2(
        identity=identity,
        dataset_artifact_root=case["context"].dataset_artifact_root,
        ridge_artifact_root=case["context"].ridge_artifact_root,
        bundle_root=case["context"].bundle_root,
    )
    root = tmp_path / "wrong-scenario-source"
    replay_calls: list[bool] = []

    def forbidden_replay(**kwargs: object) -> object:
        replay_calls.append(True)
        raise AssertionError("source mismatch must stop before replay")

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        forbidden_replay,
    )

    with pytest.raises(ValueError, match="scenario inference source differs"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root,
            context=context,
        )

    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert replay_calls == []
    assert failure["phase"] == "before_provenance"
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()


def test_scenario_attempt_failure_keeps_first_row_and_original_exception(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "interrupted-after-cash"
    original_error = KeyboardInterrupt("synthetic v2 stop after cash")
    expected_cash = (case["root"] / "families" / "cash" / "record.json").read_bytes()

    def interrupt_after_first(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]

        def interrupt(outcome: object) -> None:
            observer(outcome)  # type: ignore[operator]
            raise original_error

        kwargs["on_family_completed"] = interrupt
        return _emit_captured_replay(case, **kwargs)

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        interrupt_after_first,
    )

    with pytest.raises(KeyboardInterrupt) as caught:
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root,
            context=case["context"],
        )

    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert caught.value is original_error
    assert failure["phase"] == "four_family_scenario_replay"
    assert failure["completed_families"] == ["cash"]
    assert (root / "families" / "cash" / "record.json").read_bytes() == expected_cash


def test_wrong_callback_family_order_fails_before_persisting_a_row(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "wrong-scenario-family-order"
    outcome = case["replay_result"].common_replay.families[1]

    def emit_wrong_first(**kwargs: object) -> object:
        kwargs["on_family_completed"](outcome)  # type: ignore[operator]
        raise AssertionError("wrong family order must stop replay")

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        emit_wrong_first,
    )

    with pytest.raises(ValueError, match="fixed family order"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "four_family_scenario_replay"
    assert failure["completed_families"] == []
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()


def test_returned_native_rows_must_be_the_exact_objects_observed_by_callbacks(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "callback-return-object-mismatch"
    replay_result = case["replay_result"]

    def return_equivalent_but_new_outcome_objects(**kwargs: object) -> object:
        observer = kwargs["on_family_completed"]
        outcomes = replay_result.common_replay.families
        for outcome in outcomes:
            observer(outcome)  # type: ignore[operator]
        changed_rows = (*outcomes[:-1], replace(outcomes[-1]))
        changed_replay = replace(replay_result.common_replay, families=changed_rows)
        return replace(replay_result, common_replay=changed_replay)

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        return_equivalent_but_new_outcome_objects,
    )

    with pytest.raises(ValueError, match="not observed by callback"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "four_family_scenario_replay"
    assert failure["completed_families"] == [
        "cash",
        "fixed_trend",
        "ridge_forecast",
        "joint_ppo",
    ]
    assert all(
        (root / "families" / family / "record.json").is_file()
        for family in failure["completed_families"]
    )
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()


def test_after_provenance_equality_guard_is_independent_of_digest_pins(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "scenario-provenance-canonical-drift"
    original_build = portfolio_family_scenario_attempt_v2.build_candidate_run_provenance
    calls = 0

    def mutate_only_runtime_payload(
        *, research_context_digest: str | None = None
    ) -> dict[str, object]:
        nonlocal calls
        calls += 1
        provenance = original_build(research_context_digest=research_context_digest)
        if calls == 1:
            return provenance
        runtime = provenance["runtime_environment"]
        assert isinstance(runtime, dict)
        return {
            **provenance,
            "runtime_environment": {**runtime, "unbound_change": "changed"},
        }

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        lambda **kwargs: _emit_captured_replay(case, **kwargs),
    )
    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "build_candidate_run_provenance",
        mutate_only_runtime_payload,
    )

    with pytest.raises(RuntimeError, match="provenance changed"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    assert calls == 2
    assert (root / "provenance" / "before" / "record.json").is_file()
    assert (root / "provenance" / "after" / "record.json").is_file()
    assert all(
        (root / "families" / family / "record.json").is_file()
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    )
    assert all(
        (root / "families" / family / "record.json").read_bytes()
        == (case["root"] / "families" / family / "record.json").read_bytes()
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    )
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "verify_provenance"


def test_result_publication_failure_retains_rows_without_completion(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import (
        portfolio_family_scenario_attempt_v2,
        portfolio_family_scenario_result_artifact_v2,
    )
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "scenario-result-publication-failure"
    original_publish = portfolio_family_scenario_result_artifact_v2._publish_record

    def reject_manifest(path: Path, record: object) -> bytes:
        if path == root / "result" / "manifest.json":
            raise OSError("synthetic scenario manifest publication failure")
        return original_publish(path, record)

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        lambda **kwargs: _emit_captured_replay(case, **kwargs),
    )
    monkeypatch.setattr(
        portfolio_family_scenario_result_artifact_v2,
        "_publish_record",
        reject_manifest,
    )

    with pytest.raises(OSError, match="scenario manifest publication failure"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    assert all(
        (root / "families" / family / "record.json").is_file()
        for family in ("cash", "fixed_trend", "ridge_forecast", "joint_ppo")
    )
    assert (root / "provenance" / "after" / "record.json").is_file()
    assert not (root / "result" / "manifest.json").exists()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "publish_result"


def test_strict_result_reload_failure_keeps_published_result_uncompleted(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "scenario-result-reload-failure"

    def reject_result_reload(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic scenario result reload failure")

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        lambda **kwargs: _emit_captured_replay(case, **kwargs),
    )
    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "load_four_family_portfolio_scenario_result_artifact_v2",
        reject_result_reload,
    )

    with pytest.raises(ValueError, match="scenario result reload failure"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    assert (root / "result" / "manifest.json").is_file()
    assert not (root / "complete" / "record.json").exists()
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "strict_result_reload"


def test_failed_final_attempt_reload_keeps_completion_and_loader_rejects_conflict(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        load_four_family_portfolio_scenario_attempt_v2,
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "scenario-final-reload-failure"

    def reject_attempt_reload(*args: object, **kwargs: object) -> object:
        raise ValueError("synthetic final scenario attempt reload failure")

    real_loader = load_four_family_portfolio_scenario_attempt_v2
    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        lambda **kwargs: _emit_captured_replay(case, **kwargs),
    )
    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "load_four_family_portfolio_scenario_attempt_v2",
        reject_attempt_reload,
    )

    with pytest.raises(ValueError, match="final scenario attempt reload failure"):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    complete = json.loads((root / "complete" / "record.json").read_text("utf-8"))
    failure = json.loads((root / "failure" / "record.json").read_text("utf-8"))
    assert failure["phase"] == "strict_attempt_reload"
    assert "completion_marker_note" in failure
    with pytest.raises(ValueError, match="failed scenario attempts"):
        real_loader(
            root,
            expected_identity=case["identity"],
            expected_result_digest=complete["result_digest"],
        )


def test_failure_journal_error_does_not_replace_primary_row_write_error(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case
    root = tmp_path / "scenario-failure-journal-error"
    original_publish = portfolio_family_scenario_attempt_v2._publish_record
    expected_cash = (case["root"] / "families" / "cash" / "record.json").read_bytes()

    def reject_row_and_journal(path: Path, record: object) -> bytes:
        if path.parent.name == "fixed_trend":
            raise OSError("primary scenario row write failure")
        if path.parent.name == "failure":
            raise OSError("secondary scenario failure journal error")
        return original_publish(path, record)

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "run_four_family_shared_portfolio_scenario_replay_v2",
        lambda **kwargs: _emit_captured_replay(case, **kwargs),
    )
    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "_publish_record",
        reject_row_and_journal,
    )

    with pytest.raises(OSError, match="primary scenario row write failure") as caught:
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=root, context=case["context"]
        )

    assert any(
        "secondary scenario failure journal error" in note
        for note in caught.value.__notes__
    )
    assert (root / "families" / "cash" / "record.json").read_bytes() == expected_cash
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "failure" / "record.json").exists()
    assert (root / "families" / "cash" / "record.json").is_file()
    assert not (root / "families" / "fixed_trend" / "record.json").exists()
    assert not (root / "complete" / "record.json").exists()


def test_scenario_result_loader_rejects_tampered_native_row(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        load_four_family_portfolio_scenario_result_artifact_v2,
    )

    case = actual_scenario_attempt_case
    copied = tmp_path / "tampered-copy"
    shutil.copytree(case["root"], copied)
    row_path = copied / "families" / "cash" / "record.json"
    row_path.write_bytes(row_path.read_bytes() + b" ")

    with pytest.raises(ValueError, match="differs from its manifest"):
        load_four_family_portfolio_scenario_result_artifact_v2(
            copied,
            expected_digest=case["loaded"].result.artifact_digest,
            expected_identity=case["identity"],
        )


@pytest.mark.parametrize("mutation", ("reversed", "duplicate"))
def test_scenario_result_family_manifest_requires_exact_roster_order(
    actual_scenario_attempt_case: dict[str, Any],
    mutation: str,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        _validate_family_rows,
    )

    case = actual_scenario_attempt_case
    manifest = json.loads((case["root"] / "result" / "manifest.json").read_bytes())
    rows = manifest["payload"]["families"]
    if mutation == "reversed":
        rows.reverse()
    else:
        rows[1] = rows[0]

    with pytest.raises(ValueError, match="family order or path differs"):
        _validate_family_rows(case["root"], rows, identity=case["identity"])


@pytest.mark.parametrize(
    ("mutation", "error_type", "message"),
    (
        ("missing-family", FileNotFoundError, None),
        ("extra-file", ValueError, "missing or unexpected result files"),
        ("extra-directory", ValueError, "unexpected directory"),
        ("symlink", ValueError, "symlink"),
    ),
)
def test_scenario_result_loader_rejects_invalid_attempt_inventory(
    actual_scenario_attempt_case: dict[str, Any],
    tmp_path: Path,
    mutation: str,
    error_type: type[Exception],
    message: str | None,
) -> None:
    from trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 import (
        load_four_family_portfolio_scenario_result_artifact_v2,
    )

    case = actual_scenario_attempt_case
    copied = tmp_path / f"scenario-inventory-{mutation}"
    shutil.copytree(case["root"], copied)
    if mutation == "missing-family":
        (copied / "families" / "ridge_forecast" / "record.json").unlink()
    elif mutation == "extra-file":
        (copied / "unexpected.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "extra-directory":
        (copied / "unexpected").mkdir()
    else:
        try:
            os.symlink(copied / "started" / "record.json", copied / "unexpected-link")
        except OSError as error:
            pytest.skip(f"symlink creation is unavailable on this platform: {error}")

    with pytest.raises(error_type, match=message):
        load_four_family_portfolio_scenario_result_artifact_v2(
            copied,
            expected_digest=case["loaded"].result.artifact_digest,
            expected_identity=case["identity"],
        )


def test_existing_scenario_attempt_root_is_rejected_before_provenance(
    actual_scenario_attempt_case: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.runs import portfolio_family_scenario_attempt_v2
    from trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 import (
        run_four_family_portfolio_scenario_attempt_v2,
    )

    case = actual_scenario_attempt_case

    def must_not_build_provenance(**kwargs: object) -> object:
        raise AssertionError("existing root was not rejected before preflight")

    monkeypatch.setattr(
        portfolio_family_scenario_attempt_v2,
        "build_candidate_run_provenance",
        must_not_build_provenance,
    )
    with pytest.raises(FileExistsError):
        run_four_family_portfolio_scenario_attempt_v2(
            attempt_root=case["root"], context=case["context"]
        )
