from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

import trade_rl
from tools import ppo_conditional_diagnostic as diagnostic

TEST_RUNTIME = {
    "gymnasium": "test",
    "numpy": "test",
    "python": "test",
    "stable-baselines3": "test",
    "torch": "test",
}


class FeaturePolicy:
    num_timesteps = 8192
    observation_space = SimpleNamespace(shape=(6,))
    action_space = SimpleNamespace(n=3, start=0)

    def __init__(self, call_order: list[str] | None = None) -> None:
        self.save_calls = 0
        self.call_order = call_order
        self.observations: list[np.ndarray] = []

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[object, object]:
        assert deterministic is True
        self.observations.append(observation.copy())
        if self.call_order is not None:
            self.call_order.append("predict")
        return np.asarray(2 if observation[0] > 0.5 else 1), None

    def save(self, path: str) -> None:
        self.save_calls += 1
        if self.call_order is not None:
            self.call_order.append("save")
        Path(path).write_bytes(b"synthetic-fitted-policy")


class FlatPolicy:
    num_timesteps = 8192
    observation_space = SimpleNamespace(shape=(6,))
    action_space = SimpleNamespace(n=3, start=0)

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[object, object]:
        del observation
        assert deterministic is True
        return np.asarray(1), None

    def save(self, path: str) -> None:
        Path(path).write_bytes(b"synthetic-flat-policy")


def _fitted_strategy(policy: object) -> SimpleNamespace:
    return SimpleNamespace(
        policy=policy,
        feature_indices=(0,),
        feature_names=("synthetic_regime_label",),
        feature_normalizer=None,
        observation_schema=diagnostic.PPO_OBSERVATION_SCHEMA_V3,
        minimum_hold_bars=0,
    )


def _read_canonical(path: Path) -> dict[str, Any]:
    raw = path.read_bytes()
    value = json.loads(raw)
    assert diagnostic.canonical_json_bytes(value) == raw
    assert isinstance(value, dict)
    return value


def _install_runtime(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic, "_runtime_versions", lambda: TEST_RUNTIME.copy())


def test_fixture_is_fixed_causal_label_with_swapped_renamed_heldout() -> None:
    train = diagnostic.build_fixture_dataset()
    heldout = diagnostic.build_fixture_dataset(heldout=True)

    assert train.n_bars == heldout.n_bars == 64
    assert train.symbols == ("train_0", "train_1")
    assert heldout.symbols == ("heldout_0", "heldout_1")
    assert train.features[0, :, 0].tolist() == [1.0, 0.0]
    assert heldout.features[0, :, 0].tolist() == [0.0, 1.0]
    assert np.all(train.feature_available)
    assert np.all(heldout.feature_available)
    assert np.all(train.resolved_array("feature_staleness") == 0.0)
    assert np.all(heldout.resolved_array("feature_staleness") == 0.0)
    assert np.all(train.global_features == 0.0)
    assert np.all(heldout.global_features == 0.0)

    train_a = train.close[:, 0]
    heldout_a = heldout.close[:, 1]
    assert np.all(train_a[:8] == 100.0)
    assert train_a[15] == pytest.approx(100.0 * 1.01**8)
    assert np.all(train_a[15:] == pytest.approx(train_a[15]))
    assert np.all(heldout_a[:12] == 100.0)
    assert heldout_a[19] == pytest.approx(100.0 * 1.01**8)
    assert np.all(heldout_a[19:] == pytest.approx(heldout_a[19]))
    assert train.identity_verified
    assert heldout.identity_verified
    assert train.available_at is not None
    assert heldout.available_at is not None
    assert np.all(train.available_at <= train.timestamps[:, None])
    assert np.all(heldout.available_at <= heldout.timestamps[:, None])
    cost = diagnostic.fixed_execution_cost()
    assert cost.fee_rate == 0.005
    assert cost.maker_fee_rate == 0.0
    assert cost.taker_fee_rate == 0.0
    assert cost.spread_rate == 0.0
    assert cost.impact_rate == 0.0
    assert cost.slippage_std == 0.0
    assert cost.tail_slippage_probability == 0.0
    assert cost.tail_slippage_multiplier == 0.0
    assert cost.max_participation_rate == 1.0
    assert cost.borrow_rate_multiplier == 0.0
    assert cost.order_latency_bars == 1
    assert cost.maintenance_margin_rate == 0.25
    assert cost.collateral_haircut == 1.0
    assert cost.max_leverage == 1.0


def test_prefit_oracle_matches_delayed_gain_flat_and_fee_only_ledgers() -> None:
    oracle = diagnostic.run_prefit_oracle()

    assert oracle["passed"] is True
    assert oracle["failed_checks"] == []
    checks = oracle["checks"]
    assert checks["delayed_long_cash_and_fees_match_reference"]
    assert checks["flat_control_has_no_turnover_or_fees"]
    assert checks["fee_only_round_trip_loses_both_fees"]
    assert checks["entry_fee_precedes_first_positive_return"]
    assert checks["terminal_positions_and_orders_are_flat"]

    metrics = oracle["metrics"]
    assert metrics["delayed_long_final_cash"] > 1_000.0
    assert metrics["flat_final_cash"] == pytest.approx(1_000.0)
    assert metrics["fee_only_round_trip_final_cash"] == pytest.approx(995.0)
    assert metrics["fee_only_round_trip_total_cost"] == pytest.approx(5.0)


def test_prefit_oracle_rejects_price_rows_not_available_at_row_time() -> None:
    dataset = diagnostic.build_fixture_dataset()
    assert dataset.available_at is not None
    available_at = dataset.available_at.copy()
    available_at[-1, 0] = dataset.timestamps[-1] + np.timedelta64(1, "h")
    delayed = replace(
        dataset,
        dataset_id="0" * 64,
        identity_payload_json=None,
        information_available=None,
        available_at=available_at,
    )

    assert diagnostic._fixture_inputs_are_valid(dataset)
    assert not diagnostic._fixture_inputs_are_valid(delayed)


def test_evidence_snapshots_detach_mutable_sources_and_reject_unknown_values() -> None:
    dataset = diagnostic.build_fixture_dataset(heldout=True)
    env = diagnostic._make_environment(
        dataset,
        symbol_index=1,
        execution_observations=[],
    )
    env.reset(seed=diagnostic.FIXED_SEED)
    book_snapshot = diagnostic._book_state_snapshot(env.book, index=env.index)
    book_bytes = diagnostic.canonical_json_bytes(book_snapshot)
    env.book.quantities[1] = 0.25
    env.book.returns_history.append(0.01)
    assert diagnostic.canonical_json_bytes(book_snapshot) == book_bytes

    source_event = {"events": [{"quantity": 1.0}]}
    event_snapshot = diagnostic._canonical_object_snapshot(
        source_event,
        field="execution",
    )
    event_bytes = diagnostic.canonical_json_bytes(event_snapshot)
    source_event["events"][0]["quantity"] = 2.0
    assert diagnostic.canonical_json_bytes(event_snapshot) == event_bytes
    assert event_snapshot["events"][0]["quantity"] == 1.0

    with pytest.raises(TypeError, match="unsupported canonical JSON value"):
        diagnostic._canonical_object_snapshot(
            {"unsupported": object()},
            field="execution",
        )


def test_source_manifest_covers_tracked_execution_and_risk_python_sources() -> None:
    manifest = diagnostic._source_manifest()

    assert "tools/ppo_conditional_diagnostic.py" in manifest
    assert "pyproject.toml" in manifest
    assert "uv.lock" in manifest
    assert "trade_rl/data/market.py" in manifest
    assert "trade_rl/risk/pretrade.py" in manifest
    assert "trade_rl/simulation/execution.py" in manifest
    assert "trade_rl/simulation/stateful/execution.py" in manifest
    assert "trade_rl/strategies/rl/ppo.py" in manifest
    assert "trade_rl/strategies/rl/ppo_training.py" in manifest


@pytest.mark.parametrize(
    ("module_attribute", "relative_path"),
    [
        ("trade_rl", "trade_rl/__init__.py"),
        ("ppo_module", "trade_rl/strategies/rl/ppo.py"),
    ],
)
def test_rejects_loaded_ppo_sources_from_another_checkout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module_attribute: str,
    relative_path: str,
) -> None:
    _install_runtime(monkeypatch)
    payload = diagnostic._protocol_payload(
        prefit_oracle={"passed": True}, runtime_versions=TEST_RUNTIME
    )
    protocol = {
        **payload,
        "protocol_digest": diagnostic.content_digest(payload),
    }

    alternate_file = tmp_path / "alternate-checkout" / relative_path
    alternate_file.parent.mkdir(parents=True)
    alternate_file.write_text("# alternate checkout\n", encoding="utf-8")
    loaded_module = (
        trade_rl if module_attribute == "trade_rl" else diagnostic.ppo_module
    )
    monkeypatch.setattr(loaded_module, "__file__", str(alternate_file))

    with pytest.raises(RuntimeError, match="loaded PPO source origin"):
        diagnostic._assert_source_identity(protocol, stage="pre-fit validation")


def test_prepare_writes_canonical_recipe_without_fitting_and_never_overwrites(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        del args, kwargs
        pytest.fail("prepare must not fit a policy")

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", forbidden_fit)
    output = tmp_path / "conditional-diagnostic"
    prepared = diagnostic.prepare(output)

    assert prepared["status"] == "prepared"
    protocol = _read_canonical(output / "protocol.json")
    prefit = _read_canonical(output / "prefit-oracle.json")
    assert protocol["recipe"]["training"]["total_timesteps"] == 8192
    assert protocol["recipe"]["training"]["rollout_steps_per_env"] == 512
    assert protocol["recipe"]["training"]["seed"] == 83
    assert protocol["runtime_versions"] == TEST_RUNTIME
    assert protocol["recipe"]["costs"]["maintenance_margin_rate"] == 0.25
    assert protocol["recipe"]["effective_default_risk_config"]["max_abs_weight"] == 1.0
    assert protocol["ppo_defaults"]["gamma"] == 0.99
    assert protocol["ppo_defaults"]["gae_lambda"] == 0.95
    assert protocol["ppo_defaults"]["network"]["net_arch"] == {
        "pi": [64, 64],
        "vf": [64, 64],
    }
    assert protocol["scope"]["research_authorization"] is False
    assert protocol["scope"]["p3_status"] == "NOT_ATTEMPTED"
    assert prefit["passed"] is True
    assert not (output / "attempt-start.json").exists()
    assert not (output / "result.json").exists()

    with pytest.raises(FileExistsError):
        diagnostic.prepare(output)


def test_run_uses_one_frozen_fit_and_public_v3_observation_for_heldout(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls: list[tuple[object, dict[str, object]]] = []
    fitted_policies: list[FeaturePolicy] = []
    call_order: list[str] = []

    def fake_fit(dataset: object, **kwargs: object) -> SimpleNamespace:
        assert (output / "attempt-start.json").is_file()
        assert not (output / "result.json").exists()
        assert not (output / "failure.json").exists()
        assert _read_canonical(output / "attempt-start.json")["fit_call_budget"] == 1
        call_order.append("fit")
        fit_calls.append((dataset, kwargs))
        policy = FeaturePolicy(call_order)
        fitted_policies.append(policy)
        return _fitted_strategy(policy)

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    result = diagnostic.run_diagnostic(output)

    assert len(fit_calls) == 1
    _, fit_kwargs = fit_calls[0]
    assert fit_kwargs == {
        "feature_indices": (0,),
        "fit_symbol_indices": (0, 1),
        "start_index": 0,
        "stop_index": 63,
        "training_layout": "interleaved",
        "rollout_steps_per_env": 512,
        "total_timesteps": 8192,
        "seed": 83,
        "gross_budget": 0.5,
        "initial_capital": 1_000.0,
        "normalize_features": False,
        "minimum_hold_bars": 0,
        "observation_schema": "ppo_observation_v3",
        "settle_terminal_position": True,
        "execution_cost": diagnostic.fixed_execution_cost(),
    }
    assert result["status"] == "completed"
    assert result["fit_call_count"] == 1
    assert result["diagnostic_verdict"] == "pass"
    assert result["p3_status"] == "NOT_ATTEMPTED"
    assert result["source_identity_checks"] == {
        "before_fit": True,
        "after_fit": True,
        "after_bundle": True,
        "after_heldout": True,
    }
    assert result["failed_clauses"] == []
    assert result["scope"] == "synthetic_software_diagnostic_only"
    assert result["schema"] == "ppo_conditional_diagnostic_result_v2"
    assert result["evidence_contract_version"] == 2
    assert result["heldout_metrics"]["feature_one_first_intent"] == "LONG"
    assert result["heldout_metrics"]["feature_zero_first_intent"] == "FLAT"
    assert _read_canonical(output / "result.json")["failed_clauses"] == []
    evidence_manifest = _read_canonical(output / "evidence-manifest.json")
    trace = _read_canonical(output / "heldout-trace.json")
    bundle_manifest = _read_canonical(output / "inference-bundle" / "manifest.json")
    result_readback = _read_canonical(output / "result.json")
    assert result_readback["evidence_manifest_digest"] == diagnostic.content_digest(
        evidence_manifest
    )
    assert (
        result_readback["evidence_manifest_sha256"]
        == hashlib.sha256((output / "evidence-manifest.json").read_bytes()).hexdigest()
    )
    assert evidence_manifest["policy_bundle"]["manifest_digest"] == (
        diagnostic.content_digest(bundle_manifest)
    )
    policy_bytes = (output / "inference-bundle" / "policy.zip").read_bytes()
    assert (
        evidence_manifest["policy_bundle"]["policy_sha256"]
        == hashlib.sha256(policy_bytes).hexdigest()
    )
    assert evidence_manifest["policy_bundle"]["policy_size_bytes"] == len(policy_bytes)
    assert evidence_manifest["trace"]["digest"] == diagnostic.content_digest(trace)
    trace_bytes = (output / "heldout-trace.json").read_bytes()
    assert (
        evidence_manifest["trace"]["sha256"] == hashlib.sha256(trace_bytes).hexdigest()
    )
    assert evidence_manifest["trace"]["size_bytes"] == len(trace_bytes)
    assert result_readback["heldout_trace_digest"] == diagnostic.content_digest(trace)
    assert fitted_policies[0].save_calls == 1
    assert call_order[0:2] == ["fit", "save"]
    assert call_order.index("save") < call_order.index("predict")
    assert len(trace["episodes"]) == 2
    assert {episode["role"] for episode in trace["episodes"]} == {
        "feature_one",
        "feature_zero",
    }
    observation_offset = 0
    for episode in trace["episodes"]:
        steps = episode["steps"]
        assert steps
        assert [step["decision_index"] for step in steps] == sorted(
            step["decision_index"] for step in steps
        )
        assert steps[0]["pre_action_observation"]["dtype"] == np.dtype(np.float32).str
        assert steps[0]["pre_action_observation"]["shape"] == [6]
        for step in steps:
            saved_observation = step["pre_action_observation"]
            reconstructed = np.asarray(
                saved_observation["values"],
                dtype=np.dtype(saved_observation["dtype"]),
            ).reshape(saved_observation["shape"])
            np.testing.assert_array_equal(
                reconstructed,
                fitted_policies[0].observations[observation_offset],
            )
            observation_offset += 1
        earlier_state = steps[0]["post_step_book"]
        final_state = steps[-1]["post_step_book"]
        assert earlier_state["index"] == steps[0]["resulting_index"]
        assert final_state["index"] == steps[-1]["resulting_index"]
        assert earlier_state["index"] < final_state["index"]
        for step in steps:
            terminal_intervals = int(
                step["step_info"].get("terminal_settlement_intervals", 0)
            )
            calls = step["execution_calls"]
            assert len(calls) == 1 + terminal_intervals
            assert calls[0]["phase"] == "policy_step_execution"
            assert all(call["phase"] == "terminal_settlement" for call in calls[1:])
    assert any(
        call["phase"] == "terminal_settlement"
        for episode in trace["episodes"]
        for step in episode["steps"]
        for call in step["execution_calls"]
    )
    assert observation_offset == len(fitted_policies[0].observations)

    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(output)
    assert len(fit_calls) == 1


def test_learned_policy_failures_are_all_recorded_without_retraining(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls = 0

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return _fitted_strategy(FlatPolicy())

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    result = diagnostic.run_diagnostic(output)

    assert fit_calls == 1
    assert result["status"] == "completed"
    assert result["diagnostic_verdict"] == "fail"
    assert {
        "feature_one_initial_long",
        "feature_one_filled_before_heldout_move",
        "feature_one_exposure_through_heldout_move",
        "feature_one_terminal_cash_above_initial",
    }.issubset(set(result["failed_clauses"]))
    assert _read_canonical(output / "result.json")["diagnostic_verdict"] == "fail"


def test_fit_exception_is_recorded_and_the_attempt_cannot_be_retried(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls = 0

    def broken_fit(*args: object, **kwargs: object) -> None:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        assert (output / "attempt-start.json").is_file()
        raise RuntimeError("injected fit failure")

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", broken_fit)
    result = diagnostic.run_diagnostic(output)

    assert fit_calls == 1
    assert result["status"] == "execution_failed"
    failure = _read_canonical(output / "failure.json")
    assert failure["fit_call_count"] == 1
    assert failure["exception_type"] == "RuntimeError"
    assert not (output / "result.json").exists()
    with pytest.raises(FileExistsError):
        diagnostic.run_diagnostic(output)
    assert fit_calls == 1


def test_failure_before_fit_records_zero_invocations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    real_build = diagnostic.build_fixture_dataset
    build_calls = 0
    fit_calls = 0

    def broken_build(*, heldout: bool = False) -> Any:
        nonlocal build_calls
        build_calls += 1
        if build_calls == 3:
            raise RuntimeError("injected fixture rebuild failure")
        return real_build(heldout=heldout)

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        pytest.fail("fit must not run after the fixture rebuild fails")

    monkeypatch.setattr(diagnostic, "build_fixture_dataset", broken_build)
    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", forbidden_fit)
    result = diagnostic.run_diagnostic(output)

    assert fit_calls == 0
    assert result["status"] == "execution_failed"
    failure = _read_canonical(output / "failure.json")
    attempt = _read_canonical(output / "attempt-start.json")
    assert failure["fit_call_count"] == 0
    assert attempt["fit_call_budget"] == 1
    assert "fit_call_count" not in attempt


def test_pre_fit_source_drift_records_failure_without_invoking_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    original_manifest = diagnostic._source_manifest
    manifest_calls = 0
    fit_calls = 0

    def drifting_manifest() -> dict[str, str]:
        nonlocal manifest_calls
        manifest_calls += 1
        current = original_manifest()
        if manifest_calls >= 2:
            current["trade_rl/simulation/execution.py"] = "f" * 64
        return current

    def forbidden_fit(*args: object, **kwargs: object) -> None:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        pytest.fail("source drift before fit must stop the fit")

    monkeypatch.setattr(diagnostic, "_source_manifest", drifting_manifest)
    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", forbidden_fit)
    result = diagnostic.run_diagnostic(output)

    assert fit_calls == 0
    assert result["status"] == "execution_failed"
    failure = _read_canonical(output / "failure.json")
    assert failure["fit_call_count"] == 0
    assert failure["source_identity_stage"] == "before_fit"
    assert failure["source_identity_checks"]["before_fit"] is False
    assert not (output / "result.json").exists()


def test_post_fit_source_drift_is_failure_not_a_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    original_manifest = diagnostic._source_manifest
    fit_finished = False
    fit_calls = 0

    def drifting_manifest() -> dict[str, str]:
        current = original_manifest()
        if fit_finished:
            current["trade_rl/simulation/execution.py"] = "e" * 64
        return current

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal fit_finished, fit_calls
        del args, kwargs
        fit_calls += 1
        fit_finished = True
        return _fitted_strategy(FeaturePolicy())

    monkeypatch.setattr(diagnostic, "_source_manifest", drifting_manifest)
    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    result = diagnostic.run_diagnostic(output)

    assert fit_calls == 1
    assert result["status"] == "execution_failed"
    failure = _read_canonical(output / "failure.json")
    assert failure["source_identity_stage"] == "after_fit"
    assert failure["source_identity_checks"] == {
        "before_fit": True,
        "after_fit": False,
        "after_bundle": False,
        "after_heldout": False,
    }
    assert not (output / "result.json").exists()


def test_heldout_source_drift_is_failure_not_a_completed_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    original_manifest = diagnostic._source_manifest
    evaluation_finished = False

    def drifting_manifest() -> dict[str, str]:
        current = original_manifest()
        if evaluation_finished:
            current["trade_rl/simulation/execution.py"] = "d" * 64
        return current

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        del args, kwargs
        return _fitted_strategy(FeaturePolicy())

    def fake_heldout(
        policy: object,
        dataset: object,
    ) -> tuple[dict[str, bool], dict[str, Any], dict[str, Any]]:
        nonlocal evaluation_finished
        del policy, dataset
        evaluation_finished = True
        return (
            {"synthetic_clause": True},
            {"synthetic_metric": 1.0},
            {"episodes": []},
        )

    monkeypatch.setattr(diagnostic, "_source_manifest", drifting_manifest)
    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    monkeypatch.setattr(diagnostic, "_heldout_clauses", fake_heldout)
    result = diagnostic.run_diagnostic(output)

    assert result["status"] == "execution_failed"
    failure = _read_canonical(output / "failure.json")
    assert failure["source_identity_stage"] == "after_heldout"
    assert failure["source_identity_checks"] == {
        "before_fit": True,
        "after_fit": True,
        "after_bundle": True,
        "after_heldout": False,
    }
    assert not (output / "result.json").exists()


def test_bundle_save_failure_preserves_one_fit_failure_record(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls = 0

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return _fitted_strategy(FeaturePolicy())

    def fail_save(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise OSError("injected bundle save failure")

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    monkeypatch.setattr(diagnostic, "save_ppo_inference_bundle", fail_save)
    result = diagnostic.run_diagnostic(output)

    failure = _read_canonical(output / "failure.json")
    assert fit_calls == 1
    assert result["status"] == "execution_failed"
    assert failure["execution_phase"] == "policy_bundle"
    assert failure["fit_call_count"] == 1
    assert failure["artifact_statuses"]["inference_bundle"] == "failed"
    assert not (output / "result.json").exists()


def test_trace_write_failure_keeps_verified_bundle_and_trace_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls = 0
    original_write = diagnostic._write_exclusive_json

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return _fitted_strategy(FeaturePolicy())

    def write_then_fail(path: Path, value: object) -> None:
        original_write(path, value)
        if path.name == "heldout-trace.json":
            raise OSError("injected post-write trace failure")

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    monkeypatch.setattr(diagnostic, "_write_exclusive_json", write_then_fail)
    result = diagnostic.run_diagnostic(output)

    failure = _read_canonical(output / "failure.json")
    trace_bytes = (output / "heldout-trace.json").read_bytes()
    assert fit_calls == 1
    assert result["status"] == "execution_failed"
    assert failure["execution_phase"] == "heldout_trace"
    assert failure["fit_call_count"] == 1
    assert failure["artifact_statuses"]["inference_bundle"] == "verified"
    assert failure["artifact_statuses"]["heldout_trace"] == "failed"
    assert failure["artifact_files"]["heldout-trace.json"]["sha256"] == (
        hashlib.sha256(trace_bytes).hexdigest()
    )
    assert not (output / "result.json").exists()


def test_late_source_drift_keeps_evidence_but_never_publishes_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_runtime(monkeypatch)
    output = tmp_path / "conditional-diagnostic"
    diagnostic.prepare(output)
    fit_calls = 0
    original_assert_source = diagnostic._assert_source_identity

    def fake_fit(*args: object, **kwargs: object) -> SimpleNamespace:
        nonlocal fit_calls
        del args, kwargs
        fit_calls += 1
        return _fitted_strategy(FeaturePolicy())

    def drift_after_heldout(protocol: dict[str, Any], *, stage: str) -> None:
        if stage == "after_heldout":
            raise RuntimeError("injected late source drift")
        original_assert_source(protocol, stage=stage)

    monkeypatch.setattr(diagnostic, "fit_ppo_strategy", fake_fit)
    monkeypatch.setattr(diagnostic, "_assert_source_identity", drift_after_heldout)
    result = diagnostic.run_diagnostic(output)

    failure = _read_canonical(output / "failure.json")
    assert fit_calls == 1
    assert result["status"] == "execution_failed"
    assert failure["execution_phase"] == "after_heldout"
    assert failure["fit_call_count"] == 1
    assert failure["artifact_statuses"]["inference_bundle"] == "verified"
    assert failure["artifact_statuses"]["heldout_trace"] == "verified"
    assert failure["artifact_statuses"]["evidence_manifest"] == "not_started"
    assert "evidence-manifest.json" not in failure["artifact_files"]
    assert not (output / "result.json").exists()


def test_completed_diagnostic_failure_has_distinct_cli_exit_code(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        diagnostic,
        "run_diagnostic",
        lambda output: {
            "status": "completed",
            "diagnostic_verdict": "fail",
            "failed_clauses": ["feature_one_initial_long"],
        },
    )

    assert diagnostic.main(["run", "unused-output-dir"]) == 3
    assert '"diagnostic_verdict": "fail"' in capsys.readouterr().out
