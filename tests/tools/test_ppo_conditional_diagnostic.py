from __future__ import annotations

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

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[object, object]:
        assert deterministic is True
        return np.asarray(2 if observation[0] > 0.5 else 1), None


class FlatPolicy:
    num_timesteps = 8192

    def predict(
        self, observation: np.ndarray, *, deterministic: bool = True
    ) -> tuple[object, object]:
        del observation
        assert deterministic is True
        return np.asarray(1), None


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

    def fake_fit(dataset: object, **kwargs: object) -> SimpleNamespace:
        assert (output / "attempt-start.json").is_file()
        assert not (output / "result.json").exists()
        assert not (output / "failure.json").exists()
        assert _read_canonical(output / "attempt-start.json")["fit_call_budget"] == 1
        fit_calls.append((dataset, kwargs))
        return SimpleNamespace(policy=FeaturePolicy())

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
        "after_heldout": True,
    }
    assert result["failed_clauses"] == []
    assert result["scope"] == "synthetic_software_diagnostic_only"
    assert result["heldout_metrics"]["feature_one_first_intent"] == "LONG"
    assert result["heldout_metrics"]["feature_zero_first_intent"] == "FLAT"
    assert _read_canonical(output / "result.json")["failed_clauses"] == []

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
        return SimpleNamespace(policy=FlatPolicy())

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
        return SimpleNamespace(policy=FeaturePolicy())

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
        return SimpleNamespace(policy=FeaturePolicy())

    def fake_heldout(
        policy: object,
        dataset: object,
    ) -> tuple[dict[str, bool], dict[str, Any]]:
        nonlocal evaluation_finished
        del policy, dataset
        evaluation_finished = True
        return {"synthetic_clause": True}, {"synthetic_metric": 1.0}

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
        "after_heldout": False,
    }
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
