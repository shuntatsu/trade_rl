from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from trade_rl.artifacts import canonical_json_bytes


def _load() -> ModuleType:
    path = Path(".github/scripts/issue663_normalization_execution.py")
    spec = importlib.util.spec_from_file_location(
        "issue663_normalization_execution", path
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Issue 663 execution helper spec unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _protocol() -> dict[str, object]:
    return {
        "schema": "issue663_normalization_protocol_v2",
        "candidate_seeds": [0, 1, 2, 3, 4],
        "source_dataset_id": "d" * 64,
        "source_artifact_digest": "a" * 64,
        "control_run_id": 35287338444,
        "terminal_flatness_semantics": "issue645_reporting_quantity_abs_le_1e-10",
        "unused_data_accessed": False,
        "final_test_accessed": False,
        "production_eligible": False,
        "live_trading_authorized": False,
    }


def test_activation_and_seed_claims_are_canonical_and_remote_identity_bound() -> None:
    module = _load()
    protocol = _protocol()
    activation = module.build_activation_claim(
        protocol,
        workflow_run_id=123,
        workflow_run_attempt=1,
        implementation_head="a" * 40,
    )
    assert activation["schema_version"] == "issue663_normalization_activation_v2"
    assert activation["issue_number"] == 663
    assert activation["workflow_run_id"] == 123
    assert activation["workflow_run_attempt"] == 1
    assert activation["implementation_head"] == "a" * 40
    assert activation["local_output_root_is_authority"] is False
    assert activation["economic_execution_started"] is False
    assert activation["economic_result_inspected"] is False
    assert activation["unused_data_accessed"] is False
    assert activation["final_test_accessed"] is False
    assert module.claim_bytes(activation) == canonical_json_bytes(activation)

    seed_claim = module.build_seed_claim(
        protocol,
        seed=2,
        workflow_run_id=123,
        workflow_run_attempt=1,
        implementation_head="a" * 40,
        activation_artifact_id=77,
        activation_artifact_digest="sha256:" + "b" * 64,
        control_artifact_id=88,
        control_artifact_digest="sha256:" + "c" * 64,
    )
    assert seed_claim["schema_version"] == "issue663_normalization_seed_claim_v2"
    assert seed_claim["seed"] == 2
    assert seed_claim["workflow_run_id"] == 123
    assert seed_claim["activation_artifact_id"] == 77
    assert seed_claim["activation_artifact_digest"] == "sha256:" + "b" * 64
    assert seed_claim["control_artifact_id"] == 88
    assert seed_claim["control_artifact_digest"] == "sha256:" + "c" * 64
    assert seed_claim["economic_execution_started"] is False
    assert seed_claim["candidate_result_inspected"] is False
    assert module.claim_bytes(seed_claim) == canonical_json_bytes(seed_claim)


def test_remote_slot_state_machine_closes_crash_window() -> None:
    module = _load()
    name = module.seed_claim_name(0)
    module.validate_remote_slot(
        {"total_count": 0, "artifacts": []},
        artifact_name=name,
        state="empty",
    )
    claimed = {
        "total_count": 1,
        "artifacts": [
            {
                "id": 7,
                "name": name,
                "expired": False,
                "workflow_run": {"id": 123},
            }
        ],
    }
    with pytest.raises(RuntimeError, match="already claimed"):
        module.validate_remote_slot(claimed, artifact_name=name, state="empty")
    artifact = module.validate_remote_slot(
        claimed,
        artifact_name=name,
        state="claimed",
        workflow_run_id=123,
    )
    assert artifact["id"] == 7
    with pytest.raises(RuntimeError, match="workflow"):
        module.validate_remote_slot(
            claimed,
            artifact_name=name,
            state="claimed",
            workflow_run_id=999,
        )


def test_seed_claim_rejects_seed_runtime_or_identity_drift() -> None:
    module = _load()
    protocol = _protocol()
    common = {
        "workflow_run_id": 123,
        "workflow_run_attempt": 1,
        "implementation_head": "a" * 40,
        "activation_artifact_id": 77,
        "activation_artifact_digest": "sha256:" + "b" * 64,
        "control_artifact_id": 88,
        "control_artifact_digest": "sha256:" + "c" * 64,
    }
    with pytest.raises(ValueError, match="seed"):
        module.build_seed_claim(protocol, seed=5, **common)
    with pytest.raises(ValueError, match="attempt"):
        module.build_seed_claim(
            protocol,
            seed=0,
            **{**common, "workflow_run_attempt": 2},
        )
    with pytest.raises(ValueError, match="implementation_head"):
        module.build_seed_claim(
            protocol,
            seed=0,
            **{**common, "implementation_head": "not-a-sha"},
        )


def test_candidate_seed_changes_only_feature_normalization(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    module = _load()
    timestamps = np.arange(20, dtype="timedelta64[h]") + np.datetime64("2022-12-31T20")
    dataset = SimpleNamespace(
        dataset_id="d" * 64,
        timestamps=timestamps,
        symbols=("BTCUSDT", "ETHUSDT"),
        feature_names=("f0", "f1"),
    )
    config = SimpleNamespace(
        feature_indices=(0, 1),
        fit_symbol_indices=(0, 1),
        fit_cutoff="2023-01-01T00:00:00",
    )
    calls: dict[str, object] = {}

    monkeypatch.setattr(module, "load_market_dataset_artifact", lambda path: dataset)
    monkeypatch.setattr(module, "with_price_channels", lambda value: value)
    monkeypatch.setattr(
        module,
        "inspect_study",
        lambda path: SimpleNamespace(plan=SimpleNamespace(baseline_config=config)),
    )
    monkeypatch.setattr(module, "development_indices", lambda value: (4, 12))

    policy = object()
    normalizer = object()
    strategy = SimpleNamespace(policy=policy, feature_normalizer=normalizer)

    def fit(value: object, **kwargs: object) -> object:
        calls["fit_dataset"] = value
        calls["fit_kwargs"] = kwargs
        return strategy

    monkeypatch.setattr(module, "fit_ppo_strategy", fit)
    monkeypatch.setattr(module, "save_normalized_ppo", lambda root, value: "m" * 64)

    def evaluate(value: object, factory: object, **kwargs: object) -> dict[str, object]:
        calls["evaluate_dataset"] = value
        calls["evaluate_kwargs"] = kwargs
        produced = factory()
        calls["factory_strategy"] = produced
        return {
            "schema": "directional_arm_v1",
            "dataset_id": "d" * 64,
            "returns": [0.0, 0.0],
            "metrics": {"total_return": 0.0},
            "year_returns": {"2023": 0.0, "2024": 0.0},
            "ledger_max_drawdown": 0.0,
            "termination_reasons": [],
            "terminal_flat": True,
            "terminal_quantities": [0.0, 0.0],
            "qualified": False,
            "production_eligible": False,
        }

    monkeypatch.setattr(module, "evaluate_directional_arm", evaluate)
    monkeypatch.setattr(
        module,
        "PPOIntentStrategy",
        lambda policy, *, feature_indices, feature_normalizer: (
            policy,
            feature_indices,
            feature_normalizer,
        ),
    )

    result = module.run_candidate_seed(
        tmp_path / "source",
        tmp_path / "output",
        seed=3,
    )
    fit_kwargs = calls["fit_kwargs"]
    assert fit_kwargs["feature_indices"] == (0, 1)
    assert fit_kwargs["fit_symbol_indices"] == (0, 1)
    assert fit_kwargs["start_index"] == 0
    assert fit_kwargs["stop_index"] == 3
    assert fit_kwargs["gross_budget"] == 0.1
    assert fit_kwargs["total_timesteps"] == 262_144
    assert fit_kwargs["seed"] == 3
    assert fit_kwargs["initial_capital"] == 10_000.0
    assert fit_kwargs["training_layout"] == "sequential"
    assert fit_kwargs["rollout_steps_per_env"] is None
    assert fit_kwargs["normalize_features"] is True
    assert fit_kwargs["execution_cost"].processing_bar_volume_capacity is False
    assert result["seed"] == 3
    assert result["model_bundle_digest"] == "m" * 64
    assert result["economic_result_interpreted"] is False
    assert result["unused_data_accessed"] is False
    assert result["final_test_accessed"] is False
    assert result["production_eligible"] is False
    assert result["live_trading_authorized"] is False


def test_result_blind_messages_do_not_include_economics() -> None:
    module = _load()
    message = module.seed_result_written_message(4)
    assert message == "seed4: result bytes written; interpretation deferred"
    assert "return" not in message.lower()
    assert "qualified" not in message.lower()
