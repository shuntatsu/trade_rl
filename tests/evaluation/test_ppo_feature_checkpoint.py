from __future__ import annotations

import copy
import gc
import hashlib
import json
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

import trade_rl.evaluation.ppo_feature_checkpoint as checkpoint
import trade_rl.evaluation.ppo_feature_study as legacy
from tests.evaluation import test_ppo_feature_study as legacy_fixtures
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.evaluation.experiments import ResolvedRunConfig


class _FakePolicy:
    def __init__(self, timesteps: object = 262_144) -> None:
        self.num_timesteps = timesteps

    def predict(self, _observation: object, *, deterministic: bool) -> tuple[int, None]:
        assert deterministic is True
        return 1, None


class _FakeStrategy:
    def __init__(
        self, timesteps: object = 262_144, feature_indices: tuple[int, ...] = (0,)
    ) -> None:
        self.policy = _FakePolicy(timesteps)
        self.feature_indices = feature_indices
        self.feature_names = tuple(("x", "relative")[i] for i in feature_indices)
        self.feature_normalizer = None


class _FakeDataset:
    def __init__(
        self,
        dataset_id: str,
        symbols: tuple[str, ...],
        feature_names: tuple[str, ...],
    ) -> None:
        self.dataset_id = dataset_id
        self.symbols = symbols
        self.n_symbols = len(symbols)
        self.feature_names = feature_names


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


@pytest.fixture
def checkpoint_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    protocol = copy.deepcopy(legacy_fixtures._PROTOCOL)
    protocol["seeds"] = [0]
    protocol["source_dataset_id"] = "frozen-source-dataset"
    protocol["admission"].update(
        required_passing_seeds_per_symbol=1,
        required_symbols=1,
        required_positive_paired_seed_deltas=1,
    )
    snapshot = b"synthetic frozen source snapshot"
    protocol["source_snapshot_sha256"] = _sha256(snapshot)
    source = tmp_path / "source"
    source.mkdir()
    root = tmp_path / "checkpoint-study"
    symbols = tuple(protocol["symbols"])
    dataset = SimpleNamespace(
        dataset_id=protocol["evaluation_dataset_id"],
        symbols=symbols,
        n_symbols=len(symbols),
        feature_names=("x", "relative"),
    )
    base_config = ResolvedRunConfig(
        signal_name="signal",
        signal_index=0,
        feature_names=("x",),
        feature_indices=(0,),
        fit_symbol_names=symbols,
        fit_symbol_indices=tuple(range(len(symbols))),
        fit_cutoff="2022-12-31T00:00:00.000000000",
        rule_entry_threshold=0.1,
        rule_exit_threshold=0.02,
        forecast_entry_threshold=0.01,
        forecast_exit_threshold=0.002,
        ppo_total_timesteps=262_144,
        ppo_seed=0,
        evaluation_start="2023-01-01T00:00:00.000000000",
        evaluation_stop_exclusive="2025-01-01T00:00:00.000000000",
        gross_budget=0.5,
        initial_capital=100_000.0,
        execution_overlay="zero_overlay_dataset_fields_authoritative",
    )
    monkeypatch.setattr(
        checkpoint.legacy,
        "expected_protocol",
        lambda _source: copy.deepcopy(protocol),
    )
    monkeypatch.setattr(
        checkpoint.legacy, "_source_snapshot_bytes", lambda _provenance: snapshot
    )

    def verify_snapshot(study_root: Path, expected: dict[str, Any]) -> None:
        assert (study_root / "source-snapshot.zip").read_bytes() == snapshot
        assert expected["source_snapshot_sha256"] == _sha256(snapshot)

    monkeypatch.setattr(checkpoint.legacy, "_verify_source_snapshot", verify_snapshot)
    monkeypatch.setattr(checkpoint, "_require_current_protocol", lambda *_args: None)
    monkeypatch.setattr(
        checkpoint,
        "_load_context",
        lambda _source, _protocol: (dataset, base_config),
    )

    fit_calls: list[tuple[str, Path]] = []
    evaluator_calls: list[tuple[str, int]] = []
    loader_calls: list[Path] = []
    state: dict[str, Any] = {
        "actual_timesteps": 262_144,
        "fit_calls": fit_calls,
        "evaluator_calls": evaluator_calls,
        "loader_calls": loader_calls,
    }

    def fake_fit(
        arm: str,
        _dataset: object,
        config: ResolvedRunConfig,
        output: Path,
        **_kwargs: object,
    ) -> object:
        fit_calls.append((arm, output))
        (output / "model.zip").write_bytes(b"training export: " + arm.encode())
        return lambda: _FakeStrategy(
            state["actual_timesteps"], tuple(config.feature_indices)
        )

    def fake_save_bundle(
        bundle_root: Path,
        strategy: _FakeStrategy,
        *,
        feature_names: tuple[str, ...],
    ) -> str:
        bundle_root.mkdir()
        policy = b"inference export"
        (bundle_root / "policy.zip").write_bytes(policy)
        selected_names = tuple(
            feature_names[index] for index in strategy.feature_indices
        )
        manifest = {
            "schema": "ppo_inference_bundle_v1",
            "feature_indices": list(strategy.feature_indices),
            "feature_names": list(selected_names),
            "policy_sha256": _sha256(policy),
        }
        (bundle_root / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        return content_digest(manifest)

    def fake_load_bundle(
        bundle_root: Path,
        *,
        expected_digest: str,
        feature_names: tuple[str, ...],
    ) -> _FakeStrategy:
        loader_calls.append(bundle_root)
        manifest = json.loads((bundle_root / "manifest.json").read_bytes())
        assert content_digest(manifest) == expected_digest
        selected_names = tuple(
            feature_names[index] for index in manifest["feature_indices"]
        )
        assert tuple(manifest["feature_names"]) == selected_names
        assert (
            _sha256((bundle_root / "policy.zip").read_bytes())
            == manifest["policy_sha256"]
        )
        return _FakeStrategy(262_144, tuple(manifest["feature_indices"]))

    def fake_evaluate(
        _dataset: object,
        _factory: object,
        **kwargs: object,
    ) -> dict[str, Any]:
        if kwargs["latency_bars"] == 1:
            scenario = "latency_1"
        elif kwargs["cost_multiplier"] == 2.0:
            scenario = "cost_2x"
        else:
            scenario = "base"
        index = int(kwargs["symbol_index"])
        evaluator_calls.append((scenario, index))
        row = legacy_fixtures._replay(
            index,
            (0.01, 0.01, 0.01, 0.01),
            scenario_name=scenario,
        )
        row["scenario"] = scenario
        row["ledger_evidence"] = legacy_fixtures._ledger_payload(row)
        row.pop("ledger_trace_summary")
        return row

    monkeypatch.setattr(checkpoint, "fit_directional_candidate", fake_fit)
    monkeypatch.setattr(checkpoint, "save_ppo_inference_bundle", fake_save_bundle)
    monkeypatch.setattr(checkpoint, "load_ppo_inference_bundle", fake_load_bundle)
    monkeypatch.setattr(checkpoint, "evaluate_directional_arm", fake_evaluate)
    return {
        "source": source,
        "root": root,
        "protocol": protocol,
        "dataset": dataset,
        "base_config": base_config,
        "state": state,
    }


def _prepare(env: dict[str, Any]) -> None:
    checkpoint.prepare_checkpoint_study(env["source"], env["root"])


def test_replay_factory_preserves_verified_feature_names(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    evaluate = checkpoint.evaluate_directional_arm

    def verify_factory(dataset: Any, factory: Any, **kwargs: Any) -> dict[str, Any]:
        first, second = factory(), factory()
        assert first.feature_names == second.feature_names == ("x",)
        assert first.feature_indices == second.feature_indices == (0,)
        assert first is not second
        assert first.policy is second.policy
        return evaluate(dataset, factory, **kwargs)

    monkeypatch.setattr(checkpoint, "evaluate_directional_arm", verify_factory)
    checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)


def test_prepare_source_drift_keeps_staging_non_authoritative(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env

    def reject_drift(*_args: object) -> None:
        raise ValueError("source/data/runtime drift blocks checkpoint publication")

    monkeypatch.setattr(checkpoint, "_require_current_protocol", reject_drift)
    with pytest.raises(ValueError, match="source/data/runtime"):
        _prepare(env)

    assert not env["root"].exists()
    assert list(env["root"].parent.glob(".checkpoint-staging-*"))


def test_fit_releases_resolved_dataset_before_streamed_drift_check(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env
    _prepare(env)
    dataset = _FakeDataset(
        dataset_id=env["protocol"]["evaluation_dataset_id"],
        symbols=tuple(env["protocol"]["symbols"]),
        feature_names=("x", "relative"),
    )
    dataset_ref = weakref.ref(dataset)
    context = [(dataset, env["base_config"])]
    monkeypatch.setattr(
        checkpoint,
        "_load_context",
        lambda _source, _protocol: context.pop(),
    )

    def assert_released(*_args: object) -> None:
        gc.collect()
        assert dataset_ref() is None

    monkeypatch.setattr(checkpoint, "_require_current_protocol", assert_released)
    del dataset
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)

    assert dataset_ref() is None


def test_replay_releases_dataset_and_policy_before_ledger_persistence(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    dataset_refs: list[weakref.ReferenceType[_FakeDataset]] = []

    def load_context(_source: Path, _protocol: dict[str, Any]) -> tuple[Any, Any]:
        dataset = _FakeDataset(
            dataset_id=env["protocol"]["evaluation_dataset_id"],
            symbols=tuple(env["protocol"]["symbols"]),
            feature_names=("x", "relative"),
        )
        dataset_refs.append(weakref.ref(dataset))
        return dataset, env["base_config"]

    monkeypatch.setattr(checkpoint, "_load_context", load_context)
    load_bundle = checkpoint.load_ppo_inference_bundle
    policy_refs: list[weakref.ReferenceType[_FakePolicy]] = []

    def track_loaded_policy(*args: Any, **kwargs: Any) -> _FakeStrategy:
        strategy = load_bundle(*args, **kwargs)
        policy_refs.append(weakref.ref(strategy.policy))
        return strategy

    monkeypatch.setattr(checkpoint, "load_ppo_inference_bundle", track_loaded_policy)
    persist = legacy._persist_replay_ledger

    def assert_released_before_persistence(
        directory: Path,
        row: dict[str, Any],
        *,
        scenario_name: str,
        symbol_index: int,
    ) -> dict[str, Any]:
        gc.collect()
        assert dataset_refs[-1]() is None
        assert policy_refs[-1]() is None
        return persist(
            directory,
            row,
            scenario_name=scenario_name,
            symbol_index=symbol_index,
        )

    monkeypatch.setattr(
        legacy, "_persist_replay_ledger", assert_released_before_persistence
    )
    checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    assert dataset_refs[-1]() is None


def test_completed_fit_is_idempotent_and_never_retrains(
    checkpoint_env: dict[str, Any],
) -> None:
    _prepare(checkpoint_env)
    fit_dir = checkpoint.fit_checkpoint(
        checkpoint_env["source"], checkpoint_env["root"], "baseline", 0
    )
    before = {
        path.relative_to(fit_dir): path.read_bytes()
        for path in fit_dir.rglob("*")
        if path.is_file()
    }

    retried = checkpoint.fit_checkpoint(
        checkpoint_env["source"], checkpoint_env["root"], "baseline", 0
    )

    after = {
        path.relative_to(retried): path.read_bytes()
        for path in retried.rglob("*")
        if path.is_file()
    }
    assert after == before
    assert len(checkpoint_env["state"]["fit_calls"]) == 1


def test_replay_retry_preserves_completed_cell_and_only_runs_missing_symbol(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)

    cell = checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    original = {
        path.name: path.read_bytes() for path in cell.iterdir() if path.is_file()
    }
    checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 1)

    assert len(env["state"]["evaluator_calls"]) == 2
    assert len(env["state"]["loader_calls"]) == 2
    assert {
        path.name: path.read_bytes() for path in cell.iterdir() if path.is_file()
    } == original


def test_interrupted_cell_publication_leaves_no_completed_cell_and_can_retry(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    publish = checkpoint._publish_attempt_directory

    def fail_publication(_stage: Path, _target: Path) -> None:
        raise OSError("injected interruption before atomic rename")

    monkeypatch.setattr(checkpoint, "_publish_attempt_directory", fail_publication)
    with pytest.raises(OSError, match="injected interruption"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    assert not (
        env["root"] / "cells" / "baseline" / "ppo0" / "base" / "symbol-0"
    ).exists()
    assert any((env["root"] / "attempts").iterdir())

    monkeypatch.setattr(checkpoint, "_publish_attempt_directory", publish)
    checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    assert (env["root"] / "cells" / "baseline" / "ppo0" / "base" / "symbol-0").is_dir()


def test_incomplete_cell_roster_blocks_assembly_and_final_publication(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    for index in range(len(env["protocol"]["symbols"]) - 1):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", index)

    with pytest.raises(ValueError, match="cell roster is incomplete"):
        checkpoint.assemble_arm(env["source"], env["root"], "baseline", 0)
    with pytest.raises(ValueError, match="cell roster is incomplete"):
        checkpoint.finalize_checkpoint_study(env["source"], env["root"])
    assert not (env["root"] / "arms" / "baseline" / "ppo0").exists()
    assert not (env["root"] / "comparison").exists()


def test_model_tampering_is_rejected_before_bundle_loader_or_evaluator(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    _prepare(env)
    fit_dir = checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    policy = fit_dir / "bundle" / "policy.zip"
    policy.write_bytes(policy.read_bytes() + b"tampered")

    with pytest.raises(ValueError, match="policy"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    assert env["state"]["loader_calls"] == []
    assert env["state"]["evaluator_calls"] == []


def test_invalid_scenario_and_boolean_symbol_are_rejected_before_loading(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)

    with pytest.raises(ValueError, match="scenario"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "cost_2x", 0)
    with pytest.raises(ValueError, match="symbol index"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", True)
    assert env["state"]["loader_calls"] == []
    assert env["state"]["evaluator_calls"] == []


def test_runtime_drift_after_evaluation_blocks_cell_publication(
    checkpoint_env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    calls = 0

    def drifting(*_args: object) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise ValueError("source/data/runtime drift blocks checkpoint publication")

    monkeypatch.setattr(checkpoint, "_require_current_protocol", drifting)

    with pytest.raises(ValueError, match="source/data/runtime"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)
    assert env["state"]["evaluator_calls"] == [("base", 0)]
    assert not (
        env["root"] / "cells" / "baseline" / "ppo0" / "base" / "symbol-0"
    ).exists()


@pytest.mark.parametrize("actual_timesteps", [True, 262_143])
def test_actual_timesteps_rejects_bool_and_nonmatching_integer(
    checkpoint_env: dict[str, Any],
    actual_timesteps: object,
) -> None:
    env = checkpoint_env
    _prepare(env)
    env["state"]["actual_timesteps"] = actual_timesteps

    with pytest.raises(ValueError, match="actual_timesteps"):
        checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    assert not (env["root"] / "fits" / "baseline" / "ppo0").exists()


def test_assembled_arm_is_verified_by_legacy_result_and_ledger_chain(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    for index in range(len(env["protocol"]["symbols"])):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", index)

    arm_dir = checkpoint.assemble_arm(env["source"], env["root"], "baseline", 0)
    row = legacy._read_arm(env["root"] / "arms", "baseline", 0, env["protocol"])

    assert row["factor"] == "baseline"
    assert row["seed"] == 0
    assert set(row["by_symbol"]) == set(env["protocol"]["symbols"])
    assert (arm_dir / "result.sha256.json").is_file()


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("scenario", "cost_2x"),
        ("symbol", "ETHUSDT"),
        ("feature_indices", [False]),
        ("provenance", {"runtime": "changed"}),
    ],
)
def test_cell_manifest_tampering_fails_before_loader_or_evaluator(
    checkpoint_env: dict[str, Any],
    field: str,
    replacement: object,
) -> None:
    env = checkpoint_env
    _prepare(env)
    checkpoint.fit_checkpoint(env["source"], env["root"], "baseline", 0)
    cell_dir = checkpoint.replay_cell(
        env["source"], env["root"], "baseline", 0, "base", 0
    )
    cell_raw = (cell_dir / "cell.json").read_bytes()
    cell = json.loads(cell_raw)
    cell[field] = replacement
    tampered = canonical_json_bytes(cell)
    (cell_dir / "cell.json").write_bytes(tampered)
    (cell_dir / "cell.digest.json").write_bytes(
        canonical_json_bytes({"sha256": _sha256(tampered), "size_bytes": len(tampered)})
    )
    env["state"]["loader_calls"].clear()
    env["state"]["evaluator_calls"].clear()

    with pytest.raises(ValueError, match="identity/provenance mismatch"):
        checkpoint.replay_cell(env["source"], env["root"], "baseline", 0, "base", 0)

    assert env["state"]["loader_calls"] == []
    assert env["state"]["evaluator_calls"] == []


def test_complete_arm_and_comparison_retries_are_write_once(
    checkpoint_env: dict[str, Any],
) -> None:
    env = checkpoint_env
    env["protocol"]["seeds"] = [0, 1, 2, 3, 4]
    _prepare(env)
    expected_arms: dict[tuple[str, int], Path] = {}
    for factor in legacy.FACTORS:
        for seed in env["protocol"]["seeds"]:
            checkpoint.fit_checkpoint(env["source"], env["root"], factor, seed)
            for scenario in checkpoint._scenario_roster(env["protocol"], factor):
                for symbol_index in range(len(env["protocol"]["symbols"])):
                    checkpoint.replay_cell(
                        env["source"],
                        env["root"],
                        factor,
                        seed,
                        scenario,
                        symbol_index,
                    )
            expected_arms[(factor, seed)] = checkpoint.assemble_arm(
                env["source"], env["root"], factor, seed
            )

    arm_bytes = {
        factor: {
            path.name: path.read_bytes()
            for path in directory.iterdir()
            if path.is_file()
        }
        for factor, directory in expected_arms.items()
    }
    evaluator_count = len(env["state"]["evaluator_calls"])
    for factor, seed in expected_arms:
        checkpoint.assemble_arm(env["source"], env["root"], factor, seed)
    comparison = checkpoint.finalize_checkpoint_study(env["source"], env["root"])
    comparison_bytes = {
        path.name: path.read_bytes() for path in comparison.iterdir() if path.is_file()
    }
    retried = checkpoint.finalize_checkpoint_study(env["source"], env["root"])

    assert len(env["state"]["evaluator_calls"]) == evaluator_count
    assert retried == comparison
    assert {
        path.name: path.read_bytes() for path in retried.iterdir() if path.is_file()
    } == comparison_bytes
    assert {
        factor: {
            path.name: path.read_bytes()
            for path in directory.iterdir()
            if path.is_file()
        }
        for factor, directory in expected_arms.items()
    } == arm_bytes
