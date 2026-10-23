from __future__ import annotations

import json
import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from tests.evaluation.experiments.test_evidence import (
    _config,
    _dataset,
    _fake_execute,
    _plan,
)
from trade_rl.artifacts.hashing import content_digest
from trade_rl.data import load_market_dataset_artifact, publish_market_dataset_artifact
from trade_rl.evaluation.experiments.errors import ArtifactIntegrityError
from trade_rl.evaluation.experiments.evidence import (
    execute_evidence_set,
    load_evidence_set,
)
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import (
    build_candidate_run_provenance,
    inspect_candidate_run_artifact,
    load_candidate_run_artifact,
    publish_candidate_run,
    resolve_candidate_run_spec,
)


def _published_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    from trade_rl.evaluation.experiments import evidence as evidence_module

    dataset = _dataset()
    dataset_root = tmp_path / "dataset"
    artifact = publish_market_dataset_artifact(dataset_root, dataset)
    plan = _plan(dataset, artifact)
    monkeypatch.setattr(evidence_module, "execute_candidate_run", _fake_execute())
    context = content_digest({"kind": "baseline", "study": plan.digest})

    store = StudyStore(tmp_path / "study")
    result = execute_evidence_set(
        store=store,
        target=Path("evidence/baseline"),
        dataset_root=dataset_root,
        plan=plan,
        config=_config(),
        research_context_digest=context,
    )
    return store, plan, result


def _rebind_evidence_manifest(
    evidence_root: Path,
    *,
    semantic_config: dict[str, object] | None = None,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if semantic_config is not None:
        manifest["semantic_config"] = semantic_config
        manifest["semantic_config_digest"] = content_digest(semantic_config)
    for entry in manifest["run_digests"]:
        run_root = evidence_root / "runs" / f"seed-{entry['ppo_seed']}"
        entry["artifact_digest"] = inspect_candidate_run_artifact(
            run_root
        ).artifact_digest
    manifest["fingerprint"] = evidence_module._evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"])
            for entry in manifest["run_digests"]
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )


def _legacy_semantic_config(
    semantic_config: dict[str, object],
    *,
    version: int,
) -> dict[str, object]:
    result = dict(semantic_config)
    result["schema_version"] = f"resolved_run_config_v{version}"
    introduced_versions = {
        "ppo_observation_schema": 2,
        "ppo_global_feature_names": 2,
        "ppo_training_layout": 3,
        "ppo_rollout_steps_per_env": 3,
        "ppo_minimum_hold_bars": 4,
        "ppo_settle_terminal_position": 4,
        "pretrade_risk_config": 5,
    }
    for field, introduced_version in introduced_versions.items():
        if version < introduced_version:
            result.pop(field, None)
    return result


def _downgrade_summary_to_version(summary: dict[str, object], version: int) -> None:
    summary["schema_version"] = f"lean_candidate_result_v{version}"
    if version < 2:
        summary.pop("ppo_observation", None)
    candidate_config = summary["candidate_config"]
    evaluation = summary["evaluation"]
    assert isinstance(candidate_config, dict)
    assert isinstance(evaluation, dict)
    candidate_fields = {
        "ppo_training_layout": 3,
        "ppo_rollout_steps_per_env": 3,
        "ppo_training_timesteps": 3,
        "ppo_minimum_hold_bars": 4,
        "ppo_observation_schema": 4,
        "ppo_settle_terminal_position": 4,
        "ppo_training_minimum_hold_suppressed_count": 4,
        "pretrade_risk_config": 5,
    }
    for field, introduced_version in candidate_fields.items():
        if version < introduced_version:
            candidate_config.pop(field, None)
    evaluation_fields = {
        "ppo_settle_terminal_position": 4,
        "pretrade_risk_config": 5,
        "expected_periods": 6,
    }
    for field, introduced_version in evaluation_fields.items():
        if version < introduced_version:
            evaluation.pop(field, None)
    by_symbol = summary["by_symbol"]
    assert isinstance(by_symbol, list)
    for symbol in by_symbol:
        assert isinstance(symbol, dict)
        strategies = symbol["strategies"]
        assert isinstance(strategies, list)
        for strategy in strategies:
            assert isinstance(strategy, dict)
            if version < 5:
                for field in (
                    "final_portfolio_value",
                    "fill_count",
                    "final_quantities",
                    "active_order_remainders",
                    "terminal_order_reasons",
                    "terminal_settlement_complete",
                    "minimum_hold_audit",
                ):
                    strategy.pop(field, None)
            if version < 4:
                diagnostics = strategy["diagnostics"]
                assert isinstance(diagnostics, dict)
                diagnostics.pop("minimum_hold_suppressed_count", None)
                diagnostics.pop("minimum_hold_unlocked_count", None)


def test_evidence_set_identity_binds_seedless_semantic_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, result = _published_evidence(tmp_path, monkeypatch)
    loaded = load_evidence_set(store.root / "evidence" / "baseline")

    assert "ppo_seed" not in loaded.semantic_config
    assert result.semantic_config_digest == content_digest(loaded.semantic_config)
    assert [
        loaded.runs[seed].summary["candidate_config"]["ppo_seed"]
        for seed in plan.ppo_seeds
    ] == list(plan.ppo_seeds)


def test_evidence_loader_rejects_self_consistent_manifest_that_reintroduces_seed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    semantic_config = dict(manifest["semantic_config"])
    semantic_config["ppo_seed"] = plan.ppo_seeds[0]
    semantic_digest = content_digest(semantic_config)
    manifest["semantic_config"] = semantic_config
    manifest["semantic_config_digest"] = semantic_digest
    manifest["fingerprint"] = content_digest(
        {
            "schema_version": manifest["schema_version"],
            "semantic_config_digest": semantic_digest,
            "ppo_seeds": manifest["ppo_seeds"],
            "run_digests": manifest["run_digests"],
            "research_context_digest": manifest["research_context_digest"],
        }
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="ppo_seed|semantic config"):
        load_evidence_set(evidence_root)


def test_evidence_loader_rejects_valid_member_run_with_different_resolved_config(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from trade_rl.evaluation.experiments import evidence as evidence_module

    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    dataset = load_market_dataset_artifact(tmp_path / "dataset")
    context = content_digest({"kind": "baseline", "study": plan.digest})
    changed_config = replace(
        _config(),
        ppo_seed=plan.ppo_seeds[0],
        forecast_entry_threshold=0.02,
    )
    spec = resolve_candidate_run_spec(
        dataset,
        dataset_artifact_schema=plan.dataset_artifact_schema,
        dataset_artifact_digest=plan.dataset_artifact_digest,
        config=changed_config,
        execution_overlay=plan.baseline_config.execution_overlay,
    )
    wrong_run_root = tmp_path / "self-consistent-wrong-config-run"
    publish_candidate_run(
        wrong_run_root,
        evidence_module.execute_candidate_run(dataset, spec),
        build_candidate_run_provenance(research_context_digest=context),
    )
    wrong_run = load_candidate_run_artifact(wrong_run_root)
    assert wrong_run.summary["candidate_config"]["forecast_entry_threshold"] == 0.02

    member_root = evidence_root / "runs" / f"seed-{plan.ppo_seeds[0]}"
    shutil.rmtree(member_root)
    shutil.copytree(wrong_run_root, member_root)

    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    identity = inspect_candidate_run_artifact(member_root)
    for run_digest in manifest["run_digests"]:
        if run_digest["ppo_seed"] == plan.ppo_seeds[0]:
            run_digest["artifact_digest"] = identity.artifact_digest
    manifest["fingerprint"] = evidence_module._evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"])
            for entry in manifest["run_digests"]
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="config"):
        load_evidence_set(evidence_root)


def test_evidence_loader_rejects_newer_run_fields_after_schema_downgrade(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, _, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["run_digests"]:
        summary_path = (
            evidence_root / "runs" / f"seed-{entry['ppo_seed']}" / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["schema_version"] = "lean_candidate_result_v1"
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    _rebind_evidence_manifest(
        evidence_root,
        semantic_config=_legacy_semantic_config(manifest["semantic_config"], version=1),
    )

    with pytest.raises(ArtifactIntegrityError, match="schema|newer|field"):
        load_evidence_set(evidence_root)


@pytest.mark.parametrize("version", (1, 2, 3, 4, 5))
def test_evidence_loader_preserves_legacy_run_schema_bindings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
) -> None:
    store, _, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    for entry in manifest["run_digests"]:
        summary_path = (
            evidence_root / "runs" / f"seed-{entry['ppo_seed']}" / "summary.json"
        )
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        _downgrade_summary_to_version(summary, version)
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    _rebind_evidence_manifest(
        evidence_root,
        semantic_config=_legacy_semantic_config(
            manifest["semantic_config"], version=version
        ),
    )

    loaded = load_evidence_set(evidence_root)
    assert all(
        run.summary["schema_version"] == f"lean_candidate_result_v{version}"
        for run in loaded.runs.values()
    )


@pytest.mark.parametrize(
    ("version", "location", "field"),
    (
        (3, "candidate_config", "ppo_minimum_hold_bars"),
        (4, "strategy", "final_quantities"),
        (5, "evaluation", "expected_periods"),
    ),
)
def test_evidence_loader_rejects_later_parameter_evidence_by_location(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    version: int,
    location: str,
    field: str,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest = json.loads((evidence_root / "manifest.json").read_text(encoding="utf-8"))
    for seed in plan.ppo_seeds:
        summary_path = evidence_root / "runs" / f"seed-{seed}" / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        original = json.loads(json.dumps(summary))
        _downgrade_summary_to_version(summary, version)
        if seed == plan.ppo_seeds[0]:
            if location == "candidate_config":
                summary[location][field] = original[location][field]
            elif location == "evaluation":
                summary[location][field] = original[location][field]
            else:
                summary["by_symbol"][0]["strategies"][0][field] = original["by_symbol"][
                    0
                ]["strategies"][0][field]
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    _rebind_evidence_manifest(
        evidence_root,
        semantic_config=_legacy_semantic_config(
            manifest["semantic_config"], version=version
        ),
    )

    with pytest.raises(ArtifactIntegrityError, match="fields|schema"):
        load_evidence_set(evidence_root)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("signal_index", False), ("feature_indices", [False])),
)
def test_evidence_loader_uses_type_faithful_config_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    replacement: object,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    summary_path = evidence_root / "runs" / f"seed-{plan.ppo_seeds[0]}" / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    candidate_config = summary["candidate_config"]
    candidate_config[field] = replacement
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    _rebind_evidence_manifest(evidence_root)

    with pytest.raises(ArtifactIntegrityError, match="config"):
        load_evidence_set(evidence_root)


def test_evidence_loader_binds_each_run_dataset_to_study_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store, plan, _ = _published_evidence(tmp_path, monkeypatch)
    evidence_root = store.root / "evidence" / "baseline"
    manifest_path = evidence_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_digests = {entry["ppo_seed"]: entry for entry in manifest["run_digests"]}

    for seed in plan.ppo_seeds:
        member_root = evidence_root / "runs" / f"seed-{seed}"
        summary_path = member_root / "summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        summary["dataset_id"] = "f" * 64
        summary["dataset_artifact"]["artifact_digest"] = "e" * 64
        summary_path.write_text(
            json.dumps(summary, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        run_digests[seed]["artifact_digest"] = inspect_candidate_run_artifact(
            member_root
        ).artifact_digest

    from trade_rl.evaluation.experiments import evidence as evidence_module

    manifest["fingerprint"] = evidence_module._evidence_fingerprint(
        semantic_config_digest=manifest["semantic_config_digest"],
        ppo_seeds=tuple(manifest["ppo_seeds"]),
        run_digests=tuple(
            (entry["ppo_seed"], entry["artifact_digest"])
            for entry in manifest["run_digests"]
        ),
        research_context_digest=manifest["research_context_digest"],
    )
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    with pytest.raises(ArtifactIntegrityError, match="dataset identity"):
        load_evidence_set(evidence_root, expected_plan=plan)
