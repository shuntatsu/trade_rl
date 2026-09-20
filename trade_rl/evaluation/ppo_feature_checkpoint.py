"""Bounded, write-once checkpoint execution for the PPO feature ablation."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import uuid
from copy import deepcopy
from dataclasses import replace
from functools import partial
from hashlib import sha256
from pathlib import Path
from typing import Any, cast

from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import (
    file_digest_and_size,
    open_regular_binary,
    read_verified_bytes,
)
from trade_rl.data.artifacts import load_market_dataset_artifact
from trade_rl.data.features.price_channels import with_price_channels
from trade_rl.evaluation import ppo_feature_study as legacy
from trade_rl.evaluation.directional import evaluate_directional_arm
from trade_rl.evaluation.directional_candidates import (
    PPO_TIMESTEPS,
    fit_directional_candidate,
)
from trade_rl.evaluation.experiments import inspect_study
from trade_rl.evaluation.experiments.store import StudyStore
from trade_rl.evaluation.runs import build_candidate_run_provenance
from trade_rl.strategies.rl.ppo import PPOIntentStrategy
from trade_rl.strategies.rl.ppo_artifact import (
    load_ppo_inference_bundle,
    save_ppo_inference_bundle,
)

PROTOCOL_SCHEMA = "ppo_feature_checkpoint_protocol_v1"
FIT_SCHEMA = "ppo_feature_fit_checkpoint_v1"
CELL_SCHEMA = "ppo_feature_replay_cell_v1"
_BUNDLE_SCHEMA = "ppo_inference_bundle_v1"
_STAGING_PREFIX = ".checkpoint-staging-"


def _sha256(raw: bytes) -> str:
    return sha256(raw).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _require_strict_int(
    value: object, *, field: str, expected: int | None = None
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    if expected is not None and value != expected:
        raise ValueError(f"{field} differs from the preregistered value")
    return value


def _absolute(path: Path) -> Path:
    return Path(os.path.abspath(path))


def _check_directory_ancestors(path: Path) -> None:
    absolute = _absolute(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"directory path must not contain symlinks: {current}")
        if current.exists() and not current.is_dir():
            raise ValueError(f"directory path parent is not a directory: {current}")


def _require_directory(path: Path, *, field: str) -> None:
    _check_directory_ancestors(path)
    if path.is_symlink() or not path.is_dir():
        raise ValueError(f"{field} must be a regular directory")


def _make_directory(root: Path, parts: tuple[str, ...]) -> Path:
    _require_directory(root, field="checkpoint root")
    current = root
    for part in parts:
        if not part or part in {".", ".."} or "/" in part or "\\" in part:
            raise ValueError("checkpoint path contains an unsafe component")
        current = current / part
        if current.is_symlink():
            raise ValueError("checkpoint parent directory must not be a symlink")
        if current.exists():
            if not current.is_dir():
                raise ValueError("checkpoint parent must be a regular directory")
        else:
            current.mkdir()
    return current


def _require_relative_directory(root: Path, parts: tuple[str, ...]) -> Path:
    _require_directory(root, field="checkpoint root")
    current = root
    for part in parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("checkpoint parent directory must not be a symlink")
        if not current.exists():
            raise ValueError(f"checkpoint evidence directory is missing: {current}")
        if not current.is_dir():
            raise ValueError("checkpoint evidence parent must be a regular directory")
    return current


def _write_once(path: Path, payload: object) -> bytes:
    raw = canonical_json_bytes(payload)
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    return raw


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    with open_regular_binary(path, field=field) as stream:
        return stream.read()


def _parse_canonical_json(raw: bytes, *, field: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{field} is not valid JSON") from error
    if not isinstance(value, dict) or canonical_json_bytes(value) != raw:
        raise ValueError(f"{field} must be a canonical JSON object")
    return value


def _read_verified_json(path: Path, sidecar: Path, *, field: str) -> dict[str, Any]:
    sidecar_raw = _read_regular_bytes(sidecar, field=f"{field} digest")
    digest_record = _parse_canonical_json(sidecar_raw, field=f"{field} digest")
    if set(digest_record) != {"sha256", "size_bytes"}:
        raise ValueError(f"{field} digest record has an unexpected shape")
    digest = digest_record["sha256"]
    size = digest_record["size_bytes"]
    if not _is_sha256(digest):
        raise ValueError(f"{field} digest is malformed")
    _require_strict_int(size, field=f"{field} size")
    if size < 0:
        raise ValueError(f"{field} size must be non-negative")
    raw = read_verified_bytes(
        path,
        expected_digest=digest,
        expected_size_bytes=size,
        field=field,
    )
    return _parse_canonical_json(raw, field=field)


def _write_verified_json(path: Path, payload: object) -> dict[str, str | int]:
    raw = _write_once(path, payload)
    return {"sha256": _sha256(raw), "size_bytes": len(raw)}


def _write_json_sidecar(path: Path, *, raw: bytes) -> None:
    _write_once(path, {"sha256": _sha256(raw), "size_bytes": len(raw)})


def _scenario_roster(core_protocol: dict[str, Any], factor: str) -> tuple[str, ...]:
    if factor not in legacy.FACTORS:
        raise ValueError("factor is not preregistered")
    if factor == legacy.BASELINE_FACTOR:
        return ("base",)
    stress_names = core_protocol["evaluation"]["stress_names"]
    if not isinstance(stress_names, list) or any(
        not isinstance(name, str) or not name for name in stress_names
    ):
        raise ValueError("preregistered stress roster is malformed")
    return ("base", *tuple(stress_names))


def _execution_contract(core_protocol: dict[str, Any]) -> dict[str, Any]:
    factors = core_protocol["factors"]
    seeds = core_protocol["seeds"]
    symbols = core_protocol["symbols"]
    if (
        not isinstance(factors, dict)
        or tuple(factors) != legacy.FACTORS
        or not isinstance(seeds, list)
        or not seeds
        or not isinstance(symbols, list)
        or not symbols
    ):
        raise ValueError("legacy protocol roster cannot be checkpointed")
    fit_roster: list[dict[str, Any]] = []
    cell_roster: list[dict[str, Any]] = []
    scenario_names: dict[str, list[str]] = {}
    for factor in legacy.FACTORS:
        factor_scenarios = _scenario_roster(core_protocol, factor)
        scenario_names[factor] = list(factor_scenarios)
        for seed in seeds:
            _require_strict_int(seed, field="seed")
            fit_roster.append({"factor": factor, "seed": seed})
            for scenario in factor_scenarios:
                for symbol_index, symbol in enumerate(symbols):
                    _require_strict_int(symbol_index, field="symbol_index")
                    if not isinstance(symbol, str) or not symbol:
                        raise ValueError("symbol roster contains an invalid name")
                    cell_roster.append(
                        {
                            "factor": factor,
                            "seed": seed,
                            "scenario": scenario,
                            "symbol_index": symbol_index,
                            "symbol": symbol,
                        }
                    )
    return {
        "schema": "ppo_feature_checkpoint_execution_v1",
        "fit_roster": fit_roster,
        "cell_roster": cell_roster,
        "scenario_names_by_factor": scenario_names,
        "fit_unit": "one factor and seed per invocation",
        "replay_unit": "one scenario and symbol per invocation",
        "completed_stage_policy": "validate and no-op; never overwrite or recompute",
        "failed_attempt_policy": "retain staging as non-authoritative evidence",
        "partial_optimizer_resume": False,
        "legacy_economics_and_admission": "ppo_feature_ablation_protocol_v1",
    }


def expected_checkpoint_protocol(source: Path) -> dict[str, Any]:
    """Bind the legacy economic protocol to this module's checkpoint contract."""
    core_protocol = legacy.expected_protocol(source)
    if core_protocol["training"]["requested_timesteps"] != PPO_TIMESTEPS:
        raise ValueError("PPO fit budget differs from the frozen directional factory")
    return {
        "schema": PROTOCOL_SCHEMA,
        "core_protocol": core_protocol,
        "core_protocol_digest": content_digest(core_protocol),
        "execution": _execution_contract(core_protocol),
        "runner_sha256": _sha256(Path(__file__).read_bytes()),
    }


def _validate_checkpoint_protocol(source: Path, root: Path) -> dict[str, Any]:
    expected = expected_checkpoint_protocol(source)
    _validate_saved_protocol(root, expected)
    return expected


def _validate_saved_protocol(root: Path, expected: dict[str, Any]) -> None:
    _require_directory(root, field="checkpoint root")
    protocol_path = root / "checkpoint-protocol.json"
    protocol_digest_path = root / "checkpoint-protocol.digest.json"
    protocol_raw = _read_regular_bytes(protocol_path, field="checkpoint protocol")
    if protocol_raw != canonical_json_bytes(expected):
        raise ValueError("checkpoint protocol differs from current source/data/runtime")
    digest_raw = _read_regular_bytes(
        protocol_digest_path, field="checkpoint protocol digest"
    )
    digest_record = _parse_canonical_json(
        digest_raw, field="checkpoint protocol digest"
    )
    if digest_record != {"digest": content_digest(expected)}:
        raise ValueError("checkpoint protocol digest changed")

    core_protocol = expected["core_protocol"]
    snapshot_path = root / "source-snapshot.zip"
    snapshot_digest, snapshot_size = file_digest_and_size(
        snapshot_path, field="checkpoint source snapshot"
    )
    if snapshot_digest != core_protocol.get("source_snapshot_sha256"):
        raise ValueError("checkpoint source snapshot digest mismatch")
    snapshot = read_verified_bytes(
        snapshot_path,
        expected_digest=snapshot_digest,
        expected_size_bytes=snapshot_size,
        field="checkpoint source snapshot",
    )
    with tempfile.TemporaryDirectory(prefix="trade-rl-checkpoint-snapshot-") as temp:
        private_root = Path(temp)
        (private_root / "source-snapshot.zip").write_bytes(snapshot)
        legacy._verify_source_snapshot(private_root, core_protocol)


def validate_checkpoint_protocol(source: Path, root: Path) -> dict[str, Any]:
    """Public read-only protocol validation for checkpoint consumers."""
    return _validate_checkpoint_protocol(source, root)


def _require_current_protocol(
    source: Path, root: Path, protocol: dict[str, Any]
) -> None:
    """Recheck immutable identities without loading the large Dataset arrays."""
    _validate_saved_protocol(root, protocol)
    core_protocol = protocol["core_protocol"]
    if _sha256(Path(__file__).read_bytes()) != protocol["runner_sha256"]:
        raise ValueError("source/data/runtime drift blocks checkpoint publication")
    provenance = build_candidate_run_provenance()
    if canonical_json_bytes(provenance) != canonical_json_bytes(
        core_protocol["provenance"]
    ):
        raise ValueError("source/data/runtime drift blocks checkpoint publication")

    dataset_root = source / "dataset"
    _require_directory(dataset_root, field="source Dataset artifact")
    if {entry.name for entry in dataset_root.iterdir()} != {
        "manifest.json",
        "arrays.npz",
    }:
        raise ValueError("source/data/runtime drift blocks checkpoint publication")
    manifest = _parse_canonical_json(
        _read_regular_bytes(
            dataset_root / "manifest.json", field="source Dataset manifest"
        ),
        field="source Dataset manifest",
    )
    artifact_digest = manifest.pop("artifact_digest", None)
    arrays_digest, _arrays_size = file_digest_and_size(
        dataset_root / "arrays.npz", field="source Dataset arrays"
    )
    if (
        artifact_digest != core_protocol["source_artifact_digest"]
        or content_digest(manifest) != artifact_digest
        or manifest.get("dataset_id") != core_protocol["source_dataset_id"]
        or manifest.get("arrays_digest") != arrays_digest
    ):
        raise ValueError("source/data/runtime drift blocks checkpoint publication")

    study = inspect_study(source / "study").plan
    plan_path = source / "study" / "plan.json"
    plan_digest, _plan_size = file_digest_and_size(plan_path, field="source StudyPlan")
    if (
        study.digest != core_protocol["source_study_digest"]
        or study.dataset_id != core_protocol["source_dataset_id"]
        or plan_digest != core_protocol["source_plan_sha256"]
    ):
        raise ValueError("source/data/runtime drift blocks checkpoint publication")
    current_snapshot = legacy._source_snapshot_bytes(provenance)
    if _sha256(current_snapshot) != core_protocol["source_snapshot_sha256"]:
        raise ValueError("source/data/runtime drift blocks checkpoint publication")


def _load_context(source: Path, core_protocol: dict[str, Any]) -> tuple[Any, Any]:
    """Resolve current Dataset and frozen base config for fit or single-cell replay."""
    dataset = load_market_dataset_artifact(source / "dataset")
    if dataset.dataset_id != core_protocol["source_dataset_id"]:
        raise ValueError("current source Dataset identity differs from protocol")
    dataset = with_price_channels(dataset)
    if dataset.dataset_id != core_protocol["evaluation_dataset_id"]:
        raise ValueError("current evaluation Dataset identity differs from protocol")
    plan = inspect_study(source / "study").plan
    if plan.digest != core_protocol["source_study_digest"]:
        raise ValueError("current StudyPlan identity differs from protocol")
    if plan.dataset_id != core_protocol["source_dataset_id"]:
        raise ValueError("current StudyPlan Dataset differs from protocol")
    if not all(isinstance(name, str) and name for name in dataset.feature_names):
        raise ValueError("current Dataset feature schema is malformed")
    for factor, definition in core_protocol["factors"].items():
        names = definition["feature_names"]
        indices = definition["feature_indices"]
        if (
            not isinstance(names, list)
            or not isinstance(indices, list)
            or len(names) != len(indices)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or not 0 <= index < len(dataset.feature_names)
                for index in indices
            )
            or tuple(dataset.feature_names[index] for index in indices) != tuple(names)
        ):
            raise ValueError(f"current Dataset feature schema differs for {factor}")
    return dataset, plan.baseline_config


def _validate_factor_seed(
    core_protocol: dict[str, Any], factor: str, seed: int
) -> tuple[dict[str, Any], int]:
    if factor not in legacy.FACTORS:
        raise ValueError("factor is not preregistered")
    seed = _require_strict_int(seed, field="seed")
    seeds = core_protocol["seeds"]
    if seed not in seeds:
        raise ValueError("seed is not preregistered")
    return core_protocol["factors"][factor], seed


def _validate_cell_request(
    core_protocol: dict[str, Any],
    factor: str,
    seed: int,
    scenario_name: str,
    symbol_index: int,
) -> tuple[dict[str, Any], str, int]:
    feature, checked_seed = _validate_factor_seed(core_protocol, factor, seed)
    if not isinstance(scenario_name, str) or scenario_name not in _scenario_roster(
        core_protocol, factor
    ):
        raise ValueError("scenario is not preregistered for this factor")
    symbol_index = _require_strict_int(symbol_index, field="symbol index")
    symbols = core_protocol["symbols"]
    if not 0 <= symbol_index < len(symbols):
        raise ValueError("symbol index is outside the preregistered roster")
    return feature, scenario_name, symbol_index


def _attempt_directory(root: Path, name: str) -> Path:
    attempts = _make_directory(root, ("attempts",))
    path = attempts / f"{name}-{uuid.uuid4().hex}"
    path.mkdir()
    return path


def _publish_attempt_directory(stage: Path, target: Path) -> None:
    """Atomically publish one verified directory without replacing evidence."""
    _require_directory(stage, field="checkpoint staging attempt")
    _check_directory_ancestors(target.parent)
    if not target.parent.is_dir() or target.parent.is_symlink():
        raise ValueError("checkpoint publication parent is not a regular directory")
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"checkpoint evidence already exists: {target}")
    stage.rename(target)
    _require_directory(target, field="published checkpoint evidence")


def _publish_new_study(stage: Path, root: Path) -> None:
    _require_directory(stage, field="checkpoint prepare staging")
    _check_directory_ancestors(root.parent)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"checkpoint study already exists: {root}")
    stage.rename(root)
    _require_directory(root, field="published checkpoint root")


def prepare_checkpoint_study(source: Path, root: Path) -> dict[str, Any]:
    """Create a new checkpoint-only root with frozen source protocol and snapshot."""
    root = _absolute(root)
    parent = root.parent
    _check_directory_ancestors(parent)
    parent.mkdir(parents=True, exist_ok=True)
    with StudyStore(parent).mutation_lock():
        if root.exists() or root.is_symlink():
            raise FileExistsError(f"checkpoint study already exists: {root}")
        protocol = expected_checkpoint_protocol(source)
        core_protocol = protocol["core_protocol"]
        snapshot = legacy._source_snapshot_bytes(core_protocol["provenance"])
        if _sha256(snapshot) != core_protocol["source_snapshot_sha256"]:
            raise ValueError("current source snapshot differs from expected protocol")
        stage = parent / f"{_STAGING_PREFIX}{root.name}-{uuid.uuid4().hex}"
        stage.mkdir()
        (stage / "source-snapshot.zip").write_bytes(snapshot)
        _write_once(stage / "checkpoint-protocol.json", protocol)
        _write_once(
            stage / "checkpoint-protocol.digest.json",
            {"digest": content_digest(protocol)},
        )
        _validate_saved_protocol(stage, protocol)
        _require_current_protocol(source, stage, protocol)
        _publish_new_study(stage, root)
        return protocol


def _fit_directory(root: Path, factor: str, seed: int) -> Path:
    return root / "fits" / factor / f"ppo{seed}"


def _validate_fit_directory(
    root: Path,
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    *,
    dataset_feature_names: tuple[str, ...] | None = None,
    fit_directory: Path | None = None,
) -> tuple[dict[str, Any], str]:
    core_protocol = protocol["core_protocol"]
    feature, seed = _validate_factor_seed(core_protocol, factor, seed)
    directory = (
        _require_relative_directory(root, ("fits", factor, f"ppo{seed}"))
        if fit_directory is None
        else fit_directory
    )
    _require_directory(directory, field="fit checkpoint")
    entries = {path.name for path in directory.iterdir()}
    if entries != {"fit.json", "fit.digest.json", "model.zip", "bundle"}:
        raise ValueError("fit checkpoint file roster is incomplete or unexpected")
    bundle_directory = directory / "bundle"
    _require_directory(bundle_directory, field="PPO inference bundle")
    if {path.name for path in bundle_directory.iterdir()} != {
        "manifest.json",
        "policy.zip",
    }:
        raise ValueError("PPO inference bundle file roster is incomplete or unexpected")
    manifest = _read_verified_json(
        directory / "fit.json",
        directory / "fit.digest.json",
        field="fit checkpoint manifest",
    )
    expected_fields = {
        "schema",
        "factor",
        "seed",
        "checkpoint_protocol_digest",
        "core_protocol_digest",
        "dataset_id",
        "source_dataset_id",
        "feature_names",
        "feature_indices",
        "dataset_feature_names",
        "training",
        "actual_timesteps",
        "provenance",
        "bundle_digest",
        "bundle_manifest_sha256",
        "bundle_manifest_size_bytes",
        "policy_sha256",
        "policy_size_bytes",
        "training_model_sha256",
        "training_model_size_bytes",
    }
    if set(manifest) != expected_fields:
        raise ValueError("fit checkpoint manifest has an unexpected shape")
    expected_names = feature["feature_names"]
    expected_indices = feature["feature_indices"]
    if (
        manifest["schema"] != FIT_SCHEMA
        or manifest["factor"] != factor
        or isinstance(manifest["seed"], bool)
        or not isinstance(manifest["seed"], int)
        or manifest["seed"] != seed
        or manifest["checkpoint_protocol_digest"] != content_digest(protocol)
        or manifest["core_protocol_digest"] != content_digest(core_protocol)
        or manifest["dataset_id"] != core_protocol["evaluation_dataset_id"]
        or manifest["source_dataset_id"] != core_protocol["source_dataset_id"]
        or canonical_json_bytes(manifest["feature_names"])
        != canonical_json_bytes(expected_names)
        or canonical_json_bytes(manifest["feature_indices"])
        != canonical_json_bytes(expected_indices)
        or canonical_json_bytes(manifest["training"])
        != canonical_json_bytes(core_protocol["training"])
        or canonical_json_bytes(manifest["provenance"])
        != canonical_json_bytes(core_protocol["provenance"])
    ):
        raise ValueError("fit checkpoint identity/provenance mismatch")
    actual_timesteps = _require_strict_int(
        manifest["actual_timesteps"],
        field="fit checkpoint actual_timesteps",
        expected=core_protocol["training"]["requested_timesteps"],
    )
    if actual_timesteps != PPO_TIMESTEPS:
        raise ValueError("fit checkpoint actual_timesteps differs from PPO factory")
    full_schema = manifest["dataset_feature_names"]
    if (
        not isinstance(full_schema, list)
        or any(not isinstance(name, str) or not name for name in full_schema)
        or len(full_schema) != len(set(full_schema))
        or not isinstance(expected_indices, list)
        or any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < len(full_schema)
            for index in expected_indices
        )
        or tuple(full_schema[index] for index in expected_indices)
        != tuple(expected_names)
    ):
        raise ValueError("fit checkpoint full Dataset feature schema is malformed")
    if (
        dataset_feature_names is not None
        and tuple(full_schema) != dataset_feature_names
    ):
        raise ValueError(
            "fit checkpoint Dataset feature schema differs from current data"
        )

    bundle_digest = manifest["bundle_digest"]
    bundle_manifest_digest = manifest["bundle_manifest_sha256"]
    bundle_manifest_size = manifest["bundle_manifest_size_bytes"]
    policy_digest = manifest["policy_sha256"]
    policy_size = manifest["policy_size_bytes"]
    training_model_digest = manifest["training_model_sha256"]
    training_model_size = manifest["training_model_size_bytes"]
    for digest, field in (
        (bundle_digest, "bundle digest"),
        (bundle_manifest_digest, "bundle manifest digest"),
        (policy_digest, "bundle policy digest"),
        (training_model_digest, "training model digest"),
    ):
        if not _is_sha256(digest):
            raise ValueError(f"fit checkpoint {field} is malformed")
    for size, field in (
        (bundle_manifest_size, "bundle manifest size"),
        (policy_size, "bundle policy size"),
        (training_model_size, "training model size"),
    ):
        _require_strict_int(size, field=field)
        if size < 0:
            raise ValueError(f"{field} must be non-negative")
    bundle_manifest_raw = read_verified_bytes(
        bundle_directory / "manifest.json",
        expected_digest=bundle_manifest_digest,
        expected_size_bytes=bundle_manifest_size,
        field="PPO inference manifest",
    )
    bundle_manifest = _parse_canonical_json(
        bundle_manifest_raw, field="PPO inference manifest"
    )
    policy_path = bundle_directory / "policy.zip"
    policy_raw = read_verified_bytes(
        policy_path,
        expected_digest=policy_digest,
        expected_size_bytes=policy_size,
        field="PPO inference policy",
    )
    if (
        content_digest(bundle_manifest) != bundle_digest
        or bundle_manifest.get("schema") != _BUNDLE_SCHEMA
        or canonical_json_bytes(bundle_manifest.get("feature_names"))
        != canonical_json_bytes(expected_names)
        or canonical_json_bytes(bundle_manifest.get("feature_indices"))
        != canonical_json_bytes(expected_indices)
        or bundle_manifest.get("policy_sha256") != _sha256(policy_raw)
    ):
        raise ValueError("PPO inference bundle manifest/policy mismatch")
    read_verified_bytes(
        directory / "model.zip",
        expected_digest=training_model_digest,
        expected_size_bytes=training_model_size,
        field="PPO training model export",
    )
    return manifest, content_digest(manifest)


def fit_checkpoint(source: Path, root: Path, factor: str, seed: int) -> Path:
    """Fit one factor/seed once and publish an inference-safe immutable bundle."""
    protocol = _validate_checkpoint_protocol(source, root)
    core_protocol = protocol["core_protocol"]
    feature, seed = _validate_factor_seed(core_protocol, factor, seed)
    with StudyStore(root).mutation_lock():
        target = _fit_directory(root, factor, seed)
        if target.exists() or target.is_symlink():
            dataset, _config = _load_context(source, core_protocol)
            _validate_fit_directory(
                root,
                factor,
                seed,
                protocol,
                dataset_feature_names=tuple(dataset.feature_names),
            )
            del dataset
            _require_current_protocol(source, root, protocol)
            return target

        dataset, base_config = _load_context(source, core_protocol)
        resolved = replace(
            base_config,
            feature_names=tuple(feature["feature_names"]),
            feature_indices=tuple(feature["feature_indices"]),
        )
        attempt = _attempt_directory(root, f"fit-{factor}-ppo{seed}")
        factory = fit_directional_candidate(
            f"ppo{seed}",
            dataset,
            resolved,
            attempt,
            ppo_risk_config=legacy.TRAINING_RISK,
            initial_capital=core_protocol["training"]["initial_capital"],
            gross_budget=core_protocol["training"]["gross_budget"],
        )
        strategy = factory()
        policy = getattr(strategy, "policy", None)
        actual_timesteps = _require_strict_int(
            getattr(policy, "num_timesteps", None),
            field="fit actual_timesteps",
            expected=core_protocol["training"]["requested_timesteps"],
        )
        if actual_timesteps != PPO_TIMESTEPS:
            raise ValueError("fit actual_timesteps differs from PPO factory")
        bundle_digest = save_ppo_inference_bundle(
            attempt / "bundle",
            cast(PPOIntentStrategy, strategy),
            feature_names=tuple(dataset.feature_names),
        )
        if not _is_sha256(bundle_digest):
            raise ValueError("PPO inference bundle returned a malformed digest")
        bundle_manifest_digest, bundle_manifest_size = file_digest_and_size(
            attempt / "bundle" / "manifest.json",
            field="PPO inference manifest",
        )
        policy_digest, policy_size = file_digest_and_size(
            attempt / "bundle" / "policy.zip", field="PPO inference policy"
        )
        training_model_digest, training_model_size = file_digest_and_size(
            attempt / "model.zip", field="PPO training model export"
        )
        fit_manifest = {
            "schema": FIT_SCHEMA,
            "factor": factor,
            "seed": seed,
            "checkpoint_protocol_digest": content_digest(protocol),
            "core_protocol_digest": content_digest(core_protocol),
            "dataset_id": dataset.dataset_id,
            "source_dataset_id": core_protocol["source_dataset_id"],
            "feature_names": list(feature["feature_names"]),
            "feature_indices": list(feature["feature_indices"]),
            "dataset_feature_names": list(dataset.feature_names),
            "training": core_protocol["training"],
            "actual_timesteps": actual_timesteps,
            "provenance": core_protocol["provenance"],
            "bundle_digest": bundle_digest,
            "bundle_manifest_sha256": bundle_manifest_digest,
            "bundle_manifest_size_bytes": bundle_manifest_size,
            "policy_sha256": policy_digest,
            "policy_size_bytes": policy_size,
            "training_model_sha256": training_model_digest,
            "training_model_size_bytes": training_model_size,
        }
        manifest_raw = _write_once(attempt / "fit.json", fit_manifest)
        _write_json_sidecar(attempt / "fit.digest.json", raw=manifest_raw)
        _validate_fit_directory(
            root,
            factor,
            seed,
            protocol,
            dataset_feature_names=tuple(dataset.feature_names),
            fit_directory=attempt,
        )
        dataset_feature_names = tuple(dataset.feature_names)
        del dataset, base_config, resolved, strategy, policy, factory
        _require_current_protocol(source, root, protocol)
        fit_parent = _make_directory(root, ("fits", factor))
        _publish_attempt_directory(attempt, fit_parent / f"ppo{seed}")
        _validate_fit_directory(
            root,
            factor,
            seed,
            protocol,
            dataset_feature_names=dataset_feature_names,
        )
        _require_current_protocol(source, root, protocol)
        return target


def _cell_directory(
    root: Path, factor: str, seed: int, scenario_name: str, symbol_index: int
) -> Path:
    return (
        root
        / "cells"
        / factor
        / f"ppo{seed}"
        / scenario_name
        / f"symbol-{symbol_index}"
    )


def _read_cell_directory(
    directory: Path,
    *,
    factor: str,
    seed: int,
    scenario_name: str,
    symbol_index: int,
    protocol: dict[str, Any],
    fit_digest: str,
    dataset_feature_names: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    core_protocol = protocol["core_protocol"]
    feature, scenario_name, symbol_index = _validate_cell_request(
        core_protocol, factor, seed, scenario_name, symbol_index
    )
    seed = _require_strict_int(seed, field="seed")
    symbols = core_protocol["symbols"]
    symbol = symbols[symbol_index]
    expected_filename = f"ledger-{scenario_name}-symbol-{symbol_index}.json.gz"
    _require_directory(directory, field="checkpoint replay cell")
    if {path.name for path in directory.iterdir()} != {
        "cell.json",
        "cell.digest.json",
        expected_filename,
    }:
        raise ValueError("replay cell file roster is incomplete or unexpected")
    cell = _read_verified_json(
        directory / "cell.json",
        directory / "cell.digest.json",
        field="replay cell manifest",
    )
    expected_fields = {
        "schema",
        "factor",
        "seed",
        "scenario",
        "symbol_index",
        "symbol",
        "checkpoint_protocol_digest",
        "core_protocol_digest",
        "fit_digest",
        "dataset_id",
        "dataset_feature_names",
        "feature_names",
        "feature_indices",
        "provenance",
        "ledger_size_bytes",
        "replay",
    }
    if set(cell) != expected_fields:
        raise ValueError("replay cell manifest has an unexpected shape")
    if (
        cell["schema"] != CELL_SCHEMA
        or cell["factor"] != factor
        or isinstance(cell["seed"], bool)
        or not isinstance(cell["seed"], int)
        or cell["seed"] != seed
        or cell["scenario"] != scenario_name
        or isinstance(cell["symbol_index"], bool)
        or not isinstance(cell["symbol_index"], int)
        or cell["symbol_index"] != symbol_index
        or cell["symbol"] != symbol
        or cell["checkpoint_protocol_digest"] != content_digest(protocol)
        or cell["core_protocol_digest"] != content_digest(core_protocol)
        or cell["fit_digest"] != fit_digest
        or cell["dataset_id"] != core_protocol["evaluation_dataset_id"]
        or canonical_json_bytes(cell["dataset_feature_names"])
        != canonical_json_bytes(list(dataset_feature_names))
        or canonical_json_bytes(cell["feature_names"])
        != canonical_json_bytes(feature["feature_names"])
        or canonical_json_bytes(cell["feature_indices"])
        != canonical_json_bytes(feature["feature_indices"])
        or canonical_json_bytes(cell["provenance"])
        != canonical_json_bytes(core_protocol["provenance"])
        or not isinstance(cell["replay"], dict)
    ):
        raise ValueError("replay cell identity/provenance mismatch")
    ledger_size = _require_strict_int(
        cell["ledger_size_bytes"], field="replay ledger size"
    )
    if ledger_size < 0:
        raise ValueError("replay ledger size must be non-negative")
    replay = cell["replay"]
    ledger_digest = replay.get("ledger_evidence_gzip_sha256")
    if not isinstance(ledger_digest, str) or not _is_sha256(ledger_digest):
        raise ValueError("replay compressed ledger digest is malformed")
    ledger_path = directory / expected_filename
    ledger_bytes = read_verified_bytes(
        ledger_path,
        expected_digest=ledger_digest,
        expected_size_bytes=ledger_size,
        field="replay compressed ledger",
    )
    if replay.get("ledger_evidence_file") != expected_filename:
        raise ValueError("replay ledger filename differs from its cell")
    from tempfile import TemporaryDirectory

    with TemporaryDirectory(prefix="trade-rl-cell-verify-") as temp:
        private_directory = Path(temp)
        (private_directory / expected_filename).write_bytes(ledger_bytes)
        summary = legacy._read_ledger_artifact(
            private_directory,
            replay,
            protocol=core_protocol,
            scenario_name=scenario_name,
            symbol=symbol,
        )
    replay_for_validation = deepcopy(replay)
    replay_for_validation["ledger_trace_summary"] = summary
    legacy._validate_replay(
        replay_for_validation,
        protocol=core_protocol,
        scenario_name=scenario_name,
        symbol=symbol,
        symbol_index=symbol_index,
    )
    return cell, replay, ledger_bytes


def _validate_fit_for_context(
    root: Path,
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    dataset: Any,
) -> tuple[dict[str, Any], str]:
    manifest, digest = _validate_fit_directory(
        root,
        factor,
        seed,
        protocol,
        dataset_feature_names=tuple(dataset.feature_names),
    )
    return manifest, digest


def _load_verified_strategy(
    fit_directory: Path,
    fit_manifest: dict[str, Any],
    feature_names: tuple[str, ...],
) -> PPOIntentStrategy:
    bundle = fit_directory / "bundle"
    strategy = load_ppo_inference_bundle(
        bundle,
        expected_digest=fit_manifest["bundle_digest"],
        feature_names=feature_names,
    )
    actual_timesteps = _require_strict_int(
        getattr(strategy.policy, "num_timesteps", None),
        field="loaded PPO actual_timesteps",
        expected=fit_manifest["actual_timesteps"],
    )
    if actual_timesteps != fit_manifest["actual_timesteps"]:
        raise ValueError("loaded PPO timesteps differ from completed fit")
    if tuple(strategy.feature_indices) != tuple(fit_manifest["feature_indices"]):
        raise ValueError("loaded PPO feature indices differ from completed fit")
    return strategy


def replay_cell(
    source: Path,
    root: Path,
    factor: str,
    seed: int,
    scenario_name: str,
    symbol_index: int,
) -> Path:
    """Replay one scenario and one symbol using a privately verified frozen PPO."""
    protocol = _validate_checkpoint_protocol(source, root)
    core_protocol = protocol["core_protocol"]
    feature, scenario_name, symbol_index = _validate_cell_request(
        core_protocol, factor, seed, scenario_name, symbol_index
    )
    seed = _require_strict_int(seed, field="seed")
    with StudyStore(root).mutation_lock():
        _require_current_protocol(source, root, protocol)
        dataset, _base_config = _load_context(source, core_protocol)
        dataset_feature_names = tuple(dataset.feature_names)
        fit_manifest, fit_digest = _validate_fit_for_context(
            root, factor, seed, protocol, dataset
        )
        target = _cell_directory(root, factor, seed, scenario_name, symbol_index)
        if target.exists() or target.is_symlink():
            _read_cell_directory(
                target,
                factor=factor,
                seed=seed,
                scenario_name=scenario_name,
                symbol_index=symbol_index,
                protocol=protocol,
                fit_digest=fit_digest,
                dataset_feature_names=dataset_feature_names,
            )
            del dataset, _base_config
            _require_current_protocol(source, root, protocol)
            return target

        fit_directory = _fit_directory(root, factor, seed)
        strategy = _load_verified_strategy(
            fit_directory, fit_manifest, dataset_feature_names
        )
        _require_current_protocol(source, root, protocol)
        attempt = _attempt_directory(
            root,
            f"cell-{factor}-ppo{seed}-{scenario_name}-symbol-{symbol_index}",
        )
        scenario = core_protocol["evaluation"]["scenarios"][scenario_name]
        symbol = core_protocol["symbols"][symbol_index]
        feature_names = tuple(feature["feature_names"])

        factory = partial(
            PPOIntentStrategy,
            strategy.policy,
            feature_indices=tuple(fit_manifest["feature_indices"]),
            feature_names=strategy.feature_names,
            feature_normalizer=strategy.feature_normalizer,
        )

        replay = evaluate_directional_arm(
            dataset,
            factory,
            start_index=core_protocol["evaluation"]["start_index"],
            stop_index=core_protocol["evaluation"]["stop_index"],
            latency_bars=scenario["latency_bars"],
            cost_multiplier=scenario["cost_multiplier"],
            symbol_index=symbol_index,
            initial_capital=core_protocol["evaluation"]["initial_capital"],
            gross_budget=core_protocol["evaluation"]["gross_budget"],
            capture_ledger_evidence=True,
        )
        del dataset, strategy, factory
        _require_current_protocol(source, root, protocol)
        replay = legacy._persist_replay_ledger(
            attempt,
            replay,
            scenario_name=scenario_name,
            symbol_index=symbol_index,
        )
        filename = replay["ledger_evidence_file"]
        ledger_size = file_digest_and_size(
            attempt / filename, field="replay compressed ledger"
        )[1]
        cell_manifest = {
            "schema": CELL_SCHEMA,
            "factor": factor,
            "seed": seed,
            "scenario": scenario_name,
            "symbol_index": symbol_index,
            "symbol": symbol,
            "checkpoint_protocol_digest": content_digest(protocol),
            "core_protocol_digest": content_digest(core_protocol),
            "fit_digest": fit_digest,
            "dataset_id": core_protocol["evaluation_dataset_id"],
            "dataset_feature_names": list(dataset_feature_names),
            "feature_names": list(feature_names),
            "feature_indices": list(feature["feature_indices"]),
            "provenance": core_protocol["provenance"],
            "ledger_size_bytes": ledger_size,
            "replay": replay,
        }
        cell_raw = _write_once(attempt / "cell.json", cell_manifest)
        _write_json_sidecar(attempt / "cell.digest.json", raw=cell_raw)
        _read_cell_directory(
            attempt,
            factor=factor,
            seed=seed,
            scenario_name=scenario_name,
            symbol_index=symbol_index,
            protocol=protocol,
            fit_digest=fit_digest,
            dataset_feature_names=dataset_feature_names,
        )
        _require_current_protocol(source, root, protocol)
        _make_directory(
            root,
            ("cells", factor, f"ppo{seed}", scenario_name),
        )
        _publish_attempt_directory(attempt, target)
        _read_cell_directory(
            target,
            factor=factor,
            seed=seed,
            scenario_name=scenario_name,
            symbol_index=symbol_index,
            protocol=protocol,
            fit_digest=fit_digest,
            dataset_feature_names=dataset_feature_names,
        )
        _require_current_protocol(source, root, protocol)
        return target


def _required_cells(
    core_protocol: dict[str, Any], factor: str, seed: int
) -> list[tuple[str, int, str]]:
    feature, seed = _validate_factor_seed(core_protocol, factor, seed)
    del feature
    return [
        (scenario, index, symbol)
        for scenario in _scenario_roster(core_protocol, factor)
        for index, symbol in enumerate(core_protocol["symbols"])
    ]


def _load_required_cells(
    root: Path,
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    fit_digest: str,
    dataset_feature_names: tuple[str, ...],
) -> tuple[dict[str, dict[str, dict[str, Any]]], dict[str, dict[str, bytes]]]:
    core_protocol = protocol["core_protocol"]
    replay_rows: dict[str, dict[str, dict[str, Any]]] = {}
    ledgers: dict[str, dict[str, bytes]] = {}
    for scenario_name, symbol_index, symbol in _required_cells(
        core_protocol, factor, seed
    ):
        directory = _cell_directory(root, factor, seed, scenario_name, symbol_index)
        if not directory.exists() or directory.is_symlink():
            raise ValueError(
                f"checkpoint cell roster is incomplete: {scenario_name}/{symbol}"
            )
        _cell, replay, ledger = _read_cell_directory(
            directory,
            factor=factor,
            seed=seed,
            scenario_name=scenario_name,
            symbol_index=symbol_index,
            protocol=protocol,
            fit_digest=fit_digest,
            dataset_feature_names=dataset_feature_names,
        )
        replay_rows.setdefault(scenario_name, {})[symbol] = replay
        ledgers.setdefault(scenario_name, {})[replay["ledger_evidence_file"]] = ledger
    return replay_rows, ledgers


def _expected_arm_payload(
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    fit_manifest: dict[str, Any],
    fit_digest: str,
    replay_rows: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    core_protocol = protocol["core_protocol"]
    feature = core_protocol["factors"][factor]
    stress_names = (
        core_protocol["evaluation"]["stress_names"]
        if factor == legacy.CANDIDATE_FACTOR
        else []
    )
    return {
        "schema": "ppo_feature_arm_v1",
        "factor": factor,
        "seed": seed,
        "dataset_id": core_protocol["evaluation_dataset_id"],
        "start_index": core_protocol["evaluation"]["start_index"],
        "stop_index": core_protocol["evaluation"]["stop_index"],
        "feature_names": feature["feature_names"],
        "feature_indices": feature["feature_indices"],
        "training_risk": core_protocol["training"]["risk"],
        "requested_timesteps": core_protocol["training"]["requested_timesteps"],
        "actual_timesteps": fit_manifest["actual_timesteps"],
        "dataset_feature_names": fit_manifest["dataset_feature_names"],
        "by_symbol": replay_rows["base"],
        "stress_by_name": {name: replay_rows[name] for name in stress_names},
        "model_sha256": {"model.zip": fit_manifest["policy_sha256"]},
        "model_size_bytes": fit_manifest["policy_size_bytes"],
        "protocol_digest": content_digest(core_protocol),
        "provenance": core_protocol["provenance"],
        "checkpoint_protocol_digest": content_digest(protocol),
        "fit_digest": fit_digest,
    }


def _cell_manifest_digest(directory: Path) -> str:
    cell = _read_verified_json(
        directory / "cell.json",
        directory / "cell.digest.json",
        field="replay cell manifest",
    )
    return content_digest(cell)


def _build_arm_evidence(
    root: Path,
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    dataset_feature_names: tuple[str, ...],
) -> tuple[dict[str, Any], bytes, bytes, dict[str, dict[str, bytes]]]:
    fit_manifest, fit_digest = _validate_fit_directory(
        root,
        factor,
        seed,
        protocol,
        dataset_feature_names=dataset_feature_names,
    )
    replay_rows, ledger_rosters = _load_required_cells(
        root,
        factor,
        seed,
        protocol,
        fit_digest,
        dataset_feature_names,
    )
    model_bytes = read_verified_bytes(
        _fit_directory(root, factor, seed) / "bundle" / "policy.zip",
        expected_digest=fit_manifest["policy_sha256"],
        expected_size_bytes=fit_manifest["policy_size_bytes"],
        field="PPO inference policy",
    )
    arm = _expected_arm_payload(
        factor, seed, protocol, fit_manifest, fit_digest, replay_rows
    )
    arm["cell_digests"] = {
        scenario: {
            symbol: _cell_manifest_digest(
                _cell_directory(
                    root,
                    factor,
                    seed,
                    scenario,
                    tuple(protocol["core_protocol"]["symbols"]).index(symbol),
                )
            )
            for symbol in rows
        }
        for scenario, rows in replay_rows.items()
    }
    result_raw = canonical_json_bytes(arm)
    return arm, result_raw, model_bytes, ledger_rosters


def _legacy_validate_arm_bytes(
    factor: str,
    seed: int,
    core_protocol: dict[str, Any],
    result_raw: bytes,
    model_bytes: bytes,
    ledger_rosters: dict[str, dict[str, bytes]],
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="trade-rl-assembled-arm-") as temp:
        private_root = Path(temp)
        directory = private_root / factor / f"ppo{seed}"
        directory.mkdir(parents=True)
        (directory / "result.json").write_bytes(result_raw)
        (directory / "result.sha256.json").write_bytes(
            canonical_json_bytes({"sha256": _sha256(result_raw)})
        )
        (directory / "model.zip").write_bytes(model_bytes)
        for scenario in ledger_rosters.values():
            for filename, raw in scenario.items():
                (directory / filename).write_bytes(raw)
        return legacy._read_arm(private_root, factor, seed, core_protocol)


def _validate_arm_directory(
    directory: Path,
    *,
    factor: str,
    seed: int,
    protocol: dict[str, Any],
    result_raw: bytes,
    model_bytes: bytes,
    ledger_rosters: dict[str, dict[str, bytes]],
) -> dict[str, Any]:
    _require_directory(directory, field="assembled arm")
    expected_files = {"result.json", "result.sha256.json", "model.zip"}
    expected_files.update(
        filename for scenario in ledger_rosters.values() for filename in scenario
    )
    if {path.name for path in directory.iterdir()} != expected_files:
        raise ValueError("assembled arm file roster differs from completed cells")
    result_path = directory / "result.json"
    result_bytes = read_verified_bytes(
        result_path,
        expected_digest=_sha256(result_raw),
        expected_size_bytes=len(result_raw),
        field="assembled arm result",
    )
    if result_bytes != result_raw:
        raise ValueError("assembled arm result differs from its checkpoint cells")
    sidecar = _parse_canonical_json(
        _read_regular_bytes(
            directory / "result.sha256.json", field="assembled arm result digest"
        ),
        field="assembled arm result digest",
    )
    if sidecar != {"sha256": _sha256(result_raw)}:
        raise ValueError("assembled arm result digest mismatch")
    model_path = directory / "model.zip"
    model_digest = _sha256(model_bytes)
    copied_model = read_verified_bytes(
        model_path,
        expected_digest=model_digest,
        expected_size_bytes=len(model_bytes),
        field="assembled arm model",
    )
    if copied_model != model_bytes:
        raise ValueError("assembled arm model differs from completed fit")
    for scenario in ledger_rosters.values():
        for filename, expected_raw in scenario.items():
            copied = read_verified_bytes(
                directory / filename,
                expected_digest=_sha256(expected_raw),
                expected_size_bytes=len(expected_raw),
                field="assembled arm replay ledger",
            )
            if copied != expected_raw:
                raise ValueError("assembled arm ledger differs from completed cell")
    return _legacy_validate_arm_bytes(
        factor,
        seed,
        protocol["core_protocol"],
        result_raw,
        model_bytes,
        ledger_rosters,
    )


def assemble_arm(source: Path, root: Path, factor: str, seed: int) -> Path:
    """Validate complete cells and atomically assemble one legacy-compatible arm."""
    protocol = _validate_checkpoint_protocol(source, root)
    core_protocol = protocol["core_protocol"]
    _validate_factor_seed(core_protocol, factor, seed)
    with StudyStore(root).mutation_lock():
        _require_current_protocol(source, root, protocol)
        dataset, _base_config = _load_context(source, core_protocol)
        dataset_feature_names = tuple(dataset.feature_names)
        del dataset, _base_config
        arm, result_raw, model_bytes, ledger_rosters = _build_arm_evidence(
            root, factor, seed, protocol, dataset_feature_names
        )
        target = root / "arms" / factor / f"ppo{seed}"
        if target.exists() or target.is_symlink():
            _validate_arm_directory(
                target,
                factor=factor,
                seed=seed,
                protocol=protocol,
                result_raw=result_raw,
                model_bytes=model_bytes,
                ledger_rosters=ledger_rosters,
            )
            _require_current_protocol(source, root, protocol)
            return target

        attempt = _attempt_directory(root, f"arm-{factor}-ppo{seed}")
        (attempt / "result.json").write_bytes(result_raw)
        (attempt / "result.sha256.json").write_bytes(
            canonical_json_bytes({"sha256": _sha256(result_raw)})
        )
        (attempt / "model.zip").write_bytes(model_bytes)
        for scenario in ledger_rosters.values():
            for filename, raw in scenario.items():
                (attempt / filename).write_bytes(raw)
        _validate_arm_directory(
            attempt,
            factor=factor,
            seed=seed,
            protocol=protocol,
            result_raw=result_raw,
            model_bytes=model_bytes,
            ledger_rosters=ledger_rosters,
        )
        _require_current_protocol(source, root, protocol)
        arms_parent = _make_directory(root, ("arms", factor))
        _publish_attempt_directory(attempt, arms_parent / f"ppo{seed}")
        _validate_arm_directory(
            target,
            factor=factor,
            seed=seed,
            protocol=protocol,
            result_raw=result_raw,
            model_bytes=model_bytes,
            ledger_rosters=ledger_rosters,
        )
        _require_current_protocol(source, root, protocol)
        return target


def _comparison_payload(
    root: Path,
    protocol: dict[str, Any],
    dataset_feature_names: tuple[str, ...],
) -> dict[str, Any]:
    core_protocol = protocol["core_protocol"]
    rows: dict[str, dict[int, dict[str, Any]]] = {}
    result_hashes: dict[str, dict[str, str]] = {}
    model_hashes: dict[str, dict[str, str]] = {}
    fit_digests: dict[str, dict[str, str]] = {}
    cell_digests: dict[str, dict[str, dict[str, dict[str, str]]]] = {}
    timesteps: dict[str, dict[str, int]] = {}
    for factor in legacy.FACTORS:
        rows[factor] = {}
        result_hashes[factor] = {}
        model_hashes[factor] = {}
        fit_digests[factor] = {}
        cell_digests[factor] = {}
        timesteps[factor] = {}
        for seed in core_protocol["seeds"]:
            seed = _require_strict_int(seed, field="seed")
            fit_manifest, fit_digest = _validate_fit_directory(
                root,
                factor,
                seed,
                protocol,
                dataset_feature_names=dataset_feature_names,
            )
            arm, result_raw, model_bytes, ledger_rosters = _build_arm_evidence(
                root, factor, seed, protocol, dataset_feature_names
            )
            directory = root / "arms" / factor / f"ppo{seed}"
            if not directory.exists() or directory.is_symlink():
                raise ValueError(
                    f"assembled arm roster is incomplete: {factor}/ppo{seed}"
                )
            rows[factor][seed] = _validate_arm_directory(
                directory,
                factor=factor,
                seed=seed,
                protocol=protocol,
                result_raw=result_raw,
                model_bytes=model_bytes,
                ledger_rosters=ledger_rosters,
            )
            result_hashes[factor][f"ppo{seed}"] = _sha256(result_raw)
            model_hashes[factor][f"ppo{seed}"] = _sha256(model_bytes)
            fit_digests[factor][f"ppo{seed}"] = fit_digest
            timesteps[factor][f"ppo{seed}"] = fit_manifest["actual_timesteps"]
            cell_digests[factor][f"ppo{seed}"] = {
                scenario: dict(symbols)
                for scenario, symbols in arm["cell_digests"].items()
            }
    comparison = legacy.compare_feature_ablation(
        rows[legacy.BASELINE_FACTOR],
        rows[legacy.CANDIDATE_FACTOR],
        core_protocol,
    )
    comparison.update(
        checkpoint_protocol_digest=content_digest(protocol),
        core_protocol_digest=content_digest(core_protocol),
        result_sha256=result_hashes,
        model_sha256=model_hashes,
        fit_digests=fit_digests,
        cell_digests=cell_digests,
        actual_timesteps=timesteps,
    )
    return comparison


def _read_comparison_directory(
    directory: Path, expected: dict[str, Any]
) -> tuple[bytes, bytes]:
    _require_directory(directory, field="completed comparison")
    if {path.name for path in directory.iterdir()} != {
        "comparison.json",
        "comparison.digest.json",
    }:
        raise ValueError("completed comparison file roster is unexpected")
    expected_raw = canonical_json_bytes(expected)
    raw = read_verified_bytes(
        directory / "comparison.json",
        expected_digest=_sha256(expected_raw),
        expected_size_bytes=len(expected_raw),
        field="completed comparison",
    )
    if raw != expected_raw:
        raise ValueError("completed comparison differs from current validated arms")
    digest_raw = _read_regular_bytes(
        directory / "comparison.digest.json", field="comparison digest"
    )
    digest = _parse_canonical_json(digest_raw, field="comparison digest")
    if digest != {"digest": content_digest(expected)}:
        raise ValueError("completed comparison digest mismatch")
    return raw, digest_raw


def finalize_checkpoint_study(source: Path, root: Path) -> Path:
    """Require all fits, cells, and arms, then publish the fixed legacy decision."""
    protocol = _validate_checkpoint_protocol(source, root)
    core_protocol = protocol["core_protocol"]
    with StudyStore(root).mutation_lock():
        _require_current_protocol(source, root, protocol)
        dataset, _base_config = _load_context(source, core_protocol)
        dataset_feature_names = tuple(dataset.feature_names)
        del dataset, _base_config
        comparison = _comparison_payload(root, protocol, dataset_feature_names)
        target = root / "comparison"
        if target.exists() or target.is_symlink():
            _read_comparison_directory(target, comparison)
            _require_current_protocol(source, root, protocol)
            return target

        attempt = _attempt_directory(root, "comparison")
        comparison_raw = _write_once(attempt / "comparison.json", comparison)
        _write_once(
            attempt / "comparison.digest.json",
            {"digest": content_digest(comparison)},
        )
        if comparison_raw != canonical_json_bytes(comparison):
            raise ValueError("comparison staging bytes are not canonical")
        _read_comparison_directory(attempt, comparison)
        _require_current_protocol(source, root, protocol)
        _publish_attempt_directory(attempt, target)
        _read_comparison_directory(target, comparison)
        _require_current_protocol(source, root, protocol)
        return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare_parser = commands.add_parser("prepare")
    prepare_parser.add_argument("--source", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    for name in ("fit", "replay-cell", "assemble-arm", "finalize"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name != "finalize":
            command.add_argument("--factor", choices=legacy.FACTORS, required=True)
            command.add_argument("--seed", type=int, required=True)
        if name == "replay-cell":
            command.add_argument(
                "--scenario", choices=("base", "cost_2x", "latency_1"), required=True
            )
            command.add_argument("--symbol-index", type=int, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        prepare_checkpoint_study(args.source, args.output)
    elif args.command == "fit":
        fit_checkpoint(args.source, args.output, args.factor, args.seed)
    elif args.command == "replay-cell":
        replay_cell(
            args.source,
            args.output,
            args.factor,
            args.seed,
            args.scenario,
            args.symbol_index,
        )
    elif args.command == "assemble-arm":
        assemble_arm(args.source, args.output, args.factor, args.seed)
    else:
        finalize_checkpoint_study(args.source, args.output)


if __name__ == "__main__":
    main()


__all__ = [
    "assemble_arm",
    "expected_checkpoint_protocol",
    "finalize_checkpoint_study",
    "fit_checkpoint",
    "main",
    "prepare_checkpoint_study",
    "replay_cell",
    "validate_checkpoint_protocol",
]
