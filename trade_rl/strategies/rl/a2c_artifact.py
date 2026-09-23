"""Write-once inference bundles for A2C policies."""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from numbers import Integral
from pathlib import Path
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.verified_file import (
    file_digest,
    open_regular_binary,
    verified_private_copy,
)
from trade_rl.strategies.rl.a2c import A2CFitMetadata, A2CIntentStrategy
from trade_rl.strategies.rl.intent import ppo_observation_contract_payload
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

_SCHEMA = "a2c_inference_bundle_v1"


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    with open_regular_binary(path, field=field) as stream:
        return stream.read()


def _validated_feed_feature_names(
    feature_names: tuple[str, ...],
    feature_indices: tuple[int, ...],
) -> tuple[str, ...]:
    names = tuple(feature_names)
    if (
        not names
        or any(not isinstance(name, str) or not name for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError("feature_names must be non-empty unique strings")
    indices = tuple(feature_indices)
    if (
        not indices
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or max(indices) >= len(names)
    ):
        raise ValueError("policy feature indices are outside the feed feature schema")
    return tuple(names[index] for index in indices)


def _matches_integer(value: object, expected: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and int(value) == expected
    )


def _validate_policy_spaces(policy: object, *, feature_count: int) -> None:
    try:
        observation_shape = tuple(
            getattr(getattr(policy, "observation_space"), "shape")
        )
        action_space = getattr(policy, "action_space")
        action_count = getattr(action_space, "n")
        action_start = getattr(action_space, "start")
    except (AttributeError, TypeError) as error:
        raise ValueError("policy spaces differ from the A2C contract") from error
    if (
        len(observation_shape) != 1
        or not _matches_integer(observation_shape[0], 3 * feature_count + 2)
        or not _matches_integer(action_count, 3)
        or not _matches_integer(action_start, 0)
    ):
        raise ValueError("policy spaces differ from the A2C contract")


def _require_a2c_policy_family(policy: object) -> None:
    try:
        a2c_type = getattr(importlib.import_module("stable_baselines3"), "A2C")
    except (ImportError, AttributeError) as error:
        raise RuntimeError(
            "stable-baselines3 is required; install the train-sb3 extra"
        ) from error
    if not isinstance(a2c_type, type) or not isinstance(policy, a2c_type):
        raise ValueError("A2C policy family does not match the A2C bundle schema")


def _validate_policy_timesteps(
    policy: object,
    fit_metadata: A2CFitMetadata,
) -> None:
    try:
        timesteps = getattr(policy, "num_timesteps")
    except AttributeError as error:
        raise ValueError("A2C policy timesteps differ from fit metadata") from error
    if (
        isinstance(timesteps, bool)
        or not isinstance(timesteps, int)
        or timesteps != fit_metadata.effective_timesteps
    ):
        raise ValueError("A2C policy timesteps differ from fit metadata")


def _validate_policy_seed(
    policy: object,
    fit_metadata: A2CFitMetadata,
) -> None:
    try:
        seed = getattr(policy, "seed")
    except AttributeError as error:
        raise ValueError("A2C policy seed differs from fit metadata") from error
    if (
        isinstance(seed, bool)
        or not isinstance(seed, int)
        or seed != fit_metadata.seed
    ):
        raise ValueError("A2C policy seed differs from fit metadata")


def _validated_manifest(
    root: Path,
    *,
    expected_digest: str,
    feature_names: tuple[str, ...],
) -> tuple[
    tuple[int, ...],
    tuple[str, ...],
    PPOFeatureNormalizer | None,
    A2CFitMetadata,
    str,
]:
    require_sha256(expected_digest, field="expected_digest")
    raw = _read_regular_bytes(
        root / "manifest.json",
        field="A2C inference manifest",
    )
    manifest = json.loads(raw)
    expected_keys = {
        "schema",
        "algorithm",
        "observation",
        "feature_indices",
        "feature_names",
        "normalizer",
        "fit_metadata",
        "policy_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest["schema"] != _SCHEMA
        or manifest["algorithm"] != "a2c"
        or manifest["observation"] != ppo_observation_contract_payload()
        or content_digest(manifest) != expected_digest
        or canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("A2C inference manifest differs from its pinned contract")

    raw_indices = manifest["feature_indices"]
    raw_names = manifest["feature_names"]
    if not isinstance(raw_indices, list) or not isinstance(raw_names, list):
        raise ValueError("A2C inference feature schema must use JSON arrays")
    indices = tuple(raw_indices)
    selected_names = tuple(raw_names)
    if (
        not indices
        or len(indices) != len(selected_names)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or len(set(indices)) != len(indices)
        or any(not isinstance(name, str) or not name for name in selected_names)
        or len(set(selected_names)) != len(selected_names)
    ):
        raise ValueError("A2C inference feature schema is malformed")
    if _validated_feed_feature_names(feature_names, indices) != selected_names:
        raise ValueError("feed feature schema differs from the A2C inference bundle")

    raw_normalizer = manifest["normalizer"]
    normalizer: PPOFeatureNormalizer | None
    if raw_normalizer is None:
        normalizer = None
    else:
        if not isinstance(raw_normalizer, dict):
            raise ValueError("A2C inference normalizer payload is malformed")
        normalizer = PPOFeatureNormalizer.from_payload(raw_normalizer)
        normalizer.validate_features(indices)
        if normalizer.feature_names != selected_names:
            raise ValueError(
                "A2C inference normalizer feature schema differs from the bundle"
            )

    raw_fit = manifest["fit_metadata"]
    if raw_fit is None:
        raise ValueError("A2C inference bundle requires fit metadata")
    fit_metadata = A2CFitMetadata.from_payload(raw_fit)
    policy_digest = manifest["policy_sha256"]
    require_sha256(policy_digest, field="A2C policy digest")
    return indices, selected_names, normalizer, fit_metadata, policy_digest


def save_a2c_inference_bundle(
    root: Path,
    strategy: A2CIntentStrategy,
    *,
    feature_names: tuple[str, ...],
) -> str:
    """Publish one canonical A2C inference bundle and return its pinned digest."""

    _require_a2c_policy_family(strategy.policy)
    if strategy.fit_metadata is None:
        raise ValueError("A2C inference bundle requires fit metadata")
    indices = tuple(strategy.feature_indices)
    selected_names = _validated_feed_feature_names(feature_names, indices)
    if strategy.feature_names is None:
        raise ValueError(
            "A2C strategy feature schema must be bound before inference publication"
        )
    if tuple(strategy.feature_names) != selected_names:
        raise ValueError("strategy feature schema differs from the A2C inference feed")
    normalizer = strategy.feature_normalizer
    if normalizer is not None:
        normalizer.validate_features(indices)
        if normalizer.feature_names != selected_names:
            raise ValueError("normalizer schema differs from A2C inference feed")
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            f"A2C inference bundle destination already exists: {root}"
        )
    _validate_policy_spaces(strategy.policy, feature_count=len(indices))
    _validate_policy_timesteps(strategy.policy, strategy.fit_metadata)
    _validate_policy_seed(strategy.policy, strategy.fit_metadata)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        policy_path = staging / "policy.zip"
        getattr(strategy.policy, "save")(str(policy_path))
        manifest = {
            "schema": _SCHEMA,
            "algorithm": "a2c",
            "observation": ppo_observation_contract_payload(),
            "feature_indices": list(indices),
            "feature_names": list(selected_names),
            "normalizer": None if normalizer is None else normalizer.to_payload(),
            "fit_metadata": strategy.fit_metadata.to_payload(),
            "policy_sha256": file_digest(policy_path, field="A2C inference policy"),
        }
        encoded = canonical_json_bytes(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def _load_a2c_policy(policy_path: Path, *, feature_count: int) -> Any:
    module = importlib.import_module("stable_baselines3")
    torch_module = importlib.import_module("torch")
    getattr(torch_module, "set_num_threads")(1)
    model = getattr(module, "A2C").load(str(policy_path), device="cpu")
    _validate_policy_spaces(model, feature_count=feature_count)
    return model


def load_a2c_inference_bundle(
    root: Path,
    *,
    expected_digest: str,
    feature_names: tuple[str, ...],
) -> A2CIntentStrategy:
    """Verify A2C family, schema, feed, and private policy bytes before loading."""

    indices, selected_names, normalizer, metadata, policy_digest = _validated_manifest(
        root,
        expected_digest=expected_digest,
        feature_names=feature_names,
    )
    with verified_private_copy(
        root / "policy.zip",
        expected_digest=policy_digest,
        field="A2C inference policy",
        filename="policy.zip",
    ) as verified_policy:
        model = _load_a2c_policy(verified_policy, feature_count=len(indices))
    _validate_policy_timesteps(model, metadata)
    _validate_policy_seed(model, metadata)
    return A2CIntentStrategy(
        model,
        feature_indices=indices,
        feature_names=selected_names,
        feature_normalizer=normalizer,
        fit_metadata=metadata,
    )


__all__ = ["load_a2c_inference_bundle", "save_a2c_inference_bundle"]
