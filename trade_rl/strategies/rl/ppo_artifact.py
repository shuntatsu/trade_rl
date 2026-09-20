"""Write-once normalized PPO bundles bound to policy bytes and preprocessing."""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from hashlib import sha256
from pathlib import Path

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.strategies.rl.ppo import (
    PPOIntentStrategy,
    ppo_observation_contract_payload,
)
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

_SCHEMA = "ppo_normalized_model_v1"
_INFERENCE_SCHEMA = "ppo_inference_bundle_v1"


def save_normalized_ppo(root: Path, strategy: PPOIntentStrategy) -> str:
    """Publish a complete normalized policy; return the digest callers must pin."""
    normalizer = strategy.feature_normalizer
    if normalizer is None:
        raise ValueError("a normalized model must include its fitted normalizer")
    normalizer.validate_features(strategy.feature_indices)
    if root.exists():
        raise FileExistsError(f"normalized PPO destination already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        policy_path = staging / "policy.zip"
        getattr(strategy.policy, "save")(str(policy_path))
        manifest = {
            "schema": _SCHEMA,
            "observation": ppo_observation_contract_payload(),
            "normalizer": normalizer.to_payload(),
            "policy_sha256": sha256(policy_path.read_bytes()).hexdigest(),
        }
        encoded = canonical_json_bytes(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def load_normalized_ppo(
    root: Path, *, expected_digest: str, feature_names: tuple[str, ...]
) -> PPOIntentStrategy:
    """Verify the pinned bundle and feed schema before deserializing a policy."""
    require_sha256(expected_digest, field="expected_digest")
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    if (
        not isinstance(manifest, dict)
        or set(manifest) != {"schema", "observation", "normalizer", "policy_sha256"}
        or manifest["schema"] != _SCHEMA
        or manifest["observation"] != ppo_observation_contract_payload()
        or content_digest(manifest) != expected_digest
        or canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("normalized model manifest differs from its pinned contract")
    normalizer = PPOFeatureNormalizer.from_payload(manifest["normalizer"])
    if (
        max(normalizer.feature_indices) >= len(feature_names)
        or tuple(feature_names[index] for index in normalizer.feature_indices)
        != normalizer.feature_names
    ):
        raise ValueError("feed feature schema differs from fitted normalization")
    policy_path = root / "policy.zip"
    if sha256(policy_path.read_bytes()).hexdigest() != manifest["policy_sha256"]:
        raise ValueError("policy bytes differ from the normalized model manifest")
    module = importlib.import_module("stable_baselines3")
    torch_module = importlib.import_module("torch")
    getattr(torch_module, "set_num_threads")(1)
    model = getattr(module, "PPO").load(str(policy_path), device="cpu")
    if (
        tuple(model.observation_space.shape)
        != (3 * len(normalizer.feature_indices) + 2,)
        or model.action_space.n != 3
        or model.action_space.start != 0
    ):
        raise ValueError("saved policy spaces differ from the PPO contract")
    return PPOIntentStrategy(
        model,
        feature_indices=normalizer.feature_indices,
        feature_normalizer=normalizer,
    )



def _validated_feed_feature_names(
    feature_names: tuple[str, ...],
    feature_indices: tuple[int, ...],
) -> tuple[str, ...]:
    names = tuple(feature_names)
    if (
        not names
        or len(set(names)) != len(names)
        or any(not isinstance(name, str) or not name for name in names)
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


def _load_ppo_policy(
    policy_path: Path,
    *,
    feature_count: int,
) -> object:
    module = importlib.import_module("stable_baselines3")
    torch_module = importlib.import_module("torch")
    getattr(torch_module, "set_num_threads")(1)
    model = getattr(module, "PPO").load(str(policy_path), device="cpu")
    if (
        tuple(model.observation_space.shape) != (3 * feature_count + 2,)
        or model.action_space.n != 3
        or model.action_space.start != 0
    ):
        raise ValueError("saved policy spaces differ from the PPO contract")
    return model


def save_ppo_inference_bundle(
    root: Path,
    strategy: PPOIntentStrategy,
    *,
    feature_names: tuple[str, ...],
) -> str:
    """Publish one inference-safe PPO bundle for raw or normalized policies."""

    indices = tuple(strategy.feature_indices)
    selected_names = _validated_feed_feature_names(feature_names, indices)
    normalizer = strategy.feature_normalizer
    if normalizer is not None:
        normalizer.validate_features(indices)
        if normalizer.feature_names != selected_names:
            raise ValueError(
                "normalizer feature schema differs from the inference feed schema"
            )
    if root.exists():
        raise FileExistsError(f"PPO inference bundle destination already exists: {root}")
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        policy_path = staging / "policy.zip"
        getattr(strategy.policy, "save")(str(policy_path))
        manifest = {
            "schema": _INFERENCE_SCHEMA,
            "observation": ppo_observation_contract_payload(),
            "feature_indices": list(indices),
            "feature_names": list(selected_names),
            "normalizer": None if normalizer is None else normalizer.to_payload(),
            "policy_sha256": sha256(policy_path.read_bytes()).hexdigest(),
        }
        encoded = canonical_json_bytes(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        staging.rename(root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def load_ppo_inference_bundle(
    root: Path,
    *,
    expected_digest: str,
    feature_names: tuple[str, ...],
) -> PPOIntentStrategy:
    """Verify a PPO inference bundle and current feed schema before policy load."""

    require_sha256(expected_digest, field="expected_digest")
    raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    expected_keys = {
        "schema",
        "observation",
        "feature_indices",
        "feature_names",
        "normalizer",
        "policy_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or set(manifest) != expected_keys
        or manifest["schema"] != _INFERENCE_SCHEMA
        or manifest["observation"] != ppo_observation_contract_payload()
        or content_digest(manifest) != expected_digest
        or canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("PPO inference manifest differs from its pinned contract")

    raw_indices = manifest["feature_indices"]
    raw_names = manifest["feature_names"]
    if not isinstance(raw_indices, list) or not isinstance(raw_names, list):
        raise ValueError("PPO inference feature schema must use JSON arrays")
    indices = tuple(raw_indices)
    selected_names = tuple(raw_names)
    if (
        not indices
        or len(indices) != len(selected_names)
        or len(set(indices)) != len(indices)
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or any(not isinstance(name, str) or not name for name in selected_names)
    ):
        raise ValueError("PPO inference feature schema is malformed")

    current_selected = _validated_feed_feature_names(feature_names, indices)
    if current_selected != selected_names:
        raise ValueError("feed feature schema differs from the PPO inference bundle")

    raw_normalizer = manifest["normalizer"]
    normalizer: PPOFeatureNormalizer | None
    if raw_normalizer is None:
        normalizer = None
    else:
        if not isinstance(raw_normalizer, dict):
            raise ValueError("PPO inference normalizer payload is malformed")
        normalizer = PPOFeatureNormalizer.from_payload(raw_normalizer)
        normalizer.validate_features(indices)
        if normalizer.feature_names != selected_names:
            raise ValueError(
                "PPO inference normalizer feature schema differs from the bundle"
            )

    policy_path = root / "policy.zip"
    if sha256(policy_path.read_bytes()).hexdigest() != manifest["policy_sha256"]:
        raise ValueError("policy bytes differ from the PPO inference manifest")
    model = _load_ppo_policy(policy_path, feature_count=len(indices))
    return PPOIntentStrategy(
        model,
        feature_indices=indices,
        feature_normalizer=normalizer,
    )
