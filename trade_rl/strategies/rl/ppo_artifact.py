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
