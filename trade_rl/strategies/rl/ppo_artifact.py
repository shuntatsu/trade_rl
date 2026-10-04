"""Write-once normalized PPO bundles bound to policy bytes and preprocessing."""

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
from trade_rl.artifacts.atomic_write import atomic_rename_directory
from trade_rl.artifacts.verified_file import (
    file_digest,
    open_regular_binary,
    verified_private_copy,
)
from trade_rl.strategies.rl.intent import (
    PPO_OBSERVATION_SCHEMA,
    PPO_OBSERVATION_SCHEMA_V3,
    PPO_OBSERVATION_SCHEMAS,
    ppo_observation_contract_payload,
)
from trade_rl.strategies.rl.ppo import PPOIntentStrategy
from trade_rl.strategies.rl.ppo_normalization import PPOFeatureNormalizer

_SCHEMA = "ppo_normalized_model_v1"
_INFERENCE_SCHEMA = "ppo_inference_bundle_v1"
_INFERENCE_SCHEMA_V2 = "ppo_inference_bundle_v2"


def _read_regular_bytes(path: Path, *, field: str) -> bytes:
    with open_regular_binary(path, field=field) as stream:
        return stream.read()


def _matches_integer(value: object, expected: int) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, Integral)
        and int(value) == expected
    )


def _validate_policy_spaces(
    policy: object,
    *,
    feature_count: int,
    observation_schema: str = PPO_OBSERVATION_SCHEMA,
) -> None:
    try:
        observation_shape = tuple(
            getattr(getattr(policy, "observation_space"), "shape")
        )
        action_space = getattr(policy, "action_space")
        action_count = getattr(action_space, "n")
        action_start = getattr(action_space, "start")
    except (AttributeError, TypeError) as error:
        raise ValueError("policy spaces differ from the PPO contract") from error
    if (
        len(observation_shape) != 1
        or not _matches_integer(
            observation_shape[0],
            3 * feature_count
            + 2
            + int(observation_schema == PPO_OBSERVATION_SCHEMA_V3),
        )
        or not _matches_integer(action_count, 3)
        or not _matches_integer(action_start, 0)
    ):
        raise ValueError("policy spaces differ from the PPO contract")


def save_normalized_ppo(root: Path, strategy: PPOIntentStrategy) -> str:
    """Publish a complete normalized policy; return the digest callers must pin."""
    normalizer = strategy.feature_normalizer
    if normalizer is None:
        raise ValueError("a normalized model must include its fitted normalizer")
    if (
        strategy.observation_schema != PPO_OBSERVATION_SCHEMA
        or strategy.minimum_hold_bars != 0
    ):
        raise ValueError(
            "normalized PPO v1 only supports Observation v2 with no minimum hold"
        )
    normalizer.validate_features(strategy.feature_indices)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"normalized PPO destination already exists: {root}")
    _validate_policy_spaces(
        strategy.policy,
        feature_count=len(normalizer.feature_indices),
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        policy_path = staging / "policy.zip"
        getattr(strategy.policy, "save")(str(policy_path))
        manifest: dict[str, object] = {
            "schema": _SCHEMA,
            "observation": ppo_observation_contract_payload(),
            "normalizer": normalizer.to_payload(),
            "policy_sha256": file_digest(
                policy_path,
                field="normalized PPO policy",
            ),
        }
        encoded = canonical_json_bytes(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        atomic_rename_directory(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def load_normalized_ppo(
    root: Path, *, expected_digest: str, feature_names: tuple[str, ...]
) -> PPOIntentStrategy:
    """Verify the pinned bundle and feed schema before deserializing a policy."""
    require_sha256(expected_digest, field="expected_digest")
    raw = _read_regular_bytes(
        root / "manifest.json",
        field="normalized PPO manifest",
    )
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
    with verified_private_copy(
        policy_path,
        expected_digest=manifest["policy_sha256"],
        field="normalized PPO policy",
        filename="policy.zip",
    ) as verified_policy:
        model = _load_ppo_policy(
            verified_policy,
            feature_count=len(normalizer.feature_indices),
        )
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


def _load_ppo_policy(
    policy_path: Path,
    *,
    feature_count: int,
    observation_schema: str = PPO_OBSERVATION_SCHEMA,
) -> Any:
    module = importlib.import_module("stable_baselines3")
    torch_module = importlib.import_module("torch")
    getattr(torch_module, "set_num_threads")(1)
    model = getattr(module, "PPO").load(str(policy_path), device="cpu")
    _validate_policy_spaces(
        model,
        feature_count=feature_count,
        observation_schema=observation_schema,
    )
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
    if strategy.feature_names is None:
        raise ValueError(
            "PPO strategy feature schema must be bound before inference publication"
        )
    if tuple(strategy.feature_names) != selected_names:
        raise ValueError(
            "strategy feature schema differs from the inference feed schema"
        )
    observation_schema = strategy.observation_schema
    minimum_hold_bars = strategy.minimum_hold_bars
    if observation_schema not in PPO_OBSERVATION_SCHEMAS:
        raise ValueError("unsupported PPO observation schema")
    if minimum_hold_bars > 0 and observation_schema != PPO_OBSERVATION_SCHEMA_V3:
        raise ValueError("minimum hold requires PPO Observation v3")
    normalizer = strategy.feature_normalizer
    if normalizer is not None:
        normalizer.validate_features(indices)
        if normalizer.feature_names != selected_names:
            raise ValueError(
                "normalizer feature schema differs from the inference feed schema"
            )
    if root.exists() or root.is_symlink():
        raise FileExistsError(
            f"PPO inference bundle destination already exists: {root}"
        )
    _validate_policy_spaces(
        strategy.policy,
        feature_count=len(indices),
        observation_schema=observation_schema,
    )
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=str(root.parent))
    )
    try:
        policy_path = staging / "policy.zip"
        getattr(strategy.policy, "save")(str(policy_path))
        manifest: dict[str, object] = {
            "schema": (
                _INFERENCE_SCHEMA
                if observation_schema == PPO_OBSERVATION_SCHEMA
                else _INFERENCE_SCHEMA_V2
            ),
            "observation": ppo_observation_contract_payload(observation_schema),
            "feature_indices": list(indices),
            "feature_names": list(selected_names),
            "normalizer": None if normalizer is None else normalizer.to_payload(),
            "policy_sha256": file_digest(
                policy_path,
                field="PPO inference policy",
            ),
        }
        if observation_schema != PPO_OBSERVATION_SCHEMA:
            manifest["minimum_hold_bars"] = minimum_hold_bars
        encoded = canonical_json_bytes(manifest)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(encoded)
        atomic_rename_directory(staging, root)
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
    raw = _read_regular_bytes(
        root / "manifest.json",
        field="PPO inference manifest",
    )
    manifest = json.loads(raw)
    legacy_keys = {
        "schema",
        "observation",
        "feature_indices",
        "feature_names",
        "normalizer",
        "policy_sha256",
    }
    if (
        not isinstance(manifest, dict)
        or content_digest(manifest) != expected_digest
        or canonical_json_bytes(manifest) != raw
    ):
        raise ValueError("PPO inference manifest differs from its pinned contract")

    schema = manifest.get("schema")
    if schema == _INFERENCE_SCHEMA:
        if set(manifest) != legacy_keys or manifest[
            "observation"
        ] != ppo_observation_contract_payload(PPO_OBSERVATION_SCHEMA):
            raise ValueError("PPO inference manifest differs from its pinned contract")
        observation_schema = PPO_OBSERVATION_SCHEMA
        minimum_hold_bars = 0
    elif schema == _INFERENCE_SCHEMA_V2:
        if set(manifest) != legacy_keys | {"minimum_hold_bars"}:
            raise ValueError("PPO inference manifest differs from its pinned contract")
        raw_observation = manifest["observation"]
        if not isinstance(raw_observation, dict):
            raise ValueError("PPO inference observation contract is malformed")
        observation_schema_value = raw_observation.get("schema_version")
        if (
            not isinstance(observation_schema_value, str)
            or observation_schema_value not in PPO_OBSERVATION_SCHEMAS
            or raw_observation
            != ppo_observation_contract_payload(observation_schema_value)
        ):
            raise ValueError("PPO inference observation contract is unsupported")
        observation_schema = observation_schema_value
        raw_minimum_hold_bars = manifest["minimum_hold_bars"]
        if (
            isinstance(raw_minimum_hold_bars, bool)
            or not isinstance(raw_minimum_hold_bars, int)
            or raw_minimum_hold_bars < 0
        ):
            raise ValueError("PPO inference minimum_hold_bars is malformed")
        minimum_hold_bars = raw_minimum_hold_bars
        if minimum_hold_bars > 0 and observation_schema != PPO_OBSERVATION_SCHEMA_V3:
            raise ValueError("minimum hold requires PPO Observation v3")
    else:
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
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in indices
        )
        or len(set(indices)) != len(indices)
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
    with verified_private_copy(
        policy_path,
        expected_digest=manifest["policy_sha256"],
        field="PPO inference policy",
        filename="policy.zip",
    ) as verified_policy:
        model = _load_ppo_policy(
            verified_policy,
            feature_count=len(indices),
            observation_schema=observation_schema,
        )
    return PPOIntentStrategy(
        model,
        feature_indices=indices,
        feature_names=selected_names,
        feature_normalizer=normalizer,
        observation_schema=observation_schema,
        minimum_hold_bars=minimum_hold_bars,
    )
