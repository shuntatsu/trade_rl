from __future__ import annotations

import hashlib
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.policy_identity import SB3_POLICY_IDENTITY_ATTRIBUTE
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.structured_export import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    STRUCTURED_EXPORT_MODEL_NAME,
    canonical_structured_observation_keys,
    export_structured_policy_actor,
    load_structured_export_manifest,
)
from trade_rl.serving.structured_policy import (
    CanonicalStructuredPolicyLoader,
    StructuredTorchScriptPolicy,
)


def _digest(path: Any) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _FakeStructuredPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mapping: dict[str, spaces.Space] = {
            "current_snapshot": spaces.Box(-10.0, 10.0, shape=(3, 4)),
            "asset_state": spaces.Box(-10.0, 10.0, shape=(3, 2)),
            "global_state": spaces.Box(-10.0, 10.0, shape=(5,)),
            "active": spaces.Box(0.0, 1.0, shape=(3,)),
            "current_weights": spaces.Box(-1.0, 1.0, shape=(3,), dtype=np.float32),
        }
        for timeframe in ("15m", "1h", "4h", "1d"):
            shape = (3, 4, 2)
            mapping[f"sequence_{timeframe}_values"] = spaces.Box(
                -10.0, 10.0, shape=shape
            )
            mapping[f"sequence_{timeframe}_available"] = spaces.Box(
                0.0, 1.0, shape=shape
            )
            mapping[f"sequence_{timeframe}_staleness"] = spaces.Box(
                0.0, 100.0, shape=shape
            )
        self.observation_space = spaces.Dict(mapping)
        self.marker = nn.Parameter(torch.ones(1))
        self.logical_device = "cuda:7"

    def to(self, device: object, *args: object, **kwargs: object):
        self.logical_device = str(device)
        if str(device) == "cpu":
            return super().to("cpu", *args, **kwargs)
        return self

    def _predict(
        self,
        observation: dict[str, torch.Tensor],
        deterministic: bool = True,
    ) -> torch.Tensor:
        del deterministic
        snapshot = observation["current_snapshot"]
        active = observation["active"]
        short = observation["sequence_15m_values"].mean(dim=(2, 3))
        long = observation["sequence_1d_values"].mean(dim=(2, 3))
        quality = observation["sequence_1d_available"].mean(dim=(2, 3))
        raw = (
            snapshot.mean(dim=2)
            + short
            - long * quality
            + observation["current_weights"]
        )
        return torch.tanh(raw) * active


class _FakeModel:
    def __init__(self) -> None:
        self.policy = _FakeStructuredPolicy()
        self.device = "cuda:7"
        symbols = ("BTC", "ETH", "BNB")
        action_names = tuple(f"target_weight:{symbol}" for symbol in symbols)
        architecture = {
            "asset_identity_mode": "identity_free_v1",
            "d_model": 16,
            "n_symbols": 3,
            "schema_version": "hierarchical_sequence_policy_v4",
            "timeframes": ["15m", "1h", "4h", "1d"],
        }
        sequence_digest = content_digest(architecture)
        asset_binding = {
            "action_names": action_names,
            "n_symbols": 3,
            "schema_version": "sequence_asset_binding_v1",
            "symbols": symbols,
        }
        current_weight = {
            "bounds": (-1.0, 1.0),
            "dtype": "float32",
            "key": "current_weights",
            "observation_schema": SEQUENCE_OBSERVATION_SCHEMA,
            "shape": (3,),
            "source": "effective_book_weights",
        }
        exploration_contract = {
            "action_distribution": "masked_shared_squashed_diag_gaussian_v1",
            "change_intensity_coupling": "post_composition_gate_independent_v1",
            "log_std_parameterization": "shared_scalar_v1",
            "state_dependent_noise": False,
            "schema_version": "hierarchical_exploration_v1",
            "squashing": "tanh",
        }
        policy_architecture = {
            "actor_head": "hierarchical_gate_target_v1",
            "current_weight_observation": current_weight,
            "exploration_contract": exploration_contract,
            "gate_temperature": 1.0,
            "observation_encoder": "hierarchical_sequence_v2",
            "schema_version": "hierarchical_gate_target_policy_v3",
            "sequence_architecture_digest": sequence_digest,
        }
        identity = {
            "actor_head": "hierarchical_gate_target_v1",
            "asset_binding": asset_binding,
            "asset_binding_digest": content_digest(asset_binding),
            "current_weight_observation": current_weight,
            "exploration_contract": exploration_contract,
            "gate_temperature": 1.0,
            "observation_encoder": "hierarchical_sequence_v2",
            "policy_architecture_digest": content_digest(policy_architecture),
            "schema_version": "sb3_policy_identity_v4",
            "sequence_architecture": architecture,
            "sequence_architecture_digest": sequence_digest,
        }
        setattr(self, SB3_POLICY_IDENTITY_ATTRIBUTE, identity)


def _observation() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(77)
    observation: dict[str, np.ndarray] = {
        "current_snapshot": rng.normal(size=(3, 4)).astype(np.float32),
        "asset_state": rng.normal(size=(3, 2)).astype(np.float32),
        "global_state": rng.normal(size=(5,)).astype(np.float32),
        "active": np.asarray([1.0, 1.0, 0.0], dtype=np.float32),
        "current_weights": np.asarray([0.2, -0.4, 0.0], dtype=np.float32),
    }
    for timeframe in ("15m", "1h", "4h", "1d"):
        observation[f"sequence_{timeframe}_values"] = rng.normal(size=(3, 4, 2)).astype(
            np.float32
        )
        observation[f"sequence_{timeframe}_available"] = np.ones(
            (3, 4, 2), dtype=np.float32
        )
        observation[f"sequence_{timeframe}_staleness"] = np.zeros(
            (3, 4, 2), dtype=np.float32
        )
    observation["sequence_1d_available"][2].fill(0.0)
    observation["sequence_1d_staleness"][2].fill(100.0)
    return observation


def _bundle(root: Any, manifest: Any) -> Any:
    model_path = root / manifest.model_path
    manifest_path = root / STRUCTURED_EXPORT_MANIFEST_NAME
    files = (
        SimpleNamespace(
            path=STRUCTURED_EXPORT_MANIFEST_NAME,
            digest=_digest(manifest_path),
            size_bytes=manifest_path.stat().st_size,
        ),
        SimpleNamespace(
            path=manifest.model_path,
            digest=_digest(model_path),
            size_bytes=model_path.stat().st_size,
        ),
    )
    return SimpleNamespace(
        root=root,
        manifest=SimpleNamespace(
            observation_schema=SEQUENCE_OBSERVATION_SCHEMA,
            files=files,
        ),
    )


def test_structured_export_round_trip_and_canonical_loader(tmp_path: Any) -> None:
    model = _FakeModel()
    observation = _observation()
    manifest = export_structured_policy_actor(
        model=model,
        output_dir=tmp_path,
        example_observation=observation,
        action_size=3,
        tolerance=1e-6,
    )

    assert tuple(item.name for item in manifest.inputs) == (
        canonical_structured_observation_keys()
    )
    assert manifest.max_abs_error <= manifest.tolerance
    loaded_manifest = load_structured_export_manifest(
        tmp_path / STRUCTURED_EXPORT_MANIFEST_NAME
    )
    assert loaded_manifest == manifest

    standalone = StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)
    action = standalone.predict(observation)
    with torch.no_grad():
        expected = model.policy._predict(
            {
                key: torch.from_numpy(value).unsqueeze(0)
                for key, value in observation.items()
            }
        )[0].numpy()
    np.testing.assert_allclose(action, expected, rtol=0.0, atol=1e-6)

    loader = CanonicalStructuredPolicyLoader(
        expected_architecture_digest=manifest.architecture_digest
    )
    policy = loader.load(_bundle(tmp_path, manifest))
    np.testing.assert_allclose(
        policy.predict(observation), expected, rtol=0.0, atol=1e-6
    )
    smoke = policy.smoke_observation()
    assert tuple(smoke) == canonical_structured_observation_keys()
    assert policy.predict(smoke).shape == (3,)


def test_structured_export_excludes_training_only_decision_index(tmp_path: Any) -> None:
    model = _FakeModel()
    model.policy.observation_space.spaces["decision_index"] = spaces.Box(
        low=0,
        high=100,
        shape=(1,),
        dtype=np.int64,
    )
    observation = _observation()
    observation["decision_index"] = np.asarray([77], dtype=np.int64)

    manifest = export_structured_policy_actor(
        model=model,
        output_dir=tmp_path,
        example_observation=observation,
        action_size=3,
    )

    assert tuple(item.name for item in manifest.inputs) == (
        canonical_structured_observation_keys()
    )


def test_structured_loader_rejects_architecture_and_input_drift(tmp_path: Any) -> None:
    manifest = export_structured_policy_actor(
        model=_FakeModel(),
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=3,
    )
    wrong = CanonicalStructuredPolicyLoader(expected_architecture_digest="0" * 64)
    with pytest.raises(ValueError, match="architecture"):
        wrong.load(_bundle(tmp_path, manifest))

    policy = StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)
    missing = _observation()
    missing.pop("sequence_1d_staleness")
    with pytest.raises(ValueError, match="keys"):
        policy.predict(missing)
    invalid = _observation()
    invalid["current_snapshot"] = np.zeros((2, 4), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        policy.predict(invalid)


def test_structured_loader_rejects_tampered_model(tmp_path: Any) -> None:
    manifest = export_structured_policy_actor(
        model=_FakeModel(),
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=3,
    )
    model_path = tmp_path / STRUCTURED_EXPORT_MODEL_NAME
    model_path.write_bytes(model_path.read_bytes() + b"tamper")
    with pytest.raises(ValueError, match="size"):
        StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)


def test_structured_export_restores_policy_device_and_training_mode(
    tmp_path: Any,
) -> None:
    model = _FakeModel()
    model.policy.train(True)

    export_structured_policy_actor(
        model=model,
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=3,
    )

    assert model.policy.logical_device == "cuda:7"
    assert model.policy.training is True


def test_structured_export_restores_policy_state_after_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    model = _FakeModel()
    model.policy.train(True)
    monkeypatch.setattr(
        torch.jit,
        "trace",
        lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("trace failed")),
    )

    with pytest.raises(RuntimeError, match="trace failed"):
        export_structured_policy_actor(
            model=model,
            output_dir=tmp_path,
            example_observation=_observation(),
            action_size=3,
        )

    assert model.policy.logical_device == "cuda:7"
    assert model.policy.training is True
    assert not (tmp_path / STRUCTURED_EXPORT_MODEL_NAME).exists()
    assert not (tmp_path / STRUCTURED_EXPORT_MANIFEST_NAME).exists()


def test_structured_export_requires_current_weights(tmp_path: Any) -> None:
    model = _FakeModel()
    del model.policy.observation_space.spaces["current_weights"]
    observation = _observation()
    observation.pop("current_weights")
    with pytest.raises(ValueError, match="canonical contract"):
        export_structured_policy_actor(
            model=model,
            output_dir=tmp_path,
            example_observation=observation,
            action_size=3,
        )


def test_structured_policy_deserializes_only_a_private_verified_copy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Any,
) -> None:
    manifest = export_structured_policy_actor(
        model=_FakeModel(),
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=3,
    )
    source = tmp_path / STRUCTURED_EXPORT_MODEL_NAME
    original_load = torch.jit.load
    loaded_paths: list[Path] = []

    def recording_load(path: str, *args: Any, **kwargs: Any) -> Any:
        loaded_paths.append(Path(path))
        return original_load(path, *args, **kwargs)

    monkeypatch.setattr(torch.jit, "load", recording_load)
    StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)

    assert len(loaded_paths) == 1
    assert loaded_paths[0] != source
    assert not loaded_paths[0].exists()


@pytest.mark.skipif(os.name == "nt", reason="symlink semantics require POSIX")
def test_structured_policy_rejects_symlink_model(tmp_path: Any) -> None:
    manifest = export_structured_policy_actor(
        model=_FakeModel(),
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=3,
    )
    source = tmp_path / STRUCTURED_EXPORT_MODEL_NAME
    external = tmp_path / "external.pt"
    source.replace(external)
    source.symlink_to(external)

    with pytest.raises(ValueError, match="symlink"):
        StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)
