from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from gymnasium import spaces
from torch import nn

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.structured_policy_contract import StructuredExportManifest
from trade_rl.rl.policy_identity import SB3_POLICY_IDENTITY_ATTRIBUTE
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.rl.structured_export import export_structured_policy_actor
from trade_rl.serving.structured_policy import StructuredTorchScriptPolicy

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


class _SingleSymbolStructuredPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        mapping: dict[str, spaces.Space] = {
            "current_snapshot": spaces.Box(-10.0, 10.0, shape=(1, 3)),
            "asset_state": spaces.Box(-10.0, 10.0, shape=(1, 2)),
            "global_state": spaces.Box(-10.0, 10.0, shape=(2,)),
            "active": spaces.Box(0.0, 1.0, shape=(1,)),
            "current_weights": spaces.Box(
                -1.0, 1.0, shape=(1,), dtype=np.float32
            ),
        }
        for timeframe in _TIMEFRAMES:
            shape = (1, 4, 2)
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

    def _predict(
        self,
        observation: dict[str, torch.Tensor],
        deterministic: bool = True,
    ) -> torch.Tensor:
        del deterministic
        snapshot = observation["current_snapshot"].mean(dim=2)
        short = observation["sequence_15m_values"].mean(dim=(2, 3))
        current = observation["current_weights"]
        active = observation["active"]
        return torch.tanh(snapshot + short + current) * active


class _SingleSymbolModel:
    def __init__(self) -> None:
        self.policy = _SingleSymbolStructuredPolicy()
        self.device = "cpu"
        action_names = ("target_weight:BTCUSDT",)
        architecture = {
            "asset_fusion_mode": "single_symbol_bypass_v1",
            "asset_identity_mode": "identity_free_v1",
            "d_model": 16,
            "n_symbols": 1,
            "schema_version": "hierarchical_sequence_policy_v4",
            "timeframes": list(_TIMEFRAMES),
        }
        architecture_digest = content_digest(architecture)
        asset_binding = {
            "action_names": action_names,
            "n_symbols": 1,
            "schema_version": "sequence_asset_binding_v1",
            "symbols": ("BTCUSDT",),
        }
        current_weight = {
            "bounds": (-1.0, 1.0),
            "dtype": "float32",
            "key": "current_weights",
            "observation_schema": SEQUENCE_OBSERVATION_SCHEMA,
            "shape": (1,),
            "source": "effective_book_weights",
        }
        exploration = {
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
            "exploration_contract": exploration,
            "gate_temperature": 1.0,
            "observation_encoder": "hierarchical_sequence_v2",
            "schema_version": "hierarchical_gate_target_policy_v3",
            "sequence_architecture_digest": architecture_digest,
        }
        identity = {
            "actor_head": "hierarchical_gate_target_v1",
            "asset_binding": asset_binding,
            "asset_binding_digest": content_digest(asset_binding),
            "current_weight_observation": current_weight,
            "exploration_contract": exploration,
            "gate_temperature": 1.0,
            "observation_encoder": "hierarchical_sequence_v2",
            "policy_architecture_digest": content_digest(policy_architecture),
            "schema_version": "sb3_policy_identity_v4",
            "sequence_architecture": architecture,
            "sequence_architecture_digest": architecture_digest,
        }
        setattr(self, SB3_POLICY_IDENTITY_ATTRIBUTE, identity)


def _observation() -> dict[str, np.ndarray]:
    observation: dict[str, np.ndarray] = {
        "current_snapshot": np.asarray([[0.2, 0.1, -0.1]], dtype=np.float32),
        "asset_state": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "global_state": np.asarray([0.0, 1.0], dtype=np.float32),
        "active": np.asarray([1.0], dtype=np.float32),
        "current_weights": np.asarray([0.25], dtype=np.float32),
    }
    for timeframe in _TIMEFRAMES:
        observation[f"sequence_{timeframe}_values"] = np.full(
            (1, 4, 2), 0.05, dtype=np.float32
        )
        observation[f"sequence_{timeframe}_available"] = np.ones(
            (1, 4, 2), dtype=np.float32
        )
        observation[f"sequence_{timeframe}_staleness"] = np.zeros(
            (1, 4, 2), dtype=np.float32
        )
    return observation


def _export(tmp_path: Path) -> tuple[_SingleSymbolModel, StructuredExportManifest]:
    model = _SingleSymbolModel()
    manifest = export_structured_policy_actor(
        model=model,
        output_dir=tmp_path,
        example_observation=_observation(),
        action_size=1,
        tolerance=1e-6,
    )
    return model, manifest


def _rebuilt_manifest(
    tmp_path: Path,
    manifest: StructuredExportManifest,
    *,
    policy_identity: dict[str, object] | None = None,
    action_size: int | None = None,
) -> StructuredExportManifest:
    return StructuredExportManifest.build(
        model_path=tmp_path / manifest.model_path,
        policy_identity=(
            dict(manifest.policy_identity)
            if policy_identity is None
            else policy_identity
        ),
        inputs=manifest.inputs,
        action_size=manifest.action_size if action_size is None else action_size,
        tolerance=manifest.tolerance,
        max_abs_error=manifest.max_abs_error,
    )


def test_single_symbol_structured_export_round_trip(tmp_path: Path) -> None:
    model, manifest = _export(tmp_path)
    observation = _observation()

    policy = StructuredTorchScriptPolicy(root=tmp_path, manifest=manifest)
    action = policy.predict(observation)

    assert action.shape == (1,)
    assert manifest.action_size == 1
    raw_architecture = manifest.policy_identity["sequence_architecture"]
    assert isinstance(raw_architecture, dict)
    assert raw_architecture["asset_fusion_mode"] == "single_symbol_bypass_v1"
    np.testing.assert_allclose(
        action,
        model.policy._predict(
            {
                key: torch.from_numpy(value).unsqueeze(0)
                for key, value in observation.items()
            }
        )[0].detach().numpy(),
        rtol=0.0,
        atol=1e-6,
    )


def test_structured_policy_rejects_missing_single_symbol_fusion_identity(
    tmp_path: Path,
) -> None:
    _model, manifest = _export(tmp_path)
    identity = dict(manifest.policy_identity)
    raw_architecture = identity["sequence_architecture"]
    assert isinstance(raw_architecture, dict)
    architecture = dict(raw_architecture)
    architecture.pop("asset_fusion_mode")
    identity["sequence_architecture"] = architecture
    identity["sequence_architecture_digest"] = content_digest(architecture)
    tampered = _rebuilt_manifest(
        tmp_path,
        manifest,
        policy_identity=identity,
    )

    with pytest.raises(ValueError, match="fusion identity mismatch"):
        StructuredTorchScriptPolicy(root=tmp_path, manifest=tampered)


def test_structured_policy_rejects_manifest_action_size_drift(tmp_path: Path) -> None:
    _model, manifest = _export(tmp_path)
    tampered = _rebuilt_manifest(tmp_path, manifest, action_size=3)

    with pytest.raises(ValueError, match="action size identity mismatch"):
        StructuredTorchScriptPolicy(root=tmp_path, manifest=tampered)
