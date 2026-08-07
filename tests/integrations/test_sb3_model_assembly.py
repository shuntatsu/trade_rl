from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.observations import ObservationLayout
from trade_rl.rl.training import ResidualTrainingConfig
from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT


class _Probe(gym.Env[np.ndarray, np.ndarray]):
    metadata = {"render_modes": []}

    def __init__(self) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-1.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
        self.layout = ObservationLayout(
            n_symbols=1,
            n_features=2,
            action_size=1,
            n_factors=0,
            per_symbol_width=3,
            global_width=2,
        )
        self.asset_active_column = 2

    @property
    def unwrapped(self) -> _Probe:
        return self

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, object] | None = None,
    ) -> tuple[np.ndarray, dict[str, object]]:
        super().reset(seed=seed)
        return np.zeros(5, dtype=np.float32), {}

    def step(
        self, action: np.ndarray
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, object]]:
        return np.zeros(5, dtype=np.float32), 0.0, False, False, {}


class _SequenceDataset:
    dataset_id = "d" * 64
    symbols = ("BTCUSDT",)


class _SequenceBuilder:
    def layout_digest(self, dataset: object) -> str:
        assert dataset is _SequenceProbe.dataset
        return "e" * 64


class _SequenceProbe:
    dataset = _SequenceDataset()
    sequence_observation_builder = _SequenceBuilder()
    sequence_normalizer = None
    sequence_policy_plane = None
    sequence_layout_metadata = {"n_symbols": 1}
    pre_trade_risk = SimpleNamespace(
        config=SimpleNamespace(entry_threshold=0.1, no_trade_band=0.05)
    )

    @property
    def unwrapped(self) -> "_SequenceProbe":
        return self


def _identity() -> dict[str, object]:
    return {
        "action_names": ("target_weight:BTCUSDT",),
        "action_size": 1,
    }


def _config(**changes: object) -> ResidualTrainingConfig:
    payload: dict[str, object] = {
        "timesteps": 8,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 8,
        "n_envs": 1,
        "batch_size": 8,
        "n_epochs": 1,
        "observation_encoder": "flat_mlp",
        "device": "cpu",
    }
    payload.update(changes)
    return ResidualTrainingConfig(**payload)  # type: ignore[arg-type]


def test_ppo_policy_assembly_preserves_actor_critic_configuration() -> None:
    from trade_rl.integrations.sb3_model_assembly import (
        resolve_sb3_policy_assembly,
    )

    config = _config(
        policy_net_arch=(96, 48),
        value_net_arch=(128, 64),
        log_std_init=-1.25,
    )

    assembly = resolve_sb3_policy_assembly(
        probe=_Probe(),
        identity=_identity(),
        config=config,
        algorithm_config=build_algorithm_config(config),
    )

    assert assembly.policy_identifier == config.policy
    assert assembly.policy_kwargs == {
        "log_std_init": -1.25,
        "net_arch": {"pi": [96, 48], "vf": [128, 64]},
    }
    assert assembly.rollout_buffer_bytes is not None
    assert assembly.sequence_metadata is None
    assert assembly.sequence_reconstructor is None
    assert assembly.uses_shared_asset_actor is False


def test_off_policy_assembly_uses_q_function_architecture() -> None:
    from trade_rl.integrations.sb3_model_assembly import (
        resolve_sb3_policy_assembly,
    )

    config = _config(
        algorithm="sac",
        policy_net_arch=(64, 32),
        value_net_arch=(80, 40),
        n_steps=2_048,
        batch_size=64,
        n_epochs=10,
    )

    assembly = resolve_sb3_policy_assembly(
        probe=_Probe(),
        identity=_identity(),
        config=config,
        algorithm_config=build_algorithm_config(config),
    )

    assert assembly.policy_identifier == config.policy
    assert assembly.policy_kwargs == {"net_arch": {"pi": [64, 32], "qf": [80, 40]}}
    assert assembly.rollout_buffer_bytes is None


def test_asset_set_assembly_preserves_layout_metadata() -> None:
    from trade_rl.integrations.sb3_model_assembly import (
        resolve_sb3_policy_assembly,
    )
    from trade_rl.rl.policies import AssetSetFeatureExtractor

    config = _config(
        observation_encoder=("asset_set"),
        asset_embedding_dim=24,
        global_embedding_dim=16,
    )

    assembly = resolve_sb3_policy_assembly(
        probe=_Probe(),
        identity=_identity(),
        config=config,
        algorithm_config=build_algorithm_config(config),
    )

    assert (
        assembly.policy_kwargs["features_extractor_class"] is AssetSetFeatureExtractor
    )
    assert assembly.policy_kwargs["features_extractor_kwargs"] == {
        "n_symbols": 1,
        "per_symbol_width": 3,
        "global_width": 2,
        "active_column": 2,
        "asset_embedding_dim": 24,
        "global_embedding_dim": 16,
    }


def test_rollout_budget_remains_fail_closed() -> None:
    from trade_rl.integrations.sb3_model_assembly import (
        resolve_sb3_policy_assembly,
    )

    config = _config(max_rollout_buffer_bytes=1)

    with pytest.raises(ValueError, match="estimated PPO rollout buffer exceeds"):
        resolve_sb3_policy_assembly(
            probe=_Probe(),
            identity=_identity(),
            config=config,
            algorithm_config=build_algorithm_config(config),
        )


def test_model_assembly_dependency_boundary() -> None:
    source = (PYTHON_SOURCE_ROOT / "integrations/sb3_model_assembly.py").read_text(
        encoding="utf-8"
    )
    forbidden = (
        "trade_rl.learning",
        "trade_rl.integrations.behavior_cloning",
        "trade_rl.rl.tensorboard_logging",
        "trade_rl.rl.training_performance",
        "trade_rl.artifacts.store",
        "trade_rl.integrations.sb3_training",
    )
    assert [name for name in forbidden if name in source] == []


def test_sequence_assembly_binds_hierarchical_actor_identity() -> None:
    from trade_rl.integrations.sb3_model_assembly import _sequence_policy_assembly

    config = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        policy_actor_head="hierarchical_gate_target_v1",
        hierarchical_gate_temperature=0.75,
        behavior_cloning_epochs=1,
        behavior_cloning_gate_prediction_threshold=0.49,
    )

    _, policy_kwargs, _, _, uses_shared_actor = _sequence_policy_assembly(
        probe=_SequenceProbe(),
        identity=_identity(),
        config=config,
    )

    assert uses_shared_actor is True
    assert policy_kwargs["shared_actor_head"] == "hierarchical_gate_target_v1"
    assert policy_kwargs["shared_actor_gate_temperature"] == pytest.approx(0.75)
    assert policy_kwargs["shared_actor_gate_prediction_threshold"] == pytest.approx(
        0.49
    )
    assert policy_kwargs["shared_actor_entry_threshold"] == pytest.approx(0.1)
    assert policy_kwargs["shared_actor_minimum_deterministic_change"] == pytest.approx(
        0.05
    )


def test_hierarchical_actor_fields_are_digest_bound() -> None:
    from trade_rl.artifacts.hashing import content_digest

    first = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        hierarchical_gate_temperature=0.75,
    )
    second = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        hierarchical_gate_temperature=1.25,
    )

    assert first.digest_payload()["policy_actor_head"] == (
        "hierarchical_gate_target_v1"
    )
    assert content_digest(first.digest_payload()) != content_digest(
        second.digest_payload()
    )
