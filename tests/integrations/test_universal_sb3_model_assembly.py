from __future__ import annotations

from types import SimpleNamespace

import numpy as np
from gymnasium import spaces

from trade_rl.integrations.sb3_model_assembly import resolve_sb3_policy_assembly
from trade_rl.rl.algorithm_configs import build_algorithm_config
from trade_rl.rl.training import ResidualTrainingConfig


class _UniversalSequenceProbe:
    is_universal_single_instrument = True
    policy_symbols = ("INSTRUMENT",)
    sequence_layout_metadata = {
        "feature_counts": {"15m": 2, "1h": 2, "4h": 2, "1d": 2},
        "window_lengths": {"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        "snapshot_width": 2,
        "asset_state_width": 2,
        "global_width": 2,
        "n_symbols": 1,
        "current_weight_shape": (1,),
        "instrument_context_width": 9,
    }
    pre_trade_risk = SimpleNamespace(
        config=SimpleNamespace(entry_threshold=0.01, no_trade_band=0.005)
    )
    observation_space = spaces.Dict(
        {
            "current_snapshot": spaces.Box(-10.0, 10.0, shape=(1, 2), dtype=np.float32),
            "asset_state": spaces.Box(-10.0, 10.0, shape=(1, 2), dtype=np.float32),
            "global_state": spaces.Box(-10.0, 10.0, shape=(2,), dtype=np.float32),
            "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32),
            "current_weights": spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32),
            "instrument_context": spaces.Box(-np.inf, np.inf, shape=(1, 9), dtype=np.float32),
            **{
                f"sequence_{timeframe}_{suffix}": spaces.Box(
                    0.0 if suffix != "values" else -10.0,
                    10.0,
                    shape=(1, 4, 2),
                    dtype=np.float32,
                )
                for timeframe in ("15m", "1h", "4h", "1d")
                for suffix in ("values", "available", "staleness")
            },
        }
    )

    @property
    def unwrapped(self) -> _UniversalSequenceProbe:
        return self


def _config() -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=8,
        gamma=1.0,
        seeds=(0,),
        n_steps=8,
        n_envs=1,
        batch_size=8,
        n_epochs=1,
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        policy_actor_head="shared_target_v1",
        sequence_tcn_capacity="compact",
        sequence_d_model=32,
        sequence_timeframe_attention_heads=4,
        sequence_timeframe_attention_layers=1,
        sequence_asset_attention_heads=4,
        sequence_asset_attention_layers=1,
        sequence_dropout=0.0,
        max_rollout_buffer_bytes=100_000_000,
        device="cpu",
    )


def test_universal_sequence_assembly_uses_generic_symbol_and_full_dict_buffer() -> None:
    config = _config()
    probe = _UniversalSequenceProbe()
    assembly = resolve_sb3_policy_assembly(
        probe=probe,
        identity={
            "action_names": ("target_weight:INSTRUMENT",),
            "action_size": 1,
        },
        config=config,
        algorithm_config=build_algorithm_config(config),
    )

    assert assembly.sequence_symbols == ("INSTRUMENT",)
    assert assembly.sequence_action_names == ("target_weight:INSTRUMENT",)
    assert assembly.sequence_reconstructor is None
    assert assembly.rollout_buffer_class is None
    assert assembly.rollout_buffer_kwargs is None
    assert assembly.sequence_metadata is not None
    assert assembly.sequence_metadata["instrument_context_width"] == 9
    assert assembly.rollout_buffer_bytes is not None


def test_universal_sequence_assembly_keeps_shared_actor() -> None:
    config = _config()
    assembly = resolve_sb3_policy_assembly(
        probe=_UniversalSequenceProbe(),
        identity={
            "action_names": ("target_weight:INSTRUMENT",),
            "action_size": 1,
        },
        config=config,
        algorithm_config=build_algorithm_config(config),
    )

    assert assembly.uses_shared_asset_actor is True
    kwargs = assembly.policy_kwargs["features_extractor_kwargs"]
    assert isinstance(kwargs, dict)
    assert kwargs["instrument_context_width"] == 9
