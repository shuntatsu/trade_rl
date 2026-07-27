from __future__ import annotations

import pytest

from trade_rl.rl.training import ResidualTrainingConfig


def _config(**overrides: object) -> ResidualTrainingConfig:
    values: dict[str, object] = {
        "timesteps": 128,
        "gamma": 0.99,
        "seeds": (0,),
        "n_steps": 128,
        "batch_size": 128,
    }
    values.update(overrides)
    return ResidualTrainingConfig(**values)  # type: ignore[arg-type]


def test_observation_encoder_is_one_closed_choice() -> None:
    with pytest.raises(ValueError, match="observation_encoder"):
        _config(observation_encoder="unknown")


def test_hierarchical_encoder_requires_multi_input_policy() -> None:
    with pytest.raises(ValueError, match="MultiInputPolicy"):
        _config(observation_encoder="hierarchical_sequence_v2")


def test_timeframe_and_asset_attention_are_independent_identity_fields() -> None:
    left = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        sequence_timeframe_attention_layers=1,
        sequence_asset_attention_layers=2,
    )
    right = _config(
        observation_encoder="hierarchical_sequence_v2",
        policy="MultiInputPolicy",
        sequence_timeframe_attention_layers=2,
        sequence_asset_attention_layers=1,
    )
    assert left.digest_payload() != right.digest_payload()


def test_sequence_fields_fail_closed_for_non_sequence_encoder() -> None:
    with pytest.raises(ValueError, match="inactive"):
        _config(
            observation_encoder="asset_set",
            sequence_timeframe_attention_layers=3,
        )
