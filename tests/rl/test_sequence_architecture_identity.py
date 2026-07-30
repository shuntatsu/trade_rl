from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.rl.sequence_architecture import (
    sequence_architecture_identity,
    sequence_asset_binding_identity,
)
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _architecture() -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={timeframe: 6 for timeframe in _TIMEFRAMES},
        window_lengths={"15m": 96, "1h": 168, "4h": 120, "1d": 60},
        latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
        asset_state_width=40,
        snapshot_width=64,
        n_symbols=3,
        d_model=336,
        timeframe_attention_heads=8,
        asset_attention_heads=8,
        timeframe_attention_layers=2,
        asset_attention_layers=2,
        timeframe_ffn_multiplier=3,
        asset_ffn_multiplier=3,
        timeframe_gate_bias=-2.0,
        asset_gate_bias=-2.0,
        dropout=0.05,
    )


def test_sequence_architecture_identity_is_deterministic() -> None:
    architecture = _architecture()
    left = sequence_architecture_identity(architecture)
    right = sequence_architecture_identity(architecture)

    assert left == right
    assert left.digest == right.digest
    assert len(left.digest) == 64
    assert "symbols" not in left.digest_payload()
    assert "action_names" not in left.digest_payload()


def test_sequence_architecture_digest_changes_with_model_semantics() -> None:
    base = _architecture()
    deeper = replace(base, timeframe_attention_layers=3)

    assert (
        sequence_architecture_identity(base).digest
        != sequence_architecture_identity(deeper).digest
    )


def test_sequence_asset_binding_rejects_symbol_action_mismatch() -> None:
    with pytest.raises(ValueError, match="counts must match"):
        sequence_asset_binding_identity(
            n_symbols=3,
            symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            action_names=("target_weight:BTCUSDT", "target_weight:ETHUSDT"),
        )


def test_sequence_asset_binding_rejects_wrong_action_order() -> None:
    with pytest.raises(ValueError, match="target-weight action names"):
        sequence_asset_binding_identity(
            n_symbols=3,
            symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            action_names=(
                "target_weight:ETHUSDT",
                "target_weight:BTCUSDT",
                "target_weight:BNBUSDT",
            ),
        )


def test_sequence_architecture_identity_records_complete_receptive_fields() -> None:
    identity = sequence_architecture_identity(_architecture())

    for window, dilations in zip(
        identity.window_lengths,
        identity.dilations,
        strict=True,
    ):
        assert 1 + 2 * sum(dilations) >= window
