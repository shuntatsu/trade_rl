from __future__ import annotations

from dataclasses import replace

from trade_rl.rl.sequence_architecture import sequence_architecture_identity
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
        attention_heads=8,
        attention_layers=2,
        attention_ffn_multiplier=3,
        attention_gate_bias=-2.0,
        dropout=0.05,
    )


def test_sequence_architecture_identity_is_deterministic() -> None:
    architecture = _architecture()
    left = sequence_architecture_identity(
        architecture,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=(
            "target_weight:BTCUSDT",
            "target_weight:ETHUSDT",
            "target_weight:BNBUSDT",
        ),
    )
    right = sequence_architecture_identity(
        architecture,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=(
            "target_weight:BTCUSDT",
            "target_weight:ETHUSDT",
            "target_weight:BNBUSDT",
        ),
    )

    assert left == right
    assert left.digest == right.digest
    assert len(left.digest) == 64


def test_sequence_architecture_digest_changes_with_model_semantics() -> None:
    base = _architecture()
    deeper = replace(base, attention_layers=3)
    base_identity = sequence_architecture_identity(
        base,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=("a", "b", "c"),
    )
    deeper_identity = sequence_architecture_identity(
        deeper,
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=("a", "b", "c"),
    )

    assert base_identity.digest != deeper_identity.digest


def test_sequence_architecture_identity_rejects_symbol_action_mismatch() -> None:
    try:
        sequence_architecture_identity(
            _architecture(),
            symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
            action_names=("a", "b"),
        )
    except ValueError as error:
        assert "symbol and action counts" in str(error)
    else:
        raise AssertionError("symbol/action identity mismatch must fail closed")


def test_sequence_architecture_identity_records_complete_receptive_fields() -> None:
    identity = sequence_architecture_identity(
        _architecture(),
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=("a", "b", "c"),
    )

    for window, dilations in zip(
        identity.window_lengths,
        identity.dilations,
        strict=True,
    ):
        assert 1 + 2 * sum(dilations) >= window
