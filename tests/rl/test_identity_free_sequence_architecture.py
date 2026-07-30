from __future__ import annotations

from dataclasses import replace

import pytest

from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _architecture() -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={timeframe: 3 for timeframe in _TIMEFRAMES},
        window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
        latent_dims={timeframe: 8 for timeframe in _TIMEFRAMES},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        timeframe_attention_heads=4,
        asset_attention_heads=4,
        timeframe_attention_layers=1,
        asset_attention_layers=1,
        dropout=0.0,
    )


def _identity():
    return sequence_architecture_identity(
        _architecture(),
        symbols=("BTCUSDT", "ETHUSDT", "BNBUSDT"),
        action_names=("target_weight:BTCUSDT", "target_weight:ETHUSDT", "target_weight:BNBUSDT"),
    )


def test_sequence_identity_records_identity_free_asset_semantics() -> None:
    identity = _identity()

    assert identity.schema_version == "hierarchical_sequence_policy_v3"
    assert identity.asset_identity_mode == "identity_free_v1"
    assert identity.digest_payload()["asset_identity_mode"] == "identity_free_v1"


def test_sequence_identity_rejects_other_asset_identity_modes() -> None:
    with pytest.raises(ValueError, match="asset identity mode"):
        replace(_identity(), asset_identity_mode="fixed_slot_embedding_v1")
