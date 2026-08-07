from __future__ import annotations

import torch

from trade_rl.rl.gated_transformer import GatedTransformerStack
from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _architecture(n_symbols: int) -> SequencePolicyArchitecture:
    return SequencePolicyArchitecture(
        input_channels={timeframe: 2 for timeframe in _TIMEFRAMES},
        window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
        latent_dims={timeframe: 8 for timeframe in _TIMEFRAMES},
        asset_state_width=2,
        snapshot_width=3,
        n_symbols=n_symbols,
        d_model=16,
        timeframe_attention_heads=4,
        timeframe_attention_layers=1,
        timeframe_ffn_multiplier=2,
        timeframe_gate_bias=-2.0,
        asset_attention_heads=4,
        asset_attention_layers=1,
        asset_ffn_multiplier=2,
        asset_gate_bias=-2.0,
        dropout=0.0,
        encoder_widths={timeframe: (4, 4) for timeframe in _TIMEFRAMES},
    )


def test_one_symbol_encoder_does_not_build_cross_asset_transformer() -> None:
    encoder = MultiTimeframeAssetEncoder(_architecture(1)).eval()
    assert encoder.cross_asset is None
    assert not any(name.startswith("cross_asset.") for name, _ in encoder.named_parameters())
    sequences = {
        timeframe: torch.randn(2, 1, 4, 2) for timeframe in _TIMEFRAMES
    }
    available = {
        timeframe: torch.ones(2, 1, 4, 2, dtype=torch.bool)
        for timeframe in _TIMEFRAMES
    }
    staleness = {
        timeframe: torch.zeros(2, 1, 4, 2) for timeframe in _TIMEFRAMES
    }
    active = torch.tensor([[1.0], [0.0]])

    with torch.no_grad():
        contextual, pooled = encoder(
            sequences=sequences,
            available=available,
            staleness=staleness,
            snapshot=torch.randn(2, 1, 3),
            asset_state=torch.randn(2, 1, 2),
            active=active,
        )

    assert contextual.shape == (2, 1, 16)
    assert pooled.shape == (2, 16)
    assert torch.count_nonzero(contextual[0]) > 0
    assert torch.count_nonzero(contextual[1]) == 0
    assert torch.count_nonzero(pooled[1]) == 0


def test_multi_symbol_encoder_retains_cross_asset_transformer() -> None:
    encoder = MultiTimeframeAssetEncoder(_architecture(3))

    assert isinstance(encoder.cross_asset, GatedTransformerStack)
    assert any(name.startswith("cross_asset.") for name, _ in encoder.named_parameters())


def test_one_symbol_architecture_identity_differs_from_three_symbol_identity() -> None:
    single = sequence_architecture_identity(_architecture(1))
    legacy = sequence_architecture_identity(_architecture(3))
    single_payload = single.digest_payload()
    legacy_payload = legacy.digest_payload()

    assert single_payload["n_symbols"] == 1
    assert single_payload["asset_fusion_mode"] == "single_symbol_bypass_v1"
    assert legacy_payload["n_symbols"] == 3
    assert "asset_fusion_mode" not in legacy_payload
    assert single.digest != legacy.digest
