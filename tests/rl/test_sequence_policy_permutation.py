from __future__ import annotations

import torch

from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)

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


def _inputs() -> dict[str, object]:
    sequences = {
        timeframe: torch.randn(2, 3, 4, 3) for timeframe in _TIMEFRAMES
    }
    available = {
        timeframe: torch.ones(2, 3, 4, dtype=torch.bool)
        for timeframe in _TIMEFRAMES
    }
    return {
        "sequences": sequences,
        "available": available,
        "staleness": {
            timeframe: torch.zeros_like(value, dtype=torch.float32)
            for timeframe, value in available.items()
        },
        "snapshot": torch.randn(2, 3, 8),
        "asset_state": torch.randn(2, 3, 4),
        "active": torch.tensor([[True, True, True], [True, False, True]]),
    }


def test_asset_encoder_has_no_slot_identity_parameters() -> None:
    encoder = MultiTimeframeAssetEncoder(_architecture())

    assert not hasattr(encoder, "symbol_embedding")
    assert all(
        "symbol_embedding" not in name for name, _ in encoder.named_parameters()
    )


def test_asset_encoder_is_permutation_equivariant() -> None:
    torch.manual_seed(20260730)
    encoder = MultiTimeframeAssetEncoder(_architecture()).eval()
    inputs = _inputs()
    sequences = inputs["sequences"]
    available = inputs["available"]
    staleness = inputs["staleness"]
    snapshot = inputs["snapshot"]
    asset_state = inputs["asset_state"]
    active = inputs["active"]
    assert isinstance(sequences, dict)
    assert isinstance(available, dict)
    assert isinstance(staleness, dict)
    assert isinstance(snapshot, torch.Tensor)
    assert isinstance(asset_state, torch.Tensor)
    assert isinstance(active, torch.Tensor)
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        tokens, pooled = encoder(
            sequences=sequences,
            available=available,
            staleness=staleness,
            snapshot=snapshot,
            asset_state=asset_state,
            active=active,
        )
        permuted_tokens, permuted_pooled = encoder(
            sequences={
                key: value[:, permutation] for key, value in sequences.items()
            },
            available={
                key: value[:, permutation] for key, value in available.items()
            },
            staleness={
                key: value[:, permutation] for key, value in staleness.items()
            },
            snapshot=snapshot[:, permutation],
            asset_state=asset_state[:, permutation],
            active=active[:, permutation],
        )

    torch.testing.assert_close(permuted_tokens, tokens[:, permutation])
    torch.testing.assert_close(permuted_pooled, pooled)
