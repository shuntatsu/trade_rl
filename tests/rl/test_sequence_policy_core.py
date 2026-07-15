from __future__ import annotations

import torch

from trade_rl.rl.sequence_policy import (
    CausalTemporalBlock,
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)

torch.set_num_threads(1)


def test_causal_temporal_block_does_not_use_future_values() -> None:
    torch.manual_seed(7)
    block = CausalTemporalBlock(
        in_channels=4,
        out_channels=8,
        kernel_size=3,
        dilation=2,
        dropout=0.0,
    ).eval()
    baseline = torch.randn(2, 4, 24)
    mutated = baseline.clone()
    mutated[:, :, 13:] += 10_000.0

    with torch.no_grad():
        left = block(baseline)
        right = block(mutated)

    torch.testing.assert_close(left[:, :, :13], right[:, :, :13])


def test_multitimeframe_encoder_shapes_and_parameter_budget() -> None:
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 18, "1h": 20, "4h": 16, "1d": 12},
        latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
        asset_state_width=40,
        snapshot_width=64,
        d_model=320,
        attention_heads=8,
        attention_layers=2,
    )
    encoder = MultiTimeframeAssetEncoder(architecture)
    batch, assets = 3, 3
    sequences = {
        "15m": torch.randn(batch, assets, 12, 18),
        "1h": torch.randn(batch, assets, 16, 20),
        "4h": torch.randn(batch, assets, 10, 16),
        "1d": torch.randn(batch, assets, 8, 12),
    }
    available = {
        key: torch.ones(value.shape[:-1], dtype=torch.bool)
        for key, value in sequences.items()
    }
    snapshot = torch.randn(batch, assets, 64)
    asset_state = torch.randn(batch, assets, 40)
    active = torch.tensor([[1, 1, 1], [1, 0, 1], [0, 0, 0]], dtype=torch.bool)

    output = encoder(
        sequences=sequences,
        available=available,
        snapshot=snapshot,
        asset_state=asset_state,
        active=active,
    )

    assert output.shape == (batch, 320)
    assert torch.isfinite(output).all()
    assert torch.count_nonzero(output[2]) == 0
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    assert 2_000_000 <= parameter_count <= 10_000_000
