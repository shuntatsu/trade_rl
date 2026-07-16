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
        window_lengths={"15m": 12, "1h": 16, "4h": 10, "1d": 8},
        latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
        asset_state_width=40,
        snapshot_width=64,
        n_symbols=3,
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

    tokens, pooled = encoder(
        sequences=sequences,
        available=available,
        snapshot=snapshot,
        asset_state=asset_state,
        active=active,
    )

    assert tokens.shape == (batch, assets, 320)
    assert pooled.shape == (batch, 320)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(tokens[2]) == 0
    assert torch.count_nonzero(pooled[2]) == 0
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    assert 2_000_000 <= parameter_count <= 10_000_000


def test_timeframe_receptive_fields_cover_declared_windows() -> None:
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
        window_lengths={"15m": 96, "1h": 168, "4h": 120, "1d": 60},
        latent_dims={"15m": 16, "1h": 16, "4h": 16, "1d": 16},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    )
    encoder = MultiTimeframeAssetEncoder(architecture)

    for timeframe, window in architecture.window_lengths.items():
        assert encoder.timeframe_encoders[timeframe].receptive_field >= window


def test_oldest_declared_history_can_influence_final_timeframe_latent() -> None:
    torch.manual_seed(17)
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    encoder = CausalTimeframeEncoder(
        2,
        8,
        window_length=96,
        widths=(8, 8, 8, 8, 8, 8),
        dropout=0.0,
    ).eval()
    baseline = torch.zeros(1, 96, 2)
    mutated = baseline.clone()
    mutated[:, 0, 0] = 10.0

    with torch.no_grad():
        left = encoder(baseline)
        right = encoder(mutated)

    assert not torch.allclose(left, right)


def test_asset_tokens_retain_symbol_specific_context() -> None:
    torch.manual_seed(19)
    architecture = SequencePolicyArchitecture(
        input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
        window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
        latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=24,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    )
    encoder = MultiTimeframeAssetEncoder(architecture).eval()
    sequences = {
        timeframe: torch.zeros(1, 3, 4, 3) for timeframe in architecture.input_channels
    }
    available = {
        timeframe: torch.ones(1, 3, 4, dtype=torch.bool)
        for timeframe in architecture.input_channels
    }
    snapshot = torch.zeros(1, 3, 8)
    snapshot[:, 1, 0] = 2.0
    state = torch.zeros(1, 3, 4)
    active = torch.ones(1, 3, dtype=torch.bool)

    with torch.no_grad():
        tokens, _ = encoder(
            sequences=sequences,
            available=available,
            snapshot=snapshot,
            asset_state=state,
            active=active,
        )

    assert tokens.shape == (1, 3, 24)
    assert not torch.allclose(tokens[:, 0], tokens[:, 1])
    assert not torch.allclose(tokens[:, 1], tokens[:, 2])


def test_partial_feature_availability_keeps_latest_timestep_usable() -> None:
    from gymnasium import spaces

    from trade_rl.rl.policies import SequenceAssetFeatureExtractor

    timeframes = ("15m", "1h", "4h", "1d")
    feature_counts = {timeframe: 2 for timeframe in timeframes}
    window_lengths = {timeframe: 3 for timeframe in timeframes}
    observation_spaces = {
        "current_snapshot": spaces.Box(-1.0, 1.0, shape=(1, 8), dtype=float),
        "asset_state": spaces.Box(-1.0, 1.0, shape=(1, 4), dtype=float),
        "global_state": spaces.Box(-1.0, 1.0, shape=(3,), dtype=float),
        "active": spaces.Box(0.0, 1.0, shape=(1,), dtype=float),
    }
    for timeframe in timeframes:
        shape = (1, 3, 2)
        observation_spaces[f"sequence_{timeframe}_values"] = spaces.Box(
            -1.0, 1.0, shape=shape, dtype=float
        )
        observation_spaces[f"sequence_{timeframe}_available"] = spaces.Box(
            0, 1, shape=shape, dtype="uint8"
        )
        observation_spaces[f"sequence_{timeframe}_staleness"] = spaces.Box(
            0.0, 10.0, shape=shape, dtype="float16"
        )
    extractor = SequenceAssetFeatureExtractor(
        spaces.Dict(observation_spaces),
        feature_counts=feature_counts,
        window_lengths=window_lengths,
        snapshot_width=8,
        asset_state_width=4,
        global_width=3,
        n_symbols=1,
        d_model=16,
        attention_heads=4,
        attention_layers=1,
        dropout=0.0,
    ).eval()
    observation = {
        "current_snapshot": torch.zeros(1, 1, 8),
        "asset_state": torch.zeros(1, 1, 4),
        "global_state": torch.zeros(1, 3),
        "active": torch.ones(1, 1),
    }
    for timeframe in timeframes:
        values = torch.zeros(1, 1, 3, 2)
        values[:, :, -1, 0] = 1.0
        availability = torch.ones(1, 1, 3, 2, dtype=torch.uint8)
        availability[:, :, -1, 1] = 0
        observation[f"sequence_{timeframe}_values"] = values
        observation[f"sequence_{timeframe}_available"] = availability
        observation[f"sequence_{timeframe}_staleness"] = torch.zeros(
            1, 1, 3, 2, dtype=torch.float16
        )

    captured: dict[str, torch.Tensor] = {}
    original = extractor.asset_encoder.timeframe_encoders["15m"].forward

    def capture(value: torch.Tensor, available: torch.Tensor | None = None):
        assert available is not None
        captured["mask"] = available.detach().clone()
        return original(value, available)

    extractor.asset_encoder.timeframe_encoders["15m"].forward = capture  # type: ignore[method-assign]
    with torch.no_grad():
        output = extractor(observation)

    assert output.shape == (1, 160)
    assert bool(captured["mask"][0, -1])
