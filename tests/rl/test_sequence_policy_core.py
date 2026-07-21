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


def test_compact_sequence_capacity_preserves_all_receptive_field_depths() -> None:
    import trade_rl.rl.sequence_policy as sequence_policy

    resolver = getattr(sequence_policy, "sequence_encoder_widths", None)
    assert callable(resolver)
    widths = resolver("compact")

    assert tuple(widths) == ("15m", "1h", "4h", "1d")
    assert tuple(map(len, widths.values())) == (6, 7, 6, 5)
    assert max(max(values) for values in widths.values()) == 112


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


def test_timeframe_projection_runs_only_after_causal_timestep_selection() -> None:
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    encoder = CausalTimeframeEncoder(
        4,
        8,
        window_length=12,
        widths=(8, 8, 8, 8),
        dropout=0.0,
    )
    projection_input_shapes: list[torch.Size] = []

    def capture_projection_input(
        _module: torch.nn.Module, args: tuple[torch.Tensor, ...]
    ) -> None:
        projection_input_shapes.append(args[0].shape)

    handle = encoder.projection.register_forward_pre_hook(capture_projection_input)
    try:
        encoder(
            torch.randn(3, 12, 4),
            torch.tensor(
                [
                    [True] * 12,
                    [True] * 7 + [False] * 5,
                    [False] * 12,
                ]
            ),
        )
    finally:
        handle.remove()

    assert projection_input_shapes == [torch.Size((3, 8))]


def test_projection_after_selection_matches_legacy_outputs_and_gradients() -> None:
    from trade_rl.rl.sequence_policy import CausalTimeframeEncoder

    torch.manual_seed(23)
    encoder = CausalTimeframeEncoder(
        4,
        8,
        window_length=12,
        widths=(8, 8, 8, 8),
        dropout=0.0,
    )
    available = torch.tensor(
        [
            [True] * 12,
            [True] * 7 + [False] * 5,
            [False] * 12,
        ]
    )
    legacy_input = torch.randn(3, 12, 4, requires_grad=True)
    optimized_input = legacy_input.detach().clone().requires_grad_(True)

    legacy_sequence = encoder.projection(encoder.forward_sequence(legacy_input))
    positions = torch.arange(12).expand_as(available)
    indices = positions.masked_fill(~available, -1).max(dim=1).values
    safe = indices.clamp_min(0)
    legacy_selected = legacy_sequence[torch.arange(3), safe]
    legacy = torch.where(
        (indices >= 0).unsqueeze(1),
        legacy_selected,
        torch.zeros_like(legacy_selected),
    )
    legacy.square().sum().backward()
    legacy_parameter_gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in encoder.named_parameters()
        if parameter.grad is not None
    }

    encoder.zero_grad(set_to_none=True)
    optimized = encoder(optimized_input, available)
    optimized.square().sum().backward()

    torch.testing.assert_close(optimized, legacy, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(
        optimized_input.grad, legacy_input.grad, rtol=1e-5, atol=1e-6
    )
    for name, parameter in encoder.named_parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(
            parameter.grad,
            legacy_parameter_gradients[name],
            rtol=1e-5,
            atol=1e-6,
        )


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
