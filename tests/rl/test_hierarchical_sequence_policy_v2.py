from __future__ import annotations

import torch

from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)

_TIMEFRAMES = ("15m", "1h", "4h", "1d")


def _case(*, requires_grad: bool = False):
    architecture = SequencePolicyArchitecture(
        input_channels={timeframe: 6 for timeframe in _TIMEFRAMES},
        window_lengths={timeframe: 4 for timeframe in _TIMEFRAMES},
        latent_dims={timeframe: 8 for timeframe in _TIMEFRAMES},
        asset_state_width=4,
        snapshot_width=8,
        n_symbols=3,
        d_model=16,
        timeframe_attention_heads=4,
        asset_attention_heads=4,
        timeframe_attention_layers=1,
        asset_attention_layers=1,
        dropout=0.0,
        encoder_widths={timeframe: (8, 8) for timeframe in _TIMEFRAMES},
    )
    encoder = MultiTimeframeAssetEncoder(architecture)
    sequences: dict[str, torch.Tensor] = {}
    available: dict[str, torch.Tensor] = {}
    for timeframe in _TIMEFRAMES:
        values = torch.randn(2, 3, 4, 2)
        availability = torch.ones(2, 3, 4, 2, dtype=torch.bool)
        staleness = torch.zeros(2, 3, 4, 2)
        sequence = torch.cat(
            (values, availability.float(), torch.log1p(staleness)),
            dim=-1,
        )
        sequence.requires_grad_(requires_grad)
        sequences[timeframe] = sequence
        available[timeframe] = availability
    snapshot = torch.randn(2, 3, 8)
    asset_state = torch.randn(2, 3, 4)
    active = torch.tensor([[True, True, False], [False, False, False]])
    return encoder, sequences, available, snapshot, asset_state, active


def test_hierarchical_encoder_shapes_masks_and_parameter_budget() -> None:
    encoder, sequences, available, snapshot, asset_state, active = _case()

    tokens, pooled = encoder(
        sequences=sequences,
        available=available,
        staleness={
            key: __import__("torch").zeros_like(
                value, dtype=__import__("torch").float32
            )
            for key, value in available.items()
        },
        snapshot=snapshot,
        asset_state=asset_state,
        active=active,
    )

    assert tokens.shape == (2, 3, 16)
    assert pooled.shape == (2, 16)
    assert torch.isfinite(tokens).all()
    assert torch.isfinite(pooled).all()
    assert torch.count_nonzero(tokens[0, 2]) == 0
    assert torch.count_nonzero(tokens[1]) == 0
    assert torch.count_nonzero(pooled[1]) == 0
    parameter_count = sum(parameter.numel() for parameter in encoder.parameters())
    assert parameter_count < 12_000_000


def test_every_valid_native_timeframe_receives_gradient() -> None:
    encoder, sequences, available, snapshot, asset_state, active = _case(
        requires_grad=True
    )
    active[:] = True

    tokens, pooled = encoder(
        sequences=sequences,
        available=available,
        staleness={
            key: __import__("torch").zeros_like(
                value, dtype=__import__("torch").float32
            )
            for key, value in available.items()
        },
        snapshot=snapshot,
        asset_state=asset_state,
        active=active,
    )
    (tokens.square().mean() + pooled.square().mean()).backward()

    for timeframe, sequence in sequences.items():
        assert sequence.grad is not None, timeframe
        assert torch.count_nonzero(sequence.grad) > 0, timeframe


def test_fully_missing_timeframe_is_invariant_through_both_attention_axes() -> None:
    encoder, sequences, available, snapshot, asset_state, active = _case()
    encoder.eval()
    available["4h"].zero_()
    changed = {timeframe: value.clone() for timeframe, value in sequences.items()}
    changed["4h"] += 10_000.0

    with torch.no_grad():
        left = encoder(
            sequences=sequences,
            available=available,
            staleness={
                key: __import__("torch").zeros_like(
                    value, dtype=__import__("torch").float32
                )
                for key, value in available.items()
            },
            snapshot=snapshot,
            asset_state=asset_state,
            active=active,
        )
        right = encoder(
            sequences=changed,
            available=available,
            staleness={
                key: __import__("torch").zeros_like(
                    value, dtype=__import__("torch").float32
                )
                for key, value in available.items()
            },
            snapshot=snapshot,
            asset_state=asset_state,
            active=active,
        )

    torch.testing.assert_close(left[0], right[0])
    torch.testing.assert_close(left[1], right[1])
