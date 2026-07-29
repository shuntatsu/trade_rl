from __future__ import annotations

import numpy as np
import pytest
import torch
from torch import nn

from trade_rl.rl.sequence_policy import (
    MultiTimeframeAssetEncoder,
    SequencePolicyArchitecture,
)


class _TraceableAssetEncoder(nn.Module):
    def __init__(self, encoder: MultiTimeframeAssetEncoder) -> None:
        super().__init__()
        self.encoder = encoder

    def forward(
        self,
        sequence_15m: torch.Tensor,
        available_15m: torch.Tensor,
        staleness_15m: torch.Tensor,
        sequence_1h: torch.Tensor,
        available_1h: torch.Tensor,
        staleness_1h: torch.Tensor,
        sequence_4h: torch.Tensor,
        available_4h: torch.Tensor,
        staleness_4h: torch.Tensor,
        sequence_1d: torch.Tensor,
        available_1d: torch.Tensor,
        staleness_1d: torch.Tensor,
        snapshot: torch.Tensor,
        asset_state: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(
            sequences={
                "15m": sequence_15m,
                "1h": sequence_1h,
                "4h": sequence_4h,
                "1d": sequence_1d,
            },
            available={
                "15m": available_15m,
                "1h": available_1h,
                "4h": available_4h,
                "1d": available_1d,
            },
            staleness={
                "15m": staleness_15m,
                "1h": staleness_1h,
                "4h": staleness_4h,
                "1d": staleness_1d,
            },
            snapshot=snapshot,
            asset_state=asset_state,
            active=active,
        )


def _wrapper() -> _TraceableAssetEncoder:
    widths = {timeframe: (8, 8) for timeframe in ("15m", "1h", "4h", "1d")}
    architecture = SequencePolicyArchitecture(
        input_channels={timeframe: 1 for timeframe in widths},
        window_lengths={"15m": 4, "1h": 3, "4h": 2, "1d": 2},
        latent_dims={timeframe: 4 for timeframe in widths},
        asset_state_width=2,
        snapshot_width=2,
        n_symbols=2,
        d_model=8,
        timeframe_attention_heads=2,
        timeframe_attention_layers=1,
        timeframe_ffn_multiplier=2,
        asset_attention_heads=2,
        asset_attention_layers=1,
        asset_ffn_multiplier=2,
        dropout=0.0,
        encoder_widths=widths,
    )
    return _TraceableAssetEncoder(MultiTimeframeAssetEncoder(architecture)).eval()


def _inputs() -> tuple[torch.Tensor, ...]:
    torch.manual_seed(7)
    values: list[torch.Tensor] = []
    for window in (4, 3, 2, 2):
        sequence = torch.randn(2, 2, window, 1)
        available = torch.ones(2, 2, window, 1, dtype=torch.bool)
        staleness = torch.zeros(2, 2, window, 1)
        values.extend((sequence, available, staleness))
    values.extend(
        (
            torch.randn(2, 2, 2),
            torch.randn(2, 2, 2),
            torch.ones(2, 2, dtype=torch.bool),
        )
    )
    return tuple(values)


def _masked_cases(
    example: tuple[torch.Tensor, ...],
) -> tuple[tuple[torch.Tensor, ...], ...]:
    partial = [value.clone() for value in example]
    partial[4][0, 1].zero_()
    partial[7][1, 0].zero_()
    partial[-1][0, 1] = False

    inactive = [value.clone() for value in example]
    inactive[-1].zero_()
    for position in (1, 4, 7, 10):
        inactive[position][1].zero_()

    return example, tuple(partial), tuple(inactive)


@pytest.mark.filterwarnings("error::torch.jit.TracerWarning")
def test_traced_asset_encoder_generalizes_across_availability_and_active_masks() -> (
    None
):
    wrapper = _wrapper()
    example = _inputs()
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, example, strict=False, check_trace=False)
        for inputs in _masked_cases(example):
            expected = wrapper(*inputs)
            actual = traced(*inputs)
            for expected_tensor, actual_tensor in zip(expected, actual, strict=True):
                np.testing.assert_allclose(
                    actual_tensor.detach().numpy(),
                    expected_tensor.detach().numpy(),
                    atol=1e-5,
                    rtol=0.0,
                )
