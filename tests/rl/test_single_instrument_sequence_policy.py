from __future__ import annotations

import torch
from torch import nn

from trade_rl.rl.gated_transformer import GatedTransformerStack
from trade_rl.rl.sequence_architecture import sequence_architecture_identity
from trade_rl.rl.sequence_policy import SequencePolicyArchitecture


class _ForbiddenBlock(nn.Module):
    def forward(self, *args: object, **kwargs: object) -> torch.Tensor:
        raise AssertionError("single-token path must not execute transformer blocks")

    def diagnostic_forward(self, *args: object, **kwargs: object) -> object:
        raise AssertionError("single-token path must not execute diagnostics")


def _stack() -> GatedTransformerStack:
    stack = GatedTransformerStack(
        d_model=8,
        heads=2,
        layers=2,
        ffn_multiplier=2,
        dropout=0.0,
        gate_bias=-2.0,
    )
    stack.blocks = nn.ModuleList((_ForbiddenBlock(), _ForbiddenBlock()))
    return stack


def test_one_token_path_bypasses_attention_ffn_and_output_norm() -> None:
    stack = _stack()
    value = torch.randn(3, 1, 8)
    valid = torch.ones(3, 1, dtype=torch.bool)

    output = stack(value, valid=valid)
    diagnostic, weights = stack.diagnostic_forward(value, valid=valid)

    torch.testing.assert_close(output, value)
    torch.testing.assert_close(diagnostic, value)
    assert weights == ()


def test_one_symbol_architecture_identity_differs_from_three_symbol_identity() -> None:
    def architecture(n_symbols: int) -> SequencePolicyArchitecture:
        return SequencePolicyArchitecture(
            input_channels={"15m": 3, "1h": 3, "4h": 3, "1d": 3},
            window_lengths={"15m": 4, "1h": 4, "4h": 4, "1d": 4},
            latent_dims={"15m": 8, "1h": 8, "4h": 8, "1d": 8},
            asset_state_width=4,
            snapshot_width=8,
            n_symbols=n_symbols,
            d_model=24,
            timeframe_attention_heads=4,
            timeframe_attention_layers=1,
            timeframe_ffn_multiplier=2,
            timeframe_gate_bias=-2.0,
            asset_attention_heads=4,
            asset_attention_layers=1,
            asset_ffn_multiplier=2,
            asset_gate_bias=-2.0,
            dropout=0.0,
        )

    single = sequence_architecture_identity(architecture(1))
    legacy = sequence_architecture_identity(architecture(3))

    assert single.digest_payload()["n_symbols"] == 1
    assert legacy.digest_payload()["n_symbols"] == 3
    assert single.digest != legacy.digest
