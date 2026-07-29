"""Gated pre-norm transformer primitives for reinforcement-learning policies."""

from __future__ import annotations

import math

import torch
from torch import nn

from trade_rl.rl.export_context import graph_export_active


class GatedResidual(nn.Module):
    """Apply a learnable per-channel gate to one residual branch."""

    def __init__(self, d_model: int, *, gate_bias: float) -> None:
        super().__init__()
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        if not math.isfinite(gate_bias):
            raise ValueError("gate_bias must be finite")
        self.gate = nn.Parameter(torch.full((d_model,), float(gate_bias)))

    def forward(self, residual: torch.Tensor, branch: torch.Tensor) -> torch.Tensor:
        if not graph_export_active():
            if residual.shape != branch.shape:
                raise ValueError(
                    "residual and branch tensors must have identical shapes"
                )
            if residual.ndim != 3 or residual.shape[-1] != self.gate.numel():
                raise ValueError("gated residual expects [batch, tokens, d_model]")
        scale = torch.sigmoid(self.gate).view(1, 1, -1)
        return residual + scale * branch


class GatedTransformerBlock(nn.Module):
    """One pre-norm self-attention/FFN block with gated residual branches."""

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        ffn_multiplier: int,
        dropout: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        _validate_architecture(
            d_model=d_model,
            heads=heads,
            layers=1,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            gate_bias=gate_bias,
        )
        hidden = d_model * ffn_multiplier
        self.attention_norm = nn.LayerNorm(d_model)
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attention_gate = GatedResidual(d_model, gate_bias=gate_bias)
        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )
        self.ffn_gate = GatedResidual(d_model, gate_bias=gate_bias)

    @staticmethod
    def _zero_invalid(value: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
        return value * valid.unsqueeze(-1).to(dtype=value.dtype)

    def _forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
        need_weights: bool,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        normalized = self.attention_norm(value)
        branch, weights = self.attention(
            normalized,
            normalized,
            normalized,
            key_padding_mask=~valid,
            need_weights=need_weights,
            average_attn_weights=False,
        )
        value = self.attention_gate(value, branch)
        value = self._zero_invalid(value, valid)
        branch = self.ffn(self.ffn_norm(value))
        value = self.ffn_gate(value, branch)
        return self._zero_invalid(value, valid), weights

    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        output, _ = self._forward(value, valid=valid, need_weights=False)
        return output

    def diagnostic_forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output, weights = self._forward(value, valid=valid, need_weights=True)
        if weights is None:
            raise RuntimeError("diagnostic attention weights were not produced")
        return output, weights


class GatedTransformerStack(nn.Module):
    """Validated stack that masks invalid tokens before and after every block."""

    def __init__(
        self,
        *,
        d_model: int,
        heads: int,
        layers: int,
        ffn_multiplier: int,
        dropout: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        _validate_architecture(
            d_model=d_model,
            heads=heads,
            layers=layers,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            gate_bias=gate_bias,
        )
        self.d_model = d_model
        self.blocks = nn.ModuleList(
            GatedTransformerBlock(
                d_model=d_model,
                heads=heads,
                ffn_multiplier=ffn_multiplier,
                dropout=dropout,
                gate_bias=gate_bias,
            )
            for _ in range(layers)
        )
        self.output_norm = nn.LayerNorm(d_model)

    def _validated_inputs(
        self,
        value: torch.Tensor,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not graph_export_active():
            if value.ndim != 3 or value.shape[-1] != self.d_model:
                raise ValueError("transformer stack expects [batch, tokens, d_model]")
            if valid.shape != value.shape[:2]:
                raise ValueError("valid mask must match batch and token dimensions")
        valid = valid.to(device=value.device, dtype=torch.bool)
        if not graph_export_active() and torch.any(~valid.any(dim=1)):
            raise ValueError("every batch row requires at least one valid token")
        return value * valid.unsqueeze(-1).to(dtype=value.dtype), valid

    def forward(self, value: torch.Tensor, *, valid: torch.Tensor) -> torch.Tensor:
        value, valid = self._validated_inputs(value, valid)
        for block in self.blocks:
            value = block(value, valid=valid)
        value = self.output_norm(value)
        return value * valid.unsqueeze(-1).to(dtype=value.dtype)

    def diagnostic_forward(
        self,
        value: torch.Tensor,
        *,
        valid: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...]]:
        value, valid = self._validated_inputs(value, valid)
        weights: list[torch.Tensor] = []
        for block in self.blocks:
            value, block_weights = block.diagnostic_forward(value, valid=valid)
            weights.append(block_weights)
        value = self.output_norm(value)
        output = value * valid.unsqueeze(-1).to(dtype=value.dtype)
        return output, tuple(weights)


def _validate_architecture(
    *,
    d_model: int,
    heads: int,
    layers: int,
    ffn_multiplier: int,
    dropout: float,
    gate_bias: float,
) -> None:
    for field, value in (
        ("d_model", d_model),
        ("heads", heads),
        ("layers", layers),
        ("ffn_multiplier", ffn_multiplier),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{field} must be a positive integer")
    if d_model % heads != 0:
        raise ValueError("d_model must divide evenly across attention heads")
    if not math.isfinite(dropout) or not 0.0 <= dropout <= 0.05:
        raise ValueError("dropout must be within [0, 0.05]")
    if not math.isfinite(gate_bias):
        raise ValueError("gate_bias must be finite")


__all__ = ["GatedResidual", "GatedTransformerBlock", "GatedTransformerStack"]
