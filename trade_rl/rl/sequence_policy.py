"""Pure PyTorch causal sequence encoders shared by BC and PPO policies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn


def _group_count(channels: int) -> int:
    for candidate in (16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class CausalTemporalBlock(nn.Module):
    """Residual temporal convolution with left-only padding."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if min(in_channels, out_channels, kernel_size, dilation) <= 0:
            raise ValueError("temporal block dimensions must be positive")
        if not 0.0 <= dropout <= 0.05:
            raise ValueError("sequence dropout must be within [0, 0.05]")
        self.left_padding = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
        )
        self.norm = nn.LayerNorm(out_channels)
        self.activation = nn.SiLU()
        self.dropout = nn.Dropout(dropout)
        self.residual = (
            nn.Identity()
            if in_channels == out_channels
            else nn.Conv1d(in_channels, out_channels, kernel_size=1)
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("temporal block expects [batch, channels, time]")
        padded = torch.nn.functional.pad(value, (self.left_padding, 0))
        encoded = self.conv(padded)
        encoded = encoded.transpose(1, 2)
        encoded = self.norm(encoded).transpose(1, 2)
        encoded = self.dropout(self.activation(encoded))
        return self.activation(encoded + self.residual(value))


class CausalTimeframeEncoder(nn.Module):
    """Encode one native clock and pool its last available causal state."""

    def __init__(self, input_channels: int, latent_dim: int, *, dropout: float) -> None:
        super().__init__()
        if input_channels <= 0 or latent_dim <= 0:
            raise ValueError("timeframe encoder dimensions must be positive")
        widths = (64, 96, 128, 160, 192)
        blocks: list[nn.Module] = []
        current = input_channels
        for width, dilation in zip(widths, (1, 2, 4, 8, 16), strict=True):
            blocks.append(
                CausalTemporalBlock(
                    in_channels=current,
                    out_channels=width,
                    kernel_size=3,
                    dilation=dilation,
                    dropout=dropout,
                )
            )
            current = width
        self.blocks = nn.Sequential(*blocks)
        self.projection = nn.Sequential(
            nn.Linear(current, max(latent_dim, 192)),
            nn.LayerNorm(max(latent_dim, 192)),
            nn.SiLU(),
            nn.Linear(max(latent_dim, 192), latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )

    def forward_sequence(self, value: torch.Tensor) -> torch.Tensor:
        if value.ndim != 3:
            raise ValueError("timeframe input must be [batch, time, channels]")
        encoded = self.blocks(value.transpose(1, 2)).transpose(1, 2)
        return self.projection(encoded)

    def forward(
        self, value: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        encoded = self.forward_sequence(value)
        if available is None:
            return encoded[:, -1]
        if available.shape != value.shape[:2]:
            raise ValueError("availability mask must match batch and time dimensions")
        mask = available.to(dtype=torch.bool)
        positions = torch.arange(value.shape[1], device=value.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        safe = indices.clamp_min(0)
        selected = encoded[torch.arange(value.shape[0], device=value.device), safe]
        return torch.where(
            (indices >= 0).unsqueeze(1), selected, torch.zeros_like(selected)
        )


@dataclass(frozen=True, slots=True)
class SequencePolicyArchitecture:
    input_channels: Mapping[str, int]
    latent_dims: Mapping[str, int]
    asset_state_width: int
    snapshot_width: int
    d_model: int = 320
    attention_heads: int = 8
    attention_layers: int = 2
    dropout: float = 0.05

    def __post_init__(self) -> None:
        expected = ("15m", "1h", "4h", "1d")
        if (
            tuple(self.input_channels) != expected
            or tuple(self.latent_dims) != expected
        ):
            raise ValueError(
                "sequence architecture requires ordered 15m/1h/4h/1d clocks"
            )
        if any(value <= 0 for value in self.input_channels.values()):
            raise ValueError("sequence input channels must be positive")
        if any(value <= 0 for value in self.latent_dims.values()):
            raise ValueError("sequence latent dimensions must be positive")
        if min(self.asset_state_width, self.snapshot_width, self.d_model) <= 0:
            raise ValueError("sequence architecture widths must be positive")
        if self.d_model % self.attention_heads != 0:
            raise ValueError("d_model must be divisible by attention_heads")
        if self.attention_layers <= 0:
            raise ValueError("attention_layers must be positive")
        if not 0.0 <= self.dropout <= 0.05:
            raise ValueError("sequence dropout must be within [0, 0.05]")


class MultiTimeframeAssetEncoder(nn.Module):
    """Fuse native-clock histories, current state, and cross-asset context."""

    def __init__(self, architecture: SequencePolicyArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.timeframes = tuple(architecture.input_channels)
        self.timeframe_encoders = nn.ModuleDict(
            {
                timeframe: CausalTimeframeEncoder(
                    architecture.input_channels[timeframe],
                    architecture.latent_dims[timeframe],
                    dropout=architecture.dropout,
                )
                for timeframe in self.timeframes
            }
        )
        self.snapshot_encoder = nn.Sequential(
            nn.Linear(architecture.snapshot_width, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
        )
        self.asset_state_encoder = nn.Sequential(
            nn.Linear(architecture.asset_state_width, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
            nn.Linear(128, 96),
            nn.LayerNorm(96),
            nn.SiLU(),
        )
        fusion_input = sum(architecture.latent_dims.values()) + 256 + 96
        self.asset_fusion = nn.Sequential(
            nn.Linear(fusion_input, 640),
            nn.LayerNorm(640),
            nn.SiLU(),
            nn.Linear(640, 384),
            nn.LayerNorm(384),
            nn.SiLU(),
            nn.Linear(384, architecture.d_model),
            nn.LayerNorm(architecture.d_model),
            nn.SiLU(),
        )
        layer = nn.TransformerEncoderLayer(
            d_model=architecture.d_model,
            nhead=architecture.attention_heads,
            dim_feedforward=architecture.d_model * 3,
            dropout=architecture.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.cross_asset = nn.TransformerEncoder(
            layer, num_layers=architecture.attention_layers
        )

    def forward(
        self,
        *,
        sequences: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        snapshot: torch.Tensor,
        asset_state: torch.Tensor,
        active: torch.Tensor,
    ) -> torch.Tensor:
        if snapshot.ndim != 3 or asset_state.ndim != 3 or active.ndim != 2:
            raise ValueError("asset encoder expects batched asset tensors")
        batch, assets, _ = snapshot.shape
        if asset_state.shape[:2] != (batch, assets) or active.shape != (batch, assets):
            raise ValueError("asset tensors disagree on batch or asset dimensions")
        parts: list[torch.Tensor] = []
        for timeframe in self.timeframes:
            sequence = sequences[timeframe]
            mask = available[timeframe]
            if sequence.ndim != 4 or sequence.shape[:2] != (batch, assets):
                raise ValueError(
                    "sequence tensor has invalid batch or asset dimensions"
                )
            if sequence.shape[-1] != self.architecture.input_channels[timeframe]:
                raise ValueError("sequence channel count does not match architecture")
            if mask.ndim == 4:
                mask = mask.any(dim=-1)
            if mask.shape != sequence.shape[:3]:
                raise ValueError("sequence availability shape is invalid")
            flattened = sequence.reshape(
                batch * assets, sequence.shape[2], sequence.shape[3]
            )
            flattened_mask = mask.reshape(batch * assets, sequence.shape[2])
            encoded = self.timeframe_encoders[timeframe](flattened, flattened_mask)
            parts.append(encoded.reshape(batch, assets, -1))
        parts.append(self.snapshot_encoder(snapshot))
        parts.append(self.asset_state_encoder(asset_state))
        fused = self.asset_fusion(torch.cat(parts, dim=-1))

        active_mask = active.to(dtype=torch.bool)
        has_active = active_mask.any(dim=1)
        safe_mask = active_mask.clone()
        if torch.any(~has_active):
            safe_mask[~has_active, 0] = True
            fused = fused.clone()
            fused[~has_active, 0] = 0.0
        contextual = self.cross_asset(fused, src_key_padding_mask=~safe_mask)
        weights = active_mask.to(dtype=contextual.dtype).unsqueeze(-1)
        pooled = (contextual * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return torch.where(has_active.unsqueeze(1), pooled, torch.zeros_like(pooled))


__all__ = [
    "CausalTemporalBlock",
    "CausalTimeframeEncoder",
    "MultiTimeframeAssetEncoder",
    "SequencePolicyArchitecture",
]
