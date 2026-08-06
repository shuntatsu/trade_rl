"""Pure PyTorch causal sequence encoders shared by BC and PPO policies."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping

import torch
from torch import nn

from trade_rl.rl.export_context import graph_export_active
from trade_rl.rl.gated_transformer import GatedTransformerStack
from trade_rl.rl.timeframe_fusion import CrossTimeframeFusion

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_DEFAULT_WIDTHS: dict[str, tuple[int, ...]] = {
    "15m": (64, 96, 128, 160, 192, 192),
    "1h": (64, 96, 128, 160, 192, 192, 192),
    "4h": (48, 80, 112, 144, 160, 160, 160),
    "1d": (32, 64, 96, 112, 128),
}
_COMPACT_WIDTHS: dict[str, tuple[int, ...]] = {
    "15m": (40, 56, 72, 96, 112, 112),
    "1h": (40, 56, 72, 96, 112, 112, 112),
    "4h": (32, 48, 64, 88, 96, 96),
    "1d": (24, 40, 56, 72, 80),
}


def sequence_encoder_widths(capacity: str) -> dict[str, tuple[int, ...]]:
    """Resolve an explicit maintained temporal-capacity preset."""

    if capacity == "standard":
        return dict(_DEFAULT_WIDTHS)
    if capacity == "compact":
        return dict(_COMPACT_WIDTHS)
    raise ValueError("sequence capacity must be standard or compact")


def _required_dilations(window_length: int, *, kernel_size: int = 3) -> tuple[int, ...]:
    """Return power-of-two dilations whose receptive field covers the window."""

    if window_length <= 0 or kernel_size <= 1:
        raise ValueError(
            "window_length must be positive and kernel_size must exceed one"
        )
    dilations: list[int] = []
    receptive_field = 1
    dilation = 1
    while receptive_field < window_length:
        dilations.append(dilation)
        receptive_field += (kernel_size - 1) * dilation
        dilation *= 2
    return tuple(dilations)


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

    def __init__(
        self,
        input_channels: int,
        latent_dim: int,
        *,
        window_length: int,
        widths: tuple[int, ...],
        dropout: float,
    ) -> None:
        super().__init__()
        if input_channels <= 0 or latent_dim <= 0 or window_length <= 0:
            raise ValueError("timeframe encoder dimensions must be positive")
        dilations = _required_dilations(window_length)
        if not widths:
            raise ValueError("timeframe encoder widths must not be empty")
        if len(widths) < len(dilations):
            widths = widths + (widths[-1],) * (len(dilations) - len(widths))
        elif len(widths) > len(dilations):
            widths = widths[: len(dilations)]
        blocks: list[nn.Module] = []
        current = input_channels
        for width, dilation in zip(widths, dilations, strict=True):
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
        self.latent_dim = latent_dim
        self.window_length = window_length
        self.dilations = dilations
        self.receptive_field = 1 + 2 * sum(dilations)
        if self.receptive_field < window_length:
            raise RuntimeError(
                "timeframe encoder receptive field does not cover window"
            )
        hidden = max(latent_dim, current)
        self.projection = nn.Sequential(
            nn.Linear(current, hidden),
            nn.LayerNorm(hidden),
            nn.SiLU(),
            nn.Linear(hidden, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.SiLU(),
        )

    def forward_sequence(self, value: torch.Tensor) -> torch.Tensor:
        if not graph_export_active():
            if value.ndim != 3:
                raise ValueError("timeframe input must be [batch, time, channels]")
            if value.shape[1] != self.window_length:
                raise ValueError("timeframe input length does not match architecture")
        return self.blocks(value.transpose(1, 2)).transpose(1, 2)

    def forward(
        self, value: torch.Tensor, available: torch.Tensor | None = None
    ) -> torch.Tensor:
        if available is None:
            encoded = self.forward_sequence(value)
            return self.projection(encoded[:, -1])
        if not graph_export_active() and available.shape != value.shape[:2]:
            raise ValueError("availability mask must match batch and time dimensions")
        mask = available.to(dtype=torch.bool)
        positions = torch.arange(value.shape[1], device=value.device).expand_as(mask)
        indices = positions.masked_fill(~mask, -1).max(dim=1).values
        valid = indices >= 0
        valid_values = value[valid]
        encoded = self.forward_sequence(valid_values)
        valid_indices = indices[valid]
        selected = encoded[
            torch.arange(encoded.shape[0], device=value.device), valid_indices
        ]
        projected = self.projection(selected)
        output = projected.new_zeros((value.shape[0], self.latent_dim))
        batch_indices = torch.arange(value.shape[0], device=value.device)[valid]
        return output.index_copy(0, batch_indices, projected)


@dataclass(frozen=True, slots=True)
class SequencePolicyArchitecture:
    input_channels: Mapping[str, int]
    window_lengths: Mapping[str, int]
    latent_dims: Mapping[str, int]
    asset_state_width: int
    snapshot_width: int
    n_symbols: int
    d_model: int = 320
    timeframe_attention_heads: int = 8
    timeframe_attention_layers: int = 2
    timeframe_ffn_multiplier: int = 3
    timeframe_gate_bias: float = -2.0
    asset_attention_heads: int = 8
    asset_attention_layers: int = 2
    asset_ffn_multiplier: int = 3
    asset_gate_bias: float = -2.0
    dropout: float = 0.05
    encoder_widths: Mapping[str, tuple[int, ...]] | None = None

    def __post_init__(self) -> None:
        if (
            tuple(self.input_channels) != _TIMEFRAMES
            or tuple(self.window_lengths) != _TIMEFRAMES
            or tuple(self.latent_dims) != _TIMEFRAMES
        ):
            raise ValueError(
                "sequence architecture requires ordered 15m/1h/4h/1d clocks"
            )
        if any(value <= 0 for value in self.input_channels.values()):
            raise ValueError("sequence input channels must be positive")
        if any(value <= 0 for value in self.window_lengths.values()):
            raise ValueError("sequence window lengths must be positive")
        if any(value <= 0 for value in self.latent_dims.values()):
            raise ValueError("sequence latent dimensions must be positive")
        if (
            min(
                self.asset_state_width,
                self.snapshot_width,
                self.n_symbols,
                self.d_model,
            )
            <= 0
        ):
            raise ValueError("sequence architecture widths must be positive")
        for field_name, heads in (
            ("timeframe_attention_heads", self.timeframe_attention_heads),
            ("asset_attention_heads", self.asset_attention_heads),
        ):
            if self.d_model % heads != 0:
                raise ValueError(f"d_model must be divisible by {field_name}")
        for field_name, value in (
            ("timeframe_attention_layers", self.timeframe_attention_layers),
            ("timeframe_ffn_multiplier", self.timeframe_ffn_multiplier),
            ("asset_attention_layers", self.asset_attention_layers),
            ("asset_ffn_multiplier", self.asset_ffn_multiplier),
        ):
            if value <= 0:
                raise ValueError(f"{field_name} must be positive")
        for field_name, gate_value in (
            ("timeframe_gate_bias", self.timeframe_gate_bias),
            ("asset_gate_bias", self.asset_gate_bias),
        ):
            if not math.isfinite(gate_value):
                raise ValueError(f"{field_name} must be finite")
        if not 0.0 <= self.dropout <= 0.05:
            raise ValueError("sequence dropout must be within [0, 0.05]")
        widths = self.encoder_widths or _DEFAULT_WIDTHS
        if tuple(widths) != _TIMEFRAMES:
            raise ValueError("encoder widths must use ordered maintained clocks")
        if any(
            not value or any(item <= 0 for item in value) for value in widths.values()
        ):
            raise ValueError("encoder widths must contain positive integers")
        object.__setattr__(self, "encoder_widths", dict(widths))


class MultiTimeframeAssetEncoder(nn.Module):
    """Hierarchically fuse native-clock histories and optional asset context."""

    def __init__(self, architecture: SequencePolicyArchitecture) -> None:
        super().__init__()
        self.architecture = architecture
        self.timeframes = tuple(architecture.input_channels)
        assert architecture.encoder_widths is not None
        self.timeframe_encoders = nn.ModuleDict(
            {
                timeframe: CausalTimeframeEncoder(
                    architecture.input_channels[timeframe],
                    architecture.latent_dims[timeframe],
                    window_length=architecture.window_lengths[timeframe],
                    widths=architecture.encoder_widths[timeframe],
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
        self.context_encoder = nn.Sequential(
            nn.Linear(352, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, architecture.d_model),
            nn.LayerNorm(architecture.d_model),
            nn.SiLU(),
        )
        self.timeframe_fusion = CrossTimeframeFusion(
            latent_dims=architecture.latent_dims,
            window_lengths=architecture.window_lengths,
            d_model=architecture.d_model,
            heads=architecture.timeframe_attention_heads,
            layers=architecture.timeframe_attention_layers,
            ffn_multiplier=architecture.timeframe_ffn_multiplier,
            dropout=architecture.dropout,
            gate_bias=architecture.timeframe_gate_bias,
        )
        self.cross_asset = GatedTransformerStack(
            d_model=architecture.d_model,
            heads=architecture.asset_attention_heads,
            layers=architecture.asset_attention_layers,
            ffn_multiplier=architecture.asset_ffn_multiplier,
            dropout=architecture.dropout,
            gate_bias=architecture.asset_gate_bias,
        )

    def forward(
        self,
        *,
        sequences: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        snapshot: torch.Tensor,
        asset_state: torch.Tensor,
        active: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not graph_export_active():
            if snapshot.ndim != 3 or asset_state.ndim != 3 or active.ndim != 2:
                raise ValueError("asset encoder expects batched asset tensors")
        batch, assets, _ = snapshot.shape
        if not graph_export_active():
            if assets != self.architecture.n_symbols:
                raise ValueError("asset count does not match architecture")
            if asset_state.shape[:2] != (batch, assets) or active.shape != (
                batch,
                assets,
            ):
                raise ValueError("asset tensors disagree on batch or asset dimensions")
        latents: dict[str, torch.Tensor] = {}
        quality_available: dict[str, torch.Tensor] = {}
        quality_staleness: dict[str, torch.Tensor] = {}
        for timeframe in self.timeframes:
            sequence = sequences[timeframe]
            availability = available[timeframe]
            if not graph_export_active():
                if sequence.ndim != 4 or sequence.shape[:2] != (batch, assets):
                    raise ValueError(
                        "sequence tensor has invalid batch or asset dimensions"
                    )
                if sequence.shape[2] != self.architecture.window_lengths[timeframe]:
                    raise ValueError("sequence length does not match architecture")
                if sequence.shape[-1] != self.architecture.input_channels[timeframe]:
                    raise ValueError(
                        "sequence channel count does not match architecture"
                    )
                if (
                    availability.ndim not in {3, 4}
                    or availability.shape[:3] != sequence.shape[:3]
                ):
                    raise ValueError("sequence availability shape is invalid")
            raw_staleness = staleness[timeframe]
            if not graph_export_active() and raw_staleness.shape != availability.shape:
                raise ValueError("sequence staleness must match sequence availability")
            timestep_mask = (
                availability.any(dim=-1) if availability.ndim == 4 else availability
            )
            flattened = sequence.reshape(
                batch * assets, sequence.shape[2], sequence.shape[3]
            )
            flattened_mask = timestep_mask.reshape(batch * assets, sequence.shape[2])
            encoded = self.timeframe_encoders[timeframe](flattened, flattened_mask)
            latents[timeframe] = encoded.reshape(batch, assets, -1)
            quality_available[timeframe] = availability
            quality_staleness[timeframe] = raw_staleness
        context = self.context_encoder(
            torch.cat(
                (
                    self.snapshot_encoder(snapshot),
                    self.asset_state_encoder(asset_state),
                ),
                dim=-1,
            )
        )
        fused = self.timeframe_fusion(
            latents=latents,
            available=quality_available,
            staleness=quality_staleness,
            context=context,
        )
        asset_positions = torch.arange(assets, device=fused.device)
        active_mask = active.to(dtype=torch.bool)
        has_active = active_mask.any(dim=1)
        fallback = (~has_active).unsqueeze(1) & asset_positions.unsqueeze(0).eq(0)
        safe_mask = active_mask | fallback
        fused = torch.where(fallback.unsqueeze(-1), torch.zeros_like(fused), fused)
        if self.architecture.n_symbols == 1:
            contextual = fused * safe_mask.unsqueeze(-1).to(dtype=fused.dtype)
        else:
            contextual = self.cross_asset(fused, valid=safe_mask)
        contextual = torch.where(
            active_mask.unsqueeze(-1), contextual, torch.zeros_like(contextual)
        )
        weights = active_mask.to(dtype=contextual.dtype).unsqueeze(-1)
        pooled = (contextual * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        pooled = torch.where(has_active.unsqueeze(1), pooled, torch.zeros_like(pooled))
        return contextual, pooled


__all__ = [
    "CausalTemporalBlock",
    "CausalTimeframeEncoder",
    "MultiTimeframeAssetEncoder",
    "SequencePolicyArchitecture",
    "sequence_encoder_widths",
]
