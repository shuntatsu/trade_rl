"""Quality-aware gated attention across maintained native market timeframes."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import nn

from trade_rl.rl.export_context import graph_export_active
from trade_rl.rl.gated_transformer import GatedTransformerStack

_TIMEFRAMES = ("15m", "1h", "4h", "1d")
_TIMEFRAME_MINUTES = {"15m": 15.0, "1h": 60.0, "4h": 240.0, "1d": 1440.0}
_QUALITY_WIDTH = 5


def _positive_ordered_mapping(
    value: Mapping[str, int], *, field: str
) -> dict[str, int]:
    if tuple(value) != _TIMEFRAMES:
        raise ValueError(f"{field} must use ordered 15m/1h/4h/1d timeframes")
    resolved = dict(value)
    if any(
        isinstance(item, bool) or not isinstance(item, int) or item <= 0
        for item in resolved.values()
    ):
        raise ValueError(f"{field} must contain positive integers")
    return resolved


def _quality_summary(
    available: torch.Tensor,
    staleness: torch.Tensor,
    *,
    window_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not graph_export_active():
        if available.ndim not in {3, 4}:
            raise ValueError(
                "availability must be [batch, assets, time] or include channels"
            )
        if staleness.shape != available.shape:
            raise ValueError("staleness must match availability shape")
        if available.shape[2] != window_length:
            raise ValueError("quality plane window does not match architecture")
    available = available.to(dtype=torch.bool)
    usable = available.any(dim=-1) if available.ndim == 4 else available
    has_any = usable.any(dim=-1)
    positions = torch.arange(window_length, device=available.device).view(1, 1, -1)
    positions = positions.expand_as(usable)
    last_index = positions.masked_fill(~usable, -1).max(dim=-1).values
    safe_index = last_index.clamp_min(0)
    available_fraction = usable.to(dtype=torch.float32).mean(dim=-1)
    last_fraction = safe_index.to(dtype=torch.float32) / float(
        max(window_length - 1, 1)
    )

    if available.ndim == 4:
        channels = available.shape[-1]
        gather_index = (
            safe_index.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, channels)
        )
        selected_available = available.gather(2, gather_index).squeeze(2)
        selected_staleness = staleness.gather(2, gather_index).squeeze(2)
        selected_fraction = selected_available.to(dtype=torch.float32).mean(dim=-1)
        logged = torch.log1p(selected_staleness.to(dtype=torch.float32).clamp_min(0.0))
        weights = selected_available.to(dtype=torch.float32)
        denominator = weights.sum(dim=-1).clamp_min(1.0)
        stale_mean = (logged * weights).sum(dim=-1) / denominator
        stale_max = logged.masked_fill(~selected_available, 0.0).max(dim=-1).values
    else:
        gather_index = safe_index.unsqueeze(-1)
        selected_available = available.gather(2, gather_index).squeeze(2)
        selected_staleness = staleness.gather(2, gather_index).squeeze(2)
        selected_fraction = selected_available.to(dtype=torch.float32)
        logged = torch.log1p(selected_staleness.to(dtype=torch.float32).clamp_min(0.0))
        stale_mean = logged * selected_fraction
        stale_max = stale_mean

    quality = torch.stack(
        (
            available_fraction,
            last_fraction,
            selected_fraction,
            stale_mean,
            stale_max,
        ),
        dim=-1,
    )
    quality = torch.where(has_any.unsqueeze(-1), quality, torch.zeros_like(quality))
    return quality, has_any


class CrossTimeframeFusion(nn.Module):
    """Fuse native-clock latents into one decision-conditioned token per asset."""

    def __init__(
        self,
        *,
        latent_dims: Mapping[str, int],
        window_lengths: Mapping[str, int],
        d_model: int,
        heads: int,
        layers: int,
        ffn_multiplier: int,
        dropout: float,
        gate_bias: float,
    ) -> None:
        super().__init__()
        self.latent_dims = _positive_ordered_mapping(latent_dims, field="latent_dims")
        self.window_lengths = _positive_ordered_mapping(
            window_lengths, field="window_lengths"
        )
        if isinstance(d_model, bool) or not isinstance(d_model, int) or d_model <= 0:
            raise ValueError("d_model must be a positive integer")
        self.d_model = d_model
        self.timeframes = _TIMEFRAMES
        self.latent_projectors = nn.ModuleDict(
            {
                timeframe: nn.Sequential(
                    nn.Linear(self.latent_dims[timeframe], d_model),
                    nn.LayerNorm(d_model),
                    nn.SiLU(),
                )
                for timeframe in self.timeframes
            }
        )
        self.quality_encoder = nn.Sequential(
            nn.Linear(_QUALITY_WIDTH, d_model),
            nn.LayerNorm(d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model),
        )
        duration_features = []
        for timeframe in self.timeframes:
            minutes = _TIMEFRAME_MINUTES[timeframe]
            duration_features.append(
                (
                    math.log1p(minutes),
                    math.log1p(minutes * self.window_lengths[timeframe]),
                )
            )
        self.register_buffer(
            "duration_features",
            torch.tensor(duration_features, dtype=torch.float32),
            persistent=True,
        )
        self.duration_encoder = nn.Linear(2, d_model, bias=False)
        self.timeframe_embedding = nn.Embedding(len(self.timeframes), d_model)
        self.context_norm = nn.LayerNorm(d_model)
        self.transformer = GatedTransformerStack(
            d_model=d_model,
            heads=heads,
            layers=layers,
            ffn_multiplier=ffn_multiplier,
            dropout=dropout,
            gate_bias=gate_bias,
        )

    def _prepared_tokens(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, int, int]:
        if not graph_export_active():
            if tuple(latents) != self.timeframes:
                raise ValueError("latents must use ordered 15m/1h/4h/1d timeframes")
            if (
                tuple(available) != self.timeframes
                or tuple(staleness) != self.timeframes
            ):
                raise ValueError(
                    "quality planes must use ordered 15m/1h/4h/1d timeframes"
                )
            if context.ndim != 3 or context.shape[-1] != self.d_model:
                raise ValueError("context must be [batch, assets, d_model]")
        batch, assets, _ = context.shape
        tokens = [self.context_norm(context)]
        valid_tokens = [
            torch.ones(batch, assets, dtype=torch.bool, device=context.device)
        ]
        duration = self.duration_encoder(
            self.duration_features.to(device=context.device, dtype=context.dtype)
        )
        identities = self.timeframe_embedding(
            torch.arange(len(self.timeframes), device=context.device)
        ).to(dtype=context.dtype)

        for index, timeframe in enumerate(self.timeframes):
            latent = latents[timeframe]
            if not graph_export_active() and latent.shape != (
                batch,
                assets,
                self.latent_dims[timeframe],
            ):
                raise ValueError("timeframe latent shape does not match architecture")
            plane = available[timeframe]
            stale = staleness[timeframe]
            if not graph_export_active() and plane.shape[:2] != (batch, assets):
                raise ValueError("timeframe availability batch or asset shape mismatch")
            quality, has_any = _quality_summary(
                plane,
                stale,
                window_length=self.window_lengths[timeframe],
            )
            quality_token = self.quality_encoder(
                quality.to(device=latent.device, dtype=latent.dtype)
            )
            token = self.latent_projectors[timeframe](latent)
            token = token + quality_token + duration[index] + identities[index]
            tokens.append(token)
            valid_tokens.append(has_any.to(device=context.device))

        stacked = torch.stack(tokens, dim=2)
        valid = torch.stack(valid_tokens, dim=2)
        flattened = stacked.reshape(batch * assets, len(tokens), self.d_model)
        flattened_valid = valid.reshape(batch * assets, len(tokens))
        return flattened, flattened_valid, batch, assets

    def forward(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> torch.Tensor:
        flattened, valid, batch, assets = self._prepared_tokens(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        contextual = self.transformer(flattened, valid=valid)
        return contextual[:, 0].reshape(batch, assets, self.d_model)

    def diagnostic_forward(
        self,
        *,
        latents: Mapping[str, torch.Tensor],
        available: Mapping[str, torch.Tensor],
        staleness: Mapping[str, torch.Tensor],
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, ...], torch.Tensor]:
        flattened, valid, batch, assets = self._prepared_tokens(
            latents=latents,
            available=available,
            staleness=staleness,
            context=context,
        )
        contextual, weights = self.transformer.diagnostic_forward(
            flattened,
            valid=valid,
        )
        output = contextual[:, 0].reshape(batch, assets, self.d_model)
        return output, weights, valid


__all__ = ["CrossTimeframeFusion"]
