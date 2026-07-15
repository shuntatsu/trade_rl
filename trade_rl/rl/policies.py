"""Permutation-invariant policy feature extractors for asset-set observations."""

from __future__ import annotations

import torch
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class AssetSetFeatureExtractor(BaseFeaturesExtractor):
    """Encode each asset with shared weights and attention-pool active assets."""

    def __init__(
        self,
        observation_space: spaces.Box,
        *,
        n_symbols: int,
        per_symbol_width: int,
        global_width: int,
        active_column: int,
        asset_embedding_dim: int = 64,
        global_embedding_dim: int = 64,
    ) -> None:
        if n_symbols <= 0 or per_symbol_width <= 0 or global_width <= 0:
            raise ValueError("asset-set dimensions must be positive")
        if not 0 <= active_column < per_symbol_width:
            raise ValueError("active_column is outside the per-symbol observation")
        expected = n_symbols * per_symbol_width + global_width
        if observation_space.shape != (expected,):
            raise ValueError("observation space does not match asset-set dimensions")
        if asset_embedding_dim <= 0 or global_embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        features_dim = asset_embedding_dim + global_embedding_dim
        super().__init__(observation_space, features_dim=features_dim)
        self.n_symbols = n_symbols
        self.per_symbol_width = per_symbol_width
        self.global_width = global_width
        self.active_column = active_column
        self.asset_encoder = nn.Sequential(
            nn.Linear(per_symbol_width, asset_embedding_dim),
            nn.LayerNorm(asset_embedding_dim),
            nn.Tanh(),
            nn.Linear(asset_embedding_dim, asset_embedding_dim),
            nn.Tanh(),
        )
        self.attention_score = nn.Linear(asset_embedding_dim, 1)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, global_embedding_dim),
            nn.LayerNorm(global_embedding_dim),
            nn.Tanh(),
            nn.Linear(global_embedding_dim, global_embedding_dim),
            nn.Tanh(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        asset_width = self.n_symbols * self.per_symbol_width
        assets = observations[:, :asset_width].reshape(
            -1,
            self.n_symbols,
            self.per_symbol_width,
        )
        globals_ = observations[:, asset_width:]
        encoded_assets = self.asset_encoder(assets)
        active = assets[:, :, self.active_column] > 0.5
        scores = self.attention_score(encoded_assets).squeeze(-1)
        scores = scores.masked_fill(~active, -1e9)
        has_active = active.any(dim=1, keepdim=True)
        safe_scores = torch.where(has_active, scores, torch.zeros_like(scores))
        weights = torch.softmax(safe_scores, dim=1)
        weights = torch.where(active, weights, torch.zeros_like(weights))
        denominator = weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        weights = weights / denominator
        pooled_assets = (encoded_assets * weights.unsqueeze(-1)).sum(dim=1)
        pooled_assets = torch.where(
            has_active,
            pooled_assets,
            torch.zeros_like(pooled_assets),
        )
        encoded_globals = self.global_encoder(globals_)
        return torch.cat((pooled_assets, encoded_globals), dim=1)


class SequenceAssetFeatureExtractor(BaseFeaturesExtractor):
    """Encode native-clock sequences and fuse current cross-asset state."""

    def __init__(
        self,
        observation_space: spaces.Dict,
        *,
        feature_counts: dict[str, int],
        window_lengths: dict[str, int],
        snapshot_width: int,
        asset_state_width: int,
        global_width: int,
        n_symbols: int,
        d_model: int = 320,
        attention_heads: int = 8,
        attention_layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        from trade_rl.rl.sequence_policy import (
            MultiTimeframeAssetEncoder,
            SequencePolicyArchitecture,
        )

        timeframes = ("15m", "1h", "4h", "1d")
        if tuple(feature_counts) != timeframes or tuple(window_lengths) != timeframes:
            raise ValueError("sequence metadata must use ordered maintained clocks")
        expected_shapes: dict[str, tuple[int, ...]] = {
            "current_snapshot": (n_symbols, snapshot_width),
            "asset_state": (n_symbols, asset_state_width),
            "global_state": (global_width,),
            "active": (n_symbols,),
        }
        for timeframe in timeframes:
            sequence_shape = (
                n_symbols,
                window_lengths[timeframe],
                feature_counts[timeframe],
            )
            for suffix in ("values", "available", "staleness"):
                expected_shapes[f"sequence_{timeframe}_{suffix}"] = sequence_shape
        for key, expected_shape in expected_shapes.items():
            if key not in observation_space.spaces:
                raise ValueError(f"sequence observation space is missing {key}")
            if observation_space.spaces[key].shape != expected_shape:
                raise ValueError(f"sequence observation shape mismatch for {key}")
        super().__init__(observation_space, features_dim=d_model + 128)
        self.timeframes = timeframes
        architecture = SequencePolicyArchitecture(
            input_channels={
                timeframe: 3 * feature_counts[timeframe] for timeframe in timeframes
            },
            latent_dims={"15m": 192, "1h": 192, "4h": 160, "1d": 128},
            asset_state_width=asset_state_width,
            snapshot_width=snapshot_width,
            d_model=d_model,
            attention_heads=attention_heads,
            attention_layers=attention_layers,
            dropout=dropout,
        )
        self.asset_encoder = MultiTimeframeAssetEncoder(architecture)
        self.global_encoder = nn.Sequential(
            nn.Linear(global_width, 256),
            nn.LayerNorm(256),
            nn.SiLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.SiLU(),
        )

    def forward(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
        sequences: dict[str, torch.Tensor] = {}
        available: dict[str, torch.Tensor] = {}
        for timeframe in self.timeframes:
            values = observations[f"sequence_{timeframe}_values"].float()
            availability = observations[f"sequence_{timeframe}_available"].float()
            staleness = observations[f"sequence_{timeframe}_staleness"].float()
            sequences[timeframe] = torch.cat(
                (values, availability, torch.log1p(staleness.clamp_min(0.0))),
                dim=-1,
            )
            available[timeframe] = availability > 0.5
        assets = self.asset_encoder(
            sequences=sequences,
            available=available,
            snapshot=observations["current_snapshot"].float(),
            asset_state=observations["asset_state"].float(),
            active=observations["active"].float(),
        )
        globals_ = self.global_encoder(observations["global_state"].float())
        return torch.cat((assets, globals_), dim=-1)


__all__ = ["AssetSetFeatureExtractor", "SequenceAssetFeatureExtractor"]
