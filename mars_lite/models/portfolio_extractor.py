"""
ポートフォリオ用特徴抽出器モジュール

PortfolioTradingEnv のflat観測を再構成し、
銘柄共有エンコーダ（置換対称）でエンコードするSB3特徴抽出器。

構成:
    obs = [sym1(feat+w), sym2(feat+w), ..., global]
    → 各銘柄ベクトルを共有MLPでエンコード + 銘柄埋め込みを加算
    → 全銘柄をconcat + グローバル → 共通トランク
"""

import torch
import torch.nn as nn
from gymnasium import spaces
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PortfolioExtractor(BaseFeaturesExtractor):
    """銘柄共有エンコーダ型の特徴抽出器"""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_symbols: int,
        n_per_symbol: int,
        n_global: int,
        symbol_embed_dim: int = 8,
        encoder_dim: int = 32,
        features_dim: int = 128,
    ):
        super().__init__(observation_space, features_dim)

        expected = n_symbols * n_per_symbol + n_global
        if observation_space.shape[0] != expected:
            raise ValueError(
                f"obs dim mismatch: space={observation_space.shape[0]}, "
                f"layout expects {expected} "
                f"({n_symbols}x{n_per_symbol}+{n_global})"
            )

        self.n_symbols = n_symbols
        self.n_per_symbol = n_per_symbol
        self.n_global = n_global

        # 銘柄共有エンコーダ（全銘柄で同一重み）
        self.symbol_encoder = nn.Sequential(
            nn.Linear(n_per_symbol, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, encoder_dim),
            nn.ReLU(),
        )
        # 銘柄ごとの学習可能な埋め込み（対称性を破って銘柄個性を持たせる）
        self.symbol_embedding = nn.Parameter(
            torch.zeros(n_symbols, symbol_embed_dim)
        )
        nn.init.normal_(self.symbol_embedding, std=0.1)

        trunk_in = n_symbols * (encoder_dim + symbol_embed_dim) + n_global
        self.trunk = nn.Sequential(
            nn.Linear(trunk_in, features_dim),
            nn.LayerNorm(features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        sym_flat = observations[:, : self.n_symbols * self.n_per_symbol]
        global_part = observations[:, self.n_symbols * self.n_per_symbol:]

        sym = sym_flat.view(batch, self.n_symbols, self.n_per_symbol)
        encoded = self.symbol_encoder(sym)  # (batch, n_sym, encoder_dim)

        embed = self.symbol_embedding.unsqueeze(0).expand(batch, -1, -1)
        encoded = torch.cat([encoded, embed], dim=2).flatten(1)

        return self.trunk(torch.cat([encoded, global_part], dim=1))
