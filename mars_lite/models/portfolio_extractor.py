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


# 抽出器の規模プリセット。small は実証済み（ARCHITECTURE.mdのベンチ値の構成）、
# large は容量を増やした構成（多様/実データでの汎用性狙い、要再ベンチ）。
ARCH_PRESETS = {
    "small": dict(symbol_embed_dim=8, encoder_dim=32, features_dim=128,
                  tf_encode_dim=16, enc_hidden=(64,), tf_hidden=(),
                  trunk_hidden=()),
    "large": dict(symbol_embed_dim=16, encoder_dim=64, features_dim=256,
                  tf_encode_dim=32, enc_hidden=(128, 128), tf_hidden=(32,),
                  trunk_hidden=(256, 256)),
}


def _mlp(in_dim, hidden, out_dim, dropout=0.0, use_ln=True,
         out_ln=False, out_act=True):
    """Linear/LayerNorm/ReLU(/Dropout) の積層を作る小ヘルパ"""
    layers = []
    d = in_dim
    for h in hidden:
        layers.append(nn.Linear(d, h))
        if use_ln:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.ReLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d = h
    layers.append(nn.Linear(d, out_dim))
    if out_ln:
        layers.append(nn.LayerNorm(out_dim))
    if out_act:
        layers.append(nn.ReLU())
    return nn.Sequential(*layers)


class PortfolioExtractor(BaseFeaturesExtractor):
    """銘柄共有エンコーダ型の特徴抽出器"""

    def __init__(
        self,
        observation_space: spaces.Box,
        n_symbols: int,
        n_per_symbol: int,
        n_global: int,
        size: str = "large",
        dropout: float = 0.0,
    ):
        cfg = ARCH_PRESETS[size]
        symbol_embed_dim = cfg["symbol_embed_dim"]
        encoder_dim = cfg["encoder_dim"]
        features_dim = cfg["features_dim"]
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
        self.symbol_encoder = _mlp(
            n_per_symbol, cfg["enc_hidden"], encoder_dim,
            dropout=dropout, use_ln=True, out_ln=False, out_act=True,
        )
        # 銘柄ごとの学習可能な埋め込み（対称性を破って銘柄個性を持たせる）
        self.symbol_embedding = nn.Parameter(
            torch.zeros(n_symbols, symbol_embed_dim)
        )
        nn.init.normal_(self.symbol_embedding, std=0.1)

        trunk_in = n_symbols * (encoder_dim + symbol_embed_dim) + n_global
        self.trunk = _mlp(
            trunk_in, cfg["trunk_hidden"], features_dim,
            dropout=dropout, use_ln=True, out_ln=True, out_act=True,
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


class TFGatedPortfolioExtractor(BaseFeaturesExtractor):
    """
    TFゲート付き特徴抽出器（多時間軸の構造的解決）

    銘柄特徴を [TFブロック × n_tf | 基準特徴+ウェイト] に分解し、
    - 全TF・全銘柄で共有のTFブロックエンコーダ + TF埋め込み
    - **TFごとの学習可能スカラーゲート**（sigmoid、計n_tf個）
    を通す。役立つ時間軸のゲートは開き、ノイズの時間軸は閉じる。
    ゲートはパラメータ4個程度なので過学習リスクが極小で、学習後に
    `gate_values()` で「モデルがどのTFを使ったか」を可視化できる。
    """

    def __init__(
        self,
        observation_space: spaces.Box,
        n_symbols: int,
        n_per_symbol: int,
        n_global: int,
        n_tf_blocks: int = 4,
        tf_block_size: int = 7,
        size: str = "large",
        dropout: float = 0.0,
    ):
        cfg = ARCH_PRESETS[size]
        symbol_embed_dim = cfg["symbol_embed_dim"]
        tf_encode_dim = cfg["tf_encode_dim"]
        encoder_dim = cfg["encoder_dim"]
        features_dim = cfg["features_dim"]
        super().__init__(observation_space, features_dim)

        expected = n_symbols * n_per_symbol + n_global
        if observation_space.shape[0] != expected:
            raise ValueError(
                f"obs dim mismatch: space={observation_space.shape[0]}, expected {expected}"
            )
        self.n_symbols = n_symbols
        self.n_per_symbol = n_per_symbol
        self.n_global = n_global
        self.n_tf = n_tf_blocks
        self.tf_size = tf_block_size
        self.n_base = n_per_symbol - n_tf_blocks * tf_block_size  # 基準特徴+現ウェイト

        # TFブロック共有エンコーダ（全TF・全銘柄で同一重み。LayerNorm無し=元設計踏襲）
        self.tf_encoder = _mlp(
            tf_block_size, cfg["tf_hidden"], tf_encode_dim,
            dropout=dropout, use_ln=False, out_ln=False, out_act=True,
        )
        # TF埋め込み（TFの個性）と学習可能ゲート（初期値: sigmoid(0)=0.5で全TF半開）
        self.tf_embedding = nn.Parameter(torch.zeros(n_tf_blocks, tf_encode_dim))
        nn.init.normal_(self.tf_embedding, std=0.1)
        self.tf_gate_logits = nn.Parameter(torch.zeros(n_tf_blocks))

        # 銘柄共有エンコーダ
        sym_in = n_tf_blocks * tf_encode_dim + self.n_base
        self.symbol_encoder = _mlp(
            sym_in, cfg["enc_hidden"], encoder_dim,
            dropout=dropout, use_ln=True, out_ln=False, out_act=True,
        )
        self.symbol_embedding = nn.Parameter(torch.zeros(n_symbols, symbol_embed_dim))
        nn.init.normal_(self.symbol_embedding, std=0.1)

        trunk_in = n_symbols * (encoder_dim + symbol_embed_dim) + n_global
        self.trunk = _mlp(
            trunk_in, cfg["trunk_hidden"], features_dim,
            dropout=dropout, use_ln=True, out_ln=True, out_act=True,
        )

    def gate_values(self):
        """学習されたTFゲート値（0=閉/1=開）を返す"""
        return torch.sigmoid(self.tf_gate_logits).detach().cpu().numpy()

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        batch = observations.shape[0]
        sym_flat = observations[:, : self.n_symbols * self.n_per_symbol]
        global_part = observations[:, self.n_symbols * self.n_per_symbol:]

        sym = sym_flat.view(batch, self.n_symbols, self.n_per_symbol)
        tf_part = sym[:, :, : self.n_tf * self.tf_size].view(
            batch, self.n_symbols, self.n_tf, self.tf_size
        )
        base_part = sym[:, :, self.n_tf * self.tf_size:]

        # TFブロックをエンコードし、埋め込み加算 → ゲート乗算
        enc = self.tf_encoder(tf_part)                      # (B, S, T, D)
        enc = enc + self.tf_embedding[None, None, :, :]
        gates = torch.sigmoid(self.tf_gate_logits)          # (T,)
        enc = enc * gates[None, None, :, None]
        enc = enc.flatten(2)                                # (B, S, T*D)

        sym_encoded = self.symbol_encoder(torch.cat([enc, base_part], dim=2))
        embed = self.symbol_embedding.unsqueeze(0).expand(batch, -1, -1)
        sym_encoded = torch.cat([sym_encoded, embed], dim=2).flatten(1)

        return self.trunk(torch.cat([sym_encoded, global_part], dim=1))
