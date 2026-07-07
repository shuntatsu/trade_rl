"""
後処理モジュール（方策の生ウェイト → 執行可能ウェイト）

重要: このPostProcessorは学習環境(step)と運用推論の**両方**で同一に適用する。
学習時と運用時で異なる後処理を通すと、バックテストが実運用を予測しなくなる
（train/serve skew）。そのためロジックはここに一元化する。

パイプライン:
  raw → ①EMA平滑 → ②銘柄集中上限 → ③レバレッジ1射影 → ④ボラターゲティング
      → ⑤ドローダウン応答（デリスキング） → ⑥アンサンブル不一致縮小
      → ⑦no-tradeバンド（微小取引抑制） → 執行ウェイト
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np

BARS_PER_YEAR_1H = 24 * 365


@dataclass
class PostProcessConfig:
    """後処理パラメータ"""
    ema_alpha: float = 1.0              # 1.0=平滑なし、<1で生ウェイトへEMA追従
    max_weight: float = 0.4             # 銘柄あたりの絶対ウェイト上限
    no_trade_band: float = 0.02         # |Δw|がこの幅未満なら据え置き
    target_vol: Optional[float] = None  # 年率ボラ目標（Noneで無効）。例: 0.20
    vol_lookback: int = 48              # ボラ推定に使う直近バー数
    dd_derisk_start: float = 0.10       # このDDからグロス縮小開始
    dd_derisk_floor: float = 0.3        # DD悪化時のグロス下限倍率
    disagreement_penalty: float = 1.0   # アンサンブル不一致によるグロス縮小の強さ
    beta_neutral: bool = False          # 市場方向(等ウェイト)成分を除去しドル中立化
    bars_per_year: int = BARS_PER_YEAR_1H

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


@dataclass
class PostProcessInfo:
    """後処理の各段の状態（監視・ガードレール用）"""
    raw_gross: float = 0.0
    processed_gross: float = 0.0
    vol_scale: float = 1.0
    dd_scale: float = 1.0
    disagreement_scale: float = 1.0
    est_port_vol: float = 0.0
    extra: dict = field(default_factory=dict)


def _project_leverage(w: np.ndarray, max_leverage: float = 1.0) -> np.ndarray:
    """Σ|w| <= max_leverage に射影"""
    gross = np.abs(w).sum()
    if gross > max_leverage:
        return w * (max_leverage / gross)
    return w


class PortfolioPostProcessor:
    """方策の生ウェイトを執行可能ウェイトへ変換する後処理器"""

    def __init__(self, config: Optional[PostProcessConfig] = None, risk_overlay=None):
        """
        Args:
            risk_overlay: 省略時（既定）は④⑤⑥（ボラ目標/DDデリスク/不一致縮小）を
                このクラス内にインライン実装した従来ロジックで処理する。
                mars_lite.trading.risk_overlay.RiskOverlay 実装（RuleRiskOverlay/
                RLRiskOverlay）を渡すと④⑤⑥をそちらに委譲する（opt-in、
                docs/ARCHITECTURE.md §2.8参照）。
        """
        self.cfg = config or PostProcessConfig()
        self.risk_overlay = risk_overlay

    def process(
        self,
        raw_weights: np.ndarray,
        prev_weights: np.ndarray,
        recent_returns: Optional[np.ndarray] = None,
        drawdown: float = 0.0,
        disagreement: float = 0.0,
    ) -> tuple:
        """
        Args:
            raw_weights: 方策の生ウェイト（レバレッジ1射影済みを想定）
            prev_weights: 現在の保有ウェイト
            recent_returns: (lookback, n_symbols) 直近の銘柄別単純リターン
                            （ボラターゲティング用。Noneでスキップ）
            drawdown: 現在のドローダウン率 [0,1]
            disagreement: アンサンブル不一致度 [0,1]（0=一致）

        Returns:
            (executed_weights, PostProcessInfo)
        """
        cfg = self.cfg
        raw = np.asarray(raw_weights, dtype=np.float64)
        prev = np.asarray(prev_weights, dtype=np.float64)
        info = PostProcessInfo(raw_gross=float(np.abs(raw).sum()))

        # ① EMA平滑（生ウェイトへ部分追従）
        w = cfg.ema_alpha * raw + (1.0 - cfg.ema_alpha) * prev

        # ② 銘柄集中上限
        w = np.clip(w, -cfg.max_weight, cfg.max_weight)

        # ③ レバレッジ1射影
        w = _project_leverage(w, 1.0)

        # ④⑤⑥ リスクオーバーレイ（ボラ目標・DDデリスク・不一致縮小でグロスを調整）
        if self.risk_overlay is not None:
            w, overlay_info = self.risk_overlay.scale(w, drawdown, disagreement, recent_returns)
            info.vol_scale = overlay_info.get("vol_scale", 1.0)
            info.dd_scale = overlay_info.get("dd_scale", 1.0)
            info.disagreement_scale = overlay_info.get("disagreement_scale", 1.0)
            info.est_port_vol = overlay_info.get("est_port_vol", 0.0)
        else:
            # ④ ボラターゲティング
            if cfg.target_vol is not None and recent_returns is not None and len(recent_returns) >= 5:
                port_ret = recent_returns @ w
                est_vol = float(np.std(port_ret) * np.sqrt(cfg.bars_per_year))
                info.est_port_vol = est_vol
                if est_vol > cfg.target_vol and est_vol > 1e-9:
                    info.vol_scale = cfg.target_vol / est_vol
                    w = w * info.vol_scale

            # ⑤ ドローダウン応答（DDが進むほどグロスを線形縮小、下限あり）
            if drawdown > cfg.dd_derisk_start:
                over = (drawdown - cfg.dd_derisk_start) / max(1.0 - cfg.dd_derisk_start, 1e-9)
                info.dd_scale = float(max(cfg.dd_derisk_floor, 1.0 - over))
                w = w * info.dd_scale

            # ⑥ アンサンブル不一致縮小（意見が割れるほどグロスを落とす）
            if disagreement > 0:
                info.disagreement_scale = float(
                    max(0.0, 1.0 - cfg.disagreement_penalty * disagreement)
                )
                w = w * info.disagreement_scale

        # ⑦ no-tradeバンド（微小な変更は据え置いてコストを節約）
        delta = w - prev
        delta[np.abs(delta) < cfg.no_trade_band] = 0.0
        executed = prev + delta

        # ⑧ ベータ中立化（オプトイン）: 実行ウェイトから市場方向＝等ウェイト平均
        # 成分を除去し Σw=0（ドル中立）を保証する。暗号資産のように全銘柄が
        # BTCベータで共線なユニバースでは、ネットロングは実質レバBTCベット。
        # これを外すと相対アルファだけが残る。方向性ベータを捨てるため上昇相場
        # では不利になりうる → 既定off。バンド後の最終段に置くことで、微小レグの
        # 据え置きで中立性が崩れないよう出力での中立を保証する。
        if cfg.beta_neutral and len(executed) > 1:
            executed = executed - executed.mean()
            info.extra["beta_neutralized"] = True

        info.processed_gross = float(np.abs(executed).sum())
        return executed, info


def make_legacy_processor(min_trade_delta: float = 0.02) -> PortfolioPostProcessor:
    """従来挙動（射影＋no-tradeバンドのみ）と等価な後処理器"""
    return PortfolioPostProcessor(PostProcessConfig(
        ema_alpha=1.0, max_weight=1.0, no_trade_band=min_trade_delta,
        target_vol=None, dd_derisk_start=1.0, disagreement_penalty=0.0,
    ))


def make_default_processor(
    target_vol: Optional[float] = 0.5,
    ema_alpha: float = 0.5,
    no_trade_band: float = 0.04,
    beta_neutral: bool = False,
) -> PortfolioPostProcessor:
    """
    ポートフォリオ運用の推奨後処理器

    P0ベンチマークで生方策比リターン2倍・Sharpe1.6倍・回転71%減・DD半減を確認。
    平滑化とno-tradeバンドが過剰取引によるコスト漏れを抑えるのが主効果。

    ema_alpha/no_trade_bandはホライズンスキャンで選んだ予測ホライズンに
    合わせて呼び出し側でスケールできる（低頻度アルファほど平滑を強め、
    no-tradeバンドを広げてコストと信号の周波数を整合させる）。
    """
    return PortfolioPostProcessor(PostProcessConfig(
        ema_alpha=ema_alpha, max_weight=0.4, no_trade_band=no_trade_band,
        target_vol=target_vol, vol_lookback=48,
        dd_derisk_start=0.10, dd_derisk_floor=0.3,
        disagreement_penalty=1.0, beta_neutral=beta_neutral,
    ))
