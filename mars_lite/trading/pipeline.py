"""
方策の生ウェイトから執行ウェイトを導く決定パイプライン

env.step（学習時）と /api/signal/latest（運用時）が全く同じコードパスを
通るようにするための共有実装。以前はこの計算ロジックが env.step 内に
インラインで書かれ、serve側は別のインライン実装（かつ壊れていた）を
持っていた。train/serve一致は「同じ後処理オブジェクトを使う」だけでは
不十分で、直近リターンの切り出し方・HTFゲートの適用有無まで同一の
コードパスを通す必要がある。
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np

from mars_lite.trading.post_processor import _project_leverage, make_legacy_processor


@dataclass
class PortfolioState:
    """パイプライン呼び出し時点のポートフォリオ状態"""

    weights: np.ndarray
    portfolio_value: float = 1.0
    peak_value: float = 1.0
    disagreement: float = 0.0

    @property
    def drawdown(self) -> float:
        return 1.0 - self.portfolio_value / max(self.peak_value, 1e-9)


@dataclass
class MarketView:
    """パイプラインが必要とする市場データの切り出し（特徴量工学はここに閉じ込める）"""

    recent_returns: Optional[np.ndarray] = None  # (lookback, n_symbols) 単純リターン
    htf_trend: Optional[np.ndarray] = None  # (n_symbols,) HTFゲート用トレンド値

    @classmethod
    def from_feature_set(
        cls, fs, t: int, vol_lookback: int = 0, htf_idx: Optional[int] = None
    ) -> "MarketView":
        """FeatureSetの時点tから、ボラターゲティング用の直近リターンと
        （設定されていれば）HTFゲート用のトレンド値を切り出す。

        env.step の元実装と同一の窓定義（start = max(0, t - lookback)）。
        """
        recent_returns = None
        if vol_lookback > 0:
            start = max(0, t - vol_lookback)
            if t > start:
                recent_returns = (
                    np.diff(fs.close[start : t + 1], axis=0) / fs.close[start:t, :]
                )
        htf_trend = None
        if htf_idx is not None:
            htf_trend = fs.features[t][:, htf_idx]
        return cls(recent_returns=recent_returns, htf_trend=htf_trend)


class DecisionPipeline:
    """射影 → 後処理（EMA/集中上限/ボラ目標/DDデリスク/不一致縮小/no-tradeバンド）
    → HTFゲート、を一貫して適用するパイプライン。env.step と /api/signal/latest
    の両方がこれを呼ぶ（一致は構造で保証される）。
    """

    def __init__(
        self,
        post_processor=None,
        min_trade_delta: float = 0.04,
        htf_threshold: float = 0.3,
        htf_neutral_scale: float = 0.5,
        max_leverage: float = 1.0,
    ):
        # post_processor省略時は「射影＋no-tradeバンドのみ」の後処理器
        # （make_legacy_processor）を使う。以前はここに同じロジックを
        # インラインで複製していた（no_trade_band と min_trade_delta という
        # 2つの設定ノブが同じ概念を表す二重化の一因だった）。
        self.post_processor = post_processor or make_legacy_processor(min_trade_delta)
        self.min_trade_delta = min_trade_delta
        self.htf_threshold = htf_threshold
        self.htf_neutral_scale = htf_neutral_scale
        self.max_leverage = max_leverage

    def project(self, raw_action: np.ndarray) -> np.ndarray:
        return _project_leverage(
            np.asarray(raw_action, dtype=np.float64).flatten(), self.max_leverage
        )

    def target_weights(
        self, proj: np.ndarray, state: PortfolioState, market: MarketView
    ):
        """proj（レバレッジ射影済みの提案ウェイト）から執行ウェイトを導く。

        Returns:
            (target_weights, pp_info)
        """
        prev = np.asarray(state.weights, dtype=np.float64)
        proj = np.asarray(proj, dtype=np.float64)

        target, pp_info = self.post_processor.process(
            proj,
            prev,
            recent_returns=market.recent_returns,
            drawdown=state.drawdown,
            disagreement=state.disagreement,
        )

        if market.htf_trend is not None:
            target = self.apply_htf_gate(target, market.htf_trend)

        return target, pp_info

    def apply_htf_gate(self, w: np.ndarray, htf_trend: np.ndarray) -> np.ndarray:
        """階層MTFゲート: 上位足トレンドと逆方向のポジションを禁止し、
        トレンド無し(neutral)では縮小する。グロスは増加させない。
        """
        gated = np.asarray(w, dtype=np.float64).copy()
        for i in range(len(gated)):
            h = float(htf_trend[i])
            if h > self.htf_threshold:
                if gated[i] < 0:
                    gated[i] = 0.0
            elif h < -self.htf_threshold:
                if gated[i] > 0:
                    gated[i] = 0.0
            else:
                gated[i] = gated[i] * self.htf_neutral_scale
        return gated
