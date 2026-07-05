"""
レジーム特化アンサンブル（項目3）

単一の汎用方策は「平均的な相場」に最適化されるため、強気トレンド・弱気
トレンド・レンジのどれかで妥協が生じやすい。ここでは相場レジームごとに
専門家（specialist）方策を学習し、推論時に現在のレジームへルーティングする。

レジームは観測に含まれる `btc_trend`（BTCの24本トレーリングリターンz-score、
look-ahead無し）で判定する。学習時のラベルと推論時のルーティングが同じ
シグナルを使うため、train/serve一致が保たれる。

- regime_labels(fs): 各バーを bull/bear/range に分類（学習ラベル用）
- regime_start_pools(fs, horizon): レジーム別のエピソード開始位置プール
- RegimeEnsemble: specialist群 + generalistフォールバックのルーティング方策
  （agent.predict互換 → evaluate/推論APIにそのまま差し込める）
"""

from typing import Dict, List, Optional, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet, GLOBAL_FEATURES

REGIMES = ("bull", "bear", "range")
_DEFAULT_THRESHOLD = 0.5


def _btc_trend_index() -> int:
    """生グローバル特徴ブロック内での btc_trend の位置"""
    return list(GLOBAL_FEATURES).index("btc_trend")


def classify_trend(trend_z: float, threshold: float = _DEFAULT_THRESHOLD) -> str:
    """btc_trend の z-score をレジームへ分類"""
    if trend_z > threshold:
        return "bull"
    if trend_z < -threshold:
        return "bear"
    return "range"


def regime_labels(fs: FeatureSet, threshold: float = _DEFAULT_THRESHOLD) -> np.ndarray:
    """各バーのレジームラベル配列（dtype=object の文字列）"""
    idx = _btc_trend_index()
    trend = fs.global_features[:, idx]
    return np.array([classify_trend(float(t), threshold) for t in trend], dtype=object)


def regime_start_pools(
    fs: FeatureSet,
    horizon: int,
    threshold: float = _DEFAULT_THRESHOLD,
    min_fraction: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    レジーム別のエピソード開始位置プールを作る。

    開始位置 s のエピソード窓 [s, s+horizon) 内で、そのレジームが
    min_fraction 以上を占める s を専門家の訓練プールとする。どの
    レジームも十分な数が集まらない場合は空配列（呼び出し側で全体に
    フォールバック）。
    """
    labels = regime_labels(fs, threshold)
    max_start = max(0, fs.n_bars - horizon - 2)
    pools: Dict[str, List[int]] = {r: [] for r in REGIMES}
    for s in range(max_start + 1):
        window = labels[s:s + horizon]
        if len(window) == 0:
            continue
        counts = {r: int((window == r).sum()) for r in REGIMES}
        dominant = max(counts, key=counts.get)
        if counts[dominant] / len(window) >= min_fraction:
            pools[dominant].append(s)
    return {r: np.array(v, dtype=np.int64) for r, v in pools.items()}


class RegimeEnsemble:
    """
    レジーム別 specialist へルーティングする方策（agent.predict互換）

    Args:
        specialists: regime名 -> 学習済みエージェント
        generalist: どのレジームにも specialist が無い時のフォールバック
        obs_layout: PortfolioTradingEnv.obs_layout（グローバルブロック位置の特定に使用）
        n_raw_globals: 生グローバル特徴数（feature_set.global_features.shape[1]）
        threshold: レジーム判定のz-score閾値
    """

    def __init__(
        self,
        specialists: Dict[str, object],
        generalist: Optional[object],
        obs_layout: Dict[str, int],
        n_raw_globals: int,
        threshold: float = _DEFAULT_THRESHOLD,
    ):
        if not specialists and generalist is None:
            raise ValueError("specialists と generalist の両方が空です")
        self.specialists = specialists
        self.generalist = generalist
        self.threshold = threshold
        self.device = getattr(
            next(iter(specialists.values())) if specialists else generalist,
            "device", "cpu",
        )

        # obs 内での btc_trend の絶対位置を求める。
        # レイアウト: [per_symbol...][raw_globals...][port_globals(3)]
        n_symbols = obs_layout["n_symbols"]
        n_per_sym = obs_layout["n_per_symbol"]
        global_start = n_symbols * n_per_sym
        self._trend_pos = global_start + _btc_trend_index()
        self._n_raw_globals = n_raw_globals
        # 選択履歴（デバッグ・可視化用）
        self.route_counts = {r: 0 for r in REGIMES}

    def _regime_from_obs(self, obs: np.ndarray) -> str:
        flat = np.asarray(obs).flatten()
        trend_z = float(flat[self._trend_pos]) if self._trend_pos < flat.size else 0.0
        return classify_trend(trend_z, self.threshold)

    def _agent_for(self, regime: str) -> object:
        agent = self.specialists.get(regime)
        if agent is None:
            agent = self.generalist
        if agent is None:
            # 最も近いレジームの specialist（bull<->range<->bear）
            order = {"bull": ["range", "bear"], "bear": ["range", "bull"],
                     "range": ["bull", "bear"]}
            for alt in order.get(regime, []):
                if alt in self.specialists:
                    return self.specialists[alt]
            return next(iter(self.specialists.values()))
        return agent

    def predict(self, obs: np.ndarray, deterministic: bool = True
                ) -> Tuple[np.ndarray, None]:
        regime = self._regime_from_obs(obs)
        self.route_counts[regime] = self.route_counts.get(regime, 0) + 1
        agent = self._agent_for(regime)
        action, _ = agent.predict(obs, deterministic=deterministic)
        return np.asarray(action).flatten(), None
