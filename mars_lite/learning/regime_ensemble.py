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

from mars_lite.features.feature_pipeline import GLOBAL_FEATURES, FeatureSet
from mars_lite.learning.regime_fsm import REGIMES_8, RegimeFSM

REGIMES = ("bull", "bear", "range")
_DEFAULT_THRESHOLD = 0.5

FALLBACK_ROUTES = {
    "extreme_bull": [
        "bull_high",
        "bull_low",
        "range_high",
        "range_low",
        "bear_high",
        "bear_low",
        "extreme_bear",
    ],
    "bull_high": [
        "bull_low",
        "extreme_bull",
        "range_high",
        "range_low",
        "bear_high",
        "bear_low",
        "extreme_bear",
    ],
    "bull_low": [
        "bull_high",
        "range_low",
        "range_high",
        "extreme_bull",
        "bear_low",
        "bear_high",
        "extreme_bear",
    ],
    "range_high": [
        "range_low",
        "bull_high",
        "bear_high",
        "bull_low",
        "bear_low",
        "extreme_bull",
        "extreme_bear",
    ],
    "range_low": [
        "range_high",
        "bull_low",
        "bear_low",
        "bull_high",
        "bear_high",
        "extreme_bull",
        "extreme_bear",
    ],
    "bear_high": [
        "bear_low",
        "extreme_bear",
        "range_high",
        "range_low",
        "bull_high",
        "bull_low",
        "extreme_bull",
    ],
    "bear_low": [
        "bear_high",
        "range_low",
        "range_high",
        "extreme_bear",
        "bull_low",
        "bull_high",
        "extreme_bull",
    ],
    "extreme_bear": [
        "bear_high",
        "bear_low",
        "range_high",
        "range_low",
        "bull_high",
        "bull_low",
        "extreme_bull",
    ],
}


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
    fsm: Optional[RegimeFSM] = None,
) -> Dict[str, np.ndarray]:
    """
    レジーム別のエピソード開始位置プールを作る。

    開始位置 s のエピソード窓 [s, s+horizon) 内で、そのレジームが
    min_fraction 以上を占める s を専門家の訓練プールとする。どの
    レジームも十分な数が集まらない場合は空配列（呼び出し側で全体に
    フォールバック）。
    """
    if fsm is not None:
        vol_idx = list(GLOBAL_FEATURES).index("btc_vol_regime")
        trend_idx = list(GLOBAL_FEATURES).index("btc_trend")
        vol_series = fs.global_features[:, vol_idx]
        trend_series = fs.global_features[:, trend_idx]
        labels = fsm.classify_series(trend_series, vol_series)
        regimes_to_use = REGIMES_8
    else:
        labels = regime_labels(fs, threshold)
        regimes_to_use = REGIMES

    max_start = max(0, fs.n_bars - horizon - 2)
    pools: Dict[str, List[int]] = {r: [] for r in regimes_to_use}
    for s in range(max_start + 1):
        window = labels[s : s + horizon]
        if len(window) == 0:
            continue
        counts = {r: int((window == r).sum()) for r in regimes_to_use}
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
        fsm: 8状態 Regime FSM インスタンス
    """

    def __init__(
        self,
        specialists: Dict[str, object],
        generalist: Optional[object],
        obs_layout: Dict[str, int],
        n_raw_globals: int,
        threshold: float = _DEFAULT_THRESHOLD,
        fsm: Optional[RegimeFSM] = None,
    ):
        if not specialists and generalist is None:
            raise ValueError("specialists と generalist の両方が空です")
        self.specialists = specialists
        self.generalist = generalist
        self.threshold = threshold
        self.fsm = fsm
        self.device = getattr(
            next(iter(specialists.values())) if specialists else generalist,
            "device",
            "cpu",
        )

        # obs 内での btc_trend の絶対位置を求める。
        # レイアウト: [per_symbol...][raw_globals...][port_globals(3)]
        n_symbols = obs_layout["n_symbols"]
        n_per_sym = obs_layout["n_per_symbol"]
        global_start = n_symbols * n_per_sym
        self._trend_pos = global_start + _btc_trend_index()
        self._vol_pos = global_start + list(GLOBAL_FEATURES).index("btc_vol_regime")
        self._n_raw_globals = n_raw_globals

        # 選択履歴（デバッグ・可視化用）
        regimes_keys = REGIMES_8 if self.fsm is not None else REGIMES
        self.route_counts = {r: 0 for r in regimes_keys}

    def _regime_from_obs(self, obs: np.ndarray) -> str:
        flat = np.asarray(obs).flatten()
        trend_z = float(flat[self._trend_pos]) if self._trend_pos < flat.size else 0.0

        if self.fsm is not None:
            vol_z = float(flat[self._vol_pos]) if self._vol_pos < flat.size else 0.0
            return self.fsm.update(trend_z, vol_z)
        else:
            return classify_trend(trend_z, self.threshold)

    def _agent_for(self, regime: str) -> object:
        agent = self.specialists.get(regime)
        if agent is not None:
            return agent

        if self.fsm is None:
            # 既存の3状態の挙動: まず generalist を確認し、なければ近接レジーム
            if self.generalist is not None:
                return self.generalist
            order = {
                "bull": ["range", "bear"],
                "bear": ["range", "bull"],
                "range": ["bull", "bear"],
            }.get(regime, [])
            for alt in order:
                if alt in self.specialists:
                    return self.specialists[alt]
            return next(iter(self.specialists.values()))
        else:
            # 新規の8状態の挙動: 近接レジームを優先し、なければ generalist
            order = FALLBACK_ROUTES.get(regime, [])
            for alt in order:
                if alt in self.specialists:
                    return self.specialists[alt]
            if self.generalist is not None:
                return self.generalist
            return next(iter(self.specialists.values()))

    def predict(
        self, obs: np.ndarray, deterministic: bool = True
    ) -> Tuple[np.ndarray, None]:
        regime = self._regime_from_obs(obs)
        self.route_counts[regime] = self.route_counts.get(regime, 0) + 1
        agent = self._agent_for(regime)
        action, _ = agent.predict(obs, deterministic=deterministic)
        return np.asarray(action).flatten(), None
