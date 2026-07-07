"""
細分化レジーム分類（因果的・look-ahead無し）

既存の regime_ensemble.py は btc_trend の符号だけで bull/bear/range の3分類。
本モジュールはトレンド方向 × 局面(早期/成熟) × ボラティリティで最大6分類に
細分化する。目的は「どの局面に予測力(IC)が残っているか」を切り分けること
（レジーム分割はアルファを生まないが、条件付きで構造を露出させる）。

分類軸（すべて観測に既にある因果的グローバル特徴から算出）:
  - btc_trend      : BTC 24本トレーリングリターンz（方向・強度）
  - btc_vol_regime : BTC 24本ローリングボラz（高ボラ/低ボラ）
  - トレンド継続長 : 現在のトレンド状態が何本連続で続いたか（早期/成熟の判別。
                     過去のみ参照＝因果的）

レジーム:
  trend_up_early   : 上昇トレンド、直近 age_bars 本以内に開始（初動）
  trend_up_mature  : 上昇トレンド、成熟（継続 age_bars 本超）
  trend_down_early : 下降トレンド、初動
  trend_down_mature: 下降トレンド、成熟
  range_lowvol     : レンジ（|trend|<=閾値）かつ低ボラ
  range_highvol    : レンジかつ高ボラ
"""

from typing import List

import numpy as np

from mars_lite.features.feature_pipeline import GLOBAL_FEATURES, FeatureSet

FINE_REGIMES = (
    "trend_up_early",
    "trend_up_mature",
    "trend_down_early",
    "trend_down_mature",
    "range_lowvol",
    "range_highvol",
)


def _global_index(name: str) -> int:
    return list(GLOBAL_FEATURES).index(name)


def label_fine_regimes(
    fs: FeatureSet,
    trend_threshold: float = 0.5,
    vol_threshold: float = 0.0,
    age_bars: int = 24,
) -> np.ndarray:
    """
    各バーを6レジームへ分類する（因果的）。

    Args:
        fs: FeatureSet（global_featuresにbtc_trend/btc_vol_regimeを含む）
        trend_threshold: |btc_trend| がこの値超でトレンド、以下でレンジ
        vol_threshold: btc_vol_regime がこの値超で高ボラ
        age_bars: トレンド継続がこの本数以内なら「早期(early)」、超なら「成熟(mature)」

    Returns:
        (n_bars,) dtype=object の文字列ラベル配列
    """
    ti = _global_index("btc_trend")
    vi = _global_index("btc_vol_regime")
    trend = fs.global_features[:, ti].astype(np.float64)
    vol = fs.global_features[:, vi].astype(np.float64)

    # トレンド方向: +1(up) / -1(down) / 0(range)。因果的（そのバーの値のみ）
    direction = np.where(
        trend > trend_threshold, 1, np.where(trend < -trend_threshold, -1, 0)
    )

    # トレンド継続長: 同一方向が何本連続で続いたか（過去のみ＝因果的）
    age = np.zeros(len(direction), dtype=np.int64)
    for t in range(len(direction)):
        if t > 0 and direction[t] != 0 and direction[t] == direction[t - 1]:
            age[t] = age[t - 1] + 1
        else:
            age[t] = 0

    labels = np.empty(len(direction), dtype=object)
    for t in range(len(direction)):
        d = direction[t]
        if d == 0:
            labels[t] = "range_highvol" if vol[t] > vol_threshold else "range_lowvol"
        else:
            phase = "early" if age[t] <= age_bars else "mature"
            side = "up" if d > 0 else "down"
            labels[t] = f"trend_{side}_{phase}"
    return labels


def regime_distribution(labels: np.ndarray) -> dict:
    """レジーム別のバー数分布"""
    return {r: int((labels == r).sum()) for r in FINE_REGIMES}
