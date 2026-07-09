"""
リスクベースのクロスセクショナル配分（Phase2: サイジング強化）

背景: money_manager.build_money_manager は combined_teacher の生の予測方向・
大きさをそのまま銘柄間配分に使っており、銘柄間の相関構造（例: BTC/ETHが
強く共動する）を一切考慮していなかった。相関の高いクラスタに配分が集中し、
分散効果を発揮できていない可能性がある。

このモジュールは skfolio の Hierarchical Risk Parity（HRP、階層クラスタリング
ベース、凸最適化ソルバー不要で高速）を使い、直近リターンの共分散構造から
「リスク予算」を求め、combined_teacher の生シグナルにクロスセクショナルな
乗数として重ねる:

    final_weight_i = raw_signal_i * (hrp_weight_i * n_symbols)

hrp_weight は正でsum=1（リスクパリティ配分）なので、乗数の平均は1.0。
相関が高く分散効果の低い銘柄群は事後的に縮小され、独立性の高い銘柄群は
拡大される。素朴な逆ボラ（_vol_targeted、ポートフォリオ全体のグロスのみ調整）
とは直交する改善軸で、置き換えではなく併用できる。

因果性: 時刻tのHRP重みは fs.close[:t] のみから推定した直近lookbackバーの
リターンで適合する（未来を見ない）。

skfolio が使えない/フィットに失敗した場合は中立（全銘柄乗数1.0=無変換）に
フォールバックする。
"""

from typing import Optional

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import WeightFn


def hrp_weights(returns: np.ndarray) -> Optional[np.ndarray]:
    """(L, n_sym) のリターン行列から Hierarchical Risk Parity 重み (n_sym,) を返す。

    正でsum=1。フィット不能（skfolio未導入・特異な共分散等）なら None を返す
    （呼び出し側は等ウェイトへフォールバックすること）。
    """
    n_sym = returns.shape[1]
    if returns.shape[0] < max(20, n_sym + 5):
        return None
    try:
        import pandas as pd
        from skfolio.optimization import HierarchicalRiskParity

        model = HierarchicalRiskParity()
        model.fit(pd.DataFrame(returns))
        w = np.asarray(model.weights_, dtype=np.float64)
        if w.shape != (n_sym,) or not np.all(np.isfinite(w)) or w.sum() <= 0:
            return None
        return w / w.sum()
    except Exception:
        return None


def risk_parity_scaled(
    teacher_fn: WeightFn,
    lookback: int = 96,
    min_lookback: int = 60,
) -> WeightFn:
    """teacher の生シグナルへ、直近リターンから推定したHRPリスク予算を
    クロスセクショナル乗数として重ねるラッパー。

    HRPが適合できない（履歴不足・フィット失敗）場合は teacher の出力を
    そのまま返す（中立フォールバック、乗数1.0と同義）。
    """

    def fn(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        w = teacher_fn(fs, t, prev)
        gross = float(np.abs(w).sum())
        if gross < 1e-9:
            return w
        start = max(1, t - lookback)
        if t - start < min_lookback:
            return w
        rets = fs.close[start : t + 1] / fs.close[start - 1 : t] - 1.0  # (L, n_sym)
        hrp_w = hrp_weights(rets)
        if hrp_w is None:
            return w
        n = len(w)
        multiplier = hrp_w * n  # 平均1.0のクロスセクショナル乗数
        scaled = w * multiplier
        sg = float(np.abs(scaled).sum())
        if sg < 1e-9:
            return w
        # 元のグロス水準は維持し、銘柄間の配分だけをHRPで組み替える
        # （グロス自体の調整は _vol_targeted の責務であり重複させない）
        return scaled / sg * gross

    return fn
