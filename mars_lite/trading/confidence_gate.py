"""
相対アルファ成分の自己参照的な信頼度ゲート（トレンド希釈問題の是正）

背景（実測に基づく設計判断）: money_manager.combined_teacher は相対アルファ
（ridge/gbm）と方向性ベータ（trend）を単純に加算しグロス射影するため、
相対アルファがある期間で負けている（実データ検証: -10.65%）と、その期間中
ずっと trend_following 単体（同期間 +26.34%）を希釈し続けて敗北する
（実測: 合成後 +14.36%、trend単体の約半分）。

このモジュールは相対アルファ成分の**直近の実現損益**（因果的、未来を見ない）
を自己参照して、その時点でアルファが「効いているか」を判定し、効いていない
ときはブレンド比率を0（純trend）に、効いているときはブレンド比率を上げる
動的ゲートを提供する。

重要な注意（過学習リスク）: lookback / alpha_scale はハイパーパラメータであり、
評価対象のholdoutに対して直接グリッドサーチすると簡単に過学習する（実測:
holdout直接探索で+40%超という非現実的な結果が出たが、train/val/holdoutを
正しく3分割しvalだけでハイパラ選定すると+19%程度に落ち着いた）。
**必ずtrain区間で適合したモデルをval区間でチューニングし、test/holdout
区間では選定済みの固定値で1回だけ評価すること。**

因果性: 相対アルファ成分（ridge_teacher）は時刻tの特徴量 fs.features[t]
のみに依存し、prev（直前ウェイト）に一切依存しないため、評価区間全体の
アルファウェイト・実現リターンを1回だけベクトル化して事前計算できる
（O(n)、ループ内でtごとに再計算しない）。
"""

from typing import Callable

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.learning.baselines import WeightFn


def confidence_gated_blend(
    alpha_fn: WeightFn,
    trend_fn: WeightFn,
    lookback: int = 100,
    min_lookback: int = 50,
    alpha_scale: float = 0.02,
) -> WeightFn:
    """相対アルファ成分の直近トレーリング実現リターンに応じてtrend成分との
    ブレンド比率を動的に決める。

    conf = clip(trailing_alpha_return / alpha_scale, 0, 1)
    final = (1 - conf) * trend_w + conf * alpha_w

    - トレーリングリターンが0以下 → conf=0（純trend、希釈なし）
    - トレーリングリターンが alpha_scale 以上 → conf=1（アルファ全開）

    alpha_fn は prev に依存しない前提（ridge_teacher/combined_teacher の
    相対アルファ成分がこれに該当）。異なる fs オブジェクトが渡された場合は
    その fs 用に再計算する（同一評価区間内では1回だけ計算される）。
    """
    cache: dict[int, tuple] = {}

    def _precompute(fs: FeatureSet):
        n_bars, n_sym = fs.n_bars, fs.n_symbols
        alpha_w_all = np.zeros((n_bars, n_sym))
        for tau in range(n_bars):
            alpha_w_all[tau] = alpha_fn(fs, tau, None)
        rets = fs.close[1:] / fs.close[:-1] - 1.0
        alpha_ret = np.zeros(n_bars)
        alpha_ret[:-1] = np.einsum("ij,ij->i", alpha_w_all[:-1], rets)
        cum = np.concatenate([[0.0], np.cumsum(alpha_ret)])
        cache[id(fs)] = (alpha_w_all, cum)
        return alpha_w_all, cum

    def fn(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        trend_w = trend_fn(fs, t, prev)
        alpha_w_all, cum = cache.get(id(fs)) or _precompute(fs)

        start = max(0, t - lookback)
        if t - start < min_lookback:
            return trend_w
        trailing = float(cum[t] - cum[start])
        conf = float(np.clip(trailing / alpha_scale, 0.0, 1.0))
        if conf <= 1e-9:
            return trend_w
        blended = (1 - conf) * trend_w + conf * alpha_w_all[t]
        gross = float(np.abs(blended).sum())
        return blended / gross if gross > 1.0 else blended

    return fn
