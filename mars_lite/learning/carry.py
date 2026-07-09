"""
Fundingキャリー戦略（trendと真に分散するスリーブ）

背景: simulate_strategy は毎バー `funding = Σ(weights × funding_rate[t+1])`
を実現損益から差し引いている（無期限先物のfunding授受を検算済みで反映済み）。
つまり「高funding銘柄をショートし、低/負funding銘柄をロングする」キャリー
戦略は、新しいコストモデルの追加なしにそのまま既存の実行パスで動く。

このスリーブがtrendと分散するのは、fundingが方向性モメンタムとは独立な
需給シグナル（レバレッジロング過多→高funding、逆もまた然り）であるため。
strategy_wf の judge_carry が「trend_followingとのfoldリターン相関<0.3」を
要求するのはこの分散性を検証するため。

因果性: 時刻tの発注は fs.funding_rate[:t+1]（= t までに確定済みの授受実績）
のみに依存する。simulate_strategy が課金する funding_rate[t+1] 自体は
未来のfunding授受額だが、これは「ポジションを取った結果として発生する
コスト」であって発注判断の入力ではない（trend_following_strategy が
fs.close[t+1] の未来リターンを見ずに発注するのと同じ因果構造）。
"""

from typing import Callable

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet

WeightFn = Callable[[FeatureSet, int, np.ndarray], np.ndarray]

DEFAULT_LOOKBACK = 72
DEFAULT_REBALANCE_EVERY = 24
DEFAULT_GROSS = 0.5

# 既定値が不合格の場合のみ使う事前登録グリッド（lookbackのみ変える）。
# 撤退基準: cost2x黒字を満たさなければ廃棄し、これ以外の調整はしない。
CARRY_GRID = [{"lookback": lb} for lb in (24, 72, 168)]


def make_carry_strategy(
    lookback: int = DEFAULT_LOOKBACK,
    rebalance_every: int = DEFAULT_REBALANCE_EVERY,
    gross: float = DEFAULT_GROSS,
) -> WeightFn:
    """直近funding率のクロスセクショナル順位に基づくdollar-neutralキャリー戦略。

    高funding銘柄（ロングがショートに支払う）をショート、低/負funding銘柄
    （ショートがロングに支払う、またはロングが受け取る）をロングする。
    順位をデミーンして使うため理論上ネット≈0（対称性で厳密にゼロ）。

    Args:
        lookback: トレーリングfunding平均を測る窓（バー数）
        rebalance_every: 目標を再計算する間隔（バー数）
        gross: 目標グロス（Σ|w|）。dollar-neutralなので0.5等の抑えた値を想定
    """

    def teacher(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        if t % rebalance_every != 0 and np.any(prev):
            return prev

        start = max(0, t - lookback + 1)
        if t - start < max(4, lookback // 2):
            return np.zeros(fs.n_symbols)

        trailing_funding = fs.funding_rate[start : t + 1].mean(axis=0)
        n = fs.n_symbols
        rank = np.argsort(np.argsort(trailing_funding)).astype(np.float64)
        demeaned = rank - rank.mean()  # 対称 -> sum=0 (dollar-neutral厳密保証)
        raw = -demeaned  # 高funding(高rank)ほど負=ショート

        s = float(np.abs(raw).sum())
        if s < 1e-12:
            return np.zeros(n)
        return raw / s * gross

    return teacher
