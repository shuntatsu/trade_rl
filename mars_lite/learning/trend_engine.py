"""
TrendEngine v2: 複数ホライズンTSMOM + 銘柄別ボラスケーリング + ポートフォリオ
ボラ目標を内蔵したトレンドフォロー戦略

背景: 実データ検証で最も稼いでいたのは既存の trend_following_strategy
（mars_lite.learning.baselines、lookback=48単一・tanh正規化・24本毎
リバランス）唯一そのものだった。相対アルファ（Ridge/GBM/信頼度ゲート等）は
どう組み合わせてもtrendを希釈するだけで上回れなかったため、「trendを
上乗せで超えようとする」のではなく「trendエンジン自体を定石で強化する」
方針に転換する。

強化点（すべて trend_following_strategy の数式・思想を踏襲した拡張）:
  1. 複数ホライズン合成: 単一lookback=48は特定の周期に過適合しやすい。
     複数ホライズン（短期〜長期）のtanh正規化モメンタムを等ウェイト平均し、
     一貫してトレンドが出ている銘柄・局面ほど強いシグナルになるようにする。
  2. 銘柄別ボラスケーリング: 素のtanhシグナルは銘柄間のボラ差を無視して
     いた。inverse_vol_strategy と同じ発想でシグナルをσで割り、低ボラ銘柄の
     相対配分を厚くする（等リスク寄与に近づける）。シグナル段階のグロスへ
     再正規化するため、全体のエクスポージャー水準は変えない。
  3. ポートフォリオボラ目標（down-only）: mars_lite.trading.post_processor
     の④ボラターゲティングと同一ロジック（実現ボラが目標を超えた時だけ
     縮小、超えなければ拡大しない）を移植。急変時のドローダウンを抑える。

回帰アンカー: lookbacks=(48,)・vol_scale_symbols=False・target_vol=None・
rebalance_every=24 に設定すると、trend_following_strategy と全時刻で
厳密に同一のウェイトを出す（tests/test_trend_engine.py で保証）。これは
「拡張が既存の実証済み挙動を包含している」ことの数学的な裏付け。
"""

from typing import Callable, Optional, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H

# mars_lite.learning.baselines.WeightFn と同一定義。baselines.py が
# BASELINES登録時にこのモジュールを遅延importするため、逆方向にbaselines.py
# へ依存すると import 順序次第で壊れる循環importになる（実測: 先に
# trend_engine を単独importすると失敗）。型の実体は同じなので独立定義で回避。
WeightFn = Callable[[FeatureSet, int, np.ndarray], np.ndarray]

# 定石値（実行前に固定・グリッド探索前にまず試す既定値）
DEFAULT_LOOKBACKS: Tuple[int, ...] = (24, 72, 168, 336)
DEFAULT_VOL_LOOKBACK = 168
DEFAULT_TARGET_VOL = 0.30

# 既定値が不合格の場合のみ使う事前登録グリッド（dev内valのみで選定すること。
# strategy_wf.run_holdout_once を通す前のtest/holdoutに直接当てて選ばない）。
TREND_V2_GRID = [
    {"lookbacks": DEFAULT_LOOKBACKS, "target_vol": tv} for tv in (0.20, 0.30, 0.40)
] + [{"lookbacks": (48, 168), "target_vol": tv} for tv in (0.20, 0.30, 0.40)]


def make_trend_engine_v2(
    lookbacks: Tuple[int, ...] = DEFAULT_LOOKBACKS,
    vol_lookback: int = DEFAULT_VOL_LOOKBACK,
    target_vol: Optional[float] = DEFAULT_TARGET_VOL,
    rebalance_every: int = 24,
    vol_scale_symbols: bool = True,
    bars_per_year: int = BARS_PER_YEAR_1H,
) -> WeightFn:
    """複数ホライズンTSMOM + 銘柄別ボラスケーリング + ポートフォリオボラ目標。

    Args:
        lookbacks: TSMOMシグナルを計算するホライズン（バー数）の集合。
            各ホライズンで trend_following_strategy と同一の
            tanh(mom/mean|mom|) 正規化を行い、等ウェイト平均する
        vol_lookback: 銘柄別ボラスケーリング・ポートフォリオボラ目標の
            両方で使う直近リターンの窓（バー数）
        target_vol: >0 ならポートフォリオの実現ボラがこれを超えた時だけ
            グロスを縮小する（down-only）。Noneで無効
        rebalance_every: 目標を再計算する間隔（バー数）
        vol_scale_symbols: 銘柄別ボラスケーリングを行うか
    """

    def teacher(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        if t % rebalance_every != 0 and np.any(prev):
            return prev

        signals = []
        for length in lookbacks:
            start = max(0, t - length)
            if t - start < 4:
                continue
            mom = np.log(fs.close[t] / fs.close[start])
            scale = np.abs(mom).mean() + 1e-9
            signals.append(np.tanh(mom / scale))
        if not signals:
            return np.zeros(fs.n_symbols)
        raw = np.mean(signals, axis=0)
        raw_gross = float(np.abs(raw).sum())
        if raw_gross < 1e-12:
            return np.zeros(fs.n_symbols)

        w = raw
        if vol_scale_symbols:
            start = max(0, t - vol_lookback)
            if t - start >= 8:
                log_rets = np.diff(np.log(fs.close[start : t + 1]), axis=0)
                sigma = np.clip(log_rets.std(axis=0), 1e-6, None)
                scaled = raw / sigma
                sg = float(np.abs(scaled).sum())
                if sg > 1e-12:
                    # シグナル段階のグロスへ再正規化（総エクスポージャーは
                    # 変えず、銘柄間の配分だけを組み替える）
                    w = scaled / sg * raw_gross

        gross = float(np.abs(w).sum())
        if gross > 1.0:
            w = w / gross

        if target_vol is not None:
            start = max(1, t - vol_lookback)
            if t - start >= 5:
                rets_window = fs.close[start : t + 1] / fs.close[start - 1 : t] - 1.0
                port_ret = rets_window @ w
                est_vol = float(np.std(port_ret) * np.sqrt(bars_per_year))
                if est_vol > target_vol and est_vol > 1e-9:
                    w = w * (target_vol / est_vol)

        return w

    return teacher
