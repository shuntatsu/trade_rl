"""
因果的な適応スリーブ配分（follow-the-winner メタアロケータ）

背景（実データで確定した状況）: このユニバース/期間には「常に勝つ単一戦略」は
存在しない。代わりに逆の局面で稼ぐ2つの補完的エッジが揃った:
  - crowding（建玉クラウディング）: 市場中立アルファ。多様/レンジ局面で勝つ
    （dev 6fold cost2x で +3.89%, Sharpe 1.86, TF 6/6勝）
  - trend_following: 方向性ベータ。強トレンド局面で勝つ（holdout +29%）
static な固定比率合成は片方が足を引っ張るため悪化した（dev 50/50 = -7.33%）。

このモジュールは手組みレジームラベル（前セッションで汎化しないと実証済み）
ではなく、**各スリーブの直近の実現実績だけ**から配分を決める適応メタ
アロケータを提供する。「最近効いているスリーブに寄せる」follow-the-winner。

因果性の担保: 各スリーブは deterministic（fs が与えられれば weights も
実現リターンも一意）。評価区間全体の各スリーブのバー実現リターンを1回だけ
事前計算し、時刻tの配分は [t-lookback, t) の**過去のみ**の実績で決める。
未来を一切見ない（tests/test_regime_allocator.py の因果性テストで担保）。

注意（重要な限界）: walk-forward の各foldはアウトオブサンプルだが、我々は
このデータセット全体を広範にマイニング済みで、どのサブ区間も我々の分析に
汚染されている。この適応アロケータの walk-forward 結果は「示唆的証拠」で
あって「潔白な最終検定」ではない。真にクリーンな検証は新しい期間の
フォワード（ペーパー）テストのみ。
"""

from typing import Callable, List, Tuple

import numpy as np

from mars_lite.features.feature_pipeline import FeatureSet

WeightFn = Callable[[FeatureSet, int, np.ndarray], np.ndarray]


def _precompute_sleeve_bar_returns(fs: FeatureSet, sleeve: WeightFn) -> np.ndarray:
    """スリーブを standalone で走らせた時のバー実現リターン (n_bars,) を返す。

    時刻sのリターンは weights(s-1)·(close[s]/close[s-1]-1) - funding。
    weights(s-1) は s-1 までのデータのみに依存するため全体が因果的。
    （執行コストは含めない簡易版: 配分の"どちらが効いているか"の判定用途
    なので相対比較には十分。最終評価は simulate_strategy が正しくコストを課金）。
    """
    n_bars = fs.n_bars
    bar_ret = np.zeros(n_bars)
    prev = np.zeros(fs.n_symbols)
    for s in range(n_bars - 1):
        w = sleeve(fs, s, prev)
        r_vec = fs.close[s + 1] / fs.close[s] - 1.0
        funding = float(np.sum(w * fs.funding_rate[s + 1]))
        bar_ret[s + 1] = float(np.dot(w, r_vec)) - funding
        prev = w
    return bar_ret


def make_adaptive_allocator(
    sleeves: List[Tuple[str, WeightFn]],
    lookback: int = 336,
    min_lookback: int = 168,
    rebalance_every: int = 24,
    temperature: float = 1.0,
    floor: float = 0.0,
) -> WeightFn:
    """各スリーブの直近 lookback バーの実現リターンから配分を決める適応アロケータ。

    配分 = softmax(trailing_return / temperature) を各スリーブに与え、
    Σ(alloc_i · sleeve_weights_i) を最終ウェイトとする（グロス>1は射影）。

    Args:
        sleeves: [(名前, WeightFn), ...]
        lookback: 配分判定に使う直近実績の窓（バー数）
        min_lookback: この本数未満の履歴では等ウェイト配分
        rebalance_every: 配分と各スリーブの再計算間隔
        temperature: softmaxの温度。小さいほど勝者総取り、大きいほど均等
        floor: 各スリーブの最低配分（0で無効。分散を強制したい場合に使う）
    """
    cache: dict = {}

    def _precompute(fs: FeatureSet):
        rets = [_precompute_sleeve_bar_returns(fs, fn) for _, fn in sleeves]
        cum = [np.concatenate([[0.0], np.cumsum(r)]) for r in rets]
        cache[id(fs)] = cum
        return cum

    def fn(fs: FeatureSet, t: int, prev: np.ndarray) -> np.ndarray:
        if t % rebalance_every != 0 and np.any(prev):
            return prev

        cum = cache.get(id(fs)) or _precompute(fs)
        k = len(sleeves)
        start = max(0, t - lookback)
        if t - start < min_lookback:
            alloc = np.full(k, 1.0 / k)
        else:
            trailing = np.array([c[t] - c[start] for c in cum])
            z = trailing / max(temperature, 1e-9)
            z = z - z.max()  # 数値安定化
            alloc = np.exp(z)
            alloc = alloc / alloc.sum()
            if floor > 0:
                alloc = np.clip(alloc, floor, None)
                alloc = alloc / alloc.sum()

        w = np.zeros(fs.n_symbols)
        for a, (_, sleeve) in zip(alloc, sleeves):
            w = w + a * sleeve(fs, t, prev)
        gross = float(np.abs(w).sum())
        return w / gross if gross > 1.0 else w

    return fn
