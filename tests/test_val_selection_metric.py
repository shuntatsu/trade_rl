"""
val_selection.sortino_score（PBT/val選択の指標）のテスト

旧指標 total_return − 0.5·max_dd は弱αノイズ下で「flat（無取引）」を常に最良と
判定して保守崩壊させた。是正後のSortino指標が
  - flat（全ゼロ）→ 0
  - 正のリスク調整エッジ → 正（flatより高い＝取引を選べる）
  - 損失方策 → 負（棄却）
となることを検証する。
"""

import numpy as np

from mars_lite.learning.val_selection import sortino_score


def test_flat_scores_zero():
    assert sortino_score(np.zeros(500)) == 0.0
    assert sortino_score([]) == 0.0


def test_positive_edge_beats_flat():
    rng = np.random.default_rng(0)
    rets = rng.normal(0.0005, 0.01, size=1000)  # 正のドリフト
    assert sortino_score(rets) > 0.0  # flat(0)より高い＝取引を選べる


def test_losing_strategy_is_negative():
    rng = np.random.default_rng(1)
    rets = rng.normal(-0.0005, 0.01, size=1000)  # 負のドリフト
    assert sortino_score(rets) < 0.0


def test_no_collapse_to_flat_on_weak_positive_signal():
    """弱いが正のエッジ（実現平均が正）なら flat(0) を上回る。旧指標の
    保守崩壊（弱いエッジをmaxDD項で潰しflatを選ぶ）の是正の核心。"""
    rng = np.random.default_rng(2)
    weak = 0.0005 + rng.normal(0.0, 0.005, size=4000)  # 実現平均が確実に正のSNR
    assert weak.mean() > 0  # 前提: 実現エッジは正
    assert sortino_score(weak) > sortino_score(np.zeros(4000))


def test_higher_downside_lowers_score():
    """同じ平均でも下方偏差が大きいほどスコアは下がる（リスク調整）。"""
    base = np.full(1000, 0.001)
    calm = base.copy()
    volatile = base.copy()
    volatile[::10] = -0.02  # 時々大きく負ける
    assert sortino_score(volatile) < sortino_score(calm)
