"""
evaluate_blended_book（2スリーブ合成book）のテスト

RLの実行済みウェイトとtrend_followingベースラインの合成が正しく機能する
ことを、境界条件（trend_frac=0/1）とグロス射影で検証する。
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.eval.blended_book import evaluate_blended_book
from mars_lite.features.feature_pipeline import FeaturePipeline


def _fs(days=40, alpha="cross", seed=7):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class _FlatAgent:
    """常にゼロウェイトを提案するダミーエージェント"""

    def predict(self, obs, deterministic=True):
        return np.zeros(7, dtype=np.float32), None


def test_trend_frac_zero_is_pure_rl_weights():
    """trend_frac=0ならブレンド後の実行ウェイトはRL単体と一致する"""
    fs = _fs()
    agent = _FlatAgent()
    res = evaluate_blended_book(agent, fs, trend_frac=0.0)
    # フラットエージェント + trend_frac=0 なら常にゼロウェイト → 取引なし
    assert res["turnover_total"] == pytest.approx(0.0, abs=1e-9)
    assert res["total_return"] == pytest.approx(0.0, abs=1e-9)


def test_trend_frac_one_ignores_rl_signal():
    """trend_frac=1ならRLのウェイトは無視され、trendシグナルのみが反映される"""
    fs = _fs()
    agent = _FlatAgent()
    res_flat_rl = evaluate_blended_book(agent, fs, trend_frac=1.0)

    class _NoisyAgent:
        def predict(self, obs, deterministic=True):
            rng = np.random.default_rng(0)
            return rng.uniform(-1, 1, size=7).astype(np.float32), None

    res_noisy_rl = evaluate_blended_book(_NoisyAgent(), fs, trend_frac=1.0)
    # RL側の入力が違っても trend_frac=1 なら結果は同一（RLのウェイトが
    # 完全に上書きされている証拠）
    assert res_flat_rl["total_return"] == pytest.approx(
        res_noisy_rl["total_return"], abs=1e-9
    )


def test_gross_leverage_capped_after_blend():
    """合成後のグロスが1を超えたら射影される（レバレッジ上限の不変条件）"""
    fs = _fs()

    class _MaxLongAgent:
        def predict(self, obs, deterministic=True):
            return np.ones(7, dtype=np.float32), None

    res = evaluate_blended_book(_MaxLongAgent(), fs, trend_frac=0.5)
    # 極端な入力でも爆発的な損失/turnoverにならない（射影が効いている簡易確認）
    assert abs(res["total_return"]) < 5.0
    assert res["turnover_total"] < fs.n_bars * 2  # 妥当な範囲


def test_returns_expected_keys():
    fs = _fs()
    res = evaluate_blended_book(_FlatAgent(), fs, trend_frac=0.3)
    for key in (
        "name",
        "total_return",
        "sharpe",
        "max_drawdown",
        "turnover_total",
        "n_bars",
    ):
        assert key in res
    assert res["name"] == "blended_trend0.30"
