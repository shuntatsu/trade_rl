"""
金銭管理アロケータ（mars_lite.trading.money_manager）のテスト

要点:
  - 因果性: 適合は train_fs のみを使い、test区間の未来を覗かない
  - 分解の整合: trend成分のみ = 方向性、ridge成分のみ = 市場中立（ゼロサム）
  - サイジング: グロス上限1.0を超えない、ボラ目標が過剰レバレッジを抑える
  - アルファ有データで flat（何もしない）を上回る
"""

import numpy as np

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.trading.money_manager import (
    build_money_manager,
    evaluate_money_manager,
)


def _fs(days=90, alpha="cross", seed=5):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


def _split(fs, frac=0.6):
    purge = 24
    k = int(fs.n_bars * frac)
    return fs.slice(0, k), fs.slice(k + purge, fs.n_bars)


class TestCausality:
    def test_fit_ignores_test_future(self):
        """train_fsが同一なら、test_fsの未来を書き換えても構築される
        weight_fnの「学習済みパラメータ」は不変（=適合はtrainのみ依存）。"""
        fs = _fs()
        train_fs, test_fs = _split(fs)

        fn = build_money_manager(train_fs, horizon=4, use_ridge=True, use_trend=False)

        # 同じ時刻・同じ特徴を与えれば、testをどう壊しても出力は同一
        feats_t = test_fs.features[10].copy()
        w1 = fn(test_fs, 10, np.zeros(test_fs.n_symbols))

        test_broken = test_fs
        rng = np.random.default_rng(0)
        test_broken.features[20:] = rng.normal(size=test_broken.features[20:].shape)
        w2 = fn(test_broken, 10, np.zeros(test_broken.n_symbols))
        # 時刻10の特徴は壊していない（20以降のみ破壊）→ 出力一致
        np.testing.assert_allclose(test_broken.features[10], feats_t)
        np.testing.assert_allclose(w1, w2)


class TestDecomposition:
    def test_ridge_only_is_market_neutral(self):
        """ridge成分のみ = クロスセクショナル中心化でゼロサム（市場中立）"""
        fs = _fs()
        train_fs, test_fs = _split(fs)
        fn = build_money_manager(
            train_fs, use_ridge=True, use_trend=False, ridge_target="cs_demean"
        )
        for t in range(5, test_fs.n_bars, 37):
            w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
            assert abs(float(w.sum())) < 1e-9  # ネットエクスポージャゼロ

    def test_trend_only_can_be_directional(self):
        """trend成分のみ = ネット方向性を持ちうる（ゼロサムでない）"""
        fs = _fs(alpha="none")
        train_fs, test_fs = _split(fs)
        fn = build_money_manager(train_fs, use_ridge=False, use_trend=True)
        nets = [
            abs(float(fn(test_fs, t, np.zeros(test_fs.n_symbols)).sum()))
            for t in range(60, test_fs.n_bars, 20)
        ]
        assert max(nets) > 1e-6  # どこかで方向性が出る


class TestSizing:
    def test_gross_never_exceeds_one(self):
        fs = _fs()
        train_fs, test_fs = _split(fs)
        for tv in (0.0, 0.5, 2.0):
            fn = build_money_manager(train_fs, target_vol=tv)
            for t in range(5, test_fs.n_bars, 29):
                w = fn(test_fs, t, np.zeros(test_fs.n_symbols))
                assert float(np.abs(w).sum()) <= 1.0 + 1e-9

    def test_vol_target_scales_down_high_vol(self):
        """ボラ目標を極端に小さくするとグロスが縮む（=リスク抑制が効く）"""
        fs = _fs()
        train_fs, test_fs = _split(fs)
        fn_hi = build_money_manager(train_fs, target_vol=0.0)  # 無効=素のグロス
        fn_lo = build_money_manager(train_fs, target_vol=0.01)  # 年率1%目標=強く縮む
        g_hi = g_lo = 0.0
        for t in range(60, test_fs.n_bars, 17):
            g_hi += float(np.abs(fn_hi(test_fs, t, np.zeros(test_fs.n_symbols))).sum())
            g_lo += float(np.abs(fn_lo(test_fs, t, np.zeros(test_fs.n_symbols))).sum())
        assert g_lo < g_hi


class TestTurnoverControl:
    def test_rebalance_throttle_cuts_turnover(self):
        """rebalance_every を大きくすると回転が減る（コスト暴走の是正）。"""
        fs = _fs(alpha="cross", days=120, seed=11)
        train_fs, test_fs = _split(fs)
        res_churn = evaluate_money_manager(
            train_fs, test_fs, rebalance_every=1, no_trade_band=0.0
        )
        res_throttled = evaluate_money_manager(
            train_fs, test_fs, rebalance_every=24, no_trade_band=0.05
        )
        assert res_throttled.turnover_total < res_churn.turnover_total

    def test_holds_between_rebalances(self):
        """非リバランスバーでは前回ウェイトをそのまま保持する。"""
        fs = _fs()
        train_fs, test_fs = _split(fs)
        fn = build_money_manager(train_fs, rebalance_every=24, no_trade_band=0.0)
        prev = fn(test_fs, 24, np.zeros(test_fs.n_symbols))
        held = fn(test_fs, 25, prev)  # 25 % 24 != 0 → 保持
        np.testing.assert_allclose(held, prev)


class TestEvaluation:
    def test_beats_flat_on_alpha_data(self):
        """アルファ有データでは金銭管理アロケータはflat(0%)を上回る"""
        fs = _fs(alpha="cross", days=120, seed=7)
        train_fs, test_fs = _split(fs)
        res = evaluate_money_manager(
            train_fs, test_fs, horizon=4, use_ridge=True, use_trend=True
        )
        assert res.name == "money_manager"
        assert res.total_return > 0.0
        d = res.to_dict()
        assert set(["total_return", "sharpe", "max_drawdown"]).issubset(d.keys())

    def test_feature_mismatch_raises(self):
        fs = _fs()
        train_fs, test_fs = _split(fs)
        masked = test_fs.slice(0, test_fs.n_bars)
        # 特徴数を人為的に食い違わせる
        masked.feature_names = masked.feature_names[:-1]
        masked.features = masked.features[:, :, :-1]
        import pytest

        with pytest.raises(ValueError):
            evaluate_money_manager(train_fs, masked)
