"""評価指標モジュール（Deflated Sharpe Ratio）のテスト"""

import numpy as np

from mars_lite.utils.metrics import _norm_cdf, _norm_ppf, deflated_sharpe_ratio


class TestNormApprox:
    def test_ppf_cdf_roundtrip(self):
        """自前実装の逆正規CDF/CDFが往復一致する（scipy不使用の近似精度確認）"""
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            x = _norm_ppf(p)
            assert abs(_norm_cdf(x) - p) < 1e-6


class TestDeflatedSharpeRatio:
    def test_more_trials_lowers_dsr(self):
        """試行回数(n_trials)が増えるほど、同じリターン系列でもDSRは下がる
        （選択バイアス補正が効いている証拠）"""
        rng = np.random.default_rng(0)
        rets = rng.normal(0.001, 0.01, 2000)

        few = deflated_sharpe_ratio(
            rets, trial_sharpes=[1.5], annualization_factor=24 * 365
        )
        many = deflated_sharpe_ratio(
            rets,
            trial_sharpes=list(rng.normal(0, 1.5, 200)),
            annualization_factor=24 * 365,
        )
        assert many["dsr"] < few["dsr"]
        assert many["sr0_annualized"] > few["sr0_annualized"]

    def test_single_trial_has_no_selection_penalty(self):
        """試行が1回だけ(n_trials<=1)なら試行数補正は無効(sr0=0)"""
        rng = np.random.default_rng(1)
        rets = rng.normal(0.001, 0.01, 500)
        rep = deflated_sharpe_ratio(rets, trial_sharpes=[2.0])
        assert rep["n_trials"] == 1
        assert rep["sr0_annualized"] == 0.0

    def test_short_series_returns_empty(self):
        """観測数が少なすぎる場合は判定不能として中立値を返す"""
        rep = deflated_sharpe_ratio(np.array([0.001, 0.002]), trial_sharpes=[1.0])
        assert rep["dsr"] == 0.0

    def test_zero_variance_returns_returns_empty(self):
        """リターンが定数（分散ゼロ）の場合はゼロ除算せず中立値を返す"""
        rep = deflated_sharpe_ratio(np.full(100, 0.001), trial_sharpes=[1.0, 2.0, 3.0])
        assert rep["dsr"] == 0.0

    def test_dsr_bounded_in_unit_interval(self):
        rng = np.random.default_rng(2)
        rets = rng.normal(0.002, 0.008, 1000)
        rep = deflated_sharpe_ratio(rets, trial_sharpes=list(rng.normal(0, 1, 20)))
        assert 0.0 <= rep["dsr"] <= 1.0
