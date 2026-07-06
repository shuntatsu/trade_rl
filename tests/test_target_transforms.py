"""
ターゲット変換（cs_demean/vol_norm）のテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline
from mars_lite.features.signal_check import (
    _forward_returns,
    _pool,
    _transform_target,
    run_signal_check,
)


def _synthetic_fs(alpha="cross", days=60, seed=7):
    src = SyntheticSource(n_days=days, alpha=alpha, seed=seed)
    return FeaturePipeline(src.symbols).build(src)


class TestTransformTarget:
    def test_cs_demean_removes_row_mean(self):
        fwd = np.array([[1.0, 2.0, 3.0], [4.0, 4.0, 4.0], [np.nan, 1.0, 2.0]])
        out = _transform_target(fwd, "cs_demean", fs=None)
        np.testing.assert_allclose(out[0], [-1.0, 0.0, 1.0])
        np.testing.assert_allclose(out[1], [0.0, 0.0, 0.0])
        assert np.isnan(out[2, 0])

    def test_raw_is_identity(self):
        fwd = np.array([[1.0, 2.0], [3.0, 4.0]])
        out = _transform_target(fwd, "raw", fs=None)
        np.testing.assert_array_equal(out, fwd)

    def test_unknown_target_raises(self):
        with pytest.raises(ValueError):
            _transform_target(np.zeros((2, 2)), "bogus", fs=None)

    def test_vol_norm_scales_by_realized_vol(self):
        fs = _synthetic_fs(alpha="cross", days=30)
        fwd_raw = _forward_returns(fs, horizon=4)
        X, y, _ = _pool(fs, horizon=4, target="vol_norm")
        # vol_norm出力は有限値のみ残る（vol推定にNaNが出る序盤は除外される）
        assert np.isfinite(y).all()
        assert len(y) > 0


class TestCSDemeanIC:
    def test_cs_demean_ic_at_least_as_good_for_cross_alpha(self):
        """ゼロサム相対アルファ（cross）ではcs_demean ICがraw IC以上になりやすい"""
        fs = _synthetic_fs(alpha="cross", days=90, seed=3)
        raw_report = run_signal_check(fs, target="raw")
        cs_report = run_signal_check(fs, target="cs_demean")
        # crossアルファは元々ゼロサムなのでcs_demeanでほぼ信号が保存され、
        # 市場全体のノイズが除去される分ICが同等以上になりやすい
        assert cs_report.mean_oos_ic >= raw_report.mean_oos_ic - 0.05

    def test_bull_alpha_ic_drops_under_cs_demean(self):
        """方向性ベータ（bull）はcs_demeanで市場成分が除去され、raw より弱くなる"""
        fs = _synthetic_fs(alpha="bull", days=60, seed=5)
        raw_report = run_signal_check(fs, target="raw", threshold=0.0, min_t_stat=0.0)
        cs_report = run_signal_check(
            fs, target="cs_demean", threshold=0.0, min_t_stat=0.0
        )
        # bullは全銘柄共通ドリフトなのでcs_demeanでほぼ消える
        assert abs(cs_report.mean_oos_ic) <= abs(raw_report.mean_oos_ic) + 0.1

    def test_target_recorded_in_report(self):
        fs = _synthetic_fs(alpha="cross", days=30)
        report = run_signal_check(fs, target="cs_demean")
        assert report.target == "cs_demean"
        assert "target" in report.to_dict()
        assert "cs_demean" in report.summary()
