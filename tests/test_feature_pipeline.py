"""
特徴量パイプラインとシグナル検証のテスト
"""

import numpy as np
import pytest

from mars_lite.data.sources import SyntheticSource, CsvSource
from mars_lite.features.feature_pipeline import (
    FeaturePipeline, TF_BLOCK_FEATURES, BASE_FEATURES, CS_FEATURES,
)
from mars_lite.features.signal_check import run_signal_check


@pytest.fixture(scope="module")
def source():
    return SyntheticSource(n_days=20, alpha="cross", seed=11)


@pytest.fixture(scope="module")
def feature_set(source):
    return FeaturePipeline(source.symbols).build(source)


class TestFeaturePipeline:

    def test_shapes(self, feature_set, source):
        fs = feature_set
        n_tf = 4  # 15m, 1h, 4h, 1d
        expected_features = n_tf * len(TF_BLOCK_FEATURES) + len(BASE_FEATURES) + len(CS_FEATURES)
        assert fs.features.shape == (fs.n_bars, len(source.symbols), expected_features)
        assert fs.global_features.shape[0] == fs.n_bars
        assert fs.close.shape == (fs.n_bars, fs.n_symbols)

    def test_no_nan(self, feature_set):
        assert not np.isnan(feature_set.features).any()
        assert not np.isnan(feature_set.global_features).any()
        assert np.isfinite(feature_set.close).all()

    def test_features_bounded(self, feature_set):
        """z-score系特徴はクリップ範囲内"""
        assert feature_set.features.max() <= 5.0 + 1e-6
        assert feature_set.features.min() >= -5.0 - 1e-6

    def test_no_lookahead_in_features(self, source):
        """
        Look-ahead検査: 期間を短く切って構築した特徴量と、
        全期間で構築した特徴量の共通部分が（末尾の未確定部を除き）一致する
        → 未来のデータが過去の特徴に影響していない
        """
        pipe = FeaturePipeline(source.symbols)
        fs_full = pipe.build(source)
        # 前半だけで構築
        end_ts = str(fs_full.timestamps[fs_full.n_bars // 2])
        fs_half = pipe.build(source, end=end_ts)

        n = fs_half.n_bars - 1
        # ret_rank等は共通バーで同一のはず
        np.testing.assert_allclose(
            fs_half.features[:n], fs_full.features[:n], atol=1e-5,
            err_msg="未来データが過去の特徴量に漏れています（look-ahead bias）",
        )

    def test_funding_alignment(self, feature_set):
        """fundingは8時間毎（1h足で全バーの約1/8）に発生"""
        nonzero_ratio = (feature_set.funding_rate != 0).mean()
        assert 0.05 < nonzero_ratio < 0.25

    def test_slice(self, feature_set):
        sub = feature_set.slice(10, 50)
        assert sub.n_bars == 40
        np.testing.assert_array_equal(sub.close, feature_set.close[10:50])


class TestSignalCheck:

    def test_alpha_detected(self, feature_set):
        """アルファ注入データではICゲートがPASSする"""
        report = run_signal_check(feature_set)
        assert report.mean_oos_ic > 0.05
        assert report.passed

    def test_noise_rejected(self):
        """純ノイズデータではアルファ有データよりICが大幅に低い

        注: 短期間の標本では偶発的な相関が乗るため、絶対閾値ではなく
        陽性対照との相対比較で判定する（60日でも±0.05程度は揺れる）。
        """
        src_noise = SyntheticSource(n_days=40, alpha="none", seed=13)
        fs_noise = FeaturePipeline(src_noise.symbols).build(src_noise)
        ic_noise = run_signal_check(fs_noise).mean_oos_ic

        src_alpha = SyntheticSource(n_days=40, alpha="cross", seed=13)
        fs_alpha = FeaturePipeline(src_alpha.symbols).build(src_alpha)
        ic_alpha = run_signal_check(fs_alpha).mean_oos_ic

        assert ic_alpha > ic_noise + 0.1
        assert ic_noise < 0.1


class TestCsvSource:

    def test_roundtrip(self, tmp_path):
        """generate_sample_data出力 → CsvSource → FeaturePipeline が通る"""
        import subprocess, sys
        subprocess.run(
            [sys.executable, "scripts/generate_sample_data.py",
             "--days", "10", "--alpha", "cross",
             "--symbols", "BTCUSDT", "ETHUSDT",
             "--output", str(tmp_path), "--start-date", "2024-01-01"],
            check=True, capture_output=True,
        )
        src = CsvSource(tmp_path, ["BTCUSDT", "ETHUSDT"])
        klines = src.load_klines("BTCUSDT", "1m")
        assert len(klines) == 10 * 1440
        of = src.load_orderflow("BTCUSDT")
        assert "volume_imbalance" in of.columns
        funding = src.load_funding("BTCUSDT")
        assert len(funding) == 30  # 10日 × 3回/日

        fs = FeaturePipeline(["BTCUSDT", "ETHUSDT"]).build(src)
        assert fs.n_symbols == 2
        assert fs.n_bars > 200


class TestFeatureMask:

    def test_mask_zeroes_dropped_features(self, feature_set):
        import numpy as np
        mask = np.ones(feature_set.n_features, dtype=bool)
        mask[0] = False
        masked = feature_set.apply_mask(mask)
        assert (masked.features[:, :, 0] == 0).all()
        # 他の特徴は不変
        np.testing.assert_array_equal(masked.features[:, :, 1], feature_set.features[:, :, 1])
        # 形状・レイアウトは維持
        assert masked.n_features == feature_set.n_features

    def test_compute_mask_keeps_signal_features(self, feature_set):
        from mars_lite.features.signal_check import compute_feature_mask
        rep = compute_feature_mask(feature_set)
        # crossアルファではret_rank等の主要特徴が残る
        assert "ret_rank" in rep["kept"]
        assert len(rep["kept"]) >= 5
        assert rep["mask"].shape[0] == feature_set.n_features

    def test_mask_length_validation(self, feature_set):
        import numpy as np
        import pytest as _pt
        with _pt.raises(ValueError):
            feature_set.apply_mask(np.ones(3, dtype=bool))


class TestTrendGateAndTeachers:

    def test_trend_gate_detects_persistent_bull(self):
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.features.signal_check import run_trend_gate
        src = SyntheticSource(n_days=90, alpha="bull", alpha_strength=0.001, seed=1)
        fs = FeaturePipeline(src.symbols).build(src)
        g = run_trend_gate(fs)
        assert g["has_trend"]
        assert g["direction"] > 0                    # 上昇方向
        assert g["fold_sign_agreement"] == 1.0       # 全foldで同符号

    def test_trend_gate_rejects_noise_no_false_positive(self):
        """ランダムウォークの実現ドリフトを符号一致で弾く（偽陽性防止）"""
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.features.signal_check import run_trend_gate
        for seed in range(6):
            src = SyntheticSource(n_days=60, alpha="none", seed=seed)
            fs = FeaturePipeline(src.symbols).build(src)
            assert not run_trend_gate(fs)["has_trend"], f"false positive at seed {seed}"

    def test_ts_momentum_teacher_net_long_in_uptrend(self):
        import numpy as np
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.learning.bc_warmstart import ts_momentum_teacher
        src = SyntheticSource(n_days=40, alpha="bull", alpha_strength=0.001, seed=2)
        fs = FeaturePipeline(src.symbols).build(src)
        w = ts_momentum_teacher()(fs, fs.n_bars - 1, np.zeros(fs.n_symbols))
        assert w.sum() > 0                           # ネットロング

    def test_ts_momentum_not_zero_sum(self):
        """時系列モメンタムはネット方向性を持つ（クロスモメンタムはゼロサム）"""
        import numpy as np
        from mars_lite.learning.bc_warmstart import ts_momentum_teacher, soft_momentum_teacher
        from mars_lite.data.sources import SyntheticSource
        src = SyntheticSource(n_days=40, alpha="bull", alpha_strength=0.001, seed=3)
        fs = FeaturePipeline(src.symbols).build(src)
        ts_w = ts_momentum_teacher()(fs, fs.n_bars - 1, np.zeros(fs.n_symbols))
        cs_w = soft_momentum_teacher()(fs, fs.n_bars - 1, np.zeros(fs.n_symbols))
        assert abs(ts_w.sum()) > 0.05                # ネット方向性あり
        assert abs(cs_w.sum()) < 1e-9                # クロスはゼロサム


class TestDerivativeFeatures:

    def test_derivatives_flow_and_carry_signal(self):
        """OI/L/S比率/清算がパイプラインを通り、cross alphaで予測力を持つ"""
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.features.signal_check import run_signal_check
        src = SyntheticSource(n_days=90, alpha="cross", seed=0)
        fs = FeaturePipeline(src.symbols).build(src)
        for f in ["oi_z", "oi_change", "ls_ratio_z", "liq_z"]:
            assert f in fs.feature_names
        assert not np.isnan(fs.features).any()
        ic = run_signal_check(fs)
        # OIは正のIC、L/S比率は逆張り（負のIC）
        assert ic.per_feature_ic["oi_z"] > 0.05
        assert ic.per_feature_ic["ls_ratio_z"] < -0.05

    def test_missing_derivatives_zeroed(self):
        """デリバティブ無しソース（株式等）でも動く（ゼロ埋め）"""
        from mars_lite.data.sources import DataSource
        import pandas as pd

        class NoDerivSource(DataSource):
            def __init__(self):
                super().__init__(["A", "B"])
                rng = np.random.default_rng(0)
                n = 60 * 1440
                ts = pd.date_range("2024-01-01", periods=n, freq="1min")
                self._k = {}
                for s in ["A", "B"]:
                    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.0009, n)))
                    o = np.concatenate([[100], c[:-1]])
                    self._k[s] = pd.DataFrame({"timestamp": ts, "open": o,
                        "high": np.maximum(o, c) * 1.001, "low": np.minimum(o, c) * 0.999,
                        "close": c, "volume": rng.uniform(100, 1000, n)})
            def load_klines(self, symbol, timeframe="1m", start=None, end=None):
                df = self._k[symbol]
                if timeframe != "1m":
                    from mars_lite.data.data_utils import resample_ohlcv
                    df = resample_ohlcv(df, timeframe)
                return df

        fs = FeaturePipeline(["A", "B"]).build(NoDerivSource())
        oi_idx = fs.feature_names.index("oi_z")
        assert (fs.features[:, :, oi_idx] == 0).all()
        assert not np.isnan(fs.features).any()
