"""
データ処理モジュールのテスト
"""

import numpy as np
import pandas as pd
import pytest
from mars_lite.data.preprocessing import preprocess_ohlcv, validate_preprocessed_data
from mars_lite.data.spread import calc_corwin_schultz
from mars_lite.data.volume_profile import attach_expected_volume, build_volume_profile

from mars_lite.data.volatility import calc_garman_klass, calc_parkinson


def create_sample_ohlcv(n_bars: int = 100) -> pd.DataFrame:
    """テスト用OHLCVデータを生成"""
    np.random.seed(42)

    # ランダムウォークで価格生成
    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))

    # OHLC生成
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
    open_ = low + (high - low) * np.random.rand(n_bars)

    # 出来高生成
    volume = np.random.exponential(1000, n_bars)

    # タイムスタンプ生成
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min")

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


class TestVolatility:
    """ボラティリティ推定のテスト"""

    def test_garman_klass_positive(self):
        """GKボラティリティは非負"""
        df = create_sample_ohlcv()
        vol = calc_garman_klass(df)

        assert len(vol) == len(df)
        assert (vol >= 0).all()

    def test_parkinson_positive(self):
        """Parkinsonボラティリティは非負"""
        df = create_sample_ohlcv()
        vol = calc_parkinson(df)

        assert len(vol) == len(df)
        assert (vol >= 0).all()

    def test_volatility_order(self):
        """高ボラ期間では高い値を返す"""
        df = create_sample_ohlcv()

        # 高ボラ設定
        df.loc[50:60, "high"] *= 1.1
        df.loc[50:60, "low"] *= 0.9

        vol_gk = calc_garman_klass(df)

        # 高ボラ区間が高い
        assert vol_gk[50:60].mean() > vol_gk[:50].mean()


class TestSpread:
    """スプレッド推定のテスト"""

    def test_corwin_schultz_positive(self):
        """CSスプレッドは非負"""
        df = create_sample_ohlcv()
        spread = calc_corwin_schultz(df, period=2)

        # 最初の数行はNaN
        spread_valid = spread.dropna()
        assert (spread_valid >= 0).all()

    def test_spread_sanity(self):
        """スプレッドが妥当な範囲内"""
        df = create_sample_ohlcv()
        spread = calc_corwin_schultz(df, period=2)

        spread_valid = spread.dropna()
        # 通常、スプレッドは10%未満
        assert spread_valid.max() < 0.1


class TestVolumeProfile:
    """出来高プロファイルのテスト"""

    def test_build_profile(self):
        """プロファイルが正しく構築される"""
        df = create_sample_ohlcv(n_bars=1440)  # 1日分
        profile = build_volume_profile(df)

        # 全時刻スロットが埋まる
        assert len(profile) == 1440

        # 値が正
        assert all(v > 0 for v in profile.values())

    def test_attach_expected_volume(self):
        """v_expected列が正しく追加される"""
        df = create_sample_ohlcv(n_bars=1440)
        profile = build_volume_profile(df)

        df_with_expected = attach_expected_volume(df, profile)

        assert "v_expected" in df_with_expected.columns
        assert not df_with_expected["v_expected"].isna().any()


class TestPreprocessing:
    """前処理パイプラインのテスト"""

    def test_preprocess_adds_all_features(self):
        """全特徴量が追加される"""
        df = create_sample_ohlcv(n_bars=1440)
        result = preprocess_ohlcv(df)

        expected_cols = [
            "vol_gk",
            "vol_park",
            "vol_gk_smooth",
            "spread_cs",
            "spread_cs_smooth",
            "v_expected",
            "log_return",
        ]

        for col in expected_cols:
            assert col in result.columns, f"Missing: {col}"

    def test_validate_preprocessed_data(self):
        """検証が正常に通る"""
        df = create_sample_ohlcv(n_bars=1440)
        result = preprocess_ohlcv(df)

        assert validate_preprocessed_data(result) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
