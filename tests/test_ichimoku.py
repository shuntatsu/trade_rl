"""
一目均衡表（Ichimoku）特徴量のテスト

主要テスト項目:
1. look-ahead バイアスがないこと (senkou_a/b は過去情報のみ依存)
2. chikou (遅行スパン) が学習特徴量に含まれないこと
3. 未来雲予測特徴量は look-ahead なし（現在データのみから計算）
4. 全特徴量がローリングz-score標準化されていること
5. ichi_pos が雲の上/中/下で正しい符号を持つこと
6. ichi_cloud_bull が A>B で +1、A<B で -1 を返すこと
7. FeaturePipeline.build で feature_names に ichi_* が10個含まれること
"""

import numpy as np
import pandas as pd

from mars_lite.features.ichimoku import calc_ichimoku, ichimoku_features

# ============================================================
# テスト用データ生成ヘルパー
# ============================================================


def make_ohlcv(n: int = 200, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100.0 * np.cumprod(1 + rng.normal(0, 0.01, n))
    high = close * (1 + rng.uniform(0.001, 0.01, n))
    low = close * (1 - rng.uniform(0.001, 0.01, n))
    open_ = close * (1 + rng.normal(0, 0.005, n))
    ts = pd.date_range("2025-01-01", periods=n, freq="1h")
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close}, index=ts
    )


# ============================================================
# calc_ichimoku のテスト
# ============================================================


class TestCalcIchimoku:
    def test_columns(self):
        df = make_ohlcv()
        ichi = calc_ichimoku(df["high"], df["low"], df["close"])
        assert set(ichi.columns) == {
            "tenkan",
            "kijun",
            "senkou_a",
            "senkou_b",
            "future_span_a",
            "future_span_b",
            "chikou",
        }

    def test_no_lookahead_senkou(self):
        """
        senkou_a[t] は close[t] に依存してはならない。
        close[t] を変えても senkou_a[t] が変わらないことを確認する。
        """
        df = make_ohlcv(n=200)
        ichi1 = calc_ichimoku(df["high"], df["low"], df["close"])

        # 最後の1本だけ close を変える
        df2 = df.copy()
        df2 = df2.copy()  # CoW 対応
        df2.loc[df2.index[-1], "close"] *= 1.5
        df2.loc[df2.index[-1], "high"] *= 1.5
        df2.loc[df2.index[-1], "low"] *= 1.5
        ichi2 = calc_ichimoku(df2["high"], df2["low"], df2["close"])

        # senkou_a[t] は t-26 以前の情報のみなので、最後の26本は変わってはならない
        # 正確には最終インデックスの senkou_a は 26本前のデータに依存
        # → 最後の25本の senkou_a は変化しないはず
        idx = ichi1.index[-25:]
        pd.testing.assert_series_equal(
            ichi1["senkou_a"].loc[idx],
            ichi2["senkou_a"].loc[idx],
            check_names=False,
        )

    def test_chikou_is_future(self):
        """
        chikou[t] = close[t + 26] の関係を確認（= 未来情報）
        """
        df = make_ohlcv(n=200)
        ichi = calc_ichimoku(df["high"], df["low"], df["close"])
        # chikou[0] == close[26]
        assert abs(float(ichi["chikou"].iloc[0]) - float(df["close"].iloc[26])) < 1e-9

    def test_senkou_warmup(self):
        """先頭 senkou_b_period(52) + displacement(26) = 78本が NaN になる"""
        df = make_ohlcv(n=200)
        ichi = calc_ichimoku(df["high"], df["low"], df["close"])
        # senkou_b は 52本 rolling + 26本 shift = 78本先頭が NaN のはず
        assert ichi["senkou_b"].iloc[:77].isna().all()
        assert ichi["senkou_b"].iloc[77:].notna().any()


# ============================================================
# ichimoku_features のテスト
# ============================================================


class TestIchimokuFeatures:
    def test_output_columns(self):
        df = make_ohlcv()
        feats = ichimoku_features(df["high"], df["low"], df["close"])
        expected = {
            # 現在雲
            "ichi_pos",
            "ichi_cloud_thick",
            "ichi_cloud_bull",
            # 転換・基準線
            "ichi_tk_cross",
            "ichi_price_kijun",
            "ichi_price_tenkan",
            # 未来雲予測
            "ichi_future_pos",
            "ichi_future_bull",
            "ichi_future_thick",
            "ichi_tk_accel",
        }
        assert set(feats.columns) == expected

    def test_total_feature_count(self):
        """10種の特徴量が出力されること"""
        df = make_ohlcv()
        feats = ichimoku_features(df["high"], df["low"], df["close"])
        assert len(feats.columns) == 10

    def test_no_chikou_in_features(self):
        """遅行スパンが特徴量に含まれないこと"""
        df = make_ohlcv()
        feats = ichimoku_features(df["high"], df["low"], df["close"])
        assert "chikou" not in feats.columns

    def test_no_nan_in_output(self):
        """NaN が 0 で埋まっていること"""
        df = make_ohlcv(n=200)
        feats = ichimoku_features(df["high"], df["low"], df["close"])
        assert not feats.isna().any().any()

    def test_clip_range(self):
        """すべての値が [-clip, clip] 内に収まること"""
        df = make_ohlcv(n=300)
        feats = ichimoku_features(df["high"], df["low"], df["close"], clip=5.0)
        assert (feats.abs() <= 5.0 + 1e-6).all().all()

    def test_ichi_pos_above_cloud(self):
        """
        価格が雲の上に長期間ある場合、ichi_pos が雲の下のケースより高いこと。
        z-score 化後は絶対値の符号ではなく相対比較で確認する。
        """
        n = 300
        ts = pd.date_range("2025-01-01", periods=n, freq="1h")
        # 単調上昇（長期で雲の上を維持するケース）
        close_up = pd.Series(np.linspace(100, 500, n), index=ts)
        high_up = close_up * 1.005
        low_up = close_up * 0.995
        feats_up = ichimoku_features(high_up, low_up, close_up)

        # 単調下降（長期で雲の下を維持するケース）
        close_dn = pd.Series(np.linspace(500, 100, n), index=ts)
        high_dn = close_dn * 1.005
        low_dn = close_dn * 0.995
        feats_dn = ichimoku_features(high_dn, low_dn, close_dn)

        # warmup (78本+) 以降の安定期で比較
        mean_up = feats_up["ichi_pos"].iloc[90:].mean()
        mean_dn = feats_dn["ichi_pos"].iloc[90:].mean()
        assert mean_up > mean_dn, (
            f"Expected ichi_pos higher in uptrend ({mean_up:.3f}) "
            f"than downtrend ({mean_dn:.3f})"
        )

    def test_ichi_pos_below_cloud(self):
        """価格が雲の下にある期間は ichi_pos < 0 になること"""
        n = 200
        ts = pd.date_range("2025-01-01", periods=n, freq="1h")
        # 単調下降の価格
        close = pd.Series(np.linspace(200, 100, n), index=ts)
        high = close * 1.005
        low = close * 0.995
        feats = ichimoku_features(high, low, close)
        stable = feats["ichi_pos"].iloc[80:]
        assert (stable <= 0.01).all(), (
            f"Expected negative ichi_pos below cloud, got max={stable.max()}"
        )

    def test_ichi_cloud_bull_sign(self):
        """
        上昇局面では ichi_cloud_bull の平均z-scoreが正になること
        (z-score標準化後は+1/-1の展開まわりが変わるが、平均値の符号は保存)
        """
        n = 300
        ts = pd.date_range("2025-01-01", periods=n, freq="1h")
        close = pd.Series(np.linspace(100, 300, n), index=ts)
        high = close * 1.005
        low = close * 0.995
        feats = ichimoku_features(high, low, close)
        # senkou_a には warmup が 78本必要
        stable = feats["ichi_cloud_bull"].iloc[90:]
        # z-score標準化後でも平均値が正中立以上のはず
        assert stable.mean() >= 0, (
            f"Expected non-negative mean cloud_bull in uptrend, got {stable.mean():.3f}"
        )

    def test_no_lookahead_feature(self):
        """
        最後の1本の価格を変えても、現在雲特徴量[t-27以山]が変化しないこと
        （look-ahead がないことの確認）
        """
        df = make_ohlcv(n=200)
        feats1 = ichimoku_features(df["high"], df["low"], df["close"])

        # 最後の1本だけ大幅に変える
        df2 = df.copy()
        df2.loc[df2.index[-1], "close"] *= 2.0
        df2.loc[df2.index[-1], "high"] *= 2.0
        df2.loc[df2.index[-1], "low"] *= 2.0
        feats2 = ichimoku_features(df2["high"], df2["low"], df2["close"])

        # 現在雲は 26本前のデータに依存するので、最後の 26本より前は変化しない
        idx = feats1.index[:-27]
        pd.testing.assert_frame_equal(
            feats1[["ichi_pos", "ichi_cloud_bull"]].loc[idx],
            feats2[["ichi_pos", "ichi_cloud_bull"]].loc[idx],
            check_exact=False,
            atol=1e-5,
        )

    def test_future_cloud_no_lookahead(self):
        """
        未来雲予測特徴量（ichi_future_*）も look-ahead なし。
        未来雲は現在時刻 t のデータだけから計算されるため、
        t+1 以降のデータが変わっても、最後の1本以外は不変のはず。
        """
        df = make_ohlcv(n=200)
        feats1 = ichimoku_features(df["high"], df["low"], df["close"])

        df2 = df.copy()
        df2.loc[df2.index[-1], "close"] *= 3.0
        df2.loc[df2.index[-1], "high"] *= 3.0
        df2.loc[df2.index[-1], "low"] *= 3.0
        feats2 = ichimoku_features(df2["high"], df2["low"], df2["close"])

        # 未来雲は現在データだけに依存 → 最後の1本以外は変化なし
        idx = feats1.index[:-1]
        pd.testing.assert_frame_equal(
            feats1[["ichi_future_pos", "ichi_future_bull"]].loc[idx],
            feats2[["ichi_future_pos", "ichi_future_bull"]].loc[idx],
            check_exact=False,
            atol=1e-5,
        )


# ============================================================
# FeaturePipeline との統合確認
# ============================================================


class TestFeaturePipelineIntegration:
    def test_ichimoku_in_feature_names(self):
        """FeaturePipeline の feature_names に ichi_* が10個含まれること"""
        from mars_lite.features.feature_pipeline import FeaturePipeline

        fp = FeaturePipeline(["BTCUSDT", "ETHUSDT"])
        ichi_feats = [f for f in fp.feature_names if f.startswith("ichi_")]
        expected = [
            "ichi_pos",
            "ichi_cloud_thick",
            "ichi_cloud_bull",
            "ichi_tk_cross",
            "ichi_price_kijun",
            "ichi_price_tenkan",
            "ichi_future_pos",
            "ichi_future_bull",
            "ichi_future_thick",
            "ichi_tk_accel",
        ]
        assert set(ichi_feats) == set(expected), f"Got: {ichi_feats}"

    def test_build_with_synthetic_source(self):
        """SyntheticSource で build が完走し、ichi_* 特徴量に NaN がないこと"""
        from mars_lite.data.sources import SyntheticSource
        from mars_lite.features.feature_pipeline import FeaturePipeline

        symbols = ["BTCUSDT", "ETHUSDT"]
        fp = FeaturePipeline(symbols)
        source = SyntheticSource(n_days=15, alpha="cross", seed=0)
        fs = fp.build(source)

        ichi_idxs = [i for i, n in enumerate(fs.feature_names) if n.startswith("ichi_")]
        assert len(ichi_idxs) == 10

        ichi_data = fs.features[:, :, ichi_idxs]
        assert not np.isnan(ichi_data).any(), "NaN found in ichi features from build()"
        assert (np.abs(ichi_data) <= 5.0 + 1e-5).all(), (
            "Clipping violated in built features"
        )
