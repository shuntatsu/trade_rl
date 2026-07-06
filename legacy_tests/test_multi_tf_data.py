"""
多時間軸データローダーとデータ分割のテスト
"""

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from mars_lite.data.data_split import (
    TemporalSplitter,
    get_split_info,
    split_temporal,
    split_temporal_multi_tf,
)
from mars_lite.data.multi_timeframe_loader import MultiTimeframeLoader


def create_sample_ohlcv(
    n_bars: int = 1000, start_date: str = "2024-01-01"
) -> pd.DataFrame:
    """テスト用OHLCVデータを生成"""
    np.random.seed(42)

    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
    open_ = low + (high - low) * np.random.rand(n_bars)
    volume = np.random.exponential(1000, n_bars)
    timestamps = pd.date_range(start_date, periods=n_bars, freq="1min")

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


class TestTemporalSplit:
    """時系列分割のテスト"""

    def test_split_ratios(self):
        """分割比率が正しい"""
        df = create_sample_ohlcv(1000)
        train, val, test = split_temporal(df, 0.7, 0.15, 0.15)

        assert len(train) == 700
        assert len(val) == 150
        assert len(test) == 150

    def test_temporal_order_preserved(self):
        """時系列順序が維持される（データリーク防止）"""
        df = create_sample_ohlcv(1000)
        train, val, test = split_temporal(df)

        # 訓練の最後 < 検証の最初
        assert train["timestamp"].max() < val["timestamp"].min()
        # 検証の最後 < テストの最初
        assert val["timestamp"].max() < test["timestamp"].min()

    def test_split_info(self):
        """分割情報が正しく取得される"""
        df = create_sample_ohlcv(1000)
        train, val, test = split_temporal(df)
        info = get_split_info(train, val, test)

        assert "train" in info
        assert "val" in info
        assert "test" in info
        assert info["total_bars"] == 1000
        assert abs(info["train"]["ratio"] - 0.7) < 0.01

    def test_temporal_splitter_class(self):
        """TemporalSplitterクラスが正常動作"""
        splitter = TemporalSplitter(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
        df = create_sample_ohlcv(1000)
        train, val, test = splitter.split(df)

        assert len(train) == 800
        assert len(val) == 100
        assert len(test) == 100


class TestMultiTFSplit:
    """多時間軸データ分割のテスト"""

    def test_multi_tf_split(self):
        """複数時間軸データが同時に分割される"""
        data_1m = create_sample_ohlcv(1000, "2024-01-01 00:00:00")
        data_15m = create_sample_ohlcv(70, "2024-01-01 00:00:00")  # 約1日分

        data_dict = {"1m": data_1m, "15m": data_15m}
        train, val, test = split_temporal_multi_tf(data_dict)

        assert "1m" in train
        assert "15m" in train
        assert len(train["1m"]) > 0
        assert len(train["15m"]) > 0

    def test_multi_tf_no_data_leak(self):
        """多時間軸分割でデータリークがない"""
        data_1m = create_sample_ohlcv(1000, "2024-01-01 00:00:00")
        data_15m = create_sample_ohlcv(70, "2024-01-01 00:00:00")

        data_dict = {"1m": data_1m, "15m": data_15m}
        train, val, test = split_temporal_multi_tf(data_dict)

        # 1m時間軸でリークチェック
        if len(train["1m"]) > 0 and len(val["1m"]) > 0:
            assert train["1m"]["timestamp"].max() < val["1m"]["timestamp"].min()


class TestMultiTimeframeLoader:
    """多時間軸ローダーのテスト（ファイル操作含む）"""

    @pytest.fixture
    def temp_data_dir(self):
        """テスト用一時ディレクトリとデータを作成"""
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)

            # テストデータ作成・保存
            for tf, n_bars in [("1m", 1440), ("15m", 96), ("1h", 24)]:
                df = create_sample_ohlcv(n_bars, "2024-01-01")
                df.to_csv(tmpdir / f"BTCUSDT_{tf}_30d.csv", index=False)

            yield tmpdir

    def test_load_all(self, temp_data_dir):
        """全時間軸データが読み込まれる"""
        loader = MultiTimeframeLoader(
            data_dir=temp_data_dir,
            timeframes=["1m", "15m", "1h"],
            symbol="BTCUSDT",
            days=30,
            preprocess=True,
        )

        data = loader.load_all()

        assert "1m" in data
        assert "15m" in data
        assert "1h" in data

    def test_get_base_and_higher_tf(self, temp_data_dir):
        """ベースと上位時間軸が正しく分離される"""
        loader = MultiTimeframeLoader(
            data_dir=temp_data_dir,
            timeframes=["1m", "15m", "1h"],
            symbol="BTCUSDT",
            days=30,
        )
        loader.load_all()

        base = loader.get_base_timeframe()
        higher = loader.get_higher_timeframes()

        assert len(base) == 1440  # 1m
        assert "15m" in higher
        assert "1h" in higher
        assert "1m" not in higher


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
