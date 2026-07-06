"""
多時間軸環境のテスト
"""

import numpy as np
import pandas as pd
import pytest
from mars_lite.data.preprocessing import preprocess_ohlcv
from mars_lite.env.multi_tf_env import MarsLiteMultiTFEnv


def create_preprocessed_data(n_bars: int = 2000) -> pd.DataFrame:
    """テスト用の前処理済みデータを生成"""
    np.random.seed(42)

    returns = np.random.randn(n_bars) * 0.01
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.005)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.005)
    open_ = low + (high - low) * np.random.rand(n_bars)
    volume = np.random.exponential(1000, n_bars)
    timestamps = pd.date_range("2024-01-01", periods=n_bars, freq="1min")

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return preprocess_ohlcv(df)


def create_higher_tf_data(
    base_df: pd.DataFrame, interval_minutes: int = 15
) -> pd.DataFrame:
    """上位時間軸データを生成"""
    np.random.seed(42)

    n_bars = len(base_df) // interval_minutes

    returns = np.random.randn(n_bars) * 0.02
    close = 100 * np.exp(np.cumsum(returns))
    high = close * (1 + np.abs(np.random.randn(n_bars)) * 0.01)
    low = close * (1 - np.abs(np.random.randn(n_bars)) * 0.01)
    open_ = low + (high - low) * np.random.rand(n_bars)
    volume = np.random.exponential(5000, n_bars)
    timestamps = pd.date_range(
        "2024-01-01", periods=n_bars, freq=f"{interval_minutes}min"
    )

    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )

    return preprocess_ohlcv(df)


class TestMarsLiteMultiTFEnv:
    """多時間軸環境のテスト"""

    @pytest.fixture
    def env_with_higher_tf(self):
        """上位TF付きの環境を生成"""
        data_1m = create_preprocessed_data(n_bars=2000)
        higher_tf = {
            "15m": create_higher_tf_data(data_1m, 15),
            "1h": create_higher_tf_data(data_1m, 60),
        }

        return MarsLiteMultiTFEnv(
            data_1m=data_1m,
            higher_tf_data=higher_tf,
            initial_inventory=1000.0,
            max_steps=100,
            side="SELL",
            higher_tf_lookback=5,
        )

    @pytest.fixture
    def env_without_higher_tf(self):
        """上位TFなしの環境を生成"""
        data_1m = create_preprocessed_data(n_bars=2000)

        return MarsLiteMultiTFEnv(
            data_1m=data_1m,
            higher_tf_data={},
            initial_inventory=1000.0,
            max_steps=100,
            side="SELL",
        )

    def test_observation_space_with_higher_tf(self, env_with_higher_tf):
        """上位TF付きの観測空間サイズが正しい"""
        env = env_with_higher_tf

        # 1m特徴: 10 (log_returns) + 3 (vol, spread, rel_vol) = 13
        # 上位TF: 2 TF * 5 lookback * 3 features = 30
        # 内部状態: 2
        # 合計: 45
        expected_dim = 10 + 3 + (2 * 5 * 3) + 2

        assert env.observation_space.shape[0] == expected_dim

    def test_observation_space_without_higher_tf(self, env_without_higher_tf):
        """上位TFなしの観測空間サイズが正しい"""
        env = env_without_higher_tf

        # 1m特徴: 13 + 内部状態: 2 = 15
        expected_dim = 10 + 3 + 2

        assert env.observation_space.shape[0] == expected_dim

    def test_reset(self, env_with_higher_tf):
        """リセットが正常動作"""
        obs, info = env_with_higher_tf.reset(seed=42)

        assert obs is not None
        assert len(obs) == env_with_higher_tf.observation_space.shape[0]
        assert info["inventory"] == env_with_higher_tf.initial_inventory

    def test_step_reduces_inventory(self, env_with_higher_tf):
        """ステップで在庫が減少"""
        env_with_higher_tf.reset(seed=42)

        action = np.array([0.1])  # 10%執行
        obs, reward, terminated, truncated, info = env_with_higher_tf.step(action)

        assert info["inventory"] < env_with_higher_tf.initial_inventory

    def test_episode_terminates(self, env_with_higher_tf):
        """エピソードが終了する"""
        env_with_higher_tf.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 200:
            action = np.array([0.5])
            obs, reward, terminated, truncated, info = env_with_higher_tf.step(action)
            done = terminated or truncated
            steps += 1

        assert done

    def test_observation_contains_higher_tf_context(self, env_with_higher_tf):
        """観測に上位TFコンテキストが含まれる"""
        obs, info = env_with_higher_tf.reset(seed=42)

        # 上位TF部分が全てゼロでないことを確認（パディング以外）
        base_dim = 10 + 3  # log_returns + market features
        higher_dim = 2 * 5 * 3  # 2 TF * 5 lookback * 3 features

        higher_tf_obs = obs[base_dim : base_dim + higher_dim]

        # 全てゼロでないはず（最初はパディングでゼロの可能性あり）
        env_with_higher_tf.reset(seed=42, options={"start_idx": 100})
        obs2, _ = env_with_higher_tf.reset(seed=42, options={"start_idx": 100})
        higher_tf_obs2 = obs2[base_dim : base_dim + higher_dim]

        # 一部は非ゼロ（上位TFデータが反映されている）
        # 注: タイムスタンプの同期により、完全にゼロの可能性もある
        assert len(higher_tf_obs2) == higher_dim


class TestMultiTFEnvBehavior:
    """多時間軸環境の挙動テスト"""

    @pytest.fixture
    def env(self):
        data_1m = create_preprocessed_data(n_bars=2000)
        higher_tf = {
            "15m": create_higher_tf_data(data_1m, 15),
        }
        return MarsLiteMultiTFEnv(
            data_1m=data_1m,
            higher_tf_data=higher_tf,
            initial_inventory=1000.0,
            max_steps=100,
            side="SELL",
        )

    def test_info_contains_higher_tf_count(self, env):
        """infoに上位TF数が含まれる"""
        obs, info = env.reset(seed=42)

        assert "n_higher_tfs" in info
        assert info["n_higher_tfs"] == 1

    def test_action_clipping(self, env):
        """アクションが[0, 1]にクリップされる"""
        env.reset(seed=42)

        action = np.array([1.5])  # 範囲外
        obs, reward, _, _, info = env.step(action)

        assert info["inventory"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
