"""
環境モジュールのテスト
"""

import numpy as np
import pandas as pd
import pytest
from mars_lite.data.preprocessing import preprocess_ohlcv
from mars_lite.env.mars_lite_env import MarsLiteEnv
from mars_lite.env.matching_engine import match_order
from mars_lite.env.reward import calc_almgren_chriss_cost, calc_reward


def create_preprocessed_data(n_bars: int = 2000) -> pd.DataFrame:
    """テスト用の前処理済みデータを生成"""
    np.random.seed(42)

    # 価格生成
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


class TestMatchingEngine:
    """マッチングエンジンのテスト"""

    def test_sell_price_lower(self):
        """SELL時、執行価格は基準価格より低い"""
        p_exec, info = match_order(
            side="SELL",
            quantity=100,
            sigma=0.02,
            spread_cs=0.001,
            v_expected=1000,
            p_base=100.0,
            y_impact=0.5,
        )

        assert p_exec < 100.0
        assert info["total_cost_pct"] > 0

    def test_buy_price_higher(self):
        """BUY時、執行価格は基準価格より高い"""
        p_exec, info = match_order(
            side="BUY",
            quantity=100,
            sigma=0.02,
            spread_cs=0.001,
            v_expected=1000,
            p_base=100.0,
            y_impact=0.5,
        )

        assert p_exec > 100.0

    def test_zero_quantity_no_impact(self):
        """数量0ではインパクトなし"""
        p_exec, info = match_order(
            side="SELL",
            quantity=0,
            sigma=0.02,
            spread_cs=0.001,
            v_expected=1000,
            p_base=100.0,
        )

        assert p_exec == 100.0
        assert info["impact_pct"] == 0.0

    def test_higher_quantity_more_impact(self):
        """数量が多いほどインパクトが大きい"""
        _, info_low = match_order(
            side="SELL",
            quantity=10,
            sigma=0.02,
            spread_cs=0.001,
            v_expected=1000,
            p_base=100.0,
        )
        _, info_high = match_order(
            side="SELL",
            quantity=500,
            sigma=0.02,
            spread_cs=0.001,
            v_expected=1000,
            p_base=100.0,
        )

        assert info_high["impact_pct"] > info_low["impact_pct"]


class TestReward:
    """報酬関数のテスト"""

    def test_reward_negative_for_cost(self):
        """コストが発生すると報酬は負"""
        reward = calc_reward(
            quantity=100,
            p_base=100.0,
            p_exec=99.9,  # スリッページ発生
            sigma=0.02,
            remaining_inventory=500,
            lambda_risk=0.001,
            side="SELL",
        )

        assert reward < 0

    def test_higher_inventory_more_penalty(self):
        """在庫が多いほどリスクペナルティが大きい"""
        exec_cost, risk_low, _ = calc_almgren_chriss_cost(
            quantity=100,
            p_base=100.0,
            p_exec=99.9,
            sigma=0.02,
            remaining_inventory=100,
        )
        _, risk_high, _ = calc_almgren_chriss_cost(
            quantity=100,
            p_base=100.0,
            p_exec=99.9,
            sigma=0.02,
            remaining_inventory=500,
        )

        assert risk_high > risk_low


class TestMarsLiteEnv:
    """MarS Lite環境のテスト"""

    @pytest.fixture
    def env(self):
        """テスト用環境を生成"""
        data = create_preprocessed_data(n_bars=2000)
        return MarsLiteEnv(
            data=data,
            initial_inventory=1000.0,
            max_steps=100,
            side="SELL",
        )

    def test_reset(self, env):
        """リセットが正常動作"""
        obs, info = env.reset(seed=42)

        assert obs is not None
        assert len(obs) == env.observation_space.shape[0]
        assert info["inventory"] == env.initial_inventory

    def test_step_reduces_inventory(self, env):
        """ステップで在庫が減少"""
        env.reset(seed=42)

        action = np.array([0.1])  # 10%執行
        obs, reward, terminated, truncated, info = env.step(action)

        assert info["inventory"] < env.initial_inventory

    def test_episode_terminates(self, env):
        """エピソードが終了する"""
        env.reset(seed=42)

        done = False
        steps = 0
        while not done and steps < 200:
            action = np.array([0.5])  # 50%ずつ
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            steps += 1

        assert done

    def test_look_ahead_bias_prevention(self, env):
        """Look-ahead bias防止: Next Openを使用"""
        env.reset(seed=42, options={"start_idx": 20})

        action = np.array([0.1])
        obs, reward, terminated, truncated, info = env.step(action)

        # P_baseはNext Openであることを確認
        exec_hist = env.get_execution_history()
        assert len(exec_hist) == 1

        # 次のバーのopenと一致することを確認
        next_open = env.data.iloc[21]["open"]
        assert exec_hist.iloc[0]["p_base"] == next_open


class TestEnvironmentBehavior:
    """環境の様式化された挙動のテスト"""

    @pytest.fixture
    def env(self):
        data = create_preprocessed_data(n_bars=2000)
        return MarsLiteEnv(
            data=data,
            initial_inventory=1000.0,
            max_steps=100,
            side="SELL",
        )

    def test_action_clipping(self, env):
        """アクションが[0, 1]にクリップされる"""
        env.reset(seed=42)

        # 範囲外アクション
        action = np.array([1.5])
        obs, reward, _, _, info = env.step(action)

        # 在庫が0以上
        assert info["inventory"] >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
