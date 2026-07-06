import os
import sys
from unittest.mock import MagicMock

import pandas as pd

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from mars_lite.env.mars_lite_env import MarsLiteEnv
from stable_baselines3.common.dummy_vec_env import DummyVecEnv

from mars_lite.learning.curriculum_callback import CurriculumCallback
from mars_lite.learning.training_callback import MetricsHistory


def create_dummy_env():
    # Create minimal dummy data
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=100, freq="1h"),
            "open": [100.0] * 100,
            "high": [101.0] * 100,
            "low": [99.0] * 100,
            "close": [100.0] * 100,
            "volume": [1000.0] * 100,
            "log_return": [0.0] * 100,
        }
    )

    data_dict = {"BTCUSDT": {"1h": df}}

    env = MarsLiteEnv(
        data_dict=data_dict,
        use_dsr=False,  # Start PnL
        timeframes=["1h"],
        max_steps=50,
    )
    return env


def test_curriculum_callback():
    print("Testing Curriculum Callback Logic...")

    # 1. Setup Env
    env = create_dummy_env()
    vec_env = DummyVecEnv([lambda: env])

    # Verify Initial State
    print(f"Initial State: Use DSR = {env.use_dsr}")
    assert env.use_dsr == False, "Env should start with use_dsr=False"

    # 2. Setup Callback
    history = MetricsHistory()
    warmup_steps = 5
    callback = CurriculumCallback(
        dsr_warmup_steps=warmup_steps, metrics_history=history, verbose=1
    )

    # Mock SB3 training environment
    callback.init_callback(model=MagicMock())
    callback.model.get_env.return_value = vec_env  # Mock get_env if needed, but callback.training_env is set by init_callback logic usually?
    # SB3 init_callback setting training_env might be complex to mock perfectly without full SB3 run,
    # but let's try to set it manually.
    callback.training_env = vec_env

    # 3. Simulate Steps
    print(f"Simulating {warmup_steps + 2} steps...")

    for i in range(warmup_steps + 2):
        callback.num_timesteps = i + 1
        callback.on_step()

        current_mode = env.use_dsr
        print(f"Step {i + 1}: DSR Mode = {current_mode}")

        if i + 1 < warmup_steps:
            assert current_mode == False, f"Step {i + 1}: Should still be False"
        elif i + 1 >= warmup_steps:
            # Logic checks num_timesteps >= warmup_steps.
            # Step 5 >= 5 -> Trigger Switch.
            assert current_mode == True, f"Step {i + 1}: Should be True"

    # Check History
    events = history.get_all()
    switch_events = [e for e in events if "Switched to DSR Mode" in e["message"]]
    print(f"Switch Events found: {len(switch_events)}")
    assert len(switch_events) == 1, "Should have exactly one switch event"

    print("\nCurriculum Callback Test Passed!")


if __name__ == "__main__":
    test_curriculum_callback()
