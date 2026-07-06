import sys
from unittest.mock import MagicMock

import numpy as np

# Mock stable_baselines3 PPO if not installed or for simple testing
sys.modules["stable_baselines3"] = MagicMock()
sys.modules["stable_baselines3"].PPO = MagicMock()


# Mock PPO.load to return a mock model with predictable output
class MockModel:
    def __init__(self, value):
        self.value = value

    def predict(self, obs, **kwargs):
        # Return constant action + slight noise based on obs shape
        action = np.full((1, 1), self.value, dtype=np.float32)
        return action, None


def mock_load(path, **kwargs):
    # Determine value from path string for testing
    if "model_a" in str(path):
        return MockModel(1.0)  # Always Buy
    elif "model_b" in str(path):
        return MockModel(-0.5)  # Weak Sell
    elif "model_c" in str(path):
        return MockModel(0.2)  # Weak Buy
    return MockModel(0.0)


sys.modules["stable_baselines3"].PPO.load = mock_load

# Import after mocking
from mars_lite.models.ensemble import EnsemblePredictor


def test_ensemble():
    print("Testing EnsemblePredictor...")

    # Mock paths
    paths = ["model_a.zip", "model_b.zip", "model_c.zip"]

    # Initialize
    # Ensure paths exist check doesn't fail - create dummy files
    for p in paths:
        with open(p, "w") as f:
            f.write("dummy")

    predictor = EnsemblePredictor(paths, device="cpu")

    # Mock Observation
    obs = np.zeros((1, 10), dtype=np.float32)

    # 1. Mean Aggregation
    # A: 1.0, B: -0.5, C: 0.2 -> Mean: (1.0 - 0.5 + 0.2) / 3 = 0.7 / 3 = 0.233...
    action_mean, _ = predictor.predict(obs, method="mean")
    print(f"Mean Action: {action_mean[0]} (Expected ~0.233)")

    # 2. Vote Aggregation
    # Signs: A(+), B(-), C(+) -> + prevails.
    # Mean of prevailing (+): (1.0 + 0.2) / 2? No, logic was mean of ALL if avg sign > 0.5?
    # Logic in code:
    # signs = [1, -1, 1] -> avg = 1/3 (0.33)
    # mask = abs(0.33) >= 0.5 -> False. Result 0?
    # Let's check logic: "if abs(avg_sign) >= 0.5".
    # 2 vs 1 vote -> 1/3 net sign. 0.33 < 0.5. So it effectively filters weakness.
    # If we want simple majority (2/3), threshold should be lower?
    # Avg sign 0.33 means (2 pos - 1 neg) / 3.
    # If threshold 0.5, we need (3 pos - 0 neg) -> 1.0 or (3 pos - 1 neg)/4 = 0.5.
    # So threshold 0.5 requires strong consensus.

    action_vote, _ = predictor.predict(obs, method="vote")
    print(f"Vote Action: {action_vote[0]} (Expected 0.0 due to strict threshold)")

    # Cleanup
    import os

    for p in paths:
        if os.path.exists(p):
            os.remove(p)


if __name__ == "__main__":
    test_ensemble()
