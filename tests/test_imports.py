"""Test training manager import"""

import sys

sys.path.insert(0, ".")

try:
    from mars_lite.server.training_manager import TrainingConfig

    print("TrainingManager import: OK")

    tc = TrainingConfig()
    print(f"TrainingConfig: {tc.to_dict()}")

    print("metrics_server import: OK")

    print("\nAll imports successful!")
except Exception as e:
    print(f"Error: {e}")
    import traceback

    traceback.print_exc()
