import time

from mars_lite.server.training_manager import TrainingConfig, TrainingManager

print("Starting direct test...")
manager = TrainingManager()
config = TrainingConfig(
    total_timesteps=500,
    checkpoint_freq=100,
    symbol="BTCUSDT",
    interval="1h",
    output_dir="./output",
)

# Start
res = manager.start(config)
print(f"Start result: {res}")

# Wait for training
for i in range(20):
    time.sleep(1)
    status = manager.get_status_info()
    print(
        f"Step {i}: Status={status.get('status')}, Current Step={status.get('current_step')}"
    )
    if status.get("status") in ["completed", "error"]:
        break

# Stop
if manager.is_running:
    print("Stopping...")
    manager.stop()
    time.sleep(2)

print("Test finished.")
