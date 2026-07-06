import time
from typing import Optional

from stable_baselines3.common.callbacks import BaseCallback

from mars_lite.learning.training_callback import MetricsHistory


class CurriculumCallback(BaseCallback):
    """
    Curriculum Learning Callback

    一定ステップ数経過後にEnvの報酬モードを切り替える。
    PnL Mode (Warmup) -> DSR Mode
    """

    def __init__(
        self,
        dsr_warmup_steps: int,
        metrics_history: Optional[MetricsHistory] = None,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.dsr_warmup_steps = dsr_warmup_steps
        self.metrics_history = metrics_history
        self._switched = False

    def _on_step(self) -> bool:
        # Check if warmup steps reached
        if not self._switched and self.num_timesteps >= self.dsr_warmup_steps:
            # Switch to DSR Mode
            self._switch_to_dsr()
            self._switched = True

        return True

    def _switch_to_dsr(self):
        # Access Env
        # TrainingEnv is usually a VecEnv.
        env = self.training_env

        # Unwrap to find MarsLiteEnv
        # VecEnv -> Monitor -> MarsLiteEnv
        # Iterate over all envs if VecEnv

        # SB3 VecEnv wrapper method call
        # env.env_method("set_reward_mode", True) ?

        try:
            # Try calling method for all environments
            env.env_method("set_reward_mode", True)

            msg = (
                f"Curriculum Update: Switched to DSR Mode at step {self.num_timesteps}"
            )
            if self.verbose > 0:
                print(msg)

            if self.metrics_history:
                self.metrics_history.add(
                    {
                        "type": "log",
                        "message": msg,
                        "level": "info",
                        "timestamp": time.time(),
                    }
                )

        except Exception as e:
            print(f"Failed to switch reward mode: {e}")
            # Try manual unwrap if env_method fails?
            # MarsLiteEnv might be deep inside.
