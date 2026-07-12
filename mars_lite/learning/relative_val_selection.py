from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def rollout_aligned_eval_freq(
    *, total_timesteps: int, one_rollout_steps: int, n_eval_targets: int = 10
) -> int:
    if total_timesteps <= 0 or one_rollout_steps <= 0 or n_eval_targets <= 0:
        raise ValueError("timesteps, rollout size, and target count must be positive")
    rollouts = max(1, total_timesteps // one_rollout_steps)
    rollouts_per_eval = max(1, rollouts // n_eval_targets)
    return rollouts_per_eval * one_rollout_steps


@dataclass(frozen=True)
class RelativeCheckpointScore:
    step: int
    block_excess: tuple[float, ...]
    median_excess: float
    positive_block_ratio: float
    drawdown_excess: float = 0.0
    turnover_excess: float = 0.0
    baseline_fallback: bool = False

    @classmethod
    def from_blocks(
        cls,
        *,
        step: int,
        blocks,
        drawdown_excess: float = 0.0,
        turnover_excess: float = 0.0,
    ) -> "RelativeCheckpointScore":
        values = np.asarray(tuple(blocks), dtype=np.float64)
        if values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("blocks must contain finite values")
        if step < 0:
            raise ValueError("step must be non-negative")
        if not np.isfinite(drawdown_excess) or not np.isfinite(turnover_excess):
            raise ValueError("tie-break metrics must be finite")
        return cls(
            step=int(step),
            block_excess=tuple(float(x) for x in values),
            median_excess=float(np.median(values)),
            positive_block_ratio=float(np.mean(values > 0.0)),
            drawdown_excess=float(drawdown_excess),
            turnover_excess=float(turnover_excess),
        )

    @property
    def eligible(self) -> bool:
        return self.median_excess > 0.0 and self.positive_block_ratio >= 0.5

    @classmethod
    def identity(cls) -> "RelativeCheckpointScore":
        return cls(
            step=0,
            block_excess=(0.0,),
            median_excess=0.0,
            positive_block_ratio=0.0,
            baseline_fallback=True,
        )


def choose_relative_checkpoint(
    scores: list[RelativeCheckpointScore],
) -> RelativeCheckpointScore:
    eligible = [score for score in scores if score.eligible]
    if not eligible:
        return RelativeCheckpointScore.identity()
    return max(
        eligible,
        key=lambda score: (
            score.median_excess,
            score.positive_block_ratio,
            -score.drawdown_excess,
            -score.turnover_excess,
            -score.step,
        ),
    )


def quick_relative_evaluate(
    agent,
    fs,
    *,
    env_kwargs: Optional[dict] = None,
    n_blocks: int = 3,
    step: int = 0,
    start_idx: int = 0,
) -> RelativeCheckpointScore:
    from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv

    if not 0 <= start_idx < max(1, fs.n_bars - 2):
        raise ValueError("start_idx is outside the validation executable range")
    kwargs = dict(env_kwargs or {})
    kwargs.pop("episode_bars", None)
    episode_bars = max(1, fs.n_bars - 2 - start_idx)
    env = BaselineResidualTradingEnv(fs, episode_bars=episode_bars, **kwargs)
    obs, _ = env.reset(options={"start_idx": start_idx})
    excess: list[float] = []
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        excess.append(float(info["excess_log_return"]))
        done = term or trunc
    blocks = [float(np.sum(block)) for block in np.array_split(excess, n_blocks)]
    return RelativeCheckpointScore.from_blocks(
        step=step,
        blocks=blocks,
        drawdown_excess=env.hybrid.max_drawdown - env.shadow.max_drawdown,
        turnover_excess=env.hybrid.turnover_total - env.shadow.turnover_total,
    )


class RelativeValSelectionCallback(BaseCallback):
    """Select checkpoints only when they add validation value over base trend."""

    def __init__(
        self,
        val_fs,
        *,
        eval_freq: int,
        env_kwargs: Optional[dict] = None,
        start_idx: int = 0,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.val_fs = val_fs
        self.eval_freq = int(eval_freq)
        self.env_kwargs = dict(env_kwargs or {})
        self.start_idx = int(start_idx)
        self.history: list[RelativeCheckpointScore] = []
        self.identity_params: Optional[bytes] = None
        self.best_params: Optional[bytes] = None
        self.best_score = RelativeCheckpointScore.identity()

    @staticmethod
    def _serialize(model) -> bytes:
        buffer = io.BytesIO()
        model.save(buffer)
        return buffer.getvalue()

    def _evaluate_and_save(self) -> None:
        score = quick_relative_evaluate(
            self.model,
            self.val_fs,
            env_kwargs=self.env_kwargs,
            step=self.num_timesteps,
            start_idx=self.start_idx,
        )
        self.history.append(score)
        selected = choose_relative_checkpoint(self.history)
        if selected.step == score.step and selected.eligible:
            self.best_score = selected
            self.best_params = self._serialize(self.model)
        if self.verbose >= 1:
            print(
                f"[RelativeValSelection] step={score.step:,} "
                f"median_excess={score.median_excess:+.6f} "
                f"positive_blocks={score.positive_block_ratio:.0%}"
            )

    def _on_training_start(self) -> None:
        self.identity_params = self._serialize(self.model)
        self._evaluate_and_save()

    def _on_step(self) -> bool:
        if self.num_timesteps % self.eval_freq < self.training_env.num_envs:
            self._evaluate_and_save()
        return True

    def _on_training_end(self) -> None:
        if not self.history or self.history[-1].step != self.num_timesteps:
            self._evaluate_and_save()

    def restore_best(self, agent):
        from stable_baselines3 import PPO

        payload = self.best_params or self.identity_params
        if payload is None:
            return agent
        restored = PPO.load(io.BytesIO(payload), device=agent.device)
        agent.set_parameters(restored.get_parameters())
        if self.best_params is None:
            self.best_score = RelativeCheckpointScore.identity()
        return agent
