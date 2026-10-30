"""Fixed-budget PPO training of the residual throttle (optional ``train-sb3`` extra).

stable-baselines3 and torch are imported lazily so the rest of the package, and the
core install, never require them. A missing runtime raises a ``RuntimeError`` that
names the extra.
"""

from __future__ import annotations

import importlib
from typing import Any

import numpy as np

from trade_rl.evaluation.residual_rl.env import ResidualAllocatorEnv
from trade_rl.evaluation.residual_rl.evaluate import ActionChooser, env_config
from trade_rl.evaluation.residual_rl.market import Regime
from trade_rl.evaluation.residual_rl.protocol import ResidualProtocol


def _runtime() -> tuple[Any, Any]:
    try:
        torch = importlib.import_module("torch")
        sb3 = importlib.import_module("stable_baselines3")
    except ImportError as exc:  # pragma: no cover - exercised without the extra
        raise RuntimeError(
            "residual PPO training needs the train-sb3 extra: uv sync --extra train-sb3"
        ) from exc
    return torch, sb3


def runtime_versions() -> dict[str, str]:
    torch, sb3 = _runtime()
    return {
        "stable_baselines3": str(sb3.__version__),
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


def train_residual_ppo(
    protocol: ResidualProtocol,
    regime: Regime,
    ppo_seed: int,
    *,
    total_timesteps: int | None = None,
) -> Any:
    """Train one PPO agent on the training seed pool of ``regime``.

    ``total_timesteps`` overrides the protocol budget only for smoke runs and tests;
    the study run always uses the frozen budget.
    """

    torch, sb3 = _runtime()
    torch.set_num_threads(1)
    ppo = protocol.ppo
    config = env_config(protocol, regime, reward_scale=ppo.reward_scale)
    env = ResidualAllocatorEnv(config, protocol.seeds.train)
    model = sb3.PPO(
        "MlpPolicy",
        env,
        learning_rate=ppo.learning_rate,
        n_steps=ppo.n_steps,
        batch_size=ppo.batch_size,
        n_epochs=ppo.n_epochs,
        gamma=ppo.gamma,
        gae_lambda=ppo.gae_lambda,
        clip_range=ppo.clip_range,
        ent_coef=ppo.ent_coef,
        vf_coef=ppo.vf_coef,
        policy_kwargs={"net_arch": list(ppo.net_arch)},
        seed=ppo_seed,
        device="cpu",
        verbose=0,
    )
    model.learn(total_timesteps=total_timesteps or ppo.total_timesteps)
    return model


def model_chooser(model: Any, *, deterministic: bool) -> ActionChooser:
    def choose(observation: np.ndarray) -> int:
        action, _ = model.predict(observation, deterministic=deterministic)
        return int(np.asarray(action).reshape(-1)[0])

    return choose


__all__ = ["model_chooser", "runtime_versions", "train_residual_ppo"]
