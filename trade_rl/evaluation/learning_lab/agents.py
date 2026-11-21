"""Training and deterministic-evaluation wrappers for the overlay agents.

PPO and A2C (discrete trade fractions), SAC (continuous) and QR-DQN (discrete,
distributional) through stable-baselines3 / sb3-contrib. This is a thin wrapper:
``fit_shared_portfolio_ppo`` is not used because it is bound to ``SharedPortfolioGymEnv``'s
per-symbol ``MultiDiscrete`` action and its native-transition journals, while the overlay's
action is one trade fraction. Each run is seeded, trains on CPU, writes one checkpoint zip and
a ``.sha256`` of its exact bytes, and a checkpoint is loaded only after that digest matches.

``stable_baselines3``, ``sb3_contrib`` and ``torch`` are imported lazily (optional
``learning-lab`` extra), so importing this module costs nothing.
"""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.evaluation.learning_lab.metrics import RolloutResult
from trade_rl.evaluation.learning_lab.overlay_env import (
    ACTION_CONTINUOUS,
    ACTION_DISCRETE,
    ExecutionOverlayEnv,
)

#: algorithm -> (module, class, action mode it needs, method id)
ALGORITHMS: Final[dict[str, tuple[str, str, str, str]]] = {
    "PPO": ("stable_baselines3", "PPO", ACTION_DISCRETE, "L1-PPO"),
    "A2C": ("stable_baselines3", "A2C", ACTION_DISCRETE, "L1-A2C"),
    "SAC": ("stable_baselines3", "SAC", ACTION_CONTINUOUS, "L1-SAC"),
    "QRDQN": ("sb3_contrib", "QRDQN", ACTION_DISCRETE, "L1-QRDQN"),
}
_ON_POLICY = ("PPO", "A2C")
_MAX_SEED = (1 << 32) - 1


class AgentError(RuntimeError):
    """An agent could not be trained, stored or loaded as declared."""


def algorithm_for_method(method_id: str) -> str:
    for name, (_, _, _, method) in ALGORITHMS.items():
        if method == method_id:
            return name
    raise AgentError(f"{method_id!r} is not a learned-agent method")


def action_mode_for(algorithm: str) -> str:
    return ALGORITHMS[algorithm][2]


def method_id_for(algorithm: str) -> str:
    return ALGORITHMS[algorithm][3]


@dataclass(frozen=True, slots=True)
class AgentConfig:
    algorithm: str
    total_timesteps: int
    seed: int
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 64
    buffer_size: int = 10_000
    learning_starts: int = 64
    net_arch: tuple[int, ...] = (64, 64)
    gamma: float = 1.0

    def __post_init__(self) -> None:
        if self.algorithm not in ALGORITHMS:
            raise AgentError(f"unknown algorithm {self.algorithm!r}")
        if type(self.total_timesteps) is not int or self.total_timesteps < 1:
            raise AgentError("total_timesteps must be a positive int")
        if type(self.seed) is not int or not 0 <= self.seed <= _MAX_SEED:
            raise AgentError("seed must be an int within [0, 2**32 - 1]")
        if self.batch_size < 1 or self.n_steps < 1 or self.buffer_size < 1:
            raise AgentError("batch_size, n_steps and buffer_size must be positive")
        if not 0.0 < self.gamma <= 1.0:
            raise AgentError("gamma must lie within (0, 1]")

    def payload(self) -> dict[str, Any]:
        out = asdict(self)
        out["net_arch"] = list(self.net_arch)
        return out

    @property
    def digest(self) -> str:
        data = json.dumps(self.payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(data.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class TrainedAgent:
    algorithm: str
    seed: int
    config_digest: str
    checkpoint: Path
    checkpoint_sha256: str


def _load_runtime(algorithm: str) -> tuple[Any, Any]:
    module_name, class_name, _, _ = ALGORITHMS[algorithm]
    try:
        torch = importlib.import_module("torch")
        module = importlib.import_module(module_name)
    except ImportError as error:
        raise AgentError(
            "stable-baselines3, sb3-contrib and torch are required; "
            "install the learning-lab extra (uv sync --extra learning-lab)"
        ) from error
    return torch, getattr(module, class_name)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_matching_env(algorithm: str, env: ExecutionOverlayEnv) -> None:
    needed = action_mode_for(algorithm)
    if env.config.action_mode != needed:
        raise AgentError(
            f"{algorithm} needs a {needed} action space, the environment is "
            f"{env.config.action_mode}"
        )


def _constructor_kwargs(config: AgentConfig) -> dict[str, Any]:
    arch = list(config.net_arch)
    kwargs: dict[str, Any] = {
        "learning_rate": config.learning_rate,
        "gamma": config.gamma,
        "seed": config.seed,
        "device": "cpu",
        "verbose": 0,
    }
    if config.algorithm in _ON_POLICY:
        kwargs["n_steps"] = config.n_steps
        kwargs["policy_kwargs"] = {"net_arch": {"pi": arch, "vf": arch}}
        if config.algorithm == "PPO":
            kwargs["batch_size"] = min(config.batch_size, config.n_steps)
    else:
        kwargs["buffer_size"] = config.buffer_size
        kwargs["learning_starts"] = config.learning_starts
        kwargs["batch_size"] = config.batch_size
        kwargs["policy_kwargs"] = {"net_arch": arch}
        if config.algorithm == "QRDQN":
            kwargs["target_update_interval"] = max(10, config.learning_starts)
            kwargs["exploration_fraction"] = 0.3
    return kwargs


def train_agent(
    env: ExecutionOverlayEnv,
    config: AgentConfig,
    output_dir: Path | str,
) -> TrainedAgent:
    """Train one fresh agent and write ``<ALGO>-seed<k>.zip`` plus its ``.sha256``."""
    _require_matching_env(config.algorithm, env)
    torch, model_class = _load_runtime(config.algorithm)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"{config.algorithm}-seed{config.seed}"
    target = out / f"{stem}.zip"
    if target.exists():
        raise FileExistsError(f"refusing to overwrite checkpoint {target}")
    torch.manual_seed(config.seed)
    env.reset(seed=config.seed)
    model = model_class("MlpPolicy", env, **_constructor_kwargs(config))
    model.learn(total_timesteps=config.total_timesteps)
    model.save(str(target))
    digest = _sha256_file(target)
    (out / f"{stem}.sha256").write_text(f"{digest}  {target.name}\n", encoding="utf-8")
    return TrainedAgent(
        algorithm=config.algorithm,
        seed=config.seed,
        config_digest=config.digest,
        checkpoint=target,
        checkpoint_sha256=digest,
    )


def load_agent(algorithm: str, checkpoint: Path | str, *, sha256: str) -> Any:
    """Load a checkpoint only if its exact bytes hash to ``sha256``."""
    _, model_class = _load_runtime(algorithm)
    path = Path(checkpoint)
    if _sha256_file(path) != sha256:
        raise AgentError(f"checkpoint {path.name} does not match its recorded sha256")
    return model_class.load(str(path), device="cpu")


def evaluate_agent(
    model: Any,
    env: ExecutionOverlayEnv,
    *,
    method: str,
    seed: int = 0,
) -> RolloutResult:
    """One deterministic episode of ``model`` on ``env`` (greedy actions, fixed reset seed)."""
    observation, _ = env.reset(seed=seed)
    done = False
    while not done:
        action, _ = model.predict(observation, deterministic=True)
        if env.config.action_mode == ACTION_DISCRETE:
            action = int(np.asarray(action).reshape(-1)[0])
        observation, _, done, _, _ = env.step(action)
    book = env.session.book
    return RolloutResult(
        method=method,
        bar_returns=env.episode_bar_returns,
        final_equity=float(book.portfolio_value),
        total_cost=float(book.total_cost),
        turnover_total=float(book.turnover_total),
        initial_capital=env.config.initial_capital,
    )


__all__ = [
    "ALGORITHMS",
    "AgentConfig",
    "AgentError",
    "TrainedAgent",
    "action_mode_for",
    "algorithm_for_method",
    "evaluate_agent",
    "load_agent",
    "method_id_for",
    "train_agent",
]
