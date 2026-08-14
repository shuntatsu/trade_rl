"""Process-safe observation collection for walk-forward normalizer fitting."""

from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.signal_artifacts import (
    LoadedAlphaArtifact,
    LoadedFactorArtifact,
)
from trade_rl.rl.environment import ResidualMarketEnv
from trade_rl.workflows.training_run import TrainingRunConfig
from trade_rl.workflows.walk_forward_evaluation import build_market_environment


@dataclass(frozen=True, slots=True)
class NormalizerWorkerSpec:
    """Serializable inputs required to rebuild one normalizer environment."""

    dataset: MarketDataset
    run: TrainingRunConfig
    episode_bars: int
    alpha_provider: LoadedAlphaArtifact | None = None
    factor_provider: LoadedFactorArtifact | None = None


_NORMALIZER_WORKER_SPEC: NormalizerWorkerSpec | None = None


def _normalizer_worker_count(episode_bars: int) -> int:
    raw = os.environ.get("TRADE_RL_PREPROCESS_WORKERS", "8").strip()
    try:
        configured = int(raw)
    except ValueError as exc:
        raise ValueError("TRADE_RL_PREPROCESS_WORKERS must be an integer") from exc
    if configured <= 0:
        raise ValueError("TRADE_RL_PREPROCESS_WORKERS must be positive")
    return min(configured, episode_bars)


def _normalizer_partitions(
    start: int,
    stop: int,
    worker_count: int,
    action_size: int,
) -> tuple[tuple[int, int, int], ...]:
    boundaries = np.linspace(start, stop, worker_count + 1, dtype=np.int64)
    return tuple(
        (int(left), int(right), action_size)
        for left, right in zip(boundaries[:-1], boundaries[1:], strict=True)
        if right > left
    )


def _normalizer_start_method() -> str:
    """Choose a process start method that never forks a live parent."""

    available = multiprocessing.get_all_start_methods()
    if "forkserver" in available:
        return "forkserver"
    if "spawn" in available:
        return "spawn"
    raise RuntimeError("normalizer parallel collection requires forkserver or spawn")


def normalizer_environment(spec: NormalizerWorkerSpec) -> ResidualMarketEnv:
    """Build one unnormalized cash-start environment from serializable inputs."""

    return build_market_environment(
        spec.dataset,
        spec.run,
        normalizer=None,
        sequence_normalizer=None,
        episode_bars=spec.episode_bars,
        liquidate_on_end=False,
        alpha_provider=spec.alpha_provider,
        factor_provider=spec.factor_provider,
    )


def _collect_normalizer_chunk_from_spec(
    spec: NormalizerWorkerSpec,
    bounds: tuple[int, int, int],
) -> np.ndarray:
    start, stop, action_size = bounds
    environment = normalizer_environment(spec)
    observations: list[np.ndarray] = []
    try:
        observation, _ = environment.reset(
            seed=0,
            options={
                "episode_bars": stop - start,
                "initial_state_mode": "cash",
                "start_idx": start,
            },
        )
        terminated = False
        truncated = False
        while not terminated and not truncated:
            observations.append(
                np.asarray(observation, dtype=np.float32).copy(order="C")
            )
            observation, _, terminated, truncated, _ = environment.step(
                np.zeros(action_size, dtype=np.float32)
            )
    finally:
        environment.close()
    if len(observations) != stop - start:
        raise RuntimeError("normalizer worker observation count mismatch")
    return np.stack(observations, axis=0)


def _initialize_normalizer_worker(spec: NormalizerWorkerSpec) -> None:
    global _NORMALIZER_WORKER_SPEC
    _NORMALIZER_WORKER_SPEC = spec


def _collect_normalizer_worker_chunk(
    bounds: tuple[int, int, int],
) -> np.ndarray:
    spec = _NORMALIZER_WORKER_SPEC
    if spec is None:
        raise RuntimeError("normalizer worker specification is unavailable")
    return _collect_normalizer_chunk_from_spec(spec, bounds)


def collect_normalizer_matrix(
    spec: NormalizerWorkerSpec,
    *,
    start: int,
    episode_bars: int,
    action_size: int,
    finite_horizon: bool,
) -> np.ndarray:
    """Collect deterministic observations serially or with safe child processes."""

    worker_count = _normalizer_worker_count(episode_bars)
    if worker_count <= 1 or finite_horizon:
        return _collect_normalizer_chunk_from_spec(
            spec,
            (start, start + episode_bars, action_size),
        )

    partitions = _normalizer_partitions(
        start,
        start + episode_bars,
        worker_count,
        action_size,
    )
    context = multiprocessing.get_context(_normalizer_start_method())
    with ProcessPoolExecutor(
        max_workers=len(partitions),
        mp_context=context,
        initializer=_initialize_normalizer_worker,
        initargs=(spec,),
    ) as executor:
        chunks = tuple(executor.map(_collect_normalizer_worker_chunk, partitions))
    return np.concatenate(chunks, axis=0)
