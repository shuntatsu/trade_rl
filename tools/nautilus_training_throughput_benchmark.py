"""Measure legacy versus Nautilus dual-shadow training throughput on one fixed fixture."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.data.market import MarketDataset
from trade_rl.integrations.nautilus.rl_dual_shadow import NautilusEnvironmentDualShadow
from trade_rl.integrations.sb3_training import StableBaselines3Backend
from trade_rl.rl.actions import ActionSpec
from trade_rl.rl.dual_shadow_environment import ExecutionDualShadowResidualMarketEnv
from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.trend import TrendConfig, TrendStrategy

_SCHEMA_VERSION = "nautilus_training_throughput_benchmark_v1"
_RUNTIME_VERSION = "1.230.0"


def _market(*, timesteps: int) -> MarketDataset:
    n_bars = max(80, timesteps + 32)
    close = np.linspace(100.0, 101.0, n_bars, dtype=np.float64)[:, None]
    return MarketDataset(
        dataset_id="7" * 64,
        symbols=("BTCUSDT",),
        timestamps=np.datetime64("2026-01-01", "ns")
        + np.arange(n_bars) * np.timedelta64(1, "h"),
        features=np.zeros((n_bars, 1, 1), dtype=np.float32),
        global_features=np.zeros((n_bars, 1), dtype=np.float32),
        open=np.vstack([close[0], close[:-1]]),
        high=close + 0.1,
        low=close - 0.1,
        close=close,
        volume=np.full((n_bars, 1), 1_000_000.0),
        funding_rate=np.zeros_like(close),
        tradable=np.ones_like(close, dtype=np.bool_),
        feature_available=np.ones((n_bars, 1, 1), dtype=np.bool_),
        feature_names=("ret",),
        global_feature_names=("regime",),
        periods_per_year=8_760,
        mark_price=close.copy(),
        index_price=close.copy(),
    )


def _common_environment_kwargs(
    dataset: MarketDataset,
    *,
    timesteps: int,
) -> dict[str, Any]:
    return {
        "trend_strategy": TrendStrategy(
            TrendConfig(fast_lookback=2, base_lookback=4, slow_lookback=8)
        ),
        "action_spec": ActionSpec(
            mode="target_weight",
            risk_tilt_enabled=False,
            target_weight_count=1,
        ),
        "config": ResidualMarketEnvConfig(
            initial_capital=1_000.0,
            episode_bars=timesteps,
            decision_every=1,
            execution_cost=ExecutionCostConfig.zero(),
            initial_state_modes=("cash",),
        ),
    }


def _legacy_factory(
    dataset: MarketDataset,
    *,
    timesteps: int,
) -> Callable[[], ResidualMarketEnv]:
    def build() -> ResidualMarketEnv:
        return ResidualMarketEnv(
            dataset,
            **_common_environment_kwargs(dataset, timesteps=timesteps),
        )

    return build


def _streaming_factory(
    dataset: MarketDataset,
    *,
    timesteps: int,
) -> Callable[[], ResidualMarketEnv]:
    def build() -> ResidualMarketEnv:
        return ExecutionDualShadowResidualMarketEnv(
            dataset,
            **_common_environment_kwargs(dataset, timesteps=timesteps),
            execution_dual_shadow=NautilusEnvironmentDualShadow(
                dataset,
                no_trade_band=0.0,
            ),
        )

    return build


def _training_config(*, timesteps: int) -> ResidualTrainingConfig:
    return ResidualTrainingConfig(
        timesteps=timesteps,
        gamma=0.99,
        seeds=(0,),
        algorithm="ppo",
        n_steps=timesteps,
        n_envs=1,
        batch_size=timesteps,
        n_epochs=1,
        observation_encoder="flat_mlp",
        device="cpu",
        policy_net_arch=(16, 8),
        value_net_arch=(16, 8),
    )


def _measure(
    factory: Callable[[], ResidualMarketEnv],
    *,
    timesteps: int,
    output_path: Path,
) -> dict[str, float | int]:
    started = time.perf_counter()
    result = StableBaselines3Backend(factory).train(
        seed=0,
        config=_training_config(timesteps=timesteps),
        output_path=output_path,
    )
    elapsed_seconds = time.perf_counter() - started
    if result.actual_timesteps != timesteps:
        raise RuntimeError(
            "benchmark training did not execute the requested timesteps: "
            f"{result.actual_timesteps} != {timesteps}"
        )
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("benchmark elapsed time must be finite and positive")
    return {
        "actual_timesteps": result.actual_timesteps,
        "elapsed_seconds": elapsed_seconds,
        "steps_per_second": result.actual_timesteps / elapsed_seconds,
    }


def run_benchmark(*, timesteps: int) -> dict[str, Any]:
    if isinstance(timesteps, bool) or not isinstance(timesteps, int):
        raise TypeError("timesteps must be an integer")
    if timesteps < 2:
        raise ValueError("timesteps must be at least 2")
    runtime_version = importlib.metadata.version("nautilus_trader")
    if runtime_version != _RUNTIME_VERSION:
        raise RuntimeError(
            "benchmark requires pinned nautilus_trader==1.230.0; "
            f"found {runtime_version}"
        )

    dataset = _market(timesteps=timesteps)
    with tempfile.TemporaryDirectory(prefix="trade-rl-nautilus-benchmark-") as root:
        root_path = Path(root)
        legacy = _measure(
            _legacy_factory(dataset, timesteps=timesteps),
            timesteps=timesteps,
            output_path=root_path / "legacy" / "policy.zip",
        )
        streaming = _measure(
            _streaming_factory(dataset, timesteps=timesteps),
            timesteps=timesteps,
            output_path=root_path / "streaming" / "policy.zip",
        )

    legacy_elapsed = float(legacy["elapsed_seconds"])
    streaming_elapsed = float(streaming["elapsed_seconds"])
    return {
        "schema_version": _SCHEMA_VERSION,
        "runtime_version": runtime_version,
        "algorithm": "ppo",
        "dataset_kind": "deterministic_synthetic_btcusdt",
        "timesteps": timesteps,
        "legacy_authoritative": legacy,
        "nautilus_dual_shadow_streaming": streaming,
        "elapsed_slowdown_ratio": streaming_elapsed / legacy_elapsed,
        "performance_approved": False,
        "approval_note": (
            "Observational CI microbenchmark only; no production promotion threshold "
            "is defined by this evidence."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, default=8)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    evidence = run_benchmark(timesteps=args.timesteps)
    rendered = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
