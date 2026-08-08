"""Measure isolated legacy versus Nautilus training throughput and memory."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
import platform
import subprocess
import sys
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Literal

from trade_rl.artifacts.hashing import content_digest
from trade_rl.simulation.runtime_performance import (
    RuntimePerformanceEvidence,
    RuntimePerformanceMeasurement,
    RuntimePerformanceWorkload,
)

_RUNTIME_VERSION = "1.230.0"
_DEFAULT_TIMESTEPS = (8, 32)
_REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
_BENCHMARK_SOURCE_SCHEMA = "nautilus_training_performance_source_v1"
_WorkerMode = Literal["legacy", "streaming"]


def _normalize_timesteps(value: int | Sequence[int]) -> tuple[int, ...]:
    if isinstance(value, bool):
        raise TypeError("timesteps must contain integers")
    raw: tuple[int, ...]
    if isinstance(value, int):
        raw = (value,)
    else:
        raw = tuple(value)
    if not raw:
        raise ValueError("benchmark requires at least one timestep workload")
    if any(isinstance(item, bool) or not isinstance(item, int) for item in raw):
        raise TypeError("timesteps must contain integers")
    if any(item < 2 for item in raw):
        raise ValueError("timesteps must be at least 2")
    normalized = tuple(sorted(set(raw)))
    return normalized


def _benchmark_source_digest(workloads: tuple[int, ...]) -> str:
    """Bind measurements to the exact deterministic benchmark workload contract."""

    return content_digest(
        {
            "algorithm": "ppo",
            "dataset": {
                "close_end": 101.0,
                "close_start": 100.0,
                "dataset_id": "7" * 64,
                "feature_names": ("ret",),
                "global_feature_names": ("regime",),
                "interval": "1h",
                "n_bars_rule": "max(80,timesteps+32)",
                "periods_per_year": 8_760,
                "quote_volume": 1_000_000.0,
                "start": "2026-01-01T00:00:00",
                "symbol": "BTCUSDT",
            },
            "dataset_kind": "deterministic_synthetic_btcusdt",
            "environment": {
                "decision_every": 1,
                "episode_bars": "timesteps",
                "execution_cost": "zero",
                "initial_capital": 1_000.0,
                "initial_state_modes": ("cash",),
                "no_trade_band": 0.0,
                "target_weight_count": 1,
                "trend_lookbacks": (2, 4, 8),
            },
            "schema_version": _BENCHMARK_SOURCE_SCHEMA,
            "training": {
                "batch_size": "timesteps",
                "device": "cpu",
                "gamma": 0.99,
                "n_envs": 1,
                "n_epochs": 1,
                "n_steps": "timesteps",
                "observation_encoder": "flat_mlp",
                "policy_net_arch": (16, 8),
                "seeds": (0,),
                "value_net_arch": (16, 8),
            },
            "workloads": workloads,
        }
    )


def _linux_status(pid: int) -> tuple[int, int] | None:
    try:
        lines = (
            (Path("/proc") / str(pid) / "status")
            .read_text(
                encoding="utf-8",
                errors="replace",
            )
            .splitlines()
        )
    except (FileNotFoundError, PermissionError, ProcessLookupError):
        return None
    parent_pid: int | None = None
    rss_bytes = 0
    for line in lines:
        if line.startswith("PPid:"):
            parent_pid = int(line.split()[1])
        elif line.startswith("VmRSS:"):
            rss_bytes = int(line.split()[1]) * 1024
    if parent_pid is None:
        return None
    return parent_pid, rss_bytes


def _sample_linux_process_tree(root_pid: int) -> tuple[int, int]:
    records: dict[int, tuple[int, int]] = {}
    try:
        entries = tuple(Path("/proc").iterdir())
    except OSError:
        return 0, 0
    for entry in entries:
        if not entry.name.isdigit():
            continue
        pid = int(entry.name)
        status = _linux_status(pid)
        if status is not None:
            records[pid] = status
    descendants = {root_pid}
    changed = True
    while changed:
        changed = False
        for pid, (parent_pid, _) in records.items():
            if pid not in descendants and parent_pid in descendants:
                descendants.add(pid)
                changed = True
    rss_bytes = sum(records.get(pid, (0, 0))[1] for pid in descendants)
    present_count = sum(pid in records for pid in descendants)
    return rss_bytes, present_count


def _run_worker_subprocess(
    *,
    mode: _WorkerMode,
    timesteps: int,
    root: Path,
) -> RuntimePerformanceMeasurement:
    measurement_path = root / f"{mode}-{timesteps}-measurement.json"
    log_path = root / f"{mode}-{timesteps}.log"
    command = (
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-mode",
        mode,
        "--worker-timesteps",
        str(timesteps),
        "--worker-output",
        str(measurement_path),
    )
    peak_tree_rss_bytes = 0
    peak_process_count = 0
    with log_path.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            command,
            cwd=_REPOSITORY_ROOT,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        while process.poll() is None:
            tree_rss_bytes, process_count = _sample_linux_process_tree(process.pid)
            peak_tree_rss_bytes = max(peak_tree_rss_bytes, tree_rss_bytes)
            peak_process_count = max(peak_process_count, process_count)
            time.sleep(0.05)
        return_code = process.wait()
    if return_code != 0:
        log = log_path.read_text(encoding="utf-8", errors="replace")
        raise RuntimeError(
            f"{mode} benchmark worker failed with exit code {return_code}:\n{log}"
        )
    try:
        payload = json.loads(measurement_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as error:
        raise RuntimeError(
            f"{mode} benchmark worker did not emit valid evidence"
        ) from error
    if not isinstance(payload, dict):
        raise RuntimeError(f"{mode} benchmark worker evidence must be an object")
    expected = {
        "elapsed_seconds",
        "peak_children_rss_bytes",
        "peak_self_rss_bytes",
        "steps_per_second",
        "timesteps",
    }
    if set(payload) != expected:
        raise RuntimeError(f"{mode} benchmark worker field closure mismatch")
    self_rss = int(payload["peak_self_rss_bytes"])
    children_rss = int(payload["peak_children_rss_bytes"])
    peak_tree_rss_bytes = max(peak_tree_rss_bytes, self_rss, children_rss)
    peak_process_count = max(
        peak_process_count,
        1 + int(children_rss > 0),
    )
    return RuntimePerformanceMeasurement(
        timesteps=int(payload["timesteps"]),
        elapsed_seconds=float(payload["elapsed_seconds"]),
        steps_per_second=float(payload["steps_per_second"]),
        peak_self_rss_bytes=self_rss,
        peak_children_rss_bytes=children_rss,
        peak_process_tree_rss_bytes=peak_tree_rss_bytes,
        peak_process_count=peak_process_count,
    )


def run_benchmark(*, timesteps: int | Sequence[int]) -> dict[str, Any]:
    workloads = _normalize_timesteps(timesteps)
    if not sys.platform.startswith("linux") or not Path("/proc").is_dir():
        raise RuntimeError(
            "benchmark process-tree RSS measurement requires Linux /proc"
        )
    runtime_version = importlib.metadata.version("nautilus_trader")
    if runtime_version != _RUNTIME_VERSION:
        raise RuntimeError(
            "benchmark requires pinned nautilus_trader==1.230.0; "
            f"found {runtime_version}"
        )

    paired: list[RuntimePerformanceWorkload] = []
    with tempfile.TemporaryDirectory(prefix="trade-rl-nautilus-benchmark-") as root:
        root_path = Path(root)
        for workload_timesteps in workloads:
            legacy = _run_worker_subprocess(
                mode="legacy",
                timesteps=workload_timesteps,
                root=root_path,
            )
            streaming = _run_worker_subprocess(
                mode="streaming",
                timesteps=workload_timesteps,
                root=root_path,
            )
            paired.append(
                RuntimePerformanceWorkload(
                    timesteps=workload_timesteps,
                    legacy_authoritative=legacy,
                    nautilus_dual_shadow_streaming=streaming,
                )
            )

    evidence = RuntimePerformanceEvidence(
        runtime_version=runtime_version,
        platform=f"{sys.platform}-{platform.machine().lower()}",
        algorithm="ppo",
        dataset_kind="deterministic_synthetic_btcusdt",
        source_digest=_benchmark_source_digest(workloads),
        workloads=tuple(paired),
        performance_approved=False,
        approval_policy_digest=None,
        approval_note=(
            "Observational isolated-process CI evidence only. Process-tree RSS is "
            "sampled from Linux /proc at 50 ms intervals; no reviewed production "
            "promotion threshold is bound to this evidence."
        ),
    )
    return {"evidence_digest": evidence.digest, **evidence.to_mapping()}


def _worker_training_measurement(
    *,
    mode: _WorkerMode,
    timesteps: int,
) -> dict[str, float | int]:
    import resource
    from collections.abc import Callable

    import numpy as np

    from trade_rl.data.market import MarketDataset
    from trade_rl.integrations.nautilus.rl_dual_shadow import (
        NautilusEnvironmentDualShadow,
    )
    from trade_rl.integrations.sb3_training import StableBaselines3Backend
    from trade_rl.rl.actions import ActionSpec
    from trade_rl.rl.dual_shadow_environment import (
        ExecutionDualShadowResidualMarketEnv,
    )
    from trade_rl.rl.environment import ResidualMarketEnv, ResidualMarketEnvConfig
    from trade_rl.rl.training import ResidualTrainingConfig
    from trade_rl.simulation.execution import ExecutionCostConfig
    from trade_rl.strategies.trend import TrendConfig, TrendStrategy

    runtime_version = importlib.metadata.version("nautilus_trader")
    if runtime_version != _RUNTIME_VERSION:
        raise RuntimeError(
            "benchmark worker requires pinned nautilus_trader==1.230.0; "
            f"found {runtime_version}"
        )

    n_bars = max(80, timesteps + 32)
    close = np.linspace(100.0, 101.0, n_bars, dtype=np.float64)[:, None]
    dataset = MarketDataset(
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

    common_environment_kwargs: dict[str, Any] = {
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

    def legacy_factory() -> ResidualMarketEnv:
        return ResidualMarketEnv(dataset, **common_environment_kwargs)

    def streaming_factory() -> ResidualMarketEnv:
        return ExecutionDualShadowResidualMarketEnv(
            dataset,
            **common_environment_kwargs,
            execution_dual_shadow=NautilusEnvironmentDualShadow(
                dataset,
                no_trade_band=0.0,
            ),
        )

    factory: Callable[[], ResidualMarketEnv]
    if mode == "legacy":
        factory = legacy_factory
    else:
        factory = streaming_factory

    config = ResidualTrainingConfig(
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
    with tempfile.TemporaryDirectory(prefix=f"trade-rl-{mode}-worker-") as root:
        started = time.perf_counter()
        result = StableBaselines3Backend(factory).train(
            seed=0,
            config=config,
            output_path=Path(root) / "policy.zip",
        )
        elapsed_seconds = time.perf_counter() - started
    if result.actual_timesteps != timesteps:
        raise RuntimeError(
            "benchmark training did not execute the requested timesteps: "
            f"{result.actual_timesteps} != {timesteps}"
        )
    if not math.isfinite(elapsed_seconds) or elapsed_seconds <= 0.0:
        raise RuntimeError("benchmark elapsed time must be finite and positive")

    self_usage = resource.getrusage(resource.RUSAGE_SELF)
    child_usage = resource.getrusage(resource.RUSAGE_CHILDREN)
    peak_self_rss_bytes = int(self_usage.ru_maxrss) * 1024
    peak_children_rss_bytes = int(child_usage.ru_maxrss) * 1024
    return {
        "elapsed_seconds": elapsed_seconds,
        "peak_children_rss_bytes": peak_children_rss_bytes,
        "peak_self_rss_bytes": peak_self_rss_bytes,
        "steps_per_second": timesteps / elapsed_seconds,
        "timesteps": timesteps,
    }


def _run_worker_from_args(args: argparse.Namespace) -> None:
    if (
        args.worker_mode is None
        or args.worker_timesteps is None
        or args.worker_output is None
    ):
        raise SystemExit("worker mode requires --worker-timesteps and --worker-output")
    measurement = _worker_training_measurement(
        mode=args.worker_mode,
        timesteps=args.worker_timesteps,
    )
    rendered = json.dumps(measurement, sort_keys=True, separators=(",", ":")) + "\n"
    args.worker_output.parent.mkdir(parents=True, exist_ok=True)
    args.worker_output.write_text(rendered, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timesteps", type=int, action="append")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--worker-mode", choices=("legacy", "streaming"))
    parser.add_argument("--worker-timesteps", type=int)
    parser.add_argument("--worker-output", type=Path)
    args = parser.parse_args()

    if args.worker_mode is not None:
        _run_worker_from_args(args)
        return

    requested_timesteps = (
        _DEFAULT_TIMESTEPS if args.timesteps is None else tuple(args.timesteps)
    )
    evidence = run_benchmark(timesteps=requested_timesteps)
    rendered = json.dumps(evidence, sort_keys=True, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
