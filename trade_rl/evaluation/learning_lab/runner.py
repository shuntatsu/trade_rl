"""Plan, train and evaluate one learning-lab method on one stage.

The runner is data-source agnostic: it receives a :class:`LabSetup` (a ``MarketDataset``, the
anchor signals, costs, risk, splits) and never loads or fetches anything. The real-data gate
(the owner G4 authorization comment) is the tool layer's, ``tools/learning_lab.py``.

Rules enforced here for every setup:

* every row of the dataset is before :data:`splits.DATA_CUTOFF` (2026-08-01);
* agents train on ``train`` rows only (the environment's stop index is the split's last row);
* ``test`` is evaluated only for a ``(method, seed, config)`` that already has a completed
  ``validate`` trial in the registry, and every evaluation is one registry line.
"""

from __future__ import annotations

import hashlib
import math
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.data.market import MarketDataset
from trade_rl.evaluation.agy_anchor_oos import protocol as anchor_protocol
from trade_rl.evaluation.learning_lab import agents, baselines, splits
from trade_rl.evaluation.learning_lab.metrics import (
    RolloutResult,
    summarize_returns,
)
from trade_rl.evaluation.learning_lab.overlay_env import (
    ExecutionOverlayEnv,
    OverlayConfig,
    PerturbationSource,
)
from trade_rl.evaluation.learning_lab.registry import (
    TrialRecord,
    TrialRegistry,
    registry_deflated_sharpe,
)
from trade_rl.evaluation.learning_lab.synthetic import BAR
from trade_rl.risk.pretrade import PreTradeRiskConfig
from trade_rl.simulation.execution import ExecutionCostConfig
from trade_rl.strategies.agy_anchor.signals import AnchorSignals

STUDY_ID: Final = "anchor_rl_execution_dev_v1"
PROTOCOL_SCHEMA: Final = "learning_lab_protocol_v1"
MIN_SEEDS_BEATING: Final = 4
MAX_VALIDATION_DRAWDOWN: Final = 0.25
LABEL_PROMISING: Final = "DEV_PROMISING"
LABEL_STOP: Final = "DEV_STOP"


class RunnerError(RuntimeError):
    """A run was refused or inconsistent."""


def method_ids() -> tuple[str, ...]:
    learned = tuple(agents.method_id_for(name) for name in agents.ALGORITHMS)
    return (*learned, *baselines.baseline_methods())


def protocol_payload() -> dict[str, Any]:
    """Every outcome-relevant constant of the harness (bound by :func:`protocol_digest`)."""
    return {
        "schema": PROTOCOL_SCHEMA,
        "study_id": STUDY_ID,
        "splits": splits.splits_payload(),
        "data_cutoff": str(splits.DATA_CUTOFF),
        "overlay": OverlayConfig().payload(),
        "observation_bar_clock": "uniform dataset-native interval, not decision stride",
        "run_identity": "config digest includes actual initial_capital",
        "statistics_validity": "finite typed DEV metrics; finite numeric DSR and trial variance",
        "methods": list(method_ids()),
        "fixed_strides": list(baselines.FIXED_STRIDES),
        "reward": "net_equity_increment of the shared-portfolio engine, gamma 1",
        "anchor_cash_limit": anchor_protocol.CASH_LIMIT,
        "anchor_initial_margin_rate": anchor_protocol.INITIAL_MARGIN_RATE,
        "risk": asdict(anchor_protocol.risk_config()),
        "cost_base": asdict(anchor_protocol.scenario_cost("BASE")),
        "decision_rule": {
            "min_seeds_beating": MIN_SEEDS_BEATING,
            "max_validation_drawdown": MAX_VALIDATION_DRAWDOWN,
            "halves": 2,
        },
    }


def protocol_digest() -> str:
    return hashlib.sha256(canonical_json_bytes(protocol_payload())).hexdigest()


@dataclass(frozen=True)
class LabSetup:
    """Everything a run needs; built by the caller (synthetic builder or tool)."""

    dataset: MarketDataset
    signals: AnchorSignals
    execution_cost: ExecutionCostConfig
    risk_config: PreTradeRiskConfig
    splits: Mapping[str, splits.WalkForwardSplit]
    initial_capital: float = 100_000.0
    signal_lag_bars: int = 1
    data_kind: str = "synthetic"

    def __post_init__(self) -> None:
        if (
            type(self.initial_capital) not in (int, float)
            or not math.isfinite(self.initial_capital)
            or self.initial_capital <= 0
        ):
            raise RunnerError("initial_capital must be finite and positive")
        if self.dataset.native_clock is not None:
            raise RunnerError(
                "legacy learning-lab protocol does not admit native fixed execution"
            )
        splits.require_before_cutoff(self.dataset.timestamps)
        splits.validate_split_set(dict(self.splits))

    def window(self, split: str) -> tuple[int, int]:
        return splits.split_indices(self.dataset.timestamps, dict(self.splits))[split]

    @property
    def periods_per_year(self) -> int:
        periods = self.dataset.periods_per_year
        if type(periods) is not int:
            raise RunnerError(
                "legacy learning-lab protocol requires integer annual frequency"
            )
        return periods


def synthetic_splits(
    n_bars: int, first_stamp: str = "2022-01-01T00:00:00"
) -> dict[str, splits.WalkForwardSplit]:
    """Contiguous 50/25/25 percent windows over a synthetic 4h grid (dry runs only)."""
    origin = np.datetime64(first_stamp, "ns")
    a, b = n_bars // 2, n_bars // 2 + n_bars // 4
    edges = [origin + i * BAR for i in (0, a, b, n_bars)]
    return {
        name: splits.WalkForwardSplit(name=name, start=edges[i], stop=edges[i + 1])
        for i, name in enumerate(splits.SPLIT_NAMES)
    }


def plan(methods: Sequence[str], seeds: Sequence[int]) -> list[dict[str, Any]]:
    """The runs a study would make; reads nothing and runs nothing."""
    known = set(method_ids())
    unknown = [m for m in methods if m not in known]
    if unknown:
        raise RunnerError(f"unknown methods: {unknown}")
    runs: list[dict[str, Any]] = []
    for method in methods:
        learned = method in {agents.method_id_for(a) for a in agents.ALGORITHMS}
        for seed in seeds if learned else (0,):
            runs.append(
                {
                    "method": method,
                    "seed": seed,
                    "kind": "agent" if learned else "baseline",
                    "algorithm": agents.algorithm_for_method(method)
                    if learned
                    else None,
                    "train": splits.TRAIN if learned else None,
                    "evaluate": [splits.VALIDATE, splits.TEST],
                }
            )
    return runs


def overlay_config_for(algorithm: str, setup: LabSetup) -> OverlayConfig:
    return OverlayConfig(
        action_mode=agents.action_mode_for(algorithm),
        initial_capital=setup.initial_capital,
    )


def make_env(
    setup: LabSetup,
    split: str,
    algorithm: str,
    *,
    perturbation: PerturbationSource | None = None,
    window: tuple[int, int] | None = None,
) -> ExecutionOverlayEnv:
    first, last = setup.window(split) if window is None else window
    return ExecutionOverlayEnv(
        setup.dataset,
        start_index=first,
        stop_index=last,
        base_factory=baselines.anchor_stack_factory(
            setup.signals, start_index=first, signal_lag_bars=setup.signal_lag_bars
        ),
        execution_cost=setup.execution_cost,
        risk_config=setup.risk_config,
        config=overlay_config_for(algorithm, setup),
        perturbation=perturbation,
    )


def config_digest(
    setup: LabSetup, agent: agents.AgentConfig | None, method: str
) -> str:
    payload = {
        "protocol_digest": protocol_digest(),
        "method": method,
        "agent": None if agent is None else agent.payload(),
        "splits": splits.splits_payload(dict(setup.splits)),
        "dataset_id": setup.dataset.dataset_id,
        "data_kind": setup.data_kind,
        "initial_capital": float(setup.initial_capital),
        "signal_lag_bars": setup.signal_lag_bars,
        "cost": asdict(setup.execution_cost),
        "risk": asdict(setup.risk_config),
    }
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def _half_metrics(result: RolloutResult, ppy: int, n_first: int) -> dict[str, Any]:
    first = result.bar_returns[:n_first]
    second = result.bar_returns[n_first:]
    return {
        "half1": summarize_returns(first, periods_per_year=ppy),
        "half2": summarize_returns(second, periods_per_year=ppy),
    }


def metrics_of(
    result: RolloutResult, setup: LabSetup, window: tuple[int, int]
) -> dict[str, Any]:
    first, last = window
    out = result.metrics(periods_per_year=setup.periods_per_year)
    (h1_first, h1_last), _ = splits.validation_halves(first, last)
    out.update(_half_metrics(result, setup.periods_per_year, h1_last - h1_first + 1))
    return out


def run_baseline_method(
    setup: LabSetup, method: str, split: str
) -> tuple[RolloutResult, dict[str, Any]]:
    window = setup.window(split)
    result = baselines.run_baseline(
        method,
        setup.dataset,
        setup.signals,
        start_index=window[0],
        stop_index=window[1],
        execution_cost=setup.execution_cost,
        risk_config=setup.risk_config,
        initial_capital=setup.initial_capital,
        signal_lag_bars=setup.signal_lag_bars,
    )
    return result, metrics_of(result, setup, window)


def train_method(
    setup: LabSetup,
    method: str,
    seed: int,
    *,
    agent_config: agents.AgentConfig,
    output_dir: Path | str,
    registry: TrialRegistry,
    code_commit: str,
    perturbation: PerturbationSource | None = None,
) -> tuple[agents.TrainedAgent, TrialRecord]:
    """Train on the ``train`` split, register the trial and return the checkpoint."""
    algorithm = agents.algorithm_for_method(method)
    if agent_config.algorithm != algorithm or agent_config.seed != seed:
        raise RunnerError("agent_config does not match the method and seed")
    env = make_env(setup, splits.TRAIN, algorithm, perturbation=perturbation)
    digest = config_digest(setup, agent_config, method)
    try:
        trained = agents.train_agent(env, agent_config, output_dir)
        model = agents.load_agent(
            algorithm, trained.checkpoint, sha256=trained.checkpoint_sha256
        )
        eval_env = make_env(setup, splits.TRAIN, algorithm)
        result = agents.evaluate_agent(model, eval_env, method=method, seed=seed)
    except Exception as error:
        registry.append(
            TrialRecord(
                method_id=method,
                config_digest=digest,
                seed=seed,
                split=splits.TRAIN,
                code_commit=code_commit,
                metrics={},
                status="failed",
                stop_reason=f"{type(error).__name__}: {error}"[:300],
            )
        )
        raise
    metrics = metrics_of(result, setup, setup.window(splits.TRAIN))
    metrics["checkpoint_sha256"] = trained.checkpoint_sha256
    metrics["total_timesteps"] = agent_config.total_timesteps
    record = TrialRecord(
        method_id=method,
        config_digest=digest,
        seed=seed,
        split=splits.TRAIN,
        code_commit=code_commit,
        metrics=metrics,
        status="completed",
    )
    registry.append(record)
    return trained, record


def evaluate_method(
    setup: LabSetup,
    method: str,
    seed: int,
    split: str,
    *,
    registry: TrialRegistry,
    code_commit: str,
    agent_config: agents.AgentConfig | None = None,
    checkpoint: Path | str | None = None,
    checkpoint_sha256: str | None = None,
) -> TrialRecord:
    """Evaluate a baseline or a trained agent on ``validate`` or ``test`` and register it."""
    if split not in (splits.VALIDATE, splits.TEST):
        raise RunnerError("evaluate runs on the validate or test split only")
    learned = method not in baselines.baseline_methods()
    digest = config_digest(setup, agent_config if learned else None, method)
    if split == splits.TEST:
        seen = [
            r
            for r in registry.records()
            if r.method_id == method
            and r.seed == seed
            and r.config_digest == digest
            and r.split == splits.VALIDATE
            and r.status == "completed"
        ]
        if not seen:
            raise RunnerError(
                "test is evaluated only after a completed validate trial of the same "
                "method, seed and configuration"
            )
    window = setup.window(split)
    if learned:
        if agent_config is None or checkpoint is None or checkpoint_sha256 is None:
            raise RunnerError(
                "a learned method needs its config, checkpoint and sha256"
            )
        algorithm = agents.algorithm_for_method(method)
        model = agents.load_agent(algorithm, checkpoint, sha256=checkpoint_sha256)
        env = make_env(setup, split, algorithm)
        result = agents.evaluate_agent(model, env, method=method, seed=seed)
    else:
        result, _ = run_baseline_method(setup, method, split)
    metrics = metrics_of(result, setup, window)
    record = TrialRecord(
        method_id=method,
        config_digest=digest,
        seed=seed,
        split=split,
        code_commit=code_commit,
        metrics=metrics,
        status="completed",
    )
    registry.append(record)
    return record


def _dev_stat(row: Mapping[str, Any], field: str, context: str) -> float:
    """Fail closed on missing/invalid metrics before comparing development arms."""
    try:
        value = row[field]
    except (TypeError, KeyError) as error:
        raise RunnerError(f"{context}.{field} is missing") from error
    if type(value) not in (float, int):
        raise RunnerError(f"{context}.{field} must be a finite number")
    try:
        numeric = float(value)
    except OverflowError as error:
        raise RunnerError(f"{context}.{field} must be finite") from error
    if not math.isfinite(numeric):
        raise RunnerError(f"{context}.{field} must be finite")
    return numeric


def _dev_halves(row: Mapping[str, Any], context: str) -> dict[str, float]:
    halves: dict[str, float] = {}
    for key in ("half1", "half2"):
        subrow = row.get(key)
        if not isinstance(subrow, Mapping):
            raise RunnerError(f"{context}.{key} is not a metrics map")
        halves[key] = _dev_stat(subrow, "sharpe_per_period", f"{context}.{key}")
    return halves


def dev_decision(
    agent_validation: Mapping[int, Mapping[str, Any]],
    baseline_validation: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the frozen DEV comparison after checking every compared scalar."""
    strides = [
        m
        for m in baseline_validation
        if m in {baselines.stride_method_id(s) for s in baselines.FIXED_STRIDES}
    ]
    if not strides or not agent_validation:
        raise RunnerError("the rule needs fixed-stride baselines and agent seeds")
    baseline_sharpes = {
        name: _dev_stat(baseline_validation[name], "sharpe_per_period", name)
        for name in strides
    }
    baseline_halves = {
        name: _dev_halves(baseline_validation[name], name) for name in strides
    }
    sharpes = {
        seed: _dev_stat(row, "sharpe_per_period", f"seed {seed}")
        for seed, row in agent_validation.items()
    }
    drawdowns = {
        seed: _dev_stat(row, "max_drawdown", f"seed {seed}")
        for seed, row in agent_validation.items()
    }
    agent_halves = {
        seed: _dev_halves(row, f"seed {seed}") for seed, row in agent_validation.items()
    }
    best = max(strides, key=lambda name: baseline_sharpes[name])
    seeds = sorted(agent_validation)
    median_sharpe = statistics.median(sharpes.values())
    if not math.isfinite(median_sharpe):
        raise RunnerError("median validation Sharpe must be finite")
    median_seed = min(seeds, key=lambda s: (abs(sharpes[s] - median_sharpe), s))
    beating = [seed for seed in seeds if sharpes[seed] > baseline_sharpes[best]]
    halves = all(
        agent_halves[median_seed][half] > baseline_halves[best][half]
        for half in ("half1", "half2")
    )
    worst_dd = max(drawdowns.values())
    checks = {
        "median_beats_best_stride": median_sharpe > baseline_sharpes[best],
        "seeds_beating": len(beating) >= MIN_SEEDS_BEATING,
        "both_halves": halves,
        "drawdown_below_limit": worst_dd < MAX_VALIDATION_DRAWDOWN,
    }
    return {
        "label": LABEL_PROMISING if all(checks.values()) else LABEL_STOP,
        "checks": checks,
        "best_stride": best,
        "median_seed": median_seed,
        "seeds_beating": beating,
    }


def deflated_for(records: Sequence[TrialRecord]) -> dict[str, float]:
    """Compute DSR from one snapshot; reject an ineligible terminal trial."""
    if not records:
        raise ValueError("deflated Sharpe requires at least one registered trial")
    record = records[-1]
    if record.status != "completed":
        raise ValueError("deflated Sharpe requires a completed last trial")
    required = ("sharpe_per_period", "n_obs", "skewness", "kurtosis")
    missing = [field for field in required if field not in record.metrics]
    if missing:
        raise ValueError(f"completed trial is missing metrics: {', '.join(missing)}")
    # The registry counts every failed trial, but a *headline* deflated Sharpe
    # requires the last trial to carry a valid completed return distribution.
    # In particular 4.9 observations must not silently become four.
    count = record.metrics["n_obs"]
    if type(count) is not int or count < 2:
        raise ValueError("n_obs must be an integer of at least two")
    try:
        representable_count = float(count)
    except OverflowError as error:
        raise ValueError("n_obs is outside the finite numeric range") from error
    if not math.isfinite(representable_count):
        raise ValueError("n_obs is outside the finite numeric range")
    moments: dict[str, float] = {}
    for name in ("sharpe_per_period", "skewness", "kurtosis"):
        value = record.metrics[name]
        if type(value) not in (int, float):
            raise ValueError(f"{name} must be a finite number")
        try:
            numeric = float(value)
        except OverflowError as error:
            raise ValueError(f"{name} must be a finite number") from error
        if not math.isfinite(numeric):
            raise ValueError(f"{name} must be a finite number")
        moments[name] = numeric
    try:
        return registry_deflated_sharpe(
            records,
            sharpe=moments["sharpe_per_period"],
            n_obs=count,
            skewness=moments["skewness"],
            kurtosis=moments["kurtosis"],
        )
    except OverflowError as error:
        # A canonical JSON number can be finite yet overflow floating-point
        # Sharpe arithmetic; the CLI must refuse, not raise an uncaught error.
        raise ValueError("deflated Sharpe numerical overflow") from error


__all__ = [
    "LABEL_PROMISING",
    "LABEL_STOP",
    "PROTOCOL_SCHEMA",
    "STUDY_ID",
    "LabSetup",
    "RunnerError",
    "config_digest",
    "deflated_for",
    "dev_decision",
    "evaluate_method",
    "make_env",
    "method_ids",
    "metrics_of",
    "plan",
    "protocol_digest",
    "protocol_payload",
    "run_baseline_method",
    "synthetic_splits",
    "train_method",
]
