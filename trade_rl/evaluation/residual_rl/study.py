"""Run the frozen study and assemble the result and provenance records.

Modes (every invocation is appended to the provenance file by the CLI):

* ``calibration``: allocator-only baselines on the calibration seeds. No PPO.
* ``smoke``: a tiny PPO pipeline check on calibration seeds. It is not evidence.
* ``full``: the frozen study. Trains one PPO agent per (regime, PPO seed) on the
  training pool and evaluates every policy on the disjoint held-out pool.
"""

from __future__ import annotations

from collections.abc import Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np

from trade_rl.evaluation.residual_rl.evaluate import (
    EpisodeMetrics,
    aggregate,
    constant_chooser,
    control_verdict,
    env_config,
    evaluate_chooser,
    final_label,
    multiplier_index,
    primary_verdict,
)
from trade_rl.evaluation.residual_rl.market import Regime
from trade_rl.evaluation.residual_rl.protocol import (
    ResidualProtocol,
    protocol_payload,
    require_frozen_protocol,
)
from trade_rl.evaluation.residual_rl.train import (
    model_chooser,
    runtime_versions,
    train_residual_ppo,
)

BASELINE_NAMES = (
    "CASH",
    "ALLOCATOR_KAPPA_1",
    "ALLOCATOR_KAPPA_3",
    "STATIC_M_0.5",
    "STATIC_M_1.5",
)


@dataclass(frozen=True, slots=True)
class PolicyEvaluation:
    name: str
    metrics: tuple[EpisodeMetrics, ...]
    action_histogram: tuple[int, ...]

    def returns(self) -> np.ndarray:
        return np.asarray([m.net_log_return for m in self.metrics])

    def turnover(self) -> np.ndarray:
        return np.asarray([m.mean_daily_turnover for m in self.metrics])


def evaluate_baselines(
    protocol: ResidualProtocol, regime: Regime, seeds: Sequence[int]
) -> dict[str, PolicyEvaluation]:
    """CASH, allocator alone (kappa 1 and 3) and the two static throttles."""

    specs: tuple[tuple[str, float | None, float], ...] = (
        ("CASH", None, 0.0),
        ("ALLOCATOR_KAPPA_1", protocol.kappa_primary, 1.0),
        ("ALLOCATOR_KAPPA_3", protocol.kappa_comparison, 1.0),
        ("STATIC_M_0.5", protocol.kappa_primary, 0.5),
        ("STATIC_M_1.5", protocol.kappa_primary, 1.5),
    )
    out: dict[str, PolicyEvaluation] = {}
    for name, kappa, multiplier in specs:
        config = env_config(protocol, regime, kappa=kappa)
        index = multiplier_index(protocol, multiplier)
        metrics, histogram = evaluate_chooser(config, seeds, constant_chooser(index))
        out[name] = PolicyEvaluation(
            name, tuple(metrics), tuple(int(v) for v in histogram)
        )
    return out


def fit_and_evaluate(
    protocol: ResidualProtocol,
    regime: Regime,
    ppo_seed: int,
    seeds: tuple[int, ...],
    total_timesteps: int | None,
) -> PolicyEvaluation:
    """Train one agent and evaluate its deterministic policy on ``seeds``."""

    model = train_residual_ppo(
        protocol, regime, ppo_seed, total_timesteps=total_timesteps
    )
    config = env_config(protocol, regime)
    chooser = model_chooser(model, deterministic=protocol.ppo.deterministic_evaluation)
    metrics, histogram = evaluate_chooser(config, seeds, chooser)
    return PolicyEvaluation(
        f"RESIDUAL_PPO_seed{ppo_seed}",
        tuple(metrics),
        tuple(int(v) for v in histogram),
    )


def _fit_task(args: tuple[Any, ...]) -> tuple[str, int, PolicyEvaluation]:
    protocol, regime_value, ppo_seed, seeds, total_timesteps = args
    regime = Regime(regime_value)
    result = fit_and_evaluate(protocol, regime, ppo_seed, seeds, total_timesteps)
    return regime.value, ppo_seed, result


def _fit_all(
    protocol: ResidualProtocol,
    seeds: tuple[int, ...],
    total_timesteps: int | None,
    ppo_seeds: Sequence[int],
    workers: int,
) -> dict[str, dict[int, PolicyEvaluation]]:
    tasks = [
        (protocol, regime.value, ppo_seed, seeds, total_timesteps)
        for regime in (Regime.PLANTED_ALPHA, Regime.ZERO_ALPHA)
        for ppo_seed in ppo_seeds
    ]
    results: dict[str, dict[int, PolicyEvaluation]] = {
        Regime.PLANTED_ALPHA.value: {},
        Regime.ZERO_ALPHA.value: {},
    }
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for regime_value, ppo_seed, evaluation in pool.map(_fit_task, tasks):
                results[regime_value][ppo_seed] = evaluation
    else:
        for task in tasks:
            regime_value, ppo_seed, evaluation = _fit_task(task)
            results[regime_value][ppo_seed] = evaluation
    return results


def _evaluation_record(evaluation: PolicyEvaluation) -> dict[str, Any]:
    return {
        "aggregate": aggregate(evaluation.metrics),
        "action_histogram": list(evaluation.action_histogram),
        "net_log_return_by_seed": [m.net_log_return for m in evaluation.metrics],
        "daily_turnover_by_seed": [m.mean_daily_turnover for m in evaluation.metrics],
    }


def run_calibration(protocol: ResidualProtocol) -> dict[str, Any]:
    """Allocator-only baselines on the calibration seeds (no PPO, not evidence)."""

    seeds = protocol.seeds.calibration
    record: dict[str, Any] = {"seeds": list(seeds), "regimes": {}}
    for regime in (Regime.PLANTED_ALPHA, Regime.ZERO_ALPHA):
        baselines = evaluate_baselines(protocol, regime, seeds)
        record["regimes"][regime.value] = {
            name: aggregate(evaluation.metrics)
            for name, evaluation in baselines.items()
        }
    return record


def run_smoke(
    protocol: ResidualProtocol, *, total_timesteps: int, n_seeds: int
) -> dict[str, Any]:
    """Tiny end-to-end pipeline check on calibration seeds. Not evidence."""

    seeds = protocol.seeds.calibration[:n_seeds]
    fits = _fit_all(protocol, seeds, total_timesteps, protocol.ppo.seeds[:1], 1)
    record: dict[str, Any] = {
        "seeds": list(seeds),
        "total_timesteps": total_timesteps,
        "regimes": {},
    }
    for regime_value, by_seed in fits.items():
        record["regimes"][regime_value] = {
            f"seed{seed}": _evaluation_record(evaluation)["aggregate"]
            for seed, evaluation in by_seed.items()
        }
    return record


def run_full_study(protocol: ResidualProtocol, *, workers: int) -> dict[str, Any]:
    """The frozen study. Raises unless ``protocol`` equals the frozen digest."""

    digest = require_frozen_protocol(protocol)
    held_out = protocol.seeds.held_out
    ppo_seeds = protocol.ppo.seeds
    fits = _fit_all(protocol, held_out, None, ppo_seeds, workers)

    regimes: dict[str, Any] = {}
    evaluations: dict[str, dict[str, PolicyEvaluation]] = {}
    for regime in (Regime.PLANTED_ALPHA, Regime.ZERO_ALPHA):
        baselines = evaluate_baselines(protocol, regime, held_out)
        residual = {
            f"RESIDUAL_PPO_seed{seed}": fits[regime.value][seed] for seed in ppo_seeds
        }
        evaluations[regime.value] = {**baselines, **residual}
        regimes[regime.value] = {
            name: _evaluation_record(evaluation)
            for name, evaluation in evaluations[regime.value].items()
        }

    def stack(regime: Regime, attribute: str) -> np.ndarray:
        return np.stack(
            [
                getattr(evaluations[regime.value][f"RESIDUAL_PPO_seed{s}"], attribute)()
                for s in ppo_seeds
            ]
        )

    planted_alloc = evaluations[Regime.PLANTED_ALPHA.value]["ALLOCATOR_KAPPA_1"]
    zero_alloc = evaluations[Regime.ZERO_ALPHA.value]["ALLOCATOR_KAPPA_1"]
    primary = primary_verdict(
        stack(Regime.PLANTED_ALPHA, "returns"), planted_alloc.returns(), protocol.rules
    )
    control = control_verdict(
        stack(Regime.ZERO_ALPHA, "turnover"),
        stack(Regime.ZERO_ALPHA, "returns"),
        zero_alloc.turnover(),
        zero_alloc.returns(),
        protocol.rules,
    )
    label, reasons = final_label(
        bool(primary["passed"]), bool(control["passed"]), protocol.rules
    )

    planted = evaluations[Regime.PLANTED_ALPHA.value]
    static_names = (
        "ALLOCATOR_KAPPA_1",
        "ALLOCATOR_KAPPA_3",
        "STATIC_M_0.5",
        "STATIC_M_1.5",
    )
    best_static = max(
        static_names, key=lambda n: float(np.median(planted[n].returns()))
    )
    residual_median = float(
        np.median(np.median(stack(Regime.PLANTED_ALPHA, "returns"), axis=1))
    )
    diagnostics = {
        "best_static_policy_planted_alpha": best_static,
        "best_static_median_net_log_return": float(
            np.median(planted[best_static].returns())
        ),
        "residual_median_of_median_net_log_return": residual_median,
        "residual_beats_best_static": bool(
            residual_median > float(np.median(planted[best_static].returns()))
        ),
        "note": "diagnostic only; not part of the pre-declared label",
    }
    return {
        "schema": "issue810_residual_ppo_synthetic_results_v1",
        "label": label,
        "label_reasons": reasons,
        "label_scope": (
            "SYNTHETIC evidence only. Never a real-data profitability claim; "
            "real-data RL remains gated behind the fee-only learnability check."
        ),
        "protocol_sha256": digest,
        "protocol": protocol_payload(protocol),
        "held_out_seeds": list(held_out),
        "primary_planted_alpha": primary,
        "control_zero_alpha_fee_only": control,
        "diagnostics": diagnostics,
        "regimes": regimes,
        "runtime": runtime_versions(),
    }


__all__ = [
    "BASELINE_NAMES",
    "PolicyEvaluation",
    "evaluate_baselines",
    "fit_and_evaluate",
    "run_calibration",
    "run_full_study",
    "run_smoke",
]
