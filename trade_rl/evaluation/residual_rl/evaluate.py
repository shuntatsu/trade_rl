"""Episode rollouts, per-episode metrics and the pre-declared decision rules."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import numpy as np

from trade_rl.evaluation.residual_rl.env import (
    EnvConfig,
    ResidualAllocatorEnv,
)
from trade_rl.evaluation.residual_rl.market import Regime
from trade_rl.evaluation.residual_rl.protocol import (
    DecisionRules,
    ResidualProtocol,
)

ActionChooser = Callable[[np.ndarray], int]


@dataclass(frozen=True, slots=True)
class EpisodeMetrics:
    market_seed: int
    net_log_return: float
    mean_daily_turnover: float
    cost_paid: float
    max_drawdown: float
    trade_days: int
    mean_multiplier: float
    final_equity: float


def env_config(
    protocol: ResidualProtocol,
    regime: Regime,
    *,
    kappa: float | None = None,
    reward_scale: float = 1.0,
) -> EnvConfig:
    """Env configuration for one regime; ``kappa`` overrides the allocator's."""

    allocator = protocol.allocator
    if kappa is not None:
        allocator = replace(allocator, kappa=kappa)
    return EnvConfig(
        market=protocol.market,
        regime=regime,
        allocator=allocator,
        costs=protocol.costs,
        scales=protocol.scales,
        multipliers=protocol.multipliers,
        reward_scale=reward_scale,
    )


def multiplier_index(protocol: ResidualProtocol, multiplier: float) -> int:
    return protocol.multipliers.index(multiplier)


def episode_metrics(env: ResidualAllocatorEnv, market_seed: int) -> EpisodeMetrics:
    """Metrics recomputed from the env's append-only ledger rows."""

    rows = env.ledger
    if not rows:
        raise ValueError("episode has no ledger rows")
    equity = np.asarray([row["equity_after"] for row in rows])
    before = np.asarray([row["equity_before"] for row in rows])
    traded = np.asarray([row["traded_value"] for row in rows])
    costs = np.asarray([row["fee"] + row["spread_cost"] for row in rows])
    peak = np.maximum.accumulate(np.concatenate(([env.initial_equity], equity)))[1:]
    drawdown = float(np.max(1.0 - equity / peak))
    return EpisodeMetrics(
        market_seed=market_seed,
        net_log_return=float(sum(row["log_return"] for row in rows)),
        mean_daily_turnover=float(np.mean(traded / before)),
        cost_paid=float(costs.sum() / env.initial_equity),
        max_drawdown=drawdown,
        trade_days=int(np.sum(traded > 1e-12)),
        mean_multiplier=float(np.mean([row["multiplier"] for row in rows])),
        final_equity=float(equity[-1]),
    )


def run_episode(
    config: EnvConfig, market_seed: int, chooser: ActionChooser
) -> tuple[EpisodeMetrics, list[int]]:
    env = ResidualAllocatorEnv(config, (market_seed,))
    obs, _ = env.reset(options={"market_seed": market_seed})
    actions: list[int] = []
    done = False
    while not done:
        action = int(chooser(obs))
        actions.append(action)
        obs, _, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
    return episode_metrics(env, market_seed), actions


def constant_chooser(index: int) -> ActionChooser:
    def choose(_: np.ndarray) -> int:
        return index

    return choose


def evaluate_chooser(
    config: EnvConfig, seeds: Sequence[int], chooser: ActionChooser
) -> tuple[list[EpisodeMetrics], np.ndarray]:
    """Episodes on ``seeds`` plus the histogram of chosen action indexes."""

    histogram = np.zeros(len(config.multipliers), dtype=np.int64)
    metrics: list[EpisodeMetrics] = []
    for seed in seeds:
        episode, actions = run_episode(config, seed, chooser)
        metrics.append(episode)
        for action in actions:
            histogram[action] += 1
    return metrics, histogram


def aggregate(metrics: Sequence[EpisodeMetrics]) -> dict[str, float]:
    returns = np.asarray([m.net_log_return for m in metrics])
    turnover = np.asarray([m.mean_daily_turnover for m in metrics])
    return {
        "n_episodes": float(len(metrics)),
        "median_net_log_return": float(np.median(returns)),
        "mean_net_log_return": float(np.mean(returns)),
        "std_net_log_return": float(np.std(returns)),
        "median_daily_turnover": float(np.median(turnover)),
        "median_cost_paid": float(np.median([m.cost_paid for m in metrics])),
        "median_max_drawdown": float(np.median([m.max_drawdown for m in metrics])),
        "mean_trade_days": float(np.mean([m.trade_days for m in metrics])),
    }


def paired_bootstrap_median_lower(
    differences: np.ndarray, *, resamples: int, seed: int, quantile: float
) -> float:
    """Lower ``quantile`` of the bootstrap distribution of the median difference."""

    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2 or not np.all(np.isfinite(values)):
        raise ValueError("differences must be a finite vector of length >= 2")
    rng = np.random.default_rng(seed)
    index = rng.integers(0, values.size, size=(resamples, values.size))
    medians = np.median(values[index], axis=1)
    return float(np.quantile(medians, quantile))


def primary_verdict(
    residual_returns: np.ndarray,
    allocator_returns: np.ndarray,
    rules: DecisionRules,
) -> dict[str, object]:
    """F1: planted-alpha market, ``residual_returns`` is ``(n_ppo_seeds, n_seeds)``."""

    residual = np.asarray(residual_returns, dtype=np.float64)
    allocator = np.asarray(allocator_returns, dtype=np.float64)
    if residual.ndim != 2 or allocator.shape != (residual.shape[1],):
        raise ValueError("returns must be (ppo_seeds, seeds) and (seeds,)")
    differences = residual - allocator[None, :]
    per_ppo_seed = np.median(differences, axis=1)
    pooled = differences.mean(axis=0)
    lower = paired_bootstrap_median_lower(
        pooled,
        resamples=rules.bootstrap_resamples,
        seed=rules.bootstrap_seed,
        quantile=rules.bootstrap_quantile,
    )
    median_delta = float(np.median(per_ppo_seed))
    margin_ok = median_delta >= rules.margin
    every_seed_positive = bool(np.all(per_ppo_seed > 0.0))
    bootstrap_ok = lower > 0.0
    return {
        "per_ppo_seed_median_advantage": [float(v) for v in per_ppo_seed],
        "median_of_per_ppo_seed_advantage": median_delta,
        "margin": rules.margin,
        "margin_met": bool(margin_ok),
        "every_ppo_seed_positive": every_seed_positive,
        "bootstrap_median_lower_bound": lower,
        "bootstrap_lower_bound_positive": bool(bootstrap_ok),
        "fraction_seeds_improved": float(np.mean(pooled > 0.0)),
        "passed": bool(margin_ok and every_seed_positive and bootstrap_ok),
    }


def control_verdict(
    residual_turnover: np.ndarray,
    residual_returns: np.ndarray,
    allocator_turnover: np.ndarray,
    allocator_returns: np.ndarray,
    rules: DecisionRules,
) -> dict[str, object]:
    """F2: zero-alpha fee-only market; arrays are ``(n_ppo_seeds, n_seeds)`` / ``(n_seeds,)``."""

    res_turn = np.median(np.asarray(residual_turnover, dtype=np.float64), axis=1)
    res_ret = np.median(np.asarray(residual_returns, dtype=np.float64), axis=1)
    alloc_turn = float(np.median(allocator_turnover))
    alloc_ret = float(np.median(allocator_returns))
    median_turn = float(np.median(res_turn))
    median_ret = float(np.median(res_ret))
    ratio = median_turn / alloc_turn if alloc_turn > 0.0 else math.inf
    low_turnover = ratio <= rules.control_turnover_ratio
    loses_less = median_ret > alloc_ret
    return {
        "allocator_median_daily_turnover": alloc_turn,
        "allocator_median_net_log_return": alloc_ret,
        "per_ppo_seed_median_daily_turnover": [float(v) for v in res_turn],
        "per_ppo_seed_median_net_log_return": [float(v) for v in res_ret],
        "median_turnover_ratio_to_allocator": ratio,
        "turnover_ratio_limit": rules.control_turnover_ratio,
        "turnover_ratio_met": bool(low_turnover),
        "loses_less_than_allocator": bool(loses_less),
        "passed": bool(low_turnover and loses_less),
    }


def final_label(
    primary_passed: bool, control_passed: bool, rules: DecisionRules
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if not primary_passed:
        reasons.append("F1_primary_advantage_not_established")
    if not control_passed:
        reasons.append("F2_fee_only_learnability_control_failed")
    return (rules.label_fail if reasons else rules.label_pass), reasons


__all__ = [
    "ActionChooser",
    "EpisodeMetrics",
    "aggregate",
    "constant_chooser",
    "control_verdict",
    "env_config",
    "episode_metrics",
    "evaluate_chooser",
    "final_label",
    "multiplier_index",
    "paired_bootstrap_median_lower",
    "primary_verdict",
    "run_episode",
]
