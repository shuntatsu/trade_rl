from __future__ import annotations

from typing import Any, Optional

import numpy as np

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.trading.post_processor import BARS_PER_YEAR_1H


def _book_metrics(book, initial_capital: float) -> dict[str, float | int]:
    returns = np.asarray(book.returns_history, dtype=np.float64)
    sharpe = (
        float(returns.mean() / returns.std() * np.sqrt(BARS_PER_YEAR_1H))
        if returns.size and returns.std() > 0.0
        else 0.0
    )
    downside = np.minimum(returns, 0.0)
    downside_std = float(np.sqrt(np.mean(downside**2))) if returns.size else 0.0
    sortino = (
        float(returns.mean() / downside_std * np.sqrt(BARS_PER_YEAR_1H))
        if downside_std > 0.0
        else 0.0
    )
    return {
        "total_return": float(book.portfolio_value / initial_capital - 1.0),
        "sharpe": sharpe,
        "sortino": sortino,
        "max_drawdown": float(book.max_drawdown),
        "turnover_total": float(book.turnover_total),
        "total_cost": float(book.total_cost),
        "funding_pnl": float(book.funding_pnl),
        "n_trades": int(book.n_trades),
        "n_base_bars": int(len(returns)),
    }


def _moving_block_mean_test(
    differences: np.ndarray,
    *,
    n_bootstrap: int = 1_000,
    seed: int = 0,
) -> dict[str, float | int]:
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        return {"p_value": 1.0, "lower_ci": 0.0, "upper_ci": 0.0, "block_size": 1}
    observed = float(values.mean())
    if np.all(np.abs(values) <= 1e-15):
        return {"p_value": 1.0, "lower_ci": 0.0, "upper_ci": 0.0, "block_size": 1}
    n = len(values)
    block_size = max(1, min(n, int(np.ceil(n**0.5))))
    rng = np.random.default_rng(seed)
    means = np.empty(n_bootstrap, dtype=np.float64)
    for draw in range(n_bootstrap):
        sampled: list[int] = []
        while len(sampled) < n:
            start = int(rng.integers(0, max(1, n - block_size + 1)))
            sampled.extend(range(start, min(start + block_size, n)))
        means[draw] = float(values[np.asarray(sampled[:n])].mean())
    quantiles = np.asarray(np.quantile(means, [0.025, 0.975]), dtype=np.float64)
    lower = float(quantiles[0])
    upper = float(quantiles[1])
    if observed <= 0.0:
        p_value = 1.0
    else:
        centered = means - means.mean()
        p_value = float(np.mean(centered >= observed))
    return {
        "p_value": p_value,
        "lower_ci": lower,
        "upper_ci": upper,
        "block_size": block_size,
    }


def evaluate_relative_agent(
    agent,
    fs,
    *,
    env_kwargs: Optional[dict[str, Any]] = None,
    start_idx: int = 0,
    bootstrap_seed: int = 0,
) -> dict[str, Any]:
    kwargs = dict(env_kwargs or {})
    kwargs.pop("episode_bars", None)
    episode_bars = max(1, fs.n_bars - 2 - start_idx)
    env = BaselineResidualTradingEnv(fs, episode_bars=episode_bars, **kwargs)
    obs, _ = env.reset(options={"start_idx": start_idx})

    actions: list[np.ndarray] = []
    trend_mixes: list[float] = []
    alpha_budgets: list[float] = []
    stage_gross: dict[str, list[float]] = {
        "proposal": [],
        "htf_constrained": [],
        "executed": [],
    }
    done = False
    while not done:
        action, _ = agent.predict(obs, deterministic=True)
        action_array = np.asarray(action, dtype=np.float64).reshape(-1)
        obs, _, term, trunc, info = env.step(action_array)
        composition = info["composition"]
        actions.append(action_array.copy())
        trend_mixes.append(float(composition.trend_mix))
        alpha_budgets.append(float(composition.alpha_budget))
        stage_gross["proposal"].append(float(np.abs(composition.proposal).sum()))
        htf = info["hybrid_pp_info"].extra.get(
            "htf_constrained_weights", composition.proposal
        )
        stage_gross["htf_constrained"].append(float(np.abs(htf).sum()))
        stage_gross["executed"].append(float(np.abs(env.hybrid.weights).sum()))
        done = term or trunc

    hybrid_metrics = _book_metrics(env.hybrid, env.initial_capital)
    shadow_metrics = _book_metrics(env.shadow, env.initial_capital)
    hybrid_returns = np.asarray(env.hybrid.returns_history, dtype=np.float64)
    shadow_returns = np.asarray(env.shadow.returns_history, dtype=np.float64)
    differences = hybrid_returns - shadow_returns
    bootstrap = _moving_block_mean_test(differences, seed=bootstrap_seed)
    excess_log_return = float(
        np.log(env.hybrid.portfolio_value / env.initial_capital)
        - np.log(env.shadow.portfolio_value / env.initial_capital)
    )
    action_matrix = np.asarray(actions, dtype=np.float64)

    def stats(values: np.ndarray) -> dict[str, float]:
        if values.size == 0:
            return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
        return {
            "mean": float(values.mean()),
            "std": float(values.std()),
            "min": float(values.min()),
            "max": float(values.max()),
        }

    return {
        "identity": {
            "action_schema": "baseline_residual_v1",
            "shadow_baseline": "base_trend_v2",
        },
        "hybrid": hybrid_metrics,
        "shadow": shadow_metrics,
        "paired": {
            "excess_total_return": float(hybrid_metrics["total_return"])
            - float(shadow_metrics["total_return"]),
            "excess_log_return": excess_log_return,
            "mean_base_bar_excess": float(differences.mean())
            if differences.size
            else 0.0,
            **bootstrap,
        },
        "actions": {
            "count": int(len(actions)),
            "trend_mix": stats(
                action_matrix[:, 0] if action_matrix.size else np.array([])
            ),
            "alpha": stats(action_matrix[:, 1] if action_matrix.size else np.array([])),
            "alpha_budget": stats(np.asarray(alpha_budgets)),
            "contrarian_alpha_fraction": float(np.mean(np.asarray(alpha_budgets) < 0.0))
            if alpha_budgets
            else 0.0,
        },
        "weight_stages": {
            name: stats(np.asarray(values, dtype=np.float64))
            for name, values in stage_gross.items()
        },
        "execution": {
            "decision_every": env.decision_every,
            "decision_steps": len(actions),
            "base_bars_advanced": len(hybrid_returns),
            "annualization_factor": BARS_PER_YEAR_1H,
            "return_series": "base_bar",
        },
    }
