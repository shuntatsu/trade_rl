from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.pipeline.residual_pipeline import (
    FixedResidualAgent,
    IdentityResidualAgent,
    _evaluation_kwargs,
    _train_residual_ensemble,
    select_residual_configuration,
)
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily


@dataclass(frozen=True)
class ResidualCandidateSelection:
    development_results: dict[str, dict[str, Any]]
    development_cost2x_results: dict[str, dict[str, Any]]
    selection: dict[str, Any]
    selected_configuration: str
    selected_agent: object
    selected_policies: tuple[object, ...]
    selected_model_path: Path | None
    selected_alpha_enabled: bool


def train_select_residual_candidates(
    *,
    args,
    train_fs,
    val_fs,
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    env_kwargs: dict[str, Any],
    output: Path,
) -> ResidualCandidateSelection:
    """Train the preregistered A/B/C/D candidates using development data only."""

    output.mkdir(parents=True, exist_ok=True)
    identity = IdentityResidualAgent()
    fixed_alpha = FixedResidualAgent((0.0, 0.5))
    development_results: dict[str, dict[str, Any]] = {}
    development_cost2x_results: dict[str, dict[str, Any]] = {}

    a_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=False)
    development_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs=a_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["A"] = evaluate_relative_agent(
        identity,
        val_fs,
        env_kwargs={**a_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )

    b_agent, b_policies, b_model_path = _train_residual_ensemble(
        label="B_trend_mix",
        args=args,
        train_fs=train_fs,
        val_fs=val_fs,
        trend_family=trend_family,
        alpha=alpha,
        alpha_enabled=False,
        env_kwargs=env_kwargs,
        output=output,
    )
    b_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=False)
    development_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs=b_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["B"] = evaluate_relative_agent(
        b_agent,
        val_fs,
        env_kwargs={**b_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )

    d_agent: object | None = None
    d_policies: list[object] = []
    d_model_path: Path | None = None
    if alpha.enabled:
        c_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=True)
        development_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs=c_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            val_fs,
            env_kwargs={**c_kwargs, "cost_multiplier": 2.0},
            bootstrap_seed=args.seed,
        )
        d_agent, d_policies, d_model_path = _train_residual_ensemble(
            label="D_combined",
            args=args,
            train_fs=train_fs,
            val_fs=val_fs,
            trend_family=trend_family,
            alpha=alpha,
            alpha_enabled=True,
            env_kwargs=env_kwargs,
            output=output,
        )
        d_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=True)
        development_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs=d_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["D"] = evaluate_relative_agent(
            d_agent,
            val_fs,
            env_kwargs={**d_kwargs, "cost_multiplier": 2.0},
            bootstrap_seed=args.seed,
        )

    selection = select_residual_configuration(
        development_results,
        cost2x_results=development_cost2x_results,
    )
    selected = str(selection["selected"])
    if selected == "D":
        if d_agent is None or d_model_path is None:
            raise RuntimeError("configuration D selected without a trained D candidate")
        selected_agent = d_agent
        selected_policies = tuple(d_policies)
        selected_model_path = d_model_path
        selected_alpha_enabled = True
    elif selected == "B":
        selected_agent = b_agent
        selected_policies = tuple(b_policies)
        selected_model_path = b_model_path
        selected_alpha_enabled = False
    elif selected == "A":
        selected_agent = identity
        selected_policies = ()
        selected_model_path = None
        selected_alpha_enabled = False
    else:
        raise RuntimeError(f"unsupported selected residual configuration: {selected}")

    return ResidualCandidateSelection(
        development_results=development_results,
        development_cost2x_results=development_cost2x_results,
        selection=selection,
        selected_configuration=selected,
        selected_agent=selected_agent,
        selected_policies=selected_policies,
        selected_model_path=selected_model_path,
        selected_alpha_enabled=selected_alpha_enabled,
    )
