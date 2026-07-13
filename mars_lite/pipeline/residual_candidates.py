from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.learning.residual_ensemble import ResidualActionEnsemble
from mars_lite.pipeline.training_engine import train_ppo
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily


class FixedResidualAgent:
    def __init__(self, action: tuple[float, float]):
        value = np.asarray(action, dtype=np.float32)
        if value.shape != (2,) or not np.all(np.isfinite(value)):
            raise ValueError("fixed residual action must be finite with shape (2,)")
        self.action = value

    def predict(self, observation, deterministic: bool = True):
        observation_array = np.asarray(observation)
        if observation_array.ndim == 1:
            return self.action.copy(), None
        return np.repeat(self.action[None, :], observation_array.shape[0], axis=0), None


class IdentityResidualAgent(FixedResidualAgent):
    def __init__(self):
        super().__init__((0.0, 0.0))


def _slim_baselines(results: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        name: value.to_dict() if hasattr(value, "to_dict") else dict(value)
        for name, value in results.items()
    }


def _is_fallback(policy: object) -> bool:
    selection = getattr(policy, "validation_selection", None)
    score = getattr(selection, "best_score", None)
    return bool(getattr(score, "baseline_fallback", False))


def _eligible_relative_result(result: dict[str, Any], drawdown_slack: float) -> bool:
    excess = float(result["paired"]["excess_log_return"])
    hybrid_dd = float(result["hybrid"]["max_drawdown"])
    shadow_dd = float(result["shadow"]["max_drawdown"])
    return excess > 0.0 and hybrid_dd <= shadow_dd + drawdown_slack


def select_residual_configuration(
    development_results: dict[str, dict[str, Any]],
    *,
    cost2x_results: dict[str, dict[str, Any]],
    drawdown_slack: float = 0.05,
) -> dict[str, Any]:
    """Select A/B/D using preregistered 1x and 2x development rules."""

    if "A" not in development_results or "A" not in cost2x_results:
        raise ValueError("development matrices require configuration A")
    if set(development_results) != set(cost2x_results):
        raise ValueError(
            "1x and 2x development matrices must contain identical configs"
        )
    scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in development_results.items()
    }
    cost2x_scores = {
        name: float(result["paired"]["excess_log_return"])
        for name, result in cost2x_results.items()
    }
    base_eligible = {
        name: _eligible_relative_result(result, drawdown_slack)
        for name, result in development_results.items()
    }
    cost2x_eligible = {name: score >= 0.0 for name, score in cost2x_scores.items()}
    eligible = {
        name: base_eligible[name] and cost2x_eligible[name]
        for name in development_results
    }
    selected = "A"
    reasons = ["A is the identity baseline"]

    if eligible.get("B", False):
        selected = "B"
        reasons.append(
            "B adds positive development excess within drawdown slack and survives 2x costs"
        )

    if eligible.get("D", False):
        if not eligible.get("C", False):
            reasons.append(
                "D was rejected because fixed-alpha diagnostic C did not beat A"
            )
        else:
            hurdle = max(scores.get("B", 0.0), scores["C"])
            if scores["D"] > hurdle:
                selected = "D"
                reasons.append(
                    "D strictly beats both B and fixed-alpha diagnostic C and survives 2x costs"
                )
            else:
                reasons.append("D did not beat the stronger of B and C")

    return {
        "selected": selected,
        "policy_mode": (
            "baseline_only" if selected == "A" else "ppo_residual_ensemble"
        ),
        "scores": scores,
        "cost2x_scores": cost2x_scores,
        "base_eligible": base_eligible,
        "cost2x_eligible": cost2x_eligible,
        "eligible": eligible,
        "reasons": reasons,
        "drawdown_slack": drawdown_slack,
    }


def _train_residual_ensemble(
    *,
    label: str,
    args,
    train_fs,
    checkpoint_val_fs,
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    alpha_enabled: bool,
    env_kwargs: dict[str, Any],
    output: Path,
) -> tuple[object, list[object], Path]:
    ensemble_size = max(1, int(getattr(args, "ensemble", 3)))
    policies: list[object] = []
    for member in range(ensemble_size):
        policies.append(
            train_ppo(
                fs=train_fs,
                val_fs=checkpoint_val_fs,
                timesteps=args.timesteps,
                seed=args.seed + member,
                gamma=args.gamma,
                ent_coef=getattr(args, "ent_coef", 0.002),
                learning_rate=getattr(args, "learning_rate", 3e-4),
                verbose=args.verbose,
                action_mode="baseline-residual",
                run_tier=getattr(args, "run_tier", "research"),
                n_seeds=ensemble_size,
                trend_family=trend_family,
                alpha_provider=alpha,
                alpha_enabled=alpha_enabled,
                bc_warmstart=False,
                **env_kwargs,
            )
        )
    agent: object = (
        policies[0] if len(policies) == 1 else ResidualActionEnsemble(policies)
    )
    if len(policies) == 1:
        model_path = output / f"{label}_model.zip"
        policies[0].save(str(model_path))
    else:
        model_path = output / f"{label}_ensemble"
        agent.save(model_path)
    return agent, policies, model_path


def _evaluation_kwargs(
    env_kwargs: dict[str, Any],
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    *,
    alpha_enabled: bool,
) -> dict[str, Any]:
    return {
        **env_kwargs,
        "trend_family": trend_family,
        "alpha_provider": alpha,
        "alpha_enabled": alpha_enabled,
    }


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
    checkpoint_val_fs,
    selection_fs,
    trend_family: TrendFamily,
    alpha: FrozenResidualAlpha,
    env_kwargs: dict[str, Any],
    output: Path,
) -> ResidualCandidateSelection:
    """Train checkpoints on one window and select A/B/D on a later window."""

    output.mkdir(parents=True, exist_ok=True)
    identity = IdentityResidualAgent()
    fixed_alpha = FixedResidualAgent((0.0, 0.5))
    development_results: dict[str, dict[str, Any]] = {}
    development_cost2x_results: dict[str, dict[str, Any]] = {}

    a_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=False)
    development_results["A"] = evaluate_relative_agent(
        identity,
        selection_fs,
        env_kwargs=a_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["A"] = evaluate_relative_agent(
        identity,
        selection_fs,
        env_kwargs={**a_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )

    b_agent, b_policies, b_model_path = _train_residual_ensemble(
        label="B_trend_mix",
        args=args,
        train_fs=train_fs,
        checkpoint_val_fs=checkpoint_val_fs,
        trend_family=trend_family,
        alpha=alpha,
        alpha_enabled=False,
        env_kwargs=env_kwargs,
        output=output,
    )
    b_kwargs = _evaluation_kwargs(env_kwargs, trend_family, alpha, alpha_enabled=False)
    development_results["B"] = evaluate_relative_agent(
        b_agent,
        selection_fs,
        env_kwargs=b_kwargs,
        bootstrap_seed=args.seed,
    )
    development_cost2x_results["B"] = evaluate_relative_agent(
        b_agent,
        selection_fs,
        env_kwargs={**b_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=args.seed,
    )

    d_agent: object | None = None
    d_policies: list[object] = []
    d_model_path: Path | None = None
    if alpha.enabled:
        c_kwargs = _evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=True
        )
        development_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            selection_fs,
            env_kwargs=c_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["C"] = evaluate_relative_agent(
            fixed_alpha,
            selection_fs,
            env_kwargs={**c_kwargs, "cost_multiplier": 2.0},
            bootstrap_seed=args.seed,
        )
        d_agent, d_policies, d_model_path = _train_residual_ensemble(
            label="D_combined",
            args=args,
            train_fs=train_fs,
            checkpoint_val_fs=checkpoint_val_fs,
            trend_family=trend_family,
            alpha=alpha,
            alpha_enabled=True,
            env_kwargs=env_kwargs,
            output=output,
        )
        d_kwargs = _evaluation_kwargs(
            env_kwargs, trend_family, alpha, alpha_enabled=True
        )
        development_results["D"] = evaluate_relative_agent(
            d_agent,
            selection_fs,
            env_kwargs=d_kwargs,
            bootstrap_seed=args.seed,
        )
        development_cost2x_results["D"] = evaluate_relative_agent(
            d_agent,
            selection_fs,
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
