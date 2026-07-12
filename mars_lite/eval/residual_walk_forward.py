from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean, median
from typing import Any

import numpy as np

from mars_lite.eval.context_window import with_history_context
from mars_lite.eval.gate1_diagnostics import walk_forward_ic
from mars_lite.eval.relative_evaluation import evaluate_relative_agent
from mars_lite.features.signal_check import run_leak_self_test
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.gates import evaluate_residual_alpha_gate
from mars_lite.pipeline.residual_candidates import train_select_residual_candidates
from mars_lite.pipeline.residual_pipeline import (
    _evaluation_kwargs,
    _is_fallback,
    _slim_baselines,
)
from mars_lite.pipeline.training_engine import build_env_kwargs, build_post_processor
from mars_lite.trading.execution import FEE_KWARG_KEYS
from mars_lite.trading.residual_alpha import FrozenResidualAlpha
from mars_lite.trading.trend_family import TrendFamily, TrendFamilyConfig


@dataclass(frozen=True)
class ResidualFoldSpec:
    fold: int
    outer_train_start: int
    outer_train_end: int
    inner_train_start: int
    inner_train_end: int
    inner_validation_start: int
    inner_validation_end: int
    outer_test_start: int
    outer_test_end: int
    purge_bars: int

    @property
    def inner_train_bars(self) -> int:
        return self.inner_train_end - self.inner_train_start

    @property
    def inner_validation_bars(self) -> int:
        return self.inner_validation_end - self.inner_validation_start

    @property
    def outer_test_bars(self) -> int:
        return self.outer_test_end - self.outer_test_start


def build_residual_fold_specs(
    *,
    n_bars: int,
    n_folds: int,
    purge_bars: int,
    horizon: int,
) -> tuple[list[ResidualFoldSpec], list[dict[str, object]]]:
    if min(n_bars, n_folds, purge_bars, horizon) <= 0:
        raise ValueError("n_bars, n_folds, purge_bars, and horizon must be positive")

    effective_purge = max(int(purge_bars), int(horizon), 24)
    edges = np.linspace(int(n_bars * 0.4), n_bars, n_folds + 1).astype(int)
    specs: list[ResidualFoldSpec] = []
    skipped: list[dict[str, object]] = []

    for fold in range(n_folds):
        outer_train_end = int(edges[fold])
        outer_test_start = outer_train_end + effective_purge
        outer_test_end = int(edges[fold + 1])
        inner_train_end = int(outer_train_end * 0.8)
        inner_validation_start = inner_train_end + effective_purge
        inner_validation_end = outer_train_end

        sizes = {
            "inner_train_too_short": inner_train_end,
            "inner_validation_too_short": inner_validation_end - inner_validation_start,
            "outer_test_too_short": outer_test_end - outer_test_start,
        }
        minimums = {
            "inner_train_too_short": 200,
            "inner_validation_too_short": 100,
            "outer_test_too_short": 50,
        }
        reason = next(
            (name for name, value in sizes.items() if value < minimums[name]),
            None,
        )
        if reason is not None:
            skipped.append(
                {
                    "fold": fold,
                    "reason": reason,
                    "outer_train_end": outer_train_end,
                    "inner_train_end": inner_train_end,
                    "inner_validation_start": inner_validation_start,
                    "inner_validation_end": inner_validation_end,
                    "outer_test_start": outer_test_start,
                    "outer_test_end": outer_test_end,
                    "purge_bars": effective_purge,
                }
            )
            continue

        specs.append(
            ResidualFoldSpec(
                fold=fold,
                outer_train_start=0,
                outer_train_end=outer_train_end,
                inner_train_start=0,
                inner_train_end=inner_train_end,
                inner_validation_start=inner_validation_start,
                inner_validation_end=inner_validation_end,
                outer_test_start=outer_test_start,
                outer_test_end=outer_test_end,
                purge_bars=effective_purge,
            )
        )

    return specs, skipped


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "median": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(mean(values)),
        "median": float(median(values)),
        "min": float(min(values)),
        "max": float(max(values)),
    }


def summarize_residual_folds(
    folds: list[dict[str, object]],
    *,
    requested_folds: int,
    skipped_folds: list[dict[str, object]],
) -> dict[str, object]:
    selection_counts = {"A": 0, "B": 0, "D": 0}
    hybrid_returns_1x: list[float] = []
    hybrid_returns_2x: list[float] = []
    shadow_returns_1x: list[float] = []
    shadow_returns_2x: list[float] = []
    excess_1x: list[float] = []
    excess_2x: list[float] = []
    fallback_count = 0
    member_count = 0
    alpha_enabled_count = 0
    hybrid_zero_trade_folds = 0
    shadow_zero_trade_folds = 0
    total_scored_oos_bars = 0

    for fold in folds:
        selected = str(fold["selected_configuration"])
        if selected in selection_counts:
            selection_counts[selected] += 1
        if bool(fold.get("alpha_enabled", False)):
            alpha_enabled_count += 1

        raw_fallbacks = fold.get("selected_seed_fallbacks", [])
        if not isinstance(raw_fallbacks, list):
            raise TypeError("selected_seed_fallbacks must be a list")
        fallback_flags = [bool(value) for value in raw_fallbacks]
        fallback_count += sum(fallback_flags)
        member_count += len(fallback_flags)

        outer_oos = fold["outer_oos"]
        if not isinstance(outer_oos, dict):
            raise TypeError("outer_oos must be a mapping")
        relative_1x = outer_oos["relative_1x"]
        relative_2x = outer_oos["relative_2x"]
        if not isinstance(relative_1x, dict) or not isinstance(relative_2x, dict):
            raise TypeError("relative evaluations must be mappings")

        hybrid_1x = relative_1x["hybrid"]
        shadow_1x = relative_1x["shadow"]
        paired_1x = relative_1x["paired"]
        hybrid_2x = relative_2x["hybrid"]
        shadow_2x = relative_2x["shadow"]
        paired_2x = relative_2x["paired"]
        for value in (
            hybrid_1x,
            shadow_1x,
            paired_1x,
            hybrid_2x,
            shadow_2x,
            paired_2x,
        ):
            if not isinstance(value, dict):
                raise TypeError("relative metric sections must be mappings")

        hybrid_returns_1x.append(float(hybrid_1x["total_return"]))
        hybrid_returns_2x.append(float(hybrid_2x["total_return"]))
        shadow_returns_1x.append(float(shadow_1x["total_return"]))
        shadow_returns_2x.append(float(shadow_2x["total_return"]))
        excess_1x.append(float(paired_1x["excess_log_return"]))
        excess_2x.append(float(paired_2x["excess_log_return"]))
        hybrid_zero_trade_folds += int(int(hybrid_1x["n_trades"]) == 0)
        shadow_zero_trade_folds += int(int(shadow_1x["n_trades"]) == 0)

        split = fold["split"]
        if not isinstance(split, dict):
            raise TypeError("split must be a mapping")
        total_scored_oos_bars += int(split["outer_test_scored_bars"])

    completed = len(folds)
    return {
        "requested_folds": int(requested_folds),
        "completed_folds": completed,
        "skipped_folds": len(skipped_folds),
        "selection_counts": selection_counts,
        "alpha_enabled_folds": alpha_enabled_count,
        "selected_member_fallback_count": fallback_count,
        "selected_member_count": member_count,
        "selected_member_fallback_rate": (
            float(fallback_count / member_count) if member_count else 0.0
        ),
        "hybrid_zero_trade_folds": hybrid_zero_trade_folds,
        "shadow_zero_trade_folds": shadow_zero_trade_folds,
        "hybrid_return_1x": _stats(hybrid_returns_1x),
        "hybrid_return_2x": _stats(hybrid_returns_2x),
        "shadow_return_1x": _stats(shadow_returns_1x),
        "shadow_return_2x": _stats(shadow_returns_2x),
        "paired_excess_log_return_1x": _stats(excess_1x),
        "paired_excess_log_return_2x": _stats(excess_2x),
        "hybrid_beats_shadow_fraction_1x": (
            float(sum(value > 0.0 for value in excess_1x) / completed)
            if completed
            else 0.0
        ),
        "hybrid_beats_shadow_fraction_2x": (
            float(sum(value > 0.0 for value in excess_2x) / completed)
            if completed
            else 0.0
        ),
        "survives_cost2x_fraction": (
            float(sum(value >= 0.0 for value in excess_2x) / completed)
            if completed
            else 0.0
        ),
        "total_scored_oos_bars": total_scored_oos_bars,
    }


def save_residual_walk_forward_report(
    path: str | Path,
    payload: dict[str, Any],
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    )
    output.write_text(text + "\n", encoding="utf-8")


def _history_bars(config: TrendFamilyConfig) -> int:
    return (
        max(config.fast_lookback, config.base_lookback, config.slow_lookback)
        + config.rebalance_every
    )


def _baseline_kwargs(env_kwargs: dict[str, Any]) -> dict[str, Any]:
    return {key: env_kwargs[key] for key in FEE_KWARG_KEYS if key in env_kwargs}


def run_residual_fold(
    *,
    fs,
    spec: ResidualFoldSpec,
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Train/select on inner development data and score one frozen outer OOS fold."""

    fold_output = Path(output_dir) / "residual_wf" / f"fold_{spec.fold}"
    fold_output.mkdir(parents=True, exist_ok=True)
    fold_args = copy.copy(args)
    fold_args.seed = int(args.seed) + spec.fold * max(1, int(args.ensemble))

    inner_train_fs = fs.slice(spec.inner_train_start, spec.inner_train_end)
    leak = run_leak_self_test(inner_train_fs, horizon=args.horizon)
    if not leak["healthy"]:
        raise RuntimeError(f"fold {spec.fold} leak self-test failed")

    trend_config = TrendFamilyConfig(
        base_timeframe=getattr(args, "base_timeframe", "1h")
    )
    trend_family = TrendFamily(trend_config)
    history_bars = _history_bars(trend_config)
    validation_window = with_history_context(
        fs,
        start=spec.inner_validation_start,
        end=spec.inner_validation_end,
        history_bars=history_bars,
    )
    test_window = with_history_context(
        fs,
        start=spec.outer_test_start,
        end=spec.outer_test_end,
        history_bars=history_bars,
    )
    validation_fs = validation_window.feature_set
    test_fs = test_window.feature_set

    signal_model = str(getattr(args, "signal_model", "gbm"))
    model_gate_report = walk_forward_ic(
        inner_train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
    )
    signal_gate = evaluate_residual_alpha_gate(model_gate_report)
    alpha = FrozenResidualAlpha.fit(
        inner_train_fs,
        horizon=args.horizon,
        target="cs_demean",
        model=signal_model,
        gate_result=signal_gate,
    )
    alpha_path = fold_output / "residual_alpha.json"
    alpha.save(alpha_path)

    post_processor = build_post_processor(fold_args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(fold_args, post_processor, horizon=args.horizon)
    candidates = train_select_residual_candidates(
        args=fold_args,
        train_fs=inner_train_fs,
        val_fs=validation_fs,
        trend_family=trend_family,
        alpha=alpha,
        env_kwargs=env_kwargs,
        output=fold_output,
    )

    selected_env_kwargs = _evaluation_kwargs(
        env_kwargs,
        trend_family,
        alpha,
        alpha_enabled=candidates.selected_alpha_enabled,
    )
    relative_1x = evaluate_relative_agent(
        candidates.selected_agent,
        test_fs,
        env_kwargs=selected_env_kwargs,
        bootstrap_seed=fold_args.seed,
    )
    relative_2x = evaluate_relative_agent(
        candidates.selected_agent,
        test_fs,
        env_kwargs={**selected_env_kwargs, "cost_multiplier": 2.0},
        bootstrap_seed=fold_args.seed,
    )

    noisy_oracle = float(getattr(args, "noisy_oracle_ic", 0.0))
    baseline_kwargs = _baseline_kwargs(env_kwargs)
    baselines_1x = _slim_baselines(
        run_all_baselines(
            test_fs,
            noisy_oracle_ic=noisy_oracle if noisy_oracle > 0.0 else None,
            cost_multiplier=1.0,
            start_idx=test_window.start_idx,
            **baseline_kwargs,
        )
    )
    baselines_2x = _slim_baselines(
        run_all_baselines(
            test_fs,
            noisy_oracle_ic=noisy_oracle if noisy_oracle > 0.0 else None,
            cost_multiplier=2.0,
            start_idx=test_window.start_idx,
            **baseline_kwargs,
        )
    )

    selected_model_identity = (
        str(candidates.selected_model_path)
        if candidates.selected_model_path is not None
        else "identity:base_trend_v2"
    )
    report: dict[str, object] = {
        "fold": spec.fold,
        "split": {
            **asdict(spec),
            "inner_train_bars": inner_train_fs.n_bars,
            "inner_validation_scored_bars": validation_window.scored_bars,
            "inner_validation_context_bars": validation_window.start_idx,
            "outer_test_scored_bars": test_window.scored_bars,
            "outer_test_context_bars": test_window.start_idx,
            "history_bars_required": history_bars,
        },
        "leak_self_test": leak,
        "signal_gate": signal_gate,
        "alpha_dataset_identity": alpha.dataset_identity,
        "alpha_artifact_path": str(alpha_path),
        "alpha_artifact_gate_passed": bool(alpha.enabled),
        "development_matrix": candidates.development_results,
        "development_matrix_cost2x": candidates.development_cost2x_results,
        "selection": candidates.selection,
        "selected_configuration": candidates.selected_configuration,
        "selected_policy_mode": str(candidates.selection["policy_mode"]),
        "selected_model_identity": selected_model_identity,
        "alpha_enabled": candidates.selected_alpha_enabled,
        "selected_seed_fallbacks": [
            _is_fallback(policy) for policy in candidates.selected_policies
        ],
        "outer_oos": {
            "relative_1x": relative_1x,
            "relative_2x": relative_2x,
            "baselines_1x": baselines_1x,
            "baselines_2x": baselines_2x,
            "same_selected_model_for_cost_scenarios": True,
        },
    }
    save_residual_walk_forward_report(fold_output / "fold_report.json", report)
    return report


def run_residual_walk_forward(
    args,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run research-only nested expanding Walk-Forward for residual RL."""

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    args.action_mode = "baseline-residual"
    args.min_trade_delta = 0.0
    args.lambda_turnover = 0.0
    requested_seeds = max(1, int(getattr(args, "n_seeds", 1)))
    requested_ensemble = max(1, int(getattr(args, "ensemble", 1)))
    args.ensemble = max(requested_seeds, requested_ensemble)

    fs = build_feature_set(args, output_dir=output)
    requested_folds = int(getattr(args, "folds", 3))
    purge_bars = int(getattr(args, "purge_bars", 24))
    specs, skipped = build_residual_fold_specs(
        n_bars=fs.n_bars,
        n_folds=requested_folds,
        purge_bars=purge_bars,
        horizon=int(args.horizon),
    )

    folds: list[dict[str, object]] = []
    for spec in specs:
        folds.append(
            run_residual_fold(
                fs=fs,
                spec=spec,
                args=args,
                output_dir=output,
            )
        )

    report: dict[str, object] = {
        "mode": "baseline_residual_walk_forward_v1",
        "action_schema": "baseline_residual_v1",
        "config": {
            "requested_folds": requested_folds,
            "completed_folds": len(folds),
            "purge_bars": max(purge_bars, int(args.horizon), 24),
            "horizon": int(args.horizon),
            "decision_every": int(args.decision_every),
            "ensemble_size": int(args.ensemble),
            "requested_n_seeds": requested_seeds,
            "member_seeds_per_fold": int(args.ensemble),
            "run_tier": str(getattr(args, "run_tier", "research")),
            "n_bars_total": int(fs.n_bars),
            "outer_train_start_fraction": 0.4,
            "inner_train_fraction": 0.8,
        },
        "summary": summarize_residual_folds(
            folds,
            requested_folds=requested_folds,
            skipped_folds=skipped,
        ),
        "folds": folds,
        "skipped_folds": skipped,
        "release_eligible": False,
        "release_blocker": (
            "sealed residual release workflow remains incomplete; research output only"
        ),
    }
    save_residual_walk_forward_report(output / "residual_walk_forward.json", report)
    return report
