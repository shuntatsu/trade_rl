"""Production control-plane orchestration.

Runs the validated training/evaluation sequence, constructs one complete immutable
ServingBundle candidate, and registers it without activating it. Activation belongs
exclusively to the deployment control plane after evidence and environment approval.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.evaluator import phase_p0, phase_pbt, phase_train, phase_wf
from mars_lite.pipeline.release_eligibility import (
    ReleaseEligibility,
    derive_release_eligibility,
)
from mars_lite.pipeline.release_risk import (
    ReleaseRiskPolicy,
    load_release_risk_policy,
)
from mars_lite.pipeline.training_engine import build_env_kwargs, build_post_processor
from mars_lite.serving.candidate import create_candidate_bundle
from mars_lite.serving.registry import ModelRegistry
from mars_lite.trading.guardrails import GuardrailConfig


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _print_step(number: int, total: int, title: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {number}/{total}: {title}\n{'=' * 70}")


def _resolve_identity(args: Any) -> tuple[str, str]:
    git_sha = str(getattr(args, "git_sha", "") or os.getenv("GITHUB_SHA", "")).strip()
    if not git_sha:
        raise ValueError(
            "--git-sha or GITHUB_SHA is required to produce a registered candidate"
        )
    version = str(getattr(args, "model_version", "") or "").strip()
    if not version:
        version = f"model-{git_sha[:12]}"
    return version, git_sha


def _model_source(output_dir: Path, ensemble: int) -> Path:
    source = (
        output_dir / "portfolio_ensemble"
        if ensemble > 1
        else output_dir / "portfolio_model.zip"
    )
    if not source.exists():
        raise FileNotFoundError(f"trained model artifact not found: {source}")
    return source


def build_and_register_candidate(
    *,
    args: Any,
    output_dir: Path,
    feature_set: FeatureSet,
    train_result: dict[str, Any],
    risk_policy: ReleaseRiskPolicy,
    release_eligibility: ReleaseEligibility,
) -> Path:
    """Create a complete eligible candidate bundle and register without activation."""
    signal_layer = str(getattr(args, "signal_layer", "off"))
    if signal_layer != "off":
        raise ValueError(
            "production candidates require signal_layer=off until the Serving Plane "
            "can reproduce the causal signal transform"
        )
    if not release_eligibility.eligible:
        raise ValueError("ineligible research run cannot create a release candidate")
    version, git_sha = _resolve_identity(args)
    train_report = _load(output_dir / "train_report.json")
    feature_mask = train_report.get("feature_mask")
    if feature_mask is not None:
        if not isinstance(feature_mask, list) or not all(
            isinstance(item, bool) for item in feature_mask
        ):
            raise ValueError("train_report feature_mask must be a boolean list")

    post_processor = build_post_processor(args, horizon=args.horizon)
    env_kwargs = build_env_kwargs(args, post_processor, horizon=args.horizon)
    env = PortfolioTradingEnv(feature_set, **env_kwargs)
    observation_dim = int(env.observation_space.shape[0])
    run_config = {
        key: value for key, value in env_kwargs.items() if key != "post_processor"
    }
    run_config["base_timeframe"] = str(getattr(args, "base_timeframe", "1h"))
    run_config["observation_progress_mode"] = "zero"

    candidate_dir = output_dir / "candidates" / version
    source = _model_source(output_dir, int(args.ensemble))
    metrics = {
        "agent": {
            key: value
            for key, value in train_result["agent_res"].items()
            if key != "equity_curve"
        },
        "gate2": train_result["gate2"],
        "signal_gate": train_report.get("signal_gate"),
        "lockbox": train_report.get("lockbox"),
    }
    create_candidate_bundle(
        destination=candidate_dir,
        model_source=source,
        version=version,
        git_sha=git_sha,
        symbols=feature_set.symbols,
        feature_names=feature_set.feature_names,
        global_feature_names=feature_set.global_feature_names,
        feature_norm=str(getattr(args, "feature_norm", "none")),
        feature_mask=feature_mask,
        observation_dim=observation_dim,
        observation_schema_version=1,
        post_processor=post_processor.cfg.to_dict(),
        run_config=run_config,
        metrics=metrics,
        guardrails=asdict(GuardrailConfig()),
        risk_policy=risk_policy,
        release_eligibility=release_eligibility,
    )

    registry_dir = Path(
        getattr(args, "registry_dir", None) or output_dir / "model_registry"
    )
    registered = ModelRegistry(registry_dir).register(candidate_dir)
    print(
        f"[registry] registered immutable candidate version={registered.version} "
        f"digest={registered.bundle_digest}; not activated"
    )
    return candidate_dir


def run(args: Any) -> int:
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 5

    registration_requested = not bool(args.no_register)
    release_disqualifying_override = any(
        (args.force, args.skip_p0, args.skip_wf, args.skip_gate)
    )
    release_intent = registration_requested and not release_disqualifying_override
    if registration_requested and release_disqualifying_override:
        print(
            "[research-only] release-disqualifying override detected; "
            "candidate creation and registration are disabled"
        )
    if release_intent and getattr(args, "risk_config", None) is None:
        raise ValueError("release candidate requires --risk-config")

    p0_passed = False
    walk_forward_passed = False
    significance_passed: bool | None = None

    _print_step(1, total_steps, "P0 system sanity gate")
    if args.skip_p0:
        print("[skip]")
    else:
        original = (args.horizon, args.decision_every, args.days)
        args.horizon, args.decision_every, args.days = 4, 1, 240
        try:
            phase_p0(args, output_dir)
        finally:
            args.horizon, args.decision_every, args.days = original
        gate = _load(output_dir / "p0_report.json")["gate"]
        p0_passed = bool(gate["P0_PASSED"])
        if not p0_passed and not args.force:
            print("[STOP] P0 failed")
            return 1

    try:
        full_features = build_feature_set(args, output_dir=output_dir)
    except ValueError as exc:
        print(f"[STOP] {exc}")
        return 1

    purge = max(24, args.horizon)
    holdout_start = int(full_features.n_bars * (1.0 - args.holdout_frac))
    minimum_bars = 50
    development: FeatureSet | None = None
    holdout: FeatureSet | None = None
    if (
        full_features.n_bars - holdout_start - purge >= minimum_bars
        and holdout_start >= minimum_bars
    ):
        development = full_features.slice(0, holdout_start)
        holdout = full_features.slice(holdout_start + purge, full_features.n_bars)
        print(
            f"[holdout] development={development.n_bars} bars; "
            f"sealed holdout={holdout.n_bars} bars"
        )
    else:
        print("[WARN] insufficient data for a sealed holdout")
    sealed_holdout_available = holdout is not None and holdout.n_bars > 0
    if release_intent and not sealed_holdout_available:
        raise RuntimeError(
            "release candidate requires a non-empty sealed holdout after purge"
        )

    _print_step(2, total_steps, "PBT hyperparameter search")
    if args.skip_pbt:
        print("[skip]")
    else:
        phase_pbt(args, output_dir, fs=development)
        best = _load(output_dir / "pbt_result.json")["best_hp"]
        args.gamma = best["gamma"]
        args.ent_coef = best["ent_coef"]
        args.learning_rate = best["learning_rate"]
        args.lambda_turnover = best["lambda_turnover"]
        args.reward_scale = best["reward_scale"]

    _print_step(3, total_steps, "Walk-forward robustness gate")
    if args.skip_wf:
        print("[skip]")
    else:
        phase_wf(args, output_dir, fs=development)
        report = _load(output_dir / "walk_forward_cost2x.json")
        median_return = report["summary"]["agent_total_return"]["median"]
        walk_forward_passed = bool(median_return > args.wf_cost_gate)
        if not walk_forward_passed and not args.force:
            print(f"[STOP] cost-2x WF median {median_return:+.2%} failed")
            return 1

    _print_step(4, total_steps, "Final training and sealed gate")
    train_result = phase_train(args, output_dir, dev_fs=development, holdout_fs=holdout)
    if train_result is None:
        print("[STOP] training exited before producing a candidate")
        return 1
    gate2_passed = bool(train_result["gate2"]["passed"])
    if not gate2_passed and not args.force:
        print("[STOP] gate2 failed")
        return 1

    if args.require_significant:
        from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference

        agent_curve = train_result["agent_res"].get("equity_curve")
        baseline = train_result["baselines"].get("trend_following")
        baseline_curve = baseline.equity_curve if baseline is not None else None
        if agent_curve is None or baseline_curve is None or not len(baseline_curve):
            print("[STOP] significance gate requires both equity curves")
            return 1
        agent_returns = np.diff(agent_curve) / np.asarray(agent_curve[:-1])
        baseline_returns = np.diff(baseline_curve) / np.asarray(baseline_curve[:-1])
        length = min(len(agent_returns), len(baseline_returns))
        result = bootstrap_sharpe_difference(
            agent_returns[:length], baseline_returns[:length], seed=args.seed
        )
        significance_passed = bool(result["p_value"] < 0.05)
        if not significance_passed and not args.force:
            print(f"[STOP] superiority is not significant: p={result['p_value']:.4f}")
            return 1

    eligibility = derive_release_eligibility(
        forced=bool(args.force),
        skip_p0=bool(args.skip_p0),
        skip_pbt=bool(args.skip_pbt),
        skip_wf=bool(args.skip_wf),
        skip_gate=bool(args.skip_gate),
        sealed_holdout_used=sealed_holdout_available,
        p0_passed=p0_passed,
        walk_forward_passed=walk_forward_passed,
        gate2_passed=gate2_passed,
        significance_passed=significance_passed,
    )

    _print_step(5, total_steps, "Serving candidate construction and registration")
    if release_intent and eligibility.eligible:
        schema_features = development or full_features
        risk_policy = load_release_risk_policy(
            args.risk_config, symbols=tuple(schema_features.symbols)
        )
        build_and_register_candidate(
            args=args,
            output_dir=output_dir,
            feature_set=schema_features,
            train_result=train_result,
            risk_policy=risk_policy,
            release_eligibility=eligibility,
        )
    else:
        print("[skip] research-only or ineligible run; no candidate registered")

    print("\nControl-plane pipeline complete. Candidate activation was not performed.")
    return 0
