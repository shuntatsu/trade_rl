from __future__ import annotations

from typing import Optional

import numpy as np

from mars_lite.env.baseline_residual_env import BaselineResidualTradingEnv
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet


def validate_run_tier(
    run_tier: str,
    *,
    timesteps: int,
    n_envs: int,
    n_steps: int,
    n_seeds: int,
) -> dict[str, int | str]:
    requirements = {
        "smoke": (5, 1),
        "research": (50, 3),
        "release": (100, 5),
    }
    if run_tier not in requirements:
        raise ValueError(f"unknown run tier: {run_tier}")
    if min(timesteps, n_envs, n_steps, n_seeds) <= 0:
        raise ValueError("timesteps, n_envs, n_steps, and n_seeds must be positive")
    required_updates, required_seeds = requirements[run_tier]
    one_rollout = n_envs * n_steps
    updates = timesteps // one_rollout
    if updates < required_updates:
        raise ValueError(
            f"{run_tier} requires at least {required_updates} updates "
            f"({required_updates * one_rollout:,} timesteps)"
        )
    if n_seeds < required_seeds:
        raise ValueError(f"{run_tier} requires at least {required_seeds} seeds")
    return {
        "run_tier": run_tier,
        "updates": updates,
        "required_updates": required_updates,
        "required_seeds": required_seeds,
        "one_rollout_steps": one_rollout,
    }


def make_env_fns(
    fs: FeatureSet,
    n_envs: int,
    seed: int,
    *,
    env_class=PortfolioTradingEnv,
    **env_kwargs,
):
    from stable_baselines3.common.monitor import Monitor

    def make_one(rank: int):
        def _init():
            env = env_class(fs, **env_kwargs)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _init

    return [make_one(i) for i in range(n_envs)]


def zero_initialize_action_head(agent, exploration_log_std: float = -2.0) -> None:
    """Make the initial deterministic residual action exactly the identity action."""

    import torch

    policy = agent.policy
    with torch.no_grad():
        policy.action_net.weight.zero_()
        policy.action_net.bias.zero_()
        log_std = getattr(policy, "log_std", None)
        if log_std is not None:
            log_std.fill_(float(exploration_log_std))


def _resolve_env(
    fs: FeatureSet,
    action_mode: str,
    env_kwargs: dict,
    *,
    trend_family=None,
    alpha_provider=None,
    alpha_enabled: bool = True,
):
    if action_mode == "direct":
        return PortfolioTradingEnv, dict(env_kwargs)
    if action_mode != "baseline-residual":
        raise ValueError(f"unknown action_mode: {action_mode}")
    residual_kwargs = dict(env_kwargs)
    residual_kwargs.pop("lambda_turnover", None)
    residual_kwargs.pop("use_dsr", None)
    residual_kwargs.pop("dsr_eta", None)
    residual_kwargs.pop("disagreement_dr_max", None)
    residual_kwargs.update(
        trend_family=trend_family,
        alpha_provider=alpha_provider,
        alpha_enabled=alpha_enabled,
    )
    return BaselineResidualTradingEnv, residual_kwargs


def train_ppo(
    fs: FeatureSet,
    timesteps: int = 300_000,
    seed: int = 0,
    n_envs: int = 8,
    learning_rate: float = 3e-4,
    ent_coef: float = 0.002,
    gamma: float = 0.5,
    verbose: int = 0,
    callbacks=None,
    val_fs: FeatureSet = None,
    val_eval_freq: Optional[int] = None,
    bc_warmstart: bool = True,
    bc_epochs: int = 15,
    bc_teacher: str = "auto",
    extractor: str = "tfgated",
    horizon: int = 4,
    signal_target: str = "raw",
    oracle_noisy_ic: Optional[float] = None,
    action_mode: str = "direct",
    run_tier: Optional[str] = None,
    n_seeds: int = 1,
    trend_family=None,
    alpha_provider=None,
    alpha_enabled: bool = True,
    **env_kwargs,
):
    """Train direct-weight or baseline-residual PPO.

    Residual mode disables BC, zero-initializes the action mean, aggregates base bars
    inside the environment, and restores only checkpoints that beat the shadow trend.
    """

    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    n_steps = 256
    if run_tier is not None:
        validate_run_tier(
            run_tier,
            timesteps=timesteps,
            n_envs=n_envs,
            n_steps=n_steps,
            n_seeds=n_seeds,
        )

    full_fs = fs
    if val_fs is None and fs.n_bars > 400:
        cut = int(fs.n_bars * 0.85)
        val_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)

    env_class, resolved_env_kwargs = _resolve_env(
        fs,
        action_mode,
        env_kwargs,
        trend_family=trend_family,
        alpha_provider=alpha_provider,
        alpha_enabled=alpha_enabled,
    )
    env = DummyVecEnv(
        make_env_fns(
            fs,
            n_envs,
            seed,
            env_class=env_class,
            **resolved_env_kwargs,
        )
    )
    probe = env_class(fs, **resolved_env_kwargs)

    from mars_lite.features.feature_pipeline import TF_BLOCK_FEATURES

    tf_prefixes = []
    for name in fs.feature_names:
        prefix = name.split("_")[0]
        if (
            prefix in ("15m", "30m", "1h", "4h", "1d")
            and prefix not in tf_prefixes
        ):
            tf_prefixes.append(prefix)

    if extractor == "tfgated" and tf_prefixes:
        from mars_lite.models.portfolio_extractor import TFGatedPortfolioExtractor

        policy_kwargs = {
            "features_extractor_class": TFGatedPortfolioExtractor,
            "features_extractor_kwargs": {
                **probe.obs_layout,
                "n_tf_blocks": len(tf_prefixes),
                "tf_block_size": len(TF_BLOCK_FEATURES),
                "size": "small",
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }
    else:
        from mars_lite.models.portfolio_extractor import PortfolioExtractor

        policy_kwargs = {
            "features_extractor_class": PortfolioExtractor,
            "features_extractor_kwargs": {**probe.obs_layout, "size": "small"},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }

    def lr_schedule(progress_remaining: float) -> float:
        return learning_rate * progress_remaining

    agent = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=n_steps,
        batch_size=256,
        n_epochs=6,
        gamma=gamma,
        gae_lambda=0.9,
        ent_coef=ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        seed=seed,
        device="cpu",
        verbose=verbose,
    )

    if action_mode == "baseline-residual":
        zero_initialize_action_head(agent)
        bc_warmstart = False

    if bc_warmstart:
        from mars_lite.learning.bc_warmstart import (
            bc_pretrain,
            generate_teacher_dataset,
            ridge_teacher,
            soft_momentum_teacher,
            ts_momentum_teacher,
        )

        teacher = None
        if bc_teacher == "auto":
            from mars_lite.features.signal_check import run_signal_check, run_trend_gate
            from mars_lite.learning.bc_warmstart import combined_teacher

            ic = run_signal_check(full_fs, horizon=horizon, target=signal_target)
            trend = run_trend_gate(full_fs, horizon=horizon)
            use_ridge = ic.mean_oos_ic >= 0.025
            use_trend = trend["has_trend"]
            if use_ridge or use_trend:
                teacher = combined_teacher(
                    fs,
                    use_ridge=use_ridge,
                    use_trend=use_trend,
                    horizon=horizon,
                    ridge_target=signal_target,
                )
                if verbose:
                    components = []
                    if use_ridge:
                        components.append(f"ridge(ic={ic.mean_oos_ic:.3f})")
                    if use_trend:
                        components.append(f"trend(t={trend['t_stat']:.1f})")
                    print(f"[BC auto] teacher = {' + '.join(components)}")
            elif verbose:
                print("[BC auto] no gate passed -> BC disabled (flat prior)")
        elif bc_teacher == "ridge":
            teacher = ridge_teacher(fs, horizon=horizon, target=signal_target)
        elif bc_teacher == "ts_momentum":
            teacher = ts_momentum_teacher()
        elif bc_teacher == "oracle":
            from mars_lite.features.signal_check import run_signal_check
            from mars_lite.learning.bc_warmstart import dp_oracle_teacher

            ic = run_signal_check(full_fs, horizon=horizon, target=signal_target)
            if ic.mean_oos_ic >= 0.025:
                teacher = dp_oracle_teacher(fs, noisy_ic=oracle_noisy_ic)
                if verbose:
                    kind = (
                        f"noisy_ic={oracle_noisy_ic}"
                        if oracle_noisy_ic
                        else "perfect foresight"
                    )
                    print(
                        f"[BC oracle] IC gate passed (ic={ic.mean_oos_ic:.3f}), "
                        f"using DP-oracle teacher ({kind})"
                    )
            elif verbose:
                print(
                    f"[BC oracle] IC gate failed (ic={ic.mean_oos_ic:.3f}) "
                    "-> oracle teacher disabled (flat prior)"
                )
        else:
            teacher = soft_momentum_teacher()

        if teacher is not None:
            X, A = generate_teacher_dataset(fs, teacher, resolved_env_kwargs)
            bc_pretrain(agent, X, A, epochs=bc_epochs, verbose=verbose)

    val_cb = None
    if val_fs is not None:
        if action_mode == "baseline-residual":
            from mars_lite.learning.relative_val_selection import (
                RelativeValSelectionCallback,
                rollout_aligned_eval_freq,
            )

            effective_eval_freq = val_eval_freq or rollout_aligned_eval_freq(
                total_timesteps=timesteps,
                one_rollout_steps=n_envs * n_steps,
                n_eval_targets=10,
            )
            val_cb = RelativeValSelectionCallback(
                val_fs,
                eval_freq=effective_eval_freq,
                env_kwargs=resolved_env_kwargs,
                verbose=verbose,
            )
        else:
            from mars_lite.learning.val_selection import ValSelectionCallback

            val_cb = ValSelectionCallback(
                val_fs,
                eval_freq=val_eval_freq or 20_000,
                env_kwargs=resolved_env_kwargs,
                verbose=verbose,
            )
        callbacks = CallbackList(
            ([callbacks] if callbacks is not None else []) + [val_cb]
        )

    agent.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    if val_cb is not None:
        agent = val_cb.restore_best(agent)
        setattr(agent, "validation_selection", val_cb)
        if verbose:
            print(f"[train_ppo] Restored validation-selected model: {val_cb.best_score}")
    return agent


def build_post_processor(args, horizon: int = 4):
    from mars_lite.data.data_utils import TF_TO_MINUTES
    from mars_lite.trading.post_processor import (
        make_default_processor,
        make_legacy_processor,
    )

    action_mode = getattr(args, "action_mode", "direct")
    no_trade_band = getattr(args, "min_trade_delta", 0.04)
    if action_mode == "baseline-residual":
        no_trade_band = 0.0

    mode = getattr(args, "postproc", "full")
    if mode == "legacy":
        return make_legacy_processor(no_trade_band)
    target_vol = None if getattr(args, "target_vol", 0.5) <= 0 else args.target_vol
    ema_alpha = 0.5
    max_weight = 0.4
    cap = ema_alpha * max_weight * 0.5
    if no_trade_band > cap:
        no_trade_band = cap
    base_tf = getattr(args, "base_timeframe", "1h")
    bars_per_year = int(24 * 60 / TF_TO_MINUTES[base_tf] * 365)
    return make_default_processor(
        target_vol=target_vol,
        ema_alpha=ema_alpha,
        no_trade_band=no_trade_band,
        beta_neutral=getattr(args, "beta_neutral", False),
        bars_per_year=bars_per_year,
    )


def build_env_kwargs(args, pp, horizon: int = 4) -> dict:
    from mars_lite.trading.execution import FEE_PROFILES

    action_mode = getattr(args, "action_mode", "direct")
    min_trade_delta = getattr(args, "min_trade_delta", 0.04)
    if action_mode == "baseline-residual":
        min_trade_delta = 0.0
    env_kwargs = {
        "post_processor": pp,
        "min_trade_delta": min_trade_delta,
        "reward_scale": getattr(args, "reward_scale", 100.0),
        **FEE_PROFILES[getattr(args, "fee_profile", "taker")],
    }
    if action_mode == "direct":
        env_kwargs["lambda_turnover"] = getattr(args, "lambda_turnover", 0.04)
    if getattr(args, "htf_gate", False):
        env_kwargs["htf_gate"] = True
    if getattr(args, "obs_risk_state", False):
        env_kwargs["obs_risk_state"] = True
    if action_mode == "direct":
        disagreement_dr = float(getattr(args, "disagreement_dr", 0.0))
        if disagreement_dr > 0.0:
            env_kwargs["disagreement_dr_max"] = disagreement_dr
    explicit = getattr(args, "decision_every", 1)
    if explicit and explicit > 1:
        env_kwargs["decision_every"] = explicit
    elif getattr(args, "scan_horizons", False) and horizon > 1:
        auto_every = max(1, horizon // 2)
        if auto_every > 1:
            env_kwargs["decision_every"] = auto_every

    pp_cfg = getattr(pp, "cfg", None)
    effective_no_trade_band = getattr(pp_cfg, "no_trade_band", None)
    if pp_cfg is not None and effective_no_trade_band != min_trade_delta:
        print(
            f"[WARN] no_trade_band({effective_no_trade_band}) != "
            f"min_trade_delta({min_trade_delta})"
        )
    print(
        "[Effective config] "
        f"action_mode={action_mode} "
        f"no_trade_band={effective_no_trade_band} "
        f"ema_alpha={getattr(pp_cfg, 'ema_alpha', None)} "
        f"target_vol={getattr(pp_cfg, 'target_vol', None)} "
        f"lambda_turnover={env_kwargs.get('lambda_turnover', 0.0)} "
        f"min_trade_delta={min_trade_delta} "
        f"decision_every={env_kwargs.get('decision_every', 1)}"
    )
    return env_kwargs
