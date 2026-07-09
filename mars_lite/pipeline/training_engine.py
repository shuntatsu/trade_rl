from typing import Optional

import numpy as np

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.features.feature_pipeline import FeatureSet


def make_env_fns(fs: FeatureSet, n_envs: int, seed: int, **env_kwargs):
    from stable_baselines3.common.monitor import Monitor

    def make_one(rank: int):
        def _init():
            env = PortfolioTradingEnv(fs, **env_kwargs)
            env.reset(seed=seed + rank)
            return Monitor(env)

        return _init

    return [make_one(i) for i in range(n_envs)]


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
    val_eval_freq: int = 20_000,
    bc_warmstart: bool = True,
    bc_epochs: int = 15,
    bc_teacher: str = "auto",
    extractor: str = "tfgated",
    horizon: int = 4,
    signal_target: str = "raw",
    oracle_noisy_ic: Optional[float] = None,
    **env_kwargs,
):
    """signal_target: ICゲート判定とRidge教師の予測対象（raw/cs_demean/vol_norm）。
    絶対リターンに信号が無く相対アルファのみ有意な市場では cs_demean を指定すると、
    ゲート判定・Ridge教師の両方が同じ市場中立の対象を見る。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CallbackList
    from stable_baselines3.common.vec_env import DummyVecEnv

    from mars_lite.learning.val_selection import ValSelectionCallback

    full_fs = fs
    if val_fs is None and fs.n_bars > 400:
        cut = int(fs.n_bars * 0.85)
        val_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)

    env = DummyVecEnv(make_env_fns(fs, n_envs, seed, **env_kwargs))
    probe = PortfolioTradingEnv(fs, **env_kwargs)

    from mars_lite.features.feature_pipeline import TF_BLOCK_FEATURES

    tf_prefixes = []
    for name in fs.feature_names:
        p = name.split("_")[0]
        if p in ("15m", "30m", "1h", "4h", "1d") and p not in tf_prefixes:
            tf_prefixes.append(p)

    if extractor == "tfgated" and tf_prefixes:
        from mars_lite.models.portfolio_extractor import TFGatedPortfolioExtractor

        policy_kwargs = {
            "features_extractor_class": TFGatedPortfolioExtractor,
            "features_extractor_kwargs": {
                **probe.obs_layout,
                "n_tf_blocks": len(tf_prefixes),
                "tf_block_size": len(TF_BLOCK_FEATURES),
                # size="small"がARCHITECTURE.mdのベンチ実績構成。"large"は容量を
                # 増やした未検証構成（要再ベンチ）なのでここでは明示的に固定する
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
        n_steps=256,
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
                    comps = []
                    if use_ridge:
                        comps.append(f"ridge(ic={ic.mean_oos_ic:.3f})")
                    if use_trend:
                        comps.append(f"trend(t={trend['t_stat']:.1f})")
                    print(f"[BC auto] teacher = {' + '.join(comps)}")
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
                    "-> oracle teacher disabled (flat prior); "
                    "特権教師を模倣する意味がない（ノイズの丸暗記になる）"
                )
        else:
            teacher = soft_momentum_teacher()

        if teacher is not None:
            X, A = generate_teacher_dataset(fs, teacher, env_kwargs)
            bc_pretrain(agent, X, A, epochs=bc_epochs, verbose=verbose)

    val_cb = None
    if val_fs is not None:
        val_cb = ValSelectionCallback(
            val_fs,
            eval_freq=val_eval_freq,
            env_kwargs=env_kwargs,
            verbose=verbose,
        )
        callbacks = CallbackList(
            ([callbacks] if callbacks is not None else []) + [val_cb]
        )

    agent.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    if val_cb is not None:
        agent = val_cb.restore_best(agent)
        if verbose:
            print(
                f"[train_ppo] Restored best-val model (score={val_cb.best_score:+.4f})"
            )
    return agent


def build_post_processor(args, horizon: int = 4):
    from mars_lite.data.data_utils import TF_TO_MINUTES
    from mars_lite.trading.post_processor import (
        make_default_processor,
        make_legacy_processor,
    )

    mode = getattr(args, "postproc", "full")
    if mode == "legacy":
        return make_legacy_processor()
    tv = None if getattr(args, "target_vol", 0.5) <= 0 else args.target_vol
    # 低頻度化は decision_every が担うため、ema_alpha/no_trade_band を
    # horizon で強くスケールしない（マイクロノイズ除去に役割を限定する）。
    ema_alpha = 0.5
    no_trade_band = 0.04
    max_weight = 0.4
    # 不変条件: no_trade_band <= ema_alpha * max_weight * 0.5
    # これを破ると「1ステップで動ける最大量 < 発注しきい値」となり、
    # ウェイトが初期値(通常0)から一歩も動けず永久に据え置かれる
    # （decision_every とは独立に発生するデッドゾーン）。
    cap = ema_alpha * max_weight * 0.5
    if no_trade_band > cap:
        no_trade_band = cap
    # ④ボラターゲティングの年率換算はbase_timeframeの実バー数に合わせる。
    # 1h想定のまま4h等を使うと推定年率ボラが水増しされ、target_volへの
    # スケーリングが過剰にグロスを絞ってしまう。
    base_tf = getattr(args, "base_timeframe", "1h")
    bars_per_year = int(24 * 60 / TF_TO_MINUTES[base_tf] * 365)
    return make_default_processor(
        target_vol=tv,
        ema_alpha=ema_alpha,
        no_trade_band=no_trade_band,
        beta_neutral=getattr(args, "beta_neutral", False),
        bars_per_year=bars_per_year,
    )


def build_env_kwargs(args, pp, horizon: int = 4) -> dict:
    from mars_lite.trading.execution import FEE_PROFILES

    ekw = {
        "post_processor": pp,
        "min_trade_delta": getattr(args, "min_trade_delta", 0.04),
        "lambda_turnover": getattr(args, "lambda_turnover", 0.04),
        "reward_scale": getattr(args, "reward_scale", 100.0),
        **FEE_PROFILES[getattr(args, "fee_profile", "taker")],
    }
    if getattr(args, "htf_gate", False):
        ekw["htf_gate"] = True
    explicit = getattr(args, "decision_every", 1)
    if explicit and explicit > 1:
        ekw["decision_every"] = explicit
    elif getattr(args, "scan_horizons", False) and horizon > 1:
        auto_every = max(1, horizon // 2)
        if auto_every > 1:
            ekw["decision_every"] = auto_every
    return ekw
