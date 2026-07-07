import json
from pathlib import Path
from typing import Optional, cast

import numpy as np

from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.eval.walk_forward import (
    evaluate_agent_on_slice,
    plot_comparison,
    run_walk_forward,
)
from mars_lite.features.feature_pipeline import FeatureSet
from mars_lite.features.signal_check import run_signal_check
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.learning.manifest import generate_and_save_manifest
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.training_engine import (
    build_env_kwargs,
    build_post_processor,
    train_ppo,
)


def _as_baseline(res: dict, name: str = "generalist") -> dict:
    return {
        "name": name,
        "total_return": res.get("total_return", 0.0),
        "sharpe": res.get("sharpe", 0.0),
        "max_drawdown": res.get("max_drawdown", 0.0),
        "turnover_total": res.get("turnover_total", 0.0),
    }


def report_comparison(agent_res: dict, baselines: dict, label: str) -> None:
    print(f"\n=== {label} ===")
    print(f"{'strategy':<18} {'return':>9} {'sharpe':>8} {'maxDD':>7} {'turnover':>9}")
    print(
        f"{'RL Agent':<18} {agent_res['total_return']:>+8.2%} "
        f"{agent_res['sharpe']:>8.2f} {agent_res['max_drawdown']:>7.2%} "
        f"{agent_res['turnover_total']:>9.1f}"
    )
    for name, r in baselines.items():
        d = r.to_dict() if hasattr(r, "to_dict") else r
        print(
            f"{d['name']:<18} {d['total_return']:>+8.2%} "
            f"{d['sharpe']:>8.2f} {d['max_drawdown']:>7.2%} "
            f"{d['turnover_total']:>9.1f}"
        )
    oracle = baselines.get("oracle_dp")
    if oracle is not None:
        o = oracle.to_dict() if hasattr(oracle, "to_dict") else oracle
        denom = o["total_return"]
        if abs(denom) > 1e-9:
            print(
                f"  capture rate (RL / oracle_dp perfect)  = "
                f"{agent_res['total_return'] / denom:>+.1%}"
            )
    for name, r in baselines.items():
        if not name.startswith("oracle_ic"):
            continue
        d = r.to_dict() if hasattr(r, "to_dict") else r
        denom = d["total_return"]
        if denom > 1e-9:
            print(
                f"  capture rate (RL / {name})  = "
                f"{agent_res['total_return'] / denom:>+.1%}  <- headline"
            )
        else:
            print(
                f"  {name}: 目標ICでもコスト後は黒字化しない"
                f"（return={denom:+.2%}）。捕捉率は算出不能"
                f"（このコスト水準ではより高いICか低コストが必要）"
            )


def phase_p0(args, output_dir: Path) -> None:
    from mars_lite.data.sources import SyntheticSource
    from mars_lite.features.feature_pipeline import FeaturePipeline

    results = {}

    for label, alpha in [
        ("positive(alpha=cross)", "cross"),
        ("negative(alpha=none)", "none"),
    ]:
        print(f"\n{'=' * 60}\nP0 {label}\n{'=' * 60}")
        source = SyntheticSource(
            n_days=args.days,
            alpha=alpha,
            alpha_strength=args.alpha_strength,
            seed=args.seed,
        )
        fs = FeaturePipeline(source.symbols).build(source)

        ic = run_signal_check(fs, horizon=args.horizon)
        print(ic.summary())

        split = int(fs.n_bars * 0.7)
        purge = max(24, args.horizon)
        train_fs = fs.slice(0, split)
        test_fs = fs.slice(split + purge, fs.n_bars)

        print(
            f"\nTraining PPO: {args.timesteps:,} steps "
            f"(train {train_fs.n_bars} bars, test {test_fs.n_bars} bars)..."
        )
        pp = build_post_processor(args, horizon=args.horizon)
        ekw = build_env_kwargs(args, pp, horizon=args.horizon)
        agent = train_ppo(
            fs=train_fs,
            timesteps=args.timesteps,
            seed=args.seed,
            gamma=args.gamma,
            verbose=args.verbose,
            bc_warmstart=True,
            horizon=args.horizon,
            **ekw,
        )

        agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
        baselines = run_all_baselines(test_fs)
        report_comparison(agent_res, baselines, f"OOS comparison: {label}")

        plot_comparison(
            agent_res,
            baselines,
            output_dir / f"p0_{alpha}_equity.png",
            title=f"P0 {label}: RL vs Baselines (OOS)",
        )

        agent_res_slim = {k: v for k, v in agent_res.items() if k != "equity_curve"}
        results[label] = {
            "signal_gate": ic.to_dict(),
            "agent": agent_res_slim,
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
        }
        agent.save(str(output_dir / f"p0_{alpha}_model"))
        hyperparams = {
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "seed": args.seed,
            "decision_every": args.decision_every,
            "min_trade_delta": args.min_trade_delta,
            "lambda_turnover": args.lambda_turnover,
        }
        generate_and_save_manifest(
            output_filepath=str(output_dir / "model_manifest.json"),
            fs=fs,
            hyperparams=hyperparams,
            seed=args.seed,
            additional_metadata={"phase": "p0", "alpha": alpha},
        )

    pos = results["positive(alpha=cross)"]
    neg = results["negative(alpha=none)"]
    pos_beats_bh = (
        pos["agent"]["total_return"]
        > pos["baselines"]["equal_weight_bh"]["total_return"]
    )
    pos_beats_flat = pos["agent"]["total_return"] > 0
    neg_low_activity = (
        neg["agent"]["turnover_total"] < pos["agent"]["turnover_total"] * 0.5
        or abs(neg["agent"]["total_return"]) < 0.05
    )

    gate = {
        "positive_beats_buy_and_hold": bool(pos_beats_bh),
        "positive_beats_flat": bool(pos_beats_flat),
        "negative_stays_quiet": bool(neg_low_activity),
        "P0_PASSED": bool(pos_beats_bh and pos_beats_flat and neg_low_activity),
    }
    results["gate"] = gate

    with open(output_dir / "p0_report.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}\nP0 GATE: {'PASSED' if gate['P0_PASSED'] else 'FAILED'}")
    for k, v in gate.items():
        print(f"  {k}: {v}")
    print(f"Report: {output_dir / 'p0_report.json'}")


def phase_train(
    args,
    output_dir: Path,
    dev_fs: Optional[FeatureSet] = None,
    holdout_fs: Optional[FeatureSet] = None,
) -> Optional[dict]:
    """
    dev_fs/holdout_fs: run_pipeline.py がホールドアウト分離を行う場合に渡す。
    dev_fs 全体を学習に使い、holdout_fs（pbt/wfが一切触れていない区間）を
    最終ゲート2の評価に使う。単独 `--phase train` 実行時はどちらもNoneのままで、
    従来通り fs を自前で80/20に分割する（後方互換）。
    """
    from mars_lite.features.signal_check import run_leak_self_test

    if dev_fs is not None:
        fs = dev_fs
    else:
        try:
            fs = build_feature_set(args, output_dir=output_dir)
        except ValueError as e:
            print(f"\n[STOP] {e}")
            return None
    print(
        f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features"
    )

    leak = run_leak_self_test(fs)
    print(
        f"[Leak self-test] shuffle_ic={leak['shuffle_ic']:.4f} "
        f"future_shift_ic={leak['future_shift_ic']:.4f} healthy={leak['healthy']}"
    )
    if not leak["healthy"]:
        print("[WARN] リーク検出器の自己検査に失敗。評価結果を信用しないこと。")

    signal_target = getattr(args, "target", "raw")

    horizon = args.horizon
    scan = None
    if args.scan_horizons:
        from mars_lite.features.horizon_scan import run_horizon_scan

        scan_split = int(fs.n_bars * 0.8)
        scan = run_horizon_scan(
            fs.slice(0, scan_split),
            horizons=tuple(args.horizons),
            target=signal_target,
        )
        print(scan.summary())
        horizon = scan.best_horizon
        print(f"[Horizon scan] selected horizon={horizon}")

    ic = run_signal_check(fs, horizon=horizon, target=signal_target)
    print(ic.summary())
    if not ic.passed and not args.skip_gate:
        print(
            "\n[STOP] ゲート1不合格: 特徴量に予測力がありません。"
            "RL学習をスキップします（--skip-gate で強制続行可）。"
        )
        return None

    purge = max(24, horizon)
    if holdout_fs is not None:
        # dev_fs全体を学習に使い、pbt/wfが未接触のholdout_fsだけで評価する
        train_fs = fs
        test_fs = holdout_fs
    else:
        split = int(fs.n_bars * 0.8)
        train_fs = fs.slice(0, split)
        test_fs = fs.slice(split + purge, fs.n_bars)

    feature_mask = None
    if args.feature_mask:
        from mars_lite.features.signal_check import compute_feature_mask

        mask_rep = compute_feature_mask(train_fs, horizon=horizon)
        feature_mask = mask_rep["mask"]
        kept = cast(list[str], mask_rep["kept"])
        dropped = cast(list[str], mask_rep["dropped"])
        print(
            f"[Feature mask] kept {len(kept)}/{fs.n_features} features "
            f"(dropped: {', '.join(dropped[:8])}"
            f"{'...' if len(dropped) > 8 else ''})"
        )
        train_fs = train_fs.apply_mask(feature_mask)
        test_fs = test_fs.apply_mask(feature_mask)

    pp = build_post_processor(args, horizon=horizon)
    ekw = build_env_kwargs(args, pp, horizon=horizon)

    if args.ensemble > 1:
        from mars_lite.learning.policy_ensemble import train_ensemble

        print(f"Training {args.ensemble}-seed ensemble x {args.timesteps:,} steps...")

        def _train(train_fs_, seed):
            return train_ppo(
                fs=train_fs_,
                timesteps=args.timesteps,
                seed=seed,
                gamma=args.gamma,
                ent_coef=getattr(args, "ent_coef", 0.002),
                learning_rate=getattr(args, "learning_rate", 3e-4),
                bc_warmstart=True,
                horizon=horizon,
                signal_target=signal_target,
                bc_teacher=args.bc_teacher,
                oracle_noisy_ic=args.oracle_noisy_ic,
                **ekw,
            )

        agent = train_ensemble(
            _train, train_fs, seeds=list(range(args.ensemble)), verbose=1
        )
        agent.save(str(output_dir / "portfolio_ensemble"))
        hyperparams = {
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "ent_coef": getattr(args, "ent_coef", 0.002),
            "learning_rate": getattr(args, "learning_rate", 3e-4),
            "reward_scale": getattr(args, "reward_scale", 100.0),
            "seed": args.seed,
            "ensemble": args.ensemble,
            "decision_every": args.decision_every,
            "min_trade_delta": args.min_trade_delta,
            "lambda_turnover": args.lambda_turnover,
        }
        generate_and_save_manifest(
            output_filepath=str(output_dir / "model_manifest.json"),
            fs=train_fs,
            hyperparams=hyperparams,
            seed=args.seed,
            additional_metadata={"phase": "train", "ensemble": True},
        )
    else:
        print(f"Training PPO: {args.timesteps:,} steps...")
        agent = train_ppo(
            fs=train_fs,
            timesteps=args.timesteps,
            seed=args.seed,
            gamma=args.gamma,
            ent_coef=getattr(args, "ent_coef", 0.002),
            learning_rate=getattr(args, "learning_rate", 3e-4),
            verbose=args.verbose,
            bc_warmstart=True,
            horizon=horizon,
            signal_target=signal_target,
            bc_teacher=args.bc_teacher,
            oracle_noisy_ic=args.oracle_noisy_ic,
            **ekw,
        )

    agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
    noisy_ic = args.noisy_oracle_ic if args.noisy_oracle_ic > 0 else None
    baselines = run_all_baselines(test_fs, noisy_oracle_ic=noisy_ic)
    report_comparison(agent_res, baselines, "OOS comparison")
    plot_comparison(agent_res, baselines, output_dir / "train_equity.png")

    rl_ret = float(agent_res["total_return"])
    gate2_details = {}
    gate2_passed = True
    for bname, bres in baselines.items():
        bd = bres.to_dict() if hasattr(bres, "to_dict") else bres
        beat = bool(rl_ret > float(bd.get("total_return", 0.0)))
        gate2_details[bname] = {
            "rl_return": rl_ret,
            "baseline_return": float(bd.get("total_return", 0.0)),
            "rl_beat": beat,
        }
        if not beat:
            gate2_passed = False
    tf_baseline = gate2_details.get("trend_following", {})
    gate2 = {
        "passed": bool(gate2_passed),
        "rl_beat_trend_following": bool(tf_baseline.get("rl_beat", False))
        if "rl_beat" in tf_baseline
        else None,
        "details": gate2_details,
    }
    print(
        f"\n[Gate 2] {'PASS' if gate2_passed else 'FAIL'} "
        f"RL vs all baselines. trend_following: "
        f"{'BEAT' if tf_baseline.get('rl_beat') else 'LOST'}"
    )

    if args.ensemble <= 1:
        agent.save(str(output_dir / "portfolio_model"))
        hyperparams = {
            "timesteps": args.timesteps,
            "gamma": args.gamma,
            "ent_coef": getattr(args, "ent_coef", 0.002),
            "learning_rate": getattr(args, "learning_rate", 3e-4),
            "reward_scale": getattr(args, "reward_scale", 100.0),
            "seed": args.seed,
            "decision_every": args.decision_every,
            "min_trade_delta": args.min_trade_delta,
            "lambda_turnover": args.lambda_turnover,
        }
        generate_and_save_manifest(
            output_filepath=str(output_dir / "model_manifest.json"),
            fs=train_fs,
            hyperparams=hyperparams,
            seed=args.seed,
            additional_metadata={"phase": "train", "ensemble": False},
        )
    with open(output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "signal_gate": ic.to_dict(),
                "horizon_scan": scan.to_dict() if scan is not None else None,
                "feature_mask": (
                    [bool(x) for x in feature_mask]
                    if feature_mask is not None
                    else None
                ),
                "agent": {k: v for k, v in agent_res.items() if k != "equity_curve"},
                "baselines": {k: v.to_dict() for k, v in baselines.items()},
                "gate2": gate2,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Model & report -> {output_dir}")
    return {"agent_res": agent_res, "baselines": baselines, "gate2": gate2}


def phase_pbt(args, output_dir: Path, fs: Optional[FeatureSet] = None) -> None:
    from mars_lite.learning.pbt_search import run_pbt
    from mars_lite.learning.val_selection import quick_evaluate

    if fs is None:
        fs = build_feature_set(args)
    split = int(fs.n_bars * 0.7)
    train_fs = fs.slice(0, split)
    val_fs = fs.slice(split + 24, fs.n_bars)
    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)
    print(
        f"PBT search: pop={args.pbt_pop} gen={args.pbt_gen} "
        f"steps/individual={args.pbt_steps}"
    )

    def train_eval(hp, seed):
        ekw_hp = {
            **ekw,
            "lambda_turnover": hp["lambda_turnover"],
            "reward_scale": hp["reward_scale"],
        }
        agent = train_ppo(
            fs=train_fs,
            timesteps=args.pbt_steps,
            seed=int(seed),
            gamma=hp["gamma"],
            ent_coef=hp["ent_coef"],
            learning_rate=hp["learning_rate"],
            bc_warmstart=True,
            **ekw_hp,
        )
        return quick_evaluate(agent, val_fs, **ekw_hp)

    result = run_pbt(
        train_eval,
        population_size=args.pbt_pop,
        n_generations=args.pbt_gen,
        seed=args.seed,
    )
    print(f"\nBest HP (val score {result.best_score:+.4f}):")
    for k, v in result.best_hp.items():
        print(f"  {k} = {v:.5f}")
    with open(output_dir / "pbt_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Result -> {output_dir / 'pbt_result.json'}")


def phase_regime(args, output_dir: Path) -> None:
    """レジーム特化アンサンブル（8状態化 + 自動較正 + 専門家学習）"""
    from mars_lite.learning.regime_calibrator import RegimeCalibrator
    from mars_lite.learning.regime_ensemble import RegimeEnsemble, regime_start_pools
    from mars_lite.learning.regime_fsm import REGIMES_8, RegimeFSM

    fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")
    split = int(fs.n_bars * 0.8)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + 24, fs.n_bars)

    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)

    # 1. 自動較正の実行
    calibrate_params = {}
    if getattr(args, "calibrate_regime", True):
        n_trials = getattr(args, "regime_trials", 100)
        print(f"Running RegimeCalibrator with {n_trials} trials...")
        calibrator = RegimeCalibrator(
            n_trials=n_trials, penalty_coef=1.0, seed=args.seed
        )
        calibrate_params = calibrator.calibrate(train_fs)
        print(f"Calibrated best params: {calibrate_params}")
    else:
        # デフォルトパラメータ
        calibrate_params = {
            "t_trend_low": 0.5,
            "t_trend_extreme": 1.5,
            "t_vol": 0.0,
            "persistence_bars": 5,
        }

    # 2. 8状態 FSM の構築
    fsm = RegimeFSM(
        t_trend_low=calibrate_params["t_trend_low"],
        t_trend_extreme=calibrate_params["t_trend_extreme"],
        t_vol=calibrate_params["t_vol"],
        persistence_bars=int(calibrate_params["persistence_bars"]),
        initial_state="range_low",
    )

    spec_horizon = min(args.regime_bars, train_fs.n_bars // 4)

    # 3. 8状態開始位置プールの作成
    pools = regime_start_pools(
        train_fs, horizon=spec_horizon, min_fraction=0.45, fsm=fsm
    )

    # データ分布の集計
    from mars_lite.features.feature_pipeline import GLOBAL_FEATURES

    vol_idx = list(GLOBAL_FEATURES).index("btc_vol_regime")
    trend_idx = list(GLOBAL_FEATURES).index("btc_trend")
    vol_series = train_fs.global_features[:, vol_idx]
    trend_series = train_fs.global_features[:, trend_idx]
    labels_train = fsm.classify_series(trend_series, vol_series)
    dist = {r: int((labels_train == r).sum()) for r in REGIMES_8}

    print(f"Regime bar distribution (train): {dist}")
    print(f"Specialist episode horizon: {spec_horizon} bars")
    for r in REGIMES_8:
        print(f"  pool[{r}] = {len(pools[r])} start positions")

    # 汎用方策の学習
    print(f"\nTraining generalist: {args.timesteps:,} steps...")
    generalist = train_ppo(
        fs=train_fs,
        timesteps=args.timesteps,
        seed=args.seed,
        gamma=args.gamma,
        bc_warmstart=True,
        **ekw,
    )

    # 専門家方策の学習
    spec_ekw = {**ekw, "episode_bars": spec_horizon}
    min_pool = 20
    specialists = {}
    for r in REGIMES_8:
        if len(pools[r]) < min_pool:
            print(
                f"[skip] specialist[{r}]: pool too small "
                f"({len(pools[r])}<{min_pool}) -> generalistで代替"
            )
            continue
        print(
            f"\nTraining specialist[{r}]: {args.timesteps:,} steps "
            f"(pool={len(pools[r])})..."
        )
        specialists[r] = train_ppo(
            fs=train_fs,
            timesteps=args.timesteps,
            seed=args.seed + 1,
            gamma=args.gamma,
            bc_warmstart=True,
            regime_start_pool=pools[r],
            **spec_ekw,
        )

    # 8状態用 RegimeEnsemble 構築
    ensemble = RegimeEnsemble(
        specialists=specialists,
        generalist=generalist,
        obs_layout=PortfolioTradingEnv(train_fs, **ekw).obs_layout,
        n_raw_globals=fs.global_features.shape[1],
        fsm=fsm,
    )

    ens_res = evaluate_agent_on_slice(ensemble, test_fs, **ekw)
    gen_res = evaluate_agent_on_slice(generalist, test_fs, **ekw)
    baselines = run_all_baselines(test_fs)

    report_comparison(
        ens_res,
        {"generalist": _as_baseline(gen_res), **baselines},
        "OOS: RegimeEnsemble vs generalist vs baselines",
    )
    print(f"Route counts (test): {ensemble.route_counts}")

    plot_comparison(ens_res, baselines, output_dir / "regime_equity.png")

    with open(output_dir / "regime_report.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "calibrate_params": calibrate_params,
                "regime_distribution": dist,
                "pool_sizes": {r: len(pools[r]) for r in REGIMES_8},
                "specialists_trained": list(specialists.keys()),
                "route_counts": ensemble.route_counts,
                "ensemble": {k: v for k, v in ens_res.items() if k != "equity_curve"},
                "generalist": {k: v for k, v in gen_res.items() if k != "equity_curve"},
                "baselines": {k: v.to_dict() for k, v in baselines.items()},
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    print(f"Regime ensemble report -> {output_dir}")


def phase_wf(args, output_dir: Path, fs: Optional[FeatureSet] = None) -> None:
    if fs is None:
        try:
            fs = build_feature_set(args, output_dir=output_dir)
        except ValueError as e:
            print(f"\n[STOP] {e}")
            return
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")

    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)

    n_ensemble = max(args.ensemble, 1)
    if n_ensemble > 1:
        from mars_lite.learning.policy_ensemble import train_ensemble

        def train_fn(train_fs: FeatureSet, seed: int):
            def _inner(train_fs_: FeatureSet, _seed: int):
                return train_ppo(
                    fs=train_fs_,
                    timesteps=args.timesteps,
                    seed=_seed,
                    gamma=args.gamma,
                    ent_coef=getattr(args, "ent_coef", 0.002),
                    learning_rate=getattr(args, "learning_rate", 3e-4),
                    bc_warmstart=True,
                    **ekw,
                )

            return train_ensemble(
                _inner, train_fs, seeds=list(range(seed, seed + n_ensemble)), verbose=0
            )
    else:

        def train_fn(train_fs: FeatureSet, seed: int):
            return train_ppo(
                fs=train_fs,
                timesteps=args.timesteps,
                seed=seed,
                gamma=args.gamma,
                ent_coef=getattr(args, "ent_coef", 0.002),
                learning_rate=getattr(args, "learning_rate", 3e-4),
                bc_warmstart=True,
                **ekw,
            )

    for cost_mult in [1.0, 2.0]:
        print(f"\n--- Walk-forward (cost x{cost_mult}, ensemble={n_ensemble}) ---")
        report = run_walk_forward(
            fs,
            train_fn,
            n_folds=args.folds,
            seeds=list(range(args.n_seeds)),
            cost_multiplier=cost_mult,
            env_kwargs=ekw,
        )
        path = output_dir / f"walk_forward_cost{cost_mult:.0f}x.json"
        report.save(path)
        print(json.dumps(report.summary(), indent=2))
        print(f"Report -> {path}")
