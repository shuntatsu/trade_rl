"""
ポートフォリオRL学習スクリプト

フェーズ:
    p0    : 健全性試験。アルファ注入合成データ（陽性対照）と純ノイズ（陰性対照）
            の両方で学習し、①陽性でベースライン超え ②陰性で低回転 を確認する
    train : 指定ソースで学習（P2。実データはローカルPCで --source csv/postgres）
    wf    : ウォークフォワード検証（P3）

使い方:
    python scripts/train_portfolio.py --phase p0 --timesteps 300000
    python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000
    python scripts/train_portfolio.py --phase wf --source csv --data ./data
"""

import argparse
import json
from pathlib import Path

import numpy as np

from mars_lite.data.sources import create_source, SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline, FeatureSet
from mars_lite.features.signal_check import run_signal_check
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.models.portfolio_extractor import PortfolioExtractor
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.eval.walk_forward import (
    evaluate_agent_on_slice, run_walk_forward, plot_comparison,
)

DEFAULT_SYMBOLS = [
    "BTCUSDT", "XRPUSDT", "SUIUSDT", "BNBUSDT", "ETHUSDT", "PAXGUSDT", "ETHBTC",
]


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
    **env_kwargs,
):
    """FeatureSetでPPOを学習して返す

    val_fs を渡すと検証スライスで定期評価し、最良時点のパラメータを
    最終モデルとして採用する（小データへの過学習対策）。
    val_fs 省略時は fs の末尾15%を自動で検証に割く。
    bc_warmstart=True でBC事前学習を行う。教師はbc_teacherで選択:
    ridge（デフォルト）= 学習スライスのRidge予測（アルファの型を仮定しない）、
    momentum = クロスモメンタム固定教師。
    """
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import CallbackList
    from mars_lite.learning.val_selection import ValSelectionCallback

    if val_fs is None and fs.n_bars > 400:
        cut = int(fs.n_bars * 0.85)
        val_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)

    env = DummyVecEnv(make_env_fns(fs, n_envs, seed, **env_kwargs))
    probe = PortfolioTradingEnv(fs, **env_kwargs)

    # TFブロック構造を特徴名から導出（例: "15m_ret_z1" → TFプレフィックス）
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
            },
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }
    else:
        policy_kwargs = {
            "features_extractor_class": PortfolioExtractor,
            "features_extractor_kwargs": probe.obs_layout,
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        }

    def lr_schedule(progress_remaining: float) -> float:
        return learning_rate * progress_remaining

    agent = PPO(
        "MlpPolicy", env,
        policy_kwargs=policy_kwargs,
        learning_rate=lr_schedule,
        n_steps=256, batch_size=256, n_epochs=6,
        gamma=gamma, gae_lambda=0.9,
        ent_coef=ent_coef, vf_coef=0.5, max_grad_norm=0.5,
        seed=seed, device="cpu", verbose=verbose,
    )

    # BCウォームスタート（教師方策の模倣で方策を初期化）
    # bc_teacher="auto": 学習スライスのゲートで教師を自動選択
    #   クロスセクショナルIC合格 → ridge（相対アルファ）
    #   方向性トレンド合格     → ts_momentum（ベータ捕捉。上昇相場でB&Hに勝つ）
    #   どちらも無し           → BC無効（フラットで待つ＝安全）
    # ridge/momentum/ts_momentum を明示指定も可。
    if bc_warmstart:
        from mars_lite.learning.bc_warmstart import (
            soft_momentum_teacher, ridge_teacher, ts_momentum_teacher,
            generate_teacher_dataset, bc_pretrain,
        )
        teacher = None
        if bc_teacher == "auto":
            from mars_lite.features.signal_check import run_signal_check, run_trend_gate
            from mars_lite.learning.bc_warmstart import combined_teacher
            ic = run_signal_check(fs)
            trend = run_trend_gate(fs)
            # Ridgeは偽陽性回避のため閾値+マージンを要求
            use_ridge = ic.mean_oos_ic >= 0.025
            use_trend = trend["has_trend"]
            if use_ridge or use_trend:
                teacher = combined_teacher(fs, use_ridge=use_ridge, use_trend=use_trend)
                if verbose:
                    comps = []
                    if use_ridge: comps.append(f"ridge(ic={ic.mean_oos_ic:.3f})")
                    if use_trend: comps.append(f"trend(t={trend['t_stat']:.1f})")
                    print(f"[BC auto] teacher = {' + '.join(comps)}")
            elif verbose:
                print("[BC auto] no gate passed -> BC disabled (flat prior)")
        elif bc_teacher == "ridge":
            teacher = ridge_teacher(fs)
        elif bc_teacher == "ts_momentum":
            teacher = ts_momentum_teacher()
        else:
            teacher = soft_momentum_teacher()

        if teacher is not None:
            X, A = generate_teacher_dataset(fs, teacher, env_kwargs)
            bc_pretrain(agent, X, A, epochs=bc_epochs, verbose=verbose)

    val_cb = None
    if val_fs is not None:
        val_cb = ValSelectionCallback(
            val_fs, eval_freq=val_eval_freq, env_kwargs=env_kwargs,
            verbose=verbose,
        )
        callbacks = CallbackList(
            ([callbacks] if callbacks is not None else []) + [val_cb]
        )

    agent.learn(total_timesteps=timesteps, callback=callbacks, progress_bar=False)

    if val_cb is not None:
        agent = val_cb.restore_best(agent)
        if verbose:
            print(f"[train_ppo] Restored best-val model (score={val_cb.best_score:+.4f})")
    return agent


def _as_baseline(res: dict, name: str = "generalist") -> dict:
    """エージェント評価結果を report_comparison が読める baseline 風dictに整形"""
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
    print(f"{'RL Agent':<18} {agent_res['total_return']:>+8.2%} "
          f"{agent_res['sharpe']:>8.2f} {agent_res['max_drawdown']:>7.2%} "
          f"{agent_res['turnover_total']:>9.1f}")
    for name, r in baselines.items():
        d = r.to_dict() if hasattr(r, "to_dict") else r
        print(f"{d['name']:<18} {d['total_return']:>+8.2%} "
              f"{d['sharpe']:>8.2f} {d['max_drawdown']:>7.2%} "
              f"{d['turnover_total']:>9.1f}")


def build_post_processor(args):
    """CLIフラグから後処理器を構築（デフォルトは推奨フル後処理）"""
    from mars_lite.trading.post_processor import (
        make_default_processor, make_legacy_processor,
    )
    mode = getattr(args, "postproc", "full")
    if mode == "legacy":
        return make_legacy_processor()
    tv = None if getattr(args, "target_vol", 0.5) <= 0 else args.target_vol
    return make_default_processor(target_vol=tv)


def build_env_kwargs(args, pp) -> dict:
    """学習/評価で共有する環境kwargs（後処理器 + 任意のHTFゲート）"""
    ekw = {"post_processor": pp}
    if getattr(args, "htf_gate", False):
        ekw["htf_gate"] = True
    return ekw


def build_feature_set(args) -> FeatureSet:
    if args.source == "synthetic":
        source = SyntheticSource(
            n_days=args.days, alpha=args.alpha,
            alpha_strength=args.alpha_strength, seed=args.seed,
        )
        symbols = source.symbols
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        kwargs = {"data_dir": args.data} if args.source == "csv" else {}
        source = create_source(args.source, symbols, **kwargs)
    return FeaturePipeline(symbols).build(source)


def phase_p0(args, output_dir: Path) -> None:
    """P0健全性試験: 陽性対照（アルファ有）と陰性対照（ノイズ）"""
    results = {}

    for label, alpha in [("positive(alpha=cross)", "cross"), ("negative(alpha=none)", "none")]:
        print(f"\n{'=' * 60}\nP0 {label}\n{'=' * 60}")
        source = SyntheticSource(
            n_days=args.days, alpha=alpha,
            alpha_strength=args.alpha_strength, seed=args.seed,
        )
        fs = FeaturePipeline(source.symbols).build(source)

        # ICゲート（陽性はPASS、陰性はFAILが期待値）
        ic = run_signal_check(fs)
        print(ic.summary())

        # train/test 時系列分割（70/30、purge 24本）
        split = int(fs.n_bars * 0.7)
        train_fs = fs.slice(0, split)
        test_fs = fs.slice(split + 24, fs.n_bars)

        print(f"\nTraining PPO: {args.timesteps:,} steps "
              f"(train {train_fs.n_bars} bars, test {test_fs.n_bars} bars)...")
        pp = build_post_processor(args)
        ekw = build_env_kwargs(args, pp)
        # BCウォームスタートはICゲート合格時のみ（信号なきデータで教師を
        # 模倣するとノイズを刷り込み、陰性対照の安全性を壊すため）
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          gamma=args.gamma, verbose=args.verbose,
                          bc_warmstart=True, **ekw)

        agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
        baselines = run_all_baselines(test_fs)
        report_comparison(agent_res, baselines, f"OOS comparison: {label}")

        plot_comparison(
            agent_res, baselines,
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

    # ゲート判定
    pos = results["positive(alpha=cross)"]
    neg = results["negative(alpha=none)"]
    pos_beats_bh = pos["agent"]["total_return"] > pos["baselines"]["equal_weight_bh"]["total_return"]
    pos_beats_flat = pos["agent"]["total_return"] > 0
    neg_low_activity = neg["agent"]["turnover_total"] < pos["agent"]["turnover_total"] * 0.5 \
        or abs(neg["agent"]["total_return"]) < 0.05

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


def phase_train(args, output_dir: Path) -> None:
    """P2: 指定ソースで学習・評価（品質ゲート → リーク自己検査 → ICゲート → 学習）"""
    from mars_lite.data.quality import run_quality_gate
    from mars_lite.features.signal_check import run_leak_self_test

    # --- 品質ゲート（実データソースのみ。不合格銘柄は除外） ---
    symbols = args.symbols or DEFAULT_SYMBOLS
    if args.source != "synthetic":
        kwargs = {"data_dir": args.data} if args.source == "csv" else {}
        source = create_source(args.source, symbols, **kwargs)
        qrep = run_quality_gate(source, symbols, base_timeframe="1h")
        print(qrep.summary())
        symbols = qrep.passing_symbols
        with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
            json.dump(qrep.to_dict(), f, indent=2, ensure_ascii=False)
        if len(symbols) < 2:
            print("\n[STOP] 品質ゲート通過銘柄が2未満。データを確認してください。")
            return
        fs = FeaturePipeline(symbols).build(source)
    else:
        fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features")

    # --- リーク検出器の自己検査（評価コードの健全性確認） ---
    leak = run_leak_self_test(fs)
    print(f"[Leak self-test] shuffle_ic={leak['shuffle_ic']:.4f} "
          f"future_shift_ic={leak['future_shift_ic']:.4f} healthy={leak['healthy']}")
    if not leak["healthy"]:
        print("[WARN] リーク検出器の自己検査に失敗。評価結果を信用しないこと。")

    ic = run_signal_check(fs)
    print(ic.summary())
    if not ic.passed and not args.skip_gate:
        print("\n[STOP] ゲート1不合格: 特徴量に予測力がありません。"
              "RL学習をスキップします（--skip-gate で強制続行可）。")
        return

    split = int(fs.n_bars * 0.8)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + 24, fs.n_bars)

    # IC安定性マスク（オプトイン。時間軸の冗長性はTFゲート構造が既定で処理する）
    feature_mask = None
    if args.feature_mask:
        from mars_lite.features.signal_check import compute_feature_mask
        mask_rep = compute_feature_mask(train_fs)
        feature_mask = mask_rep["mask"]
        print(f"[Feature mask] kept {len(mask_rep['kept'])}/{fs.n_features} features "
              f"(dropped: {', '.join(mask_rep['dropped'][:8])}"
              f"{'...' if len(mask_rep['dropped']) > 8 else ''})")
        train_fs = train_fs.apply_mask(feature_mask)
        test_fs = test_fs.apply_mask(feature_mask)

    pp = build_post_processor(args)
    ekw = build_env_kwargs(args, pp)

    if args.ensemble > 1:
        from mars_lite.learning.policy_ensemble import train_ensemble
        print(f"Training {args.ensemble}-seed ensemble x {args.timesteps:,} steps...")

        def _train(train_fs_, seed):
            return train_ppo(fs=train_fs_, timesteps=args.timesteps, seed=seed,
                             gamma=args.gamma, bc_warmstart=True, **ekw)
        agent = train_ensemble(_train, train_fs, seeds=list(range(args.ensemble)),
                               verbose=1)
        agent.save(str(output_dir / "portfolio_ensemble"))
    else:
        print(f"Training PPO: {args.timesteps:,} steps...")
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          gamma=args.gamma, verbose=args.verbose,
                          bc_warmstart=True, **ekw)

    agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
    baselines = run_all_baselines(test_fs)
    report_comparison(agent_res, baselines, "OOS comparison")
    plot_comparison(agent_res, baselines, output_dir / "train_equity.png")

    if args.ensemble <= 1:
        agent.save(str(output_dir / "portfolio_model"))
    with open(output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "signal_gate": ic.to_dict(),
            "feature_mask": ([bool(x) for x in feature_mask]
                             if feature_mask is not None else None),
            "agent": {k: v for k, v in agent_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"Model & report -> {output_dir}")


def phase_pbt(args, output_dir: Path) -> None:
    """項目4: PBTハイパーパラメータ探索（gamma/ent_coef/lambda_turnover等）"""
    from mars_lite.learning.pbt_search import run_pbt
    from mars_lite.learning.val_selection import quick_evaluate

    fs = build_feature_set(args)
    split = int(fs.n_bars * 0.7)
    train_fs = fs.slice(0, split)
    val_fs = fs.slice(split + 24, fs.n_bars)
    pp = build_post_processor(args)
    ekw = build_env_kwargs(args, pp)
    print(f"PBT search: pop={args.pbt_pop} gen={args.pbt_gen} "
          f"steps/individual={args.pbt_steps}")

    def train_eval(hp, seed):
        agent = train_ppo(
            fs=train_fs, timesteps=args.pbt_steps, seed=int(seed),
            gamma=hp["gamma"], ent_coef=hp["ent_coef"],
            learning_rate=hp["learning_rate"],
            lambda_turnover=hp["lambda_turnover"],
            reward_scale=hp["reward_scale"],
            bc_warmstart=True, **ekw,
        )
        return quick_evaluate(agent, val_fs, **ekw)

    result = run_pbt(train_eval, population_size=args.pbt_pop,
                     n_generations=args.pbt_gen, seed=args.seed)
    print(f"\nBest HP (val score {result.best_score:+.4f}):")
    for k, v in result.best_hp.items():
        print(f"  {k} = {v:.5f}")
    with open(output_dir / "pbt_result.json", "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
    print(f"Result -> {output_dir / 'pbt_result.json'}")


def phase_regime(args, output_dir: Path) -> None:
    """項目3: レジーム特化アンサンブル（bull/bear/range専門家 + ルーティング）"""
    from mars_lite.learning.regime_ensemble import (
        regime_labels, regime_start_pools, RegimeEnsemble, REGIMES,
    )

    fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")
    split = int(fs.n_bars * 0.8)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + 24, fs.n_bars)

    pp = build_post_processor(args)
    ekw = build_env_kwargs(args, pp)

    # 専門家は短いエピソード（レジームが一貫する長さ）で学習する。
    # 30日窓では強気/弱気/レンジが混在し「純粋な」エピソードが作れないため、
    # 5日窓（120本）でレジーム優勢な開始位置プールを作る。
    spec_horizon = min(args.regime_bars, train_fs.n_bars // 4)
    pools = regime_start_pools(train_fs, horizon=spec_horizon, min_fraction=0.45)
    labels = regime_labels(train_fs)
    dist = {r: int((labels == r).sum()) for r in REGIMES}
    print(f"Regime bar distribution (train): {dist}")
    print(f"Specialist episode horizon: {spec_horizon} bars")
    for r in REGIMES:
        print(f"  pool[{r}] = {len(pools[r])} start positions")

    # 汎用方策（フォールバック兼、専門家プールが薄いレジーム用）
    print(f"\nTraining generalist: {args.timesteps:,} steps...")
    generalist = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                           gamma=args.gamma, bc_warmstart=True, **ekw)

    # 十分なプールがあるレジームのみ専門家を学習（少数は汎用にフォールバック）
    spec_ekw = {**ekw, "episode_bars": spec_horizon}
    min_pool = 20
    specialists = {}
    for r in REGIMES:
        if len(pools[r]) < min_pool:
            print(f"[skip] specialist[{r}]: pool too small "
                  f"({len(pools[r])}<{min_pool}) -> generalistで代替")
            continue
        print(f"\nTraining specialist[{r}]: {args.timesteps:,} steps "
              f"(pool={len(pools[r])})...")
        specialists[r] = train_ppo(
            fs=train_fs, timesteps=args.timesteps, seed=args.seed + 1,
            gamma=args.gamma, bc_warmstart=True,
            regime_start_pool=pools[r], **spec_ekw,
        )

    ensemble = RegimeEnsemble(
        specialists=specialists, generalist=generalist,
        obs_layout=PortfolioTradingEnv(train_fs, **ekw).obs_layout,
        n_raw_globals=fs.global_features.shape[1],
    )

    # 評価: レジームアンサンブル vs 汎用単体 vs ベースライン
    ens_res = evaluate_agent_on_slice(ensemble, test_fs, **ekw)
    gen_res = evaluate_agent_on_slice(generalist, test_fs, **ekw)
    baselines = run_all_baselines(test_fs)
    report_comparison(ens_res, {"generalist": _as_baseline(gen_res), **baselines},
                      "OOS: RegimeEnsemble vs generalist vs baselines")
    print(f"Route counts (test): {ensemble.route_counts}")

    plot_comparison(ens_res, baselines, output_dir / "regime_equity.png")
    with open(output_dir / "regime_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "regime_distribution": dist,
            "pool_sizes": {r: len(pools[r]) for r in REGIMES},
            "specialists_trained": list(specialists.keys()),
            "route_counts": ensemble.route_counts,
            "ensemble": {k: v for k, v in ens_res.items() if k != "equity_curve"},
            "generalist": {k: v for k, v in gen_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
        }, f, indent=2, ensure_ascii=False)
    print(f"Regime ensemble report -> {output_dir}")


def phase_wf(args, output_dir: Path) -> None:
    """P3: ウォークフォワード検証（複数シード、コスト感度）"""
    fs = build_feature_set(args)
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")

    pp = build_post_processor(args)
    ekw = build_env_kwargs(args, pp)

    def train_fn(train_fs: FeatureSet, seed: int):
        return train_ppo(fs=train_fs, timesteps=args.timesteps, seed=seed,
                         gamma=args.gamma, **ekw)

    for cost_mult in [1.0, 2.0]:
        print(f"\n--- Walk-forward (cost x{cost_mult}) ---")
        report = run_walk_forward(
            fs, train_fn,
            n_folds=args.folds,
            seeds=list(range(args.n_seeds)),
            cost_multiplier=cost_mult,
            env_kwargs=ekw,
        )
        path = output_dir / f"walk_forward_cost{cost_mult:.0f}x.json"
        report.save(path)
        print(json.dumps(report.summary(), indent=2))
        print(f"Report -> {path}")


def main():
    parser = argparse.ArgumentParser(description="ポートフォリオRL学習")
    parser.add_argument("--phase", choices=["p0", "train", "wf", "pbt", "regime"],
                        default="p0")
    parser.add_argument("--source", choices=["synthetic", "csv", "postgres"],
                        default="synthetic")
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--days", type=int, default=240, help="syntheticの生成日数")
    parser.add_argument("--alpha", default="cross",
                        choices=["none", "cross", "meanrev", "multi", "bull"])
    parser.add_argument("--alpha-strength", type=float, default=0.002)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument("--gamma", type=float, default=0.5,
                        help="割引率。行動効果が即時のため低い値が有効")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="./output/portfolio")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument("--postproc", choices=["full", "legacy"], default="full",
                        help="後処理: full=推奨(平滑/バンド/ボラ目標/DDデリスク), legacy=射影のみ")
    parser.add_argument("--target-vol", type=float, default=0.5,
                        help="年率ボラ目標。0以下で無効")
    parser.add_argument("--ensemble", type=int, default=1,
                        help="シードアンサンブルの個体数（1で単一モデル）")
    parser.add_argument("--feature-mask", action="store_true",
                        help="IC安定性による特徴マスクを有効化（実験では中立〜微減。"
                             "実データでジャンク特徴が多い場合のオプション）")
    parser.add_argument("--pbt-pop", type=int, default=6,
                        help="PBT個体数（--phase pbt）")
    parser.add_argument("--pbt-gen", type=int, default=4,
                        help="PBT世代数（--phase pbt）")
    parser.add_argument("--pbt-steps", type=int, default=40_000,
                        help="PBT各個体の学習ステップ数（--phase pbt）")
    parser.add_argument("--regime-bars", type=int, default=120,
                        help="レジーム専門家のエピソード長（--phase regime、5日=120本）")
    parser.add_argument("--htf-gate", action="store_true",
                        help="階層MTF: 上位足(4h)トレンドで方向を制約し1hはサイジング")
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.phase == "p0":
        phase_p0(args, output_dir)
    elif args.phase == "train":
        phase_train(args, output_dir)
    elif args.phase == "pbt":
        phase_pbt(args, output_dir)
    elif args.phase == "regime":
        phase_regime(args, output_dir)
    else:
        phase_wf(args, output_dir)


if __name__ == "__main__":
    main()
