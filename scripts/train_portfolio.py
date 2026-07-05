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
from typing import Optional

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
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
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
    horizon: int = 4,
    oracle_noisy_ic: Optional[float] = None,
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
            ic = run_signal_check(fs, horizon=horizon)
            trend = run_trend_gate(fs, horizon=horizon)
            # Ridgeは偽陽性回避のため閾値+マージンを要求
            use_ridge = ic.mean_oos_ic >= 0.025
            use_trend = trend["has_trend"]
            if use_ridge or use_trend:
                teacher = combined_teacher(fs, use_ridge=use_ridge, use_trend=use_trend, horizon=horizon)
                if verbose:
                    comps = []
                    if use_ridge: comps.append(f"ridge(ic={ic.mean_oos_ic:.3f})")
                    if use_trend: comps.append(f"trend(t={trend['t_stat']:.1f})")
                    print(f"[BC auto] teacher = {' + '.join(comps)}")
            elif verbose:
                print("[BC auto] no gate passed -> BC disabled (flat prior)")
        elif bc_teacher == "ridge":
            teacher = ridge_teacher(fs, horizon=horizon)
        elif bc_teacher == "ts_momentum":
            teacher = ts_momentum_teacher()
        elif bc_teacher == "oracle":
            from mars_lite.features.signal_check import run_signal_check
            from mars_lite.learning.bc_warmstart import dp_oracle_teacher
            ic = run_signal_check(fs, horizon=horizon)
            if ic.mean_oos_ic >= 0.025:
                teacher = dp_oracle_teacher(fs, noisy_ic=oracle_noisy_ic)
                if verbose:
                    kind = f"noisy_ic={oracle_noisy_ic}" if oracle_noisy_ic else "perfect foresight"
                    print(f"[BC oracle] IC gate passed (ic={ic.mean_oos_ic:.3f}), "
                          f"using DP-oracle teacher ({kind})")
            elif verbose:
                print(f"[BC oracle] IC gate failed (ic={ic.mean_oos_ic:.3f}) "
                      "-> oracle teacher disabled (flat prior); "
                      "特権教師を模倣する意味がない（ノイズの丸暗記になる）")
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
    # 捕捉率 = RL収益 / オラクル収益。完全オラクル（IC=1、到達不能な天井）と
    # ノイズ入りオラクル（現実的な目標IC天井）の両方があれば併記する。
    # ヘッドラインはノイズ入りオラクル比（現実的に目指すべき水準）。
    oracle = baselines.get("oracle_dp")
    if oracle is not None:
        o = oracle.to_dict() if hasattr(oracle, "to_dict") else oracle
        denom = o["total_return"]
        if abs(denom) > 1e-9:
            print(f"  capture rate (RL / oracle_dp perfect)  = "
                  f"{agent_res['total_return'] / denom:>+.1%}")
    for name, r in baselines.items():
        if not name.startswith("oracle_ic"):
            continue
        d = r.to_dict() if hasattr(r, "to_dict") else r
        denom = d["total_return"]
        if denom > 1e-9:
            print(f"  capture rate (RL / {name})  = "
                  f"{agent_res['total_return'] / denom:>+.1%}  <- headline")
        else:
            print(f"  {name}: 目標ICでもコスト後は黒字化しない"
                  f"（return={denom:+.2%}）。捕捉率は算出不能"
                  f"（このコスト水準ではより高いICか低コストが必要）")


def build_post_processor(args, horizon: int = 4):
    """
    CLIフラグから後処理器を構築（デフォルトは推奨フル後処理）

    horizon（採用した予測ホライズン、既定4=基準TF換算4本先）に応じて
    EMA平滑・no-tradeバンドをスケールする。horizon=4で従来値
    （ema_alpha=0.5, no_trade_band=0.04）と一致し後方互換。
    horizonが大きい（低頻度アルファ）ほど平滑を強め・no-tradeバンドを
    広げ、コストと信号の周波数を整合させる。--decision-every を
    明示指定した場合はそちらを優先し、このスケーリングは変えない。
    """
    from mars_lite.trading.post_processor import (
        make_default_processor, make_legacy_processor,
    )
    mode = getattr(args, "postproc", "full")
    if mode == "legacy":
        return make_legacy_processor()
    tv = None if getattr(args, "target_vol", 0.5) <= 0 else args.target_vol
    ema_alpha = float(np.clip(0.5 * (4.0 / max(horizon, 1)), 0.05, 1.0))
    no_trade_band = float(0.04 * np.sqrt(max(horizon, 1) / 4.0))
    return make_default_processor(target_vol=tv, ema_alpha=ema_alpha, no_trade_band=no_trade_band)


def build_env_kwargs(args, pp, horizon: int = 4) -> dict:
    """
    学習/評価で共有する環境kwargs（後処理器 + 任意のHTFゲート + 意思決定間隔）

    --decision-every が明示指定されていればそれを使う。未指定かつ
    ホライズンスキャンでhorizonが選ばれている場合は
    decision_every = max(1, horizon // 2) を既定にする
    （例: horizon=8時間 -> 4時間毎に意思決定。1h毎の回転コストで
    低頻度アルファを削るのを防ぐ）。
    """
    ekw = {
        "post_processor": pp,
        "min_trade_delta": getattr(args, "min_trade_delta", 0.04),
        "lambda_turnover": getattr(args, "lambda_turnover", 0.04),
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


def build_feature_set(args, output_dir: Optional[Path] = None) -> FeatureSet:
    if args.source == "synthetic":
        source = SyntheticSource(
            n_days=args.days, alpha=args.alpha,
            alpha_strength=args.alpha_strength, seed=args.seed,
        )
        symbols = source.symbols
    else:
        symbols = args.symbols or DEFAULT_SYMBOLS
        if args.source == "csv":
            kwargs = {"data_dir": args.data}
        elif args.source == "hyperliquid":
            kwargs = {"days": args.days}
        elif args.source == "postgres":
            kwargs = {"dsn": args.pg_dsn, "source": args.pg_source,
                     "derivatives_source": args.pg_derivatives_source,
                     "orderflow_source": args.pg_derivatives_source}
        else:
            kwargs = {}
        source = create_source(args.source, symbols, **kwargs)
        from mars_lite.data.quality import run_quality_gate
        qrep = run_quality_gate(source, symbols, base_timeframe="1h")
        print(qrep.summary())
        if output_dir is not None:
            with open(output_dir / "data_quality_report.json", "w", encoding="utf-8") as f:
                json.dump(qrep.to_dict(), f, indent=2, ensure_ascii=False)
        symbols = qrep.passing_symbols
        if len(symbols) < 2:
            raise ValueError("品質ゲート通過銘柄が2未満です。データを確認してください。")
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
        ic = run_signal_check(fs, horizon=args.horizon)
        print(ic.summary())

        # train/test 時系列分割（70/30、purge = max(24,horizon)本）
        split = int(fs.n_bars * 0.7)
        purge = max(24, args.horizon)
        train_fs = fs.slice(0, split)
        test_fs = fs.slice(split + purge, fs.n_bars)

        print(f"\nTraining PPO: {args.timesteps:,} steps "
              f"(train {train_fs.n_bars} bars, test {test_fs.n_bars} bars)...")
        pp = build_post_processor(args, horizon=args.horizon)
        ekw = build_env_kwargs(args, pp, horizon=args.horizon)
        # BCウォームスタートはICゲート合格時のみ（信号なきデータで教師を
        # 模倣するとノイズを刷り込み、陰性対照の安全性を壊すため）
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          gamma=args.gamma, verbose=args.verbose,
                          bc_warmstart=True, horizon=args.horizon, **ekw)

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
    from mars_lite.features.signal_check import run_leak_self_test

    # --- 品質ゲートおよび特徴量生成（実データソースのみ自動剪定） ---
    try:
        fs = build_feature_set(args, output_dir=output_dir)
    except ValueError as e:
        print(f"\n[STOP] {e}")
        return
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols x {fs.n_features} features")

    # --- リーク検出器の自己検査（評価コードの健全性確認） ---
    leak = run_leak_self_test(fs)
    print(f"[Leak self-test] shuffle_ic={leak['shuffle_ic']:.4f} "
          f"future_shift_ic={leak['future_shift_ic']:.4f} healthy={leak['healthy']}")
    if not leak["healthy"]:
        print("[WARN] リーク検出器の自己検査に失敗。評価結果を信用しないこと。")

    horizon = args.horizon
    scan = None
    if args.scan_horizons:
        from mars_lite.features.horizon_scan import run_horizon_scan
        # test分離: スキャンは学習スライス相当（先頭80%）のみで行う
        scan_split = int(fs.n_bars * 0.8)
        scan = run_horizon_scan(fs.slice(0, scan_split), horizons=tuple(args.horizons))
        print(scan.summary())
        horizon = scan.best_horizon
        print(f"[Horizon scan] selected horizon={horizon}")

    ic = run_signal_check(fs, horizon=horizon)
    print(ic.summary())
    if not ic.passed and not args.skip_gate:
        print("\n[STOP] ゲート1不合格: 特徴量に予測力がありません。"
              "RL学習をスキップします（--skip-gate で強制続行可）。")
        return

    split = int(fs.n_bars * 0.8)
    purge = max(24, horizon)
    train_fs = fs.slice(0, split)
    test_fs = fs.slice(split + purge, fs.n_bars)

    # IC安定性マスク（オプトイン。時間軸の冗長性はTFゲート構造が既定で処理する）
    feature_mask = None
    if args.feature_mask:
        from mars_lite.features.signal_check import compute_feature_mask
        mask_rep = compute_feature_mask(train_fs, horizon=horizon)
        feature_mask = mask_rep["mask"]
        print(f"[Feature mask] kept {len(mask_rep['kept'])}/{fs.n_features} features "
              f"(dropped: {', '.join(mask_rep['dropped'][:8])}"
              f"{'...' if len(mask_rep['dropped']) > 8 else ''})")
        train_fs = train_fs.apply_mask(feature_mask)
        test_fs = test_fs.apply_mask(feature_mask)

    pp = build_post_processor(args, horizon=horizon)
    ekw = build_env_kwargs(args, pp, horizon=horizon)

    if args.ensemble > 1:
        from mars_lite.learning.policy_ensemble import train_ensemble
        print(f"Training {args.ensemble}-seed ensemble x {args.timesteps:,} steps...")

        def _train(train_fs_, seed):
            return train_ppo(fs=train_fs_, timesteps=args.timesteps, seed=seed,
                             gamma=args.gamma, bc_warmstart=True, horizon=horizon,
                             bc_teacher=args.bc_teacher, oracle_noisy_ic=args.oracle_noisy_ic,
                             **ekw)
        agent = train_ensemble(_train, train_fs, seeds=list(range(args.ensemble)),
                               verbose=1)
        agent.save(str(output_dir / "portfolio_ensemble"))
    else:
        print(f"Training PPO: {args.timesteps:,} steps...")
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          gamma=args.gamma, verbose=args.verbose,
                          bc_warmstart=True, horizon=horizon,
                          bc_teacher=args.bc_teacher, oracle_noisy_ic=args.oracle_noisy_ic,
                          **ekw)

    agent_res = evaluate_agent_on_slice(agent, test_fs, **ekw)
    noisy_ic = args.noisy_oracle_ic if args.noisy_oracle_ic > 0 else None
    baselines = run_all_baselines(test_fs, noisy_oracle_ic=noisy_ic)
    report_comparison(agent_res, baselines, "OOS comparison")
    plot_comparison(agent_res, baselines, output_dir / "train_equity.png")

    # Phase E: ゲート2判定（RLが全ベースラインを上回ったか）を自動記録
    # 特にtrend_following（固定ルール）との比較が最重要な基準
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
    # trend_following は特に重要なので別途記録
    tf_baseline = gate2_details.get("trend_following", {})
    gate2 = {
        "passed": bool(gate2_passed),
        "rl_beat_trend_following": bool(tf_baseline.get("rl_beat", False)) if "rl_beat" in tf_baseline else None,
        "details": gate2_details,
    }
    print(f"\n[Gate 2] {'PASS' if gate2_passed else 'FAIL'} "
          f"RL vs all baselines. trend_following: "
          f"{'BEAT' if tf_baseline.get('rl_beat') else 'LOST'}")

    if args.ensemble <= 1:
        agent.save(str(output_dir / "portfolio_model"))
    with open(output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "signal_gate": ic.to_dict(),
            "horizon_scan": scan.to_dict() if scan is not None else None,
            "feature_mask": ([bool(x) for x in feature_mask]
                             if feature_mask is not None else None),
            "agent": {k: v for k, v in agent_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
            "gate2": gate2,  # Phase E: ゲート2判定結果
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
    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)
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

    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)

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
    """P3: ウォークフォワード検証（3fold×3seed×コスト感度、Phase E標準）”"""
    try:
        fs = build_feature_set(args, output_dir=output_dir)
    except ValueError as e:
        print(f"\n[STOP] {e}")
        return
    print(f"FeatureSet: {fs.n_bars} bars x {fs.n_symbols} symbols")

    pp = build_post_processor(args, horizon=args.horizon)
    ekw = build_env_kwargs(args, pp, horizon=args.horizon)

    # Phase E: --ensemble 3 を推奨デフォルトに。wfでは内側でアンサンブル学習をサポートする。
    n_ensemble = max(args.ensemble, 1)
    if n_ensemble > 1:
        from mars_lite.learning.policy_ensemble import train_ensemble
        def train_fn(train_fs: FeatureSet, seed: int):
            def _inner(train_fs_: FeatureSet, _seed: int):
                return train_ppo(fs=train_fs_, timesteps=args.timesteps, seed=_seed,
                                 gamma=args.gamma, bc_warmstart=True, **ekw)
            return train_ensemble(_inner, train_fs,
                                  seeds=list(range(seed, seed + n_ensemble)), verbose=0)
    else:
        def train_fn(train_fs: FeatureSet, seed: int):
            return train_ppo(fs=train_fs, timesteps=args.timesteps, seed=seed,
                             gamma=args.gamma, bc_warmstart=True, **ekw)

    for cost_mult in [1.0, 2.0]:
        print(f"\n--- Walk-forward (cost x{cost_mult}, ensemble={n_ensemble}) ---")
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
    parser.add_argument("--source", choices=["synthetic", "csv", "postgres", "hyperliquid"],
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
                        help="シードアンサンブルの個体数（1で単一モデル）。"
                             "実データでは3推奨（シード運のばらつき低減+不一致度スケーリング））")
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
    parser.add_argument("--horizon", type=int, default=4,
                        help="予測ホライズン（バー数）。ICゲート/BC教師/特徴マスクに使う")
    parser.add_argument("--scan-horizons", action="store_true",
                        help="--phase train で学習前にホライズンスキャンを行い、"
                             "OOS ICが最大のホライズンを自動選択する（--horizonを上書き）")
    parser.add_argument("--horizons", type=int, nargs="+",
                        default=[1, 2, 4, 8, 24, 48, 72],
                        help="--scan-horizons で走査するホライズン候補")
    parser.add_argument("--decision-every", type=int, default=1,
                        help="環境の意思決定間隔（バー数）。1バーでシグナルが立たない"
                             "低頻度アルファをホライズンスキャンで見つけた場合に使う")
    parser.add_argument("--min-trade-delta", type=float, default=0.04,
                        help="微小リバランス禁止バンド（デフォルト0.04=4%未満の変更はスキップ）")
    parser.add_argument("--lambda-turnover", type=float, default=0.04,
                        help="ターンオーバー罰則係数（デフォルト0.04=回転コストの抑制）")
    parser.add_argument("--noisy-oracle-ic", type=float, default=0.05,
                        help="現実的な天井として併記するノイズ入りオラクルの目標IC。"
                             "0以下で無効")
    parser.add_argument("--bc-teacher", choices=["auto", "ridge", "ts_momentum", "momentum", "oracle"],
                        default="auto",
                        help="BC事前学習の教師。oracle=DPオラクル（特権教師）を蒸留。"
                             "ICゲート合格時のみ有効化される")
    parser.add_argument("--oracle-noisy-ic", type=float, default=None,
                        help="--bc-teacher oracle で使う劣化オラクルの目標IC。"
                             "省略時は完全予知（学習不能なパターンを丸暗記するリスクあり）")
    parser.add_argument("--pg-dsn", type=str, default=None,
                        help="--source postgres 用の接続文字列。省略時は環境変数 "
                             "PLATFORM_DB_URL、それも無ければ docker-compose.yml の既定値")
    parser.add_argument("--pg-source", type=str, default="binance",
                        help="--source postgres で rl_klines/rl_funding_rate を絞り込む"
                             "source列の値（例: binance, hyperliquid）")
    parser.add_argument("--pg-derivatives-source", type=str, default=None,
                        help="--source postgres でrl_derivatives/rl_orderflow_1mを絞り込む"
                             "source列の値。省略時は--pg-sourceと同じ"
                             "（例: klines/fundingはhyperliquid、OI等はbinance代理）")
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
