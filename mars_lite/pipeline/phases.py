"""
学習フェーズ実装（p0/train/wf/pbt/regime）

scripts/train_portfolio.py から移動。CLIは引数パースとフェーズ選択のみを行い、
実処理はここに集約する（サーバー等の別エントリポイントからも同じ実装を使うため）。
"""

import json
from pathlib import Path
from typing import Optional

import numpy as np

from mars_lite.data.sources import create_source, SyntheticSource
from mars_lite.features.feature_pipeline import FeaturePipeline, FeatureSet
from mars_lite.features.signal_check import run_signal_check
from mars_lite.env.portfolio_env import PortfolioTradingEnv
from mars_lite.learning.baselines import run_all_baselines
from mars_lite.learning.trainer import train_ppo
from mars_lite.eval.walk_forward import (
    evaluate_agent_on_slice, run_walk_forward, plot_comparison,
)

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT",
    "SUIUSDT", "DOGEUSDT",
    "ADAUSDT", "AVAXUSDT", "LINKUSDT",
    "LTCUSDT", "BCHUSDT", "APTUSDT", "ARBUSDT", "OPUSDT",
]


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
    return make_default_processor(
        target_vol=tv, ema_alpha=ema_alpha, no_trade_band=no_trade_band,
        beta_neutral=getattr(args, "beta_neutral", False),
    )


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
    fs = FeaturePipeline(symbols).build(source)
    # 入力の分布正規化（汎用性向上）: 各特徴チャネルをローリング・ガウスランクで
    # N(0,1)に写像。資産・レジーム間のスケール差への過適合を抑える。既定はoff
    # （原則1: 証拠なき機能は既定にしない）。
    # 注意: warmup切り捨てより前に適用する。rank_gaussの正規化窓(既定250本)
    # 自体もwarmupなので、後段のwarmup切り捨てに自然に吸収される。
    if getattr(args, "feature_norm", "none") == "rank_gauss":
        fs = fs.gaussian_rank_normalized()
        print("[Feature norm] rank_gauss適用（各特徴をローリングN(0,1)に正規化）")

    # ウォームアップ切り捨て: 最長のローリング窓（1dTFのvol_ratio長期側=100日
    # ≈2400本@1h）が埋まるまで特徴が不完全（min_periods未満はゼロ埋め）。
    # 「実効学習期間をNdays確保したい」場合は取得を(N+warmup_days)日分にして
    # 先頭warmup_days日を切り捨てる運用にする（例: 365日学習→465日取得）。
    warmup_days = getattr(args, "warmup_days", 0)
    if warmup_days > 0:
        bar_minutes = 60  # base_timeframe既定"1h"
        warmup_bars = int(warmup_days * 24 * 60 / bar_minutes)
        if warmup_bars >= fs.n_bars:
            raise ValueError(
                f"--warmup-days {warmup_days} はデータ長({fs.n_bars}本)以上です。"
            )
        fs = fs.slice(warmup_bars, fs.n_bars)
        print(f"[Warmup] 先頭{warmup_days}日（{warmup_bars}本）を切り捨て。"
              f"残り{fs.n_bars}本を学習/検証に使用。")
    return fs


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
        if getattr(args, "feature_norm", "none") == "rank_gauss":
            fs = fs.gaussian_rank_normalized()

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
                          bc_warmstart=True, horizon=args.horizon,
                          net_size=args.net_size, dropout=args.dropout, **ekw)

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

    # --- ロックボックス（最終封印テスト）: 末尾を全工程から隔離 ---
    # 何度もwalk-forward/train実行を繰り返すほど、検証データへの
    # 暗黙の過学習（p-hacking）が蓄積する。ここで切り出した区間は
    # ゲート・特徴マスク・ホライズン選択・fold分割のいずれにも使わず、
    # 最終モデルの評価に一度だけ使う。
    lockbox_fs = None
    if args.lockbox_frac > 0:
        cut = int(fs.n_bars * (1.0 - args.lockbox_frac))
        lockbox_fs = fs.slice(cut, fs.n_bars)
        fs = fs.slice(0, cut)
        print(f"[Lockbox] 末尾{args.lockbox_frac:.0%}（{lockbox_fs.n_bars}本）を"
              "最終封印テストとして隔離。以降の全工程はこの区間を参照しない。")

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
                             net_size=args.net_size, dropout=args.dropout, **ekw)
        agent = train_ensemble(_train, train_fs, seeds=list(range(args.ensemble)),
                               verbose=1)
        agent.save(str(output_dir / "portfolio_ensemble"))
    else:
        print(f"Training PPO: {args.timesteps:,} steps...")
        agent = train_ppo(fs=train_fs, timesteps=args.timesteps, seed=args.seed,
                          gamma=args.gamma, verbose=args.verbose,
                          bc_warmstart=True, horizon=horizon,
                          bc_teacher=args.bc_teacher, oracle_noisy_ic=args.oracle_noisy_ic,
                          net_size=args.net_size, dropout=args.dropout, **ekw)

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
        from mars_lite.serving.model_store import save_bundle, ModelMetadata
        save_bundle(output_dir, "portfolio_model", agent, ModelMetadata(
            symbols=fs.symbols,
            post_processor=pp.cfg.to_dict(),
            feature_mask=([bool(x) for x in feature_mask]
                          if feature_mask is not None else None),
            metrics={
                "signal_gate": ic.to_dict(),
                "gate2": gate2,
            },
        ))

    lockbox_report = None
    if lockbox_fs is not None:
        marker = output_dir / "lockbox_used.marker"
        if marker.exists():
            print(f"\n[Lockbox] 警告: このロックボックスは既に "
                  f"{marker.read_text().strip()} に一度使用済みです。"
                  "同じ区間を繰り返し見て判断を調整するのは過学習の抜け道になります。")
        lockbox_res = evaluate_agent_on_slice(agent, lockbox_fs, **ekw)
        lockbox_baselines = run_all_baselines(lockbox_fs, noisy_oracle_ic=noisy_ic)
        report_comparison(lockbox_res, lockbox_baselines,
                          "LOCKBOX（最終封印・一度きりの検定）")

        # ロックボックス最終判定: RLが未使用区間で全ベースライン(特にtrend_following)
        # を上回れて初めて「本物」。ここを閉じないと診断値を出すだけで終わる。
        lb_ret = float(lockbox_res["total_return"])
        lb_beat = {}
        lb_passed = True
        for bname, bres in lockbox_baselines.items():
            if bname.startswith("oracle"):
                continue  # オラクルは到達不能な天井なので合格条件から除外
            bd = bres.to_dict() if hasattr(bres, "to_dict") else bres
            beat = bool(lb_ret > float(bd.get("total_return", 0.0)))
            lb_beat[bname] = beat
            if not beat:
                lb_passed = False
        print(f"[Lockbox GATE] {'PASS' if lb_passed else 'FAIL'} "
              f"（RLが未使用区間で全ベースライン超え）. "
              f"trend_following: {'BEAT' if lb_beat.get('trend_following') else 'LOST'}")

        lockbox_report = {
            "n_bars": lockbox_fs.n_bars,
            "passed": bool(lb_passed),
            "beat_by_baseline": lb_beat,
            "rl_beat_trend_following": bool(lb_beat.get("trend_following", False)),
            "agent": {k: v for k, v in lockbox_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in lockbox_baselines.items()},
        }
        import datetime
        marker.write_text(datetime.datetime.now(datetime.timezone.utc).isoformat())

    with open(output_dir / "train_report.json", "w", encoding="utf-8") as f:
        json.dump({
            "signal_gate": ic.to_dict(),
            "horizon_scan": scan.to_dict() if scan is not None else None,
            "feature_mask": ([bool(x) for x in feature_mask]
                             if feature_mask is not None else None),
            "agent": {k: v for k, v in agent_res.items() if k != "equity_curve"},
            "baselines": {k: v.to_dict() for k, v in baselines.items()},
            "gate2": gate2,  # Phase E: ゲート2判定結果
            "lockbox": lockbox_report,  # 最終封印テスト（--lockbox-frac指定時のみ）
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
            bc_warmstart=True,
            net_size=args.net_size, dropout=args.dropout, **ekw,
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
                           gamma=args.gamma, bc_warmstart=True,
                           net_size=args.net_size, dropout=args.dropout, **ekw)

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
            net_size=args.net_size, dropout=args.dropout,
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
    """P3: ウォークフォワード検証（3fold×3seed×コスト感度、Phase E標準）"""
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
                                 gamma=args.gamma, bc_warmstart=True,
                                 net_size=args.net_size, dropout=args.dropout, **ekw)
            return train_ensemble(_inner, train_fs,
                                  seeds=list(range(seed, seed + n_ensemble)), verbose=0)
    else:
        def train_fn(train_fs: FeatureSet, seed: int):
            return train_ppo(fs=train_fs, timesteps=args.timesteps, seed=seed,
                             gamma=args.gamma, bc_warmstart=True,
                             net_size=args.net_size, dropout=args.dropout, **ekw)

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
        if report.dsr:
            d = report.dsr
            print(f"[Deflated Sharpe] dsr={d['dsr']:.1%} (>=95%推奨) "
                  f"sr_hat={d['sr_hat_annualized']:+.2f} "
                  f"sr0(試行数補正後の基準)={d['sr0_annualized']:+.2f} "
                  f"n_trials={d['n_trials']}")
        print(f"Report -> {path}")
