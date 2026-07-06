"""
学習パイプラインを1コマンドで通しで実行するオーケストレータ。

p0(健全性試験) -> pbt(ハイパラ探索) -> wf(walk-forward評価/ゲート)
    -> train(最終モデル学習/ゲート2) -> bootstrap(統計的有意性の参考値)
    -> model_registry登録

各ステップは train_portfolio.py の各 --phase を内部で順番に呼ぶだけで、
個別に `--phase p0` 等を手で叩くのと同じ関数を実行する。
既定では各ゲート不合格時にそこで停止する（--force で強制続行）。

pbt が探索した best_hp (gamma/ent_coef/learning_rate/lambda_turnover/
reward_scale) は、以降の wf・train にそのまま引き継がれる。

【ホールドアウト分離】
特徴量セットは1回だけ構築し、末尾 `--holdout-frac`（既定15%）を
pbt・wfが一切参照しない完全ホールドアウト区間として切り出す。
pbt のハイパラ探索・wf の全foldはこのホールドアウトより前の「dev区間」だけで
完結し、train の最終ゲート2判定だけがホールドアウト区間を1回だけ参照する。
これをしないと、pbtが「後ろ30%で良いハイパラ」を探し、wf/trainの
アウトオブサンプル評価が同じ後ろ20〜30%を再利用する
（=ハイパラ探索に使ったデータを"証拠"として提示する）データスヌーピングになる。

使い方:
    uv run python scripts/run_pipeline.py --source postgres \\
        --timesteps 500000 --ensemble 3 --folds 3 --n-seeds 3
"""

import json
import sys
from pathlib import Path

from mars_lite.pipeline.cli import build_parser
from mars_lite.pipeline.dataset_builder import build_feature_set
from mars_lite.pipeline.evaluator import phase_p0, phase_pbt, phase_train, phase_wf


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _print_step(n: int, total: int, title: str) -> None:
    print(f"\n{'=' * 70}\nSTEP {n}/{total}: {title}\n{'=' * 70}")


def main() -> int:
    parser = build_parser()
    parser.add_argument("--skip-p0", action="store_true", help="p0健全性試験をスキップ")
    parser.add_argument(
        "--skip-pbt", action="store_true", help="pbtハイパラ探索をスキップ"
    )
    parser.add_argument(
        "--skip-wf",
        action="store_true",
        help="walk-forward評価をスキップ（実データでは非推奨）",
    )
    parser.add_argument(
        "--wf-cost-gate",
        type=float,
        default=0.0,
        help="walk-forward(コスト2倍)の中央値リターンがこの値以下なら停止",
    )
    parser.add_argument(
        "--require-significant",
        action="store_true",
        help="ブートストラップ検定(対trend_following)のp値<0.05をゲートにする",
    )
    parser.add_argument(
        "--no-register",
        action="store_true",
        help="学習後にモデルレジストリへ登録しない",
    )
    parser.add_argument(
        "--registry-dir",
        type=str,
        default=None,
        help="モデルレジストリのディレクトリ（既定: <output>/model_registry）",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="ゲート不合格でも後続ステップを続行する（自己責任）",
    )
    parser.add_argument(
        "--holdout-frac",
        type=float,
        default=0.15,
        help="pbt/wfが一切触れない最終ホールドアウト区間の割合（末尾から）",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    total_steps = 5

    # --- STEP 1: p0 健全性試験 ---
    _print_step(1, total_steps, "p0 健全性試験（学習システム自体が壊れていないか）")
    if args.skip_p0:
        print("[skip]")
    else:
        phase_p0(args, output_dir)
        gate = _load(output_dir / "p0_report.json")["gate"]
        print(f"[p0] P0_PASSED = {gate['P0_PASSED']}")
        if not gate["P0_PASSED"] and not args.force:
            print(
                "[STOP] p0が不合格。学習システムの前提から疑うこと。--force で強制続行可能。"
            )
            return 1

    # --- 特徴量セットを1回だけ構築し、ホールドアウトを切り出す ---
    if args.source == "synthetic":
        fs_full = None  # p0以外は各phaseが自前でsynthetic生成（既定挙動を維持）
    else:
        try:
            fs_full = build_feature_set(args, output_dir=output_dir)
        except ValueError as e:
            print(f"\n[STOP] {e}")
            return 1

    fs_dev = fs_holdout = None
    if fs_full is not None:
        purge = max(24, args.horizon)
        holdout_start = int(fs_full.n_bars * (1.0 - args.holdout_frac))
        min_bars = 50
        if (
            fs_full.n_bars - holdout_start - purge
        ) < min_bars or holdout_start < min_bars:
            print(
                f"[WARN] データが短すぎてホールドアウト分離できません "
                f"(n_bars={fs_full.n_bars})。従来通り各phaseが自前で分割します。"
            )
        else:
            fs_dev = fs_full.slice(0, holdout_start)
            fs_holdout = fs_full.slice(holdout_start + purge, fs_full.n_bars)
            print(
                f"[holdout] dev={fs_dev.n_bars}本 (pbt/wfが使用) / "
                f"holdout={fs_holdout.n_bars}本 (最終ゲート2でのみ使用、pbt/wf未接触)"
            )

    # --- STEP 2: pbt ハイパーパラメータ探索 ---
    _print_step(2, total_steps, "pbt ハイパーパラメータ探索（devのみ）")
    if args.skip_pbt:
        print("[skip]（--gamma 等をCLI指定値のまま使用）")
    else:
        phase_pbt(args, output_dir, fs=fs_dev)
        best_hp = _load(output_dir / "pbt_result.json")["best_hp"]
        args.gamma = best_hp["gamma"]
        args.ent_coef = best_hp["ent_coef"]
        args.learning_rate = best_hp["learning_rate"]
        args.lambda_turnover = best_hp["lambda_turnover"]
        args.reward_scale = best_hp["reward_scale"]
        print(f"[pbt] 採用ハイパーパラメータ -> {best_hp}")

    # --- STEP 3: walk-forward評価（ゲート、devのみ） ---
    _print_step(
        3,
        total_steps,
        "walk-forward評価（devのみ、実データで安定して機能するかの本丸）",
    )
    if args.skip_wf:
        print("[skip]（非推奨: 実データでは必ず通すこと）")
    else:
        phase_wf(args, output_dir, fs=fs_dev)
        wf = _load(output_dir / "walk_forward_cost2x.json")
        median_return = wf["summary"]["agent_total_return"]["median"]
        print(f"[wf] コスト2倍時の中央値リターン = {median_return:+.2%}")
        if median_return <= args.wf_cost_gate and not args.force:
            print(
                f"[STOP] walk-forwardがゲート({args.wf_cost_gate:+.2%})を下回りました。"
                "--force で強制続行可能。ここで止まるのは「効いていない」という結論そのもの。"
            )
            return 1

    # --- STEP 4: 最終モデル学習（devで学習、holdoutでゲート2） ---
    _print_step(
        4, total_steps, "最終モデル学習（devで学習 + 未接触holdoutでゲート判定）"
    )
    train_result = phase_train(args, output_dir, dev_fs=fs_dev, holdout_fs=fs_holdout)
    if train_result is None:
        print("[STOP] train フェーズが早期終了しました（ゲート1不合格など）。")
        return 1
    gate2_passed = train_result["gate2"]["passed"]
    print(
        f"[train] ゲート2（全ベースラインに勝っているか） = {'PASS' if gate2_passed else 'FAIL'}"
    )
    if not gate2_passed and not args.force:
        print("[STOP] ゲート2不合格。--force で強制登録可能。")
        return 1

    if args.require_significant:
        from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference

        agent_curve = train_result["agent_res"].get("equity_curve")
        tf_baseline = train_result["baselines"].get("trend_following")
        base_curve = tf_baseline.equity_curve if tf_baseline is not None else None
        if agent_curve and base_curve is not None and len(base_curve):
            import numpy as np

            a = np.diff(agent_curve) / np.asarray(agent_curve[:-1])
            b = np.diff(base_curve) / np.asarray(base_curve[:-1])
            n = min(len(a), len(b))
            stat = bootstrap_sharpe_difference(a[:n], b[:n], seed=args.seed)
            print(
                f"[bootstrap] observed_diff={stat['observed_diff']:+.3f} "
                f"CI=[{stat['lower_ci']:+.3f}, {stat['upper_ci']:+.3f}] "
                f"p={stat['p_value']:.4f}"
            )
            if stat["p_value"] >= 0.05 and not args.force:
                print(
                    "[STOP] trend_followingに対する優位性が統計的に有意ではありません"
                    "（p >= 0.05）。--force で強制登録可能。"
                )
                return 1
        else:
            print("[bootstrap] equity_curveが見つからずスキップ")

    # --- STEP 5: モデルレジストリ登録 ---
    _print_step(5, total_steps, "モデルレジストリ登録")
    if args.no_register:
        print("[skip]")
    else:
        from mars_lite.server.model_registry import ModelRegistry

        model_name = "portfolio_ensemble" if args.ensemble > 1 else "portfolio_model"
        model_path = output_dir / f"{model_name}.zip"
        if not model_path.exists():
            print(f"[WARN] {model_path} が見つからず登録をスキップします")
        else:
            registry_dir = args.registry_dir or str(output_dir / "model_registry")
            registry = ModelRegistry(registry_dir)
            agent_metrics = train_result["agent_res"]
            entry = registry.register(
                model_path,
                metrics={
                    "sharpe": agent_metrics.get("sharpe", 0.0),
                    "total_return": agent_metrics.get("total_return", 0.0),
                    "max_drawdown": agent_metrics.get("max_drawdown", 0.0),
                },
            )
            print(f"[registry] version={entry.version} を登録し、アクティブ化しました")
            print(
                f"[registry] 一覧確認: uv run python -m mars_lite.server.model_registry "
                f"--registry-dir {registry_dir} list"
            )

    print(f"\n{'=' * 70}\nパイプライン完了\n{'=' * 70}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
