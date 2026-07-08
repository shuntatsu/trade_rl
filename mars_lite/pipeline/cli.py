import argparse
from pathlib import Path

from mars_lite.pipeline.dataset_builder import DEFAULT_SYMBOLS
from mars_lite.pipeline.evaluator import (
    phase_p0,
    phase_pbt,
    phase_regime,
    phase_train,
    phase_wf,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ポートフォリオRL学習")
    parser.add_argument(
        "--phase", choices=["p0", "train", "wf", "pbt", "regime"], default="p0"
    )
    parser.add_argument(
        "--source",
        choices=["synthetic", "csv", "postgres", "hyperliquid", "bitget", "okx"],
        default="synthetic",
    )
    parser.add_argument("--data", type=str, default="./data")
    parser.add_argument("--symbols", type=str, nargs="+", default=None)
    parser.add_argument("--days", type=int, default=240, help="syntheticの生成日数")
    parser.add_argument(
        "--alpha",
        default="cross",
        choices=["none", "cross", "meanrev", "multi", "bull"],
    )
    parser.add_argument("--alpha-strength", type=float, default=0.002)
    parser.add_argument("--timesteps", type=int, default=300_000)
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.5,
        help="割引率。行動効果が即時のため低い値が有効",
    )
    parser.add_argument(
        "--ent-coef",
        type=float,
        default=0.002,
        help="PPOエントロピー係数。--phase pbt の探索結果を流し込む用途にも使う",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=3e-4,
        help="PPO学習率。--phase pbt の探索結果を流し込む用途にも使う",
    )
    parser.add_argument(
        "--reward-scale",
        type=float,
        default=100.0,
        help="報酬スケール。--phase pbt の探索結果を流し込む用途にも使う",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--n-seeds", type=int, default=3)
    parser.add_argument("--output", type=str, default="./output/portfolio")
    parser.add_argument("--verbose", type=int, default=0)
    parser.add_argument("--skip-gate", action="store_true")
    parser.add_argument(
        "--warmup-days",
        type=float,
        default=0,
        help="先頭Ndaysをウォームアップとして切り捨てる（最長ローリング窓"
        "=1dTFのvol_ratio長期側100日分が埋まるまで特徴が不完全なため）。"
        "実効学習期間をNdays確保したいなら、取得を(N+warmup_days)日分"
        "にして本フラグで切り捨てる運用にする。既定0=無効",
    )
    parser.add_argument(
        "--postproc",
        choices=["full", "legacy"],
        default="full",
        help="後処理: full=推奨(平滑/バンド/ボラ目標/DDデリスク), legacy=射影のみ",
    )
    parser.add_argument(
        "--target-vol", type=float, default=0.5, help="年率ボラ目標。0以下で無効"
    )
    parser.add_argument(
        "--ensemble",
        type=int,
        default=1,
        help="シードアンサンブルの個体数（1で単一モデル）。"
        "実データでは3推奨（シード運のばらつき低減+不一致度スケーリング））",
    )
    parser.add_argument(
        "--feature-mask",
        action="store_true",
        help="IC安定性による特徴マスクを有効化（実験では中立〜微減。"
        "実データでジャンク特徴が多い場合のオプション）",
    )
    parser.add_argument(
        "--pbt-pop", type=int, default=6, help="PBT個体数（--phase pbt）"
    )
    parser.add_argument(
        "--pbt-gen", type=int, default=4, help="PBT世代数（--phase pbt）"
    )
    parser.add_argument(
        "--pbt-steps",
        type=int,
        default=40_000,
        help="PBT各個体の学習ステップ数（--phase pbt）",
    )
    parser.add_argument(
        "--regime-bars",
        type=int,
        default=120,
        help="レジーム専門家のエピソード長（--phase regime、5日=120本）",
    )
    parser.add_argument(
        "--htf-gate",
        action="store_true",
        help="階層MTF: 上位足(4h)トレンドで方向を制約し1hはサイジング",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=4,
        help="予測ホライズン（バー数）。ICゲート/BC教師/特徴マスクに使う",
    )
    parser.add_argument(
        "--scan-horizons",
        action="store_true",
        help="--phase train で学習前にホライズンスキャンを行い、"
        "OOS ICが最大のホライズンを自動選択する（--horizonを上書き）",
    )
    parser.add_argument(
        "--horizons",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8, 24, 48, 72],
        help="--scan-horizons で走査するホライズン候補",
    )
    parser.add_argument(
        "--decision-every",
        type=int,
        default=1,
        help="環境の意思決定間隔（バー数）。1バーでシグナルが立たない"
        "低頻度アルファをホライズンスキャンで見つけた場合に使う",
    )
    parser.add_argument(
        "--min-trade-delta",
        type=float,
        default=0.04,
        help="微小リバランス禁止バンド（デフォルト0.04=4%%未満の変更はスキップ）",
    )
    parser.add_argument(
        "--lambda-turnover",
        type=float,
        default=0.04,
        help="ターンオーバー罰則係数（デフォルト0.04=回転コストの抑制）",
    )
    parser.add_argument(
        "--noisy-oracle-ic",
        type=float,
        default=0.05,
        help="現実的な天井として併記するノイズ入りオラクルの目標IC。0以下で無効",
    )
    parser.add_argument(
        "--bc-teacher",
        choices=["auto", "ridge", "ts_momentum", "momentum", "oracle"],
        default="auto",
        help="BC事前学習の教師。oracle=DPオラクル（特権教師）を蒸留。"
        "ICゲート合格時のみ有効化される",
    )
    parser.add_argument(
        "--oracle-noisy-ic",
        type=float,
        default=None,
        help="--bc-teacher oracle で使う劣化オラクルの目標IC。"
        "省略時は完全予知（学習不能なパターンを丸暗記するリスクあり）",
    )
    parser.add_argument(
        "--pg-dsn",
        type=str,
        default=None,
        help="--source postgres 用の接続文字列。省略時は環境変数 "
        "PLATFORM_DB_URL、それも無ければ docker-compose.yml の既定値",
    )
    parser.add_argument(
        "--pg-source",
        type=str,
        default="binance",
        help="--source postgres で rl_klines/rl_funding_rate を絞り込む"
        "source列の値（例: binance, hyperliquid）",
    )
    parser.add_argument(
        "--pg-derivatives-source",
        type=str,
        default=None,
        help="--source postgres でrl_derivatives/rl_orderflow_1mを絞り込む"
        "source列の値。省略時は--pg-sourceと同じ"
        "（例: klines/fundingはhyperliquid、OI等はbinance代理）",
    )

    # 新規追加引数
    parser.add_argument(
        "--calibrate-regime",
        action="store_true",
        default=True,
        help="Regime FSM の自動較正を実行",
    )
    parser.add_argument(
        "--no-calibrate-regime",
        action="store_false",
        dest="calibrate_regime",
        help="Regime FSM の自動較正を実行しない",
    )
    parser.add_argument(
        "--regime-trials", type=int, default=100, help="Regime FSM の自動較正の試行回数"
    )
    return parser


def main():
    parser = build_parser()
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
