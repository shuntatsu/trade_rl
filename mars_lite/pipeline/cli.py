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
        "--p0-days",
        type=int,
        default=240,
        help="P0合成データの生成日数。候補のhorizon/decision_everyは変更しない",
    )
    parser.add_argument(
        "--base-timeframe",
        choices=["15m", "1h", "4h", "1d"],
        default="1h",
        help="意思決定の基準時間軸。既定1h。gate1_diagnostic.py で1h不合格でも"
        "4h等では合格することがある（低頻度なほどコスト後ブレークイーブンICが"
        "下がるため）。品質ゲート・ウォームアップ日数換算もこれに追従する",
    )
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
        "--target",
        choices=["raw", "cs_demean", "vol_norm"],
        default="raw",
        help="ICゲート判定とRidge教師の予測対象。raw=絶対リターン(既定)、"
        "cs_demean=市場中立の相対アルファ(銘柄間平均を除去)、vol_norm=ボラ正規化。"
        "絶対リターンに信号が無く相対アルファのみ有意な市場では cs_demean を使う"
        "（gate1_diagnostic.py の診断結果に合わせる）",
    )
    parser.add_argument(
        "--beta-neutral",
        action="store_true",
        help="後処理で市場方向(等ウェイト平均)成分を除去しドル中立化する。"
        "全銘柄がBTCベータで共線なユニバースで相対アルファのみ残す。"
        "--target cs_demean と組で使うと出力も市場中立になる（方向性ベータを"
        "捨てるため純粋な上昇相場では不利になりうる opt-in）",
    )
    parser.add_argument(
        "--fee-profile",
        choices=["taker", "maker"],
        default="taker",
        help="執行コストプロファイル。taker=成行想定(既定、片道7bps)、"
        "maker=指値想定(片道2bps、スプレッドは払わない)。RL/全ベースライン/"
        "オラクルに同一プロファイルを適用するので比較の公平性は保たれるが、"
        "makerは未約定リスク・逆選択を表現しない楽観シナリオである点に注意",
    )
    parser.add_argument(
        "--signal-layer",
        choices=["off", "append", "only"],
        default="off",
        help="因果的Ridgeアルファ信号レイヤー（予測とトレードの責務分離）。"
        "append=既存特徴+信号、only=特徴を信号だけに置き換え"
        "（RLは予測の発見から解放されサイジング/タイミングに専念、"
        "観測次元が劇的に減る）。信号は過去データのみのローリング再学習で"
        "生成されるためwalk-forward検証でもリークしない。既定off",
    )
    parser.add_argument(
        "--signal-model",
        choices=["ridge", "gbm"],
        default="ridge",
        help="--signal-layer の予測器。ridge=線形（既定）、gbm=LightGBM勾配"
        "ブースティング（表形式金融データで線形を上回りやすい。要 [research] extra）。"
        "因果ローリング構造・embargoはどちらも同一",
    )
    parser.add_argument(
        "--signal-train-window",
        type=int,
        default=4000,
        help="--signal-layer のRidge/GBMローリング学習窓（バー数）",
    )
    parser.add_argument(
        "--signal-refit-every",
        type=int,
        default=400,
        help="--signal-layer のRidge再学習間隔（バー数）",
    )
    parser.add_argument(
        "--trend-sleeve",
        type=float,
        nargs="+",
        default=[],
        help="--phase train専用。RLの実行済みウェイトとtrend_followingベース"
        "ラインを w_blend=(1-f)*w_rl+f*w_trend で合成した2スリーブ合成book"
        "をOOS比較に追記する（複数値可、例: 0.3 0.5）。RLをmarket-neutral"
        "(--target cs_demean --beta-neutral)で学習した場合に失う方向性ベータ"
        "を、ルールベースのトレンドフォローで補う狙い。ゲート2の合否判定には"
        "使わず参考表示のみ（既定[]=無効）",
    )
    parser.add_argument(
        "--eval-money-manager",
        action="store_true",
        help="--phase train専用。因果的な金銭管理アロケータ"
        "（Ridge相対アルファ+時系列モメンタム方向性ベータ、学習スライスのみで"
        "適合しholdout/testで評価）をOOS比較に追記し、ゲートに "
        "money_manager_beat_trend_following / rl_beat_money_manager を加える。"
        "「教師あり予測+ルールサイジングはRL/純トレンドに勝てるか」の判定用。"
        "RL学習不要で数秒（既定off）",
    )
    parser.add_argument(
        "--mm-vol-target",
        type=float,
        default=0.0,
        help="--eval-money-manager のボラ目標（年率）。0で無効（combined_teacher"
        "のグロス上限のみ）。例: 0.5 で年率50%ボラにグロスを因果的に調整",
    )
    parser.add_argument(
        "--mm-components",
        choices=["ridge", "trend", "both"],
        default="both",
        help="--eval-money-manager の成分。ridge=相対アルファのみ、"
        "trend=方向性ベータのみ、both=両者合成（既定）",
    )
    parser.add_argument(
        "--mm-rebalance-every",
        type=int,
        default=24,
        help="--eval-money-manager の回転抑制間隔（バー数）。目標をNバー毎に"
        "のみ再計算し中間は保持する（弱IC実データでのコスト暴走を防ぐ）。既定24",
    )
    parser.add_argument(
        "--mm-risk-parity",
        action="store_true",
        help="--eval-money-manager にskfolio Hierarchical Risk Parityを重ね、"
        "直近リターンの共分散構造からクロスセクショナルなリスク予算を組み替える"
        "（相関の高い銘柄群への配分集中を是正）。要 [research] extra（既定off）",
    )
    parser.add_argument(
        "--mm-risk-parity-scope",
        choices=["ridge_only", "full"],
        default="ridge_only",
        help="--mm-risk-parity の適用範囲。ridge_only(既定・推奨)=相対アルファ"
        "成分のみにHRP適用。full=合成後の全体に適用（実データ検証で強トレンド期"
        "に悪化を確認済み、比較用）。方向性ベータ成分は意図的に相関した市場全体"
        "の動きを取りに行く設計のため、HRPで縮小すると trend_following 由来の"
        "エッジを損なう",
    )
    parser.add_argument(
        "--mm-risk-parity-lookback",
        type=int,
        default=96,
        help="--mm-risk-parity のHRP適合に使う直近リターンの窓（バー数）。既定96",
    )
    parser.add_argument(
        "--mm-confidence-gate",
        action="store_true",
        help="--eval-money-manager で、相対アルファ成分の直近実現損益（因果的）を"
        "自己参照し、負けている間はtrendへ、勝っている間はアルファへ動的に"
        "ブレンド比率を変える（combined_teacherの単純合算がtrendを常時希釈する"
        "問題の是正）。--mm-risk-parityより優先（併用は未対応）。"
        "cg-lookback/cg-alpha-scaleを評価対象のholdoutに直接チューニングすると"
        "過学習するため、別のval区間で選定した値を渡すこと（既定off）",
    )
    parser.add_argument(
        "--mm-cg-lookback",
        type=int,
        default=100,
        help="--mm-confidence-gate のトレーリング実現損益を測る窓（バー数）。既定100",
    )
    parser.add_argument(
        "--mm-cg-min-lookback",
        type=int,
        default=50,
        help="--mm-confidence-gate でこの本数未満の履歴では純trend（ゲート無効）。既定50",
    )
    parser.add_argument(
        "--mm-cg-alpha-scale",
        type=float,
        default=0.02,
        help="--mm-confidence-gate でconf=1(アルファ全開)に達するトレーリング"
        "リターンの基準値。既定0.02",
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
