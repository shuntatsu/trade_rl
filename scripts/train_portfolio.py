"""
ポートフォリオRL学習CLI

フェーズ:
    p0    : 健全性試験。アルファ注入合成データ（陽性対照）と純ノイズ（陰性対照）
            の両方で学習し、①陽性でベースライン超え ②陰性で低回転 を確認する
    train : 指定ソースで学習（P2。実データはローカルPCで --source csv/postgres）
    wf    : ウォークフォワード検証（P3）

使い方:
    python scripts/train_portfolio.py --phase p0 --timesteps 300000
    python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000
    python scripts/train_portfolio.py --phase wf --source csv --data ./data

実処理は mars_lite.pipeline.phases / mars_lite.learning.trainer にある
（サーバー等の別エントリポイントからも同じ実装を使うため、このファイルは
引数パースとフェーズ選択のみを行う薄いCLI）。
"""

import argparse
from pathlib import Path

from mars_lite.pipeline.phases import phase_p0, phase_train, phase_pbt, phase_regime, phase_wf

# 後方互換: 以前はこのモジュールが train_ppo/make_env_fns を定義していた
# （server/training_manager.py がsys.path操作でこのファイルをimportしていた）。
# 現在の実装は mars_lite.learning.trainer にある。
from mars_lite.learning.trainer import train_ppo, make_env_fns  # noqa: F401


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
    parser.add_argument("--net-size", choices=["small", "large"], default="small",
                        help="方策/価値ネットの規模。small=実証済み（ARCHITECTURE.md"
                             "ベンチ構成、既定）、large=大容量・多層（多様/実データでの"
                             "汎用性狙い、要再ベンチ。dropout併用推奨）")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="抽出器の隠れ層dropout率（large時の過学習/seed分散抑制。"
                             "例: 0.1）。0で無効（既定）")
    parser.add_argument("--feature-norm", choices=["none", "rank_gauss"], default="none",
                        help="入力特徴の分布正規化。rank_gauss=各特徴をローリング・"
                             "ガウスランクでN(0,1)に写像し、資産・レジーム差への過適合を"
                             "抑える（汎用性狙い、因果的・リークなし）。既定none")
    parser.add_argument("--warmup-days", type=float, default=0,
                        help="先頭Ndaysをウォームアップとして切り捨てる（最長ローリング窓"
                             "=1dTFのvol_ratio長期側100日分が埋まるまで特徴が不完全なため）。"
                             "実効学習期間をNdays確保したいなら、取得を(N+warmup_days)日分"
                             "にして本フラグで切り捨てる運用にする。既定0=無効")
    parser.add_argument("--beta-neutral", action="store_true",
                        help="後処理で市場方向(等ウェイト平均)成分を除去しドル中立化。"
                             "全銘柄がBTCベータで共線なユニバースで相対アルファのみ"
                             "残す。方向性ベータを捨てるため上昇相場で不利になりうる（opt-in）")
    parser.add_argument("--lockbox-frac", type=float, default=0.0,
                        help="--phase train専用。末尾このfractionを、ゲート/特徴マスク/"
                             "fold分割/学習の全工程から隔離し、最終モデル評価に一度だけ"
                             "使う最終封印テストにする。0で無効（既定）。過学習検査用に"
                             "0.1〜0.15程度を推奨。同じ区間の再利用は警告される。")
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
    parser.add_argument("--obs-risk-state", action="store_true",
                        help="opt-in: 前ステップの後処理状態(vol_scale/dd_scale/"
                             "disagreement_scale/est_port_vol)を観測に追加し、"
                             "方策がルール層の挙動を予見できるようにする。"
                             "既定off（証拠なき機能は既定にしない、Stage A実験用）")
    parser.add_argument("--disagreement-dr", type=float, default=0.0,
                        help="opt-in: 学習中もエピソード毎に不一致度をU(0,x)で"
                             "ランダムに与え、方策がアンサンブル不一致縮小レイヤーを"
                             "経験できるようにする（単独方策学習中は常に0という"
                             "train/eval不一致の緩和策）。0で無効（既定、Stage A実験用）")
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
                        help="微小リバランス禁止バンド（デフォルト0.04=4%%未満の変更はスキップ）")
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
