# Trade RL（MarS Lite）

複数銘柄のポートフォリオ配分を強化学習（PPO）で決定するトレーディングシステム。
多時間軸（15m/1h/4h/1d）の特徴量・オーダーフロー・funding rate を観測に使い、
取引コストを差し引いた後のリターンを最大化するように学習する。

学習だけでなく、**リスク管理・監視・モデル管理・デプロイまで一気通貫**で揃っている。
「学習スクリプトが1本あるだけ」ではなく、以下がひとつのパイプラインとしてつながっている。

```
データ取得 → 特徴量生成 → ゲート判定 → RL学習 → 評価（Walk-Forward/ブートストラップ）
   → モデル登録 → Shadow/Canary/Production デプロイ → 監視（ドリフト/ガードレール）
```

**設計の正典**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（既定値・ゲート体系・ベンチマーク根拠の理由が全て書いてある）

---

## 目次

1. [全体像（はじめての人向け）](#全体像はじめての人向け)
2. [セットアップ](#セットアップ)
3. [5分でわかる動作確認（P0）](#5分でわかる動作確認p0)
4. [データ取得](#データ取得)
5. [学習フェーズの使い分け](#学習フェーズの使い分け)
6. [評価とゲート体系](#評価とゲート体系)
7. [リスク制御・モデル管理・監視](#リスク制御モデル管理監視)
8. [デプロイパイプライン](#デプロイパイプライン)
9. [ダッシュボード](#ダッシュボード)
10. [主なCLIオプション一覧](#主なcliオプション一覧)
11. [ディレクトリ構成](#ディレクトリ構成)
12. [テスト・CI](#テストci)
13. [レガシー: 執行エージェント（v1）](#レガシー-執行エージェントv1)
14. [トラブルシューティング](#トラブルシューティング)

---

## 全体像（はじめての人向け）

このリポジトリは大きく3つの部分に分かれている。

| 部分 | 何をするか | 主な場所 |
|---|---|---|
| **① 学習パイプライン** | データを取得し、特徴量を作り、RLエージェントを学習・評価する | `mars_lite/pipeline/`, `mars_lite/env/`, `mars_lite/learning/` |
| **② 評価・監視** | 学習したモデルが「本番で使って良いか」を定量的に判定する | `mars_lite/eval/`（リプレイ検証・ドリフト監視・統計的有意性検定） |
| **③ 運用・デプロイ** | モデルのバージョン管理、段階的リリース、事故対応の手順 | `mars_lite/server/`, `mars_lite/trading/`, `docs/runbook_*.md` |

**まず動かして感触を掴みたいだけなら「[5分でわかる動作確認](#5分でわかる動作確認p0)」だけ読めば十分。**
実データで本番学習をしたい場合は「[データ取得](#データ取得)」→「[学習フェーズの使い分け](#学習フェーズの使い分け)」の順に読む。

### 用語ミニ辞典

| 用語 | 意味 |
|---|---|
| **フェーズ (`--phase`)** | 学習スクリプトの動作モード。p0（健全性試験）/ train（単発学習）/ wf（walk-forward評価）/ pbt（ハイパラ探索）/ regime（レジーム特化アンサンブル） |
| **ゲート** | 「次の工程に進んで良いか」を機械的に判定するチェックポイント。品質ゲート・ICゲート・ゲート2（対ベースライン）・デプロイゲートなど複数ある |
| **Walk-Forward評価** | 時系列データを複数の学習/検証期間に分割し、シードを変えて繰り返し学習・評価すること。1回の幸運な結果に騙されないための標準的な検証方法 |
| **レジーム (Regime)** | 相場の状態（強気/弱気/レンジ × 高ボラ/低ボラなど）。本プロジェクトは8状態のステートマシンで分類し、状態ごとに専門方策を使い分ける |
| **リプレイシミュレータ** | 1分足の実売買データ（aggTrades）に対して注文を約定させ、より粗い「バー単位」の近似シミュレーションとの乖離を定量化する仕組み |

---

## セットアップ

依存管理は **[uv](https://docs.astral.sh/uv/)** に一本化されている。
Python **3.12推奨**。

```bash
# uv 未導入の場合
pip install uv

# 依存関係を uv.lock どおりに再現インストール
uv sync --all-extras --dev

# 動作確認
uv run python -c "import mars_lite; print('OK')"
```

以降のコマンドは全て `uv run <コマンド>` の形で実行する。

---

## 5分でわかる動作確認（P0）

外部データ不要。合成データで「学習システムが正しく機能しているか」を確認する健全性試験。

```bash
uv run python scripts/train_portfolio.py --phase p0 --timesteps 300000
```

- **合格条件**: ①アルファ有データで Buy&Hold・フラット両方に勝つ ②ノイズだけのデータではほぼ取引しない
- 出力: `output/portfolio/p0_report.json`、エクイティカーブ図（`output/portfolio/p0_*_equity.png`）
- 所要時間: CPUで約20〜30分

`--phase p0` が失敗する（RLが機能していない）状態で他のフェーズに進んでも意味がないので、まずここから。

---

## データ取得

3つのデータソースがあり、`--source` で切り替える。

| ソース | コマンド例 | 特徴 |
|---|---|---|
| `synthetic` | （何もフェッチ不要、`--phase p0` などで自動生成） | 外部依存なし。動作確認・アルゴリズム検証向け |
| `postgres` | 下記参照 | 本番データの本命。1分足オーダーフローまで含む最も情報量の多いソース |
| `hyperliquid` | 下記参照 | 認証不要の公開APIから直接取得。手早く実データを試したいとき向け |

### ① 専用PostgreSQL（Docker・本番データ）

**ポート5433**の専用コンテナを使う。

```bash
docker compose up -d
```

| 項目 | 値 |
|---|---|
| コンテナ | `trade_rl_db` |
| DSN | `postgresql://trade_rl:trade_rl@localhost:5433/trade_rl` |
| 永続化 | ボリューム `trade_rl_pgdata` |

テーブル（`fetch_futures.py --to postgres` / `fetch_hyperliquid.py --to postgres` が自動作成）:

| テーブル | 内容 |
|---|---|
| `rl_klines` | OHLCV（`source` = `binance` / `hyperliquid`、時間軸別） |
| `rl_funding_rate` | funding rate |
| `rl_orderflow_1m` | 1分オーダーフロー集計（Binanceのみ） |
| `rl_derivatives` | OI（建玉） / ロングショート比率 / 清算代理指標（Binanceのみ） |

データ投入（Binance先物: klines + funding + orderflow + derivatives のフルセット）:

```bash
# vision = data.binance.vision の日次ZIPから取得（長期バックフィル向け・推奨）
uv run python scripts/fetch_futures.py \
  --days 600 --to postgres --klines-source vision \
  --dsn "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"

# aggTrades集計(orderflow)は重い処理なので、まず kline/funding だけ先に入れたい場合
uv run python scripts/fetch_futures.py \
  --days 90 --to postgres --skip-orderflow \
  --dsn "postgresql://trade_rl:trade_rl@localhost:5433/trade_rl"
```

- `--dsn` を省略する場合は環境変数 `PLATFORM_DB_URL` を設定しておく
- `--symbols` を省略すると `mars_lite/pipeline/dataset_builder.py` の `DEFAULT_SYMBOLS`（15銘柄）が対象になる
- 既に取り込み済みの日付は自動的にスキップされるので、再実行しても差分だけ追加取得される
- **銘柄ごとにDB内のヒストリー長が異なる**点に注意。`--source postgres` は `--days` を見ずDB内の全期間をそのまま使うため、
  古参銘柄（BTCUSDT/ETHUSDTなど）と新規追加銘柄で実際に使えるヒストリー長に差が出ることがある
- デリバティブAPI（OI/ロングショート比率）は取引所の仕様上、REST経由では直近分しか取れない。長期分は `data.binance.vision` の日次metrics ZIPを自動フォールバック利用する

投入状況の確認:

```bash
set PLATFORM_DB_URL=postgresql://trade_rl:trade_rl@localhost:5433/trade_rl
uv run python scripts/check_pg.py
```

### ② Hyperliquid（認証不要・OHLCV + funding）

上位足（15m/1h/4h/1d）と funding rate を公開APIから直接取得する。CSVキャッシュ・Postgresの両方に書ける。
オーダーフロー・デリバティブ指標はHyperliquidでは取得不可のため特徴量はゼロ埋めになる。

```bash
uv run python scripts/fetch_hyperliquid.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT \
  --days 180 --to csv postgres
```

- 銘柄は `BTCUSDT` でも `BTC` でも可（末尾 USDT/USDC を自動正規化）
- 学習時は `--source hyperliquid --days <N>` でそのまま接続できる

> **1h足の保持期間に上限あり**: Hyperliquid公開APIは1h足を約209日分しか
> 保持していない（実測、それ以前は0件）。`--days`をこれ以上増やしても
> 1h足では取得量は増えない（4h足は約833日、1d足は2000日以上保持）。
> より長期の実データが必要なら Binance（`scripts/fetch_futures.py`）を使う
> か`--days`をこの上限内に収めること。

### ③ Bitget / OKX（認証不要・USDT建て無期限先物）

Bitget・OKXの公開APIから直接OHLCV + fundingを取得する。オーダーフロー・
デリバティブ指標は両取引所とも非対応のため特徴量はゼロ埋めになる。
学習時は `--source bitget --days <N>` / `--source okx --days <N>` で
接続でき、初回アクセス時に自動でCSVキャッシュ（`./data/bitget/` /
`./data/okx/`）へ保存される（専用フェッチスクリプトは不要）。

- OKXは3ソース中もっとも長期の1h足履歴を持つ（実測: 1000日以上前まで取得可能）
- Bitgetは90日/1000本ずつページングして取得する

### ③ サンプルデータ（オフライン）

```bash
uv run python scripts/generate_sample_data.py --days 60 --alpha cross --output ./data
```

`--source csv --data ./data` で読み込める。

---

## 学習フェーズの使い分け

エントリーポイントは1本、`--phase` で挙動を切り替える。

```bash
uv run python scripts/train_portfolio.py --phase <phase> --source <source> [オプション...]
```

| フェーズ | 用途 | いつ使うか |
|---|---|---|
| `p0` | 合成データでの健全性試験 | 最初の動作確認、コード変更後のリグレッション確認 |
| `train` | 単発学習（最終モデル保存 + ゲート2判定） | 本番用モデルを1本仕上げたいとき |
| `wf` | Walk-Forward評価（複数fold×複数シード×コスト感度） | 「このアプローチは実データで安定して機能するか」を検証したいとき（本番判断の本丸） |
| `pbt` | Population Based Training によるハイパラ自動探索 | ハイパーパラメータをチューニングしたいとき |
| `regime` | 8状態Regime FSM + レジーム特化アンサンブル学習 | 相場状況ごとに専門方策を使い分けたいとき |

### 代表的なコマンド例

```bash
# Walk-forward評価（実データ検証の標準コマンド）
# 3fold × 3seed × コスト感度2倍 = 18ランの中央値で判断する（単発1本の結果で判断しない）
uv run python scripts/train_portfolio.py --phase wf --source postgres \
    --timesteps 500000 --ensemble 3 --folds 3 --n-seeds 3

# 本番用モデルの学習・保存（train_report.json に gate2 判定が記録される）
uv run python scripts/train_portfolio.py --phase train --source postgres \
    --timesteps 2000000 --ensemble 3

# ハイパーパラメータ探索
uv run python scripts/train_portfolio.py --phase pbt --source postgres \
    --pbt-pop 6 --pbt-gen 4 --pbt-steps 40000

# 8状態レジーム特化アンサンブル（自動較正つき）
uv run python scripts/train_portfolio.py --phase regime --source postgres \
    --timesteps 500000 --regime-trials 100

# 階層MTF（4hトレンドで方向を制約、1hでサイジング）
uv run python scripts/train_portfolio.py --phase train --source postgres \
    --timesteps 2000000 --htf-gate
```

> **ゲート1（IC）**: OOSランクIC ≥ 0.02 が既定の合格ライン。不合格ならRL学習に進まない（`--skip-gate` で強制続行可）。

---

## 評価とゲート体系

学習したモデルを「勘」ではなく段階的なゲートで判定する。

1. **品質ゲート** (`data/quality.py`): 欠損率・スパイク・timestamp整合性 → 不合格銘柄は自動除外
2. **リーク自己検査** (`signal_check.py`): シャッフルするとICが消え、未来にシフトするとICが増えることを確認（データリークの検出）
3. **ゲート1（IC）**: 特徴量に予測力があるか。実データでの本当の勝負どころ
4. **ゲート2（対ルール）**: `train_report.json` の `gate2.passed` で自動判定。特に `rl_beat_trend_following`（トレンドフォロー則に勝てているか）が重要
5. **ベースライン比較**: フラット / 等ウェイトBuy&Hold / ボラ逆数 / クロスモメンタム則 / DPオラクル（手数料込み理論上限）
6. **Walk-Forward**: コスト2倍でも中央値プラスを保てるか（`--phase wf`）
7. **リプレイ検証（M1）**: `mars_lite/eval/replay_sim.py`。1分足の実約定データに対する `ReplaySimulator` の結果と、バー単位近似シミュレーションとのSharpe差を定量化する。乖離が大きい場合は学習時の執行コスト前提が現実と合っていない可能性がある

```python
from mars_lite.eval.replay_sim import ReplaySimulator, compare_bar_vs_replay

result = ReplaySimulator(fee_rate=0.0005, max_participation_rate=0.1).simulate(
    agg_trades_df, orders, initial_cash=1_000_000.0
)
comparison = compare_bar_vs_replay(bar_returns, result.returns, tolerance=0.3)
# comparison["within_tolerance"] が False ならバー近似と実売買の乖離が許容範囲外
```

8. **ブートストラップ有意性検定（M3）**: `mars_lite/eval/bootstrap_eval.py`。候補モデルとベースラインのSharpe差について、信頼区間とp値をブートストラップで算出する

```python
from mars_lite.eval.bootstrap_eval import bootstrap_sharpe_difference

result = bootstrap_sharpe_difference(candidate_returns, baseline_returns, ci=0.95)
# result: mean, lower_ci, upper_ci, p_value, observed_diff
# lower_ci > 0 かつ p_value が十分小さければ「偶然ではなくベースラインに勝っている」と言える
```

---

## リスク制御・モデル管理・監視

### 発注前リスク検証（`mars_lite/trading/pre_trade_risk.py`）

目標ウェイト・想定元本・単一銘柄比率・レバレッジ上限などを**発注前**に検証し、違反時は `PreTradeRejection` を送出して発注をブロックする。事後のガードレール（損失発生後に検知）とは異なり、そもそも危険な発注を出させない。

### 学習マニフェスト（`mars_lite/learning/manifest.py`）

学習完了時に `output/portfolio/model_manifest.json` を自動生成する。Gitコミットハッシュ・データハッシュ・ハイパーパラメータ・シードを記録し、同一条件での再学習・再現性検証を可能にする。

### モデルレジストリ（`mars_lite/server/model_registry.py`）

学習済みモデルをバージョン管理し、アクティブモデルの切り替え・ロールバックをCLIで行う。

```bash
# 登録
uv run python -m mars_lite.server.model_registry register output/portfolio/portfolio_model.zip \
    --metrics '{"sharpe": 1.8}'

# 一覧表示
uv run python -m mars_lite.server.model_registry list

# 指定バージョンをアクティブ化
uv run python -m mars_lite.server.model_registry activate <version>

# 1つ前のアクティブバージョンにロールバック
uv run python -m mars_lite.server.model_registry rollback
```

### ドリフト監視（`mars_lite/eval/drift_monitor.py`）

特徴量分布（PSI / KS検定）とモデル予測分布を継続的に監視し、閾値超過時にアラートを発報してポジション縮小（flatten）を促す。

```python
from mars_lite.eval.drift_monitor import DriftMonitor, DriftMonitorConfig

monitor = DriftMonitor(reference_features, config=DriftMonitorConfig(psi_threshold=0.1))
report = monitor.evaluate(current_features)
if report.should_flatten:
    ...  # ポジション縮小処理へ
```

### 実行時ガードレール（`mars_lite/trading/guardrails.py`）

データ鮮度・NaN混入・損失上限超過を検知して自動フラット化する、事後防御の最終ライン。

### 8状態 Regime FSM（`mars_lite/learning/regime_fsm.py`）

トレンド3分類（Bull/Bear/Range）×ボラティリティ2分類（High/Low）の6状態 + 極端なトレンド2状態（`extreme_bull`/`extreme_bear`）＝計8状態。
`mars_lite/learning/regime_calibrator.py` がOptunaで閾値を自動較正し、`--phase regime` で状態ごとの専門方策（Specialist）を学習・ルーティングする。

---

## デプロイパイプライン

Shadow → Canary → Production の3段階を経なければ本番反映できないCIゲート（`mars_lite/server/deployment_gate.py`）を用意している。GitHub Actions `.github/workflows/deploy.yml` から手動実行（`workflow_dispatch`）する。

```bash
gh workflow run deploy.yml -f stage=canary -f shadow_passed=true -f canary_passed=false
```

`production` ステージへの遷移には `shadow_passed` / `canary_passed` に加えて `approval_ticket`（承認チケット番号）が必須。

運用ドキュメント:

| ドキュメント | 内容 |
|---|---|
| [docs/runbook_incident_response.md](docs/runbook_incident_response.md) | Shadow/Canary/Production障害時の対応手順・重大度定義 |
| [docs/runbook_compliance.md](docs/runbook_compliance.md) | モデル昇格時に保存すべき証跡（マニフェスト・ドリフトレポート・デプロイゲート証跡） |
| [docs/model_decision_log.md](docs/model_decision_log.md) | モデル採用/却下の意思決定ログ様式 |

---

## ダッシュボード

```bash
# ターミナル1（リポジトリルートから）
uv run python scripts/run_server.py                 # http://localhost:8001

# ターミナル2
cd frontend && npm install && npm run dev            # http://localhost:5173
```

- `mode: portfolio` で学習起動・メトリクス監視・バックテストが可能
- **`GET /api/signal/latest`**: 推奨ウェイト（後処理・ガードレール適用後）を返す

```json
{
  "weights": {"BTCUSDT": 0.28, "ETHUSDT": -0.11},
  "net_exposure": 0.35, "gross_exposure": 0.82,
  "data_age_hours": 0.5, "stale": false,
  "guardrail": {"action": "proceed", "triggered": []}
}
```

---

## 主なCLIオプション一覧

`uv run python scripts/train_portfolio.py --help` で全オプションを確認できる。よく使うものだけ抜粋:

| フラグ | 既定値 | 説明 |
|---|---|---|
| `--phase` | `p0` | `p0` / `train` / `wf` / `pbt` / `regime` |
| `--source` | `synthetic` | `synthetic` / `csv` / `postgres` / `hyperliquid` |
| `--days` | 240 | 取得日数（`synthetic`/`hyperliquid`のみ有効。`postgres`はDB内の全期間を使用） |
| `--symbols` | `DEFAULT_SYMBOLS`（15銘柄） | 対象銘柄を上書き指定 |
| `--timesteps` | 300,000 | 学習ステップ数。実データ本番は200万〜推奨 |
| `--ensemble` | 1 | シードアンサンブル数。**実データでは3推奨**（シード運のばらつき低減 + 不一致度スケーリング） |
| `--gamma` | 0.5 | 割引率（行動効果が即時のため低い値が有効。0.995は崩壊する） |
| `--folds` | 3 | Walk-forward の fold 数 |
| `--n-seeds` | 3 | Walk-forward で試すシード数 |
| `--postproc` | `full` | 後処理（平滑化/バンド/ボラ目標/DDデリスク）。`legacy`は射影のみ |
| `--target-vol` | 0.5 | 年率ボラティリティ目標。0以下で無効 |
| `--feature-mask` | off | IC安定性による特徴量選別を有効化 |
| `--htf-gate` | off | 上位足（4h）トレンドで方向を制約、1hでサイジング |
| `--scan-horizons` | off | 学習前にホライズンスキャンを行い、最良ホライズンを自動選択 |
| `--bc-teacher` | `auto` | Behavior Cloning事前学習の教師（`ridge`/`ts_momentum`/`momentum`/`oracle`） |
| `--calibrate-regime` | on | `--phase regime` でOptunaによる閾値自動較正を実行するか |
| `--regime-trials` | 100 | Regime FSM自動較正の試行回数 |
| `--pg-dsn` | 環境変数 `PLATFORM_DB_URL` | `--source postgres` の接続文字列 |
| `--skip-gate` | off | ゲート1不合格でも学習を強制続行 |

学習は既定で **TFゲート抽出器 + Ridge教師BC + 検証ベースモデル選択** を使う（詳細は [ARCHITECTURE.md](docs/ARCHITECTURE.md) §2.1）。

---

## ディレクトリ構成

```
trade_rl/
├── mars_lite/
│   ├── data/          # DataSource実装・品質ゲート・Postgres投入
│   ├── env/           # PortfolioTradingEnv（RL環境本体）
│   ├── features/      # FeaturePipeline・IC/リークの自己検査
│   ├── learning/       # PPO学習・BC・アンサンブル・Regime FSM・較正
│   ├── trading/        # 発注前リスク検証・後処理・実行時ガードレール
│   ├── eval/           # リプレイシミュレータ・ドリフト監視・ブートストラップ評価・walk-forward
│   ├── server/         # FastAPIダッシュボードAPI・モデルレジストリ・デプロイゲート
│   └── pipeline/       # train_portfolio.py の実体（cli/dataset_builder/training_engine/evaluator）
├── scripts/            # fetch_* / train_portfolio.py / evaluate / backtest 等
├── frontend/           # React製ダッシュボードUI
├── docs/
│   ├── ARCHITECTURE.md              # 設計の正典
│   ├── runbook_incident_response.md # 障害対応手順
│   ├── runbook_compliance.md        # コンプライアンス証跡要件
│   └── model_decision_log.md        # モデル意思決定ログ
├── .github/workflows/
│   ├── ci.yml          # lint/型検査/テスト（カバレッジ70%以上必須）
│   └── deploy.yml       # Shadow/Canary/Productionデプロイゲート
├── docker-compose.yml  # 専用Postgres（ポート5433）
├── uv.lock             # 依存関係ロックファイル
└── tests/
```

---

## テスト・CI

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run mypy mars_lite
uv run pytest tests/ -v --cov=mars_lite --cov-report=term-missing
```

PR作成時、GitHub Actions（`.github/workflows/ci.yml`）が ruff / mypy / pytest（カバレッジ70%以上）を自動実行する。

主なテスト（用途がわかりやすいものを抜粋）:

| テストファイル | 検証内容 |
|---|---|
| `tests/test_pre_trade_risk.py` | 想定元本・単一銘柄比率・レバレッジ超過時に `PreTradeRejection` が発生するか |
| `tests/test_model_manifest.py` | マニフェストのハッシュ・パラメータ記録、再学習時のOOS Sharpe再現性 |
| `tests/test_regime_fsm.py` | 8状態すべての遷移シナリオとSpecialistルーティング |
| `tests/test_replay_sim.py` | バーシミュレータとリプレイシミュレータのSharpe乖離定量化 |
| `tests/test_drift_monitor.py` | 合成ドリフト注入時のPSI/KSアラートとポジション縮小発火 |
| `tests/test_bootstrap_eval.py` | Sharpe差分ブートストラップCI・p値の正しさ |
| `tests/test_model_registry.py` | モデル一覧・アクティブ化・ロールバック |
| `tests/test_deployment_gate.py` | Shadow経由なしでのProduction反映がブロックされるか |

---

## レガシー: 執行エージェント（v1）

OHLCVのみからマーケットインパクトを推定し、注文分割・Almgren-Chriss報酬で学習する旧ワークフロー。ポートフォリオRL（本体）とは独立した古い機能で、現在はメンテナンスモード。

```bash
# データ取得
uv run python scripts/fetch_binance.py --symbol BTCUSDT --output ./data

# 学習・評価・バックテスト
uv run python scripts/train.py --data ./data --symbol BTCUSDT --multi-tf --timesteps 100000 --output ./output
uv run python scripts/evaluate.py --model ./output/final_model.zip --episodes 10
uv run python scripts/backtest.py --model ./output/final_model.zip --data ./data --symbol BTCUSDT

# 進化学習（PBT-MAP-Elites）
uv run python scripts/run_evolution.py --generations 10 --population 25 --steps-per-gen 10000 --data-dir data
```

Windowsでは `num_envs=1` を推奨（`SubprocVecEnv` のpickle問題のため）。

---

## トラブルシューティング

| 症状 | 原因・対処 |
|---|---|
| `--to postgres には --dsn か PLATFORM_DB_URL が必要です` | `--dsn` を明示指定するか、環境変数 `PLATFORM_DB_URL` を設定する |
| `品質ゲート通過銘柄が2未満です` | 指定銘柄の実データ期間が短すぎるか未取得。`--days` を短くするか、対象銘柄のバックフィルを先に行う |
| `--source postgres` で `--days` を変えても取得期間が変わらない | 仕様。Postgresソースは `--days` を見ずDB内の全期間を使う。期間を絞りたい場合は取得済みデータそのものを調整する |
| Binanceへの接続がブロックされる | 地域制限の可能性。`scripts/generate_sample_data.py` でのオフライン検証に切り替える |
| `psycopg` 関連の `SyntaxError` / import エラー | ローカルの依存キャッシュ破損の可能性。`uv pip install --reinstall --no-cache "psycopg[binary]>=3.1"` で再インストールする |

---

## ライセンス
一旦観閲のみ許可する。問い合わせは本人まで
