# Trade RL（MarS Lite）

7銘柄のポートフォリオ配分を1時間ごとに決定するRLエージェント。
多時間軸（15m/1h/4h/1d）特徴・オーダーフロー・funding rateを観測に使い、
コスト控除後リターンを最大化する。

**設計の正典**: [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)（既定値・ゲート体系・ベンチマーク根拠）

学習はCLI（`scripts/train_portfolio.py`）に一本化。運用は
`GET /api/signal/latest` のポーリングのみ（下記「シグナルサーバー」参照）。

---

## RLの強さは引き出せているか？ → 合成データ上では証拠あり

判定基準は「**チューニング済みルールに勝てるか（ゲート2）**」。

| 市場 | RLフルスタック | 最良ルール（クロスモメンタム則） | 判定 |
|---|---|---|---|
| 強モメンタム（cross） | +2129% | +1436% | RL勝ち |
| 平均回帰（meanrev） | +175% | −48% | RL圧勝 |
| 弱シグナル（IC 0.11） | +81.5% | +56% | RL勝ち |

**限界**: 上記は合成データ。実データ（IC 0.02〜0.05想定）での証明はP1以降。
Hyperliquid生特徴のみでは OOSランクIC ≈ 0.006 で**ICゲート不合格**（正しい撤退動作）。

### 累積効果（P0陽性・240日・各10万ステップOOS）

| 構成 | リターン | Sharpe | maxDD | 回転率 | 捕捉率 |
|---|---|---|---|---|---|
| baseline | +207% | 18.0 | 8.6% | 889 | 14% |
| ＋後処理 | +506% | 32.4 | 5.8% | 259 | 35% |
| ＋BCウォームスタート | +1131% | 48.9 | 2.4% | 90 | 79% |
| ＋3シードアンサンブル | **+1496%** | **53.6** | **2.0%** | 62 | **104%** |

---

## セットアップ

Python **3.12推奨**（3.13/3.14はtorch wheel要確認）。

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate  /  Linux・Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
pip install "psycopg[binary]"   # Postgres投入時のみ
```

`uv` を使う場合:

```bash
uv venv --python 3.11
# Windows PowerShell: .venv\Scripts\activate
uv pip install -e . -r requirements.txt
```

---

## クイックスタート（データ不要）

まず健全性試験(P0)で学習システムが正しく機能するか確認する:

```bash
python scripts/train_portfolio.py --phase p0 --timesteps 300000
```

- **合格条件**: ①アルファ有データでB&H・フラット両方に勝つ ②ノイズデータでほぼ取引しない
- 出力: `output/portfolio/p0_report.json`、エクイティカーブ図
- 所要: CPUで約20〜30分

---

## データ取得

### 専用PostgreSQL（Docker・推奨）

既存のTrade Platform DB（5432）と干渉しないよう、**ポート5433**の専用コンテナを使う。

```bash
docker compose up -d
```

| 項目 | 値 |
|---|---|
| コンテナ | `trade_rl_db` |
| DSN | `postgresql://trade_rl:trade_rl@localhost:5433/trade_rl` |
| 永続化 | ボリューム `trade_rl_trade_rl_pgdata` |

テーブル（`fetch_* --to postgres` が自動作成）:

| テーブル | 内容 |
|---|---|
| `rl_klines` | OHLCV（`source` = `binance` / `hyperliquid`、TF別） |
| `rl_funding_rate` | funding rate |
| `rl_orderflow_1m` | 1分オーダーフロー（Binanceのみ） |
| `rl_derivatives` | OI / L-S比率 / 清算代理（Binanceのみ・直近30日） |

投入状況の確認:

```bash
set PLATFORM_DB_URL=postgresql://trade_rl:trade_rl@localhost:5433/trade_rl
python scripts/check_pg.py
```

### Hyperliquid（認証不要・OHLCV + funding）

上位足（15m/1h/4h/1d）と funding を公開APIから取得。CSVキャッシュとPostgresの両方に書ける。

```bash
set PLATFORM_DB_URL=postgresql://trade_rl:trade_rl@localhost:5433/trade_rl

python scripts/fetch_hyperliquid.py \
  --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT BNBUSDT SUIUSDT DOGEUSDT \
  --days 180 --to csv postgres
```

- オーダーフロー・デリバティブ指標はHLでは取得不可（特徴はゼロ埋め）
- 銘柄は `BTCUSDT` でも `BTC` でも可（末尾USDT/USDCを自動正規化）

### Binance先物（フルセット: kline + funding + orderflow + derivatives）

```bash
set PLATFORM_DB_URL=postgresql://trade_rl:trade_rl@localhost:5433/trade_rl

# オーダーフロー(aggTrades)は重い。まずは --skip-orderflow で kline/funding を投入
python scripts/fetch_futures.py \
  --symbols BTCUSDT ETHUSDT XRPUSDT BNBUSDT SUIUSDT PAXGUSDT \
  --days 90 --to csv postgres --skip-orderflow

# オーダーフローも必要なら（日×銘柄で時間がかかる）
python scripts/fetch_futures.py \
  --symbols BTCUSDT ETHUSDT --days 30 --to postgres
```

- デリバAPI（OI/L-S）は**直近30日のみ**取得可能
- **長期OI/L-S**: `data.binance.vision` 日次 metrics ZIP（5分足・`fetch_derivatives` が自動使用）
- 長期間（例: 3年・10銘柄）の取得は数時間以上かかる。途中で停止しても
  再実行すれば既存データへの追記・重複スキップで再開できる
- `ETHBTC` は先物に無いためデフォルト銘柄から除外済み
- Binanceは一部地域からブロックされる。オフライン検証は下記サンプル生成を使う

### サンプルデータ（オフライン）

```bash
python scripts/generate_sample_data.py --days 60 --alpha cross --output ./data
```

---

## 本番ワークフロー（学習・検証）

### CSVソース（Binance fetch出力 or サンプル）

```bash
# Phase E 標準: walk-forward が実データ評価の既定コマンド
# 3fold × 3seed × コスト感度2倍 = 18ランの中央値で判断する（単発1曲線で判断しない）
python scripts/train_portfolio.py --phase wf --source csv --data ./data \
    --timesteps 500000 --ensemble 3 --folds 3 --n-seeds 3

# 確認用（最終モデルの保存・ゲート2判定を train_report.json に記録）
python scripts/train_portfolio.py --phase train --source csv --data ./data \
    --timesteps 2000000 --ensemble 3
```

> **Phase E 受入基準:**
> - `--phase wf` の中央値リターンがコスト2倍でもプラス（P3ゲート）
> - `train_report.json` の `gate2.passed = true`（RL が全ベースライン、特に `trend_following` を上回る）
> - `--ensemble 3` を実データの既定推奨（シード運のばらつき低減 + 不一致度スケーリング）

### Hyperliquid（`--source hyperliquid`、CSVキャッシュ不要）

`train_portfolio.py --source hyperliquid` は `HyperliquidSource` 経由で
公開APIから直接データを取得する（`fetch_hyperliquid.py`によるCSV事前取得は不要。
取得結果は `./data/hyperliquid/` に自動キャッシュされる。両者はキャッシュを共有する）。

```bash
python scripts/train_portfolio.py --phase train --source hyperliquid \
    --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT \
    --days 220 --warmup-days 100 --timesteps 2000000
```

> **注意（既知の問題）**: キャッシュは「ファイルが存在するか」しか見ておらず、
> 要求日数をカバーしているかは確認しない。先に短い日数で取得すると、後で
> `--days` を増やしても古いキャッシュが黙って再利用される。日数を増やす際は
> `rm -rf ./data/hyperliquid` してから再取得すること（docs/ARCHITECTURE.md §6）。

> **ウォームアップに注意**: 最長のローリング窓（1dTFのvol_ratio長期側=100日分）
> が埋まるまで特徴が不完全になる。実効学習期間をNdays確保したいなら、
> `--days` は (N+100) 程度を確保し `--warmup-days 100` で先頭を切り捨てる。

> **1h足の保持期間に上限あり**: Hyperliquid公開APIは1h足を約209日分しか
> 保持していない（実測、それ以前は0件）。`--days`をこれ以上増やしても
> 1h足では取得量は増えない（4h足は約833日、1d足は2000日以上保持）。
> より長期の実データが必要なら Binance（`scripts/fetch_futures.py`）を使う
> か`--days`をこの上限内に収めること（docs/ARCHITECTURE.md §6）。

> **ゲート1**: OOSランクIC ≥ 0.02。不合格ならRL学習に進まない（`--skip-gate` で強制続行可）。

### 収益最適化フェーズ

```bash
# PBT: ハイパーパラメータ自動探索
python scripts/train_portfolio.py --phase pbt --source csv --data ./data --pbt-pop 6 --pbt-gen 4 --pbt-steps 40000

# レジーム特化（bull/bear/range）
python scripts/train_portfolio.py --phase regime --source csv --data ./data --timesteps 500000

# 階層MTF（4hトレンドで方向制約）
python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000 --htf-gate
```

### RL強化の実験（オプトイン、未昇格）

`docs/ARCHITECTURE.md` §2.8/2.9 参照。いずれも既定offで、既定への昇格には
P0＋汎用性スイート×3シードでの再測定が必要（未実施）。

```bash
# 観測強化: 前ステップの後処理状態(vol_scale/dd_scale/disagreement_scale/est_port_vol)を観測に追加
python scripts/train_portfolio.py --phase train --source csv --data ./data --obs-risk-state

# リスクオーバーレイRL: グロスのスケールだけを別の小型PPOに学習させ、ルールベースと比較
python scripts/train_portfolio.py --phase overlay --source csv --data ./data \
    --timesteps 500000 --overlay-timesteps 100000
```

---

## 主なCLIオプション

| フラグ | 既定 | 説明 |
|---|---|---|
| `--phase` | p0 | `p0` / `train` / `wf` / `pbt` / `regime` / `overlay` |
| `--source` | synthetic | `synthetic` / `csv` / `postgres` / `hyperliquid` |
| `--ensemble` | 1 | シードアンサンブル数。**実データでは3推奨**（シード運低減 + 不一致度スケーリング） |
| `--gamma` | 0.5 | 割引率（0.995は崩壊する） |
| `--postproc` | full | 後処理（平滑/集中上限/ボラ目標/DDデリスク） |
| `--target-vol` | 0.5 | 年率ボラ目標。0以下で無効 |
| `--feature-mask` | off | IC安定性による特徴選別 |
| `--htf-gate` | off | 4hトレンドで方向制約 |
| `--obs-risk-state` | off | opt-in: 後処理状態を観測に追加（§2.8、未昇格） |
| `--disagreement-dr <x>` | 0 | opt-in: 学習中の不一致度ドメインランダム化上限（§2.8、未昇格） |
| `--overlay-timesteps` | 50000 | `--phase overlay`専用。リスクオーバーレイRLの学習ステップ数（§2.9、未昇格） |
| `--timesteps` | 300000 | 学習ステップ。実データ本番は200万〜 |
| `--folds` | 3 | walk-forward fold数 |
| `--n-seeds` | 3 | walk-forward で試すシード数 |
| `--lockbox-frac` | 0 | `--phase train`専用。末尾このfractionを最終封印テストに隔離（例: 0.15） |
| `--net-size` | small | 方策/価値ネット規模。`small`=実証済み、`large`=大容量・多層（要再ベンチ・`--dropout`併用推奨） |
| `--dropout` | 0 | 抽出器の隠れ層dropout率（large時の過学習/seed分散抑制。例 0.1） |
| `--feature-norm` | none | `rank_gauss`で各特徴をローリング・ガウスランクでN(0,1)に正規化（汎用性狙い・因果的） |
| `--beta-neutral` | off | 後処理で市場方向成分を除去しドル中立化（相対アルファのみ残す・opt-in） |
| `--warmup-days` | 0 | 先頭Ndaysを切り捨て（最長ローリング窓=100日分が埋まるまで特徴が不完全なため）。実効学習期間365日を確保したいなら取得465日+`--warmup-days 100` |

学習は既定で **TFゲート抽出器 + Ridge教師BC + 検証ベースモデル選択**（詳細は ARCHITECTURE.md §2.1）。
ハイパーパラメータの単一の正は `mars_lite/config.py`（`RunConfig`）。既定値を変える場合は
`tests/test_config.py`（憲法テスト）も合わせて更新すること。

---

## ゲート体系

1. **品質ゲート**: 欠損率・スパイク・timestamp整合 → 不合格銘柄除外
2. **リーク自己検査**: shuffleでIC消失 / 未来シフトでIC増大
3. **ゲート1（IC）**: 特徴に予測力があるか（実データの本丸）
4. **ゲート2（対ルール）**: `train_report.json` の `gate2.passed` で自動判定。`rl_beat_trend_following` が重要。
5. **ベースライン比較**: フラット / 等ウェイトB&H / ボラ逆数 / クロスモメンタム則
6. **オラクル則**: 手数料込みDP上限との捕捉率
7. **ウォークフォワード**: コスト2倍でも中央値プラスか
8. **Deflated Sharpe Ratio**: `--phase wf` の各コスト水準で自動計算し
   `walk_forward_cost*x.json` の `deflated_sharpe` に記録。fold×seedの
   試行回数で「そのSharpeが偶然でない確率」を補正する（目安 `dsr >= 0.95`）。
   何度も条件を変えて再実行するほど選択バイアスが乗るため、点推定の
   Sharpeだけでなくこちらも確認すること。
9. **ロックボックス（最終封印テスト）**: `--phase train --lockbox-frac 0.15` で
   末尾区間を全工程（ゲート・特徴マスク・fold分割・学習）から隔離し、
   最終モデルの評価に一度だけ使う。`train_report.json` の `lockbox` に記録。
   同じ出力ディレクトリで再実行すると使用済み警告が出る
   （繰り返し覗き見て判断を調整するのを防ぐため）。

---

## シグナル品質管理（全層）

| 層 | モジュール | 役割 |
|---|---|---|
| 前処理 | `data/quality.py` | 品質ゲート・銘柄除外 |
| 前処理 | `signal_check.py` | ICゲート・リーク自己検査 |
| 学習 | `bc_warmstart.py` | Ridge教師BC（IC合格時のみ） |
| 学習 | `policy_ensemble.py` | 多シード平均＋不一致縮小 |
| 後処理 | `trading/post_processor.py` | EMA平滑・ボラ目標・DDデリスク（train/serve一致） |
| 実行中 | `trading/guardrails.py` | 鮮度・NaN・損失上限 → フラット化 |

---

## シグナルサーバー

学習トリガー・リアルタイムメトリクス配信のUIは持たない（学習はCLIで行う設計。
`docs/ARCHITECTURE.md` §3「運用設計」参照）。サーバーは学習済みモデルから
推奨ウェイトを配信する薄いAPIのみ:

```bash
python scripts/run_server.py                 # http://localhost:8001
```

- **`GET /api/signal/latest`**: 推奨ウェイト（後処理・ガードレール適用後）を返す。
  任意で `prev_weights`（実ポジション、カンマ区切り）・`portfolio_value`・
  `peak_value` を渡すとEMA平滑・DDデリスクが実状態基準で機能する
- `GET /api/models`, `GET /api/models/{id}`, `DELETE /api/models/{id}`: 保存済みモデル管理
- `GET /api/data/available`: 利用可能なデータセット
- `GET /health`

```json
{
  "weights": {"BTCUSDT": 0.28, "ETHUSDT": -0.11},
  "net_exposure": 0.35, "gross_exposure": 0.82,
  "data_age_hours": 0.5, "stale": false,
  "guardrail": {"action": "proceed", "triggered": []}
}
```

`frontend/`（React UI）は旧アーキテクチャ（サーバー主導の学習制御）向けに
作られたもので、上記のAPI縮小に伴いコード凍結中。詳細は
[frontend/README.md](frontend/README.md) を参照。

---

## Trade Platform との連携

接点は2つ（Django側改修不要）:

1. 同一PostgreSQLの `rl_*` テーブル（`fetch_futures.py --to postgres` / `fetch_hyperliquid.py --to postgres`）
2. `GET /api/signal/latest`（Bots画面からポーリング）

---

## ディレクトリ構成

```
trade_rl/
├── mars_lite/
│   ├── config.py       # RunConfig: ハイパーパラメータの単一の正
│   ├── data/           # DataSource・品質ゲート・Postgres投入
│   ├── env/            # PortfolioTradingEnv
│   ├── features/       # FeaturePipeline・シグナル検証
│   ├── learning/       # PPO学習(trainer.py)・BC・アンサンブル・PBT
│   ├── pipeline/       # 学習フェーズ実装(p0/train/wf/pbt/regime/overlay)
│   ├── trading/        # 後処理・執行・ガードレール・DecisionPipeline・リスクオーバーレイ
│   ├── serving/        # モデル永続化・バージョン管理(model_store)
│   └── server/         # シグナルサーバー(signal_server)
├── scripts/            # fetch / train_portfolio / run_server
├── frontend/           # React UI（コード凍結中、frontend/README.md参照）
├── docs/ARCHITECTURE.md
├── docker-compose.yml  # 専用Postgres（ポート5433）
└── tests/
```

---

## テスト

```bash
python -m pytest tests/ -v
```

---

## ロードマップ（抜粋）

1. **実データP1**: fetch → ゲート1。ここが本当の勝負
2. 弱シグナル領域: `--ensemble 3`
3. RL強化Stage A/B（`--obs-risk-state` / `--phase overlay`等）の正式ベンチマーク・既定化判断
4. 紙上運用2週間 → バックテスト乖離分析 → 資金投入判断

`--source postgres` / `--source hyperliquid` は実装済み（後者は動作確認済み、
docs/ARCHITECTURE.md §6の既知の問題に注意）。

---

## ライセンス

MIT
