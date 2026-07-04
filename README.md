# MarS Lite

OHLCVデータから仮想マーケットインパクトをシミュレートし、強化学習エージェントに**注文分割（Iceberg相当）**・**時間リスクとのトレードオフ（Almgren-Chriss）**を学習させる環境。

## 特徴

- **板情報不要**: OHLCVのみからマーケットインパクトを推定
- **Almgren-Chriss報酬**: 執行コスト最小化 + 在庫リスクペナルティ
- **Look-ahead bias防止**: Next Open基準価格、時刻別期待出来高使用
- **PBT-MAP-Elites**: 多様なスペシャリスト群の同時学習
- **環境適応推論**: 市場レジームに応じた個体選択
- **Multi-Timeframe**: 1m, 15m, 1h, 4h, 1dのデータを統合して学習
- **Cross-Symbol Learning**: 複数通貨によるランダム交差学習で汎化性能を向上
- **Smart Data Fetching**: 通貨ごとの上場日を自動検出し、無駄なAPIリクエストを削減
- **Market Cap Selection**: 時価総額（Market Cap）順での上位通貨選択に対応（ステーブルコイン除外機能付き）

## Architecture

```mermaid
graph TD
    subgraph Data Pipeline
        Binance[Binance API] -->|Fetch| RawData[Raw OHLCV]
        RawData -->|Split| DailyFiles[Daily Files (YYYY-MM-DD.csv)]
        DailyFiles -->|Load| MultiTF[MultiTimeframeLoader]
        MultiTF -->|Align| AlignedData[Aligned Multi-TF Data]
        AlignedData -->|Split| SplitData[Train/Val/Test Split]
    end

    subgraph Environment
        SplitData -->|Feed| SimEnv[MarsLiteEnv]
        SimEnv -->|Wrap| TFEnv[MarsLiteMultiTFEnv]
        TFEnv -->|Wrap| CrossEnv[CrossSymbolEnv]
        
        subgraph Market Simulation
            TFEnv -->|Obs| Agent
            Agent -->|Action| TFEnv
            TFEnv -->|Step| Match[Matching Engine]
            Match -->|Exec Price| Reward[Reward Function]
            Reward -->|Scalar| Agent
        end
    end

    subgraph Learning System
        CrossEnv -->|Samples| PPO[PPO Agent]
        PPO -->|Update| Policy[Policy Network]
        
        Sampler[RandomEpisodeSampler] -->|Reset Idx| CrossEnv
    end
```

## インストール

Python **3.12推奨**（3.13/3.14はtorchのwheel提供状況を確認してください）。

```bash
# リポジトリルートで実行
python -m venv .venv
# Windows: .venv\Scripts\activate  /  Linux・Mac: source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

## クイックスタート（Windows・全機能）

```bat
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

REM 1. データ取得（Binanceに繋がらない場合は下の「サンプルデータ生成」へ）
python scripts\fetch_binance.py --symbol BTCUSDT --output .\data

REM 2. CLI学習・評価・バックテスト
python scripts\train.py --data .\data --symbol BTCUSDT --multi-tf --timesteps 100000 --output .\output
python scripts\evaluate.py --model .\output\final_model.zip --episodes 10
python scripts\backtest.py --model .\output\final_model.zip --data .\data --symbol BTCUSDT

REM 3. ダッシュボード（ターミナル1: APIサーバー、必ずリポジトリルートから起動）
python scripts\run_server.py

REM ターミナル2: フロントエンド（Node 20以上）
cd frontend
npm install
npm run dev
REM → http://localhost:5173 を開く
```

注意点:
- `run_server.py` はリポジトリルートから起動してください（`data/`・`output/` はルート基準）
- Windowsでは学習設定の `num_envs` は `1` を推奨（SubprocVecEnvのspawnで大きなデータのpickleが遅い/失敗するため）
- pandas 2.2以降では頻度文字列は `"1min"` 形式です（`"1m"` は月と解釈されます）。時間軸ラベルの変換は `mars_lite/data/data_utils.py` の `TF_TO_PANDAS_FREQ` に一元化されています

### サンプルデータ生成（オフライン環境用）

Binance APIに接続できない環境では、合成データで全パイプラインを検証できます:

```bash
python scripts/generate_sample_data.py --symbols BTCUSDT ETHUSDT --days 14 --output ./data
```

## ポートフォリオ配分RL（v2・推奨ワークフロー）

7銘柄（btc/xrp/sui/bnb/eth/paxg/ethbtc）の目標ウェイトを1時間ごとに決定する
ポートフォリオRLエージェント。多時間軸（15m/1h/4h/1d）特徴・オーダーフロー・
funding rateを観測に使い、コスト控除後リターンを最大化する。

**合否ゲート方式**: 各フェーズに定量的な通過条件を設け、「儲かるか」を証拠付きで判定する。

### シグナルの生涯を通した品質管理（前処理・後処理・実行中）

学習アルゴリズムだけでなく、データ→特徴→方策→注文→運用の全層で品質を担保する。

**前処理**
- `mars_lite/data/quality.py`: 品質ゲート（欠損率・スパイク・timestamp重複/逆行・funding被覆）。不合格銘柄は学習から自動除外
- `signal_check.run_leak_self_test`: リーク検出器の自己検査（シャッフルでIC消失／未来シフトでIC増大を確認）

**学習**
- BCウォームスタート（`learning/bc_warmstart.py`）: クロスモメンタム教師の模倣で方策を初期化してからPPO微調整（デフォルトON）
- シードアンサンブル（`learning/policy_ensemble.py`, `--ensemble N`）: 多シード平均で分散低減＋不一致度でグロス縮小
- 検証ベースモデル選択（過学習対策）

**後処理**（`mars_lite/trading/post_processor.py`・学習と運用で同一適用）
- EMA平滑 → 集中上限 → レバレッジ1射影 → ボラターゲティング → DDデリスク → no-tradeバンド
- 学習環境の`step`内でも適用し、報酬が執行後の挙動を反映（train/serve skew防止）
- **効果**: P0で生方策比リターン倍増・Sharpe1.6倍・回転70%減・DD半減

**実行中**（`mars_lite/trading/guardrails.py`）
- データ鮮度・NaN・全ゼロ特徴 → フラット化
- 日次損失・DD上限 → フラット化、連続負け・回転異常 → グロス半減
- `/api/signal/latest` が生/後処理済みウェイト・鮮度・ガードレール状態を返す

### P0: 健全性試験（どこでも実行可、データ不要）

アルファを注入した合成データ（陽性対照）と純ノイズ（陰性対照）で
「学習システムが正しく機能しているか」を検証する:

```bash
python scripts/train_portfolio.py --phase p0 --timesteps 300000 --output ./output/portfolio_p0
```

通過条件: ①陽性でRLが等ウェイトB&Hとフラット両方に勝つ ②陰性でほぼ取引しない。
レポートは `p0_report.json`、エクイティカーブ図も出力される。

### P1〜P3: 実データ（ローカルPCで実行）

```bash
# 1. 先物データ取得（kline + funding rate + オーダーフロー集計）
python scripts/fetch_futures.py --symbols BTCUSDT ETHUSDT XRPUSDT BNBUSDT SUIUSDT PAXGUSDT --days 180 --to csv

# （Trade PlatformのPostgreSQLと同居させる場合）
pip install "psycopg[binary]"
set PLATFORM_DB_URL=postgresql://postgres:postgres@localhost:5432/trade
python scripts/fetch_futures.py --symbols BTCUSDT --days 180 --to csv postgres

# 2. P1: シグナルICゲート + P2: RL学習（ゲート不合格なら自動停止）
python scripts/train_portfolio.py --phase train --source csv --data ./data --timesteps 2000000

# 3. P3: ウォークフォワード検証（複数シード・コスト2倍感度込み）
python scripts/train_portfolio.py --phase wf --source csv --data ./data --timesteps 500000
```

ゲート1はOOSランクIC≥0.02。不合格なら特徴量・データを変えるまでRL学習に進まない
（`--skip-gate`で強制続行可）。評価は常にベースライン4種
（フラット/等ウェイトB&H/ボラ逆数/クロスモメンタムルール）と並記される。

### ダッシュボード・Platform連携

- ダッシュボードからは学習設定で `mode: "portfolio"` を指定して起動可能
- **`GET /api/signal/latest`**: 学習済みモデルの最新推奨ウェイトを返す。
  Trade Platform 側はこのAPIをポーリングするだけでBots画面に統合できる

```json
{"weights": {"BTCUSDT": 0.31, "ETHUSDT": -0.12, ...}, "net_exposure": 0.4, ...}
```

## 使い方（v1: 執行エージェント）

### データ取得
```bash
# 上位30通貨のデータを取得（時価総額順・上場日から全期間）
# --sort marketcap で時価総額順（ステーブルコイン除外）、デフォルトは volume
# デフォルトでは 1m 足のみ取得します（学習時に自動で上位足を作成するため推奨）
# 全ての時間軸ファイルを物理的に保存したい場合は --multi を追加してください
python scripts/fetch_binance.py --top 30 --sort marketcap --all --output ./data --clean --multi
```

### データ整理（クリーンアップ）
指定した通貨リストに含まれない古いデータを削除したい場合は `--clean` オプションを使用します。
例えば、現在の上位30通貨**以外**のデータを削除するには：

```bash
python scripts/fetch_binance.py --top 30 --clean --output ./data --all --sort marketcap
```

全データを削除したい場合は、`data` ディレクトリを直接削除してください。

### 学習
```bash
# 複数通貨・多時間軸で学習
python scripts/train.py --data ./data --top 10 --multi-tf --timesteps 500000

# 単一通貨で学習
python scripts/train.py --data ./data --symbol BTCUSDT --multi-tf
```

### 評価
```bash
python scripts/evaluate.py --model ./output/final_model.zip --episodes 20
```

### バックテスト
```bash
python scripts/backtest.py --model ./output/final_model.zip --data ./data --symbol BTCUSDT --episodes 10
```

### 進化学習（PBT-MAP-Elites）
```bash
python scripts/run_evolution.py --generations 10 --population 25 --steps-per-gen 10000 --data-dir data
```

### ダッシュボード

FastAPIサーバー + React UIで学習の開始/停止・メトリクス可視化・モデル管理・バックテストが行えます。

```bash
# ターミナル1（リポジトリルートから）
python scripts/run_server.py     # http://localhost:8001

# ターミナル2
cd frontend
npm install
npm run dev                      # http://localhost:5173
```

APIのURLを変える場合は `frontend/.env.example` を `.env` にコピーして編集してください。

主なAPI:
- `POST /api/training/start` / `POST /api/training/stop` / `GET /api/training/status`
- `GET /api/models` / `POST /api/backtest` （`{model_id, symbol, n_episodes}`）
- `WS /ws/metrics` （学習メトリクス・取引データのリアルタイム配信）

### Pythonから直接使用
```python
import pandas as pd
from mars_lite.env import MarsLiteEnv, MarsLiteMultiTFEnv, CrossSymbolEnv
from mars_lite.data import load_multi_symbol_data, MarsLiteConfig

# データ読み込み
config = MarsLiteConfig()
data_map = load_multi_symbol_data("./data", symbols=["BTCUSDT", "ETHUSDT"], config=config)

# 環境構築（手動）
envs = {}
for sym, (base, higher) in data_map.items():
    envs[sym] = MarsLiteMultiTFEnv(
        data_1m=base, 
        higher_tf_data=higher, 
        **create_env_kwargs(config)
    )

env = CrossSymbolEnv(envs)

# Gymnasium標準インターフェース
obs, info = env.reset()
action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
```

## ディレクトリ構成

```
mars_lite/
├── mars_lite/
│   ├── data/          # データ前処理
│   ├── env/           # Gymnasium環境
│   ├── learning/      # PPO/Population管理
│   ├── evolution/     # PBT/MAP-Elites
│   └── utils/         # 設定/評価指標
├── tests/             # ユニットテスト
├── scripts/           # 学習/評価スクリプト
└── requirements.txt
```

## テスト実行

```bash
python -m pytest tests/ -v
```

## 主要パラメータ

| パラメータ | 説明 | デフォルト |
|-----------|------|-----------|
| `y_impact` | インパクト係数 | 0.5 |
| `lambda_risk` | 在庫リスク係数 | 0.001 |
| `initial_inventory` | 初期在庫 | 1000 |
| `max_steps` | 最大ステップ数 | 1440 |

## ライセンス

MIT
