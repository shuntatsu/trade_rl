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

## 使い方

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
