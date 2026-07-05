# Binance Futures 学習データ取得手順

強化学習モデルの学習に必要な過去の相場データ（kline, metrics, funding rate 等）を `fetch_futures.py` を用いて取得・保存する際の手順とコマンドのまとめです。

## 1. データベースの起動

データはローカルの PostgreSQL コンテナに保存します。まずは Docker で DB を起動します。

```bash
# プロジェクトルートで実行
docker-compose up -d db
```

これにより、ポート `5433` で Postgres コンテナ (`trade_rl_db`) が立ち上がります。

## 2. 仮想環境の構築と依存パッケージのインストール

`uv` を使って、指定したバージョンの Python（ここでは 3.11）で仮想環境を構築し、パッケージをインストールします。

```bash
# 仮想環境の作成
uv venv --python 3.11

# 仮想環境の有効化 (Windows PowerShell の場合)
.venv\Scripts\activate

# 依存パッケージとローカルモジュールのインストール
uv pip install -e . -r requirements.txt
```

## 3. データ取得コマンドの実行

仮想環境が有効な状態で、以下のコマンドを実行して Binance Vision から長期間のデータをダウンロードし、DB へ格納します。

### 今回実行したコマンド（3年分・10通貨の取得）

```bash
.venv\Scripts\python scripts/fetch_futures.py `
  --days 1095 `
  --to postgres `
  --skip-orderflow `
  --dsn postgresql://trade_rl:trade_rl@localhost:5433/trade_rl
```

**パラメータの解説:**
- `--days 1095`: 過去1095日（約3年）分のデータを取得します。
- `--to postgres`: 取得したデータを CSV ではなく Postgres へ保存します。
- `--skip-orderflow`: 非常に時間がかかる `aggTrades` (オーダーフロー) の REST API 経由での取得をスキップします（長期間のデータ取得では必須です）。
- `--dsn ...`: Postgres コンテナの接続先 URL です。

### 注意事項
- 約10通貨 × 3年分の処理となるため、数時間以上の時間がかかります。
- もし途中で停止して再開した場合でも、基本的に既存のデータに追記（または重複スキップ/上書き）される設計になっています。
