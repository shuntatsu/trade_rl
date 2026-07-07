# Trade RL Frontend

`mars_lite/server/metrics_server.py`（`scripts/run_server.py` から起動）に
接続するダッシュボードUI。学習起動・メトリクス監視（`/ws/metrics`）・
モデル管理・バックテスト（`/api/backtest`）・`/api/signal/latest` の
確認ができる。詳細はリポジトリルートの README.md「ダッシュボード」節を参照。

`mars_lite/server/signal_server.py` は `/api/signal/latest` とモデル管理のみに
絞った軽量版サーバー（本番シグナル配信専用の候補、現状は未配線）。
このフロントエンドは metrics_server.py 側のフルAPIを前提にしている。

## セットアップ

```bash
npm install
npm run dev
```
