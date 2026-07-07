# Trade RL Frontend — 現状と注意事項

このReact/Viteフロントエンドは、レガシー実行エージェントv1時代の
学習モニタリングUI（WebSocketでのリアルタイムメトリクス配信・
ブラウザからの学習開始/停止）としてつくられたもの。

## アーキテクチャ再設計（バックエンドの変更）に伴う影響

バックエンドはCLI一本化した設計に整理され、学習は
`scripts/train_portfolio.py` から実行する運用に統一された
（docs/ARCHITECTURE.md 参照）。これに伴い、サーバー側の
以下のエンドポイントは**削除済み**:

- `/ws/metrics`（学習メトリクスのWebSocket配信）
- `/api/training/*`（学習の開始・停止・状態取得・設定取得）
- `/api/metrics`, `/api/metrics/latest`

現在のサーバー（`mars_lite/server/signal_server.py`）が提供するのは:

- `GET /api/signal/latest` — ポートフォリオRLエージェントの最新推奨ウェイト（Trade Platform連携用）
- `GET /api/models`, `GET /api/models/{id}`, `DELETE /api/models/{id}` — 保存済みモデルの一覧・削除
- `GET /api/data/available` — 利用可能なデータセット
- `GET /health`

## このフロントエンドの現状

**このディレクトリはコード凍結状態**（意図的に未更新）。以下のコンポーネント・
フックは削除されたエンドポイントに依存しており、現状は接続エラー/404になる:

- `useWebSocket.ts`, `useTrainingControl.ts`
- `TrainingConfigPanel.tsx`, `TrainingChart.tsx`, `LossChart.tsx`,
  `DetailedLossChart.tsx`, `LogTerminal.tsx`, `TradingVisualizer.tsx`,
  `BacktestPanel.tsx`（`/api/backtest`も削除済み）

`ModelPanel.tsx` / `ModelSelect.tsx` / `StatsCard.tsx` は現行の
`/api/models`系エンドポイントとおおむね互換だが、`Dashboard.tsx` からの
配線は上記の削除済みフックに依存する形で書かれているため、そのままでは
動作しない。

新しいシグナル/モデル中心のダッシュボード（`/api/signal/latest`を
ポーリングするSignalPanel等）への作り直しは未着手（今後の作業）。
UIが必要な場合は、当面はAPIを直接叩く（`curl`/Postman等）か、
軽量な専用スクリプトで代替することを推奨する。
