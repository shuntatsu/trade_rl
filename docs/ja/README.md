# Trade RL 日本語ドキュメント

[English documentation](../ARCHITECTURE.md) | **日本語**

このディレクトリには、現在の利用者向け正典ドキュメントの日本語版を配置しています。

英語版が正式な正典です。日本語版との間に差異が見つかった場合は、コードとテスト、続いて英語版を優先し、日本語版を修正してください。

## 文書一覧

- [`ARCHITECTURE.md`](ARCHITECTURE.md) — システム境界、ServingBundle、Registry、オンライン推論フロー
- [`MODEL_LIFECYCLE.md`](MODEL_LIFECYCLE.md) — 学習から登録、証拠、activation、rollbackまで
- [`OPERATIONS.md`](OPERATIONS.md) — Control Plane実行、デプロイ、Serving起動、インシデント対応
- [`SECURITY.md`](SECURITY.md) — 認証、Artifact完全性、request完全性、fail-closed条件
- [`TESTING.md`](TESTING.md) — 必須CI、テスト層、テスト置換方針
- [`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md) — 本番GO／NO-GOの証拠付きチェックリスト
- [`DECISIONS.md`](DECISIONS.md) — 主要なArchitecture Decision Record
- [`RESEARCH_HISTORY.md`](RESEARCH_HISTORY.md) — 本番判断を変更しない研究履歴

ルートの日本語案内は[`../../README.ja.md`](../../README.ja.md)を参照してください。

## 状態

現在の本番判断は**NO-GO**です。コードやCIの成功だけでライブ取引は許可されません。