# Trade RL Interactive Guide

This guide is non-authoritative.
Technical truth lives in docs/architecture/* and docs/research/current-status.md.
Human-facing content lives in guide/content/topics/*.json.

このGuideは、人間がTrade RLの全体像・データフロー・責務境界・実験手順・現在の研究状態を視覚的かつ対話的に理解するための説明層です。

技術仕様や研究状態の正本を置き換えません。実装と説明が食い違う場合は、`docs/architecture/`、`docs/research/current-status.md`、現行source、contract testsを正本として確認します。

## 開発

Node.js 24 LTSを使用します。React / Vite / Tailwindの依存関係とビルドはこの`guide/` workspace内に閉じ込めます。

説明内容は`content/topics/*.json`を更新し、UI componentへ研究固有の説明を直接埋め込まない方針です。
