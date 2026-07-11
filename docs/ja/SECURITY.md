# セキュリティ

[English](../SECURITY.md) | **日本語**

## 信頼境界

Control PlaneとServing Planeは、別process・別identityとして動作します。

- Control Plane: 権限付き学習、証拠生成、Registry write、activation、rollback
- Serving Plane: 認証付き・読み取り専用のsignal配信
- Trade Platform: 権威ある口座状態、注文執行、最終pre-trade enforcement

## 認証とnetwork exposure

`POST /api/signal/latest`には`Authorization: Bearer <token>`が必要です。期待tokenは`TRADE_RL_SERVING_TOKEN`から供給され、constant-timeで比較されます。Credentialがない場合は`401`、不正なcredentialは`403`を返します。

Servingは既定で`127.0.0.1`へbindします。CORSは明示的allowlistを使用します。Credential付きwildcard originは有効化しません。

## Artifact完全性

すべてのServingBundleは、全ファイルのSHA-256 digestとcanonical bundle digestを持ちます。登録、activation、起動、hot-swapの各境界でBundleを再検証します。Version directoryは不変です。

証拠は、model version、Git SHA、bundle identity、source run、evaluation lineageに拘束されなければなりません。候補側が指定するthresholdは信頼しません。

## Request完全性

Trade Platformは固有のrequest IDとmarket snapshot identityを提供します。SQLiteはclaim済みrequest IDと不変audit eventを保存します。同一request IDの再利用は拒否し、異なるpayloadでの再利用はintegrity violationとして扱います。

SQLiteは口座状態のsource of truthではありません。現在positionとrisk stateは、認証済みTrade Platformがrequestごとに送信しなければなりません。

## Fail-closed条件

次の場合、実行可能なweightを返しません。

- Bearer credentialが無効
- active Bundleが存在しない、または不健全
- digest、schema、symbol順序、feature順序、observation次元の不一致
- stale、NaN、all-zero、または不正形式の市場データ
- 無効な口座値またはpending order
- request replay
- guardrailによるflatten／rejection
- pre-trade risk判定の失敗

Bundle refreshに失敗した場合、以前の健全なin-memory Bundleを維持し、readinessを`degraded`にします。

## Secret

Secret、live endpoint、private key、API key、運用連絡先をcommitしてはいけません。Deployment secret managementとstageごとのGitHub Environmentを使用してください。

## テストで扱う脅威

- path traversalとmanifest改ざん
- 別modelからの証拠流用
- non-finiteおよび範囲外metric
- 無効なfeature maskとschema次元
- replayされたrequest ID
- 未認証signal request
- destructive routeの公開
- 原子的activation失敗と破損hot-swap候補

## 外部セキュリティ対応

Production承認には、deployment固有のnetwork policy、secret rotation policy、machine identity、reviewer policy、audit retention決定、incident連絡先、GameDay証拠が引き続き必要です。