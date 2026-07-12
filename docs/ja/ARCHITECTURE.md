# アーキテクチャ

[English](../ARCHITECTURE.md) | **日本語**

この文書は、現在のTrade RLアーキテクチャを説明する日本語版です。正式な正典は英語版[`../ARCHITECTURE.md`](../ARCHITECTURE.md)です。不一致が見つかった場合はコードとテストを優先し、続いて英語版と日本語版を修正してください。

## 状態

[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)が証拠付きで完了するまで、Productionは**NO-GO**です。

## 原則

1. 各Production責務に対して、権威ある実装を1つだけ持つ。
2. 無効なArtifact、状態、schema、認証情報、市場データ、release identityはfail-closedで拒否する。
3. 証拠は、1つの正確なBundle digestとGit commitに拘束する。
4. 学習・評価とServingは、Observationおよび意思決定契約を共有する。
5. オンラインServingは認証付き・読み取り専用とする。
6. Registryのversionは不変とし、activationは原子的なpointer更新で行う。
7. 口座状態は明示的に扱い、Trade Platformがrequestごとに提供する。
8. 研究結果やoverride実行だけではProduction動作を許可しない。
9. live Serving Planeが承認済みversion、Bundle digest、稼働中release Git SHAを返すまで、deployment完了としない。

## システム境界

### Control Plane

オフラインControl Planeは次を担当します。

- データ準備と品質ゲート
- 学習、PBT、Walk-Forward、sealed holdout評価、統計検査
- 解決済みpipeline状態からのrelease eligibility導出
- 完全なrelease risk policyの検証
- 完全な`ServingBundle`の構築
- デプロイ証拠の生成と検証
- 不変Registryへの登録
- CLIおよび信頼済みGitHub Actionsによるactivationとrollback

Control Planeはライブシグナルを配信しません。サポート対象interfaceは、CLI commandと信頼済みGitHub Actions workflowです。

### Serving Plane

オンラインServing Planeが公開するのは次だけです。

- `GET /health`
- `GET /ready`
- 認証付き`POST /api/signal/latest`

学習、モデル削除、Registry変更、昇格、rollbackは公開しません。

## ServingBundle

Bundleは、決定論的推論に必要な完全な単位です。

```text
serving_candidate/
  manifest.json
  model.zip | ensemble/
  metadata.json
  preprocessing.json
  risk.json
```

Bundleには次を含めます。

- model versionと学習時Git SHA
- override状態と必須gate状態を含む不変release eligibility
- sealed holdoutを使用したことの記録
- 順序付きsymbol一覧
- 順序付きの銘柄別feature名とglobal feature名
- feature正規化とzero-mask設定
- Observation schemaと次元
- Serving互換のprogress mode
- 推論用environment設定
- post-processing設定
- 完全なguardrailとpre-trade risk設定
- 評価summary
- 全ファイルのSHA-256とcanonical bundle digest

forced実行、必須gateのskip、sealed holdout記録の欠落、必須gate失敗、必須risk limit欠落のいずれかがあるrelease Bundleは拒否します。Liquidity capは、順序付きBundle symbolを過不足なくカバーしなければなりません。

Digest、ファイル集合、次元、symbol順序、feature順序、schema、release eligibility、risk policyのいずれかが一致しなければBundleを拒否します。

## Registry

唯一のRegistry実装は`mars_lite.serving.registry.ModelRegistry`です。

```text
registry/
  versions/<version>/...   不変Bundle
  active.json              原子的なactive identity
  activation-history.jsonl
```

登録処理は、候補を不変versionディレクトリへコピーして再検証します。登録だけではactivationしません。Activationは登録済みBundleを検証し、`active.json`を原子的に置換します。登録またはactivationが失敗した場合、以前のactive versionを維持します。

Deployment runnerは、stage Serving Planeと同じpersistent Registry storageへアクセスできなければなりません。GitHub-hosted runnerの一時filesystemはdeployment先ではありません。

## 学習・昇格フロー

```text
データ構築
  -> 品質・leak検査
  -> sealed development／holdout分割
  -> P0
  -> development data上の任意PBT
  -> 必須の複数fold Walk-Forwardとcost sensitivity
  -> development dataで最終学習
  -> sealed holdoutでGate 2
  -> 不変release eligibilityを導出
  -> 明示的release risk policyを検証
  -> 完全なServingBundle候補
  -> 不変登録
  -> Bundle digestに拘束されたShadow／Canary証拠
  -> deployment gateとEnvironment承認
  -> persistent stage Registryで原子的activation
  -> live /ready identity検証
```

適格な学習実行が成功しても、候補を登録するだけでactivationは行いません。`--force`、`--skip-p0`、`--skip-wf`、`--skip-gate`のいずれかを使うと、候補構築・登録の対象外になります。`--skip-pbt`は記録されますが、PBTは安全gateではなく最適化手順であるため、それだけではrelease不適格になりません。

## Deployment identity handshake

CanaryおよびProductionでは、deployment workflowが次を実施します。

1. 正確な不変ServingBundleに対して証拠を検証する。
2. persistent stage Registryへアクセスできるself-hosted deployment runnerを使う。
3. 承認済みversionを登録して原子的にactivationする。
4. 設定済みstage `/ready` endpointをpollする。
5. `active_version`、`bundle_digest`、`release_git_sha`が承認済みidentityと一致することを要求する。

`degraded`は、新しく承認されたidentityを報告している場合にだけ許容します。以前のBundleをServingしている`degraded` responseはdeployment失敗です。Endpointが到達不能またはidentity不一致の場合も失敗し、運用者による明示的なrollback判断を必要とします。

## オンライン推論フロー

Trade Platformは次を送信します。

- request IDとmarket snapshot identity
- Bundleのsymbol順序に対応する現在weight
- portfolio value、day-start value、peak value
- 連続損失数とturnover状態
- pending order
- model schemaが要求する場合のdisagreementおよびrisk-state値

Serving Planeは次を実行します。

```text
認証
  -> request IDをclaimし、replayを拒否
  -> キャッシュ済みの不変feature snapshotを取得
  -> symbolとfeature schemaを検証
  -> preprocessingを復元
  -> 実際の現在positionでbuild_observation
  -> policy.predict
  -> DecisionPipeline
  -> 実注文turnoverを使うstateful guardrails
  -> pending-order-aware PreTradeRiskVerifier
  -> responseと構造化audit event
```

Trade Platformが最終的な執行・risk enforcement境界です。Responseが有効で、risk判定がapprovedでない限り、注文を実行してはいけません。

## 共有契約

- `mars_lite.env.observation.build_observation` — 共通Observation Builder
- `mars_lite.trading.pipeline.DecisionPipeline` — actionからtargetまでの共通path
- `mars_lite.trading.guardrails.evaluate_guardrails` — 実口座状態の評価
- `mars_lite.trading.pre_trade_risk.PreTradeRiskVerifier` — delta、pending order、liquidity、restriction、reduce-onlyの評価
- `mars_lite.pipeline.release_eligibility.derive_release_eligibility` — 唯一のrelease分類path
- `mars_lite.pipeline.release_risk.load_release_risk_policy` — release risk policy fileの検証
- `mars_lite.serving.runtime.ServingRuntime` — キャッシュload、安全なhot-swap、Git SHA拘束、推論orchestration、readinessの所有者

## 可用性とhot-swap

Production servingは`TRADE_RL_RELEASE_GIT_SHA`を受け取り、strict release bindingを有効にして起動します。Active BundleのGit SHAは、稼働中release SHAと一致しなければなりません。

Servingは、現在load済みBundleを維持したまま新しいactive Bundleをloadします。Digest、release eligibility、risk policy、schema、preprocessing、Git SHA拘束、model load、readiness checkがすべて成功した後にだけ、新Bundleを公開します。不正な新Bundleは、既存の健全なin-memory Bundleを置き換えず、readinessを`degraded`にします。健全なBundleが1つもない場合、signal routeは`503`を返し、実行可能なweightを返しません。

## 信頼境界

Control processとServing processは別のcredentialを使用します。ServingはBearer token、明示的origin allowlist、既定のlocal bind、audit logging、request replay防止を使用します。RegistryへのwriteにはControl Plane identityが必要です。Deploymentには専用self-hosted runner labelとstage scoped GitHub Environment variableを使用します。Secretはdeployment secret managementから供給し、repositoryへ保存しません。

## Package map

- `mars_lite/data`, `mars_lite/features` — データ・feature構築
- `mars_lite/env`, `mars_lite/learning` — RL environmentと学習
- `mars_lite/eval` — 評価とreplay simulation
- `mars_lite/pipeline` — オフラインorchestration、release eligibility、risk validation
- `mars_lite/serving` — Bundle、Registry、契約、runtime、audit、feature snapshot
- `mars_lite/server` — 読み取り専用Serving HTTP境界とdeployment gate
- `mars_lite/trading` — execution cost、decision processing、guardrail、pre-trade risk

## 明示的な非保証事項

CI合格は、テスト対象の契約が成立していることを示すだけです。収益性やProduction readinessを証明しません。合成結果、過去backtest、単発実験は、ライブ取引の許可ではありません。
