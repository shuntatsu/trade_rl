# 運用

[English](../OPERATIONS.md) | **日本語**

[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)が完了するまで、Productionは**NO-GO**です。

## Control Plane実行

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N
```

P0、Walk-Forward、最終baseline、設定済みsignificance gateのいずれかが失敗した場合、明示的に文書化された研究専用`--force`実行でない限り、pipelineは停止しなければなりません。Forced runを昇格してはいけません。

出力候補は`output/.../candidates/<version>`に配置され、不変Registryへ登録されます。学習処理はactivationを行いません。

## 証拠とデプロイ

CanaryおよびProductionデプロイには、次を含む`deployment-evidence` Artifactを生成した、成功済みのGitHub Actions runが必要です。

- `candidate.json`
- deployment gateが要求するmodel／reportファイル
- 正確な不変ServingBundleを含む`serving_candidate/`
- Shadow、drift、incident、Productionの場合はCanary report

Deployment workflowは、source runの成功、Git SHA、model version、report hash、bundle digest、証拠lineage、incident、approval ticket、Environment承認を検証します。その後、stageごとの`TRADE_RL_REGISTRY_DIR`で指定されたRegistryへ、正確な`serving_candidate/`を登録し、原子的にactivationします。

## Serving開始

必要なsecretおよび設定:

```text
TRADE_RL_SERVING_TOKEN
TRADE_RL_REGISTRY_DIR
TRADE_RL_AUDIT_DB
TRADE_RL_DATA_DIR
TRADE_RL_ALLOWED_ORIGINS
```

起動:

```bash
uv run python scripts/run_server.py
```

確認:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/ready
```

最後の健全なin-memory Bundleが継続してServingしている間、`/ready`は`degraded`になる場合があります。`unavailable`は、実行可能なsignalを返せないことを意味します。

## Rollback

Registryを確認します。

```bash
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" list
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" show-active
```

Rollback:

```bash
uv run python scripts/manage_registry.py \
  --registry-dir "$TRADE_RL_REGISTRY_DIR" rollback \
  --target-version <known-good-version>
```

`/ready`が期待するversionとdigestを報告していることを確認してください。Serving Planeは通常の検証済みhot-swap pathを使用します。運用者がmodelファイルをcopyまたはrenameしてはいけません。

## インシデント対応

1. Trade Platformで新規live riskをブロックする。
2. request ID、bundle digest、active version、market snapshot ID、Registry状態、audit databaseを保存する。
3. Exposureがある場合、固有のidempotency keyを使用して、実際のplatform固有emergency adapterを実行する。
4. Flatten成功を報告する前に、注文取消、reconciliation、reduce-only closure、残存exposureがゼロであることの検証を必須とする。
5. 登録済みでdigestが有効なknown-good Bundleにのみrollbackする。
6. 根本原因、証拠、復旧検証が文書化されるまでProductionを無効のまま維持する。

Repository codeには、架空のexchange adapterや運用連絡先を意図的に含めていません。

## GameDay最低要件

GO承認前に、testnetで次を証明する演習を実施してください。

- stale dataでは実行可能なsignalを返さない
- 無効またはreplayされたrequestはfail-closedになる
- 破損した候補が健全なBundleを置き換えない
- activationとrollbackにより、Servingされるversion／digestが期待どおり変化する
- emergency cancellation／flattenがidempotentで、reconciliation済みである
- audit recordからeventを再構築できる

Command、timestamp、identity、output、reviewer承認をreadiness checklistへ添付してください。