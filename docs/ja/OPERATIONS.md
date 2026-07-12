# 運用

[English](../OPERATIONS.md) | **日本語**

[`PRODUCTION_READINESS.md`](PRODUCTION_READINESS.md)が完了するまで、Productionは**NO-GO**です。

## Control Plane実行

release riskの例をコピーし、`symbol_liquidity_caps`が実際のBundle symbolと完全一致するよう編集します。

```bash
cp config/release-risk.example.json config/release-risk.local.json
```

適格なrelease pipelineを実行します。

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N \
  --risk-config config/release-risk.local.json
```

release可能な実行には、空でないsealed holdout、P0・Walk-Forward・Gate 2・設定済みsignificance gateの合格、完全なrelease risk policyが必要です。

`--force`、`--skip-p0`、`--skip-wf`、`--skip-gate`のいずれかを使うと研究専用実行になります。Reportは生成できますが、コードが候補構築とRegistry登録を禁止します。意図的な研究実行には`--no-register`を使用します。`--skip-pbt`は記録されますが、それだけではrelease不適格になりません。

適格な出力候補は`output/.../candidates/<version>`に配置され、不変Registryへ登録されます。学習処理はactivationを行いません。

## 証拠とデプロイ

CanaryおよびProductionデプロイには、次を含む`deployment-evidence` Artifactを生成した、成功済みのGitHub Actions runが必要です。

- `candidate.json`
- deployment gateが要求するmodel／reportファイル
- 正確な不変ServingBundleを含む`serving_candidate/`
- Shadow、drift、incident、Productionの場合はCanary report

Canary／Productionの各GitHub Environmentに次を設定します。

```text
TRADE_RL_REGISTRY_DIR       stage Servingと共有する絶対persistent path
TRADE_RL_SERVING_READY_URL  /readyを含むstage readiness URL
```

Deployment jobには、`trade-rl-deploy` labelを持ち、stage Serving processと同じpersistent Registry storageへアクセスできるself-hosted runnerが必要です。一時的なGitHub-hosted runnerをdeployment先として使用してはいけません。

Deployment workflowは、source runの成功、Git SHA、model version、report hash、Bundle digest、release eligibility、risk policy、証拠lineage、incident、approval ticket、Environment承認を検証します。その後、正確な`serving_candidate/`を登録して原子的にactivationします。

Activationだけではdeployment成功ではありません。Workflowは`TRADE_RL_SERVING_READY_URL`をpollし、`/ready`が承認済みmodel version、Bundle digest、release Git SHAを返した場合にだけ成功します。`degraded`は新しく承認されたidentityを報告する場合にだけ許容します。以前のidentityまたは到達不能endpointはworkflow失敗です。

## Serving開始

必要なsecretおよび設定:

```text
TRADE_RL_SERVING_TOKEN
TRADE_RL_RELEASE_GIT_SHA
TRADE_RL_REGISTRY_DIR
TRADE_RL_AUDIT_DB
TRADE_RL_DATA_DIR
TRADE_RL_ALLOWED_ORIGINS
```

`TRADE_RL_RELEASE_GIT_SHA`は、稼働中Serving releaseの正確な40桁Git SHAでなければなりません。Productionのstrict servingは、別revisionで構築されたBundleを拒否します。

起動:

```bash
export TRADE_RL_RELEASE_GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/run_server.py
```

確認:

```bash
curl http://127.0.0.1:8001/health
curl http://127.0.0.1:8001/ready
```

`/ready`は`status`、`active_version`、`bundle_digest`、`release_git_sha`、任意のfailure reasonを返します。最後の健全なin-memory Bundleが継続してServingしている間、`degraded`になる場合があります。`unavailable`は、実行可能なsignalを返せないことを意味します。

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

`/ready`が期待するversion、Bundle digest、release Git SHAを報告していることを確認してください。Rollback先は、現在稼働中のrelease SHAと互換でなければなりません。互換でない場合、strict bindingが拒否し、直前のin-memory Bundleを維持します。運用者がmodelファイルをcopyまたはrenameしてはいけません。

Post-activation handshake失敗後の自動rollbackは意図的に無効です。Identity不一致は、Registry状態、Serving readiness、audit証拠に基づく明示的な運用判断が必要なincidentです。

## インシデント対応

1. Trade Platformで新規live riskをブロックする。
2. request ID、Bundle digest、active version、稼働中release SHA、market snapshot ID、Registry状態、audit databaseを保存する。
3. Exposureがある場合、固有のidempotency keyを使用して、実際のplatform固有emergency adapterを実行する。
4. Flatten成功を報告する前に、注文取消、reconciliation、reduce-only closure、残存exposureがゼロであることの検証を必須とする。
5. 登録済みでdigestが有効かつcode互換なknown-good Bundleにのみrollbackする。
6. 根本原因、証拠、復旧検証が文書化されるまでProductionを無効のまま維持する。

Repository codeには、架空のexchange adapterや運用連絡先を意図的に含めていません。

## GameDay最低要件

GO承認前に、testnetで次を証明する演習を実施してください。

- stale dataでは実行可能なsignalを返さない
- 無効またはreplayされたrequestはfail-closedになる
- 破損またはGit SHA不一致の候補が健全なBundleを置き換えない
- activationとrollbackにより、Servingされるversion／digestが期待どおり変化する
- live served identityが承認済みidentityと一致しない場合、deploymentが失敗する
- emergency cancellation／flattenがidempotentで、reconciliation済みである
- audit recordからeventを再構築できる

Command、timestamp、identity、output、reviewer承認をreadiness checklistへ添付してください。
