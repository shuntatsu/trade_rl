# Production Readiness

[English](../PRODUCTION_READINESS.md) | **日本語**

現在の判断: **NO-GO**。

チェック済み項目には、主張ではなく添付証拠が必要です。Code ownerはrepository内で検証可能な項目を確認できます。運用、法務、セキュリティ、取引所関連の項目は、責任を持つownerによる確認が必要です。

## CodeとCI

- [ ] 正確なrelease headでRuff lintに合格している。
- [ ] 正確なrelease headでRuff format checkに合格している。
- [ ] 正確なrelease headでmypyに合格している。
- [ ] 正確なrelease headで完全なpytest suiteに合格している。
- [ ] Coverageが70%以上である。
- [ ] Architecture reviewのCriticalおよびImportant findingが解消されるか、責任ownerによって明示的に受容されている。

## Control Plane release eligibility

- [ ] 承認済み実行で`--force`、`--skip-p0`、`--skip-wf`、`--skip-gate`を使用していない。
- [ ] 空でないsealed holdoutが、PBT、Walk-Forward選択、feature選択、最終学習から分離されている。
- [ ] Gate 2が最終modelをそのsealed holdout上で評価している。
- [ ] P0、Walk-Forward、Gate 2、必要なsignificance gateに合格している。
- [ ] Bundleの`release_eligibility` recordが承認済み実行証拠と一致している。
- [ ] 研究専用実行から登録可能な候補が生成されないことを実証している。

## Artifactとmodel identity

- [ ] 承認されたControl Plane runから、完全なServingBundleが1つ生成されている。
- [ ] Bundleのmodel version、Git SHA、file digest、canonical digestが記録されている。
- [ ] Bundle検証が、適格なrelease metadataと完全なrisk policyを確認している。
- [ ] ShadowおよびCanary証拠が、正確に同じBundle identityを参照している。
- [ ] Deployment gateのsource run制限とrelease branch制限が設定されている。
- [ ] 稼働中Serving release Git SHAがBundle Git SHAと一致している。
- [ ] Activation済みRegistry identityが、Servingの返すversion／digestと一致する。
- [ ] Post-activation workflowがlive `/ready`を通じてversion、digest、release Git SHAを検証している。
- [ ] code互換な登録済みknown-good versionへのrollbackが実証されている。

## Serving Plane

- [ ] Servingがhealth、readiness、認証付きsignal routeだけを公開している。
- [ ] Machine-to-machine tokenとrotation手順が設定されている。
- [ ] Network bind、proxy、TLS、origin allowlistが設定されている。
- [ ] `TRADE_RL_RELEASE_GIT_SHA`が不変deployed releaseから注入されている。
- [ ] 現在positionと口座状態がrequestごとに提供される。
- [ ] Request IDとmarket snapshot identityが固有で、auditされている。
- [ ] 破損またはGit SHA不一致activation時に、以前の健全なBundleを維持する。
- [ ] 健全なBundleがない場合、実行可能なweightを含めず`503`を返す。

## Riskと執行

- [ ] Release risk policyがleverage、single-symbol weight、net exposure、worst-case notional、minimum order notionalの有限limitを含む。
- [ ] Liquidity capが順序付きBundle symbolを過不足なくカバーし、forbidden symbolが明示的に記録されている。
- [ ] Trade Platformが注文前に、返されたpre-trade risk判定を強制する。
- [ ] Pending order、symbol restriction、liquidity、reduce-only、exposure limitをend-to-endで検証している。
- [ ] 実取引所／platform用`EmergencyExecutionAdapter`が実装され、review済みである。
- [ ] Emergency cancellation、reconciliation、reduce-only closure、residual position検査がtestnetで合格している。
- [ ] Idempotency keyがemergency executionの重複を防止する。

## Deployment governance

- [ ] `shadow`、`canary`、`production` GitHub Environmentが存在する。
- [ ] Productionに必要な独立reviewerが設定されている。
- [ ] `trade-rl-deploy` labelを持つself-hosted runnerが隔離・管理されている。
- [ ] `TRADE_RL_REGISTRY_DIR`がServingと共有されるstage適切な絶対persistent pathである。
- [ ] `TRADE_RL_SERVING_READY_URL`が正しいstage `/ready` endpointを指している。
- [ ] 証拠producer identityが信頼され、branch制限されている。
- [ ] 意図的にidentityを不一致にした場合、deploymentが失敗することを実証している。
- [ ] Secretがrepository外に保存され、rotationがテストされている。

## 運用とコンプライアンス

- [ ] On-call、risk、security、complianceの連絡先が実在し、テスト済みである。
- [ ] Incident severity、escalation、communication手順が承認されている。
- [ ] 適用jurisdictionとretention periodが、資格を持つownerによって決定されている。
- [ ] Audit storage、access、backup、deletion policyが承認されている。
- [ ] Testnet GameDay証拠が添付されている。
- [ ] 最終運用ownerがGO承認へ署名している。

## GOルール

適用されるすべての項目が証拠付きでチェックされ、未解決のCriticalまたはImportant findingが存在しない場合に限り、ProductionをGOへ変更できます。それまでは、すべての文書とinterfaceで**NO-GO**を維持しなければなりません。

## ローカル検証証拠

- [ ] 正確なrelease headでP0が候補タイミングを記録し、`--p0-days`を合成データ期間だけに使用している。
- [ ] Content-addressed snapshot testで、選択済み値のmutationによりidentityが変化する。
- [ ] `1h`、`4h`、`1d`のcompleted bar鮮度testが合格している。
- [ ] `uv run python scripts/run_local_gameday.py`が7つのローカルscenarioすべてに合格している。
- [ ] 旧dashboard起動に`TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1`または明示的なdevelopment opt-inが必要である。
- [ ] filesystem Registryがsingle-nodeのままであり、ローカル証拠はmulti-nodeやtestnet証拠ではないとreviewerが確認している。

