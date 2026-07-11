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
- [ ] Architecture reviewのCriticalおよびImportant findingが解消されている。

## Artifactとmodel identity

- [ ] 承認されたControl Plane runから、完全なServingBundleが1つ生成されている。
- [ ] Bundleのmodel version、Git SHA、file digest、canonical digestが記録されている。
- [ ] ShadowおよびCanary証拠が、正確に同じbundle identityを参照している。
- [ ] Deployment gateのsource run制限とrelease branch制限が設定されている。
- [ ] Activation済みRegistry identityが、Servingの返すversion／digestと一致する。
- [ ] 登録済みknown-good versionへのrollbackが実証されている。

## Serving Plane

- [ ] Servingがhealth、readiness、認証付きsignal routeだけを公開している。
- [ ] Machine-to-machine tokenとrotation手順が設定されている。
- [ ] Network bind、proxy、TLS、origin allowlistが設定されている。
- [ ] 現在positionと口座状態がrequestごとに提供される。
- [ ] Request IDとmarket snapshot identityが固有で、auditされている。
- [ ] 破損activation時に、以前の健全なBundleを維持する。
- [ ] 健全なBundleがない場合、実行可能なweightを含めず`503`を返す。

## Riskと執行

- [ ] Trade Platformが注文前に、返されたpre-trade risk判定を強制する。
- [ ] Pending order、symbol restriction、liquidity、reduce-only、exposure limitをend-to-endで検証している。
- [ ] 実取引所／platform用`EmergencyExecutionAdapter`が実装され、review済みである。
- [ ] Emergency cancellation、reconciliation、reduce-only closure、residual position検査がtestnetで合格している。
- [ ] Idempotency keyがemergency executionの重複を防止する。

## Deployment governance

- [ ] `shadow`、`canary`、`production` GitHub Environmentが存在する。
- [ ] Productionに必要な独立reviewerが設定されている。
- [ ] `TRADE_RL_REGISTRY_DIR`がstageに適したpersistent storageを指している。
- [ ] 証拠producer identityが信頼され、branch制限されている。
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