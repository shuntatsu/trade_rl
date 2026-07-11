# アーキテクチャ上の意思決定

[English](../DECISIONS.md) | **日本語**

## ADR-001: Control PlaneとServing Planeを分離する

**決定:** 学習、証拠生成、Registry変更、activation、rollbackはオフラインで実行する。オンラインServingは認証付き・読み取り専用とする。

**理由:** 権限付き管理機能とライブsignal配信を統合すると、不要な信頼境界と障害境界が生じるため。

## ADR-002: 不変Bundle Registryを1つだけ使用する

**決定:** `mars_lite.serving.registry.ModelRegistry`をmodel lifecycleの唯一の権威とする。登録済みversion directoryは不変で、`active.json`を唯一のactive pointerとする。

**理由:** 複数Registryと固定file nameにより、昇格identityとServing identityが乖離していたため。

## ADR-003: 登録とactivationを分離する

**決定:** 学習処理は候補を構築・登録できるが、activationしてはならない。Activationにはdeployment証拠とEnvironment承認が必要である。

**理由:** 学習processの成功はdeployment許可ではないため。

## ADR-004: 推論依存関係をすべてBundle化する

**決定:** Model、symbol、feature schema、preprocessing、observation contract、post-processing、risk設定、Git SHA、metrics identity、digestを1つのServingBundleに含める。

**理由:** 不完全なmetadataにより、train／serve間の分布とshapeが一致しない問題が発生したため。

## ADR-005: Policy推論前に実positionを使用する

**決定:** `predict()`より前に、認証済みの現在口座状態からpolicy observationを構築する。

**理由:** 推論後にのみprevious weightを適用する方式は、学習時observation contractに違反するため。

## ADR-006: 決定論的なonline progressを使用する

**決定:** Production互換modelはobservation progress modeとして`zero`を使用する。オンラインで同等状態を設計・検証するまでは、episode相対progressを研究専用とする。

**理由:** 任意の学習episode位置は、stateless online inferenceで再現できないため。

## ADR-007: Trade Platformを執行権威とする

**決定:** Servingはguardrailとpre-trade risk判定を計算し、Trade Platformが最終enforcementと注文執行を行う。

**理由:** Servingは権威あるexchange connectionやaccount storeを持たず、注文を実行したと主張してはいけないため。

## ADR-008: Serving状態を最小限にする

**決定:** 口座状態は認証済みrequestごとに受け取る。SQLiteにはaudit eventとreplay claimだけを保存する。

**理由:** Serving内部にportfolioの真実を複製すると、reconciliationとstale stateのriskが生じるため。

## ADR-009: Known-good serviceを維持しながらfail-closedにする

**決定:** 無効requestには実行可能signalを返さない。無効な新Bundleは健全なin-memory Bundleを置き換えず、readinessを`degraded`にする。

**理由:** 可用性を完全性より優先してはならないが、不正候補によって検証済み既存serviceまで不要に停止させるべきではないため。

## ADR-010: 正典ドキュメントを1系列にする

**決定:** `docs/ARCHITECTURE.md`を唯一のアーキテクチャ正典とする。研究結果は分離して保存し、Productionを許可できないものとする。

**理由:** 過去文書では、歴史的実験、将来構想、実行可能な現行動作が混在していたため。