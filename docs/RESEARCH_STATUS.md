# Research Status

## Current status

```text
RepositoryIntegrity: VERIFIED_BY_MAIN_CI
ResearchWorkflows: AVAILABLE
UniversalCausalAlphaTeacherSoftware: IMPLEMENTED_AND_CI_VERIFIED
UniversalCausalAlphaTeacherEmpiricalAdmission: NOT_COMPLETED
UniversalFullResearchEmpiricalEvaluation: NOT_COMPLETED
StageAZeroShotSoftware: IMPLEMENTED_AND_CI_VERIFIED
StageAEmpiricalEvaluation: NOT_COMPLETED
StageBSpotFuturesGeneralization: NOT_IMPLEMENTED
SpotLongBook: FUTURE_LONG_ONLY_ROLE
USDSMShortBook: FUTURE_SHORT_ONLY_ROLE
HierarchicalSequenceV2: IMPLEMENTED_AND_CPU_VERIFIED
StatefulOHLCExecution: AVAILABLE_WITH_OHLCV_LIMITATIONS
TradeRLStudio: AVAILABLE_FOR_DIAGNOSTIC_REPLAY_WITH_EPISODE_ISOLATION
AttestedPaperServing: AVAILABLE_FOR_ELIGIBLE_SELECTED_FINAL_BUNDLES
DirectExchangeRouting: NOT_IMPLEMENTED
EmpiricalProductionGate: NO-GO
ProfitabilityClaim: NONE
```

この表は能力境界です。実装済み、CI検証済み、研究上有効、収益性あり、Production認可済みは別の状態です。

Universal causal alpha teacherのsoftware pathは実装・CI検証済みです。ただし、維持対象実データでのteacher admission、複数Seed/Foldのfull-research comparison、zero-shot、sealed evaluationはまだ別のempirical evidenceです。Software completionをteacher qualityや収益性の証明として扱いません。

Stage Aのソフトウェア契約とCLIは実装・CI検証済みですが、維持対象の実データ、複数Seed・Fold、対象GPUによる実証評価は完了していません。Stage BのSpotとUSDⓈ-M先物を横断する一般化は未実装です。

## 現在のModel契約

維持対象の系列Modelは`hierarchical_sequence_v2`です。

- Clock別Causal TCN
- Gated Cross-Timeframe Attention
- Maintained one-symbol pathでは`single_symbol_bypass_v1`
- Historical multi-symbol pathだけがGated Cross-Asset Attentionを使用
- AvailabilityとStalenessの明示入力
- 組み立て済みModelから生成するArchitecture identity
- BC、PPO、CostCriticPPO、LagrangianPPOで同一構造
- Checkpoint、構造化Export、ServingでのDigest照合

旧Encoder Booleanと`training_run_config_v1`は維持対象ではありません。

## Universal causal teacher status

Canonical Universal U6 teacherは`causal_alpha_ridge`です。OracleとTrend teacherは診断・互換経路として保持します。

Software contractは次を実装済みです。

- train-only pooled deterministic ridgeによる24h/72h causal alpha prediction
- prefix-only scalingとlabel cutoff
- 各train symbolのlatest complete 720h episodeをteacher holdoutとして予約
- holdoutをcandidate selectionへ使わない分離
- fit/prediction cacheによる重複計算の再利用
- selection artifactをholdout replayより前に永続化
- per-symbol holdoutのexactly-once replayとshared teacher admission
- U5 architecture / U6 algorithm間で同じteacher packageを共有
- teacher admission failure時にBC、critic warm start、PPOへ進まないfail-closed順序
- teacher build progressと経済telemetryのread-only monitoring

詳細な現行契約は[UNIVERSAL_TRAINING.md](UNIVERSAL_TRAINING.md)を参照してください。

これらはsoftware contractです。Teacher holdoutの経済成績が実データで合格した、最終Policyがbaselineを上回った、unseen symbolへ一般化した、という意味ではありません。

## Software verification

Main CIで確認する項目:

- Ruff、Formatting、MyPy
- Import LinterとDead-code検査
- 全pytest、Branch coverage、Critical coverage
- Windows/Linux互換
- Training image buildとNon-root runtime
- Sequence projection反復安定性
- PostgreSQL catalog
- CPU training capability audit

CUDA実機Evidenceは、指定Labelを持つSelf-hosted runnerが接続されている場合だけ取得できます。Runner未接続による`queued`をCode成功やGPU成功として扱いません。

## Empirical status

Pipeline完走は収益性を意味しません。過去の維持対象比較では、RL候補がBaselineを一貫して上回らず、Production選択は`NO-GO`でした。

新しいUniversal causal teacherとHierarchical sequence v2も、次を固定した比較Evidenceが揃うまで優位性を主張しません。

- 同じDataset、Fold、Seed、Teacher、Timesteps
- 同じAction、Reward、Risk、Execution policy
- Legacy相当の参照Commit
- Parameter-countを合わせたMLP control
- OOS growth、Baseline uplift、Regret、Drawdown、Turnover、Cost
- Seed dispersion、Worst fold、Throughput、GPU memory

Architecture改善やteacher software改善は、性能改善の証明ではありません。

## Stateful execution status

通常Transition、Baseline pre-roll、Sensitivity replay、Deterministic replayは同じStateful order engineを使用します。

維持対象のPrimary promotion条件:

- Conservative OHLC path
- Processing-bar shared capacity
- Partial-fill carry
- Complete OrderEvent evidence
- Matching `execution_policy_digest`

OHLCVはQueue position、Hidden liquidity、Auction、L2 depthを表現しません。Paper reconciliationと実Venue比較が別途必要です。

## Causal and sealed evaluation

NormalizerはFold train capabilityだけでFitし、その後Freezeします。Outer testはConfiguration選択後に一度だけ開きます。

Universal teacher holdoutはteacherをBCへ入れてよいか判定するadmission evidenceであり、最終Policyのsealed testではありません。Candidate eligibilityと最終評価はSeed分布、Worst seed、Dispersion、Turnover、Cost、Drawdownを含む別のzero-shot/sealed evaluationで判定します。Sealed returnを学習や選択へ戻しません。

独立Foldを、Account-state handoffなしに連続Portfolio returnや1つのMaximum drawdownとして扱いません。

## Studio and diagnostics

`training_telemetry_v1`はAppend-only診断Dataです。Producer-issued`episode_id`を優先し、選択したVector environmentのCurrent episodeだけを表示します。Historical records with`null` identityはTerminalとCounter rollbackで分割します。

Universal monitorはmember heartbeat前のcausal teacher progressも読み、学習開始後はgross/net PnL、baseline excess、turnover、execution cost、target delta、sign flip等をtrend化します。OOM、non-finite値、traceback、container failure、stale heartbeatもfindingとして扱います。

TensorBoardには最適化ScalarとSequence attention/gate/gradient診断を表示できます。MonitorとTensorBoardはいずれもModel-selection evidenceではありません。

## Paper Serving and release

`serving_bundle_v6`は、Selected-final run、Dataset、Environment、Normalizer、Execution policy、Evaluation evidence、Policy loaderを結合します。`policy_mode`と`action_mode`を分離し、学習済みTarget-weight PolicyがResidualとして誤表示されることを防ぎます。

外部`ReleaseAttestation`はBundle digest、Source、Selection、Fresh confirmation、Paper reconciliation、Approver、Expiryを署名します。Private keyはOfflineに保持します。

次のいずれかが不正ならActivation前にFail closedします。

- SignatureまたはPublic key
- Bundle/File digest
- Observation/Action contract
- Architecture identity
- Normalizer
- Execution evidence
- Expiry

## Production GOに必要なもの

少なくとも次が必要です。

1. 十分な期間と市場状態を覆うOOS Evidence
2. Paired block-bootstrap下限が正
3. Predeclared複数Seed・Fold・AUMでの安定性
4. Conservative execution sensitivityの通過
5. Fresh signed confirmation
6. 実Paper environmentとのReconciliation
7. CUDAを含む対象Runtimeでの再現
8. Direct exchange connector、Secret管理、Kill switch、Alerting
9. 運用責任者による外部Authorization

現在は満たしていないため、Production statusは**NO-GO**です。
