# Universal Training

この文書は、維持対象コードベースのUniversal U3-U6学習経路の正本です。Universalは複数のtrain symbolから1つのPolicyを学習しますが、各episodeは1つのconcrete instrumentだけを取引し、Policy-facing symbol/actionは`INSTRUMENT`へ固定します。複数銘柄を同時配分するmulti-asset portfolio policyではありません。

> **研究状態**
>
> - Universal runtime / U3-U6 software: 実装済み
> - Canonical U6 teacher: `causal_alpha_ridge`
> - Oracle / Trend teacher: 診断・互換経路
> - 実データでのTeacher admission、複数Seed/Fold、zero-shot、sealed evaluation: 別途実証が必要
> - Production status: **NO-GO**
> - 収益性の主張: なし

## 1. Runtimeとデータ境界

U6はsecret-freeなUniversal runtime manifestとfrozen metadataを入力にします。Runtimeは宣言済みtrain symbolのDataset、shared normalizer、feature schema、instrument descriptor、execution metadataのDigestを再検証してから学習資源を組み立てます。

```text
runtime manifest
  -> train-symbol dataset bindings
  -> target-local causal features
  -> continuous instrument descriptors
  -> shared train-only normalizer
  -> episode-routed single-instrument environments
```

Teacher fitting、candidate selection、BCにはtrain symbolだけを使用します。Validation/Test symbolをUniversal teacher前処理へ混入させません。NormalizerもFold train capabilityだけでFitし、その後Freezeします。

## 2. Canonical teacher: `causal_alpha_ridge`

Canonical U6は、hindsight Oracleや固定Trend ruleではなく、train-only fitted causal alpha teacherを使用します。

Teacherは既存の因果的なtarget-local featureとinstrument descriptorから、24hと72hのgross forward log returnを予測する決定論的なpooled ridge modelです。FitとScalerはknowledge cutoffより前に完全実現したlabelだけを使用します。

```text
causal features + instrument descriptor
  -> prefix-only scaling
  -> pooled deterministic ridge
  -> 24h / 72h gross-return prediction
  -> bounded target exposure controller
```

各labelは、そのlabel realizationの終端がknowledge cutoffより前にある場合だけFitへ入ります。将来label、Validation/Test symbol、holdout performanceをteacher action生成へ戻しません。

Feature availabilityが欠ける場合は、Fit済みmean相当、すなわちstandardized value `0`として扱います。Inactiveまたはnon-tradableなdecisionでは新しいtargetを提出せず、直前targetを維持します。

## 3. Episode partitionとholdout

各train symbolについて、train range内のcomplete episodeを時系列順に分割します。

```text
earlier complete episodes -> candidate selection / BC scope
latest complete 720h episode -> causal teacher holdout
```

最新のcomplete 720h episodeはcandidate selectionへ使用しません。各holdout episodeについて、model fitはそのepisode開始前に完全実現したlabelだけを利用します。

Holdoutはteacher candidateを選ぶためのtestではなく、選択済みteacherをBCへ入れてよいか判定するteacher admissionです。最終Policyの一般化評価はzero-shot / sealed evaluationで別に行います。

## 4. Exposure controller

Return predictionはbounded target exposureへ変換します。維持対象controllerは次を使用します。

- `tanh`によるbounded exposure
- entry / exit hysteresis
- existing no-trade band
- 1 decisionあたりのmaximum target-weight delta

**Minimum holding periodはありません。** 次のdecisionでsignalとcontroller条件が反転すれば、positionを反転できます。不要なturnoverを抑える責務はhysteresis、no-trade band、target delta capが持ちます。

## 5. Candidate selectionと計算再利用

Candidate gridはearlier selection episodeだけをproduction execution replayへ通し、after-costの経済指標とrisk violationを使って決定論的にrankします。Selection中にholdoutを評価しません。

同じsample scope、knowledge cutoff、ridge configで必要になるpooled fitは`CausalAlphaExpandingFitCache`で共有します。Predictionもcandidate間で再利用可能なcacheを使います。これにより、同一ridge fitをsymbol/candidateごとに繰り返す経路を避けます。

Selection progressはreplay count、fit count/cache hit、prediction count/cache hitを含むprogress artifactへ逐次保存します。

### Selection checkpointのresume identity

Candidate replayは途中再開できますが、永続化済みMetricを現在のgenerator実装へ無条件に流用しません。v2 checkpoint row schemaは`causal_alpha_selection_checkpoint_metric_v2`で、少なくとも次の2つのIdentityを束縛します。

```text
grid_digest
  = candidate grid / candidate identityのDigest

generator_code_digest
  = causal-alpha selection metric generator実装のCode identity
```

Resume時は現在のcandidate gridから`grid_digest`を、現在のgenerator実装から`generator_code_digest`を再計算し、保存済みrowと一致することを確認してからreplay Metricを受理します。どちらかが一致しなければ**Fail closed**し、保存済みMetricを再利用しません。

この境界により、candidate gridが同じでもselection replayを生成する実装が変わった場合、旧checkpointを新しいgeneratorの結果として扱うことを禁止します。Checkpoint identityは計算再利用のためのCache keyではなく、再開してよいEvidence closureの一部です。

### Research-only Causal Alpha V3 lane

`Causal Alpha V3`は、r3で観測したprediction-to-target境界とlearner-state distribution shiftを検証するための**research-only**経路です。Canonical U6を置き換えず、既存の`behavior_cloning_teacher: "causal_alpha_ridge"`、`action.mode=target_weight`、teacher admission、risk、execution、reward契約を変更しません。

Historical v2 selection checkpointはread-only診断だけに使用します。診断出力は常に`promotion_eligible=false`で、旧`generator_code_digest`を保持したままpaired scope比較と重複prediction evidenceのde-duplicationを行います。この出力をselection resume、candidate promotion、teacher admissionの代替Evidenceとして使用してはいけません。

V3 predictorはknowledge cutoffより前に完全実現したtrain/selection labelだけを使い、次をresearch evidenceへ束縛します。

- label intervalの**overlap** concurrencyから求めるuniqueness weight
- symbolごとのeligible weight massを揃えるsymbol-balanced pooling
- weighted/objective-normalized ridge
- 72h predictionを24h-equivalentへ正規化したmulti-horizon forecast
- 24h/72h disagreementとfit residual scaleから得る**uncertainty**

**teacher-admission holdout**、Validation/Test symbol、sealed evaluationをV3 fit、weight、model selection、threshold調整へ戻しません。V3を将来canonical候補として凍結する場合も、未開封のteacher admissionを別途通過させる必要があります。

V3 target compilerは絶対positionではなく`delta_weight`のconservative incremental edgeを評価し、HOLDをobjective `0`のfirst-class actionとして扱います。新規turnoverにだけexecution costとedge marginを課し、既存positionのsunk costを再課金しません。通常のalpha rebalance cadence外ではHOLDし、strong reversalまたはliquidity cap contraction時だけ早期変更を許可します。**Reward unchanged**: scalar rewardは既存の**pure net-log-growth**のままで、V3 cost/uncertaintyをreward shapingとして二重計上しません。

`anchored_target_residual`はopt-inのresearch action modeです。Target-weight alphaをteacher anchorとして、Policyは小さなbounded residualだけを出力します。Zero residualはrisk projection前のteacher anchorを厳密に再現し、既存`residual`/`target_weight`の意味を変更しません。Canonical U6 configはこのmodeを選択しません。

**DAgger**はlearner actionでsimulatorを進め、learnerが実際に訪れた同じstateをcausal teacherで再ラベルします。Dataset、Environment、ActionSpec、Teacher identityがずれたrolloutはmerge時にfail closedします。DAgger datasetはBCのstate-distribution診断・改善用であり、**does not bypass teacher admission**です。


### Artifact-bound V3 research runner

V3 primitivesを実データで最後まで評価するため、`scripts/run_universal_causal_alpha_v3_research.py`をresearch-only entrypointとして使用します。Runnerは次の順序を固定し、途中結果を次段階の選択条件へ流用しません。

```text
strict authored JSON
  -> artifact/runtime identity closure
  -> chronological V3 fit
  -> signal gate
  -> immutable candidate freeze
  -> resumable production replay
  -> economic selection
  -> untouched teacher admission
  -> research-only teacher package
```

Signal gateはearlier selection contractだけを使用し、fitは各contract開始時点より前に完全実現したlabelへ限定します。72h label intervalが重複するpredictionは独立標本として水増しせず、deterministic non-overlapping cohortへ落としてからscope-level block bootstrapを行います。Signal gateに失敗した場合、candidate freeze、production replay、teacher-admission holdoutへ進みません。

`signal/records/<fit_config_digest>/<symbol>/<episode>.json`は引き続き`causal_alpha_v3_signal_scope_v2`のcanonical Signal Gate evidenceです。Fresh V3 runでは同じscope計算中に、別artifactとして`signal/diagnostics/<fit_config_digest>/<symbol>/<episode>.json`へ`causal_alpha_v3_signal_diagnostic_scope_v1` sidecarも保存します。Sidecarは24h/72h/fused predictionとcontract内で完全実現したlabel、feature availability、fit済みcoefficient/scaler/intercept、residual RMSE、overlap-weight identity、pooled/per-symbol weighted ESSを保持しますが、常に`research_only=true`かつ`promotion_eligible=false`です。

Diagnostic sidecarはSignal Gate、candidate freeze、economic selection、Teacher admission、BC/RL、promotion判断へ入力しません。Gateへ渡すのはcanonical `CausalAlphaV3SignalScopeMetric`だけです。Canonical metricとdiagnosticはrun/fit/symbol/episode/contract/fit/forecast/cohort identityで相互束縛し、片方だけがvalidに永続化されたpartial writeは同じscopeを再計算してpersist済み側とのdigest一致を確認した場合だけ欠損側を補修します。Corrupt/stale/wrong-path/wrong-run evidenceは上書きせずfail closedします。旧Signal V2 runでdiagnostic sidecarが存在しない場合、そのcanonical Gate artifactは歴史的Evidenceとしてそのまま扱いますが、24h/72h分解やcoefficient/availability/ESS診断をpost-hoc refitで捏造・移行しません。

Candidate freeze後のproduction replayはproduction environment、既存execution cost、causal liquidity capを使用します。Replay結果は`(candidate_digest, symbol, episode_index)`ごとのimmutable recordとしてatomicに保存し、resume時にはrun/freeze/contract/code identityを再検証します。Valid recordが存在するscopeだけskipし、unknown/corrupt/identity-drifted recordはfail closedします。`below_minimum_notional`と`zero_quantity_after_rounding`は既存契約どおりexplained no-fillとして扱い、それ以外のexecution rejection、hard-risk violation、`-5%` lower-tail breachは不可逆reject条件です。

Teacher admissionはcandidate selection完了後にselected candidateだけへ開きます。Persist済みper-symbol admission recordは再利用しますが、process crashがevaluation完了とrecord永続化の間に発生した場合まで「物理的にexactly once」とは主張しません。保証するのは、persist済みholdout resultを再評価せず、各symbolについてexactly one durable admission recordを受理することです。Admission failure時にはresearch-only teacher packageを生成しません。

CLIのterminal research outcomeは、admittedがexit code 0、signal rejectionが**exit code 2**、economic selection rejectionが**exit code 3**、teacher admission rejectionが**exit code 4**です。いずれのV3 evidenceも`promotion_eligible=false`で、Software successをProduction GOへ読み替えません。

Downstream learner pathはこのrunnerの非目標です。`DAgger -> BC`、critic warm start、`anchored PPO`/Lagrangian系は**only after teacher admission**で別工程として実行します。Admission前に`anchored_target_residual`やDAggerを使ってcanonical gateを迂回しません。

## 6. Teacher admissionとfail-closed順序

選択後の順序は固定です。

```text
candidate selection
  -> selection artifactを永続化
  -> selected teacher batchを構築
  -> 各train symbolのholdoutをexactly once replay
  -> teacher admissionを固定
  -> admission PASSならBC
  -> BC economic/reconstruction gates
  -> optional critic warm start
  -> PPO-family updates
```

Teacher holdoutはshared package生成時に各symbol 1回だけ評価します。Seedごと、Architectureごと、Algorithmごとにholdoutを再評価しません。

Teacher admissionが失敗した場合はBCを開始せず、critic warm startやPPO updateへ進みません。既存のBC reconstruction/economic gateも緩めません。

Teacher admissionはBC後のbootstrap gateを複製する統計検定ではなく、未開封holdoutに対する軽量なpre-BC safety gateです。維持対象の共通判定は、aggregate gross returnとafter-cost aggregate net returnが非負、gross-negative symbolが過半数でない、worst-symbol net returnが`-0.05`以上、かつholdout全体のtrade countが正であることを要求します。V3はこの共通経済gateを再利用し、その上にhard-risk violationとunexplained execution rejectionのreject条件だけを追加します。

## 7. U5/U6で共有するteacher identity

U5 architecture ablationでは、同一のcausal teacher packageを一度だけ構築し、全architecture candidateへ渡します。U6では同一packageをPPO、Lagrangian PPO、Discounted Lagrangian PPOへ共有します。

```text
one runtime/data identity
  -> one selected causal teacher package
     -> U5 architecture candidates
     -> U6 PPO
     -> U6 Lagrangian PPO
     -> U6 Discounted Lagrangian PPO
```

Algorithm差分のためにteacher actionやholdout admissionを作り直しません。Discount factorなどAlgorithm側の差分はcritic/optimization contractに属し、teacher identityとは分離します。

## 8. Shared teacher artifacts

U5/U6のshared teacher evidence rootは通常`<output-root>/_shared-causal-teacher/`です。主要artifact:

```text
_shared-causal-teacher/
├── causal-teacher-selection.json
├── causal-teacher-progress.json
├── causal-teacher-admission.json
└── causal-teacher-package.json
```

- `causal-teacher-selection.json`: candidate grid、selection metrics、selected candidate identity
- `causal-teacher-progress.json`: teacher selection/buildの進捗とcache counters
- `causal-teacher-admission.json`: per-symbol holdout economicsとadmission result
- `causal-teacher-package.json`: teacher config/code/data/selection/admissionを束ねるimmutable identity

各training member側では`universal-pretraining.json`とpolicy-stage snapshotを保存し、random、BC、BC+criticなどの段階を区別します。

## 9. U6 full-research training

事前にUniversal runtime manifestをmaterializeしてから、3つのcanonical configを同じ非Algorithm条件で実行します。

```bash
uv sync --extra dev --extra train-sb3 --extra postgres

uv run python scripts/run_universal_full_research.py \
  --selected-architecture u_medium_direct \
  --ppo-config examples/binance-multitimeframe/universal-u6-ppo.json \
  --lagrangian-config examples/binance-multitimeframe/universal-u6-lagrangian.json \
  --discounted-config examples/binance-multitimeframe/universal-u6-discounted.json \
  --runtime-manifest artifacts/universal/runtime-manifest.json \
  --frozen-metadata-root data/runtime/frozen-metadata/usds-m \
  --baseline supervised_allocator \
  --fold 0 \
  --fold 1 \
  --output-root artifacts/universal/full-research
```

Canonical U6 configは`behavior_cloning_teacher: "causal_alpha_ridge"`を使用します。Rewardは既存のpure net-log-growth contractを維持し、teacher導入のための追加scalar cost penaltyを入れません。

## 10. Universal training monitor

Teacher selection中はまだmember heartbeatが存在しないため、monitorはshared teacher progressも読みます。

```bash
uv run python scripts/monitor_universal_training.py \
  --generation-root artifacts/universal/full-research \
  --output-root artifacts/universal/monitor
```

Docker containerも監視する場合は`--container <name>`を追加します。Monitorは次を生成します。

```text
monitor-snapshot.json
reward-trends.json
```

Telemetry trendには少なくとも次を含めます。

- reward / portfolio value / baseline portfolio value
- drawdown / interval cost / interval return
- filled turnover / fill count
- interval gross return / baseline excess return
- target delta / sign flip
- command target delta / command sign flip
- gross PnL / net PnL

OOM、non-finite値、traceback、container failure、stale heartbeatもfail-closedなfindingとして扱います。これらのmonitor値は診断用であり、sealed model-selection evidenceではありません。

## 11. Software successとResearch success

次を混同しません。

```text
CI PASS
  != teacher economic admission PASS
  != full-training completion
  != zero-shot / sealed evaluation PASS
  != profitability
  != Production authorization
```

長時間学習へ進む前に、実データteacher admission、決定論的再現、短いeconomic smokeを確認します。その後も複数Seed/Fold、zero-shot unseen-symbol evaluation、sealed test、execution sensitivity、baseline comparison、hard-safety gateが必要です。

現在のProduction statusは**NO-GO**です。

## 12. 関連文書

- [学習クイックスタート](../START.md)
- [設定リファレンス](CONFIGURATION.md)
- [アーキテクチャ](ARCHITECTURE.md)
- [研究状態](RESEARCH_STATUS.md)
- [Reward objective](REWARD_OBJECTIVE.md)
- [Multi-Timeframe research](MULTITIMEFRAME_RESEARCH.md)
- [Docker GPU full training](operations/docker-gpu-full-training.md)

`docs/implementation-plans/`配下は設計・実装時点の履歴資料です。現在の運用契約や正本としては、この文書と上記の現行文書を使用してください。