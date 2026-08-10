# Universal Single-Instrument Zero-Shot Policy Design

## 結論

維持対象の次世代モデルは、**1つの共通checkpointを複数銘柄の学習データで訓練し、推論・発注時は常に1銘柄だけを扱うUniversal Single-Instrument Policy**とする。

```text
1 model
1 generic instrument slot
1 scalar target-weight action
N training symbols
1 traded symbol per environment / deployment
zero-shot deployment to symbols absent from training
```

複数銘柄のtarget weightを同時に出すportfolio modelにはしない。既存の3-symbol triplet training runtimeも、そのまま新しいmaintained runtimeにはしない。既存資産から再利用するのは、symbol-disjoint manifest、zero-shot evaluationの統計契約、identity-free feature/policy architecture、artifact/evidence境界である。

報酬とEpisode境界はPR #369の契約、すなわち実コスト控除後net log growth、`gamma=1.0`のfinite-horizon termination、time-to-go observationを正本とする。PR #369相当が最新`main`へ統合される前に、新しいuniversal training generationを開始しない。

## 目的

- Docker内の検証済み保存データから、利用可能な銘柄集合を実行時に発見する。
- 実行開始時に銘柄集合を不変manifestへ固定し、学習中の増減でrun identityを変えない。
- train、validation、sealed testの銘柄集合を完全に分離する。
- trainに含まれない銘柄へ同一checkpointをzero-shot適用できることを必須gateにする。
- すべての環境、checkpoint、deploymentでaction shapeを`(1,)`に固定する。
- 銘柄名、symbol ID、ticker hash、one-hot embeddingをpolicy inputへ入れない。
- 既存のhard risk、execution、artifact、checkpoint、sealed-test境界を維持する。
- 現行BTC generationと既存artifactを再解釈、上書き、暗黙移行しない。

## 非目標

- 複数銘柄を同時保有するportfolio policy。
- 複数actionを同時出力するtriplet model。
- 学習中にDockerへ追加された銘柄をactive runへ自動追加すること。
- zero-shot対象でのonline fine-tuning。
- symbol embeddingによる銘柄暗記。
- average-reward RLの導入。
- この設計だけによるProduction authorization。
- 現行`training-full.json`や既存checkpointの意味変更。

## 現状から再利用する契約

### 維持するもの

- maintained environmentの1 instrument / 1 target-weight action境界。
- `hierarchical_sequence_v2`のcausal native-timeframe encoder。
- policy architectureと具体的asset bindingを分離するidentity-free設計。
- `SymbolDisjointManifest`のtrain/validation/test閉包とcontent digest。
- Stage A zero-shot evaluationのcandidate × symbol-group × fold × seed閉包、paired baseline、bootstrap、worst-group、worst-seed、pass-fractionの考え方。
- PostgreSQL indicator manifestの検証済みsymbol inventory、feature config digest、期間、artifact closure。
- PR #368のexclusive `terminated` / `truncated` telemetryとsmooth exploration計測。
- PR #369のpure growth rewardとfinite-horizon boundary。

### そのまま再利用しないもの

- `full_research_pipeline.py`の`BTCUSDT`固定。
- 3 generic slots、3 simultaneous target weightsを前提とするsymbol-triplet training runtime。
- concrete symbolをcheckpointの唯一のdeployment先として固定するasset binding。
- 単一銘柄で退化するcross-asset feature群。
- train symbolを順番にcheckpoint transferするだけのcontinual training。

## 検討した方式

### A. 銘柄ごとの逐次transfer training

1銘柄を学習し、そのcheckpointを次の銘柄へtransferする。

利点:

- 既存のstage cursorとcheckpoint transferを再利用しやすい。
- 1 stageのデータ量とmemoryが小さい。

欠点:

- 銘柄順序に依存する。
- catastrophic forgettingを起こしやすい。
- 最後に学習した銘柄がpolicyを支配しやすい。
- 同一optimizer step内で複数銘柄のgradientを比較できない。

採用しない。

### B. 3-symbol generic-slot joint policy

既存Stage Aのtriplet datasetを用い、3銘柄を同時観測・同時action出力する。

利点:

- 既存symbol-disjoint triplet infrastructureを広く再利用できる。
- cross-asset attentionを直接使える。

欠点:

- action shapeが3になり、要件の1銘柄運用と異なる。
- portfolio allocation問題へ変質する。
- slot interactionを学習したpolicyは、単一symbol deploymentと同一ではない。

採用しない。

### C. Episode-routed single-instrument mixture

各environmentは常に1銘柄だけを保持し、resetごとにtrain split内の銘柄を決定論的に切り替える。すべてのenvironmentが同じmodel weightsを共有する。

利点:

- action shapeと実運用契約が常に1。
- 同一rolloutへ複数銘柄のtransitionを混合できる。
- continual-trainingの順序依存を避けられる。
- concrete symbolをpolicy inputへ含めずに学習できる。
- unseen symbolも同じgeneric slotへbindできる。

欠点:

- multi-dataset episode routingとbalanced normalizationが必要。
- SB3 vector environmentとartifact identityを拡張する必要がある。

**Cを採用する。**

## 1. Docker保存銘柄の発見と不変化

### 1.1 Source of truth

Universal training v1のsource of truthは、Docker内PostgreSQLに保存された検証済みnative indicator manifestとする。manifestが保持する次の情報を利用する。

- cache ID
- market
- ordered symbol inventory
- start/end time
- feature configuration digest
- symbol × timeframe artifact closure

Binance Vision file cacheを使う既存BTC pathはlegacy/single-symbol互換として残す。Universal trainingは、複数銘柄の完全な共通schemaとcoverageを一括検証できるPostgreSQL manifestを使用する。

### 1.2 `StoredInstrumentCatalog`

新しい不変artifact `stored_instrument_catalog_v1`をgeneration開始時にmaterializeする。

必須field:

```text
source_cache_id
source_manifest_digest
market
feature_config_digest
required_timeframes
research_start
research_end
eligible_symbols
excluded_symbols_with_reasons
per_symbol_artifact_digests
per_symbol_metadata_digests
catalog_digest
```

銘柄は次をすべて満たす場合だけeligibleとする。

- 15m / 1h / 4h / 1d artifactが完全に存在する。
- 全銘柄でfeature schemaが同一。
- research interval全体の必要coverageを持つ。
- point-in-time execution metadataまたは明示された保守的metadata evidenceを持つ。
- linear USDS-M contractとして現在のaccounting modelで表現可能。
- listing前、delisting後、欠損期間がactivity/tradability evidenceへ正しく反映される。

Active generationはcatalog digestを固定する。Dockerへ後から銘柄やbarが追加されても、active generationは変更しない。更新を取り込む場合は新catalog、新source/image identity、新generationを作る。

### 1.3 最小銘柄数とsplit数

Zero-shotを必須評価にするため、eligible universeは最低15銘柄を要求する。

銘柄数を`N`としたとき、split countは次で固定する。

```text
validation_count = max(3, floor(N / 5))
test_count       = max(3, floor(N / 5))
train_count      = N - validation_count - test_count
```

`train_count < 9`ならfail closedする。N=15ではtrain 9、validation 3、test 3となり、既存Stage Aの最小構成と一致する。

`SymbolDisjointManifest`を用い、seedとcatalog digestからsplitを決定する。validation/test symbolのdataset、normalizer statistics、teacher labelsはtrainingへ渡さない。

## 2. Generic single-instrument environment

### 2.1 Generic slot

Policy-facing symbol vocabularyは常に次とする。

```text
symbols = ("INSTRUMENT",)
action_names = ("target_weight:INSTRUMENT",)
action_shape = (1,)
```

Concrete symbolはenvironment内部の`InstrumentEpisodeBinding`だけが保持する。

```text
concrete_symbol
source_dataset_id
symbol_dataset_digest
episode_start / episode_stop
execution_metadata_digest
instrument_descriptor_digest
split
```

Policy、checkpoint、structured exportはconcrete tickerを入力にもarchitecture identityにも含めない。

### 2.2 Episode router

各vector environmentはreset時にtrain symbolを選択する。

選択順は乱数抽選ではなく、run seed、environment index、completed episode countから作る決定論的balanced permutationとする。

```text
every train symbol appears once per environment cycle
before any symbol appears twice
```

同一rolloutでは複数environmentが異なる銘柄を処理できるが、各environmentのstate、book、order、reward、actionは完全に単一銘柄である。

銘柄選択後のepisode開始点は既存の`episode_sampling_mode`を使用する。Universal profileの初期defaultは`regime_balanced`とし、uniform/stress-tailは明示的ablationまたは評価scenarioとする。

### 2.3 Failure behavior

- 銘柄datasetをloadできない場合、そのepisodeだけ別銘柄へfallbackしない。runをfail closedする。
- reset途中でschemaやmetadata digestがcatalogと違えばrunを停止する。
- validation/test symbolがtraining routerへ渡された場合は即時例外にする。
- environmentのauto-reset後も終了episodeのconcrete symbol identityをtelemetryへ残す。

## 3. Observation contract

### 3.1 Symbol identityを入力しない

禁止する入力:

- ticker文字列
- symbol index
- one-hot symbol ID
- hash embedding
- train-universe内の固定ordinal

Policyはmarket state、execution state、continuous instrument descriptorsだけから判断する。

### 3.2 Target-local feature contract

Universal profile v1は、各timeframeのtarget-local featuresだけを使用する。現在の226-feature presetから、単一symbolで退化する次のcross-asset featureを各timeframeで除外する。

- relative return to BTC
- rolling correlation to BTC
- rolling beta to BTC
- cross-sectional momentum rank
- cross-asset dispersion

4 timeframe合計で20 channelを除外し、target-local feature数は206とする。

```text
15m: 54
1h : 54
4h : 50
1d : 48
```

Market-context branchはv1の必須入力にしない。将来、train-symbolだけから作るaggregate contextを独立ablationとして追加できるが、zero-shot core modelは対象銘柄のdataだけで推論可能でなければならない。

### 3.3 Continuous instrument descriptors

`instrument_context_v1`として、tickerを表さない連続値を追加する。

```text
listing_age_log_days
trailing_30d_quote_notional_log
tick_to_mark_ratio
lot_notional_to_equity
minimum_notional_to_equity
fee_rate
spread_rate
impact_rate
max_participation_rate
```

すべてdecision時点で因果的に取得できる値だけを使用する。borrow availability/rate、mark/index basis、position、pending order、fill、costは既存asset/execution stateに残す。

### 3.4 Time-to-go

PR #369のgamma-one profileでは、finite-horizon terminationとtime-to-go observationを必須にする。Discounted continuing ablationだけはexternal truncationかつtime-to-goなしを維持する。

## 4. Symbol-balanced normalization

Universal modelはconcrete symbolごとの固定normalizerを持たない。Validation/test symbolのstatisticsも使用しない。

新しい`symbol_balanced_standard_normalizer_v1`を定義する。

1. 各train symbolのfold train rangeから、各featureについて同数のvalid observationを決定論的な等間隔indexで抽出する。
2. 1 symbolあたりの抽出数は、全train symbolの最小valid countと上限100,000の小さい方とする。
3. 全symbolを同じ重みで結合してmean/stdを計算する。
4. stdが数値epsilon以下のchannelはconstant channelとして0へ写像する。
5. transform後を`[-10, 10]`へclipする。
6. mask、missing reason、activity、tradability、categorical stateは既存passthrough contractを維持する。

Normalizer artifactは次へbindする。

```text
catalog_digest
symbol_disjoint_manifest_digest
train_symbols
fold_train_range
feature_schema_digest
sampling_contract
statistics_digest
```

## 5. Model architecture

### 5.1 Initial maintained candidate: U-Medium Direct

最初の本命candidateは次とする。

```text
sequence_tcn_capacity = compact
sequence_d_model = 256
timeframe attention heads = 4
timeframe attention layers = 1
timeframe FFN multiplier = 3
asset attention = inactive for n_symbols=1
sequence dropout = 0.0
actor head = shared_target_v1
actor MLP = 256 -> 128
critic MLP = 256 -> 128
shared scalar log_std
```

Current standard d336/2-layer modelを即時削除しない。U-Large controlとして残す。

### 5.2 Shared and separate responsibilities

共有する部分:

- per-timeframe causal TCN
- timeframe fusion
- target-local market representation
- instrument-context encoder

分離する部分:

- actor adapter
- critic adapter
- action head
- value head

単一symbolではasset tokenとpooled tokenが同一になるため、actorへ両方を連結しない。Generic instrument latentは1回だけactor/criticへ渡す。

### 5.3 Critic input

Criticは圧縮済みmarket latentだけでなく、次をexact stateとして直接受け取る。

- current effective weight
- drawdown
- margin utilization
- cash weight
- pending target/order summary
- recent fill/cost summary
- time-to-go
- instrument descriptors

これにより、value estimationに必要なportfolio stateが大規模encoder内の圧縮表現だけに依存しない。

### 5.4 Architecture ablations

同じreward、split、normalizer、BC、PPO条件で次を比較する。

```text
U-Small Direct:
  compact TCN, d192, 4 heads, 1 layer, direct head

U-Medium Direct:
  compact TCN, d256, 4 heads, 1 layer, direct head

U-Medium Gate:
  U-Mediumとactor headだけhierarchical_gate_target_v1

U-Large Direct:
  standard TCN, d336, 8 heads, 2 layers, direct head
```

Gate/Direct profileは`target_weight_count=1`に固定する。既存3-action ablation profileをuniversal evaluationへ流用しない。

## 6. Behavior Cloning and critic warm start

### 6.1 Universal BC dataset

Oracle teacher episodeはtrain symbolsからだけ生成する。Batch samplingは次の二段階とする。

```text
uniform train symbol
-> uniform eligible teacher episode/sample within that symbol
```

長いhistoryや高activityを持つ銘柄がdatasetを支配しないようにする。

BC artifactはsymbolごとのsample count、teacher digest、execution identity、return distributionを記録する。

### 6.2 BC evaluation

BC採用条件を二つに分ける。

- seen-symbol temporal holdout: train symbolの未来期間
- unseen-symbol validation: validation splitの銘柄

Validation symbolのteacher labelは評価だけに使用し、optimizerへ渡さない。

必須比較:

```text
no BC
universal Oracle BC
```

BC gate passだけでは採用しない。固定causal evaluationでno-BC初期policyを上回ることを要求する。

### 6.3 Critic warm start

PR #369のfinite-horizon returnを用いて、teacher trajectoryの各時点からEpisode終端までの実コスト控除後net log growthをvalue targetとして作る。

Critic warm start中はactorとshared encoderをfreezeし、critic adapter/value headだけを学習する。その後PPO開始時に全parameterをunfreezeする。

比較candidate:

```text
random critic initialization
finite-horizon critic warm start
```

Critic warm startはarchitecture screenで有効性を確認してからdefaultへ昇格する。

## 7. RL algorithms

### 初期maintained比較

- PPO: universal policyの必須control。
- Lagrangian PPO: PR #369で分離された7 cost channelを管理する本命constraint candidate。
- Discounted Lagrangian PPO: time-preference ablation。

通常PPOが学習能力を示す前に、Lagrangian candidateだけを主力にしない。

### 後続候補

- SAC
- TQC

Off-policy法はsymbol-balanced replay buffer、generic single-instrument sequence encoder、artifact identityを実装した独立phaseで扱う。Universal v1の実装へ同時投入しない。

### 非RL baseline

同じtarget-local featureとdescriptorを用い、将来after-cost return/riskからtarget weightを出すsupervised allocatorを必須baselineにする。RLがこのbaselineをzero-shot評価で上回れなければ、RL採用根拠は不足と判定する。

## 8. Zero-shot evaluation

### 8.1 Evaluation cell

Triplet単位ではなく、次のcellを正本にする。

```text
candidate × concrete symbol × time fold × seed
```

各symbol/fold/seedに同一baselineを一度だけ評価し、candidate間で共有する。

### 8.2 Selection and sealed test

- Architecture、BC、critic、algorithm selectionはvalidation symbolsだけを使用する。
- Test symbolsはcandidate選択後までアクセスしない。
- Sealed testではselected candidateだけを評価する。
- Test accessは既存durable ledgerと同じ一回性・identity closureを要求する。

### 8.3 Zero-shot gate

Validation/testの各splitで次を要求する。

- paired baseline excessのfold-cluster bootstrap 95%下限が0より大きい。
- symbol別median paired excessのworst valueが0以上。
- symbol pass fractionが2/3以上。
- seed別fold medianのworst valueが0以上。
- aggregate after-cost net log growth medianが正。
- forced liquidation、margin deficit、insolvency、hard-safety violationが全cellで0。
- nominalだけでなくrequired execution stressでもaggregate paired excessが正。
- 特定1銘柄の利益を除くと全体が失敗するcandidateを拒否する。

Thresholdはevaluation planのcanonical payloadへ固定する。Raw training rewardはprofile間比較に使わない。

### 8.4 Leakage guard

次を静的・動的に拒否する。

- train routerがvalidation/test symbol datasetをloadする。
- normalizerがvalidation/test rowsを参照する。
- BC optimizerがvalidation/test teacher sampleを受け取る。
- candidate selectionがsealed-test resultを参照する。
- policy observationにconcrete ticker/IDが含まれる。
- training artifactがdeployment target symbolを固定する。

## 9. Policy identity and deployment

### 9.1 Universal model manifest

新しい`universal_single_instrument_policy_v1` manifestを定義する。

```text
policy_digest
architecture_digest
observation_schema_digest
action_schema = one target-weight scalar
instrument_descriptor_schema_digest
normalizer_digest
reward/environment digest
training_catalog_digest
training_symbol_split_digest
training_symbols_digest
zero_shot_evidence_digest
```

Concrete symbolはmodel architecture identityへ含めない。

### 9.2 Deployment binding

Concrete symbolは`single_instrument_deployment_binding_v1`へ記録する。

```text
policy_digest
concrete_symbol
market/instrument contract digest
dataset/live feature schema digest
execution metadata digest
instrument descriptor evidence
seen_in_training
binding_digest
```

`seen_in_training=false`を許容し、zero-shot deploymentとして明示する。

### 9.3 OOD gate

Zero-shotは無制限な外挿許可ではない。Deployment時に次を確認する。

- 必要historyと4 timeframe coverageが揃う。
- feature/descriptor schemaが一致する。
- descriptorのtrain-universe standard scoreの最大絶対値が6以下。
- tick/lot/minimum-notionalが現在equityで実行可能。
- execution cost、liquidity、volatilityがpredeclared support envelope外でない。

Failした場合はmodelをloadして発注せず、`NO-GO: out_of_distribution_instrument`として記録する。

## 10. Data flow

```text
Docker PostgreSQL indicator manifest
  -> discover and verify stored symbols
  -> immutable StoredInstrumentCatalog
  -> SymbolDisjointManifest
       train symbols
         -> fold-local symbol-balanced normalizer
         -> symbol-balanced Oracle BC
         -> optional critic warm start
         -> episode-routed PPO/Lagrangian training
       validation symbols
         -> zero-shot candidate selection
       sealed test symbols
         -> selected-candidate-only zero-shot gate

Selected universal checkpoint
  -> Universal policy manifest
  -> concrete SingleInstrumentDeploymentBinding
  -> one-symbol serving environment
  -> one scalar target weight
```

## 11. Compatibility and migration

- 現行BTC generationは停止、resume、再解釈、上書きしない。
- Existing BTC single-symbol profilesは明示的legacy/controlとして保持する。
- Existing symbol-triplet Stage A contractsとartifactsは読取り可能なまま残す。
- Universal policyは新しいpolicy/environment/catalog/normalizer identityを持つ。
- Existing checkpointをuniversal modelとして自動変換しない。
- PR #369相当を最新mainへ統合した新source commit、新Docker image、新generationでだけUniversal trainingを開始する。

## 12. Implementation boundaries

### PR U0 — Reward/boundary prerequisite

- PR #369を最新mainへrebaseまたは同等変更を再適用する。
- Conflict、exact-head CI、draft状態を解消する。
- Universal workとreward式の再変更を混ぜない。

### PR U1 — Stored universe and symbol-disjoint catalog

- PostgreSQL manifestからeligible symbolsを発見。
- `StoredInstrumentCatalog`を追加。
- split countとminimum universeを固定。
- train/validation/test data-access guardを追加。
- Training behaviorは変更しない。

### PR U2 — Episode-routed single-instrument environment and identity

- Generic `INSTRUMENT` slot。
- Deterministic balanced symbol router。
- Universal model/deployment binding contracts。
- One action invariant。

### PR U3 — Universal features and normalization

- 206 target-local feature profile。
- `instrument_context_v1`。
- symbol-balanced normalizer。
- Existing profileは維持。

### PR U4 — Universal BC and critic warm start

- Symbol-balanced teacher artifact。
- Seen/unseen BC evaluation。
- No-BC comparison。
- Optional finite-horizon critic warm start。

### PR U5 — Architecture ablation and zero-shot evaluation

- U-Small/U-Medium/U-Large、Direct/Gate comparison。
- Candidate × symbol × fold × seed evidence。
- Validation selectionとsealed test。
- Zero-shot gate。

### PR U6 — Full-research integration

- Evidenceで選ばれたarchitectureだけをPPO/Lagrangian/discounted full comparisonへ接続。
- Current defaultを変更するのは、zero-shot gateが実データで通過した後だけ。

## 13. Test strategy

### Unit

- Docker manifest symbol discoveryとfield closure。
- Eligible/excluded reasonの決定性。
- 15未満、train 9未満のfail-closed。
- Balanced symbol permutation。
- Generic slotとone-action validation。
- Descriptor causal transform。
- Symbol-balanced normalizer。
- OOD envelope。

### Contract

- Train/validation/test symbol disjoint closure。
- Policy observationにsymbol IDがない。
- Policy architecture identityにconcrete tickerがない。
- Training catalog/split digestがrun identityへ入る。
- Deployment bindingだけがconcrete symbolを持つ。
- PR #369のreward/boundary contractを全universal gamma-one profileが共有する。

### Integration

- 複数toy symbolを4 vector envで混合し、各envが常に1 actionを返す。
- 1 rolloutで複数symbol transitionが入るが、book/order stateが交差しない。
- Validation/test symbolをtraining routerへ注入すると失敗する。
- Unseen symbol deployment bindingがschema/OOD pass時にloadできる。
- Seen and unseen deterministic evaluationが同じcheckpointを使用する。
- BC、PPO、checkpoint save/load、structured export、serving parity。

### Evidence

- Architecture screenはProduction evidenceではないと明示する。
- Final gateは複数time fold、3 seeds、validation/test symbols、execution stressを同一headで実行する。
- Exact candidate closureとshared baseline closureを検証する。

### Repository verification

- Focused tests
- Full pytest and branch coverage
- Ruff and format
- MyPy
- Import Linter
- Dead-code report
- Windows/Ubuntu compatibility
- Training image and non-root runtime
- PostgreSQL Catalog integration
- Structured serving/export parity

## 14. Completion criteria

Software implementation完了は次をすべて満たすこととする。

1. Docker保存銘柄がhard-coded poolなしで不変catalogへmaterializeされる。
2. train/validation/test symbolの完全分離がartifactとruntimeで強制される。
3. 同一policy checkpointがgeneric `(1,)` actionで複数concrete symbolへbindできる。
4. Policy inputとarchitecture identityにsymbol IDが存在しない。
5. Trainingは複数train symbolをbalancedに混合する。
6. Validation/test symbolはtraining、normalization、BCに使われない。
7. Unseen symbolのsealed-test gateが実行可能である。
8. OOD symbolはfail closedする。
9. Existing BTC/triplet artifactsの互換読取りを壊さない。
10. Exact-head CIが全成功する。

Research成功はsoftware完了と別であり、次を追加で要求する。

- Unseen-symbol testでpaired excess bootstrap下限が正。
- Worst-symbol、worst-seed、pass-fraction gateを満たす。
- Hard-safety violationが0。
- Supervised allocator baselineを上回る。
- PPO controlが学習可能で、その後にLagrangian candidateを比較できる。

満たさない場合、Universal RL policyはProductionへ昇格しない。