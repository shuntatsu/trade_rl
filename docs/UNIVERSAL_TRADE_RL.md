# Universal Trade RL U0 / U1 Maintained Contract

Universal Trade RLは、複数銘柄から1つのsymbol-independent policyを学習しつつ、1つのenvironment / inference / deployment candidateでは常に**1つのconcrete instrumentだけ**を扱う研究契約です。

> **Production status: NO-GO**
>
> U0/U1は研究境界、Observation / Action / Reward、normalization、artifact identityを固定します。RLの学習可能性、zero-shotの経済価値、収益性、Admission performance、実市場execution fidelity、Production readinessは証明しません。

## 1. 基本アーキテクチャ

将来のBase RLは複数のTrain銘柄からshared checkpointを学習しますが、policy側のinstrument slotはgenericです。

```text
1 shared model
1 generic INSTRUMENT slot
1 scalar target-exposure action
N Train symbols
1 traded symbol per environment
```

```text
one deployment candidate
  = one selected symbol
  = one fixed capital budget
  = one target-exposure policy application
```

BTC・ETH・SOLなどへ1つのActionで同時に資金配分するPortfolio policyは、このU0/U1 maintained contractでは定義しません。複数銘柄へ資本を配る場合は、銘柄ごとの独立した実行単位の外側にCapital allocation責務を置きます。

## 2. U0: Zero-shot Universe Isolation

U0は、利用可能な全source symbolを次のいずれか1つへ完全に割り当てます。

| Role | 用途 | Fitへの利用 | Phase evaluation |
| --- | --- | --- | --- |
| Train | Base RL、normalization、calibration、threshold/reward estimationなどのfit source | **可** | Train phaseではevaluation scopeへ公開しない |
| Development | 設計判断、失敗分析、candidate selectionの開発用外部確認 | **不可** | Development phaseでDevelopmentのみ |
| Admission | frozen generationを最後に1回だけzero-shot確認するholdout | **不可** | authorization後、Admission phaseでAdmissionのみ |
| Excluded | データ不足など、事前に理由を明記した非対象symbol | 不可 | 不可 |

Train、Development、Admissionは非空・sorted・unique・pairwise disjointです。Excluded symbolも理由付きで明示し、他のroleと重複できません。

ExclusionをArtifactに残す理由は、結果を見た後で都合の悪い銘柄を黙って外すcherry-pickingを防ぐためです。除外理由そのものが妥当かどうかは別途研究判断ですが、少なくとも「何を外したか」はUniverse identityに固定されます。

## 3. U0 source identityとcomplete manifest

role configだけではUniverseを固定したことになりません。各symbolは次のsource-data identityにも結び付けられます。

- dataset SHA-256 digest
- first / last timestamp
- row count

`universe.json`は、利用可能なsource catalogの**全symbol**をroleまたはexclusionへ割り当てたcomplete manifestです。未割当の利用可能symbol、設定だけに存在してsourceがないsymbol、role overlap、source digest driftはfail-closedです。

Manifestはentriesからrole configとsource catalogを再構築し、両方のdigestを再計算して検証します。したがってArtifact digestだけを作り直してroleやsource identityを差し替えることは契約整合性を満たしません。

## 4. U0 phase firewall

U0 access contractはphaseごとに利用可能な集合を固定します。evaluation scopeは累積公開しません。

```text
TRAIN
  evaluate = none
  fit      = Train

DEVELOPMENT
  evaluate = Development
  fit      = Train

ADMISSION
  evaluate = Admission
  fit      = none
```

Admissionへ入るには、次の3つを一致させたauthorizationが必要です。

1. universe manifest digest
2. frozen generation digest
3. Selection evidence digest

DevelopmentやTrainの時点でAdmission authorizationを持ち込むことも拒否します。Admissionを開いた後でも`fit_symbols`は空で、新しいfit、normalization、calibration、threshold estimation、reward coefficient estimation、RL trainingはすべて禁止です。

Admission metadataはdataset digestやtimestampなどの**integrity verification**には利用できますが、その値をfitへ使うことはできません。

## 5. U0 Train-only provenance

U0はfit artifactに、少なくとも次を結び付けます。

- Universe manifest digest
- fit purpose
- Train source symbols
- symbolごとのdataset digest
- knowledge cutoff

fit provenance builderはsource lookupより先にTrain-only scopeを検証します。Development / Admission symbolを「同じデータが存在するから」という理由でfitへ混入させることはできません。

現在のfit purposeには、feature normalization、calibration、threshold estimation、reward coefficient estimation、RL trainingを含みます。Admission symbolはどのfit purposeにも使えません。

## 6. U0 versioned run identity

| Stage | Universe | Model config | Fit provenance | Admission authorization |
| --- | --- | --- | --- | --- |
| `UNIVERSE_MATERIALIZATION` | 必須 | 禁止 | 禁止 | 禁止 |
| `BASE_TRAINING` | 必須 | 必須 | 必須 | 禁止 |
| `DEVELOPMENT_SELECTION` | 必須 | 必須 | 必須 | 禁止 |
| `ZERO_SHOT_ADMISSION` | 必須 | 必須 | 必須 | 必須 |

## 7. U0 atomic materialization

U0はstrict role configとstrict source catalogから次の2 Artifactを同時にmaterializeします。

```text
output-root/
  universe.json
  identity.json
```

実行例:

```powershell
trade-rl-universe `
  --config examples/binance/universal-trade-rl-universe.example.json `
  --source-catalog examples/binance/universal-trade-rl-source-catalog.example.json `
  --output-root var/runs/universal-trade-rl-u0-example
```

同梱exampleは契約形式を示すillustrative dataであり、Production datasetの証拠ではありません。

Materializationはstaging directory内で両Artifactを完成・flush/fsyncした後、directory単位で1回だけpublishします。既存outputへの再実行は、2 Artifactがcanonical byte単位で完全一致するときだけidempotent successです。既存Artifactの編集、role config drift、source digest drift、余分なfileは自動修復せず拒否します。

## 8. U0が証明しないこと

U0の成功から、RL policyの学習可能性、zero-shot transferの経済価値、収益性、Execution modelの完全性、Production認可を推論してはいけません。U0は**研究デザインの汚染を防ぐ境界**です。

## 9. U1実装検証とproduction-candidate freezeを分離する

U1のObservation / Action / Reward / normalization / artifact contractそのものは、synthetic fixtureや検証済みtest artifactを使って実装・TDD・反証検証できます。この段階は**contract implementation verification**であり、実際のproduction-candidate研究generationを開始したことにはなりません。

一方、**U2 Base RLを開始する前**には、研究用exampleではなく実際に使用するproduction-candidate dataについて次をすべて固定しなければなりません。

1. 事前確定したTrain / Development / Admission / Excluded role config
2. 各symbolの実source identityを持つsource catalog
3. U0 `universe.json` / `identity.json`
4. そのU0 generationからTrain-onlyでfitしたU1 normalizer
5. U1 `normalizer.json` / `u1_contract.json`

Roleやdataset generationを変更した場合は同じ研究runを継続せず、新しいUniverse generationとしてmaterializeし直します。Admission dataを見た後でTrain/Developmentへ戻すことも同一experimentのzero-shot契約違反です。

## 10. U1 Observation contract

U1は1-symbol environmentのみを受け入れ、policy observationは次のmarket sequenceとendogenous policy/runtime stateだけで構成します。

```text
15m × 96   = 24 hours
1h  × 168  = 7 days
4h  × 120  = 20 days
1d  × 60   = 60 days
```

market featureはsymbol-localでdimensionless / economically comparableなものに限定します。主なカテゴリはreturn/candle geometry、volatility、trend geometry、liquidity/activity、derivatives stateです。

policy tensorへ入れてはいけないもの:

- concrete symbol / ticker / role / dataset ID
- TrendTargets
- alpha output
- factor prior / cross-asset prior
- shadow book state
- baselineとの差分やunderperformance state
- manually latched ownership state
- remaining episode fraction
- raw nominal price scaleを識別できる値

future market mutationは`Observation(t)`を変えてはいけません。symbol renameや価格unitの変更だけでもpolicy tensorは変化してはいけません。

## 11. U1 policy state: exposureとpendingを分離する

U1は少なくとも次の4段階を別々に観測します。

```text
policy_requested_weight
  -> signal-delay pending_target_weight + pending_target_active
  -> risk_projected_weight
  -> realized current_weight
```

`pending_target_active`を別fieldに持つため、`pending_target_weight == 0.0`は「pendingなし」と同義ではありません。

さらにsignal-delay pending targetと、execution order lifecycleのpending stateは別物です。pending notional、order type/status、age、eligible delay、triggered、expiry distanceなどを別状態として保持します。

## 12. U1 Action / episode / reset contract

Action semantic:

```text
normalized_target_exposure_v1
shape = (1,)
range = [-1, +1]
policy_requested_weight = action * policy_weight_scale
```

`policy_weight_scale`はstatic/precommittedで、U1 V1 defaultは`1.0`です。dynamic Risk capをAction semanticへ混ぜません。NaN / inf / out-of-rangeはfail closedで、hidden clippingをしません。

U1 V1 runtimeは次を固定します。

```text
decision_hours = 0.25
signal_delay_decisions = 1
episode_hours = 720
initial_state_modes = ("cash",)
episode_boundary_mode = external_truncation
finite_horizon_observation = false
liquidate_on_end = false
```

cash-only resetはTrendStrategy由来のinitial position priorをpolicy stateへ注入しないためです。sample horizon到達は`truncated=True`であり、無料liquidationやterminal rewardを作りません。

## 13. U1 Train-only universal normalization

normalizer fitは、numeric arraysへ触れる前に次の順序でsource identityを検証します。

1. U0 Train-only normalization scopeを検証
2. `FEATURE_NORMALIZATION` provenanceを作成
3. published market artifactを独立inspection
4. inspected artifact digestとU0 manifest entryを照合
5. verified artifactをload
6. exactly-one expected symbolとfeature orderを確認
7. その後初めてnumeric valuesへアクセス

multi-timeframe featureはbase clockへcarryされるため、`feature_staleness_hours`からsource event timeを復元し、同一`(symbol, feature, source_event_time)`を1回だけcountします。knowledge cutoffより後のeventはfitできません。

統計量はrow数ではなく**symbolごとに1 vote**です。各symbolでmeanとsecond momentを作り、それらをsymbol間で等重み平均します。

### 13.1 Missing value semantics

`available=false`のraw placeholder値はpolicy tensorにもfit statisticsにも影響してはいけません。policy valueは`0.0`にmaskし、availabilityとstalenessは別channelとして保持します。

### 13.2 Fixed ±10 clipping

U1 V1はavailableなcontinuous market sequence valueへ:

```text
z = (x - mean) / scale
policy_value = clip(z, -10.0, +10.0)
```

を適用します。`normalizer_clip_value = 10.0`はstatic/precommittedで、Development / Admission結果を見て調整するparameterではありません。clip valueはnormalizer statistics identityとU1 artifact identityへbindします。

Normative amendment: `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u1-normalization-clip-amendment.md`。

## 14. U1 Reward contract

Reward semantic:

```text
universal_net_log_growth_reward_v1
reward_t = 100 * log(W_after / W_before)
```

`W_before/W_after`は同じrealized hybrid accounting pathの`BookState.portfolio_value`です。fee、spread、impact/slippage、funding、borrow costなどはwealthへ一度だけ反映され、U1 reward側で二重にpenalizeしません。

valid trajectoryではnormal path、execution-cost path、funding path、borrow path、partial-fill pathのいずれでも:

```text
sum(reward_t) / 100 == log(final_wealth / initial_wealth)
```

がfloating tolerance内で成立しなければなりません。non-positive / NaN / inf wealthはfail closedです。U1追加reward shapingはありません。

## 15. U1 artifact identityとatomic materialization

U1は次の2 Artifactを同じgenerationとして扱います。

```text
u1-output/
  normalizer.json
  u1_contract.json
```

`u1_contract.json`は少なくとも次をbindします。

- U0 universe manifest identity
- U0 materialization identity
- U1 policy contract digest
- normalizer digest / provenance / knowledge cutoff / fixed clip
- observation schema digest
- ordered policy-state layout
- runtime config digest
- execution policy digest
- pretrade / portfolio risk digests
- `production_status = "NO-GO"`

publicationはcanonical UTF-8 JSON + trailing newline、staging、file/directory fsync、directory単位publishを使います。既存outputはexact 2 filenamesかつbyte-identicalの場合だけidempotent successで、drift、extra file、partial publicationはfail closedです。

## 16. U1 Quality Gate

U1をimplementation-completeと表現するには、少なくとも次をexact final HEADで確認します。

- one-symbol/action/clock/horizon/reset contract
- future causality、symbol/price-unit identity invariance
- missing placeholder invariance
- request/pending/risk/realized/order-pending state separation
- Train-only source firewall、published artifact identity、source-event dedup、equal-symbol moments、knowledge cutoff
- fixed ±10 normalizer clip
- realized after-cost wealth rewardのtelescoping
- external truncationでfree liquidationなし
- U1 artifactsのcanonical/atomic/idempotent publication
- existing normalizer/router/accounting/Causal Alpha compatibility
- targeted tests、full suite、Ruff、format、MyPy、import architecture、build
- self-reviewと独立/falsification review
- exact final HEAD CI

テストがGreenであることだけではQuality Gateを満たしません。

## 17. U1が証明しないこと

U1の成功から次を推論してはいけません。

- Base RLが実際に学習できる
- unseen symbolへ経済的にzero-shot generalizeできる
- 期待収益が正である
- Development / AdmissionでSelection gateを通る
- 実市場executionを完全再現する
- Production投入してよい

U1 Production statusは引き続き**NO-GO**です。

## 18. U2 handoff

U2はSection 9のreal production-candidate U0/U1 generationをfreezeした後にだけ開始します。

U2 Base RLは:

1. U0 Train symbolsだけから`UniversalTradeRLFitPurpose.RL_TRAINING` provenanceを作る
2. frozen U1 contract digestを`UniversalTradeRLRunStage.BASE_TRAINING` identityへbindする
3. `EpisodeRoutedSingleInstrumentEnv`でTrain symbolをepisode単位にrouteする
4. U1 V1では`instrument_context_provider=None`、`v4_context_provider=None`を維持する
5. Developmentはevaluation-only、Admissionはsealedのままにする

U2 Selectionではsymbol-OOSだけでなくtime-OOSを別軸として評価します。同じ歴史期間を別銘柄で見ているだけではfuture-regime generalizationの証拠にならないため、最も強いgeneralization cellは**unseen symbol × unseen future time**です。
