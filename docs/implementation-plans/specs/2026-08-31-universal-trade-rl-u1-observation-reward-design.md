# Universal Trade RL U1: Observation / Action / Reward Contract Design

作成日: 2026-08-31 (JST)

> **Production status: NO-GO**
>
> 本specはU1の設計契約を定義する。RL学習、収益性、zero-shot性能、Admission合格、Production投入を証明しない。

## 1. 背景とU0との関係

Universal Trade RL U0は、Train / Development / Admission / ExcludedのUniverse、source-data identity、Train-only fit provenance、Admission access firewallを固定する。

U1はその境界の内側で、**どの銘柄にも同じ意味で適用できるObservation / Action / Reward契約**を定義する。

PR #426本文では「次のU1」としてproduction-candidate universeの実materializationも言及される一方、U0 implementation planではU1をobservations/rewardとして扱っている。このspecでは役割を次のように固定する。

1. **U1 prerequisite gate**: illustrative exampleではなく、実研究対象のproduction-candidate source catalogとTrain / Development / Admission / Excluded role configがU0でmaterializeされ、`universe.json` / `identity.json` digestがfreezeされていること。
2. **U1本体**: frozen U0 Universeを入力として、Observation / Action / Reward / normalization / episode semanticsを定義すること。
3. U1 implementationはsynthetic fixtureで契約テスト可能だが、実Universe digestが存在しない状態ではU2 Base RL trainingを開始しない。

U1は既存Causal Alpha V9/V10/V11の経済ロジックを変更しない。既存Universal U3-U6経路を上書きしない。新しいzero-shot-first経路として追加する。

## 2. Objective

複数のTrain銘柄から1つのshared policyを将来学習でき、未学習銘柄でも同じ意味の入力・出力・経済評価を持つように、次を固定する。

- symbol-independentかつcausalなObservation schema
- one-symbol deploymentに対応する1次元target-exposure Action semantics
- after-cost wealthと厳密に整合するpure net-log-growth Reward
- U0 Train symbolsだけでfitするUniversal normalization
- training horizonを市場終端と混同しないepisode/truncation semantics
- すべてをU0 Universe digestとcontent-addressed identityにbindする契約

## 3. Non-goals

U1では以下を実装・選択・主張しない。

- PPO / SAC / TD3等のRL algorithm比較
- neural network architectureの選択や大型化
- Behavior Cloning / teacher policy
- Causal Alphaをteacher/anchorとして使うこと
- V12 label/horizon研究
- 1分足execution fidelity研究
- 複数銘柄へ同時にcapital allocationするportfolio policy
- Admissionの開封
- transfer learning / fine-tuning
- profitability / Sharpe / production readinessの主張
- Development結果を見てfeature/reward係数を事後調整すること

## 4. 設計原則

### 4.1 One deployment candidate = one concrete symbol

推論単位は常に1 symbol、1 fixed capital budget、1 target-exposure controlとする。

```text
one policy application
  = one concrete symbol
  = one current account/execution state
  = one scalar target-exposure action
```

shared policyは複数Train symbolsから学習可能だが、一つのactionでBTC/ETH/SOLを同時配分しない。

### 4.2 Strategy priorをObservationから外す

現行`ObservationBuilder`は市場・execution状態に加え、`TrendTargets`、alpha、shadow baseline、hybrid-vs-shadow差分等もPolicy inputへ含められる。

U1 V1では以下を禁止する。

- `TrendTargets.fast/base/slow`
- alpha provider output
- factor basis / factor tilt prior
- shadow book weight/state
- hybrid - shadow relative state
- baseline underperformance state

V10まで人間が定義していたentry/exit ownershipやbaseline policyをshared RLへ暗黙に持ち込まず、「市場状態 + 実現した自分の状態 -> target exposure」を学習対象に戻す。

### 4.3 Risk / executionは環境のhard contract

RLは「どれだけlong/short exposureを要求するか」を出す。

Risk / execution layerは、leverage、margin、position limit、tradability、liquidity、minimum order、latency、partial fill等を現実的な制約として適用する。

**Policy request、Risk projection、realized positionは3つの別状態として扱う。**

### 4.4 Rewardはwealthだけを最適化する

fee、spread、impact、funding、borrow等がwealthへ既に反映される場合、追加のturnover/cost penaltyをRewardへ重ねない。

DrawdownもU1 Rewardへ加えず、破産防止・leverage・margin・position制約はRisk / termination layerで扱う。

## 5. Reuseする既存契約

U1は以下を再利用する。

- `MarketDataset`
- `FeatureSpec` / `FeatureKind`
- `SequenceObservationBuilder`
- feature availability / staleness / missing reason
- `ObservationExecutionState`
- `PendingOrderObservationState`
- `BookState`
- `ActionMode.TARGET_WEIGHT`の基本概念
- PreTradeRisk / portfolio risk / execution simulator
- after-cost accounting
- `RewardConfig.is_pure_net_log_growth()`が表すpure growth意味論
- U0 Train-only fit provenance

ただし既存`baseline_residual_observation_v5`をそのままU1 Policy contractには使わない。U1専用schemaを追加する。

## 6. Architecture

```text
U0 Frozen Universe
       |
       +-- Train symbols --------------------------+
       |                                           |
       |                                  Train-only fit
       |                                           |
       |                              Universal Normalizer
       |                                           |
       v                                           v
MarketDataset -> UniversalObservationBuilder -> UniversalObservationV1
                       ^
                       |
          realized Book / Execution / Pending state
                       |
                       +-----------------------------+
                                                     |
                                                     v
                                                Shared Policy
                                                     |
                                              scalar action [-1, 1]
                                                     |
                                                     v
                                      NormalizedTargetExposureV1
                                                     |
                                                     v
                                         PreTradeRisk / Execution
                                                     |
                                                     v
                                            realized BookState
                                                     |
                                                     v
                                    100 * after-cost net log growth
```

責務境界:

- `UniversalObservationBuilder`: causalでsymbol-independentなPolicy inputだけを生成する。
- `UniversalObservationNormalizer`: U0 Train-only pooled statisticsをfreezeする。
- `NormalizedTargetExposureV1`: scalar actionを固定semanticの**pre-risk policy request**へ変換する。
- Risk layer: pre-risk requestを**post-risk projected target**へ変換する。
- Execution/accounting: projected targetからfill/cost/wealthを計算する。
- Reward: accounting結果を再解釈せずnet log growthへ写像する。

## 7. Observation Contract

### 7.1 Schema

新schema名:

```text
universal_trade_observation_v1
```

U1 environmentは`dataset.n_symbols == 1`を必須とする。shared trainingは複数のsingle-symbol environmentを使用する。

Observationはstructured Dictとし、次の2 planeに分ける。

1. Market sequence plane
2. Endogenous policy-state plane

symbol name / symbol index / exchange-specific IDをtensorへ入れない。

### 7.2 Market sequence plane

既存`SequenceObservationBuilder`のcausal source-index semanticsを再利用する。

初期windowは既存Universal U6と同じ時間範囲を採用する。

| timeframe | length | history |
| --- | ---: | ---: |
| 15m | 96 | 24h |
| 1h | 168 | 7d |
| 4h | 120 | 20d |
| 1d | 60 | 60d |

各timeframeについて以下を公開する。

```text
sequence_<tf>_values
sequence_<tf>_available
sequence_<tf>_staleness
```

`source_index <= decision_index`を全要素で保証する。future row backfillは禁止する。

### 7.3 U1 V1 Feature semantics

U1 V1は`FeatureKind`全種を自動許可しない。次のsemantic groupに限定する。

**Return / candle geometry**

- `LOG_RETURN`
- `BODY_RETURN`
- `HIGH_LOW_RANGE`
- `GAP_RETURN`

**Volatility**

- `REALIZED_VOLATILITY`
- `DOWNSIDE_VOLATILITY`
- `UPSIDE_VOLATILITY`
- `VOLATILITY_OF_VOLATILITY`
- `ATR_PCT`
- `ATR_CHANGE`

**Trend geometry**

- `EMA_DISTANCE`
- `EMA_SLOPE`
- `LINEAR_REGRESSION_SLOPE`
- `TREND_R2`

**Liquidity / activity**

- `VOLUME_ZSCORE`
- `VOLUME_LOG_CHANGE`
- `RELATIVE_VOLUME`

**Derivatives state**

- `FUNDING_BPS`
- `FUNDING_CHANGE`
- `FUNDING_ZSCORE`

具体的な`FeatureSpec`順序、lookback、normalization window、timeframe割当はU1 model configの一部として事前登録し、feature contract digestへbindする。Development / Admission結果を見て変更した場合は新generationとする。

U1 V1で明示的に禁止するFeatureKind:

- `RELATIVE_RETURN_TO_BTC`
- `ROLLING_CORRELATION_TO_BTC`
- `ROLLING_BETA_TO_BTC`
- `CROSS_SECTIONAL_MOMENTUM_RANK`
- `CROSS_ASSET_DISPERSION`

理由は、target symbolによってreference relationshipが非対称になり、one-symbol zero-shot契約の初期検証を不必要に複雑化するためである。

raw absolute OHLC price、raw nominal volume、raw position quantity、contract multiplier、symbol IDをPolicy tensorへ直接入れない。

### 7.4 Missingness

missing featureを単純な0として意味付けしない。

```text
value = 0, available = 1
```

と

```text
value = 0, available = 0
```

は別状態である。

stalenessも維持し、Funding等のevent-sparse featureを「0 funding」と「未更新/未取得」で区別する。

### 7.5 Endogenous policy-state plane

既存`ObservationExecutionState`、`PendingOrderObservationState`、`BookState`およびdecision/execution traceから、次のscale-freeまたは共通単位の状態だけを生成する。

```text
current_weight
policy_requested_weight
risk_projected_weight
previous_action
pending_target_weight
fill_ratio
unfilled_turnover_ratio
participation_ratio
execution_cost_rate
position_age_hours
pending_notional_ratio
pending_order_type_code
pending_order_status_code
pending_order_age_hours
pending_order_eligible_delay_hours
pending_order_triggered
pending_order_expiry_distance_hours
asset_active
tradable
borrow_available
borrow_rate
mark_index_basis
current_drawdown
current_gross_exposure
current_net_exposure
cash_weight
risk_scale
margin_utilization
```

意味を固定する。

- `policy_requested_weight`: `a_t * policy_weight_scale`で得た**PreTradeRisk適用前**のPolicy request。
- `risk_projected_weight`: Risk layer適用後、Executionへ渡されたtarget。reject/flatten/project結果を反映する。
- `pending_target_weight`: order latency / partial-fill carry等によりExecution lifecycle内で未完了のtarget。
- `current_weight`: **実現したeffective `BookState.weights`**。

同一fieldへpre-risk/post-risk値を混在させない。既存`ObservationExecutionState.requested_weights`の現在の意味が上記どちらに対応するかを実装前に監査し、曖昧ならU1専用trace fieldを追加する。名前を合わせるためだけに意味を変更しない。

その他:

- `execution_cost_rate`はnominal currency costではなく、直前stepのcost / reference equityとして表す。既存fieldがrateでない場合はU1 builderでrateへ変換する。
- position/order ageはhoursという共通物理単位で表し、その後Train-only normalizerへ渡す。
- `borrow_rate`、`mark_index_basis`のtransform式はschema identityへ固定する。
- current portfolio value、cash nominal amount、peak nominal value、quantity、mark price absolute valueは公開しない。

### 7.6 Forbidden endogenous inputs

以下はU1 V1 Policy inputへ入れない。

- shadow book state
- baseline strategy target
- hybrid-vs-shadow wealth gap
- baseline underperformance penalty state
- Causal Alpha score / confidence / target
- manually latched slow/fast ownership state
- remaining episode fraction
- Admission role/name
- dataset ID / artifact digestそのもの

## 8. Action Contract

新semantic identity:

```text
normalized_target_exposure_v1
```

Action shape:

```text
(1,)
```

Policy output:

```text
a_t in [-1, +1]
```

Pre-risk Policy request:

```text
policy_requested_weight = a_t * policy_weight_scale
```

`policy_weight_scale`はstaticかつprecommittedなmodel configであり、U0 Train/Development/Admissionで共通とする。Development結果からfitしない。U1 V1 defaultは`1.0`とし、変更時は別config digestとする。

重要: `policy_weight_scale`は現在のdynamic risk capではない。Actionの意味を状態依存にしない。

Risk layerは`policy_requested_weight`を`risk_projected_weight`へproject/reject/flattenできる。Executionはその後realized `current_weight`を更新する。

例:

```text
policy action          = +0.80
policy_requested_weight= +0.80
risk_projected_weight  = +0.35
realized current_weight= +0.22
```

次Observationでは4段階を区別して返す。

U1 V1ではbuy/sell/hold gate、manual confirmation、slow ownership、neutral expiry等をAction semanticへ入れない。

## 9. Reward Contract

新semantic identity:

```text
universal_net_log_growth_reward_v1
```

各decision transitionについて:

```text
reward_t = 100 * log(W_after / W_before)
```

ここで`W_before` / `W_after`は同一realized accounting pathのportfolio valueである。

`W_after`には実行モデルが計上する以下を含む。

- trading PnL
- fee
- spread
- market impact / slippage
- funding PnL
- borrow cost
- その他既存accounting contractでnet wealthへ入るcost

追加Reward shapingはU1 V1で0固定とする。

- excess/baseline component = 0
- drawdown reward penalty = 0
- turnover penalty = 0
- projection penalty = 0
- terminal equity bonus/penalty = 0
- margin penalty = 0 in Reward

margin/insolvency safetyはRisk / termination contractで扱う。

### 9.1 Reward reconciliation oracle

正しいepisodeでは:

```text
sum(reward_t) / 100
  == log(final_wealth / initial_wealth)
```

がfloating-point tolerance内で成立しなければならない。

forced cost、funding、partial fillを含む場合も同一oracleを使う。

### 9.2 Non-positive wealth

`W_before <= 0`または`W_after <= 0`はlog rewardとして定義しない。

小さい正値へsilent clipしてrewardを続行することは禁止する。Risk / accounting / termination contract違反としてfail closedし、そのepisodeを有効な学習evidenceに含めない。

## 10. Universal Normalization

### 10.1 Principle

既存`ObservationNormalizer`はexplicit training rangeとdataset identityを持つが、U1では複数Train symbolsを1つのshared policyへ入れるため、Universal pooled normalizerを定義する。

新artifact semantic:

```text
universal_observation_normalizer_v1
```

### 10.2 Fit scope

fit sourceはU0 `Train` symbolsだけ。

Development / Admission / Excludedはfitに使用しない。

normalizer build前にU0 Train-only provenanceを検証し、その後source datasetへアクセスする。

### 10.3 Equal-symbol weighting

単純な全row連結では、historyが長いsymbolが統計を支配するため禁止する。

各continuous feature channelについて:

1. 各Train symbol内でavailableなeligible sampleからfirst/second momentを計算する。
2. symbolごとのmomentをequal weightで集約する。
3. global mean/scaleを全symbolへ共通適用する。

したがってhistoryが長いsymbolほどnormalizerへ大きなvoteを持つことを避ける。

### 10.4 Availability-aware fit

`available == false`のplaceholder valueをmean/std fitへ入れない。

availability / missing reason / categorical code / boolean maskはpassthroughとし、continuous featureと同じstandardizationをしない。

### 10.5 Identity binding

Normalizer artifactは最低限以下をbindする。

- U0 universe manifest digest
- U0 Train fit provenance digest
- ordered Train symbols
- symbolごとのTrain source dataset digest
- U1 observation contract digest
- ordered feature contract digest
- sequence window contract
- endogenous state layout digest
- equal-symbol weighting mode
- mean / scale / clip
- knowledge cutoff

ここでU0 universe digestはDevelopment / Admission source identityも含むため、**Development / Admission sourceが変われば同一generationのnormalizer artifactを再利用しない**。source driftはU0 validationで先にfail closedする。

一方、normalizerの数値統計そのものはTrain-onlyである。別Universe generationをsyntheticに作り、Train sourceを完全に同一、Development / Admissionだけを変更した場合、`mean` / `scale`は一致しなければならない。ただしU0 universe digestが異なるためnormalizer artifact digest全体は異なってよい。

## 11. Episode / Truncation Semantics

U1をcontinuing control problemとして扱う。

training上のfinite sampled trajectoryは「市場の終端」ではない。

U1 V1では:

- remaining episode fractionをObservationへ入れない
- sampled horizon到達はGymnasium `truncated=True` semanticとする
- market/risk failure等だけを`terminated=True` semanticとする
- training truncation時に無料でpositionをliquidateしない
- `liquidate_on_end=False`
- U1 V1 initial stateは`cash`のみ

U2 algorithmはtruncationをterminal-zero-valueとして扱ってはならず、final observationからbootstrap可能でなければならない。この点はU2実装時の必須Integration Testとする。

U1 V1のtrajectory horizonは既存30日/720hを初期contractとして再利用してよいが、horizonはmodel configへbindし、Developmentを見て同一generation内で変更しない。

## 12. Shared Training Symbol Sampling Contract

U2で利用するsymbol samplerはU1 handoffとして次を前提にする。

1. U0 Train symbolsからsymbolをuniformに選ぶ。
2. 選ばれた1 symbol内でeligible causal startを選ぶ。
3. 1 environmentはepisode中そのsymbolだけを扱う。
4. row数が多いsymbolが単純にsampling probabilityを増やさない。

これはportfolio allocationではない。複数parallel envが別symbolを同時に担当することは許可する。

## 13. Identity / Artifact Contract

U1は次をcanonical artifactとしてmaterializeできる設計とする。

```text
u1_contract.json
normalizer.json
```

`u1_contract.json`は最低限以下を含む。

- schema version
- U0 universe digest
- U0 identity digest
- feature contract digest
- observation contract digest
- action contract digest
- reward contract digest
- episode/truncation contract
- policy_weight_scale
- source schema compatibility evidence
- production_status = NO-GO

U1 contract digestはU2 Base RL model config / checkpoint identityへ必須bindする。

## 14. Invariants

1. Policy tensorにsymbol IDが存在しない。
2. U1 environmentはexactly one concrete symbolを扱う。
3. Action semanticは全symbolで同一である。
4. Reward semanticは全symbolで同一である。
5. Observationはdecision time以下の情報だけから生成される。
6. Development / Admissionはnormalizationを含むfitへ入らない。
7. `policy_requested_weight` / `risk_projected_weight` / `pending_target_weight` / `current_weight`を混同しない。
8. trading costをRewardで二重計上しない。
9. training trajectory endを市場終端としてPolicyへ知らせない。
10. baseline/shadow/Causal Alpha manual priorをU1 V1 Policy inputへ入れない。
11. missingnessはvalueとavailabilityを分離する。
12. 同一feature contractなら全source symbolで同一input shapeになる。
13. U1 contract変更は新digest/new generationになる。

## 15. Failure Modes

### Critical

- future mutationが現在Observationを変える
- Admission dataでnormalizerをfitする
- symbol name/IDがPolicy tensorへ入る
- Reward sumとwealth変化が一致しない
- fee/spread/impact等のcost二重計上
- pre-risk request / post-risk target / realized positionを同一fieldへ混在させる
- training truncationで無料liquidationまたはreward bonusが発生する
- non-positive wealthをsilent clipして学習継続する

### High

- per-symbol normalizerを使いzero-shot時だけre-fitする
- row-count weighted poolingで特定symbolがnormalizerを支配する
- funding unavailableをfunding=0と同一視する
- baseline/shadow stateがU1 inputへ混入する
- cross-asset BTC reference featureによるtarget-dependent representation asymmetry
- raw nominal price/quantity/cashをPolicy tensorへ入れる
- episode remaining-time依存policyを学習可能にする

### Medium

- stale featureをavailabilityなしで利用する
- action saturationを診断できない
- order latency/partial fill stateをPolicyから隠してPOMDPを悪化させる
- source symbols間でfeature order/schemaがずれる

## 16. Test Oracle

U1の正しさは単なる`env.step()`成功では判定しない。

必須Oracle:

- exact observation component values
- source indices <= decision index
- policy request -> risk projection -> pending target -> realized weightのstate transition
- order state transition
- `BookState.portfolio_value`
- execution cost/funding/borrow side effects
- Reward-to-wealth reconciliation
- normalizer provenance / mean / scale / digest
- U0 source drift rejection
- exact schema/shape/dtype
- forbidden manual prior field absence

## 17. Required Test Layers

### Unit

- observation schema/layout
- feature allowlist/prohibited kinds
- action scaling/parsing
- pre-risk/post-risk/realized state capture
- policy-state ratio conversion
- Reward formula
- universal moment aggregation
- artifact digest/canonicalization

### Property / Falsification

- future mutation invariance
- symbol rename invariance
- price-unit scale invariance
- quantity/contract scaling invariance where economic path is equivalent
- Train-only statistics invariance across synthetic Universe generations with identical Train sources
- Development/Admission source drift fail-closed behavior
- missing-value vs true-zero distinction
- reward telescoping identity

### Integration

- Policy action -> Risk projection -> partial fill -> next Observation
- flat market round trip -> exact after-cost loss
- funding/borrow path -> exact net wealth/reward
- pending order lifecycle -> Observation state
- truncation without forced liquidation/free reward
- U0 phase/provenance -> U1 normalizer materialization

### Compatibility

- existing Causal Alpha V9/V10/V11 behavior unchanged
- existing Universal U3-U6 config behavior unchanged unless explicitly routed to U1 schema
- existing `baseline_residual_observation_v5` remains readable

### Static / Architecture

- Ruff
- format check
- MyPy for affected modules
- import-linter / architecture contract
- package build
- full test suite

## 18. Falsification Tests

### 18.1 Symbol rename

同じnumeric datasetを`BTCUSDT`から`FOOUSDT`へrenameする。

U1 Policy tensorは数値一致しなければならない。Artifact identity自体はsource identity差分により変化してよい。

### 18.2 Price-scale transformation

OHLC priceを一律`x1000`し、contract/quantity側を経済的に等価に変換したsynthetic marketを作る。

return/range/volatilityベースMarket planeとportfolio-weight policy stateは同一または定義されたtolerance内で一致しなければならない。

### 18.3 Future mutation

index `t+1`以降のprice/volume/fundingを任意値へ変更する。

`Observation(t)`とU1 normalizer transform結果はbitwiseまたは指定float tolerance内で不変でなければならない。

### 18.4 Development / Admission poisoning

同一U0 generationのDevelopment / Admission sourceを改変した場合、U0 source identity validationがnormalizer fitより先にfail closedしなければならない。

さらに、2つのsynthetic Universe generationでTrain sourcesを完全に同一、Development / Admissionだけを変更した場合:

- Universal normalizer `mean` / `scale`は一致する。
- U0 universe digestは異なる。
- したがってnormalizer artifact digest全体は異なってよい。

これにより「fit統計はTrain-only」と「artifactは完全Universe identityへbind」の両方を検証する。

### 18.5 Cost reconciliation

価格不変marketで`flat -> long -> flat`を実行する。

final wealth lossはexecution/accountingが定義するfee/spread/impact/funding/borrowだけと一致し、追加Reward penaltyによる差分がないこと。

### 18.6 Request / projection / fill mismatch

```text
policy action           = +0.80
policy_requested_weight = +0.80
risk_projected_weight   = +0.40
pending_target_weight   = +0.40
realized current_weight = +0.25
```

を作り、次Observationが各値を別fieldで保持することを検証する。

## 19. Acceptance Criteria

1. U1 contractはfrozen U0 universe digestなしではresearch generationとしてmaterializeできない。
2. U1 Policy tensorにsymbol ID、dataset ID文字列表現、manual Causal Alpha/baseline targetが存在しない。
3. U1 environmentは1 symbol以外をfail closedする。
4. Market sequenceは15m/1h/4h/1dのcausal windowsを提供する。
5. 全sequence source indexがdecision index以下である。
6. prohibited cross-asset feature kindsをU1 V1が拒否する。
7. raw nominal price/quantity/cashをPolicy tensorへ公開しない。
8. missing value / availability / stalenessを分離する。
9. `policy_requested_weight`、`risk_projected_weight`、`pending_target_weight`、`current_weight`の意味がschemaで分離される。
10. `current_weight`が実際のRisk/Execution結果と一致する。
11. scalar action `[-1,1]`がstatic `policy_weight_scale`を介して同一意味のpre-risk exposure requestへ変換される。
12. dynamic risk capによってAction semanticそのものを変更しない。
13. Rewardが`100 * net_log_return`のみで構成される。
14. `sum(reward)/100 == log(final/initial wealth)`が全valid integration episodeで成立する。
15. fee/spread/impact/funding/borrowがRewardで二重計上されない。
16. Universal normalizerはU0 Train symbolsだけからfitする。
17. Universal normalizerはequal-symbol weightingを使う。
18. unavailable placeholderをcontinuous statisticsへfitしない。
19. Development / Admission source driftはnormalizer fitより先にfail closedする。
20. Train sourceが同じsynthetic generationsではDevelopment / Admissionが異なってもnormalizer `mean` / `scale`が一致する。
21. episode remaining fractionをPolicyへ公開しない。
22. sampled horizonはtruncationであり、無料liquidationやterminal rewardを作らない。
23. existing Causal Alpha / Universal legacy pathのcompatibility regressionがない。
24. U1 implementationのtargeted tests、property tests、integration tests、static checks、full suite、buildが必要なQuality Gateを満たす。

## 20. Quality Gate

U1を「完了」と呼ぶためには以下をすべて満たす。

- 全Acceptance Criteriaの観測可能な証拠がある。
- Critical / High Failure Modesに対応するnegative/falsification testがある。
- Reward reconciliationを複数execution pathで確認している。
- U0 Train-only provenanceとのintegration testがある。
- Feature/action/reward/normalizer identityがcontent-addressedである。
- Targeted Unit / Property / Integration testsがgreen。
- Ruff / format / MyPy / import architecture checksがgreen。
- Full test suiteにfeature-only unexplained failureがない。
- package buildが成功する。
- final HEADに対するCI Required Checksを確認する。
- diff self-reviewとindependent/falsification reviewを行う。
- Admissionは未開封のままである。
- Production statusはNO-GOのままである。

## 21. Implementation Boundary

実装時は既存`ResidualMarketEnv`を全面置換しない。

推奨境界:

- 新規U1専用Observation contract/builderを追加する。
- 既存`SequenceObservationBuilder`、execution/accounting/risk state dataclassをreuseする。
- legacy `ObservationBuilder`からmanual prior fieldを削除しない。既存path互換性のため残す。
- U1 pathだけが新schemaへroutingされる。
- Universal normalizerは既存single-dataset normalizerを無理に多目的化せず、U1用の明確なTrain-only pooled artifactとして分離する。
- target exposureは既存Action/PreTradeRisk/Execution契約へadapterで接続し、既存action semanticsを黙って変更しない。
- `environment.py`へ多数のU1責務を直接追加せず、builder / contract / adapterへ分離する。

具体的なファイル名はimplementation plan作成時に、U0がmainへ入った最終HEADとimport architectureを再確認して確定する。

## 22. U2 Handoff

U2 Base RL trainingへ進める条件:

1. production-candidate U0 Universeがfreeze済み。
2. U1 contract artifactがfreeze済み。
3. U1 Universal normalizerがTrain-only provenance付きでfreeze済み。
4. U1 falsification / integration Quality Gateが通過。
5. Admissionは未開封。

U2では初めてRL algorithm / architecture / seed robustness / checkpoint selectionを扱う。U2の結果が良くても、AdmissionはU0 authorization contractを満たすまで開かない。

## 23. Spec Self-Review Record

- **Placeholder scan:** TBD/TODOはない。具体的ファイル名だけはU0最終HEAD確認後にimplementation planで決めるとscope上明示している。
- **Internal consistency:** one-symbol deployment、Train-only statistics、complete Universe identity binding、pure net-log reward、risk/execution separationは全sectionで一致する。
- **Scope check:** Observation / Action / Reward / normalization / episode semanticsに限定し、RL algorithm/transfer/Admission/productionを除外した。
- **Ambiguity check:** pre-risk request / post-risk projection / pending target / realized weightを別fieldへ固定した。Development/Admission変更時のnormalizer数値統計とartifact digestの違いも明文化した。
