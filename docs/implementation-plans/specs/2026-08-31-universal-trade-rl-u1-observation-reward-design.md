# Universal Trade RL U1: Observation / Action / Reward Contract Design

作成日: 2026-08-31 (JST)

> **Production status: NO-GO**
>
> 本specはU1の研究・実装契約を定義する。RL学習、zero-shot性能、収益性、Admission合格、Production投入は証明しない。

## 1. U0との関係

Universal Trade RL U0は、Train / Development / Admission / Excludedの役割、source-data identity、Train-only fit provenance、Admission access firewallを固定する。

U1はその凍結済みUniverseを入力として、**どの銘柄にも同じ意味で適用できるObservation / Action / Reward / normalization / episode contract**を固定する。

U1開始前提は次の2段階に分ける。

1. **U1 contract implementation**はsynthetic fixtureとU0 example manifestで検証可能。
2. **U2 Base RL training開始前**には、illustrative exampleではなく実production-candidate source catalog / role configをU0でmaterializeし、`universe.json` / `identity.json` digestをfreezeする。

U1は既存Causal Alpha V9/V10/V11やUniversal U3-U6を置換しない。zero-shot-firstの新経路として追加する。

## 2. Objective

- symbol-independentかつcausalなPolicy Observationを固定する。
- one-symbol deployment用の1次元target-exposure Actionを固定する。
- after-cost realized wealthと厳密に一致するRewardを固定する。
- U0 Train symbolsだけでfitするUniversal market normalizerを固定する。
- training horizonを市場終端と混同しないcontinuing/truncation semanticsを固定する。
- U1 contract / normalizer / runtime identityをcontent-addressed artifactとして再現可能にする。

## 3. Non-goals

U1では以下を行わない。

- PPO / SAC / TD3等のalgorithm比較・学習
- network architectureの選択・大型化
- Behavior Cloning / teacher policy
- Causal Alphaをteacher/anchorとして利用
- V12 label/horizon研究
- 1分足execution fidelity研究
- multi-asset portfolio allocation policy
- Developmentでfeature/reward係数を事後fit
- Admission開封
- transfer / fine-tuning
- profitability / Sharpe / Production readinessの主張

## 4. 現行コード監査で確定した既存意味論

U1は既存`ResidualMarketEnv`のexecution/accountingを再利用する。コード監査により以下を確定する。

### 4.1 Risk-projected target

`ResidualMarketEnv.step()`は、

```text
decision.executed_hybrid_target
 -> risk_projector.project(...)
 -> hybrid_risk.weights
 -> execute_target(... target=hybrid_risk.weights)
```

の順でExecutionへ渡す。

さらにstep後の`ObservationExecutionState.requested_weights`は、

```text
requested_weights = hybrid_risk.weights
```

として保存される。

したがってU1では既存`ObservationExecutionState.requested_weights`を**risk_projected_weight**として再利用し、意味を変更しない。

### 4.2 Signal-delay pending targetとpending orderは別物

既存`_pending_hybrid_target`は`signal_delay_decisions`用のpending targetであり、partial-fill/order-latency stateではない。

U1 V1は因果的なnext-decision executionを固定するため、

```text
signal_delay_decisions = 1
```

を要求する。

したがって次の意味を固定する。

- `policy_requested_weight`: 今回Policyが提出したpre-risk target。
- `pending_target_weight`: signal-delay queueで次decision executionを待つtarget。
- `pending_target_active`: pending targetが存在するかを表すmask。`0.0` targetと「pendingなし」を区別する。
- `risk_projected_weight`: 今回実際にExecutionへ渡されたpost-risk target。signal delayがあるため、通常は直前decisionでpendingだったtargetに対応する。
- `current_weight`: Execution後のrealized `BookState.weights`。

partial fill / order latency / order expiry等は既存`PendingOrderObservationState`で別途観測する。

### 4.3 Execution cost observation

既存`EnvironmentExecutionCoordinator.execution_observation_state()`は、

```text
execution_cost = result.cost_by_symbol / initial_capital
```

を保存する。

U1の`execution_cost_rate`はこの**cost / initial_capital**意味論をそのまま使用する。nominal costへ戻さない。

## 5. Architecture

U1は新しいSimulatorを作らない。

```text
U0 Frozen Universe
        |
        +---- Train-only market statistics ----+
        |                                      |
        v                                      v
ResidualMarketEnv                         Universal Market Normalizer
(single symbol, maintained execution)          |
        |                                      |
        +---- runtime snapshot ----------------+
        |                                      |
        v                                      v
              UniversalTradeMarketEnv wrapper
                         |
                  U1 Observation
                         |
                    Shared Policy
                         |
                 scalar action [-1,1]
                         |
             policy_requested_weight
                         |
                signal-delay queue
                         |
                 PreTradeRisk
                         |
              risk_projected_weight
                         |
              maintained Execution
                         |
                  current_weight
                         |
               realized BookState
                         |
             100 * net log growth
```

責務:

- `ResidualMarketEnv`: risk / execution / accountingの唯一のauthority。
- U1 runtime snapshot: named read-only stateだけを公開する。
- U1 observation builder: manual strategy priorを受け取らない。
- U1 normalizer: market sequence continuous channelsだけをTrain-only equal-symbol fitする。
- endogenous policy state: dimensionless valueまたはversioned deterministic transform。policy-dependent state distributionからfitしない。
- U1 wrapper: action/observation/reward semanticを固定し、base environmentの経済ロジックを変更しない。

## 6. Observation Contract

Schema:

```text
universal_trade_observation_v1
```

U1 environmentは`dataset.n_symbols == 1`を必須とする。

Observationは次の2 planeから構成する。

1. Market sequence plane
2. Endogenous policy-state plane

symbol name / symbol index / dataset ID / artifact digestをtensorへ入れない。

### 6.1 Market sequence windows

固定window:

| timeframe | length | history |
| --- | ---: | ---: |
| 15m | 96 | 24h |
| 1h | 168 | 7d |
| 4h | 120 | 20d |
| 1d | 60 | 60d |

各timeframeで公開:

```text
sequence_<tf>_values
sequence_<tf>_available
sequence_<tf>_staleness
```

既存`SequenceObservationBuilder`のcausal source-index semanticsを再利用し、全source rowで

```text
source_index <= decision_index
```

を保証する。future backfillは禁止する。

### 6.2 U1 V1 allowed FeatureKinds

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

U1 V1では明示的に禁止:

- `RELATIVE_RETURN_TO_BTC`
- `ROLLING_CORRELATION_TO_BTC`
- `ROLLING_BETA_TO_BTC`
- `CROSS_SECTIONAL_MOMENTUM_RANK`
- `CROSS_ASSET_DISPERSION`

FeatureSpecの順序、lookback、native timeframe、rolling normalization設定はprecommitted policy contractへbindする。

raw absolute OHLC、raw nominal volume、raw nominal cash、raw quantity、contract multiplier、symbol IDはPolicy tensorへ直接入れない。

### 6.3 Missingness

```text
value = 0, available = 1
```

と

```text
value = 0, available = 0
```

は別状態である。stalenessも別channelとして維持する。

### 6.4 Endogenous policy state

固定順序で最低限次を公開する。

```text
policy_requested_weight
pending_target_weight
pending_target_active
risk_projected_weight
current_weight
previous_action
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

`PendingOrderObservationState`はorder lifecycleを表し、signal-delay `pending_target_*`とは別責務である。

### 6.5 Deterministic state transforms

Endogenous stateはTrain rollout分布へfitしない。理由は、normalization state自体が探索policy/action distributionへ依存するとU1 contractがpolicy依存になるためである。

例:

```text
position_age_days = log1p(position_age_hours / 24)
pending_order_age_days = log1p(pending_order_age_hours / 24)
mark_index_basis_scaled = tanh(100 * mark_index_basis)
borrow_rate_scaled = tanh(borrow_rate)
```

weights / ratios / masksは元のdimensionless semanticを維持する。transform式とfield順序はstate-layout digestへbindする。

### 6.6 Forbidden policy inputs

- `TrendTargets.fast/base/slow`
- alpha provider output
- factor basis / factor tilt prior
- shadow book state
- hybrid-shadow差分
- baseline underperformance state
- manually latched ownership state
- remaining episode fraction
- role/name/ID文字列

## 7. Action Contract

Semantic identity:

```text
normalized_target_exposure_v1
```

Action shape `(1,)`、値域`[-1,+1]`。

```text
policy_requested_weight = action * policy_weight_scale
```

`policy_weight_scale`はstatic precommitted config。U1 V1 default `1.0`。dynamic risk capをAction semanticへ混ぜない。

U1 action parserはout-of-range / NaN / infをfail closedし、hidden clipしない。Gym/policy側がaction spaceを守る責務を持つ。

signal delay 1のため、action_tは`pending_target_weight`としてqueueされ、今回Executionされるtargetは原則として直前pending targetである。

## 8. Reward Contract

Semantic identity:

```text
universal_net_log_growth_reward_v1
```

各transition:

```text
reward_t = 100 * log(W_after / W_before)
```

`W_before` / `W_after`は同じrealized hybrid accounting pathの`BookState.portfolio_value`。

wealthには既存accountingが計上するtrading PnL、fee、spread、impact/slippage、funding、borrow cost等を含む。

U1 V1追加shapingは全て0:

- baseline/excess component
- drawdown reward penalty
- turnover penalty
- projection penalty
- terminal equity bonus/penalty
- margin reward penalty

valid episodeでは:

```text
sum(reward_t) / 100 == log(final_wealth / initial_wealth)
```

がfloating tolerance内で成立する。

`W_before <= 0`、`W_after <= 0`、NaN、infはfail closed。U1 reward層でepsilon clipしない。

## 9. Universal Market Normalization

Artifact semantic:

```text
universal_trade_sequence_normalizer_v1
```

U1でfitするのは**Market sequence continuous valuesだけ**。Endogenous stateはSection 6.5のdeterministic transformを使用する。

### 9.1 Fit firewall

source lookupより先に:

```text
UniversalTradeRLUniverseAccess.require_normalization_scope(train_symbols)
```

を通し、その後

```text
UniversalTradeRLFitPurpose.FEATURE_NORMALIZATION
```

のTrain-only provenanceを作る。

Development / Admission / Excludedを数値fitに使わない。

### 9.2 Equal-symbol weighting

row数の多いsymbolがnormalizerを支配しないよう、各channelでsymbol単位momentsをequal weight集約する。

symbol `s` のavailable sampleについて:

```text
mu_s = mean(x_s)
q_s  = mean(x_s^2)
```

Train symbols数を`S`として:

```text
mu = (1/S) * sum(mu_s)
q  = (1/S) * sum(q_s)
var = max(q - mu^2, 0)
scale = sqrt(var)
```

`scale <= epsilon`は`1.0`。

各symbolが同じ1 voteを持ち、row countそのものはsymbol間weightにならない。

### 9.3 Native unique source sampling

高timeframe値がbase clockで繰り返されることによる過剰weightを避けるため、normalizer fit sampleは各timeframeの**unique native source rows**を使う。

sampling semantic:

```text
native_unique_source_rows_v1
```

同一native source rowをoverlapping sequence window回数だけ重複countしない。

`available == false` sampleはmomentへ入れない。必要channelでavailable sampleが0ならfail closed。

### 9.4 Identity

Normalizerは最低限以下をbindする。

- U0 universe manifest digest
- Train normalization provenance digest
- ordered Train symbols / source digests
- U1 policy contract digest
- ordered feature/layout digest
- sequence windows
- sampling semantic `native_unique_source_rows_v1`
- equal-symbol weighting semantic
- mean / scale / clip / epsilon
- knowledge cutoff

別Universe generationでTrain source numeric samplesが完全同一、Development / Admissionだけが異なる場合:

- `statistics_digest`は同一
- artifact `digest`はU0 universe identity差により異なる

ことを要求する。

## 10. Episode / Reset Contract

U1はcontinuing control problemとして扱う。

固定条件:

```text
episode_hours = 720h initial contract
episode_boundary_mode = external_truncation
finite_horizon_observation = false
liquidate_on_end = false
signal_delay_decisions = 1
initial_state_modes = ("cash",)
```

cash-onlyの理由は、existing `baseline` / `stress` / `partial_fill` resetが`TrendStrategy` targetを利用し、`current_weight`経由でmanual priorを注入するためである。

sample horizon到達は`truncated=True`、market/risk failureのみ`terminated=True`。sample endで無料liquidationやterminal bonusを発生させない。

U1 V1 training wrapperはsampled resetとしてcash以外を受け入れない。explicit restore/serving stateはU2以降で別contractとして扱う。

## 11. Runtime Wrapper Contract

`UniversalTradeMarketEnv`は既存`ResidualMarketEnv`をwrapする。

constructorは以下をfail closed検証する。

- exactly one symbol
- `ActionMode.TARGET_WEIGHT`
- target count 1
- structured sequence observation enabled
- sequence windowsがU1 contractと一致
- pure net-log-growth reward config
- `signal_delay_decisions == 1`
- external truncation
- finite-horizon observation false
- `liquidate_on_end == false`
- sampled initial state exactly cash-only

step:

```text
parse scalar action
 -> policy_requested_weight
 -> W_before
 -> base_env.step([policy_requested_weight])
 -> W_after
 -> recompute U1 reward
 -> require base pure reward == U1 reward
 -> read named runtime snapshot
 -> build U1 observation
```

wrapperはRisk / Execution / Accountingを再実装しない。

## 12. Artifact / Identity Contract

U1はcanonical artifactとして:

```text
output-root/
  u1_contract.json
  normalizer.json
```

をmaterializeする。

`u1_contract.json`は最低限:

- schema version
- U0 universe manifest digest
- U0 materialization identity digest
- policy contract digest
- observation/action/reward schema/digest
- state-layout digest
- normalizer digest / provenance digest
- episode/reset/signal-delay contract
- canonical runtime config digest
- execution policy digest
- risk config digests
- production_status = `NO-GO`

をbindする。

materializationはU0と同じくcanonical JSON、atomic publish、byte-identical rerunのみidempotent successとする。partial artifact、自動修復、drift overwriteは禁止する。

U1 contract digestはU2 `BASE_TRAINING` model config/checkpoint identityへ必須bindする。

## 13. Invariants

1. Policy tensorにsymbol/dataset IDがない。
2. environmentはexactly one concrete symbol。
3. Action semanticは全symbolで同一。
4. Reward semanticは全symbolで同一。
5. Observationはdecision time以下だけ。
6. Development / Admissionはnormalizer fitへ入らない。
7. policy request / signal-delay pending / risk projection / realized weightを混同しない。
8. pending order lifecycleとsignal-delay pending targetを混同しない。
9. costをRewardで二重計上しない。
10. training horizonをPolicyへ知らせない。
11. manual strategy priorをPolicy input/reset stateへ入れない。
12. missing value / availability / stalenessを分離。
13. 同一feature contractなら全sourceで同一input shape。
14. U1 contract変更はnew digest/new generation。

## 14. Critical / High Failure Modes

### Critical

- future mutationがObservation(t)を変える
- Admission dataでnormalizer fit
- Reward sumとwealth変化が不一致
- request/projection/pending/currentの意味混同
- signal-delay pendingとpending orderの意味混同
- truncation時のforced free liquidation/bonus
- non-positive wealth silent clip

### High

- row-count weighted pooling
- repeated high-timeframe base rowsの過剰weight
- per-symbol normalizer / zero-shot時refit
- raw nominal price/quantity/cashのtensor混入
- baseline/shadow/Causal Alpha prior混入
- baseline/stress/partial-fill sampled resetによるTrend prior注入
- unavailable fundingとtrue zeroの混同
- cross-asset BTC-reference representation asymmetry

## 15. Required Test Oracles

- exact observation values / shape / dtype
- source index causality
- policy request -> pending -> risk projection -> execution -> current state transition
- pending order state transition
- realized `BookState.portfolio_value`
- exact execution/funding/borrow side effects
- reward-to-wealth reconciliation
- normalizer per-symbol moments / aggregate mean/scale / statistics digest / artifact digest
- U0 source drift and access rejection
- forbidden prior field absence
- exact artifact canonicalization/idempotency

## 16. Falsification Tests

### 16.1 Future mutation

`t+1`以降を任意変更してObservation(t) / normalized Observation(t)が不変。

### 16.2 Symbol rename

同一numeric dataでsymbol textだけ変更しPolicy tensorが一致。Artifact identityは変化してよい。

### 16.3 Price/unit scaling

OHLC等を一律scaleし経済的に等価なcontract unitsへ変換して、dimensionless market/policy stateがtolerance内一致。

### 16.4 Admission poisoning

Train numeric samplesを同一に保ちDevelopment / Admission identityだけ変えるsynthetic generationsで、`statistics_digest`一致、artifact digest不一致。

### 16.5 Cost reconciliation

flat-price `flat -> long -> flat`でlossが既存accounting costだけに由来し、reward telescopingが一致。

### 16.6 Four-stage exposure trace

signal delay + risk projection + partial fillを同時に作り:

```text
policy_requested_weight != risk_projected_weight
risk_projected_weight != current_weight
pending_target_active == 1
```

を確認する。`pending_target_weight`はcurrent submitted signal-delay targetであり、pending order quantityではない。

### 16.7 Reset prior leakage

U1 wrapperがbaseline/stress/partial-fill sampled resetを拒否し、cash reset後に前episode action/order/position stateが残らない。

## 17. Acceptance Criteria

1. frozen U0 universe/identityなしではU1 artifactsをmaterializeできない。
2. U1 tensorにsymbol/dataset ID、manual prior、raw nominal OHLC/volume/quantity/cashがない。
3. one-symbol以外fail closed。
4. fixed 15m/1h/4h/1d causal windows。
5. prohibited cross-asset FeatureKinds拒否。
6. missing value / availability / staleness分離。
7. `policy_requested_weight` / `pending_target_weight`+active / `risk_projected_weight` / `current_weight`の意味が分離。
8. pending order lifecycleは別fieldsで観測。
9. scalar actionはstatic policy scaleを使う。
10. `signal_delay_decisions == 1`をcontractとして固定。
11. Rewardはpure `100 * log(W_after/W_before)`。
12. valid episodeでreward telescoping成立。
13. cost二重計上なし。
14. normalizer fitはU0 Trainのみ。
15. normalizerはequal-symbol moment aggregation。
16. native unique source rowsを使いhigh-timeframe repeated valuesを重複countしない。
17. unavailable placeholderをfitしない。
18. Admission-only generation driftでstatisticsは不変、artifact identityは変化。
19. sample horizonはexternal truncation、forced liquidationなし。
20. sampled initial stateはcash-only。
21. existing Causal Alpha / U3-U6経路に意図しないbehavior changeなし。
22. `u1_contract.json` / `normalizer.json`がatomic/canonical/idempotentにmaterialize可能。
23. Unit / Property / Integration / Compatibility / Static / full-suite / build / final-HEAD CIのQuality Gateを満たす。

## 18. Quality Gate / U2 Handoff

U1完了にはAcceptance CriteriaとCritical/High failure falsification、targeted tests、full suite、Ruff、format、MyPy、import architecture、build、diff self-review、independent/falsification review、exact final HEAD CIが必要。

全てgreenでも、保証するのは**U1 contractの実装整合性と研究汚染防止**までであり、RL learnability、zero-shot economic value、profitability、real execution fidelity、Admission、Productionは保証しない。

U2開始には:

1. real production-candidate U0 Universeがfreeze済み。
2. U1 artifactsがfreeze済み。
3. U1 Quality Gate完了。
4. U0 Train-only `RL_TRAINING` provenanceを作成。
5. frozen U1 contract digestを`UniversalTradeRLRunStage.BASE_TRAINING`のmodel config/checkpoint identityへbind。

Developmentはevaluation-only、Admissionは未開封のまま開始する。
