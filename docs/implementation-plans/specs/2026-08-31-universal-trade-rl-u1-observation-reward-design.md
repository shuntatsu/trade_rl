# Universal Trade RL U1: Observation / Action / Reward Contract Design

作成日: 2026-08-31 (JST)

> **Production status: NO-GO**
>
> U1は研究環境のObservation / Action / Reward / normalization / artifact contractを固定する。RL学習、zero-shot性能、収益性、Admission合格、Production投入は証明しない。

## 1. U0との関係

Universal Trade RL U0は、Train / Development / Admission / Excludedの役割、完全なsource-data identity、Train-only fit provenance、Admission firewallを固定する。

U1はその境界を消費して、**どの銘柄にも同じ意味で適用できるone-symbol RL環境契約**を作る。

段階を明確に分ける。

1. U1 contract implementationはsynthetic/published fixtureとU0 example generationで検証できる。
2. U2 Base RL training開始前には、実production-candidate source catalog / role configをU0でmaterializeし、`universe.json` / `identity.json` digestをfreezeする。
3. U1 artifactsもfreezeし、そのdigestをU2 model/checkpoint identityへbindする。

既存Causal Alpha V9/V10/V11とUniversal U3-U6は変更しない。U1はzero-shot-firstの新経路である。

## 2. Objective

- causal / symbol-independentなPolicy Observationを固定する。
- one-symbol / one-capital-budget用のscalar target-exposure Actionを固定する。
- realized after-cost wealthと厳密に一致するRewardを固定する。
- U0 Train symbolsだけを使うUniversal market normalizerを固定する。
- sampled horizonを市場終端と混同しないcontinuing/truncation semanticsを固定する。
- U1 contractとnormalizerをcanonical / content-addressed / atomic artifactとして再現可能にする。

## 3. Non-goals

U1では以下を行わない。

- PPO / SAC / TD3等の学習・比較
- network architecture選択/大型化
- Behavior Cloning / teacher policy
- Causal Alphaをteacher/anchorとして利用
- V12 label/horizon研究
- 1分足execution fidelity研究
- multi-asset capital allocation policy
- Development結果を見たfeature/reward再fit
- Admission開封
- transfer/fine-tuning
- profitability / Sharpe / Production readinessの主張

## 4. 現行コードから固定する意味論

### 4.1 Execution/Accounting authority

`ResidualMarketEnv`を唯一のRisk / Execution / Accounting authorityとして再利用する。U1専用Simulatorは作らない。

### 4.2 Exposure stage semantics

現行`ResidualMarketEnv.step()`は:

```text
decision.executed_hybrid_target
  -> risk_projector.project(...)
  -> hybrid_risk.weights
  -> execute_target(... target=hybrid_risk.weights)
```

の順に処理する。

step後の`ObservationExecutionState.requested_weights`は`hybrid_risk.weights`から作られるため、その現行意味は**post-risk execution target**である。U1ではこれを`risk_projected_weight`として読む。既存field名/意味をlegacy path全体で変更しない。

### 4.3 Signal-delay targetとpending orderを分離

`_pending_hybrid_target`は`signal_delay_decisions`用のtarget queueであり、partial fill / order latency / order expiryを表すものではない。

U1 V1はcausal next-decision executionを固定するため:

```text
signal_delay_decisions = 1
```

を必須にする。

各fieldの意味:

- `policy_requested_weight`: 現decisionでPolicyが提出したpre-risk target。
- `pending_target_weight`: 現decision後、signal-delay queueに残る次decision用target。
- `pending_target_active`: pending targetの有無。pending flat `0.0`と「pendingなし」を分離する。
- `risk_projected_weight`: 現decisionで実際にExecutionへ渡されたpost-risk target。通常は前decisionのpending target由来。
- `current_weight`: Execution後のrealized `BookState.weights`。

order lifecycleは既存`PendingOrderObservationState`で別途観測する。

### 4.4 Execution cost observation

現行`EnvironmentExecutionCoordinator.execution_observation_state()`は:

```text
execution_cost = cost_by_symbol / initial_capital
```

を保存する。U1 `execution_cost_rate`はこのdimensionless意味論をそのまま使い、再度nominal変換しない。

## 5. Architecture

```text
U0 Frozen Universe
      |
      +-----------------------------+
      |                             |
      v                             v
published Train datasets      FEATURE_NORMALIZATION provenance
      |                             |
      +----------> Universal Market Normalizer
      |                             |
      v                             v
ResidualMarketEnv ------> UniversalTradeMarketEnv wrapper
(single symbol)              |          |
                             |          +--> normalized causal Observation
                             |                     |
                             |                 Shared Policy
                             |                     |
                             |              scalar action [-1,+1]
                             |                     |
                             |          policy_requested_weight
                             |                     |
                             |            signal-delay queue
                             |                     |
                             +----------> maintained Risk/Execution
                                                   |
                                            current_weight
                                                   |
                                          realized BookState
                                                   |
                                      100 * net log growth
```

責務:

- `ResidualMarketEnv`: market execution/accounting authority。
- U1 runtime snapshot: named read-only stateだけを公開。
- U1 observation builder: allowed market history + realized endogenous stateだけを生成。
- U1 market normalizer: published U0 Train artifactsだけからfit。
- endogenous state transform: deterministic/versioned。rollout distributionからfitしない。
- U1 wrapper: fixed action/observation/reward/runtime contractを検証してbase envへ委譲。

## 6. Observation Contract

Schema:

```text
universal_trade_observation_v1
```

U1 environmentは`dataset.n_symbols == 1`を必須とする。

Observationは:

1. Market sequence plane
2. Endogenous policy-state plane

だけで構成する。

symbol name、symbol index、dataset ID、artifact digestをtensorへ入れない。

### 6.1 Market windows

| timeframe | length | history |
| --- | ---: | ---: |
| 15m | 96 | 24h |
| 1h | 168 | 7d |
| 4h | 120 | 20d |
| 1d | 60 | 60d |

各timeframe:

```text
sequence_<tf>_values
sequence_<tf>_available
sequence_<tf>_staleness
```

既存`SequenceObservationBuilder`のcausal contractをreuseし、全source rowで`source_index <= decision_index`を満たす。future backfillは禁止。

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

U1 V1で禁止:

- `RELATIVE_RETURN_TO_BTC`
- `ROLLING_CORRELATION_TO_BTC`
- `ROLLING_BETA_TO_BTC`
- `CROSS_SECTIONAL_MOMENTUM_RANK`
- `CROSS_ASSET_DISPERSION`

FeatureSpecのexact order、lookback、timeframe、rolling-normalization contractはpolicy contract digestへbindする。

raw absolute OHLC、raw nominal volume/cash/quantity、contract multiplier、symbol IDはPolicy tensorへ直接入れない。

### 6.3 Missingness

```text
value=0, available=1
```

と

```text
value=0, available=0
```

を別状態とする。stalenessも別channelで維持する。

### 6.4 Endogenous policy state

固定順序:

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

signal-delay `pending_target_*`と`PendingOrderObservationState`は別責務である。

### 6.5 Deterministic state transforms

Endogenous stateはpolicy-generated rollout分布からfitしない。

versioned transform例:

```text
position_age_days      = log1p(position_age_hours / 24)
pending_order_age_days = log1p(pending_order_age_hours / 24)
eligible_delay_days    = log1p(pending_order_eligible_delay_hours / 24)
expiry_distance_days   = log1p(pending_order_expiry_distance_hours / 24)
mark_index_basis       = tanh(100 * raw_basis)
borrow_rate            = tanh(raw_borrow_rate)
```

weights / ratios / masks / fill / participation / cost rate / drawdown / gross / net / cash / risk / marginはdimensionless意味論を保持する。transform versionとfield orderをstate-layout digestへbindする。

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

Semantic:

```text
normalized_target_exposure_v1
```

Action shape `(1,)`、range `[-1,+1]`。

```text
policy_requested_weight = action * policy_weight_scale
```

`policy_weight_scale`はstatic/precommitted。U1 V1 default `1.0`。dynamic risk capをAction semanticへ混ぜない。

out-of-range / NaN / infはfail closed。U1 parserでhidden clipしない。

signal delay 1によりaction_tは次decision用pending targetになり、現decisionでは前pending targetがRisk/Execution pathへ進む。

## 8. Reward Contract

Semantic:

```text
universal_net_log_growth_reward_v1
```

```text
reward_t = 100 * log(W_after / W_before)
```

`W_before/W_after`は同じrealized hybrid accounting pathの`BookState.portfolio_value`。

既存accountingのtrading PnL、fee、spread、impact/slippage、funding、borrow cost等はwealthに一度だけ反映する。

U1追加shapingは全て0:

- baseline/excess
- drawdown reward penalty
- turnover penalty
- projection penalty
- terminal bonus/penalty
- margin reward penalty

valid trajectoryでは:

```text
sum(reward_t)/100 == log(final_wealth/initial_wealth)
```

がfloating tolerance内で成立する。

non-positive / NaN / inf wealthはfail closed。U1 reward層でepsilon clipしない。

## 9. Universal Market Normalization

Artifact semantic:

```text
universal_trade_sequence_normalizer_v1
```

fit対象は**Market sequence continuous valuesだけ**。Policy-stateはSection 6.5 deterministic transformsを使う。

### 9.1 Source identity must be proven, not asserted

normalizer fit APIは`MarketDataset`とdigest文字列の組だけを信用してはならない。

各Train sourceはpublished market artifact rootで指定し、次の順序を守る。

1. U0 `require_normalization_scope(symbols)`を実行。
2. U0 `FEATURE_NORMALIZATION` provenanceを作る。
3. `inspect_published_market_dataset_artifact(root)`でcanonical artifactを検証。
4. artifact manifestの`artifact_digest`がU0 universe entry `dataset_digest`と一致することを検証。
5. `load_market_dataset_artifact(root)`でMarketDatasetをload。
6. loaded datasetがsingle-symbolかつ期待symbolであることを検証。
7. その後初めてnumeric arraysへアクセスする。

これにより呼び出し側がdigestだけを偽装して別arraysをnormalizerへ渡す経路を閉じる。

### 9.2 Knowledge cutoff

U0 provenance `knowledge_cutoff`はUTC nanosecond timestampとして扱う。

```text
bar_close_timestamp_ns <= knowledge_cutoff_ns
```

のsource eventだけをfit対象にする。

### 9.3 Carried feature de-duplication

multi-timeframe/base-timeframe featureはbase decision clockへcarryされるため、base rowsを単純連結すると同一feature eventを複数回countする。

各available feature sampleについて、maintained `feature_staleness_hours`からfeature source event timeを復元する。

```text
source_event_time_ns
  = bar_timestamp_ns
    - round(feature_staleness_hours * 3_600_000_000_000)
```

同一 `(symbol, feature, source_event_time_ns)` はfit時に1回だけcountする。

semantic identity:

```text
unique_feature_event_time_v1
```

この方法は「値が変わった行」を使わない。同値の連続native eventsもsource event timeが異なれば別sampleとして保持する。

future/invalid event time、available=false、knowledge cutoff超過は拒否/除外する。

### 9.4 Equal-symbol weighting

各feature channelでsymbolごとのunique available event sampleから:

```text
mu_s = mean(x_s)
q_s  = mean(x_s^2)
```

Train symbol数`S`について:

```text
mu    = (1/S) * sum(mu_s)
q     = (1/S) * sum(q_s)
var   = max(q - mu^2, 0)
scale = sqrt(var)
```

`scale <= epsilon`なら`1.0`。

各symbolが同じ1 voteを持つ。history lengthやbase-clock carry回数をsymbol間weightにしない。

### 9.5 Identity separation

Normalizerは:

- `statistics_digest`: fitted numeric statistics + feature/layout/sampling semantics
- artifact `digest`: statisticsに加えU0 universe/provenance identity等

を持つ。

別Universe generationでTrain published artifacts/knowledge cutoff/policy contractが完全同一、Development/Admissionだけが異なる場合:

```text
statistics_digest: equal
artifact digest: different
```

を要求する。

## 10. Episode / Reset Contract

U1 V1固定:

```text
episode_hours = 720h
signal_delay_decisions = 1
episode_boundary_mode = external_truncation
finite_horizon_observation = false
liquidate_on_end = false
initial_state_modes = ("cash",)
```

cash-onlyの理由: existing `baseline` / `stress` / `partial_fill` sampled resetは`TrendStrategy` targetを利用し、`current_weight`経由でmanual priorを注入するため。

sample horizon到達は`truncated=True`、economic/risk failureだけが`terminated=True`。sample endで無料liquidation/terminal bonusを作らない。

explicit restore/serving stateはU1 V1 training contract外とし、U2以降で別契約化する。

## 11. U1 Wrapper Contract

`UniversalTradeMarketEnv`は既存`ResidualMarketEnv`をwrapする。

constructorでfail closed:

- one symbol
- `ActionMode.TARGET_WEIGHT`
- target count 1
- structured sequence enabled
- exact U1 windows/feature layout
- signal delay 1
- pure net-log-growth base reward
- external truncation
- finite-horizon observation false
- liquidate_on_end false
- cash-only sampled reset

step:

```text
parse scalar action
 -> policy_requested_weight
 -> record W_before
 -> base_env.step([policy_requested_weight]) exactly once
 -> record W_after
 -> recompute U1 reward
 -> require base pure reward reconciliation
 -> read runtime snapshot
 -> build/normalize U1 observation
 -> preserve terminated/truncated/info
```

wrapperはPreTradeRisk / execution / accountingを直接呼ばない。

## 12. Artifact / Identity Contract

U1 canonical output:

```text
output-root/
  normalizer.json
  u1_contract.json
```

`u1_contract.json`は最低限:

- schema version
- U0 universe manifest digest
- U0 materialization identity digest
- policy contract digest
- observation/action/reward/state-layout digests
- normalizer digest / provenance digest
- episode/reset/signal-delay/runtime config digest
- execution policy digest
- pretrade/portfolio risk digests
- `production_status = NO-GO`

をbindする。

materializationはU0同様、canonical JSON、staging、file+directory fsync where supported、directory-level atomic publish、byte-identical rerunだけidempotent success。partial final output、drift overwrite、自動修復は禁止。

## 13. Invariants

1. Policy tensorにsymbol/dataset identityがない。
2. U1 environmentはexactly one symbol。
3. Action/Reward semanticsは全symbolで同一。
4. Observationはdecision time以下だけ。
5. Development/Admissionはfitへ入らない。
6. published artifact identityを検証してからTrain numeric arraysを読む。
7. request/pending/risk/current exposureを混同しない。
8. signal-delay pendingとpending orderを混同しない。
9. costをRewardで二重計上しない。
10. training horizonをPolicyへ知らせない。
11. manual priorをObservation/resetへ入れない。
12. missing/availability/stalenessを分離。
13. carried feature eventをfitで重複countしない。
14. row数でTrain symbol間weightを変えない。
15. contract変更はnew digest/new generation。

## 14. Critical / High Failure Modes

### Critical

- future mutationがObservation(t)を変える
- Admission dataがnormalizer fitへ入る
- caller-supplied fake digestで別dataset arraysをfitできる
- Reward sumとwealth変化が不一致
- request/pending/risk/current意味混同
- signal-delay pendingとpending order意味混同
- sample truncationでforced liquidation/bonus
- non-positive wealth silent clip

### High

- row-count weighted pooling
- carried high-timeframe eventの重複count
- per-symbol/zero-shot refit normalizer
- raw nominal/ID/manual priorのtensor混入
- baseline/stress/partial-fill resetによるTrend prior注入
- unavailable fundingとtrue zero混同
- cross-asset BTC-reference representation asymmetry

## 15. Test Oracles

- exact observation values/shape/dtype
- source index causality
- future mutation invariance
- policy request -> signal pending -> risk projection -> execution -> current transition
- pending order lifecycle
- exact BookState portfolio value and accounting side effects
- reward-to-wealth telescoping
- published dataset artifact digest vs U0 manifest
- per-symbol unique-event sample set
- per-symbol moments / equal-symbol aggregate mean/scale
- statistics digest vs U0-bound artifact digest
- canonical materialized bytes/idempotency
- forbidden prior/identity field absence

## 16. Required Test Layers

- Unit: contract/action/state/reward/normalizer/artifact codecs
- Property/Falsification: future mutation, symbol rename, price-unit scaling, fake digest rejection, equal-symbol weighting, carried-event dedupe, missing-vs-zero, reward telescoping
- Integration: action→delay→risk→partial fill→next obs, flat-market costs, funding/borrow, truncation, U0 provenance→normalizer→U1 artifact
- Compatibility: existing Causal Alpha V9/V10/V11 and Universal U3-U6
- Static: Ruff, format, MyPy, import-linter/architecture checks
- Full suite + package build + final-HEAD CI

## 17. Acceptance Criteria

1. U1 artifacts bind a valid U0 universe/materialization identity。
2. normalizer source artifact digest is independently inspected and must equal U0 manifest `dataset_digest`。
3. U1 tensor has no IDs/raw nominal/manual prior/horizon fraction。
4. exactly one symbol。
5. exact 15m/1h/4h/1d causal windows。
6. prohibited cross-asset FeatureKinds rejected。
7. missing/availability/staleness separate。
8. `policy_requested_weight` / `pending_target_weight`+active / `risk_projected_weight` / `current_weight` separate。
9. pending order lifecycle separate。
10. scalar action uses static policy scale。
11. signal delay 1 fixed。
12. pure `100*log(W_after/W_before)` reward。
13. valid trajectories telescope to final wealth。
14. no cost double-counting。
15. Train-only normalization firewall occurs before artifact/data read。
16. feature events are de-duplicated by recovered source event time。
17. equal-symbol moment weighting used。
18. unavailable placeholder excluded。
19. identical Train artifacts across synthetic generations give same statistics digest despite Development/Admission changes, while artifact digest changes。
20. cash-only sampled reset。
21. external truncation/no forced end liquidation。
22. existing U3-U6/Causal Alpha path has no intended behavior regression。
23. `normalizer.json` + `u1_contract.json` materialize atomically/canonically/idempotently。
24. Unit/Property/Integration/Compatibility/Static/full-suite/build/final-HEAD CI Quality Gate passes。

## 18. Quality Gate / U2 Handoff

U1を完了と呼ぶにはAcceptance Criteria、Critical/High falsification、targeted tests、full suite、Ruff、format、MyPy、import architecture、build、diff self-review、independent/falsification review、exact-final-HEAD CIを全て確認する。

それでも保証するのは**U1 contractの実装整合性と研究汚染防止**まで。RL learnability、zero-shot economics、profitability、Admission、real-market fidelity、Production readinessは保証しない。

U2開始条件:

1. real production-candidate U0 generation freeze。
2. U1 artifacts freeze。
3. U1 Quality Gate完了。
4. U0 Train-only `RL_TRAINING` provenance作成。
5. frozen U1 contract digestをexisting `UniversalTradeRLRunStage.BASE_TRAINING` model/checkpoint identityへbind。

Developmentはevaluation-only、Admissionは未開封のままとする。
