# Universal Trade RL U2 Base PPO / Selection Preregistration

Status: **Normative design for U2 V1**  
Production status: **NO-GO**  
U2 execution status: **NO-GO until the real production-candidate U0/U1 generation is materialized and frozen**

## 1. 結論

U2 V1は、U1で固定したone-symbol policy surfaceが、複数Train銘柄から**実際に学習可能で、銘柄未見・時間未見の両方へafter-costで一般化できるか**を最小構成で検証する。

最初の候補は1つだけとする。

```text
algorithm      = PPO
architecture   = U_MEDIUM_DIRECT
seeds          = (0, 1, 2)
primary seed   = 0
checkpoint     = fixed final checkpoint only
BC / teacher   = disabled
Lagrangian     = disabled
instrument ctx = disabled
V4 ctx         = disabled
Admission      = sealed
Production     = NO-GO
```

U2 V1でPPOが不合格になった場合、そのDevelopment結果を見て同じgeneration内でSAC、TD3、Lagrangian、BC、reward shaping、別architecture、別gammaへ切り替えない。失敗理由を分類し、次のresearch generationとして設計を変更する。

この制約は「最も良い数字を探す」ことではなく、**U1 representation/action/runtimeの基本learnabilityを反証可能にする**ためである。

---

## 2. Authority / supersession

U2は次を正本とする。

1. `docs/UNIVERSAL_TRADE_RL.md`
2. `docs/implementation-plans/specs/2026-08-31-universal-trade-rl-u1-observation-reward-design.md`
3. `docs/implementation-plans/specs/2026-09-03-universal-trade-rl-u1-normalization-clip-amendment.md`
4. 本spec

`docs/architecture/universal-single-instrument-zero-shot-design.md`や既存`universal-u6-*.json`に残る次の旧契約はU2 V1へ流用しない。

- `finite_horizon_termination`
- time-to-go observation
- cash以外のinitial state
- action validation=`clip`
- Behavior Cloning prerequisite
- Lagrangian prerequisite
- `instrument_context_v1`
- V4 cross-market context
- old U6 architecture sweep

U2 concrete environmentはU1 contractをそのまま使用する。

```text
decision_hours            = 0.25
signal_delay_decisions    = 1
episode_hours             = 720
initial_state_modes       = ("cash",)
episode_boundary_mode     = external_truncation
finite_horizon_observation= false
liquidate_on_end           = false
action validation          = strict
reward                     = pure realized net log growth
normalizer clip            = ±10
```

---

## 3. Objective

U2 V1の目的は次の4点だけである。

1. U1 policy surfaceでPPOが非自明なpolicyを学習できるか確認する。
2. concrete symbol identityなしでDevelopment symbolへzero-shot transferできるか確認する。
3. Train期間より後のfuture regimeへgeneralizeできるか確認する。
4. **unseen symbol × unseen future time**でafter-cost economicsがcashを上回るか確認する。

---

## 4. Non-goals

U2 V1では次を行わない。

- Admission dataの開封
- Production authorization
- hyperparameter sweep
- checkpoint sweep / best checkpoint selection
- seed winner selection
- PPO / SAC / TD3 / Lagrangianのalgorithm sweep
- BC / DAgger / Oracle teacher
- critic warm start
- reward shaping
- dynamic action scaling
- instrument descriptor追加
- V4 / cross-market context追加
- 1分足execution fidelity
- multi-asset portfolio action
- online / zero-shot fine-tuning
- Development結果を見たthreshold緩和

---

## 5. Preconditions

U2 training processは次をすべて満たすまで作動してはならない。

1. real production-candidate U0 `universe.json` / `identity.json`がfreeze済み。
2. real production-candidate U1 `normalizer.json` / `u1_contract.json`がfreeze済み。
3. U1 implementation Quality Gateがexact headで成功済み。
4. U1 `production_status == "NO-GO"`を維持している。
5. U0 Train-only `RL_TRAINING` provenanceが存在する。
6. U1 normalizer knowledge cutoffがU2 `fit_end`と完全一致する。
7. U0 current/final headに対するstrong exact-head verificationがfreshである。
8. Developmentはevaluation-only、Admissionはunopenedである。

どれか1つでも不一致ならU2はfail closedする。

---

## 6. Orthogonal symbol × time split

### 6.1 なぜsymbol splitだけでは不十分か

TrainとDevelopmentが同じ歴史期間なら、Development symbol自体は未見でも、同じBTC/crypto macro regimeをTrain symbolsから学習済みである。

したがって:

```text
                Symbol
             seen      unseen
Time seen      A          B
Time future    C          D
```

を分離する。

最重要cellは`D = unseen symbol × unseen future time`である。

### 6.2 Common interval

U0 manifestのsource identityから、対象generationの全non-excluded symbolについて共通に存在するdecision-clock intervalを決める。

```text
common_start = max(first_timestamp)
common_end   = min(last_timestamp)
```

これはU0でfreeze済みのmetadataだけから計算し、numeric market outcomeを見て境界を動かしてはならない。

### 6.3 Fixed chronological partition

15m decision barsの個数を基準に、common intervalを次へ固定分割する。

```text
FIT                = first 60%
DEV_FUTURE_1       = next 10%
DEV_FUTURE_2       = next 10%
ADMISSION_FUTURE   = final 20%
```

丸めは古い側へ寄せる。境界barを二つのpartitionで共有しない。

追加で`SEEN_TIME_PROBE`をFIT末尾60日として定義する。

```text
SEEN_TIME_PROBE = last 1,440 hours inside FIT
```

### 6.4 Minimum data contract

U2 V1はcommon intervalが最低600日未満ならfail closedする。

また各evaluation windowは、各symbolについて少なくとも2本の完全な720h episodeを収容できなければならない。

### 6.5 Cell definitions

| Cell | Symbol role | Time | Selection use |
|---|---|---|---|
| A | Train | `SEEN_TIME_PROBE` | diagnostic only |
| B | Development | `SEEN_TIME_PROBE` | mandatory symbol-OOS gate |
| C1 | Train | `DEV_FUTURE_1` | mandatory time-OOS gate |
| C2 | Train | `DEV_FUTURE_2` | mandatory time-OOS gate |
| D1 | Development | `DEV_FUTURE_1` | mandatory joint-OOS gate |
| D2 | Development | `DEV_FUTURE_2` | mandatory joint-OOS gate |
| E | Admission | `ADMISSION_FUTURE` | sealed until authorization |

Aはtrainingと同じmacro timeを使うためResearch Selectionには使用しない。learnability / regression diagnosis専用である。

### 6.6 Episode outcome isolation

- Train episodeの`episode_stop`は必ず`fit_end`以下。
- C/D episodeのoutcome barはFITと重複しない。
- E episodeのoutcome barはDevelopment timeと重複しない。
- observation lookbackがpartition境界より前へ伸びることは許可する。これはdecision時点で既知のhistoryでありfuture leakageではない。
- evaluation episodeはwindow startへanchorした**非重複720h tile**を使用する。
- scope数を増やすためにoverlap episodeを後から追加しない。

---

## 7. U1 normalizer / U2 fit cutoff contract

実U1 normalizerはU2 time partitionを事前にmaterializeした後、**FITだけ**から作る。

```text
u1.normalizer_knowledge_cutoff_ns == u2.fit_end_ns
rl_training_provenance.knowledge_cutoff == u2.fit_end_ns
```

Development/Admission symbolまたは`fit_end`より後のTrain barを:

- normalizer
- network training
- threshold estimation
- architecture selection
- checkpoint selection

へ使ってはならない。

---

## 8. U2 Base PPO V1 — fixed training recipe

### 8.1 Algorithm

```text
algorithm                      = PPO
seeds                          = (0, 1, 2)
primary_candidate_seed         = 0
total_timesteps_per_seed       = 524,288
n_envs                         = 8
n_steps                        = 128
batch_size                     = 256
n_epochs                       = 10
learning_rate                  = 0.00012
learning_rate_schedule         = linear
learning_rate_final_ratio      = 0.1
gamma                          = 0.998969062762624
gamma_half_life                = 7 days at 15m decisions
gae_lambda                     = 0.95
clip_range                     = 0.2
normalize_advantage            = true
ent_coef                       = 0.0
vf_coef                        = 0.5
max_grad_norm                  = 0.5
target_kl                      = 0.02
log_std_init                   = -4.0
use_sde                        = false
```

`524,288`は`8 × 128 = 1,024`のrollout quantumで割り切れるため、seed間でactual timestep countがずれない。

### 8.2 Gamma rationale

U1はexternal truncationかつtime-to-goなしなので、U2 V1ではcontinuing-style discounted PPOを使用する。

`gamma=0.998969062762624`は15分decisionでちょうど7日half-lifeであり、既存Universal discounted controlですでに使用された値である。新しいDevelopment結果から選ばない。

Economic Selectionはdiscounted training objectiveではなく、**undiscounted realized after-cost wealth**で行う。

### 8.3 Exploration rationale

`log_std_init=-4.0`を固定する。過去のreal-data Universal auditでは`-2.3`付近のstochastic explorationが約8x capital/day turnoverを生み、`-4.0`でcostが大幅に低下した一方、position-selection failure自体は隠れなかった。

したがって`-4.0`は「経済結果を良く見せるtuning」ではなく、既知のexploration-churn failureを再導入しないための固定mechanics contractとする。

### 8.4 Architecture

U2 V1は既存presetの`U_MEDIUM_DIRECT`だけを使用する。

```text
observation encoder  = hierarchical_sequence_v2-compatible U1 adapter
TCN capacity         = compact
d_model              = 256
timeframe heads       = 4
timeframe layers      = 1
FFN multiplier        = 3
sequence dropout      = 0.0
asset attention       = inactive for one-symbol surface
actor head            = shared_target_v1
actor MLP             = 256 -> 128
critic MLP            = 256 -> 128
action shape          = (1,)
```

`U_SMALL` / `U_MEDIUM_GATE` / `U_LARGE`はU2 V1で比較しない。

### 8.5 Context providers

U2 V1は明示的に:

```text
instrument_context_provider = None
v4_context_provider         = None
```

とする。

### 8.6 Episode routing

既存`DeterministicBalancedInstrumentRouter` semanticsを使用する。

各environment local cycleで:

```text
every Train symbol appears once
before any Train symbol appears twice
```

symbol selectionはrun seed、environment index、completed episode count、partition identityで決定する。

concrete symbolはpolicy inputへ入れない。

### 8.7 Episode start sampling

FIT内のeligible start indexからuniformに選ぶ。

- episode outcomeがFITを越えるstartは禁止。
- regime-balanced / stress-tail oversamplingはU2 V1では使わない。
- seedごとに決定論的で、resume時も同一trajectoryを再構築できること。

### 8.8 No BC / no warm start

U2 V1はrandom PPO initializationから開始する。

```text
behavior cloning          = disabled
teacher                    = none
critic warm start          = disabled
joint warm start           = disabled
```

過去のBC failureを修正することはU2 V1の目的ではない。BCを足すと「U1 surfaceそのものがPPOで学習可能か」の反証性が落ちるためである。

---

## 9. Checkpoint anti-cherry-picking contract

### 9.1 Final checkpoint only

Economic Selectionに使えるcheckpointは各seedの**exact final checkpoint**だけ。

```text
seed 0 -> final@524288
seed 1 -> final@524288
seed 2 -> final@524288
```

intermediate checkpointはcrash recovery用に保存してよいが、Development economicsを見てbest checkpointを選んではならない。

### 9.2 Primary seed

Admission候補は事前に`seed=0`へ固定する。

seed 1/2はrobustness evidenceであり、Development結果を見てseed 0からseed 1/2へcandidateを差し替えない。

### 9.3 Infrastructure retry

OOM、runner loss、network/storage failureなどの非経済的failureは、同じcontract digest / seed / checkpoint lineageからexact resumeしてよい。

次は禁止する。

- seed replacement
- config変更後の「retry」扱い
- failed seedの黙った除外
- partial evidenceを捨てて別runを同一identityとして扱うこと

---

## 10. Evaluation policy

Selection / Admissionはdeterministic mean actionで評価する。

stochastic action evaluationはexploration diagnosticsとして保存してよいが、Selection数値へ使用しない。

各candidate scopeは次へbindする。

```text
checkpoint digest
seed
symbol role
concrete symbol
cell
window digest
episode start/stop
U0 manifest digest
U1 contract digest
U2 training contract digest
execution/risk identity
```

---

## 11. Baselines

### 11.1 Mandatory baseline

Selectionのprimary comparatorはcash:

```text
action = 0.0
wealth = realized cash path under same accounting/runtime
```

### 11.2 Diagnostic static baselines

同一scope・同一Risk/Execution/Accountingで次も記録する。

```text
constant_long  = +1.0 requested exposure
constant_short = -1.0 requested exposure
```

これらはmarket regime contextを理解するためのdiagnosticであり、U2 V1 pass/failのprimary comparatorにはしない。

Legacy V9/V10等は同一scope・同一U1 execution contractで厳密に再現できる場合だけ別reportへ載せる。Selection gateの分母にはしない。

---

## 12. Evidence metrics

各scopeで最低限:

- gross log growth / gross wealth
- net log growth / net wealth
- maximum drawdown
- turnover per day
- total execution cost
- funding PnL
- borrow cost
- mean / p95 absolute realized exposure
- target change count
- submitted / executed change count
- closed trades
- sign flips
- meaningful execution flag
- hard-risk violation count
- unexplained execution rejection count

を記録する。

各cell / seedで:

- symbol-balanced gross wealth
- symbol-balanced net wealth
- median symbol net wealth
- minimum symbol net wealth
- positive net scope fraction
- worst scope net return
- scope net-return CVaR10
- turnover p50 / p95
- gross-to-net log-growth retention
- meaningful-execution symbol fraction

を集計する。

---

## 13. Core economic gate — primary seed 0

Primary seed 0はB、C1、C2、D1、D2の**すべて**で次を満たす必要がある。

```text
symbol coverage                       = complete
symbol_balanced_gross_wealth          > 1.0
symbol_balanced_net_wealth            > 1.0
median_symbol_net_wealth              >= 1.0
minimum_symbol_net_wealth             >= 1.0
positive_net_scope_fraction           >= 0.50
scope_net_return_cvar_10              >= -0.01
turnover_p95_per_day                  <= 1.0
meaningful_execution_symbol_fraction  = 1.0
hard_risk_violation_count             = 0
unexplained_execution_rejection_count = 0
```

さらにgross aggregate log growthが正なら:

```text
net_log_growth / gross_log_growth >= 0.50
```

とする。

このretention gateは「gross alphaの大部分をcostで失うpolicy」をrejectする。

### 13.1 なぜminimum wealthを1.0にするか

U2が主張したいのは「平均で一部のwinnerがloserを隠すstrategy」ではなく、symbol-independent policyのgeneralizationである。

cashを選べるaction spaceでminimum symbol aggregate wealthが1未満なら、そのsymbolではpolicyがcashより悪い。U2 V1ではこれを許容してuniversal successとは呼ばない。

---

## 14. Seed robustness gate

D1、D2、およびD1+D2 aggregateについて3 seedをまとめ、次を満たす。

```text
median_seed_symbol_balanced_net_wealth > 1.0
worst_seed_symbol_balanced_net_wealth  >= 1.0
all_seed_hard_risk_violations           = 0
all_seed_turnover_p95_per_day           <= 1.0
```

さらにperiod-aligned return seriesをcashとpairし、seed間period medianをmoving-block bootstrapする。

```text
bootstrap lower 95% CI of excess return > 0
```

Development結果を見てbootstrap block size、confidence level、resample countを変更しない。

初期固定値:

```text
confidence = 0.95
resamples  = 2,000
bootstrap seed = 0
block-size rule = maintained moving-block implementation
```

---

## 15. Cross-cell decision rule

U2 Development Selectionがpassする条件は全部ANDである。

1. primary seed 0がBをpass。
2. primary seed 0がC1とC2をpass。
3. primary seed 0がD1とD2をpass。
4. D1 seed-robustnessがpass。
5. D2 seed-robustnessがpass。
6. D1+D2 aggregate seed-robustnessがpass。
7. 全evidence identity closureがpass。
8. Admission artifactは一度もnumeric openされていない。

Aはpass/failへ使わない。

1条件でもfailなら:

```text
selected_checkpoint = null
admission_eligible   = false
production_eligible  = false
```

とする。

---

## 16. Admission preregistration

Development Selectionがpassした場合でもAdmissionを自動実行しない。

U0 authorizationは最低限:

- U0 manifest digest
- U1 contract digest
- U2 training contract digest
- U2 Development Selection evidence digest
- exact seed-0 final checkpoint digest
- time partition digest

へbindする。

Admissionで評価するcandidateはseed 0 final checkpointだけ。

Admission scopeは:

```text
Admission symbols × ADMISSION_FUTURE
```

のみ。

Admission gateはDevelopment中に固定した同じcore economicsを使い、追加でcash paired bootstrap lower 95% CI > 0を要求する。

Admissionを見た後で:

- retrain
- renormalize
- checkpoint swap
- seed swap
- threshold change
- action scale change
- feature change
- risk config change

を行った場合、そのAdmission結果は同一generationのzero-shot evidenceとして無効とする。

Admission passでもProductionは自動GOにしない。

---

## 17. Stop conditions

次の場合はU2を止め、同じgenerationでparameter fishingしない。

### Technical stop

- NaN / inf action, reward, loss, value, gradient diagnostics
- U0/U1/U2 identity drift
- wrong symbol routing
- Development/Admission data access during fit
- source/checkpoint artifact drift
- resume lineage mismatch

### Learnability stop

- final policyが全mandatory cellでmeaningful executionを持たない
- deterministic actionが事実上constant cashへcollapseし、balanced net wealth >1を満たさない
- gross wealth自体が1以下
- seed 0がB/C/D gateを1つでもfail

### Robustness stop

- seed 1/2を含めるとD gateがfail
- worst seed < cash
- bootstrap lower CI <= 0
- turnover / costによってgross edgeが保持できない

Failure後はAdmissionを開かない。

---

## 18. Artifact / identity contract

### 18.1 U2 time partition artifact

`u2_time_partition.json`は最低限:

- U0 manifest digest
- common start/end
- FIT start/end
- SEEN_TIME_PROBE start/end
- DEV_FUTURE_1 start/end
- DEV_FUTURE_2 start/end
- ADMISSION_FUTURE start/end
- exact decision-bar counts
- episode tiling contract
- digest

を持つ。

### 18.2 U2 training contract

`u2_training_contract.json`は最低限:

- U0 manifest digest
- U1 contract digest
- U1 normalizer digest
- U1 normalizer cutoff
- `RL_TRAINING` provenance digest
- time partition digest
- exact PPO resolved hyperparameters
- architecture spec digest
- seeds / primary seed
- router digest contract
- episode sampling contract
- evaluation/checkpoint rule
- baseline contract
- Selection thresholds
- `production_status = NO-GO`
- digest

をbindする。

### 18.3 Existing run identity integration

U2 V1では`UniversalTradeRLRunStage.BASE_TRAINING.model_config_digest`へ**U2 training contract digest**を入れる。

U2 training contract自体がU1 contract digestをbindするため、既存U0 run-identity schemaを曖昧に再解釈せず、U1→U2 dependencyをcontent-addressedに固定できる。

### 18.4 Checkpoint identity

各final checkpointは:

- BASE_TRAINING run identity
- U2 training contract digest
- seed
- actual timesteps
- source git SHA
- container/image identity where applicable
- model architecture digest
- observation/action/U1 contract digests

へbindする。

---

## 19. Invariants

1. U1 observation/action/reward/runtime semanticsを変更しない。
2. fit dataはTrain symbols × FIT timeだけ。
3. Development/Admissionはfitへ入らない。
4. Development結果からcheckpoint/seed/hyperparameterを選ばない。
5. concrete tickerはpolicy input/architecture identityへ入らない。
6. routerは各symbolをcycle内で1回ずつ扱う。
7. seed 0が唯一のAdmission candidate。
8. seed 1/2 failureを隠さない。
9. economic gateはrealized after-cost wealthを使う。
10. cash baselineとexact scopeをpairする。
11. B/C/Dの意味を混同しない。
12. AdmissionはDevelopment pass + authorization前に開かない。
13. Production statusは常にNO-GO。

---

## 20. Critical / High Failure Modes

### Critical

- C/D training future leakage
- Development/Admission fit leakage
- U1 contractと異なるenvironmentで学習
- symbol ID/tickerがpolicyへ入る
- seed/checkpoint cherry-pick
- economic resultを見た後のthreshold変更
- cash baselineと異なるscope/accountingで比較
- Admissionを見た後のrefit
- checkpoint identity mismatch

### High

- one winner symbolがbalanced averageを支配
- one lucky seedだけがpass
- stochastic exploration churnをSelectionへ混入
- gross positive/net negativeを「学習成功」と扱う
- near-flat policyをuniversal successと扱う
- overlapping episode増量によるpseudo sample inflation
- old U6 finite-horizon/BC/context semanticsの混入
- Training scopeにfit_endを越えるepisodeが入る

---

## 21. Test oracles

- exact time partition bar boundaries
- no overlap between outcome partitions
- exact U1 normalizer cutoff == FIT end
- RL_TRAINING provenance source/cutoff
- balanced router cycle coverage
- one-symbol / one-action at every routed env
- no context provider
- fixed final checkpoint identity
- seed set completeness
- deterministic evaluation
- per-scope BookState wealth reconciliation
- cash paired scope closure
- B/C/D cell assignment
- symbol-balanced aggregations
- CVaR / turnover / retention calculations
- moving-block bootstrap reproducibility
- Development Selection AND rule
- Admission sealed-access audit

---

## 22. Required test layers

### Unit

- time partition codec
- U2 training contract codec
- gate calculations
- seed/primary candidate validation

### Property / falsification

- future bars do not enter FIT
- Development/Admission arrays cannot be requested by training factory
- shuffled input order does not change canonical cell/gate result
- one losing symbol cannot be hidden by winners
- one lucky seed cannot select a candidate
- intermediate checkpoint cannot be admitted

### Integration

- U1 environment factory + episode router + PPO vector env smoke
- 8 env × multiple symbols, no book/order state crossing
- exact resume identity
- final checkpoint deterministic evaluation on B/C/D
- cash/static baseline same-scope replay

### Compatibility

- current U1 exact observation/action/reward tests
- existing router/binding/checkpoint/structured serving contracts
- existing Causal Alpha path behavior unchanged

### Static / repository

- Ruff
- format
- MyPy
- import-linter
- dead-code report
- full pytest + branch coverage
- package build
- Ubuntu/Windows compatibility
- training image / non-root probe
- PostgreSQL Catalog / Nautilus capability as applicable
- exact-final-HEAD CI

---

## 23. Quality Gate

U2 implementationを「training-ready」と呼ぶには:

1. 本preregistrationをcodeより先にfreeze。
2. time partition / training contract / Selection evidenceがcanonical artifact化。
3. U1 contractを一切弱めない。
4. Train-only + FIT-only access firewallがruntimeで強制。
5. PPO configがexact identity-bound。
6. primary seed / final checkpoint anti-cherry-pickingを強制。
7. B/C/D evaluationとseed robustness gateを実装。
8. Admission sealed enforcementを実装。
9. targeted/falsification/integration/full/static/build/CIをpass。
10. independent reviewでholdout leakageとsuccess-condition loopholeを再探索。
11. real production-candidate U0/U1 artifactsをfreeze。
12. fresh U0/U1 stack verificationをpass。

それまではtraining executionはNO-GO。

---

## 24. What U2 Development pass would and would not prove

### Proveできる範囲

- 固定U1 surface上でPPOが学習可能であること
- Development symbolsへのzero-shot evidence
- future-time evidence
- Developmentにおけるunseen symbol × unseen future time evidence
- 3 seedでの限定的robustness
- after-cost cash excessの統計的support

### Proveしないこと

- Admission success
- Production profitability
- future live regimeでの永続generalization
- one-minute execution fidelity
- capacity scalingの最適性
- PPOが最良algorithmであること
- U_MEDIUM_DIRECTが最良architectureであること

U2 Development passはAdmissionを**開く資格**までであり、Production GOではない。
