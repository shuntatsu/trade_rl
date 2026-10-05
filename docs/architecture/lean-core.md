# Lean core architecture

## 結論

Trade RLの現行coreは、**causalなmarket data、1つのexecution/accounting ledger、hard risk、small strategy interface、銘柄ごとの独立評価**に絞る。

旧U-series、旧Causal Alpha世代、mandatoryな `teacher -> admission -> BC -> RL`、研究DB、UI、世代別runnerは現行研究経路の必須条件ではない。Git historyに残る旧architectureをcurrent treeへ戻さない。

現在の研究目的は、**実運用では一度に1銘柄を独立accountとして売買する**一方、銘柄IDに依存しない共通strategy/model/policyを複数銘柄の学習・検証へ適用し、未知・未使用の銘柄や期間でも転用可能な汎用性を検証することである。複数銘柄のtraining dataを使うことは、複数銘柄を同時保有するshared-cash portfolioを意味しない。各銘柄への適用ではpoint-in-time情報と同一の約定・会計条件を使い、コスト控除後の結果がunused dataでも維持されるかを確認する。

## Core flow

```text
causal raw / venue evidence
        ↓
MarketDataset + immutable dataset artifact
        ↓
strategy logical intent (SHORT / FLAT / LONG)
        ↓
hard risk / feasibility
        ↓
MarketExecutor + BookState
        ↓
independent per-symbol replay
        ↓
metrics / comparison / robustness / immutable run artifact
```

## Data contract

`MarketDataset` はpoint-in-timeで観測可能な情報だけをdecisionへ渡す。

必須条件:

- `feature_available_time <= decision_time` を満たす。featureのinformation/availability timeはdecision timeを越えない。
- supervised labelはfit cutoffの内側だけで完結する。
- scaler、normalization、imputation、feature selectionにfuture情報を混ぜない。
- developmentやfinal futureをthreshold/model選択へ戻さない。
- 15分barの行数を独立標本数と解釈しない。
- 同calendar shockを共有する複数銘柄を完全独立標本と仮定しない。

`fit_symbol_indices` はtraining rowの選択だけでなく、**選択featureが依存してよいsymbol universeの情報scope**でもある。content-verified Datasetで全symbolより狭いfit scopeを使う場合、cross-sectional rank / dispersionのように全universeへ依存するselected featureはrejectし、reference-relative / correlation / betaはそのreference symbolがfit scope内にある場合だけ許す。依存kindはfeature名のheuristicではなくDataset build identityのFeatureKindから復元する。source Dataset identityをネストして保持する正式transformでは元build configまで遡って同じ検査を行い、verified identityなのにdependency provenanceを復元できないstrict subsetはfail closedにする。full-symbol trainingは従来どおり許し、identity provenanceを持たないlegacy/synthetic Datasetの互換は維持するが、その経路だけをunseen-symbol isolationの強い証拠にはしない。

Dataset identityは内容にbindされ、canonical artifactはdeterministicでなければならない。publication先が既に存在する場合は上書きせずfailする。

Canonical Datasetのidentity-bound feature numericsは `trade_rl.data.features.numerics` を単一authorityとし、scalar `math.log` と固定順序の `math.fsum` を基礎にmean / variance / standard deviation / dot / covariance / correlationを定義する。identityを一致させるためのrounding、quantization、tolerance-based hash canonicalizationは行わない。現行buildは `market_build_v3` と `portable_feature_numerics_v1` をbuild identityへ明示bindし、保存feature dtypeは従来どおり`float32`とする。

このportable contractは、同一code・config・sealed sourceから構築した完全Datasetについて、現行のUbuntu x86_64 hosted runner上の複数AMD EPYC系と複数Intel Xeon系で `features`、`global_features`、normalization digest、Dataset IDのbit-exact一致を実証済みである。一方、任意のARM、任意libm、任意platformまでの普遍的なbit-identical保証は主張しない。historical `market_build_v2` artifactは書き換えず、current readerでそのidentityのまま読み取れる互換を維持する。

## Frozen prequential Ridge stream

The optional forecast producer fits the existing symbol-balanced Ridge on each
declared prefix exactly once and stores predictions only in its following,
non-overlapping block. Both label endpoints and their recorded source publication
must precede the cutoff. The training trace records symbols, endpoint prices and
publication clocks; the training object and vintage bind selected features,
sample weights and fit-only scaling/model parameters. Unused Dataset suffixes
are excluded from causal identities.

`ForecastBlock` declares historical model-completion and inference-delay
assumptions. The stream identifies these as `declared_simulation_v1`, never as
observed runtime receipts. Each packet binds a symbol, decision snapshot, selected
inputs, source and forecast availability, exact horizon, conditional mean **log**
return and model vintage. Only the newest ready, unexpired packet from the active
block may reach the existing cost-aware intent controller. A gap, unavailable
selected input, missing symbol or stale packet is an error, not a zero forecast.

Whole-Dataset ID is stored as lineage; the separate causal scope identity covers
only consumed fit and prediction inputs. The JSON reader requires an externally
pinned expected content digest, checks nested identities and recipe consistency,
and recalculates each prediction from the frozen selected input and model. It
does not refit or authenticate the original fitting process, inspect the source
Dataset, prove historical point-in-time availability or attest runtime latency.

The existing controller's switching-cost gate is a declared log-return proxy.
It does not estimate expected simple return, uncertainty or optimal portfolio
allocation. There is no new ledger, candidate Run/Study integration, joint RL
training, live execution or profitability evidence in this capability.

## Strategy contract

Strategyが返すlogical intentは小さく保つ。

```text
SHORT
FLAT
LONG
```

Strategyの責任:

- entry
- hold
- exit
- reversal
- signal / forecast / policyに基づく経済判断

同じLONG→LONGまたはSHORT→SHORTなら、価格変動でweightがdriftしただけを理由に毎decisionでtarget weightへ戻さない。標準は**quantity-preserving hold**であり、intentが変わった場合かhard riskがde-riskを要求する場合にquantityを変える。

Split changes the units of both the filled book and the cached strategy proposal.
PPO training, single-symbol replay, and shared-cash replay rebase the proposal by
the processed split factor, including any unfilled entry remainder. A same-side
hold therefore keeps split-adjusted exposure rather than trading back to an old
unit count. This unit conversion does not replace a pending proposal with the
actual partial fill or bypass hard risk.

### PPO Observation v2

初回canonical real-data M2で使うteacher-free PPOのpolicy observationは、fit-scope leakageを避けるため最小のcausal contractへ固定する。tensor順序は次である。

1. selected local feature values
2. selected local availability / finite mask
3. selected normalized local feature staleness
4. current intent
5. current weight

unavailableまたはnon-finiteなlocal valueは0へmaskする。stalenessはselected featureと同じ順序で`[0, 1]`の正規化値を渡し、unavailable featureは最大stalenessでなければならない。PPO v2はstalenessが欠けたobservationをfail-closedにする。一方、`StrategyObservation`はPPO専用型ではないため、既存のrule/forecast/control caller向けconstructor互換を維持し、generic contractではstaleness省略を許す。canonical replayとPPO environmentは実datasetのstalenessを明示的に渡す。

初回M2のpolicy tensorにはsymbol IDを入れず、dataset-global featureも入れない。現行`MarketDatasetBuilder`の`active_fraction`、`tradable_fraction`、`market_return_mean`、`market_return_dispersion`はdataset全symbol universeから集計されるため、fit symbol subsetだけでPPOを学習するStudyで使うとfit-scope外symbolがtraining observationへ間接混入し得る。このためObservation v2のglobal rosterは明示的な空集合である。global regimeを将来試す場合は、fit-scope外evaluation symbolを含まないreference universeを事前固定した別Controlled Factorとして扱う。

Observation contractは暗黙のimplementation detailにしない。新規Candidate Runと新規Studyのresolved configはObservation schema、staleness利用、空のglobal rosterをsemantic identityへbindする。execution economics、reward、action、risk、PPO network architectureはObservation v2のpolicy inputへ追加しない。

### PPO Observation v3 and minimum-hold treatment

Observation v3 keeps the v2 tensor prefix and appends
`min(position_age_bars, 504) / 504`. Age comes from the actual signed filled
quantity, not the last requested action. The first nonzero fill starts an
episode at age 1; partial/no fills and same-side changes advance age once per
completed hourly interval; actual flat resets it to 0; a sign crossing starts a
new episode at 1. For the hourly comparison, `minimum_hold_bars=H` unlocks
voluntary target intents at `age >= H`, counting the first filled interval as
age 1. The freshly trained H=0 baseline uses the same v3 observation.

While locked, a raw FLAT or opposite-side intent is recorded as requested but
maps to the current filled side and exact signed quantity. This cancels any
unfilled order remainder and prevents new exposure during the lock. At unlock,
the current intent is reapplied even when it matches the held side, so a partial
entry does not remain stuck at its earlier fill size. Hard-risk projection runs
after this voluntary-action constraint and can always reduce or flatten. PPO
training info and replay decisions retain requested/effective intents,
suppression/unlock flags, and suppression counts; new candidate artifacts
record those counts. SB3 still samples its ordinary three-action distribution
and retains sampled actions/log-probabilities. The constraint changes the
environment transition, not the economic action contract.

Minimum-hold duration requires Observation v3. A duration Study binds the same
v3 observation, terminal settlement, explicit `PreTradeRiskConfig`, Dataset,
execution overlay, capital, and evaluation window for its H=0 baseline and all
candidates. New `resolved_run_config_v5` and `lean_candidate_result_v6`
identities bind these values; H alone is the `PPO_MINIMUM_HOLD` factor. The
explicit risk config is shared by PPO training and every strategy replay. A
drawdown stop at 0.20 is a hard pre-trade guard, not a guarantee that a price gap
or terminal move cannot exceed 20% realized drawdown.

`run_shared_cash_replay` is a separate multi-symbol evaluation boundary. Each
symbol has one fresh strategy instance, all instances observe the same pre-trade
portfolio snapshot, and their complete target vector passes through the shared
`PreTradeRisk` and `MarketExecutor` once per interval. `current_weight` is measured
against the single portfolio value. Its optional `minimum_hold_bars` can be one
common duration or an ordered per-symbol vector; age advances from actual signed
fills, the locked target is rebound to the exact filled quantity, and risk
projection still runs afterward. Optional terminal settlement reserves the
latency-aware close interval and sends a portfolio-wide flat target through the
same risk and execution path. Any residual quantity or active order remains
visible at the end. Duration/settlement runs emit shared-cash ledger schema v2,
including requested/effective intents, per-symbol ages and quantities, and
suppression/unlock evidence; calls without those options retain v1 ledger shape.
This replay capability does not make the existing per-symbol PPO training
environment a joint portfolio learner, and does not change the independent-account
meaning of the currently frozen v1 Study.

Adaptive rule exits use the optional `StrategyObservation.gross_position_return`
provided by canonical replay. Replay derives it from the actual average entry
fill price and the current bar-close mark, signed by the filled position. It is
a gross mark-to-fill return: fees after entry, funding, and borrow are excluded.
Replay also supplies the exact `current_position_quantity` from the filled book.
Adaptive state follows that signed quantity rather than `current_intent`, which
records the last effective target and can already be FLAT while a missed or
partial exit leaves the book invested.
Take-profit, stop-loss, and trailing thresholds are checked at a decision bar
close. A trigger sends a flat intent to the next eligible execution step and
remains latched until the filled quantity is actually zero, including through a
partial or missed fill and a mark recovery below the trigger. Protective exits
bypass the voluntary minimum-hold constraint.
Latency, gaps, liquidity, and costs can move realized results past the threshold;
these triggers do not guarantee a profit or cap a loss.

New Observation-v3 Candidate Runs that include the shared-cash PPO replay use
`lean_candidate_result_v7`. The artifact binds the combined return series,
terminal cash / quantities, active-order and settlement state, and versioned
shared-cash ledger digest; loading recomputes return and maximum drawdown from
the return series. This provides the v2 Study's single-account comparison input
while preserving v1 per-symbol selection semantics.

### PPO training layout

`fit_ppo_strategy(normalize_features=True)` explicitly fits one immutable
local-feature standardizer on finite/available training decisions in the selected
symbol/window scope. Each symbol contributes equal total weight per feature;
the pooled variance includes between-symbol mean differences. Scales at most
1e-12 use 1; clipping and online statistic updates are absent. Both layouts and
the returned strategy share that fitted transform. Missing inputs stay zero,
and masks/staleness/intent/weight keep their v2 semantics. The default remains
the raw v2 encoder and persisted default observation payload.

Persisted PPO inference must bind feature semantics as well as model bytes. The
current `ppo_inference_bundle_v1` is write-once and supports both raw and
normalized PPO: it binds policy bytes, Observation v2, selected feature
indices/names and the optional fitted normalizer under one manifest digest.
`load_ppo_inference_bundle` requires that digest and the current feed's ordered
feature names. The fitted PPO strategy itself retains the selected feature names
derived from its training Dataset; publication rejects a caller-provided feed
whose selected names differ from that strategy-owned binding. The loader restores
that binding from the verified manifest, rejects selected-feature semantic drift before policy
deserialization, verifies policy bytes and SB3 spaces, and restores inference on
CPU. Historical `ppo_normalized_model_v1` bundles remain readable through
`load_normalized_ppo`; historical standalone research `model.zip` evidence is
not silently promoted to an inference-safe bundle.

Inference-bundle publication uses the shared atomic directory primitive. A
transient Windows permission failure is retried a bounded number of times only
while staging remains a regular directory and the target is absent. Exhaustion
or changed source/target state fails closed, and the publisher removes its own
staging directory; it does not publish a partial copy.

The no-refit replication verifier validates the stored slot-result schema
separately from the replay payload. Slot publication replaces the directional
evaluator's top-level `schema`, so fresh replay has no persisted schema peer; the
verifier excludes only that replay field before canonical payload comparison and
still requires every other field to match exactly. This repairs a verifier
comparison contract and does not by itself verify an existing execution artifact.

PPO environment/fitter callers can explicitly provide an immutable
`PreTradeRiskConfig` via `risk_config`. The same configuration applies when the
environment is created and after every reset, in both sequential and interleaved
layouts. Omission preserves the legacy execution-leverage-limited risk with
drawdown start/stop 1.0. This is a training-only opt-in; it does not alter policy
observations, rewards, action meanings, or replay risk. Research callers must
bind the explicit training configuration in their protocol and separately verify
that evaluation risk matches the intended deployment objective.
With the implicit default, PPOTradingEnv skips projection only when the proposal
is finite and within the execution-derived weight limit and drawdown is within
`[0, 1]`, where the default risk transform is an exact no-op. Explicit risk
configurations, invalid drawdown values, and proposals outside that limit
continue through `PreTradeRisk.constrain` so its validation and projection remain
in force.

Directional PPOのfitとdevelopment評価は `DIRECTIONAL_BASE_EXECUTION_COST` を共通authorityとして使う。zero overlayでもDataset由来のfee / spread / funding / borrowは消さず、特に `borrow_rate_multiplier=1.0` を学習・評価の両方で維持する。過去のPPO evidenceは生成時の旧implementation SHAにbindされたままであり、このcorrected execution contractのcontrolとして自動再利用しない。

`fit_ppo_strategy` の既定は従来どおり `sequential` であり、単一 `PPOTradingEnv` がfit symbolをfull-window episode単位でround-robinする。既存Studyやcandidateがlayoutを明示しない場合の意味は変えない。

fit symbolが複数あるsequential trainingでは、requested `total_timesteps` をSB3の`n_steps=2048` rollout単位へ切り上げた実効budgetが、各fit symbolへ最低1回のfull agent episodeを割り当てられることをfit前に要求する。episode長は`policy_stop_index - start_index`であり、terminal settlement用の内部barはagent transition数へ数えない。これは全symbolのnon-zero coverageを保証する下限であり、各symbolのsample数を完全に等しくする契約ではない。single-symbol fitとinterleaved layoutにはこのsequential coverage gateを適用しない。

Directional PPOはfinite-horizon endpointをdevelopment replayと揃えるため `settle_terminal_position=True` を明示する。agent decisionは `stop_index - order_latency_bars - 1` より前だけで行い、その後の予約区間では環境が `FLAT` proposalを同じ `PreTradeRisk` と `MarketExecutor` へ1 barずつ流す。forced settlementをagent actionとして偽装せず、settlement区間はagent transitionではなくepisode末尾のterminal economic costとして1つのterminal transitionへ畳み込む。このためsettlement内部へPPOのdiscount factorを別途適用せず、terminal rewardにはagent intervalのlog returnとsettlement各intervalの実現log wealth changeを加算する。capacity / turnover / venue admissionで完全flatにならない場合は残余を隠さない。normalizerはagentが実際に観測するdecision rowsだけでfit/validateする。これはper-symbol training endpointの補正であり、developmentのshared-cash cross-symbol accountingまで同一になったとは主張しない。generic PPOの既定は `settle_terminal_position=False` のままである。

Age-aware Observation v3の保有期間比較は、generic PPO既定値をそのまま使わない。candidate-run、resolved-run、candidate-suite、candidate-result各境界は、v3なら `settle_terminal_position=True` と明示的な `PreTradeRiskConfig` を要求し、`drawdown_stop` が20%を超える設定を拒否する。学習と全symbol replayは同じrisk configを受け取る。これは注文前のstopであり、gapや執行損で実現drawdownが20%を超えない保証ではない。

`interleaved` は明示選択する学習layout capabilityである。fit symbolごとに同じ `PPOTradingEnv` を `symbol_indices=(その1銘柄,)` で固定して1個ずつ作り、in-process `DummyVecEnv` で同一policyへ束ねる。観測、reward、execution/accounting、hard risk、network、entropy係数、総 `total_timesteps` は変更しない。callerは `rollout_steps_per_env` を結果を見る前に明示し、`rollout_steps_per_env × env数` が既存PPO minibatch size 64で割り切れることを要求する。

Stable-Baselines3は全rollout単位で学習するため、requested `total_timesteps`と実際の`model.num_timesteps`は一致しない場合がある。`expected_ppo_realized_timesteps`がlayout別の丸め後step数を定義し、fit直後に実値を照合する。`lean_candidate_result_v3`はrequested/realized step数、layout、rollout長を記録し、load時にfit symbol数から再計算して検証する。新規のduration/risk Runは`lean_candidate_result_v6`を使い、保有期間、Observation schema、terminal settlement、全評価期間のカバレッジ、training suppression count、明示pre-trade risk configも記録・検証する。従来のv5 artifactは互換読込するが、全期間カバレッジ検証済みとは扱わない。layout比較では同じrequested値だけでは不十分であり、baselineとcandidateのrealized transition数も一致させる。

`A2CIntentStrategy` と `fit_a2c_strategy` は、PPOと同じprivate 3-action intent adapter、`PPOTradingEnv`、Observation v2、fit-scope専用 `PPOFeatureNormalizer` を再利用する。A2Cはsequential layoutだけを許し、各fit symbolに最低1 nominal full-window episode分のstep budgetを割り当てられるか、rollout `n_steps=5` 単位へ切り上げたeffective step数でfit前に検証する。このcoverageはbudget上の容量であり、risk termination等が起きる実行中に各symbolのtransitionを観測した証拠ではない。

requested step数 `T` は整数演算で `ceil(T / 5) × 5` へ丸める。fit metadataにはrequested/effective step数、rounding rule、seed、fit symbol scope、episode長、nominal coverage下限、および設定を記録する。A2Cは `MlpPolicy`、learning rate `0.0007`、gamma `0.99`、GAE lambda `1.0`、entropy coefficient `0.0`、value coefficient `0.5`、max gradient norm `0.5`、RMSprop epsilon `1e-5`、`normalize_advantage=False`、shared `[64, 64]` Tanh policy/value nets、CPU、seedを明示する。RMSpropはalpha `0.99`、weight decay/momentum `0`、centered `false` とし、policy/optimizer引数もfit metadataへ記録する。rollout単位の丸めによる追加は最大4 stepで、PPOと同じ計算予算や公平な比較を意味しない。これはsoftware capabilityであり、economic upliftは示さない。

`a2c_inference_bundle_v1` はPPO bundleとは分離したA2C専用schemaで、fit metadataを必須とする。manifest digest、Observation v2、選択featureのindex/name、normalizer、A2C policy bytesのSHA-256を結ぶ。load前にmanifest・feed feature schema・fit metadataを検証し、policy bytesはdigest確認済みのprivate copyからCPU上のSB3 `A2C.load` へ渡す。読み込んだpolicyのspacesを確認してからstrategyを返す。保存時にはPPO policyをA2C bundleとして扱うことも拒否する。この形式はsoftware上の安全な再利用契約であり、profitabilityやdeployment適格性の証拠ではない。

このlayoutは学習sampleの並び方を変える実装能力であり、性能改善・profitability・winnerを意味しない。developmentで比較する場合は、exact layoutとrollout stepsを別Controlled Factorとして結果前にpreregisterする。PPOの学習deviceはCPUへ固定し、同じsource/runtime identityがGPU有無だけで別のSB3 execution deviceを選ばないようにする。interleavedではSB3がsub-envへ異なるreset seedを配るため、execution RNGをfactorへ混ぜないよう`slippage_std > 0`の確率的slippageは現時点でfail closedにする。

PPOの**明示constructor / policy construction surface**はSB3 defaultへ暗黙委譲しない。current sourceはlearning rate `3e-4`、sequential `n_steps=2048`、minibatch `64`、`n_epochs=10`、`gamma=0.99`、`gae_lambda=0.95`、clip range `0.2`、value clipなし、advantage normalizationあり、entropy coefficient `0.0`、value coefficient `0.5`、gradient clip `0.5`、gSDE無効、`target_kl=None`を明示する。MlpPolicy側も既存のsmall `net_arch`に加えてTanh activation、orthogonal initialization、`FlattenExtractor`、shared feature extractor、Adam optimizer、Adam epsilon `1e-5`を明示する。interleavedでは`n_steps`だけresult-blindに指定された`rollout_steps_per_env`へ置換する。

これは**列挙したconstructor/policy defaultへの暗黙依存を除く契約**であり、Stable-Baselines3内部の完全な学習アルゴリズムをrepository sourceへ複製・freezeしたという意味ではない。rollout buffer、loss/advantage計算、optimizer実装その他のlibrary内部semanticsは、pinned Stable-Baselines3 / PyTorch versionとruntime provenanceが引き続きimplementation authorityの一部である。dependency versionやその内部semanticsが変わった場合は「同じPPO contract」と推定せず、新implementation identityとしてsource review・real integration・必要なresearch preregistrationをやり直す。これらの明示値は既存pinned runtimeで既に有効だった値の固定であり、economic resultを見たhyperparameter tuningではない。

## StrategyとRiskの責任分離

Executionの `max_leverage` から導く既定pre-trade riskは
`PreTradeRisk.default_for_execution` を単一のsemantic authorityとする。
PPO trainingとcanonical replayはprivateなdefault-risk factoryを持たず、この同じ
authorityを使う。これは閾値変更ではなく、training/replay間の将来driftを防ぐ
ownership契約である。


Risk / executionが担当するもの:

- maximum gross exposure
- maximum absolute weight
- leverage / margin / insolvency
- drawdown emergency
- liquidity / participation capacity
- minimum notional
- tick / lot constraints
- inactive / untradable state
- malformed/invalid stateのfail closed

同じentry/exit判断をstrategyとriskへ二重実装しない。

strategyのlogical intentから作る `desired_quantity` はrisk適用前のproposal stateである。静的なhard cap（drawdown縮小を伴わないmax gross / max absolute weight等）は、transient projectionが同時発火していない場合だけbounded targetへ再束縛できる。`max_turnover` が理由に含まれるtargetはhard capや `hard_risk_turnover_override` が併記されても収束途中のstepであり、次barの `desired_quantity` へ書き戻さない。これにより、価格drift後のLONG→SHORT反転でも最初のdeleveraging targetへproposalが凍結せず、元のSHORT proposalから毎barturnover stepを再計算する。`drawdown_deleveraging` もその時点のdrawdownから毎bar再計算するtransient projectionであり、制約後quantityを次barの `desired_quantity` へ書き戻さない。同じdrawdown・同じ価格・同じintentならdrawdown scaleはidempotentに同じtargetを返し、50%→25%→12.5%のように同じscaleを自己再適用しない。single-symbol replay、shared-cash replay、PPO training envはこのproposal/risk state分離を共有する。

## Execution / accounting authority

P&Lの正本は `MarketExecutor + BookState` の一経路である。

The opt-in `MarketExecutor.insolvency_valuation="retain_debt"` policy preserves
signed marked equity as cash when an economic termination flattens quantity and
margin. A short gap can therefore leave a negative terminal cash balance. The
default `floor_zero` policy retains the historical zero floor and its existing
execution-policy digest. The debt policy wraps the resolved economics digest
with a distinct versioned identity; orders from the other policy are rejected.
Validated policy assignment participates in digest-cache invalidation.

In debt mode, processing stops at the first economic termination phase, including
open marking, fills, dividends and carry. The terminating mark and diagnostics
remain recorded, but later carry, fills and bars are not consumed. Multi-bar
results report the actual `bars_advanced` and matching `next_index`; the legacy
zero-floor mode keeps its historical full-interval processing.

This is an absorbing marked-debt valuation, not a new liquidation execution
model: it adds no liquidation fill, extra liquidation fee or later debt interest.
Legacy interval net/log-return diagnostics keep their bounded semantics. New
fixed-capital profit consumers must use signed book equity rather than those
legacy ratios; the flag alone does not change a reward or authorize research.

Weight-to-quantity sizing uses the current book's mark prices, the same valuation
basis as its equity and weights. An entirely cash book uses the current dataset
mark for its first entry sizing, including `BookState.zero` callers that omitted
initial prices; placeholder book marks must not determine entry units. This does
not mutate the caller's book. Trading close remains the submission reference
for order identity, limit/stop offsets, and order-completion diagnostics. Distinct
mark and trading prices must not generate a rebalance for an unchanged quantity
proposal. Direct low-level reconciliation callers that omit valuation prices
retain their explicit reference-price sizing contract.

不変条件:

- order submissionとfillを区別する。
- feeはrealized fillに対して一度だけ計上する。
- spread / impactを複数channelで二重控除しない。
- `interval_net_return` は実約定・全明示cash flow反映後の最終equityを正本とする。`interval_gross_return` は同じ実約定経路について、最終equityへexecution costとborrowを戻し、signed funding・dividend・cash interestを除いて価格損益を分離する。OPENからのasset returnへ事後weightを掛ける近似や、cost-zero条件で戦略を再実行した反実仮想とは扱わない。
- partial fill後のpositionはrealized fill quantityで更新する。
- `fill_ratio` と `unfilled_turnover` は注文の消化状態を表すため、requested側と同じsubmission reference priceでfilled quantityを評価する。adverse/favorableな実約定価格の変化だけで注文残量が消えたように見せない。`filled_turnover` は実際に売買した金額を表すためactual fill notional / starting equityを維持し、このcompletion指標とはprice basisを分ける。
- lot数量はdecimal表記をexact rationalへ変換し、承認された整数lot数を保有・注文残量の共通authorityとする。任意の初期端数は保持し、float表示はゼロ方向へ保守的に射影する。float表示値の足し引きで次の残高を作らない。
- 数量のcanonical文字列解析とFractionからfloatへの射影は、入力の意味を変えずに反復計算を省くため、標準型かつサイズ上限内の値だけを保持する2,048-entry LRU cachesを使う。型サブクラスや大きな入力はcacheを迂回し、canonical判定・有限範囲判定・保守的射影は従来どおり実行する。
- per-barのcorporate-action処理は、すべてのsplit factorが厳密に`1.0`のときだけ`BookState.apply_split`を省略する。1.0と異なる係数は、注文cancel閾値より小さい差でも正確に適用し、order-cancellation toleranceは従来どおり注文をcancelするかの判定だけに使う。
- signed fillのcash移動は承認された数量のfloat射影・価格・contract multiplierから求め、feeを一度だけ引く。clone、split、settlementはexact残高を引き継ぐ。明示的なabsolute target指定だけはexact旧残高との差額を会計してから新残高へ置換する。
- capacityによる部分約定は、元注文の整数lot上限内で、実際のfloat約定金額がcapacity以下となる最大lot数を探索する。逆算の割り算誤差で1 lotを失わず、quantity/capacity上限へ丸め許容幅を加えない。
- PendingOrderはcanonical rational文字列の累積約定数量を保存し、JSON再読込後も残量を再現する。旧float-only payloadは記録済み値として読めるが、過去に失われた精度を回復したとは扱わない。最小発注額や真のsub-lot rejectionは緩和しない。

Explicit `OrderIntent.reduce_only=True` supports MARKET orders only. Admission
rejects absent, same-direction or insufficient actual exact inventory, independently
of hypothetical pending fills in the economic projection. Allocation then
checks the actual priority sequence, including ordinary fills, and caps each
closing fill at the opposite inventory remaining. Exhausted active remainders
expire explicitly; true sub-lot inventory stays visible. Fees and capacity apply
only to actual accepted lots. Without an explicit source profile, minimum notional
and all existing constraints still apply. A legacy zero-target MARKET request is
represented internally as a reduce-only exact-inventory close, but this does not
waive its minimum-notional or other execution constraints; ordinary same-side
reductions remain ordinary orders without an explicit profile.

The optional `MarketOrderProfile` binds verified Dataset identity/full symbol order,
selected source rules, raw source hash/retrieval, one-way USD-M assumption and the
strict `reduce_only_exits` flag into execution-policy identity, including rule stress.
Only the Binance raw-source builder/loader constructs supported immutable profiles;
artifact loading requires an external digest and rederives all fields from saved
duplicate-free raw JSON. Selected contract multipliers must be one. Current filters
applied to historical bars are an explicit current-snapshot assumption, not historical
filter reconstruction or a verified account configuration.

Selected MARKET orders intersect dataset, runtime and source lot grids by exact
decimal LCM. Stress intersects this common grid with its scaled grid; rule-burden
diagnostics report the actual stressed/nominal common-grid ratio. Admission and
allocation enforce source min/max quantity, including capacity-clipped fills; an
over-maximum request is rejected, not split. Ordinary notional retains the maximum
dataset/runtime/source floor. With the flag enabled, actual same-side reductions or
zero targets create reduce-only orders and waive declared venue notional only;
the explicit runtime floor still receives notional stress. Reversals and unselected
symbols retain ordinary minima. False provides a matched ordinary-order profile.
Across target reconciliation, an equal outstanding residual is reused only when
its Dataset identity, execution-policy identity and reduce-only flag all match the
current request. A stale-identity residual is cancelled and replaced before
admission rather than being reused only to fail later with `identity_mismatch`.
Exact closing deltas still project conservatively to float requests. A
capacity-sufficient no-lot reduce-only fill consumes the exact inventory when its
request covers the projected position; lot- or capacity-limited fills retain their
exact remainder. No dust is written off.
Selected non-MARKET orders and the compatibility `liquidate_at_close` shortcut fail
closed in profile mode; use explicit stateful orders. Side permissions, funding,
costs and margin keep their existing owners. The profile is not a complete live
exchange-filter emulator or a strategy qualification.

Shared-cash directional evaluation accepts this optional profile through the
common replay/executor path. Explicit-profile results use
`directional_market_profile_arm_v1`, bind the evaluated policy/profile identity,
and retain exact terminal quantities. Their flatness gate requires exact zeros;
the omitted-profile `directional_arm_v1` result and historical float tolerance
remain unchanged. Strategy factories, costs, hard risk and training are not
altered by passing a profile.

Stateful execution rechecks exact inventory immediately before each closing fill:
margin handling after a previous fill can invalidate precomputed allocations.
An exhausted or newly insufficient position expires the closing remainder without
a fill. Other orders retain their original capacity reservations; any released
capacity stays unused. Events and final capacity evidence count actual fills only.

The flag is strictly boolean and true changes the order ID. Intent and event
readers accept legacy mappings without the field as false. Their explicit
`canonical_payload()` omits false to preserve default payloads; generic dataclass
serialization includes the new false field and is not byte-identical to old
mappings. Restoring an intent recomputes its identity, so changing or stripping a
true flag without changing the ID fails. Pending partials preserve the flag.
- 金額は既存のfloat契約を維持し、allocationとcashで同じ約定数量の射影・価格・multiplierの乗算順を使う。OrderEvent v1はfloat数量のままで、極端な非表現可能lot積のlossless ledgerとは主張しない。
- minimum-notionalのadmission価格はそのbarで決定可能なLIMIT実行価格に合わせる。LIMITがprocessing openですでにmarketableならopen価格を使い、まだmarketableでなくbar内touch待ちなら `limit_price` を使う。したがってfavorable gapで実際にopen約定できるorderをlimit値だけで誤rejectせず、逆にopen約定のactual notionalがminimum未満なのにlimit値だけで誤admitしない。MARKET / STOP_MARKETの既存reference-price admission、projected leverageのreference-price評価、fill-time allocationのactual execution-price再確認は変更しない。
- `ExecutionRuleStress` が有効な場合、tick / lot / minimum-notional / adverse tick-roundingの変更は `MarketExecutor.execution_policy_digest` へbindする。nominal/default stressは既存cost digestをそのまま維持する。これによりnominal policyで作られたworking residualをstressed executorが同一policyとして再利用しない。
- effective `tick_size > 0` のとき、外部/manual LIMITの `limit_price` と STOP_MARKETの `stop_price` はそのpoint-in-time tick grid上でなければadmissionでfail closedする。floatへの射影で生じる数ULPの表現誤差だけは同一grid点として扱い、実質的なoff-tick boundを後段のexecution-price roundingで別価格へ自動変換しない。一方、canonical target reconciliationがoffsetから内部生成するLIMIT/STOP boundは、未来のeligible-bar ruleを先読みせずsubmit時点のeffective tickだけを使って保守的にgridへsnapする（LIMIT buyは切り下げ/sellは切り上げ、STOP buyは切り上げ/sellは切り下げ）。eligible時点でruleが変わっていれば通常admissionが再検証する。`tick_size == 0` は従来どおりprice grid未指定として扱う。
- maker/taker cost分類はorder typeだけでなくrealized liquidity roleに合わせる。LIMITがそのprocessing barで初めてeligibleになり、同じbarのopenで既にmarketableなら、そのfillはliquidity-takingとしてtaker feeとfull spreadを使う。新規LIMITがbar内touchまでrestする場合、および以前のeligible barからcarryされていたLIMITが後続bar openでcrossする場合はresting orderとしてmaker feeとhalf spreadを使う。MARKET / STOP_MARKETは従来どおりtakerである。この分類はfill価格・数量・capacityを変更しない。
- fundingは対象時刻・符号・quantityに対して一度だけ計上する。
- borrow、mark-to-market、liquidationを別channelで追跡する。session calendarでclose-to-close間隔がnominal barより長い場合、closed-session gap分のcash interest / borrowはnext-open fill前のbookへ、processing bar分はfill後のbookへ適用する。continuous cadenceではこの分割は発生せず、従来の1-bar elapsed carryと等価である。
- terminal mark-to-marketとforced closeを混同しない。
- OHLCVだけからqueue positionやhidden liquidityを再現したとは主張しない。

`MarketExecutor` の標準executionはnext-openであり、decision row `t` の注文は最初にrow `t+1` のopenで約定可能になる。participation capacityのvolume authorityは `ExecutionCostConfig.processing_bar_volume_capacity` に明示bindする。

- `True` は既存互換のlegacy modeであり、processing bar全体のvolumeをcapacity poolへ使う。同一barのopen時点ではbar最終volumeは未確定なので、これはpoint-in-time liquidity forecastではない。既存canonical Dataset / Study / Runの意味を変えないためdefaultとして保持する。
- `False` はcausal stress modeであり、processing barの直前に完全終了したbarのvolumeをcapacity poolへ使う。base-volumeをmarket notionalへ換算するときも、その前barのcloseをreference priceに使う。current processing barの最終volumeはcapacityへ使わない。
- volume unitもcapacity authorityの一部である。QUOTE_NOTIONALは既存のquote-notional poolだけを使う。BASE_ASSETとCONTRACTSは既存reference-price quote-notional poolに加えて、raw base quantity / raw contract countから導くnative quantity poolも同じparticipation limitで拘束し、actual fill priceがreferenceより低いだけでsource volume以上のquantityを生成しない。BASE_ASSETの注文quantity換算は `raw base volume / contract_multiplier`、CONTRACTSはraw countをそのまま使う。trigger-segmentのavailable-volume fractionはquote/native両poolへ同じ割合で適用する。
- どちらのmodeも同じspread / impact / fee / accounting経路を使い、mode差だけで別のexecution-policy digestになる。`False` は将来volumeの予測モデルではなく、同一bar最終volumeへの依存を除くための保守的stressである。

Candidate runのexecution overlayがzeroでも、datasetに含まれるpoint-in-time fee/spread等までzeroになるわけではない。既存execution fieldはcanonical executorを通る。

## Independent per-symbol evaluation

同じfrozen model/policyを共有しつつ、strategy/controller wrapperは各銘柄ごとにfresh instanceを生成して独立Replayする。前の銘柄で更新されたwrapper内部状態を次の銘柄へ持ち越さない。

```text
universal frozen model/policy
  ├─ fresh wrapper → symbol A independent replay
  ├─ fresh wrapper → symbol B independent replay
  └─ ...
```

Aggregate P&Lだけを成功判定の正本にしない。ある銘柄の利益で別銘柄の損失を隠さず、各symbol × strategyについてreturns、drawdown、turnover、execution cost、funding、borrow、trade/fill/rebalance diagnostics、terminationを保持する。

### Trading-bot tuning and holdout reporting

`evaluation/bot.py` はshared-cash replayへ既存のnon-zero `ExecutionCostConfig()` を渡す。zero-cost replayは呼び出し側が明示的に指定した場合だけ使う。Hyperparameter selectionは時系列の先行windowだけを使い、baselineと選定candidateのperformance reportは後続holdout windowのfresh replayから作る。`tune_all_strategies` の順位もholdout returnではなく同じtuning window上のobjective scoreに従う。

Bot CLIの`--signal-feature`はDatasetの一意なfeature名を実行前に解決し、対応する`signal_index`をbaseline、cash、全candidate、全walk-forward foldへ固定する。未指定時は従来の0列目を維持する。公開tuning/comparison APIも同じindexを明示でき、boolean・負値・非整数・source Dataset範囲外のindexはchannel追加やreplay前に拒否する。configのindexとDataset identityがfeature semanticsをbindする。channel戦略とconstant/cash controlは従来どおり各自の入力・intent契約を使い、この指定でchannel定義やcontrolを変更しない。

上限付きparameter searchは各parameter axisをまたぐdeterministic sampleを使い、grid先頭のprefixだけに偏らない。`BotReport` のreturn interval数、positive rate、profit factorはbar interval単位のmetricsであり、closed-trade metricsとは呼ばない。Sharpe annualizationはreturn seriesのperiod metadataを使う。`--mode compare` は全期間のin-sample diagnostic rankingであり、holdout selectionではない。baseline P&Lがほぼzeroの場合、relative improvement percentageはundefinedとして報告する。

CLIは既存の `--dataset` directoryまたは明示的な `--demo` のどちらかを要求する。省略時や明示されたpathが存在しない・directoryでない場合は入力errorとして終了し、optimize/compareをsynthetic demo dataへ暗黙にフォールバックしない。

単一strategyのtuning resultは、parameter selection後の後続window reportを返す。一方、`tune_all_strategies` は五つのfamilyそれぞれの後続window reportを表示するため、familyを選ぶ目的で比較した時点でそのwindowはdevelopment evidenceになる。出力の `report_scope=development_family_comparison` はこの意味を示す。family選択後にfinal out-of-sample claimを行うには、さらに後の未閲覧windowで再評価する。

Candidateの選定適格性はtuning-window shared ledger maximum drawdownが20%以下であることを要求する。20%はselection vetoであり、pre-trade stopやholdoutのrealized drawdownを20%以内に保証しない。価格gap、約定損、terminal settlementで観測drawdownが20%を超える場合があるため、holdout drawdownはそのまま報告する。

`balanced` は符号付きのtotal return / max(drawdown, 0.1%)へinterval profit-factor bonusを掛ける。損失の符号を反転せず、損失candidateをcashや正returnより高く評価しない。この比率は年率換算Calmarではない。tuning candidateは終端のexact quantityが全て0、active order remainderなし、economic terminationなしであることも必要とする。Bot reportはmarked equity/P&Lと`terminal_settled`、残余quantity、active order、termination reasonを同時に保持する。決済不能の後続reportはそのまま表示し、利益や決済完了へ書き換えない。ledger evidenceを伴わず手動作成されたreportの`terminal_settled`は`None`とし、未確認の決済を真として扱わない。

各tuning prefixでは既存baselineとparameter gridに加え、同じcapital・cost・windowのcash replayを必ず比較する。cashはgrid budgetを消費しない独立controlであり、同点ならcashを優先する。cashの実損益を使い、cash interestが存在するDatasetをzero returnへ書き換えない。zero-P&L cashのSharpe objective scoreは0とする。後続windowの損益は選定に使わず、正のprefixから選んだ取引candidateが後続で損失になっても、その結果をcashへ差し替えない。requested familyは`TuningResult.strategy_name`、実際に選んだfamilyは`optimized_config.strategy_name` / `optimized_report.strategy_name`で区別する。取引候補が全て不適格でも、cashの実replayが適格ならcashを返す。cashを含む全候補が不適格なら成功を捏造せずerrorにする。

Bot reportはbookの`total_execution_cost`、符号付き`funding_pnl`、`borrow_cost`、`turnover_total`、`fill_count`、`rebalance_events`も保持する。cost/funding/borrowはaccount currency、turnoverは各fill notional / interval開始equityの和であり、ドル額やclosed-trade countとは呼ばない。診断値はP&Lへ再加算・再課金せず、execution/accounting ownerの実測値を報告する。手動で作る従来の`BotReport`では未提供の診断値を`None`とし、未計測を0へ偽装しない。

`walk_forward_tune` / `--mode walk-forward` は同じ探索・score・eligibilityを使い、直前foldで選定して次foldをfresh capital / strategy stateでreplayする。各windowの`report_scope`も`development_walk_forward`とする。`profitable_windows`にはmarked P&Lが正で、かつ終端決済が確認されたreportだけを数える。最後のfoldには割り切れない残余barを含め、隣接するevaluation intervalは重複しない。最後のmarkは次windowの開始markにもなる。後のtuningで既に評価したfoldを再利用するため、全体は`development_walk_forward`であり、独立標本やsealed final evidenceではない。各windowは同じinitial capitalへresetし、`cumulative_return_pct`は正規化returnの仮想積である。capacityやorder sizingを再投資capitalでreplayしたcontinuous wealth pathを表さない。

Botのchannel戦略は`channel_entry_upper/lower`、`channel_exit_upper/lower`をfeature名で解決し、不足時はreplay前にrejectする。閾値ではなく正規channelの符号を使う。synthetic demoはEMA signalと既存のprior-candle channel builderを使い、現在足をchannel extremaへ含めない。adaptive botのregime判定は設定signalの絶対値を使うmomentum判定であり、独立したrealized volatilityの推定ではない。

`AdaptiveProfitConfig`はfinite・nonnegativeなthresholdと、nonnegative integerのfeature index / max holdingを要求し、booleanも拒否する。entryはpositiveで対応exitより大きいことを要求する。zero regime thresholdによるtrend固定と、zero protective thresholdによるexit無効化は明示的な有効設定として維持する。NaNでprotective comparisonを黙って無効にする設定はreplay前に拒否する。

## Artifact and evidence rules

- Canonical JSON/digestのauthorityは `trade_rl.artifacts` に置く。
- Market dataset artifactのcodec/publicationは `trade_rl.data.artifacts` が持つ。
- Candidate/evaluation runはimmutable filesystem artifactとして残す。
- Candidate RunとStudy EvidenceSetのdirectory publicationはstagingからatomic renameする。Windowsで一時的なdirectory lockによるpermission errorが出た場合は有界retryし、sourceの消失やdestinationの出現を検知したら失敗する。部分copyへ切り替えない。
- Candidate Runはresolved result `summary.json`、raw interval return `returns.npz`、implementation/runtime/research-context evidence `provenance.json` の3ファイルを一体としてpublishする。
- Candidate Runは実行前後でimplementation/runtime provenanceが一致する場合だけpublishする。実行中にsource/runtime provenanceが変化したRunを正当なevidenceとして残さない。
- Candidate artifact identityはNPZのZIP圧縮表現そのものではなく、summary/provenanceと検証済みreturn arrayのsemantic contentへbindする。一方、各fileのraw SHA-256/sizeもtamper検出用evidenceとして保持できる。
- 同じ入力・設定・identityからはdeterministicなidentityを得る。
- resultを見た後にevidence条件やthresholdを都合よく変更しない。

## Separate directional development composition

`data.features.price_channels` appends prior-window high/low boundaries relative
to the current completed close. It excludes the decision candle from extrema,
requires every window member to have been available, and derives a new content
identity while preserving source prices and economic arrays. The channel rule
only emits intent. The directional evaluator uses the existing shared-cash
executor, records ledger drawdown, and schedules real terminal closing orders.
Failed terminal fills remain holdings and fail the screen. This composition
does not change the canonical five-candidate suite or its historical decisions.

## Separate funding carry development composition

The paired carry capability assembles separate spot/perpetual quote-volume
series with published funding boundaries. It is separate from the directional
SingleSymbolStrategy/Study selection path. Monthly equal-base quantities use
observed prices and remain fixed between rebalances; matched partial orders keep
completing the same target. All orders, fees and funding use the canonical ledger.

Idle USDT is assumed available to futures; spot market value and synthetic short
sale proceeds are not futures collateral. Collateral equals canonical equity
minus spot value. Guard observed-close collateral, pre-fill open maintenance and
post-fill adverse-high maintenance before later funding credits. The fixed
research wallet ratios require observed-close collateral to cover 100%
of perpetual notional and maintenance must cover 50%. These are research guards,
not historical exchange liquidation tiers. Existing gap breaches
cannot be erased by an exit or future funding. Canonical forced termination is
invalid evidence, with unresolved pre-forced-flat position evidence retained.
Carry stops submit actual exits and cannot discard residuals or reset the stop.

The initial capability uses disclosed fixed research fees and perpetual-close
mark proxies, which block production eligibility. A profitable development
replay alone does not establish an operational winner.

The optional carry spot parser requires an externally evidenced halt calendar
bound before replay. It may insert a stale preceding-close valuation mark only
for a missing whole bin contained in a declared halt. It blocks fills and zeros
capacity in every intersecting hourly bin, preserves published prices and elapsed
time, and rejects unexplained, partial-bin or unanchored leading gaps. The default
Binance source remains strict. These stale marks are a valuation limitation,
never executable prices; reopened prices cannot backfill an earlier observation.

Prospective public evidence capture preserves spot/perpetual depth, mark quotes,
settled funding and venue clocks with raw bytes and request/receipt timestamps.
Eligibility means that all ten responses passed source checks; it is not order
permission or paper profitability. Clock reversal, response duration above five
seconds, cross-request span above ten seconds, future/stale futures quotes,
crossed/unordered books and invalid settlement history prevent publication.
The total span uses a shared monotonic anchor. All quotes must still be at most
five seconds old at final receipt, with at most one second of forward clock skew.
Spot REST depth is explicitly receipt-timed because its payload has no exchange
event timestamp. Quoted funding rates remain distinct from settled payments.

Recorded-depth paper matching uses only quotes received after a saved decision,
at most five seconds before execution and ten seconds after decision. Signed
integer lots consume a bounded fraction of the appropriate ordered bid/ask side.
Capacity accumulates exact decimal quantities across levels before lot rounding;
partial residuals remain explicit. The weighted fill includes an adverse buffer,
and taker fees are charged once with contract multipliers. Quantity and notional
admission can reject an order without altering the account. Market-notional
averaging and account-specific venue restrictions remain unmodeled.

The canonical book accepts optional independent valuation prices for a fill;
cash changes at the execution price while existing positions retain their marks.
The historical default still marks at fill prices. Paper execution must not
create a temporary equity peak by revaluing an existing holding at a new order's
fill price. Displayed depth is a modeling input, not a guarantee of real fills.
One snapshot's instrument capacity may be consumed only once by its future journal.

Forward evidence readers reconstruct market and rule summaries from exact raw
responses, verify sidecars, official URL rosters, hashes and capture-wide timing,
and reject partial/failed evidence or duplicate JSON keys. A supplied expected
digest binds the manifest to its caller. Historical verification does not imply
current freshness: consumption-time reads reject future receipts and expired
quotes (five seconds) or rule captures (one hour).

Current rule capture supports trading BTC/ETH spot and USDT perpetual pairs with
market orders. It intersects base and market quantity bounds and their positive
lot quanta, preserves disabled zero market steps and validates price filters.
Market notional applicability flags and averaging windows are retained. These
public metadata bounds do not reproduce private account restrictions or venue
average-price admission and cannot establish production order eligibility.

The prospective paper journal binds an immutable protocol digest to a contiguous
event hash chain. SQLite immediate transactions serialize compare-and-append;
identical idempotency keys/content return the original event, while different
content or a stale expected parent fails. Updates/deletes are unavailable and
database triggers reject them. FULL synchronization and startup recovery of hot
rollback journals preserve the committed prefix across process death. Protocol
identity is checked before recovery; event replay rechecks every canonical body,
hash, sequence and parent. Persistence alone assigns no execution or P&L semantics.

The paper engine rebuilds each decision/execution/gap command from its verified
source references and compares the complete result with the committed event.
It applies transitions on isolated copies and adopts them only after a successful
append. Captures must begin after the previous command; execution additionally
uses its saved decision and still-fresh bound rules. Each instrument's capacity
is consumed at most once per execution capture. Spot midpoint and published
perpetual mark value the account; actual exits use later depth and pay costs.
BookState exposes accepted exact quantities without a reporting-float round trip.

Newly published funding uses exact holdings strictly before settlement time,
including payments first received after an exit. Funding revisions cannot rewrite
cash; late/missing settlements and gaps remain permanent failures. Risk checks
precede funding credits and follow each chronological settlement group; only
simultaneous settlements net. Every later breach remains visible even after a
different permanent stop or the planned terminal close. New nonzero funding at
or before an earlier capture's settlement watermark is a permanent coverage
failure, including later simultaneous fragments; cash still settles once at
receipt, without retroactive edits. This may reject asynchronous publication.
Gaps preserve quantities
and the last actual observation time. No implicit liquidation or flatness is
allowed. Runtime-bound collection and future economic qualification are separate
from deterministic engine correctness.

Public collection checks its pinned protocol and complete source/runtime identity
before acquisition and commands. A process lock admits one collector. A separate
durable control database begins every cycle before network I/O and acknowledges
only fully committed cycles; restart with an unfinished cycle halts even if the
process died before the top-level failure marker. Untracked existing captures
are rejected. Failures preserve a permanent marker and, when the verified journal
can accept it, a gap command after reconciling any uncertain commit. No automatic
retry follows a collection failure. Before beginning another cycle, compare the
collector clock with the last committed command (or the frozen start before the
first command); an elapsed gap above the configured maximum is recorded as a
non-trading gap before network acquisition, preserving exact holdings and the
last actual marks at that time. No missed observations are backfilled. A fresh
quote can still be used to attempt actual exits, while the permanent quality
failure prevents the screen from qualifying. This detects a stale wake after a
completed cycle without claiming why the prior process stopped. Pending decisions
expire after ten seconds;
the terminal observation window ends 180 seconds after the fixed close, including
when acquisition itself crosses that boundary. These are operational controls,
not evidence that a prospective economic gate passed.

New forward captures use `binance_forward_market_snapshot_v2`, explicitly binding
100 requested depth levels per side for each spot/perpetual symbol. All returned
levels are retained; more than the requested bound fails. The offline reader
still verifies historical v1 evidence against its exact 20-level URLs and bound,
without relabelling old captures. The fixed 10% participation, independent leg
fills, fee assumptions and unmatched-hedge stop are unchanged. Deeper observation
is not invented liquidity, and it cannot repair a previously rejected run.

The prospective screen is fixed at ninety UTC days and three thirty-day blocks,
sealed at least five minutes before start with 10000 virtual USDT and unchanged
paper cost/depth defaults. Require positive net profit, a positive causal marked
equity change in every block, and net profit exceeding recorded fees at the
realized trajectory. This fee headroom is not a dynamically replayed cost stress.
Require observed drawdown below 10%, funding receipts in every block, no unpaid
announced settlement for held quantity, no quality failure or nonterminal stop,
all four instruments actually filled, and a final flat account without intent.
Block marks use the last observation within 180 seconds before each boundary;
the final block includes actual exit costs and late settlement receipts.

The offline screen cannot run before close plus 180 seconds. It requires first
observation within 180 seconds, no observation gap beyond 180 seconds and a last
observation from close plus 120 to strictly before close plus 180 seconds. Rebuild
against the external protocol and event-tip anchors under the collector lock,
verify exact source/runtime identity, and reject incomplete control cycles or
unconsumed/missing source directories. Cadence follows future sixty-second slots
without fabricated catch-up observations. A quality-failed run may stop early
once actual exits have made it flat; it cannot qualify. Operational status checks
the journal chain only and must not claim raw-source financial validation.
Even `PAPER_SCREEN_PASSED` retains `production_eligible=false`: observed paper
marks, cost assumptions and public depth do not establish live fill guarantees,
intraminute drawdown, optimal returns or private-account suitability.

## 非目標

Lean coreが保証しないもの:

- profitability
- Production/live order authorization
- OHLCVから観測不能なmicrostructureの完全再現
- model complexityがedgeを生むという前提
- DB/UI/teacher pipelineが研究成立に必須であること

利益やlive suitabilityはarchitectureではなく、凍結した研究条件とunused-data evidenceで別途判断する。

## Optional after-cost scalar allocation

`strategies/allocation.py` owns a pure, independent-symbol allocator. Its input
is a declared expected **simple** return from the current decision valuation to
one explicit `horizon_end`; variance, asymmetric buy/sell costs, future exit,
signed funding, short borrow and cash return use that same horizon. Mean log
return cannot be converted to expected simple return by `expm1(mean_log)`.
`source_identity` and aggregate `available_at` bind declared estimate provenance
and availability; this software does not verify estimator receipts or calibrate
costs. Older forecasts starting at an earlier valuation are rejected rather
than reused as expected remaining return.

For actual weight `w0`, the bounded scalar surrogate is:

```text
U(w) = cash_return * (1-w) + expected_simple_return * w
       - risk_aversion * return_variance * w*w
       - buy_cost * max(w-w0, 0) - sell_cost * max(w0-w, 0)
       - exit_cost * abs(w) - funding_return * w
       - borrow_return * max(-w, 0)
```

This is a concave piecewise quadratic with fixed exposure/optional turnover
bounds. Endpoints, feasible kinks at zero/actual weight, and smooth stationary
points determine its maximum. Exact ties choose actual HOLD, then minimum
turnover, smaller absolute exposure, then signed weight; exact rational
comparisons of the supplied IEEE coefficients prevent rounded endpoint scores
from inventing gains on flat segments. Stationary points are represented as
floating weights. No epsilon hides a small positive improvement. Current exposure outside the feasible interval is
not a valid HOLD. Empty bounds and nonfinite arithmetic fail rather than pass.

`evaluation/allocation.py` composes this optimizer with canonical `PreTradeRisk`
and `MarketExecutor`. `propose_nonrl_target` builds a detached context from the
actual `BookState`, accepted exact quantities and entire `OrderBookState`.
The caller supplies a stable account ID; BookState itself has no account-ID
registry. Context identity includes cash, marks, peak/latched drawdown, margin,
termination, active and terminal orders, decision revision, dataset identity,
execution policy and risk config. Proposal identity additionally includes every
estimate, horizon, resolved allocator config, proposed weight and objective.
`execute_nonrl_proposal` rejects changed or altered inputs before admission,
recomputes the scalar proposal, and applies hard risk once. The returned
`NonRLExecutionResult` keeps the proposal, risk target and canonical execution
result so requested, approved and filled exposures remain distinguishable.

V1 accepts only an independent-symbol account, MARKET orders, zero additional
latency and one processing bar per call. Other-symbol positions/orders,
nonmarket/protective orders, stale valuation marks and terminated accounts are
rejected. Pending clocks must permit the next processing bar; already-expired
or later-eligible residuals and future active/terminal transitions are rejected.
The horizon must cover at least that first processing bar. Canonical margin and
drawdown refresh on a detached BookState copy reject already-dead margin states
and missing latched drawdown evidence; the original book is not mutated.
Risk requires `drawdown_start < drawdown_stop <= 20%`; the canonical latched maximum
drawdown is used. This guardrail cannot cap losses across gaps or missed fills.
Static exposure limits are intersected before optimization; subsequent risk,
quantization, order rejection, capacity and margin may change the submitted or
realized allocation. Thus this is a scalar surrogate optimum **before final
hard-risk projection**, not an executable or globally optimal net-profit claim.

If final approved weights equal actual weights exactly, the execution-owned
`execute_quantity_hold_statefully` cancels all selected-symbol pending MARKET
orders, including reduce-only residuals, and advances without new intents or
weight-to-quantity sizing. Canonical split/carry/mark processing still runs;
splits change quantity units. Hard-risk reductions use the normal target path.
Holding actual quantity differs from retaining an old partially filled target.
The caller continues with both returned book and order book; no second ledger,
free-cash reservation balance or execution compatibility cache is introduced.

Current-close estimates remain a declared proxy because first eligible fills
occur on the next processing bar. Future exit and carry estimates are charged
on initial exposure in the surrogate, whereas realized costs/carry use canonical
fills and marks. The signed self-financing cash term reflects this ledger's
cash, including short proceeds or negative cash; it is not a universal futures
collateral model. Existing intent replay, PPO defaults and historical artifacts
retain their meanings. This family supplies no fit, Study, terminal-liquidation
protocol, shared-capital solver, RL environment or live execution authorization.

## Net-profit objective declarations

Issue #810の新規研究向けに `evaluation/objectives` が事業目的と金融時計を宣言する。既存の独立銘柄口座、PPO log報酬、Run/Study schema、過去の選定基準は変更しない。

`CapitalContract` は独立口座ごとの初期資本、または共有口座1つの初期資本を保持する。`ObjectiveContract.net_profit_rate` は `sum(terminal_equity - initial_equity - signed_net_deposits) / sum(initial_equity)` を計算する。独立口座の単純なリターン平均を共有資金の利益へ変換せず、不等資本では資本加重になる。負の終端equityも残債を含む損失として保持する。金額は同じaccount currencyのcanonical ledgerから渡す。費用をこの計算でもう一度控除しない。

`net_profit_objective_v1` は期間のaware UTC境界、期末決済/継続mark評価、economics/risk/deployment recipeのSHA-256参照、20%以下の研究DD基準、税引前・固定インフラ費別報告・符号付き入出金の意味をidentityへbindする。digest参照はprofile内容の検証や執行の適合性の証明ではない。20%はgapや執行不能時にも守られる損失保証ではない。

`FinancialClockContract` はregular clockのdecision/execution/reward間隔、有限horizon、rollout長、gamma、GAE lambda、reward schemaを明示する。初期契約は1 decisionにつき1 reward、execution刻みへの整合、horizonのdecision刻みへの整合を要求する。時間を揃えたdiscountを比較できるが、rollout切断と経済終端のruntime処理は実装しない。固定初期資本を分母とするequity増分の総和はgamma=1で終端純利益へ一致し、log報酬とは異なる。`terminal_profit_aligned` はこの代数的関係だけを表す。

`BoundObjectiveClock` はUTC評価期間の正確な整数秒数と金融時計のeconomic horizonが一致することを要求し、両宣言のdigestを `bound_objective_clock_v1` に結び付ける。この有限評価期間はfitデータ区間や最大保有期間とは別の意味である。小数秒は丸めず拒否する。これは任意の新規bindingであり、個別宣言のconstructorを変更しない。

これらは独立した宣言・算術capabilityであり、既存runner/env/Studyへ接続されていない。共有口座訓練、費用校正、最終評価、研究実行の認可、利益性は証明しない。

## Direct-simple forecast allocation

The separate `simple_return_ridge_stream_v1` calls the existing mature-row
selector once per prefix, preserves row order and absolute symbol-balanced
weights, and derives realized labels as `end_close / start_close - 1`.
`expm1(mean_log)` does not supply an expected simple return. The shared
numeric-only Ridge solver preserves old log model/payload arithmetic, covered
by fixed and same-runtime regression oracles.

The new model is an **uncalibrated regularized linear projection** of raw
same-close price returns. Its expected-simple unit describes the quantity being
estimated, not a verified conditional expectation. Its frozen
`fit_prefix_marginal_variance` is the weighted population variance of those same
mature labels, unannualized and in squared simple-return units. It is pooled
across fit symbols, not conditional uncertainty, confidence, a covariance matrix
or a loss guarantee. Absolute weights remain unchanged because rescaling them
changes Ridge regularization.

Vintages bind selected inputs, actual traced endpoints/maturity, fit-only model
parameters and marginal variance. Packets bind decision close, selected snapshot,
availability and exact horizon. Delayed simulation packets can be recorded;
allocation v1 consumes only the exact current packet with completion at that
decision and zero inference delay, without stale fallback. Nonfinite arithmetic
or predictions below -1 fail without clipping. The reader pins an external
content digest and checks reconstructed nested schemas, labels, variance, recipe
and predictions; it neither refits nor independently authenticates the original
fit or historical source availability.

`evaluation.forecast_allocation` checks Dataset lineage, symbol, selected feature
names/values/availability and exact clocks. Packet close, Dataset close/mark and
actual BookState mark must agree. Separate `HorizonCostEstimates` bind source,
availability, horizon and initial-notional rate basis. Aggregate availability is
the later model/cost time. `execute_forecast_proposal` rebinds these inputs and
actual account state before existing risk and canonical execution; changed
inputs fail. The caller continues with the returned book, order book and
`next_index` together.

This consumer requires an explicit BookState processing clock: Dataset digest
and last completed index. Canonical stateful execution validates it before
admission, preserves it through cloning and advances it after each completed
bar. Reusing a returned account at an earlier index fails even when marks are
unchanged, preventing repeated dividend/carry processing. Initial clock values
are caller declarations, not authenticated historical receipts. Existing
unmarked books retain their default behavior; old artifacts are not migrated.

The label is a **price-return surrogate**, not held-quantity wealth or net
profit. It excludes dividends, split-adjusted wealth, funding, borrow and cash
carry; their realized events belong to the executor. No future split/dividend
scan admits or suppresses present predictions. A later split can change raw
price units while canonical quantity adjustment preserves wealth. Next-open
gaps also separate realized fills from same-close predictions. These mismatches
need a prospective instrument/label/economics contract and empirical calibration
before market utility can be claimed; recorded prices alone cannot establish an
event-free future horizon. No Study, RL consumer or economic authorization is
introduced here.

## Opt-in allocation PPO account transition

The allocation PPO path uses the same account-bound proposal, final hard risk
and MarketExecutor/BookState transition as the non-RL allocator. Discrete4
records the raw categorical action before projection: quantity HOLD, negative
request, direct FLAT request/residual baseline, positive request. Direct targets
are bounded weights; residual actions move from the exact optimizer target
toward a feasible endpoint. The residual center retains the existing baseline
order identity. Final hard risk runs once and can override HOLD.

Its finite independent account uses regular one-bar decisions, gamma=1 and
after-cost equity differences divided by fixed initial capital. Reward sums
equal the canonical signed terminal equity difference. GAE remains a surrogate;
this does not prove an unbiased terminal-profit gradient. A rollout cut continues
the account and bootstraps. The declared finite endpoint terminates without
another market bar or free liquidation, retaining marked positions/orders.
Economic termination is absorbing in this episode; retain_debt keeps signed
Book equity as terminal cash. Legacy floor_zero and interval/log arithmetic
retain their meanings. This is not a realistic liquidation-fee/debt-interest
model or a 20% loss guarantee.

The versioned observation contains available raw selected features and ordered
forecast/cost/account projections. Pending gross/count and maximum drawdown
are partial state, not complete order history or a Markov-state claim.
The recipe freezes this layout, action mapping, capital, horizon, actual
risk/economics, calendar kind and effective processing-bar duration. A SESSION
calendar's nominal bar duration changes carry timing even with regular decision
timestamps, so it cannot silently share a continuous-calendar recipe. Runtime
declarations are recomputed, not cached assertions.
Execution RNG uses its declared cost seed independently of the policy seed.
The inference bundle pins canonical manifest and policy bytes before loading a
verified private copy. It does not resume an account, optimizer or execution RNG.

## Immutable allocation account snapshot contract

`strategies/allocation_snapshot.py` は、独立した1口座のdecision前factsを
`independent_allocation_account_snapshot_v1` として深くfreezeする下位DTOである。
global symbol indexを保持し、quantity・mark・multiplierのvectorは選択symbolの1要素だけを受け取る。
正のequityを持つlive口座を対象とし、負のcashは許す。terminal口座は対象外とする。
canonical nanosecond clock、source Dataset、processing index、利用可能時刻を整合させる。
active native MARKET orderのdirection、exact requested / cumulative / remaining quantity、
TIF・expiry・reduce-only・statusと因果的なtransition clockを保持する。
報告用quantityはexact rationalから同方向へ保守的に丸めた1 ULP以内の値を許す。
受け取った完全contextの`source_state_digest`と射影factsのdigestは別のidentityである。
このDTOは口座を読み出さず、ledgerの再計算、source・order IDの真正性、予約cash、
position age、numeric PPO observation、policyへの接続を保証しない。

## Independent account snapshot observer

`evaluation.allocation_snapshot.snapshot_allocation_account` is an opt-in,
observer-only reader for a live independent-symbol account with MARKET orders
and zero extra latency. It requires the known matching Dataset/index clock,
current selected-source availability and the existing allocation context checks.
The caller must refresh canonical margin before reading, including bootstrap
accounts whose stored maintenance rate may otherwise retain its default value.
Margin is checked on a detached clone; the reader never repairs the source book.

`independent_allocation_account_snapshot_v1` freezes account facts and full active
order details. Quantity, exact-quantity, mark and multiplier vectors contain one
selected coordinate; `symbol_index` retains its global Dataset slot. Signed cash,
equity, peak/current/max drawdown and margin facts come from BookState. The full
native account and order history remain bound by `source_state_digest`, rather
than exposing another symbol's unavailable mark. Exact order remainders derive
from requested quantity minus exact cumulative fills, not reporting floats.

The producer detaches before cache-refreshing account reads and validates orders
through their existing native reader. Filled active remainders within native
completion tolerance are rejected using the canonical order owner's rule.
Nonzero active filled progress at or below the native absolute minimum fill
quantity is also rejected. Legal small fills above that minimum remain visible,
even when they are below the request-scaled completion tolerance.
IDs/digests are evidence, not numeric
policy inputs. DTO structural checks do not prove source/account authenticity;
bootstrap clock and account ID remain caller declarations. This is neither a
numeric observation encoder nor a terminal reader. Ages, free/reserved cash,
shared accounts, account restart and research/execution authorization are absent.

## Allocation observation v2 layout declaration

`strategies/rl/allocation_observation_v2.py` defines the immutable
`allocation_account_observation_v2` layout. Its required constructor fields are
ordered feature names, order-slot limit K (1..64), fixed initial capital and
normalization episode steps. Capital normalizes to the same positive finite
native float used by AllocationDecision. Detached payload/digest bind these
four declarations and the complete `F+9+13+24K` ordered field names.

The nine forecast/baseline, thirteen account and twenty-four per-order fields
separate current/historical drawdown, signed quantity notionals, margin facts,
TIF/status and optional-clock masks. The payload declares conservative economic
float32 projection, guarded raw features, exact economic sorting, rejected slot
overflow and zero padding. This declaration module generates no numeric
observation and reads no snapshot; the separate encoder implements its layout.
Existing allocation v1 recipes/tensors and policy consumers are unchanged.

## Allocation observation v2 pure encoder

`strategies/rl/allocation_observation_encoder_v2.py` projects a lower snapshot
and AllocationDecision into a fresh `F+9+13+24K` float32 array. It checks matching
account/source/time/symbol, exact position, reported cash/equity/current weight,
historical drawdown, feature order, fixed capital and normalization horizon.
Exact signed order quantities are independent of v1 pending reporting summaries.

Economic ratios use exact quantities and decimal interpretations of reporting
scalars before fixed-capital division, then the closest conservative float32
value toward zero. Overflow and nonzero-to-zero underflow reject; raw features
use guarded ordinary float32 conversion. Order rows sort by exact economic facts
before projection. Offsets subtract integer causal clocks before episode-length
division; masks separate absent clocks from index zero. Excess orders reject;
unused slots are zero. IDs, digests, absolute clocks and symbol index are absent
from numeric inputs. This sort does not reproduce native ID/capacity priority.
The encoder reads no runtime source, invents no ages/reserved cash, establishes
no authenticity/full Markov state. The opt-in runtime connection is below.

## Allocation observation v2 runtime connection

AllocationTradingEnv accepts `observation_schema=None` for unchanged v1 behavior.
Explicit v2 requires matching selected features, fixed capital and episode length.
Only v2 reset refreshes native margin before preparing the decision and admitted
live snapshot; subsequent observations compose that observer and pure encoder.
The same action/risk/execution/reward path owns all native account transitions.

Distinct `allocation_ppo_recipe_v2` / `allocation_ppo_inference_bundle_v2` bind
the complete schema, including F once. Strict reconstruction validates layout,
capital, horizon, feature order and terminal semantics before deserialization;
actual loaded model spaces are checked before prediction. Default v1 recipe,
receipt and bundle bytes are unchanged. True horizon/insolvency termination uses
an all-zero sentinel with terminated=True/truncated=False, without live reading.
SB3 autoresets to live state; its done mask excludes terminal critic contribution.
Only nonterminal rollout boundaries permit bootstrap. Truncation is not generated.

V2 adds an actual-input receipt: actor `obs_tensor` rows, rollout-end `new_obs`
with done/current/last-transition indices, and separate terminal-info sentinels.
Each event hashes immutable C-order little-endian float32 bytes framed by 8-byte
big-endian header/data lengths and canonical phase/index/reset metadata. Actor
count equals actual timesteps; boundary count equals completed rollouts. Empty
phases require SHA256(empty). These consistency receipts authenticate neither
sources nor fitting and do not record RNG/account restart state. Terminal counts
also agree with sampled reset starts and horizon-ending decisions. Correspondence
of boundary new_obs to current SB3 critic tensors is checked by independent spies.
Dataset-source
receipts keep their existing sampled-row meaning. Ages/reservations, native ID
priority, fit-only preprocessing and a complete training protocol remain separate.
