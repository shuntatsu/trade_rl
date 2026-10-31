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

## Financial objective declarations

`trade_rl.simulation.objectives` は、新しいfinite-horizon Study用のpure contractを
宣言する。`CapitalContract` は銘柄別の独立accountなら各bookの初期equityを、shared
portfolioなら一つのbookの初期equityを保持し、分母を初期capitalの合計へ固定する。
`ObjectiveContract` の `net_profit_objective_v1` は、callerが渡すterminal after-cost
equityとnet external cash flowから、一回のrealized endpointを
`(Σ terminal equity − Σ initial equity − Σ net external flow) / Σ initial equity`
として表す。depositは正、withdrawalは負で、external flowはprofitに数えない。
`net_equity_increment` は同じfixed capitalを使うinterval incrementであり、ledgerの
equityやcostを再計算しない。従って、after-cost valuesの供給元と端末処理は既存の
accounting/replay ownerに残る。canonical ID `expected_terminal_net_profit_v1` は
objective identityであり、この宣言が期待値を推定したり利益を保証したりはしない。

`ObjectiveContract` はaware UTCへ正規化した `[evaluation_start,
evaluation_stop_exclusive)`、`settled` / `marked_continuation` endpoint、economics・risk・
deployment recipe digests、最大drawdownの宣言をidentityへ含める。`FinancialClockContract`
は正のinteger秒単位のdecision/execution/reward intervalとfinite horizon、gamma、
GAE lambda、reward schemaを検証し、discountを経過時間から計算する。
`BoundObjectiveClock` は評価期間のelapsed secondsが宣言horizonと完全一致する場合だけ
両contractを結ぶ。`terminal_profit_aligned` は `equity_delta_v1` と `gamma == 1.0`
の代数的条件だけを示し、実環境でのreward parityを証明しない。

これらはopt-in declarationsであり、宣言そのものは台帳検算・Study認可・profitabilityを
証明しない。現行のopt-in `SharedPortfolioGymEnv` は `BoundObjectiveClock` を要求し、
`fit_shared_portfolio_ppo` は金融clockとgamma-one equity-delta rewardを接続する。
ただしlegacy単銘柄PPOの従来reward `log1p(interval_net_return)`、`gamma=0.99`、
`settle_terminal_position=False` の既定は変わらない。これらの接続も新しいfinancial
ledger、winner rule、実市場学習・収益性の証拠を成立させない。

## Data contract

`MarketDataset` はpoint-in-timeで観測可能な情報だけをdecisionへ渡す。

必須条件:

- `feature_available_time <= decision_time` を満たす。featureのinformation/availability timeはdecision timeを越えない。
- `MarketDataset.periods_per_year` はannualization clockを表すpositive integerであり、booleanやfractional/non-finite frequencyは拒否する。
- `trade_rl.data._timestamps` はtyped `datetime64` 値をsigned nanosecond clockへ正確に正規化するinternal helperを一つだけ所有する。raw tickとunit multiplierを整数演算し、範囲とsub-nanosecond precisionをcast前に検証する。public data facadeへはexportしない。
- `timestamps` と明示された `available_at` は、計算に使う `datetime64[ns]` へlosslessに変換できる値だけを受け入れる。nanosecond範囲外へのunit変換によるwrapと、sub-nanosecond精度の切り捨ては拒否する。NaTも拒否し、有効なcoarser-unit timestampの表現は維持する。
- `CsvMarketDataSource` はISO日時本体のfractionをnanosecondまで整数で保持し、integer epochは従来どおり `abs(value) >= 10^12` をmillisecond、それ未満をsecondとして正確に変換する。日時本体のnanosecond範囲外とnon-zero sub-nanosecond fractionは拒否する。小数秒を含むtimezone offsetはmicrosecond精度で反映し、6桁より細かい桁にnon-zero digitがあるoffsetは拒否する（余分な桁がすべてzeroなら値は失われない）。`RawMarketSeries` のtyped datetime配列もshared helperの整数演算でnsへ正確に変換し、1ns遅れた `available_at` を同時刻へ丸めず後段へ渡す。時系列順序は隣接値を直接比較し、Datasetのperiod deltaが正のint64 nanosecondsへ収まらない入力は拒否する。
- `CandidateRunConfig` の `fit_cutoff`、`evaluation_start`、`evaluation_stop_exclusive` もresolve前に元のISO calendar/range/precisionを検証してからnsへ正規化する。timezone offsetのUTC等価化を維持し、nanosecond範囲外の元時刻とnon-zero sub-nanosecond桁は拒否する。typed `datetime64` はshared data helperのexact integer arithmeticで検証するが、呼出し前に既にwrapされた `datetime64[ns]` から元時刻を復元することはできない。
- supervised labelは `label_end_time < fit_cutoff` で完結し、Datasetが宣言するstart/end価格sourceの `available_at <= fit_cutoff` を満たす。
- forecast training builderへ渡す`fit_cutoff`と、`CausalForecastTrainingSet`へ直接渡す`label_end_times`は、`data._timestamps`のshared helperを通し、元のdatetime64 unitと正のmultiplier metadataから正確な整数演算でns値を求めてからsigned-nanosecond範囲を確認する。multiplierが0以下のmetadata、`NaT`、範囲外へのwrap、nanosecond未満の端数はscopeやmaturity比較より前に拒否し、正確に表現できるfixed-duration unitとcalendar `Y` / `M` unitを受け入れる。cast後の逆変換一致だけをoverflow判定には使わない。
- Ridge / LightGBM model constructorも同じlossless cutoff normalizerを使う。両strategy adapterは、保存済みobservation timestampが`NaT`の場合と`fit_cutoff > observation.timestamp`の場合に、feature fallbackやpredictionより前に拒否し、同時刻は許す。これはmodelの宣言fit cutoffに対するas-of floorであり、modelの生成・配布時刻、vintage、decision時点での実運用可能性を証明しない。`StrategyObservation`構築前のtimestamp変換で既にwrapされた入力時刻も復元できない。
- `LeanCandidateConfig`と`CandidateRunConfig`のtyped `datetime64` 値も同じshared helperで正確に正規化する。`NaT`、signed-nanosecond範囲外へのwrap、nanosecond未満の端数は拒否し、losslessなcoarser / multiplier / fine unitは受け入れる。ISO文字列のparse semanticsは変えない。入力前に既にwrapされた`datetime64[ns]`から元時刻は復元できない。
- scaler、normalization、imputation、feature selectionにfuture情報を混ぜない。
- CandidateSuiteのPPO training preflightはevent-time window `timestamp < fit_cutoff` を保ち、`run_lean_candidate_suite` が使う選択fit symbolのraw bar価格について、inclusiveな `[0, ppo_training_stop_index]` のDataset `available_at` がすべてcutoff以下であることを、Ridge / LightGBM / PPOのいずれかをfitする前に確認する。window末尾はPPO rewardとterminal-settlement reserveの価格を含み、cutoffと同時刻のavailabilityは許す。遅着barを黙って除外してfitを続けず、suite全体をfail closedにする。このguardはCandidateSuite固有であり、cutoff parameterを持たないgeneric `fit_ppo_strategy` の契約は変えない。
- developmentやfinal futureをthreshold/model選択へ戻さない。
- 15分barの行数を独立標本数と解釈しない。
- 同calendar shockを共有する複数銘柄を完全独立標本と仮定しない。

`fit_symbol_indices` はtraining rowの選択だけでなく、**選択featureが依存してよいsymbol universeの情報scope**でもある。content-verified Datasetで全symbolより狭いfit scopeを使う場合、cross-sectional rank / dispersionのように全universeへ依存するselected featureはrejectし、reference-relative / correlation / betaはそのreference symbolがfit scope内にある場合だけ許す。依存kindはfeature名のheuristicではなくDataset build identityのFeatureKindから復元する。source Dataset identityをネストして保持する正式transformでは元build configまで遡って同じ検査を行い、verified identityなのにdependency provenanceを復元できないstrict subsetはfail closedにする。full-symbol trainingは従来どおり許し、identity provenanceを持たないlegacy/synthetic Datasetの互換は維持するが、その経路だけをunseen-symbol isolationの強い証拠にはしない。

このCandidateSuite preflightが検証するのはDatasetが宣言する `available_at` であり、過去のvenue arrivalを独立に証明しない。runtimeではsingle-symbol replay、shared-cash replay、PPO環境が、処理する行のraw-row markについて `available_at[row, symbol] > timestamps[row]` なら、その行を初期帳簿・観測・risk・約定・terminal settlementへ渡す前にfail closedにする。同時刻は許し、`information_available=False`だけでは拒否しない。single-symbol replayとPPOは選択銘柄および実保有中の銘柄を対象とし、shared-cash replayはstrategy rosterの全銘柄を対象とする。unselectedなflat銘柄と未処理のdataset tailは走査しない。直接`MarketExecutor.execute_interval`のtarget admissionも、既存のshape・finite・gross検証を満たす有効targetについて、nonzero target weightと`BookState`のnonzero exact quantityの和集合を同じrow clockで確認し、遅着ならvaluation・reconciliation・注文作成より前にfail closedにする。flat accountのzero targetからpending orderをcancelする処理は、現在markに依存しないため許す。この範囲はDatasetが宣言する行時刻の保護であり、独立時刻のない明示`mark_price`や外部`BookState.mark_prices`、明示`OrderIntent`を実行するAPI、直接`liquidate_at_close`、過去のvenue arrivalを保証しない。full runtime account-mark point-in-time authorityは未確立である。

Dataset identityは内容にbindされ、canonical artifactはdeterministicでなければならない。publication先が既に存在する場合は上書きせずfailする。

Canonical Datasetのidentity-bound feature numericsは `trade_rl.data.features.numerics` を単一authorityとし、scalar `math.log` と固定順序の `math.fsum` を基礎にmean / variance / standard deviation / dot / covariance / correlationを定義する。identityを一致させるためのrounding、quantization、tolerance-based hash canonicalizationは行わない。現行buildは `market_build_v3` と `portable_feature_numerics_v1` をbuild identityへ明示bindし、保存feature dtypeは従来どおり`float32`とする。

このportable contractは、同一code・config・sealed sourceから構築した完全Datasetについて、現行のUbuntu x86_64 hosted runner上の複数AMD EPYC系と複数Intel Xeon系で `features`、`global_features`、normalization digest、Dataset IDのbit-exact一致を実証済みである。一方、任意のARM、任意libm、任意platformまでの普遍的なbit-identical保証は主張しない。historical `market_build_v2` artifactは書き換えず、current readerでそのidentityのまま読み取れる互換を維持する。

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

`strategies/portfolio_adapter.py` は、exact built-in `ConstantIntentStrategy`、
`TrendIntentStrategy`、`RidgeForecastStrategy` だけを、固定symbol/feature rosterと
有限な`(0, 1]`範囲の明示per-symbol intent budgetを持つnumeric `PortfolioPolicy` adapterへ
接続するopt-in pathである。各snapshotの`exact_quantities`から実filled sideを導き、
current weight / quantity、current rowとbudgetを既存`StrategyObservation`へ渡して、
SHORT / FLAT / LONGを`-budget / 0 / +budget`へ写す。cash armには固定rosterの明示zero-target
policyを使える。全snapshotとroster/window contextはstrategy callback前に検証し、
callbackまたはoutput失敗がsymbol走査の途中で起きたadapterはpoisonされ、後続proposalを
返さない。factoryはadapter内でdistinct instanceを返す必要があり、比較orchestratorは各armの
ためにfresh instanceを構築する。このadapterはarm間の共有を検出しない。strategy factoryは
caller codeでありprovenance/sandboxではない。

このadapterはlegacy replayのcached target、minimum-hold、pending-aware intent、
exact-quantity HOLDを再現せず、任意の`SingleSymbolStrategy`やsubclassにも対応しない。
それらのsemanticsが必要ならdirect `PortfolioPolicy`を実装する。native sessionは同じ
BookState/MarketExecutorでrisk、order reconciliation、fill、cost、accountingと終端処理を
引き続き行う。Ridgeの`one_way_switch_cost`はintent選択のproxyであり、ledger feeを変更しない。

`strategies/allocation/` は同じ`PortfolioPolicy` seamへ、forecast sourceが供給するper-symbolの期待単純収益から費用控除後のtarget weightを作るallocatorと、等weight buy-and-hold対照を追加する。`CostAwareAllocationPolicy`は各scheduled decisionで`mu.w - (gamma/2) w'Sigma w - sum_i c_i |w_i - w_cur_i|`（`|w_i| <= w_max`、`sum|w| <= gross_max`、任意でlong-only）を許容誤差内で解き、KKT最適性certificateが通った解だけを採用する（通らなければHOLD）。L1 costがno-trade regionを作り、`min_trade_fraction`未満の変更は`hold_current_quantity_symbols`にする。Sigmaはdecision時刻以前の日次mark価格のEWMA共分散を対角へshrinkしたもので、未来barは使わない。decision schedule以外のbar、forecast/共分散/solverの失敗、口座終端では全symbolをHOLDし、黙ってtradeしない。account・P&L ledger・fill・feeは持たず、既存のshared portfolio sessionが所有する。`forecast_source`の中身（モデル・fit・provenance）はこのcontractの外にある。この実装はsoftware seamであり、経済結果やprofitability evidenceではない。
このstrategy wrapperはjoint PPO learner、training parity、Dataset/profile/fit provenance、
経済的比較やprofitability evidenceを作らない。

Canonical replayは各`StrategyObservation`へoptionalな`intent_gross_budget`と`same_intent_target_weight`を渡す。後者はbudgetがある場合だけ有効で、cached desired quantityを現在のmark/equityで表したweightである。`one_way_switch_cost`を設定したRidge / LightGBM adapterは、budgetがある場合に`CostAwareForecastIntentController.decide_for_position`を呼び、signed filled quantity（quantityが`None`ならcurrent weight）の符号からactual sideを求める。candidateは`[actual side, actual sideからのbaseline hysteresis proposal, FLAT]`の重複を除いた順序で比べる。candidateのtarget exposure `a`は通常`intent * budget`だが、candidateがlogical `current_intent`と一致しhintがある場合はcached weightを使う。従ってsame-intent holdは価格変動後の旧targetへ不要な再約定を要求せず、partial fillやpending residualは`current_weight`からcached targetまでのturnoverとして残る。direct callerも同じmethodを使えるが、返すのは`PositionIntent`だけで、fillや会計は行わない。

各target exposure `a`のutilityは`U(a) = a*expm1(f) - c*abs(a - w)`で、`f`はlog forecast、`w`はactual weight、`c`はone-way costである。選択はutilityが厳密に高い場合だけ置き換えるため、tieは先のcandidateを維持する。minimum-hold中およびunlock decisionではhintを渡さず、既存constraintがquantity holdとnominal target resetを制御する。pairwise比較はtarget間のturnover傾斜をboundedな相対値として求め、`log1p`と比較する。`1 + c*k`がfloat精度で0へ潰れる境界だけexact binary-rational fallbackを使い、正のgapを保持する。相対値は比較のためだけに使い、actual weightやcached targetをclipしない。任意の部分数量を連続最適化するものではなく、選択後には通常のhard-risk projectionとexecutionが適用される。`intent_gross_budget=None`の直接callerは従来のintent-based controller経路を保つ。

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

Run Core also binds a strict positive-integer
`supervised_forecast_horizon_hours` (default 24, at most 2,562,047 hours so the
exact duration fits signed `datetime64[ns]`) for both Ridge and LightGBM.
The shared training-set builder uses that exact elapsed-hour endpoint and only
admits labels that mature strictly before `fit_cutoff`; fitted strategy names
include the horizon. `resolved_run_config_v6` and candidate result schemas v8
(per-symbol) / v9 (shared-cash) persist the selected value. This is a per-run
capability: canonical Studies remain fixed to 24 hours and reject another value
before Dataset resolution or candidate fitting. It adds no controlled factor,
paired-comparison authorization, PPO behavior change, or economic claim.

`run_shared_cash_replay` is a separate multi-symbol evaluation boundary. Each
symbol has one fresh strategy instance, all instances observe the same pre-trade
portfolio snapshot, and their complete target vector passes through
`simulation.native_transition.execute_native_target_transition`, which applies
the shared `PreTradeRisk` then advances the canonical `MarketExecutor` once per
interval. `current_weight` is measured
against the single portfolio value. Its optional `minimum_hold_bars` can be one
common duration or an ordered per-symbol vector; age advances from actual signed
fills, the locked target is rebound to the exact filled quantity, and risk
projection still runs afterward. Optional terminal settlement reserves the
latency-aware close interval and sends a portfolio-wide flat target through the
same risk and execution path. In this explicit terminal-only step, an
`emergency_flatten_mask` covers nonzero holdings in the selected strategy
roster, so ordinary `max_turnover` does not scale down the requested close;
ordinary strategy decisions continue to obey that turnover bound. The target
still passes through the configured hard-risk and execution stages. Venue
rules, available liquidity, fill capacity, fees and pending-order handling are
unchanged: if they prevent a full close, the actual residual quantity or active
order remains visible and exact-flat validation still fails. Duration/settlement
runs emit shared-cash ledger schema v2, including requested/effective intents,
per-symbol ages and quantities, and suppression/unlock evidence; calls without
those options retain v1 ledger shape.
This replay capability does not make the existing per-symbol PPO training
environment a joint portfolio learner, and does not change the independent-account
meaning of the currently frozen v1 Study.

`run_shared_portfolio_replay` is the opt-in batch consumer over the
simulation-owned `SharedPortfolioSession`. It invokes one
`PortfolioPolicy.decide_portfolio` for each eligible native decision row. The
session retains the exact returned `BookState` and one `MarketExecutor` across
calls, advances its native index once per target, and uses the existing
`execute_native_target_transition` for hard risk and execution. Its public
account view is a detached clone, and evidence collection remains in evaluation.
`strategies/portfolio_observation.py` builds the detached snapshot from the
current `MarketDataset` row, `BookState`, active-order tuple, risk config, and
decision window; evaluation passes those source-owned inputs without giving the
strategy an executor or evaluation config. The complete symbol-ordered numeric
target vector is validated before a native transition can mutate execution
state. The snapshot copies only the current `StrategyObservation` row,
primitive account facts, active-order quantities/status/timing and available
order price context (`limit_price`, `stop_price`, and
`submission_reference_price`), and risk-limit scalars into tuples and immutable
values. Snapshots built through the current observation builder carry the
source intent's price context; directly constructed legacy pending-order
snapshots may leave all three price fields `None` and are explicitly incomplete
for price-aware consumers. The snapshot does not include the
`MarketDataset`, executor, risk or book objects, opaque order IDs, or future
rows. Cash is reported as raw account
cash, not as free or reserved cash, so this input is not a complete Markov account
state. An explicit exact-quantity HOLD is accepted only when its target exactly
matches the current book weight, and it survives only if hard risk leaves that
weight unchanged; the normal executor then cancels any residual order while
preserving the signed filled quantity. Reconciliation applies that cancellation
before processing the next execution interval, so an unfilled remainder cannot
fill in that interval after a HOLD decision. The actor cannot request the terminal
emergency-flatten mask. Policy calls stop at the existing latency-aware agent
boundary, after which the session reserves terminal intervals for its existing
risk-owned emergency-flatten path. A failed observer/transition consumes the
session because executor state may already have advanced; it cannot be retried.
The raw-v1 codec's `account` matrix columns are exported as `ACCOUNT_FIELDS`
(`quantity`, `weight`, `mark_price`, `contract_multiplier`). Its 21-column
`active_orders` matrix supports at most one order per symbol and rejects
out-of-roster symbol indices. Optional limit, stop, or submission-reference
prices use their existing presence flags; absent values are encoded as zero
with a zero flag. This keeps all-`None` legacy snapshots encodable without
inventing missing price facts.
This
numeric path has its own `shared_cash_portfolio_replay_ledger_v1` action schema;
the existing intent replay result and ledger schemas remain unchanged. This seam
does not change PPO's per-symbol training path or establish an economic result.

`evaluation/runs/portfolio_family_comparison.py` is an opt-in in-memory
coordinator for four fixed families: explicit cash, built-in Trend, a
caller-pinned Ridge artifact, and a verified joint-PPO inference bundle. It
loads and validates the paired Dataset, Ridge binding, PPO bundle, roster,
window, and shared configuration before starting any native arm. The PPO Gym
is validation-only and is not reset or stepped. Each arm gets a fresh policy,
risk controller, `SharedPortfolioSession`, and ledger while sharing the same
Dataset and immutable execution/risk/profile configuration values. Every arm
uses terminal settlement and ledger capture. Raw early termination and
insolvency remain visible in the outcome flags. An optional completion observer
receives each outcome immediately before the next arm begins; callback failure
or a later arm failure propagates without starting subsequent arms. The
coordinator does not reserve attempts, persist results, fit models, select a
winner, or establish research or profitability evidence.

`evaluation/runs/portfolio_family_attempt.py` is a separate opt-in filesystem
wrapper around that comparator. It reserves one attempt root, records declared
identity and before/after provenance, and persists each raw family outcome in
the comparator's completion callback before the next family begins. It then
publishes and strictly reloads the result before writing the completion marker;
the public attempt loader accepts only a complete, nonfailed root. A failure
record retains the phase and already completed family rows, while the native
executor and ledger remain the sole financial-state owners. The wrapper records
failure phase and earlier rows when its failure journal is writable; if that
journal write also fails, the original exception remains primary and the
secondary write error is attached as an exception note. This local write-once
boundary is not a global retry budget and adds no fit, winner selection,
research approval, or profitability evidence.
If the comparator raises inside a family before its completion callback, that
callback does not expose the in-flight family's partial native ledger; the
failure record therefore contains only rows delivered by earlier callbacks.
It does not infer a zero count for the unfinished family.

`strategies/rl/portfolio_transform.py` owns opt-in preprocessing for the joint
Gym Dict. Its versioned payload binds the raw/transformed layouts and
Dataset/objective/clock/risk/execution context. Local features use the existing
PPO normalizer; global features and per-symbol geometric mark-price references
are fitted only on `[start_index, agent_stop_index)`. Availability masks remain
explicit, account and active-order notionals use fixed initial capital, while
the dimensionless maintenance-margin rate remains a rate. UTC nanoseconds are
differenced before conversion to floating point. Loading serialized transform
parameters requires an independently retained expected digest; this detects
payload changes but does not prove authorship or fit provenance. Raw validation
also checks risk flags and order-index fields against their declared types. The
wrapper preserves native rewards and termination facts, uses absorbing padding
after termination, and records then poisons itself if transformation fails after
a raw step may have advanced. This does not make observations a complete account
state, synthesize free/reserved cash, refit on evaluation rows, train a policy,
or change legacy PPO behavior.

`PortfolioRuntimeExecutionScenarioV2` is the typed declaration used by a
separate opt-in scenario inference path. The v2 loader requires the typed
declaration and external scenario digest, all existing bundle/context/checkpoint
pins, and a separately expected current scenario-source digest. Before PPO
deserialization it validates that the declared base execution policy matches
the fitted context and the transform's existing execution-config digest, while
the runtime policy matches the actual forward `SharedPortfolioGymEnv`. The
versioned runtime binding retains both execution identities, the original fit
and actor-stop window, the runtime bound/window/actor stop, and scenario,
checkpoint, context, transform, and source identities. Fit statistics and
checkpoint bytes stay unchanged: the training actor stop uses base latency,
while runtime clock/order features and the native terminal reservation use
runtime latency. A dedicated v2 observation wrapper checks the actual runtime
environment before reset, each native step, and observation transformation;
it poisons itself after a failed post-step transform. The v1 loader, binding, payloads, and strict full-cost
matching remain unchanged; no compatibility bridge admits older selected
source/runtime checkpoints. Selected source manifests are direct-owner records,
not a complete transitive source or in-memory-code proof. The explicit
`portfolio_family_scenario_replay_v2` evaluation entry now wires this loader
into four actual native arms under the scenario's runtime cost configuration.
The v2 coordinator rechecks its scenario digest and current source digest, then
validates the loaded PPO policy's bound runtime environment before each arm and
before each completion observer. A mismatch stops before the next arm or before
an observer receives the completed outcome; the strict v1 entry leaves this
private boundary check unset.
The in-memory entry remains separate from the opt-in
`run_four_family_portfolio_scenario_attempt_v2` one-shot local wrapper. That
wrapper records a versioned external identity, started marker, before/after
provenance and scenario-source records, and each existing native family row
before starting the next arm. A strict result reload checks the caller-held
identity and expected result digest before completion; failures retain the
primary error and rows already written. This is software-integrity evidence:
it does not select a numeric stress recipe, repeat native accounting, alter the
v1 artifact schema, or establish economic validation, profitability,
unused-period approval, or live eligibility.
`PortfolioPolicySampleJournal` is a separate opt-in SB3 forward-hook recorder.
It stores actual policy forward inputs, sampled action codes, the returned joint
log-probability, and critic value before the caller advances the environment.
Each sample-v2 row also records the forward batch size and actual action/value/
log-probability tensor dtype and full batch shape, without changing the saved
action or scalar values. Attachment requires empty policy-local and global
forward-hook layouts; those layouts and their dispatch flags are frozen after
attachment. Any later change fails closed before a sample is written, preventing
a replacing hook from silently changing what SB3 receives. Caller-declared
context, transform, and policy-version labels are not Dataset, weight, or
checkpoint-lineage attestations. The recorder does not store optimizer updates
or join policy samples to native financial transitions, and it leaves legacy
PPO defaults and serialization unchanged.

`run_scripted_shared_cash_replay` is an opt-in evaluation seam for fixed action
sequences. It validates every symbol schedule before execution. `None` means no
new actor decision and keeps the cached target and pending order progressing;
`HOLD_QUANTITY` derives the side from the actual signed book quantity, cancels
any remaining order, and carries the book's exact quantity through reconciliation
only while the post-risk weight still equals the current book weight. Risk-driven
shrink or flatten remains effective. The strategy protocol and shared-cash ledger
schemas are unchanged.

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

`simulation.native_transition.execute_native_target_transition` owns the shared
one-bar numeric-target → pre-trade-risk → canonical-executor step. It returns
the existing `RiskConstrainedTarget` and `ExecutionResult`; it does not add a
second account or ledger. Replay and PPO callers keep their policy/action state,
pre-execution cached-target rebinding, split handling and evidence capture around
that transition. PPO's historical no-risk shortcut remains unchanged.

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
of hypothetical pending fills in the economic projection. When the solvent
processing-open book is already above the gross leverage cap, an otherwise valid
MARKET reduce-only close may be admitted only if it lowers gross exposure. Such
recovery closes reserve their exact accepted inventory during the admission pass
and do not reduce projected exposure before a fill, so an unfilled recovery close
cannot authorize a duplicate close or later opening. Under-cap passes keep their
existing priority projection and fill-time expiry behavior. Allocation then
checks the actual priority sequence, including ordinary fills, and caps each
closing fill at the opposite inventory remaining. Exhausted active remainders
expire explicitly; true sub-lot inventory stays visible. Fees and capacity apply
only to actual accepted lots. Without an explicit source profile, minimum notional
and all existing constraints still apply. A legacy zero-target MARKET request is
represented internally as a reduce-only exact-inventory close, but this does not
waive its minimum-notional or other execution constraints. Without an explicit
reduce-only roster, a same-side smaller MARKET target is also represented as
reduce-only only when actual holdings at decision reference prices exceed
`maximum_gross`. This inference does not apply to LIMIT/STOP orders or when a
roster is explicit, including an empty roster. The omitted-roster legacy
zero-target behavior remains unchanged; an explicit roster controls whether its
symbols' zero targets are reduce-only.

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
and retain exact terminal quantities. Every current arm result, including the
omitted-profile `directional_arm_v1` schema, requires exact zeros from the
account's exact quantities for terminal flatness. Previously saved results are
not rewritten; their recorded terminal fields continue to describe their
original evaluation. Strategy factories, costs, hard risk and training are not
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

共通のwalk-forward OOS stitching API (`stitch_oos`) は、明示されたfold `elapsed_years` を合計し、durationがないfoldは `len(values) / periods_per_year` のdefault clockを加える。独立fold間のgapは別に記録し、returnやdurationを補わない。全foldがdefault clockなら stitched `elapsed_years` は未設定のまま保つ。

`compare_shared_cash_strategy_factories` は、順序付きの各strategy armを一つのshared-cash accountで全Dataset銘柄に適用する比較APIである。armごとに各symbol用のfresh strategy adapterを作り、frozen model objectの共有は許すが、adapter instanceはarm間・symbol間で共有しない。全adapterの検証後、armごとに同じDataset、range、capital、gross budget、execution cost、risk、market-order profile、minimum hold、terminal settlement設定で既存shared-cash replayを一度実行し、各replayのreturnsとexecution diagnosticsからmetricsを作る。gross budgetは既存replayどおりsymbolごとのintent sizingへ適用され、その後portfolio hard-risk projectionを受ける。shared initial capitalはarmごとに一口座であり、銘柄数倍にはしない。fitting、ranking、selectionは行わない。

`evaluation/runs/ridge_artifact_comparison.py` は、同じ仕組みを使う限定的なfilesystem比較入口である。公開済みDatasetのpaired loaderと、呼出し側が期待digest・typed bindingを渡したfrozen Ridge artifactを一度ずつ読み、Dataset identity / schema / artifact digest / ordered feature names、Ridgeのselected featuresとfit-symbol mapping、decision開始時刻に対するfit cutoffをreplay前に照合する。`cash`、`legacy_hysteresis`、`position_aware_zero_cost`、`position_aware_cost` の4 armへ同じ明示的なcapital・risk・execution・range設定を渡し、既存shared-cash replayのterminal settlementとledger evidenceを返す。返り値は使用した正確なvalidated configを保持する。legacyからzero-costへの差はcontroller semanticsも含むbundleで、zero-costから正costへの差がposition-aware経路上のcost proxyを変える。proxyはexecution feeを帳簿へ加算せず、既存ledgerがexecution economicsを一度だけ計上する。このAPIはmodelをfitせず、winnerを選ばず、result artifactをpublishしない。artifact bindingは宣言されたfit contextを固定するもので、過去のfit手順を独立証明するものではない。

The comparator can opt in to forecast-versus-zero diagnostics. It calls the same frozen Ridge model's `predict` with the model's ordered feature indices and scores log-return labels only when the model features and decision-time starting close are available, and the exact horizon endpoint is strictly before the requested stop timestamp while the target close is available by that scoring-as-of timestamp. Forecast and zero-baseline errors use identical pairs; candidate, eligible, and per-reason exclusion counts are retained. The opt-in result uses comparison artifact v2; default calls continue to publish v1. Hourly horizon labels overlap, so pair counts are not independent market samples or significance evidence. This diagnostic is out-of-sample relative to the bound fit cutoff, but supplied predictions are not prequential vintages and the metric is not a P&L or causal-alpha claim.

`evaluation/runs/ridge_comparison_artifact.py` は、その返り値を一つのcanonical `comparison.json` へwrite-onceで保存し、外部digest必須のloaderで完全な4 arm・ledger・config・Dataset/model境界・caller-supplied provenanceを検証する。returnsはproducerが返す`BASE_BAR`、`elapsed_years=None`、全arm共通の正の`periods_per_year`に固定する。flatでorder/capacity eventがない区間は、cash/equity変化を記録済みのcash interest/dividendへ結び、funding boundary合計をinterval fundingへ照合する。cash armはintent・quantityともFLATで、order/capacity、execution cost、funding amount、borrow、dividend、turnoverを持たず、trade/rebalance、closed/open-position、realized net/gross P&Lのdiagnosticsもゼロでなければならない。正負どちらの明示cash interestも保持できる。interval-end returnsから再計算したdrawdownとledgerのintrainterval maximum drawdownは別の測定値であり、前者が後者を超えないことだけを確認する。ledger v2は一部のdelisting/split settlement proceedsを独立fieldに記録しないため、全armに対するuniversal fill-to-cash再構築は行わない。profileは保存された宣言データとして扱い、verifiedな実行profileを復元しない。publisherは渡されたprovenanceを検証するだけで、replay前後のsource/runtime採取、Dataset cadenceの再認証、fitの独立証明は行わない。これは保存・再読込のsoftware capabilityであり、Study protocol、比較結果の採否、G0-G3やprofitability evidenceを追加しない。

`evaluation/runs/ridge_comparison_attempt.py` は上記比較を一回のwrite-once attemptとして実行する薄いfilesystem wrapperである。typed入力とcallerが宣言するresearch-context digestを検証した後、attempt rootを排他予約し、supplied path・binding・configの開始記録を残す。既存provenance ownerを比較の前後で呼び、実際のimplementation/runtime/contextが変化しない場合だけ既存result publisherを一度呼び、strict loaderで再読込してからcompletionを記録する。失敗時はfailure recordを残し、比較が返った後のpublication失敗では利用可能なarm ledger mappingsをpartial evidenceとして保存する。比較呼出しが返る前の内部partial ledgerはpublic APIから取得できない。root単位のwrite-onceはlocal attempt reservationであり、global retry budgetではない。context digestはcaller declarationで、approvalやauthorizationの証明ではない。このsoftware boundaryはfit・data retrieval・winner selection・research approvalを行わない。

### Trading-bot tuning and holdout reporting

`evaluation/bot.py` はshared-cash replayへ既存のnon-zero `ExecutionCostConfig()` を渡す。zero-cost replayは呼び出し側が明示的に指定した場合だけ使う。Hyperparameter selectionは時系列の先行windowだけを使い、baselineと選定candidateのperformance reportは後続holdout windowのfresh replayから作る。`tune_all_strategies` の順位もholdout returnではなく同じtuning window上のobjective scoreに従う。

Bot CLIの`--signal-feature`はDatasetの一意なfeature名を実行前に解決し、対応する`signal_index`をbaseline、cash、全candidate、全walk-forward foldへ固定する。未指定時は従来の0列目を維持する。公開tuning/comparison APIも同じindexを明示でき、boolean・負値・非整数・source Dataset範囲外のindexはchannel追加やreplay前に拒否する。configのindexとDataset identityがfeature semanticsをbindする。channel戦略とconstant/cash controlは従来どおり各自の入力・intent契約を使い、この指定でchannel定義やcontrolを変更しない。

上限付きparameter searchは各parameter axisをまたぐdeterministic sampleを使い、grid先頭のprefixだけに偏らない。`BotReport` のreturn interval数、positive rate、profit factorはbar interval単位のmetricsであり、closed-trade metricsとは呼ばない。Sharpe annualizationはreturn seriesのperiod metadataを使う。`MarketDataset` と `ReturnSeries` の `periods_per_year` はpositive integerであり、booleanやfractional/non-finite frequencyは拒否する。`--mode compare` は全期間のin-sample diagnostic rankingであり、holdout selectionではない。baseline P&Lがほぼzeroの場合、relative improvement percentageはundefinedとして報告する。

CLIは既存の `--dataset` directoryまたは明示的な `--demo` のどちらかを要求する。省略時や明示されたpathが存在しない・directoryでない場合は入力errorとして終了し、optimize/compareをsynthetic demo dataへ暗黙にフォールバックしない。

単一strategyのtuning resultは、parameter selection後の後続window reportを返す。一方、`tune_all_strategies` は五つのfamilyそれぞれの後続window reportを表示するため、familyを選ぶ目的で比較した時点でそのwindowはdevelopment evidenceになる。出力の `report_scope=development_family_comparison` はこの意味を示す。family選択後にfinal out-of-sample claimを行うには、さらに後の未閲覧windowで再評価する。

Candidateの選定適格性はtuning-window shared ledger maximum drawdownが20%以下であることを要求する。20%はselection vetoであり、pre-trade stopやholdoutのrealized drawdownを20%以内に保証しない。価格gap、約定損、terminal settlementで観測drawdownが20%を超える場合があるため、holdout drawdownはそのまま報告する。

`balanced` は符号付きのtotal return / max(drawdown, 0.1%)へinterval profit-factor bonusを掛ける。損失の符号を反転せず、損失candidateをcashや正returnより高く評価しない。この比率は年率換算Calmarではない。tuning candidateは終端のexact quantityが全て0、active order remainderなし、economic terminationなしであることも必要とする。Bot reportはmarked equity/P&Lと`terminal_settled`、残余quantity、active order、termination reasonを同時に保持する。決済不能の後続reportはそのまま表示し、利益や決済完了へ書き換えない。ledger evidenceを伴わず手動作成されたreportの`terminal_settled`は`None`とし、未確認の決済を真として扱わない。

各tuning prefixでは既存baselineとparameter gridに加え、同じcapital・cost・windowのcash replayを必ず比較する。cashはgrid budgetを消費しない独立controlであり、同点ならcashを優先する。cashの実損益を使い、cash interestが存在するDatasetをzero returnへ書き換えない。zero-P&L cashのSharpe objective scoreは0とする。後続windowの損益は選定に使わず、正のprefixから選んだ取引candidateが後続で損失になっても、その結果をcashへ差し替えない。requested familyは`TuningResult.strategy_name`、実際に選んだfamilyは`optimized_config.strategy_name` / `optimized_report.strategy_name`で区別する。取引候補が全て不適格でも、cashの実replayが適格ならcashを返す。cashを含む全候補が不適格なら成功を捏造せずerrorにする。

Bot reportはbookの`total_execution_cost`、符号付き`funding_pnl`、`borrow_cost`、`turnover_total`、`fill_count`、`rebalance_events`も保持する。cost/funding/borrowはaccount currency、turnoverは各fill notional / interval開始equityの和であり、ドル額やclosed-trade countとは呼ばない。診断値はP&Lへ再加算・再課金せず、execution/accounting ownerの実測値を報告する。手動で作る従来の`BotReport`では未提供の診断値を`None`とし、未計測を0へ偽装しない。

`walk_forward_tune` / `--mode walk-forward` は同じ探索・score・eligibilityを使い、直前foldで選定して次foldをfresh capital / strategy stateでreplayする。各windowの`report_scope`も`development_walk_forward`とする。`profitable_windows`にはmarked P&Lが正で、かつ終端決済が確認されたreportだけを数える。最後のfoldには割り切れない残余barを含め、隣接するevaluation intervalは重複しない。最後のmarkは次windowの開始markにもなる。後のtuningで既に評価したfoldを再利用するため、全体は`development_walk_forward`であり、独立標本やsealed final evidenceではない。各windowは同じinitial capitalへresetし、`cumulative_return_pct`は正規化returnの仮想積である。capacityやorder sizingを再投資capitalでreplayしたcontinuous wealth pathを表さない。

Botのchannel戦略は`channel_entry_upper/lower`、`channel_exit_upper/lower`をfeature名で解決し、不足時はreplay前にrejectする。閾値ではなく正規channelの符号を使う。synthetic demoはEMA signalと既存のprior-candle channel builderを使い、現在足をchannel extremaへ含めない。adaptive botのregime判定は設定signalの絶対値を使うmomentum判定であり、独立したrealized volatilityの推定ではない。

`AdaptiveProfitConfig`はfinite・nonnegativeなthresholdと、nonnegative integerのfeature index / max holdingを要求し、booleanも拒否する。entryはpositiveで対応exitより大きいことを要求する。zero regime thresholdによるtrend固定と、zero protective thresholdによるexit無効化は明示的な有効設定として維持する。NaNでprotective comparisonを黙って無効にする設定はreplay前に拒否する。

### Direct decision-value study contract

`trade_rl.evaluation.decision_value_dev_study` は、事前固定された直接的意思決定価値学習（Direct Decision-Value Learning）の研究契約と、現行の合成データ診断の実行境界を定義する。**旧TD8 protocol v2（実データ）は履歴閲覧・検証の対象であり、現行runnerからの再実行は禁止する。** 新しいsolver・model schema・paired projection・proposalで古い研究認可ダイジェストを使い回すことはできない。

- **現行TD8 software-only gate（protocol v3）**: `run_decision_value_dev_study` は `DecisionValueStudyPlan(data_scope="synthetic_only")` と `real_scope_authorization_digest=None` を要求し、`artifact_root` の公開済みCanonical Datasetをmanifest/配列/ID/artifact digest照合で検証する。`identity_payload_json` の `synthetic_only: true` は発行元の申告にすぎず、データの合成由来や独立認可を暗号学的に証明しない。実行時は凍結した3学習器・4シナリオとの一致を検査し、既存出力dirを事前拒否。出力は `decision_value_dev_study_protocol_v3_software` / `decision_value_dev_study_result_v3_software` で、正常診断は `SOFTWARE_ONLY`、原因判明した不正は `INVALID / BLOCKED` とし、収益性・新規G4/G5/liveの証拠として扱わない。独立した研究審査と新しい研究認可が成立するまで、旧Tier D実データの新sourceでの再実行経路を設けない。

- **Source scope契約**: `DecisionValueStudyPlan` の型は、従来の `"synthetic_only"` と履歴用 `"real_development"` を識別し、後者にSHA-256設定ダイジェストを要求する。ただし、設定ダイジェストは独立研究認可ではない。**現行`run_decision_value_dev_study`は`"real_development"`を一律に拒否する。** scope不一致・旧実データの無断再実行はfail-closedとなる。
- **旧実データ仕様のダイジェスト（認可そのものではない）**: `compute_real_scope_authorization_digest` により、Dataset ID、期待artifact digest、銘柄、初期資本、horizon、候補行動、コスト率、リスク制約、ブロック定義、Ridge設定および既消費開示文言を含む仕様のSHA-256を再計算し、旧プロトコルのフィールドと照合する。呼び出し元でも作成できるハッシュであり、外部G0–G3承認・新sourceへの実行許可は含まない。
- **時系列ブロック構造**: 各ブロック（`DecisionValueBlockSpec`）は `train_start < train_stop < fit_cutoff < eval_start < eval_stop` の厳格な時系列順序を要求し、過去情報のみで学習（B0 Ridge、B1 線形価値、B1c 保守的線形価値、B2 浅層非線形価値）を行い、評価区間をreplayする。
- **保守的意思決定価値モデルと不確実性割引 (B1c)**: `ConservativeLinearDecisionValueModel`（`schema_version="conservative_linear_decision_value_v2"`）は、候補行動ごとの回帰係数ベクトルに加え、重なり群の平均多重度 $\bar{m}_k = \frac{1}{N_k} \sum_i m_i$ と有効自由度 $\text{dof}_k = \max(1.0, N_k / \bar{m}_k - p_{\text{eff}})$ に基づく残差共分散行列 `covariance_matrices`、不確実性ペナルティ重み `uncertainty_penalty_weight`、および片道切替コスト `one_way_switch_cost` を保持する。標準化特徴空間でのRidge回帰係数および共分散行列は、元座標系 $x$ へのアフィン変換行列 $A$ により $w = A^\top \beta$、$Cov(w) = A^\top Cov(\beta) A$ として代数的に折りたたまれ、モデルのシリアル表現および推論コードとの完全な互換性を維持する。ポリシー `ConservativeLinearDecisionValuePolicy` は、予測標準誤差 $se = \sqrt{x^\top \Sigma_k x}$ を用いた下側信頼限界（LCB: `mean - lambda * se - c * extra_turnover`）により不確実性と切替ターンオーバーを保守的に割り引く。
- **B1c有効自由度（実装との対応）**: 現行実装の残差分散は、重なり相関行列 $\Omega$ と射影 $H$ から $\text{dof}_k = \max(1,\ \mathrm{tr}[(I-H)\Omega(I-H)])$ を用い、`fit_conservative_linear_decision_value(residual_dof_out=...)` で候補headごとに取り出せる。`overlap_groups` は区間が接する（`decision_time == 直前のterminal`）branchを同一groupに連結する開示用の数であり、自由度には使わない。非重複ラベル（branch step = horizon）では $\Omega$ は単位行列に近づき、dofは標本数からパラメータ数を引いた値に近い。
- **TD7旧v1とTD8の後方互換境界**: TD8の新モデルartifactにはB1 `synthetic_linear_decision_value_v2`、B1c `conservative_linear_decision_value_v3`、B2 `synthetic_shallow_decision_value_v2`を用い、旧B1 v1/B1c v2/B2 v1を新solverの結果で偽装しない。学習済みv1モデル・古いpaired介入・旧プロトコルの意味を、TD8改修の効果で暗黙に書き換えない。v1は元来の切片にもRidge penaltyを掛けるsolver、旧B1/B1c/B2のHOLD・FLAT選択/score、pendingを除いたv1予測と実pending口座へのproposal、旧v1 paired action codecを維持する。TD8のみ別途version指定したsolver、hold-relative推論、native proposal、paired介入を使用する。異なるversionで実行した研究を旧実験の再現と呼ばず、新しいコードの結果で凍結済みTD7数値を書き換えない。
- **旧B1/B1c/B2とBLAS互換性の検証境界**: 旧B1/B1c/B2の既存artifactは保存済みSHA-256で検証し続け、旧B2 v1のcanonical JSONはTD8専用の入力正規化フィールドを持たない。BLASの違いによる新規fitの係数・共分散の最終bit差を想定し、新規fitの互換テストでは元mainの全係数配列を厳格な数値誤差で比較し、同一計算環境で旧Ridgeの正規方程式を独立に検算する。旧B1/B1cの係数・共分散も元mainから採取した完全な数値goldenと厳格な許容差で比較する。旧B1/B1c/B2の保存形式そのものは係数固定モデルのcanonical JSON/SHA-256でバイト単位比較する。異なる計算環境で新たに学習したモデルのSHA-256一致までは保証しない。
- **TD8補正設計（protocol v2, `decision_value_dev_study_protocol_v2` / `decision_value_dev_study_result_v2` / `decision_value_real_scope_authorization_v2`）**: TD5-TD7の直接価値学習は、(i) 全学習状態がflat（`_HoldQuantityPolicy`参照を`start_index=t`で起動）で、ラベルが実質「tでエントリーし24バー保有して清算」であり、口座ウェイト係数が恒等的に0だった、(ii) 判断に閾値が`1e-12`しかなくノイズでchurnした、(iii) 非定常な`funding_bps`が後続ブロックで線形外挿された、という3点が診断で確認された。TD8はこれを結果前に固定した次のパラメータで補正する（変更は新protocol versionと新authorization digestを要求する）。
  - **到達可能状態ラベル** (`evaluation/decision_value_reachable_labels.py`): branch時刻 $t$（`train_step`=horizon=24でラベル窓は重ならない）ごとに、`start_index = max(train_start, t - label_prefix_bars)`（`label_prefix_bars=48`）から平坦な初期口座でB0 cost-aware policyを走らせ、$t$で候補行動を1回だけ介入し（baselineはexact-quantity HOLD）、以後B0が $t+24$ まで継続して終端で強制清算する。ラベルは候補とHOLDのネイティブ終端equity差 / 初期資本である。これは「到達可能状態」の**切り詰めprefix近似**であり、$t$時点の口座は48バーのprefixの長さとB0がflat開始であることに依存する。全候補（all-FLAT、損失・ゼロを含む）を保持し、設定済み決済に至らないpairは`ineligible_pairs`として計数して黙って落とさない。現行software-only v3は、各非適格pairについて`decision_value_ineligible_pair_v1`の監査レコード（baseline/candidateの終端評価額・cash・正確な保有数量・費用・funding・停止理由・insolvency・未約定残・観測区間数）を`ineligible_pair_outcomes`に保持して結果JSONへ加える。学習用の有効なlabelには変換せず、値を0と捏造しない。ただし全pairが非適格で学習可能なlabelが一つもない場合は、現行runnerはfit前に例外停止するため結果JSONの永続化は保証しない。
  - **B0参照の因果的再fit**: `RidgeForecastStrategy`は`fit_cutoff`より前の意思決定を拒否するため、block-levelのB0（`fit_cutoff`=ブロックfit cutoff）はtrain window内で動作できない。`causal_b0_reference_provider`は同一recipe（特徴index・horizon・alpha・entry/exit閾値）を各branchのprefix開始時刻 `fit_cutoff = timestamps[start]` で再fitし、prefix開始前に`reference_min_fit_bars=96`バー未満しかないbranchは参照なしとしてスキップ・計数する。評価アームのB0は従来どおりブロックfit cutoffでfitする。参照が評価アームのB0と同一modelでない点は既知の限界である。
  - **HOLD相対action rule**: `LinearDecisionValuePolicy` / `ConservativeLinearDecisionValuePolicy` / `ShallowDecisionValuePolicy` は `hold_relative=True` で、$Q(s,\text{HOLD})=0$ の構造的ゼロ、全候補（FLAT含む）を学習headとして扱い、`select_hold_relative_action` が $\max_a(\hat\Delta_a - m_a)$ が厳密に正（かつ`1e-12`超）のときだけ行動し、同点はHOLDとする。ここで $m_a = k \cdot 2c_{\text{eval}}\sum_i |w^{(a)}_i|$（`round_trip_cost_fraction`、$w^{(a)}$ は候補の目標ウェイト、HOLD対象は現行有効ウェイト）。HOLDと同一のネイティブproposalになる候補はno-opとして無視する。ストレスcost補正は `(c_eval - c_train) * (Σ|t-e| + Σ|t| - Σ|e|)` のみで、訓練costと等しければ0である（旧flat起点の`compute_posture_cost_adjustment`は`hold_relative=False`の既定でのみ残る）。
  - **事前固定のmargin選択** (`evaluation/decision_value_margin.py`): 倍率 $k \in \{0, 0.5, 1, 2\}$ をtrain window内だけで選ぶ。成熟したbranchラベルを時間順に前半75%（fit）と後半25%（calibration）に分け、fitで一時モデルを学習し、calibration区間（fit最終ラベルのmaturityの後ろに1バーのembargo）でネイティブ口座のclosed-loop replayを各 $k$ について実行して**費用控除後net return**が最大の $k$ を採る。厳密な同点は大きい $k$。評価block・fit cutoff以降のデータは一切使わない。B1/B1c/B2に同じ手順を適用し、選択結果・各 $k$ のreturnをartifactに記録する。
  - **入力特徴の一様変換** (`evaluation/decision_value_feature_transform.py`): 全アーム（B0を含む）の入力を、train windowの(symbol, feature)ごとの平均・標準偏差で標準化し $[-3, 3]$ にclipする。統計はtrain windowだけから計算し、`normalization_digest`へ変換digestを束縛する。`funding_bps`を含め特徴は削除しない。`covariate_shift_report`は評価区間のtrain基準z（平均・最大絶対値・clip率）を診断として出力するが何も自動除外しない。clip規則は診断で`funding_bps`のシフトを観測した**後**に選ばれた事実を研究上の自由度として開示する。
  - **事後prior撤回**: `unexcited_prior_std`の既定をmain相当の無情報値`1.0`へ戻し、TD7再実行で導入した`1e-3`の事後調整は撤回する。protocol fieldとして凍結し、authorization digestに束縛する。
- **姿勢認識プロポーザルと未約定注文保全 (TD7)**: `preserve_posture_aware_proposal` は、`LinearDecisionValuePolicy`、`ConservativeLinearDecisionValuePolicy`、`ShallowDecisionValuePolicy`、および `_FixedEntryPolicy` に共通する統一的な姿勢保全セマンティクスを提供する。(1) レイテンシ下（`latency_1bar`, `latency_2bar`）で未約定注文が進行中の場合、目標ウェイトを実効ウェイト $w_{\text{eff}} = w_{\text{account}} + w_{\text{pending}}$ に一致させつつ `hold_current_quantity_symbols` への追加を避けることで、`reconcile_target` による進行中注文の早期キャンセルを防止して約定を保護する。(2) 約定完了後は `hold_current_quantity_symbols` に追加して毎時の微小ドリフトによるリバランスターンオーバーを完全に抑止する。(3) 損失ポジションを保有中に全候補の方向スコアが非正となった場合、自動的に現金化（`ACTION_FLAT`）を選択して損失の拡大を防止する。
- **因果性プローブと台帳突合**: 各ブロックの評価開始直前に、未来データ（t+1以降）を無作為摂動させたデータセット上での意思決定が元データセットと完全一致することを確認する因果性プローブ（Causality Perturbation Probe）を実行する。protocol v2ではfit cutoff以降を摂動したデータで特徴変換・B0・ラベル収集・全head・margin選択のパイプライン全体を再実行し、係数SHA-256・B1c共分散・選択margin・評価開始時の意思決定が不変であることを要求する。また、各ブロックの全アーム・全シナリオについて、台帳の注文・約定・資産推移・手数料・終端清算が整合していることを検証する（Ledger Evidence Reconciliation）。
- **現行TD8の教師vintage（合成software-only）**: B0教師を歴史上のprefix開始で再fitする場合、ブロック全体の将来の統計量を流用しない。raw Datasetから `train_start` 以上 `prefix_start` 未満で、かつ `prefix_start` 時点に利用可能な記録だけを使った標準化・clipのvintageを作り、fitとprefix/continuation判断の双方へ渡す。学習器の事後的なblock訓練変換と教師vintageは意図的に分ける。教師の参考方策digestをv3へ更新し、historical TD8 protocol v2・旧2024Q1研究結果は再計算/再分類しない。
- **TD8のprefix別教師・paired evidence（software-only）**: `ReferenceVintage`は参照B0の実際のprefix開始、model fit cutoff、normalizer digest、係数/設定からのmodel digest、Dataset ID、コードとruntime manifestのdigestを保存する。同じblock内の異なるprefixを共通recipe digestだけで同一教師と扱わない。各eligible labelにはこの教師vintage、baseline/candidate両側のnative終端値・損益・残数量・ledger digest、介入前native prefix digestを結び、`eligible_pair_outcomes`としてblockの`result.json`へ保存する。digestや教師identityの入替え、元labelとnative結果の不整合は学習前に拒否する。ineligible/terminated pairも教師vintageを残し、成功sampleに変換しない。これらはcompact evidenceであり、全native interval ledger本文は含めないため、後日のdigest検証には同じsource/runtimeでの再現/原ledgerの保全が別途必要である。ここでのprovenanceは外部データの真正性やG4を証明しない。旧TD7モデルや2024Q1履歴は再fit・再分類しない。
- **TD8のfit・校正での時間境界（修正版software-only）**: ブロック最終B0/B1/B1c/B2の共通変換は`train_start..train_stop`のうちblock `fit_cutoff`時点までに到着した値だけから標準化統計を作り、学習・評価で同じ変換を利用する。margin選定の75%一時fitは**別の**`fit_cutoff`時点までのraw sourceから統計をfitし、元labelの意思決定特徴をそのvintageで再encodeする。残り25%のcalibration特徴を一時fit統計に含めず、校正replayも一時fit時点の凍結normalizerで行う。marginの校正replayは全候補でnative source-backed ledger oracleの整合を要求する。学習器の最終block-wide変換を校正時点へ再利用しない。
- **TD8の比較不能pairと失敗attempt**: insolvent/risk termination・終端の残建玉/未約定等によりpairが比較不能な場合、baseline/candidate両側の終端資産・途中までの変化・終了理由・残数量・未約定残を候補/時刻に紐づけて保持する。対象blockの一時fit・最終fitはfail-closedとし、比較可能なsurvivorだけを自動採用しない。指定`output_dir`には成功resultではなく`incomplete-pairs.json`（`INVALID / BLOCKED`）をwrite-onceで保存する。all-FLAT候補の0件を根拠なくゼロ係数/ゼロ共分散に代入しない。部分観測終端値はsettled return・成功labelとして扱わない。
- **現行TD8のsource-backed会計診断**: `evaluation/decision_value_ledger_oracle.py` の区間照合を教師ラベルのbaseline/candidate両分岐（ラベル採用前）と評価全アームに適用する。 未約定イベントも含む全注文のsource ID・execution digest・処理時刻を先に照合してから、注文イベントの約定数量と執行価格から現金・保有数量を再計算し、執行価格にはraw OHLC ± 1tickの必要条件（厳密なlimit/stop約定検証ではない）を課し、各区間の売買回転率とbar markから観測可能な最大DD下限を独立再構築する。これらの反例を検出した上で、Canonical Marketのmark価格と契約乗数で各バーの評価額・期間収益率を復元する。途中のコスト・資金調達・借株・配当・現金利息はobserver記録額を利用し、それらの**料率・経済価格を独立再計算するものではない**。このoracleは株式分割等のcorporate actionを再現せず、split_factor!=1では合格させない。最終BookStateとの一致だけで「独立会計全面検証済み」と主張しない。
- **Write-once出力**: 現行 `run_decision_value_dev_study` は既存出力dirをfit前に拒否し、成功時に `protocol.json`、`result.json`、`summary.md` を作成する。最終書込直前にも既存dirを拒否するが、3ファイルを原子的に一括保存する仕組みは備えていないため、途中例外時の部分出力がないことまで保証しない。

## Artifact and evidence rules

- Canonical JSON/digestのauthorityは `trade_rl.artifacts` に置く。
- Market dataset artifactのcodec/publicationは `trade_rl.data.artifacts` が持つ。
- Candidate/evaluation runはimmutable filesystem artifactとして残す。
- Candidate RunとStudy EvidenceSetのdirectory publicationはstagingからatomic renameする。Windowsで一時的なdirectory lockによるpermission errorが出た場合は有界retryし、sourceの消失やdestinationの出現を検知したら失敗する。部分copyへ切り替えない。
- Candidate Runはresolved result `summary.json`、raw interval return `returns.npz`、implementation/runtime/research-context evidence `provenance.json` の3ファイルを一体としてpublishする。
- Candidate Runは実行前後でimplementation/runtime provenanceが一致する場合だけpublishする。実行中にsource/runtime provenanceが変化したRunを正当なevidenceとして残さない。
- Candidate artifact identityはNPZのZIP圧縮表現そのものではなく、summary/provenanceと検証済みreturn arrayのsemantic contentへbindする。一方、各fileのraw SHA-256/sizeもtamper検出用evidenceとして保持できる。
- Opt-in conditional PPO software diagnosticのcompleted resultは `ppo_conditional_diagnostic_result_v2` / `evidence_contract_version=2` を使う。fit済みpolicyを既存の `ppo_inference_bundle_v2` 形式で一度保存した後、manifest digestと完全なfeed feature schemaを指定して既存loaderで再読込し、observation schema等がfit契約と一致するpolicyだけをheld-out deterministic実行に使う。pre-action observation、action、step info、各step後のBookState、callback時点でcopyしたstateful execution evidenceは実行順を保つtraceとして保存する。evidence manifestはprotocol/source/recipe/runtime/held-out Dataset identityと、再読込・hash確認したbundleおよびtraceのdigest/raw SHA-256/sizeへbindする。`result.json` のpublish直前にもbundle manifest/policy、trace、evidence manifestのbytes・digest・SHA-256・sizeを再照合し、各artifactが同じreadbackのままの場合だけpublishする。失敗時はphase・fit回数・残存artifactのhashをfailure sidecarへ記録して再fitしない。旧completed resultは変更・backfillせず、policy bytesやtraceが保存されていない過去fitからは復元できない。このsynthetic software diagnostic evidenceはlearnability、G0-G3、economic performance、production/live authorityを与えない。PPO環境のexecution observerは省略時 `None` であり、通常のfitへ自動接続しない。
- 同じ入力・設定・identityからはdeterministicなidentityを得る。
- resultを見た後にevidence条件やthresholdを都合よく変更しない。

## Verified Dataset artifact pair

`load_published_market_dataset_artifact` returns
`(PublishedDatasetArtifact, MarketDataset)` in identity-first order from one
verified manifest parse and its matching array payload. The inspector delegates
to this pair loader; the existing dataset-only loader keeps returning
`MarketDataset`. The pair represents one loader verification pass and does not
lock the directory or promise an operating-system-level atomic filesystem
snapshot.

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

Collection liveness is separate from that evidence. The collector process writes an
atomic heartbeat; every exit, signal or console-close path and the external stale
check create the same permanent failure marker exclusively, so the first cause is
kept. A supervisor may relaunch only a child that died between cycles, inside the
frozen 180-second gap including a launch margin, with no marker, no unacknowledged
cycle and a bounded restart count. It never resumes an interrupted cycle, retries a
failed source request or edits the journal; a hard kill during acquisition therefore
still ends the screen. A duplicate or restarted launch against a live collector is
detected through the collector lock and ends as non-destructive contention (exit 4,
no marker). The external check defers its stale limit only while a live supervisor
is inside the bounded startup grace of a relaunched child. The attempt-2 tooling is plan-only until an explicit seal
and, separately, an explicit network launch, both gated by an Issue #810
authorization URL; they do not prove authorization or establish profitability.

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

## Prequential forecast availability

Prequential forecast packet（`strategies/forecasts/prequential.py`）は、Data contractのsupervised label規則（`label_end_time < fit_cutoff`、`available_at <= fit_cutoff`）を持つ `build_causal_forecast_training_set` だけからvintageを作り、fitterへdatasetを渡さずfrozenな `CausalForecastTrainingSet` だけを渡す。packetは `max_training_label_available_at < fit_time <= forecast_available_at <= decision_time` を構築時に強制し、`available_at > timestamp` の決定行はfail closedにする（`activation_delay_hours` はlabel availabilityとfit時刻の分離だけで、決定行のavailabilityは緩めない）。vintageのresidual varianceはin-sampleで、`max_training_label_available_at` は保守的な上界である。

## 非目標

Lean coreが保証しないもの:

- profitability
- Production/live order authorization
- OHLCVから観測不能なmicrostructureの完全再現
- model complexityがedgeを生むという前提
- DB/UI/teacher pipelineが研究成立に必須であること

利益やlive suitabilityはarchitectureではなく、凍結した研究条件とunused-data evidenceで別途判断する。
