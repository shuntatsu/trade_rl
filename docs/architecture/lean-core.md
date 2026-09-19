# Lean core architecture

## 結論

Trade RLの現行coreは、**causalなmarket data、1つのexecution/accounting ledger、hard risk、small strategy interface、銘柄ごとの独立評価**に絞る。

旧U-series、旧Causal Alpha世代、mandatoryな `teacher -> admission -> BC -> RL`、研究DB、UI、世代別runnerは現行研究経路の必須条件ではない。Git historyに残る旧architectureをcurrent treeへ戻さない。

現在の研究目的は、銘柄IDに依存しない共通strategy/model/policyを複数銘柄へ適用し、point-in-time情報と同一の約定・会計条件で、コスト控除後の結果がunused dataでも維持されるかを検証することである。

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

### PPO training layout

`fit_ppo_strategy(normalize_features=True)` explicitly fits one immutable
local-feature standardizer on finite/available training decisions in the selected
symbol/window scope. Each symbol contributes equal total weight per feature;
the pooled variance includes between-symbol mean differences. Scales at most
1e-12 use 1; clipping and online statistic updates are absent. Both layouts and
the returned strategy share that fitted transform. Missing inputs stay zero,
and masks/staleness/intent/weight keep their v2 semantics. The default remains
the raw v2 encoder and persisted default observation payload.

Normalized models require their preprocessing state. `save_normalized_ppo` binds
policy bytes and transform metadata in a write-once bundle. `load_normalized_ppo`
requires the expected manifest digest and the feed's full ordered feature names,
checks both before policy deserialization, and validates policy spaces. Changing
evaluation statistics or loading a model without its transform is not supported.

PPO environment/fitter callers can explicitly provide an immutable
`PreTradeRiskConfig` via `risk_config`. The same configuration applies when the
environment is created and after every reset, in both sequential and interleaved
layouts. Omission preserves the legacy execution-leverage-limited risk with
drawdown start/stop 1.0. This is a training-only opt-in; it does not alter policy
observations, rewards, action meanings, or replay risk. Research callers must
bind the explicit training configuration in their protocol and separately verify
that evaluation risk matches the intended deployment objective.

Directional PPOのfitとdevelopment評価は `DIRECTIONAL_BASE_EXECUTION_COST` を共通authorityとして使う。zero overlayでもDataset由来のfee / spread / funding / borrowは消さず、特に `borrow_rate_multiplier=1.0` を学習・評価の両方で維持する。過去のPPO evidenceは生成時の旧implementation SHAにbindされたままであり、このcorrected execution contractのcontrolとして自動再利用しない。

`fit_ppo_strategy` の既定は従来どおり `sequential` であり、単一 `PPOTradingEnv` がfit symbolをfull-window episode単位でround-robinする。既存Studyやcandidateがlayoutを明示しない場合の意味は変えない。

Directional PPOはfinite-horizon endpointをdevelopment replayと揃えるため `settle_terminal_position=True` を明示する。agent decisionは `stop_index - order_latency_bars - 1` より前だけで行い、その後の予約区間では環境が `FLAT` proposalを同じ `PreTradeRisk` と `MarketExecutor` へ1 barずつ流す。forced settlementをagent actionとして偽装せず、settlement区間はagent transitionではなくepisode末尾のterminal economic costとして1つのterminal transitionへ畳み込む。このためsettlement内部へPPOのdiscount factorを別途適用せず、terminal rewardにはagent intervalのlog returnとsettlement各intervalの実現log wealth changeを加算する。capacity / turnover / venue admissionで完全flatにならない場合は残余を隠さない。normalizerはagentが実際に観測するdecision rowsだけでfit/validateする。これはper-symbol training endpointの補正であり、developmentのshared-cash cross-symbol accountingまで同一になったとは主張しない。generic PPOの既定は `settle_terminal_position=False` のままである。

`interleaved` は明示選択する学習layout capabilityである。fit symbolごとに同じ `PPOTradingEnv` を `symbol_indices=(その1銘柄,)` で固定して1個ずつ作り、in-process `DummyVecEnv` で同一policyへ束ねる。観測、reward、execution/accounting、hard risk、network、entropy係数、総 `total_timesteps` は変更しない。callerは `rollout_steps_per_env` を結果を見る前に明示し、`rollout_steps_per_env × env数` が既存PPO minibatch size 64で割り切れることを要求する。

このlayoutは学習sampleの並び方を変える実装能力であり、性能改善・profitability・winnerを意味しない。developmentで比較する場合は、exact layoutとrollout stepsを別Controlled Factorとして結果前にpreregisterする。PPOの学習deviceはCPUへ固定し、同じsource/runtime identityがGPU有無だけで別のSB3 execution deviceを選ばないようにする。interleavedではSB3がsub-envへ異なるreset seedを配るため、execution RNGをfactorへ混ぜないよう`slippage_std > 0`の確率的slippageは現時点でfail closedにする。

## StrategyとRiskの責任分離

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

不変条件:

- order submissionとfillを区別する。
- feeはrealized fillに対して一度だけ計上する。
- spread / impactを複数channelで二重控除しない。
- `interval_net_return` は実約定・全明示cash flow反映後の最終equityを正本とする。`interval_gross_return` は同じ実約定経路について、最終equityへexecution costとborrowを戻し、signed funding・dividend・cash interestを除いて価格損益を分離する。OPENからのasset returnへ事後weightを掛ける近似や、cost-zero条件で戦略を再実行した反実仮想とは扱わない。
- partial fill後のpositionはrealized fill quantityで更新する。
- `fill_ratio` と `unfilled_turnover` は注文の消化状態を表すため、requested側と同じsubmission reference priceでfilled quantityを評価する。adverse/favorableな実約定価格の変化だけで注文残量が消えたように見せない。`filled_turnover` は実際に売買した金額を表すためactual fill notional / starting equityを維持し、このcompletion指標とはprice basisを分ける。
- lot数量はdecimal表記をexact rationalへ変換し、承認された整数lot数を保有・注文残量の共通authorityとする。任意の初期端数は保持し、float表示はゼロ方向へ保守的に射影する。float表示値の足し引きで次の残高を作らない。
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
and all existing constraints still apply and target reconciliation does not opt in.

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
Exact closing deltas still project conservatively to float requests, so unusual
non-representable inventories can retain an executable lot. No dust is written off.
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
- effective `tick_size > 0` のとき、外部/manual LIMITの `limit_price` と STOP_MARKETの `stop_price` はそのpoint-in-time tick grid上でなければadmissionでfail closedする。floatへの射影で生じる数ULPの表現誤差だけは同一grid点として扱い、実質的なoff-tick boundを後段のexecution-price roundingで別価格へ自動変換しない。一方、canonical target reconciliationがoffsetから内部生成するLIMIT/STOP boundは、未来のeligible-bar ruleを先読みせずsubmit時点のeffective tickだけを使って保守的にgridへsnapする（LIMIT buyは切り下げ/sellは切り上げ、STOP buyは切り上げ/sellは切り下げ）。eligible時点でruleが変わっていれば通常admissionが再検証する。`tick_size == 0` は従来どおりprice grid未指定として扱う。
- maker/taker cost分類はorder typeだけでなくrealized liquidity roleに合わせる。LIMITがそのprocessing barで初めてeligibleになり、同じbarのopenで既にmarketableなら、そのfillはliquidity-takingとしてtaker feeとfull spreadを使う。新規LIMITがbar内touchまでrestする場合、および以前のeligible barからcarryされていたLIMITが後続bar openでcrossする場合はresting orderとしてmaker feeとhalf spreadを使う。MARKET / STOP_MARKETは従来どおりtakerである。この分類はfill価格・数量・capacityを変更しない。
- fundingは対象時刻・符号・quantityに対して一度だけ計上する。
- borrow、mark-to-market、liquidationを別channelで追跡する。session calendarでclose-to-close間隔がnominal barより長い場合、closed-session gap分のcash interest / borrowはnext-open fill前のbookへ、processing bar分はfill後のbookへ適用する。continuous cadenceではこの分割は発生せず、従来の1-bar elapsed carryと等価である。
- terminal mark-to-marketとforced closeを混同しない。
- OHLCVだけからqueue positionやhidden liquidityを再現したとは主張しない。

`MarketExecutor` の標準executionはnext-openであり、decision row `t` の注文は最初にrow `t+1` のopenで約定可能になる。participation capacityのvolume authorityは `ExecutionCostConfig.processing_bar_volume_capacity` に明示bindする。

- `True` は既存互換のlegacy modeであり、processing bar全体のvolumeをcapacity poolへ使う。同一barのopen時点ではbar最終volumeは未確定なので、これはpoint-in-time liquidity forecastではない。既存canonical Dataset / Study / Runの意味を変えないためdefaultとして保持する。
- `False` はcausal stress modeであり、processing barの直前に完全終了したbarのvolumeをcapacity poolへ使う。base-volumeをmarket notionalへ換算するときも、その前barのcloseをreference priceに使う。current processing barの最終volumeはcapacityへ使わない。
- どちらのmodeも同じspread / impact / fee / accounting経路を使い、mode差だけで別のexecution-policy digestになる。`False` は将来volumeの予測モデルではなく、同一bar最終volumeへの依存を除くための保守的stressである。

Candidate runのexecution overlayがzeroでも、datasetに含まれるpoint-in-time fee/spread等までzeroになるわけではない。既存execution fieldはcanonical executorを通る。

## Independent per-symbol evaluation

同じfrozen strategy objectを各銘柄へ独立Replayする。

```text
universal frozen strategy
  ├─ symbol A independent replay
  ├─ symbol B independent replay
  └─ ...
```

Aggregate P&Lだけを成功判定の正本にしない。ある銘柄の利益で別銘柄の損失を隠さず、各symbol × strategyについてreturns、drawdown、turnover、execution cost、funding、borrow、trade/fill/rebalance diagnostics、terminationを保持する。

## Artifact and evidence rules

- Canonical JSON/digestのauthorityは `trade_rl.artifacts` に置く。
- Market dataset artifactのcodec/publicationは `trade_rl.data.artifacts` が持つ。
- Candidate/evaluation runはimmutable filesystem artifactとして残す。
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
retry follows a collection failure. Pending decisions expire after ten seconds;
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
