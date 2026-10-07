# Package boundaries

## 結論

Trade RLはtop-level responsibilityを `artifacts / data / integrations / risk / simulation / strategies / evaluation` に分ける。各package内の物理フォルダは研究・実行責務と一致させ、旧private pathを残すためだけのforwarding shimは置かない。

`trade_rl/_validation.py` はstandard-library-onlyの最下層validation authorityである。

## Current package tree

```text
trade_rl/
├── __init__.py
├── _version.py
├── _validation.py
├── artifacts/
│   ├── canonical.py
│   ├── hashing.py
│   ├── atomic_pointer.py
│   ├── atomic_write.py
│   ├── store.py
│   └── verified_file.py
├── data/
│   ├── market.py
│   ├── market_order_rules.py
│   ├── contracts.py
│   ├── identity.py
│   ├── source.py
│   ├── view.py
│   ├── artifacts/{codec.py,publication.py}
│   ├── build/{config.py,builder.py,economics.py}
│   └── features/{core.py,cross_asset.py,economic.py,multitimeframe.py,numerics.py,price_channels.py}
├── integrations/
│   └── binance/
│       ├── types.py
│       ├── transport.py
│       ├── cache.py
│       ├── vision.py
│       ├── book_depth.py
│       ├── agg_trades.py
│       ├── metadata.py
│       ├── carry.py
│       ├── forward.py
│       ├── forward_evidence.py
│       ├── forward_rules.py
│       ├── market_order_profile.py
│       └── dataset.py
├── risk/
│   ├── inputs.py
│   ├── portfolio.py
│   ├── pretrade.py
│   └── emergency.py
├── simulation/
│   ├── accounting.py
│   ├── execution.py
│   ├── bar_path.py
│   ├── liquidity.py
│   ├── quantities.py
│   ├── depth.py
│   ├── orders/{model.py,admission.py,reconciliation.py}
│   ├── stateful/{runtime.py,execution.py,bar_lifecycle.py,order_transitions.py,symbol_fills.py}
│   ├── targets/execution.py
│   └── diagnostics/{execution_stress.py,funding.py,runtime_performance.py,runtime_performance_io.py}
├── strategies/
│   ├── dataset_scope.py
│   ├── position_duration.py
│   ├── interface.py
│   ├── position_intent.py
│   ├── controls.py
│   ├── allocation.py
│   ├── carry.py
│   ├── rules/{trend.py,mean_reversion.py,channel_breakout.py}
│   ├── forecasts/{controller.py,supervised.py,training_trace.py,ridge.py,lightgbm.py,stream.py,prequential.py,_ridge_math.py,simple_return.py,simple_stream.py,simple_prequential.py}
│   └── rl/{intent.py,ppo.py,a2c.py,ppo_normalization.py,ppo_artifact.py,a2c_artifact.py}
└── evaluation/
    ├── replay.py
    ├── bot.py
    ├── metrics.py
    ├── evidence.py
    ├── series.py
    ├── directional.py
    ├── allocation.py
    ├── forecast_allocation.py
    ├── objectives/{__init__.py,contract.py,clock.py,binding.py}
    ├── carry.py
    ├── directional_candidates.py
    ├── directional_selection.py
    ├── directional_study.py
    ├── ppo_risk_study.py
    ├── ppo_feature_study.py
    ├── ppo_feature_checkpoint.py
    ├── ppo_normalization_replication.py
    ├── ppo_normalization_execution.py
    ├── ppo_normalization_activation.json
    ├── rl_family_comparison/{__init__.py,contract.py,cells.py,comparison.py}
    ├── paper/{__init__.py,store.py,account.py,engine.py,control.py,supervisor.py}
    ├── gates/{models.py,resolve.py}
    ├── comparison/{bootstrap.py,paired.py,seed_robustness.py,strategies.py}
    ├── robustness/
    │   ├── capacity.py
    │   ├── closed_trades.py
    │   ├── fold_metrics.py
    │   ├── perfect_information/{bound.py,solver.py}
    │   └── walk_forward/{capabilities.py,folds.py,sealed_test.py,stitching.py}
    ├── runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py}
    ├── final_test/{__init__.py,contracts.py,workflow.py}
    └── experiments/
        ├── errors.py
        ├── codec.py
        ├── inspection.py
        ├── store.py
        ├── evidence.py
        ├── analysis.py
        ├── delta.py
        ├── workflow.py
        ├── contracts/
        └── bootstrap/{__init__.py,config.py,binance.py,workflow.py,cli.py}
```

`evaluation/experiments/` はdevelopment-onlyのhigher-level Study lifecycleを所有し、`evaluation/runs/` のverified Run Coreを再利用する。`evaluation/experiments/bootstrap/` はそのStudyを実行する前のcanonical preparationだけを所有する。`evaluation/final_test/` はfrozen WINNER Studyをread-onlyでinspectionし、unused-futureを開くone-shot authorizationだけを別rootへ発行する。final Dataset、Replay/P&L、stress、Production/live authorizationは所有しない。

`evaluation/rl_family_comparison/` はPPO/A2C比較の固定設定、replay-cell schema validation、および純粋な開発判定oracleだけを所有する。SB3 constructor mocksと合成セルが固定設定・判定規則の回帰を検査する。このpackageはDataset/artifact I/O、fit/replay、ledger、paper、production/live authorizationを所有せず、`trade_rl.evaluation` の公開APIも拡張しない。

## Provider evidence boundary

The separate paired-carry path keeps provider assembly in
`integrations/binance/carry.py`, deterministic monthly matched-quantity sizing
in `strategies/carry.py`, and replay composition/diagnostics in
`evaluation/carry.py`. It submits explicit quantities through canonical orders
and preserves GTC residuals; it does not convert a quantity hold to a newly sized
weight each hour. The existing BookState and stateful executor own all P&L.
Futures collateral checks are a projection of that same book excluding spot
assets, not an additional ledger. No live order connection is provided.

`integrations/binance/forward.py` owns fresh public spot/perpetual evidence
capture. It preserves exact raw response bytes, request/receipt timing and
hashes, validates clock/depth/settlement schemas and publishes a source-eligible
snapshot only when all required responses pass. It does not simulate fills or
own a paper account. The bounded Binance transport remains the HTTP owner.
`forward_evidence.py` revalidates the immutable response graph and reconstructs
market summaries from raw bytes; consumption-time freshness is separate from
offline historical verification. `forward_rules.py` captures current public
exchange metadata and derives supported market lot/notional/price bounds. It
retains averaging semantics and unsupported account-rule limitations. Neither
module consumes a trading account or authorizes an order.

Forward capture emits the v2 fixed 100-level depth profile. The reader supports
the separately identified historical v1 20-level profile only for exact artifact
verification; each schema must match its own URL roster and depth bound. Profile
selection does not change the provider-independent execution model.

`evaluation/paper/store.py` owns protocol-bound, append-only event persistence
with canonical JSON hashes, contiguous sequence/parent checks, atomic
compare-and-append and idempotency. Reopening permits SQLite recovery only after
validating the expected protocol digest. This is an evidence store, not another
financial ledger. `paper/account.py` composes the existing FundingCarryBot,
BookState and depth matcher, tracks exact fill-time holdings for later funding
receipts, and owns permanent risk/quality stops. `paper/engine.py` binds verified
source references to saved commands, commits isolated account transitions, and
replays every result on restart. `paper/supervisor.py` binds public collection
to frozen source/runtime/protocol identity and owns the single-collector lock,
rule refresh, pending-decision resumption, terminal window and durable failure.
`paper/control.py` records cycle start before acquisition and completion only
after account commits, preventing restart from hiding failed or unused captures.
These control records assign no financial meaning. `paper/operations.py` freezes
the ninety-day economic screen, runs sixty-second slots and exposes explicitly
chain-only operational status. `paper/assessment.py` excludes a running collector,
replays the pinned event tip and all financial inputs, audits control/source
rosters, then measures and applies the fixed screen. `paper/cli.py` exposes seal,
run, status and write-once evaluation commands. The account/engine expose due
announced funding for previously held exact quantities, including after exits.
These operations contain no production routing.

The directional development CLI composes the existing shared-cash replay and
maintained strategy fitters. `directional.py` owns terminal-close scheduling and
screen metrics, `directional_candidates.py` owns the fixed fit roster,
`directional_selection.py` owns family aggregation, and `directional_study.py`
owns write-once study evidence. These modules do not own execution accounting,
exchange connectivity, or the canonical Study lifecycle. Price-channel rolling
bounds and availability belong to data; the rule consumes them as observations.

`simulation/quantities.py` owns exact decimal-rational conversion, canonical
state parsing, conservative float projection and lot quantization. Liquidity
allocations carry accepted integer lots and their quantum; accounting and the
pending-order state consume that same signed fill. Exact state is part of the
book clone and pending-order persistence contract, not strategy preprocessing.
`simulation/depth.py` is a provider-independent paper depth matcher. It submits
accepted integer-lot fills to BookState, owns displayed-size participation and
bid/ask weighted prices, and preserves unfilled lots. It owns no source capture,
journal, policy, or independent account. Callers must prohibit snapshot-capacity
reuse. Optional fill valuation prices in BookState separate cash execution from
position marks without changing the default historical execution contract.
`ppo_risk_study.py` binds a separate five-seed, training-risk-only comparison to
the completed directional baseline. It reuses the same fit and replay owners,
checks source/runtime isolation, and reports relative loss reduction separately
from the existing absolute qualification gate.

`strategies/rl/ppo_normalization.py` owns optional fit-only local-feature
standardization and its immutable metadata. `ppo.py` applies one shared fitted
transform in training and inference while keeping the raw v2 default unchanged.
`intent.py` owns the private deterministic three-action observation/intent
adapter shared by PPO and A2C, and the shared `ppo_observation_v2` semantic
contract (`PPO_OBSERVATION_SCHEMA`, `PPO_GLOBAL_FEATURE_NAMES`, and
`ppo_observation_contract_payload()`). `ppo.py` keeps the historical public
names as compatibility re-exports, but is not the semantic contract owner.
`a2c.py` owns explicit sequential CPU fitting, rollout rounding, and fit-scope
metadata over the shared `PPOTradingEnv`.
Its nominal episode coverage describes budget capacity, not observed transitions
or economic performance.

`a2c_artifact.py` owns the separate A2C inference-bundle schema. Save validates
the A2C policy family, spaces, realized `num_timesteps`, and policy `seed`
against fit metadata before publication. Load requires fit metadata, binds the
feature feed and policy digest in a canonical manifest, and verifies the
manifest, feed, metadata, and private policy copy before deserialization; it
checks the loaded policy spaces, realized timesteps, and seed before returning
the strategy.

`ppo_artifact.py` owns durable PPO inference bundles. Both publishers treat any
existing destination filesystem entry, including a dangling symlink, as occupied
under the write-once contract. The current bundle binds raw or normalized policy
bytes, Observation v2, selected feature semantics and optional preprocessing
metadata under one manifest digest; load validates that digest, feed feature
schema and policy spaces before inference. The historical normalized-only bundle
remains a compatibility reader/writer contract.
`integrations/binance/book_depth.py` と `integrations/binance/agg_trades.py` は、Binance Visionのprovider-specific historical evidenceを所有し、`MarketDataset` assemblyやexecution/P&L semanticsから分離する。

`book_depth.py` はUSD-M daily `bookDepth` の `timestamp,percentage,depth,notional` を厳密にdecodeし、maintained percentage bands、snapshot completeness、累積depth/notional、implied average price、timestamp orderingをfail-closedに検証する。top-of-book quote、bid/ask spread、market impact、slippage、`MarketDataset`、`ExecutionEconomicsProfile`、execution/accounting、P&Lのauthorityにはしない。

`agg_trades.py` はUSD-M daily `aggTrades` のaggregate trade ID、price、quantity、underlying first/last trade ID、event timestamp、buyer-is-makerを厳密にdecodeする。archiveはheaderless、またはmaintained seven-column headerが1行だけ存在する形式だけを受理し、それ以外をfail closedに拒否する。multi-million-row daily archiveを全Python string行列へ展開せずstreaming CSV + compact typed buffersで処理する。aggregate tradesはrealized trade-flow evidenceであり、order-book depth、top-of-book spread、queue position、market impact、slippage、hypothetical fill probabilityのauthorityにはしない。

raw archive bytesと既存Vision cache sidecarのURL / SHA-256 / size / `acquired_at` が取得証拠のauthorityであり、`BinanceBookDepthSeries` と `BinanceAggTradesSeries` はその再現可能なderivativeである。bookDepth rowの`timestamp`と現行`available_at`はarchive内に記録されたmarket-observation timestampを表す。aggTradesの`transact_time`もmarket event timestampであり、いずれもarchive publication timestampやそのevent時点でVision ZIPがpublicだったことを意味しない。providerが保証していないpublication lagや固定sampling cadenceを捏造しない。

別sourceの価格を使ってbookDepth品質を検査する場合は、reference valueの`available_at`が対象liquidity snapshotの`available_at`以下であることを必須にする。未来available referenceによる品質判定を許さない。historical archiveの異常値はraw evidenceを修復・丸めして隠さず、構造検証と明示的なcausal reference-alignment検証を分けてfail closedに扱う。

これらprovider evidenceをbar-level participation、spread、impact、slippageやstrategy featureへ変換する規則は別の研究変更である。導入する場合は結果を見る前にpreregisterし、必要なavailability/join semanticsとexecution-economics identityを固定した新しいDataset / Studyを構築する。既存canonical Studyやfrozen Experiment evidenceを書き換えない。

## Ownership

### `artifacts`

汎用のcanonical encoding、digest、atomic publication primitive、verified fileを持つ。`atomic_rename_directory`はstaging directoryの同一filesystem renameを使い、Windowsの一時的なpermission errorだけをsource/target再確認付きで有界retryする。copy fallbackでatomic性を弱めない。market data、strategy、evaluation等のupper layerを知らない。

### `data`

`MarketDataset`、point-in-time contract/source、identity、bounded view、artifact codec/publication、dataset build、causal feature computationを持つ。strategy/evaluation/simulationへ依存しない。`data/features/numerics.py` はcanonical Dataset identityへ入るfeature計算のportable scalar/reduction semanticsを所有し、`core.py`、`cross_asset.py`、`builder.py` が共有する。モデル学習、simulation P&L、汎用evaluationの数値計算まではこのauthorityへ含めない。現行 `MarketBuildConfig` は `market_build_v3` / `portable_feature_numerics_v1` をbuild identityへbindし、historical `market_build_v2` artifactのreader互換はartifact contractとして維持する。`data/build/economics.py` は build-level `ExecutionEconomicsProfile` の単一ownerであり、`MarketBuildConfig` のfeature/build semanticsとは分離する。profile省略時はlegacy economic behaviorを維持し、明示profileは既存economic-semantics経路を通してimmutable Dataset fields/content identityへbindする。

### `integrations`

外部venue/providerを内部data contractへ変換するadapter層。Binanceはtransport、cache、Vision archive、metadata、dataset assemblyを分離する。strategy/evaluationを知らない。`BinancePublicTransport`の既定値はnetwork-enabledの既存互換を維持し、bootstrapだけがsource freeze後に`allow_network=False`を明示してcache-only化する。

### `risk`

portfolio/pretrade/emergencyのhard safety・feasibilityを持つ。strategyのalpha判断やevaluationを所有しない。

### `simulation`

`MarketExecutor` owns the opt-in insolvency valuation policy as part of resolved
execution identity. Both the historical zero floor and signed marked-debt mode
use the same `BookState` ledger and terminal quantity/margin transition. Strategy
and evaluation consumers must not implement a second debt or profit ledger.

execution/accountingの経済正本と、order/stateful/target/diagnosticsを持つ。strategy/evaluationから独立することで、同じexecution semanticsを複数研究候補で共有できる。

`orders/model.py` owns explicit MARKET reduce-only identity, strict decoding and
event evidence; `orders/admission.py` rejects requests beyond exact inventory.
`liquidity.py` takes an explicit exact initial position for reduce-only requests
and advances it in allocation priority order for all accepted fills. It owns no
BookState mutation. `stateful/symbol_fills.py` supplies and rechecks inventory
after intervening margin handling, applies accepted lots to BookState, reconciles
capacity to actual fills and expires exhausted closing remainders; `runtime.py`
projects the order flag into events. `data/market_order_rules.py` owns immutable
profile/rule values and decimal grid intersection without venue or simulation
imports. `integrations/binance/market_order_profile.py` alone owns supported
profile construction, raw evidence publication and full source rederivation,
reusing the strict `forward_rules.py` parser. It has no order API or P&L authority.
The private factory capability prevents supported public construction/replacement,
not arbitrary Python reflection. Profiles add no fields to the Dataset schema.
`execution.py` binds an opt-in profile and stress into policy identity and resolves
per-order rules; `orders/reconciliation.py` activates same-side reduce-only exits.
Admission and allocation enforce quantity bounds and distinct per-order notional
floors. Unselected symbols and omitted-profile behavior retain their contracts.

### `strategies`

The PPO environment owns rebasing its cached quantity proposal after a processed
split; `evaluation/replay.py` owns the same transition for its single-symbol and
shared-cash callers. `simulation/targets/execution.py` supplies book mark prices
to `orders/reconciliation.py` for weight sizing while keeping trading close as
the order reference. With no held quantities, it resolves current dataset marks
before entry sizing so default initial book marks are harmless. The accounting
and execution owners remain unchanged.

small strategy interfaceとlogical intent、controls、rule、forecast、teacher-free RLを持つ。evaluationを知らない。`dataset_scope.py` はdatasetに束縛されたfeature/symbol selection validationの単一ownerであり、forecastとRLのsibling familyが互いの内部実装へ依存せず共有する。`position_duration.py` は実際のsigned quantityから保有episode ageを導き、minimum-hold中のintent制約を共通定義する。model自身やcandidate config自身の不変条件validationは各ownerに残す。

`StrategyObservation.gross_position_return` と `current_position_quantity` はoptionalなexecution-derived inputである。`evaluation/replay.py` はexecutionのfill `OrderEvent.execution_price`と現行book markからsigned mark-to-average-fill returnを計算し、現時点の実約定quantityとともに `RegimeAdaptiveStrategy` へ渡す。strategy packageはfill ledgerやreplayへ依存せず、adaptive exit requestのlatchを公開する。`current_intent` は直近のeffective targetであり、未約定・部分約定後の実保有側とは異なることがあるため、adaptive latchはsigned filled quantityで管理し、数量が0になるまで維持する。replayはそのlatchがあるFLAT intentに限りminimum-hold constraintをbypassする。gross returnはentry後fee、funding、borrowを含まない。exit fillはtrigger後のeligible execution stepに発生し、gapやliquidityを含む経済保証ではない。

### `evaluation`

lower layerを利用してReplay・metrics・gate・comparison・robustness・concrete runを構成する。

`evaluation/runs/` の責務は一回の計算とimmutable Run evidenceである。

- `candidate_suite.py`: 5 candidates + 3 controlsのfit/replay構成。
- `config.py`: Run JSONの単一parse/resolution authority。
- `execute.py`: resolved specから既存candidate suiteを一度実行するin-memory seam。
- `provenance.py`: implementation/runtime/research-context provenance生成。
- `artifact.py`: summary/raw returns/provenanceのpublication、verified load、semantic identity。Observation-v3 shared-cash PPO replayを含むRunは`lean_candidate_result_v7`へ追加portolio return seriesとsettlement / ledger evidenceをbindし、loaderがreturn / maximum drawdownをraw seriesから再計算する。
- `candidate.py`: 上記を順番に呼ぶ薄いfilesystem CLI/facade。

`trade_rl.evaluation.runs` はcandidate-run contract、execution、artifact inspection/publication、provenance constructionのTier-2 public facadeである。`config.py`、`candidate_suite.py`、`execute.py`、`artifact.py`、`provenance.py` は引き続き実装ownerであり、facadeはこれらをwrapperなしでre-exportするだけとする。production codeは `evaluation/runs/` の外からRun Coreを利用するときfacadeを経由し、package内部は循環を避けるためowner moduleを直接参照してよい。Tier-1 `trade_rl.evaluation` の公開面はこの規則によって拡大しない。candidate-runのpersisted schema互換契約はPython import pathとは独立して維持する。

`runs` はhigher-level experiment lifecycleを知らない。`evaluation/experiments/` はStudy/Experiment contract、append-only store、multi-seed EvidenceSet、analysis、controlled delta、lineage/budget/freeze workflowを所有する。`contracts/research.py` の `StudyResearchContext` / `ConsumedEvidence` はStudyをまたいで既知development evidenceが次の研究定義へ流入した事実をmachine-readableに表し、context-bound `StudyPlan` digestの一部となる。これはresult/selection oracleではなくprovenance authorityである。`contracts/study.py` がversioned `StudyProtocol` identityと、PPO holding-duration protocolの完全一致risk profileを所有し、`protocols.py` はv1 independent-account / v2 shared-cash result eligibilityとwinner orderingをpure selectorとして共有する。`analysis.py` はv3 seed-symbol comparisonとv4 combined shared-cash portfolio comparisonを所有し、v4はpersisted portfolio seriesから再計算されたreturn / drawdownを選定へ渡す。`codec.py` はpersisted JSONから既存contractへのfail-closed decodeとstable payload/identity変換を所有し、`inspection.py` はdisk graphからのread-only state reconstruction・tamper validation・`inspect_study`を所有する。`workflow.py` はmutation lock下のcommand orchestrationだけを所有し、各mutation前のdisk再構築と既存failure-injection seamを維持する。

`evaluation/experiments/bootstrap/` は次だけを所有する。

- `config.py`: strict `CanonicalM2BootstrapConfig` parse/normalization/preflightと単一seed-policy authority。historical v1-v5のpayload/digest/read semanticsを維持する。v2は明示的なbuild-level execution economics、v3はpreregistered final window、v4はさらに `StudyResearchContext` をbootstrap identityへbindし、新規final-eligible research lineの正本となる。v5はlegacy independent-account PPO holding selector、v6はnew shared-cash PPO holding selectorを別StudyPlan schemaへbindする。
- `binance.py`: exact exchange-info / Vision source freeze、raw-source roster、cache-only transport composition。
- `workflow.py`: source → canonical dataset → immutable StudyPlanをwhole-root stagingで構築し、manifest検証後に一回だけpublishする。v2ではbuildへ渡したexecution economicsと、生成/reloadしたDataset economic arraysおよびidentity-bound profileの一致もfail-closedで検証する。
- `cli.py`: `--config` / `--output` をparseしてworkflowを呼ぶだけのfilesystem adapter。
- `__init__.py`: intentionally narrow public facade。

`trade_rl.evaluation.experiments` から公開するbootstrap APIは `CanonicalM2BootstrapConfig`、`CanonicalM2BootstrapResult`、`bootstrap_canonical_m2_study`、`inspect_canonical_m2_bootstrap` の4つだけである。source-freeze private helperはpublic contractではない。

Bootstrapはpreparation-onlyであり、baseline、Controlled Experiment、winner freeze、sealed final-test authorizationを実行しない。`evaluation/runs -> evaluation/experiments` の逆依存を作らず、`integrations`から`evaluation`へ依存させず、`evaluation/experiments/bootstrap`からsealed final-test ownerへ依存させない。`evaluation/final_test` は逆向きのread-only consumerとして `evaluation/experiments` のinspection/contractsだけへ依存し、data/integrations/strategies/replay/runs/robustnessをimportしない。

## Frozen forecast ownership

`strategies.forecasts.supervised` remains the sole supervised-row selector.
`training_trace` freezes its actual pooled symbol, feature row, label endpoint,
prices and publication clocks. Ridge fitting consumes that same selected-row
object through one solver; it does not select rows a second time.

`strategies.forecasts.stream` owns the simulated availability block, Ridge
vintage, packet and immutable stream JSON contracts. `prequential` fits each
declared prefix and produces only its following prediction block. It owns no
intent selection, cash book, execution, portfolio risk or RL environment.

The family facade directly exports `ForecastBlock` and `FrozenForecastStream`
from `stream`, and `fit_prequential_ridge` from `prequential`. The
prequential owner generates causal artifacts only and does not own an intent
adapter. The Tier 1 `strategies` facade is unchanged. The forbidden
`strategies.rl -> strategies.forecasts` import remains forbidden; downstream
decision adapters require a separately reviewed ownership contract.

## Private development study boundary

`evaluation/ppo_feature_study.py` owns a write-once, development-only paired PPO
feature ablation. It reuses the frozen Dataset, existing PPO fitter, shared-cash
directional replay, and ledger evidence rather than adding a second execution or
accounting implementation. Its private CLI and result schema bind the baseline
feature roster against the same roster plus three BTC-relative return features,
the fixed seeds, independent per-symbol evaluation accounts, candidate-only
stress scenarios, and the result-blind admission rules. This module is not a
public package facade and does not expand `trade_rl.evaluation` or authorize
paper/live orders; a pass can only require a later prospective paper study.

`evaluation/ppo_feature_checkpoint.py` separates that fixed development study
into completed fits, individual replay cells, arm assembly, and comparison.
It owns a distinct checkpoint protocol and publication lifecycle, while reusing
`ppo_feature_study` for economic validation and comparison. It uses the existing
PPO inference bundle and verified-file primitives; it does not implement a
second model serializer. A completed fit or cell is reusable only after its
protocol, source, runtime, feature schema, and artifact digests are verified.
Interrupted work is not a completed checkpoint. Legacy partial study roots
cannot be imported into this runner. The evaluation public facade is unchanged.

`evaluation/ppo_normalization_replication.py` owns the sealed, result-blind
corrected-economics PPO fit-only feature-standardization protocol. It freezes the
five matched seeds, raw/normalized arms, common current directional execution
contract, 2023-2024 development window, runtime/artifact identity, relative gate,
absolute family gate, and no-rescue boundary. It does not fit a model or publish
P&L.

`evaluation/ppo_normalization_execution.py` owns only the later software/evidence
boundary for that protocol: the exact ten fresh-fit slots, prepared-root state machine,
PPO fit delegation, inference-bundle publication/reload, current shared-cash directional
replay, no-refit verification, and comparison recomputation. It does not implement a
second PPO trainer, normalizer, executor, accounting path, selection oracle, or repository-
global one-shot transport. `prepare_replication_execution` is the only root-creation
transition: it validates current activation provenance before filesystem mutation, builds
the complete root in sibling staging, validates it, and publishes by atomic rename.
Claim/failure transitions are private and derive activation/implementation identity from
that prepared root rather than caller-supplied digests. Per-root immutability therefore
does not by itself establish repository-global exactly-once execution. The separate
`tools/ppo_normalization_actions.py` transport and
`.github/workflows/ppo-normalization-execution.yml` own that repository-global boundary:
an open Draft request PR may change only the canonical execution-request record, must
contain current `main`, and must have exact-head Core / real-PPO / Guide / generic-review
Green before a write-authorized canonical request comment can proceed. That request HEAD
is authorization provenance only: the request record separately binds the reviewed/sealed
implementation source SHA, and economic execution / no-refit verification checkout that
sealed source rather than inheriting later Python changes from current `main`. The transport
revalidates the merged implementation seal/review tags and the frozen source Artifact,
requires the fixed activation tag to be absent, and creates that tag before crossing any
economic slot boundary. A failed activated run may publish only a non-economic failure
receipt; partial slot evidence is not uploaded as execution evidence.

The sealed protocol's `source_blobs` remain historical preregistration provenance; later
correctness fixes to the maintained PPO/artifact path are not rewritten into those bytes.
The exact current execution implementation/runtime is instead bound by separately reviewed
activation provenance and rechecked around long fit/replay before durable publication.
`evaluation/ppo_normalization_activation.json` is a canonical non-Python activation
authority. It is committed with `activation_sha256=null` in the software-only state.
The candidate implementation identity intentionally hashes `trade_rl/**/*.py` only and
canonicalizes Python source line endings (`CRLF -> LF`, bare `CR` rejected) before hashing,
so clean checkouts on different host policies reconstruct one source identity. A later
result-blind activation commit can therefore bind the reviewed Python implementation without
changing that implementation digest. The activation itself must bind result-blind
implementation-seal, fresh-reconstruction, and assurance-review evidence digests.

A local `verified.json` is no longer sufficient to make a comparison independent.
Before comparison publication, the ten verification-record identities must be transitively
bound to a fresh verifier artifact authority carrying repository/run/artifact identity,
raw artifact SHA-256 and the matching GitHub API digest. The one-shot workflow therefore
uploads execution evidence only after all ten slots complete, re-downloads that complete
artifact by immutable id/run/raw digest into a separate no-refit verifier job, uploads a
complete verification artifact, and only then re-downloads it in the finalizer to construct
the verifier authority and reveal `comparison.json`. Execution requires the exact
activation runtime; the verifier uses a separately frozen stable-runtime contract requiring
the same Python implementation/version, machine architecture, OS family, and complete
bound package map while recording kernel release without making it an equality gate.
Bundle manifest and all parent paths are validated before SB3 deserialization, and realized
PPO timesteps must equal the sealed 262,144 budget. The committed activation resource
remains null until the authenticated request lifecycle actually creates the one-shot
activation; transport capability by itself does not authorize or execute economics. These
modules remain private evaluation surfaces that do not expand
`trade_rl.evaluation.__all__`.

`evaluation/ppo_feature_checkpoint.py`'s private CLI provides `prepare`, `fit`,
`replay-cell`, `assemble-arm`, and `finalize`, each with `--source` and `--output`.
`prepare` requires a fresh root
and writes `checkpoint-protocol.json`, which embeds the unchanged economic
`core_protocol` and the checkpoint execution contract. `fit` and `assemble-arm`
select `--factor` and `--seed`; `replay-cell` also selects `--scenario` and
`--symbol-index`. Subsequent commands validate existing completed stages and
return without recomputation. Fits live under `fits/`, cells under `cells/`,
legacy-compatible assembled results under `arms/`, and the final decision under
`comparison/`. Failed staging under `attempts/` is retained and never counted
as completed evidence.

A fit also retains the legacy fitter's hash-bound `model.zip` as a diagnostic
export. Replay authority is the inference `bundle/policy.zip`; the assembled
arm's `model.zip` is a verified copy of that inference policy. The diagnostic
export is not loaded for replay or treated as a second candidate.

The manual `.github/workflows/ppo-feature-checkpoint.yml` workflow transports
this study between GitHub-hosted runners. Repository tooling at
`tools/ppo_checkpoint_actions.py` verifies the source and checkpoint artifact
identities and archive bytes, stages files without conflicting overwrites,
and invokes the existing checkpoint commands. It owns remote transport and
process limits, not training, accounting, comparison, or research admission.
Published checkpoint roots retain their original protocol; changing hosts does
not permit a source or runtime mismatch. Completed evidence is reused only
through the checkpoint runner's validators.
Fresh replay strategy wrappers preserve the verified policy's fitted feature
names as well as its indices and normalizer, matching the strategy-owned
schema contract used by inference bundle publication.
Transport receipts also bind the actual checkout and workflow revision. A
failed producer job may supply a completed checkpoint, but its failure status
must remain visible; a success conclusion is not an evidence-integrity oracle.

## Dependency direction

`tests/architecture/test_lean_dependency_boundaries.py` が実行可能な正本であり、少なくとも次を禁止する。

```text
_validation -> standard library only
artifacts   -X-> data/risk/simulation/strategies/evaluation/integrations
data        -X-> strategies/evaluation/simulation
integrations -X-> strategies/evaluation
risk        -X-> strategies/evaluation
simulation  -X-> strategies/evaluation
strategies  -X-> evaluation
strategies/rl -X-> strategies/forecasts
evaluation/runs -X-> evaluation/experiments
evaluation/experiments/bootstrap -X-> sealed final-test authorization
trade_rl -X-> tools/agent_repo
```

`evaluation` はlower core packagesを利用してよい。ただしlower layerからbootstrapへ逆依存しない。strategy family間で共有するdataset-bound selectionはroot `strategies/dataset_scope.py` を経由し、RLからforecast内部へ依存させない。依存方向を逆転させる必要が出た場合、循環依存や責務漏れを先に疑う。

### Static import ownership gate

`tools/agent_repo/source_index.py` の `ImportCollector` は、production sourceを実行せず、physical module treeを使って絶対・相対import、親packageからの子module import、選択したsymbolのstatic import re-exportを解決する。module名の区切りまで比較し、似たprefixの別moduleやfacade内の無関係なexportを禁止依存にしない。module scopeの条件分岐は保守的に両方検査し、関数・classのlocal importを公開exportと混同しない。

star importはliteral `__all__`、または明示的なpublic import re-exportを追跡する。動的に組み立てた`__all__`は実行して推測せず検査を失敗させる。循環re-exportも有限に走査する。source-derived mapは一回のscan内だけに保持し、生成catalogをcurrent treeへ保存しない。

`ImportCollector.collect()` はstatic re-exportを追跡してsemantic ownerまで展開する一方、`collect_direct()` はsourceに直接綴られたmodule pathだけを解決し、facade symbolのre-export先までは追わない。semantic dependencyとdirect-import policyは異なるoracleとして使い分け、facade経由の正当な利用をowner moduleの直接依存と誤認しない。

このgateはstatic import ownershipの検査であり、runtime sandboxや任意のPython到達可能性の証明ではない。動的import、実行時のattribute再束縛、反射や関数実行で生じる依存は別のreview/contract testが必要である。

### Repository tooling boundary

`tools/agent_repo/` はAgentのpreflight/context/impact/semantic diff/verification routingを行う**repository-local development tooling**であり、`trade_rl` runtime packageの一部ではない。Git/source/current docsを読む側であり、domain owner/schema registry/public runtime APIにはならない。

- `trade_rl/**` から `tools/agent_repo` への依存は禁止する。
- `tools/agent_repo` はproduction wheelへ入れない。`setuptools.packages.find.include = ["trade_rl*"]` を維持する。
- source-derived index/reportはmemory/stdoutだけに保持し、generated catalog/reportをcurrent treeへcommitしない。
- toolingのstatic analysisはruntime reachabilityの完全証明ではなく、source reviewとfinal full CIを置き換えない。
- local `verify` は開発中の検査選択を支援するが、完了判定ではpermanent CIのfull gateへ収束する。

### Agent Coordination Plane

`tools/agent_repo/coordination/` は複数AgentのTask分離・ownership・dependency・semantic collision・review/verification freshnessを扱うrepository toolingである。既存Control PlaneがRepositoryのsource-derived factを読むのに対し、Coordination Planeはそのfactを使って「誰が何を同時に実行できるか」を管理する。どちらも`trade_rl` runtime/domain authorityではなく、**production wheelへ混入させない**。

Coordinationの耐久契約は次である。

- **Task Packet** はtask id/revision、execution mode、dependency、write scope、capability、Acceptance Criteria/Test Oracle、base SHA、**Resource Key**を持ち、semantic contractをcanonical SHA-256 digestへbindする。
- stateは単一statusではなく `phase + condition`。phaseはplanned→ready→executing→review→verification→integration→completeを基本線とし、conditionはhealthy/blocked/stale/failed/conflictedを独立に表す。
- `read_only` Taskはimmutable snapshotを共有できるがwritable leaseを持たない。`write` Taskはsingle active ownerとisolated writable stateを必要とする。
- write ownershipは **lease epoch** をfencing tokenにし、reassignmentごとにnew epoch branch/worktreeを使う。expired leaseはbranch/PR/CI/artifact side effectをreconcileしてからresume/reassignする。
- Resource Key namespaceは最低限 `file:` / `authority:` / `identity:` / `schema:` / `workflow:` / `artifact:` / `side-effect:` を区別し、file-disjointでも同一identity/schema/side effectを変更するTaskをhard conflictとして表せる。
- review/verification evidenceはTask revision + Task Contract digest + base + exact PR HEADにbindする。HEAD movement後の古いreview/CIをvalidにしない。integration evidenceはcurrent `main`にもbindする。
- durable coordination evidenceはIssue contract/status record、branch/commit、PR/review、CI/artifact、main commitに置き、ready-set/conflict graph/dashboardは再構築可能なderived stateとする。checked-in mutable global task ledgerは置かない。
- `python -m tools.agent_repo task ...` はnetwork-freeなparse/digest/readiness/status/dashboard操作だけを持つ。GitHub claim/comment/mergeのwrite authorityはこのCLIへ埋め込まない。

このlayerは複数Coordinator間のdistributed consensusを保証しない。v1のoperational invariantはactive Coordinatorが1つであることとし、Worker自身の完了主張やlabelだけをmerge authorityにしない。

## Repository integration boundary

GitHub PR / CI / branch-protection・ruleset設定はRepository統合の安全性を管理するが、`trade_rl` runtime packageのauthorityではない。checked-in architecture testはPR/CI policy fileの契約を検証できるが、実際のbranch protection状態はGitHub側のread-backで別途確認する。

Integration invariant: tested PR head contains current `main`. merge直前のcurrent `main` commitがtested PR headのancestorであり、その同一PR HEADにpermanent CI successが存在することを統合証拠とする。`main` が進んだ後の古いPR-head Greenは再利用せず、current `main` を含む新HEADを再検証する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等の証拠としてよい。

このGit tree内のproseやarchitecture testだけでbranch protectionが有効とは判断しない。ruleset/protectionの設定変更後はGitHub stateをread-backし、required check、PR requirement、force-push/deletion、maintainer/admin bypassを確認する。管理surfaceが利用できない場合は未設定/未検証として扱う。

PRに要求される独立研究レビュー（`Generic Independent Research Review` / `Independent Research Review`）は、PR authorとは異なるGitHub principalによるexact HEADレビューを要求する。承認は正式な `APPROVED` review stateと本文の独立レビューmarker、単独の `### Disposition: APPROVED` 見出しを満たす場合だけ認める。`### Disposition: BLOCKED` を含む該当レビューが一つでもあれば承認より優先し、保留・曖昧・古いHEADのレビューはfail-closedに扱う。重大な指摘事項（Medium / High）が解消され全必須CIがGreenであれば、承認済みレビューはIntegratorによるマージ・クローズの対象となる。

`tools/ppo_4h_gemini_review.py` は4h PPO smokeのresult-blind external-AI reviewをdefault-branch trust rootから実行する**repository-local reviewer transport**であり、`trade_rl` runtime packageやeconomic evaluatorの一部ではない。`.github/workflows/ppo-4h-gemini-review.yml` はcanonical PR comment requestを入口に、(1) targetをdataとして読むpacket生成job、(2) Gemini secretを持たずexact reviewed SHAへ固定command setを実行するCore / PPO Runtime / Human Guide verification jobs、(3) canonical packetだけを受け取るfresh secret-bearing review jobを分離する。review jobはtarget checkoutやtarget-generated executable artifactをauthorityとして受け取らず、trusted runner/workflow identity、request/tag identity、packet digest、trusted verification job identity、Gemini request/response identityをreviewer-run attestationへbindする。Gemini API key/model selectionはworkflow configurationであり、concrete model versionはprovenanceとして記録するがproduction trading/runtime schemaやvalidator allow-listへ固定しない。

## Public API policy

Intentionally maintainedなpackage-level importは、内部private file移動より優先して安定させる。

例:

```python
from trade_rl.data import MarketDataset
from trade_rl.strategies import RidgeForecastStrategy
from trade_rl.evaluation import UniversalStrategyComparison
from trade_rl.evaluation.experiments import bootstrap_canonical_m2_study
from trade_rl.simulation import BookState, MarketExecutor
from trade_rl.risk import PreTradeRisk
```

一方、historicalに存在したprivate module pathはpublic contractの証拠ではない。private path移動のためだけのshimは原則追加しない。

## Architecture change rule

Packageを追加・移動・削除するときは同じ変更で次を行う。

1. 責務と依存方向をこの文書へ反映する。
2. `tests/architecture/` でrequired/forbidden pathと依存を契約化する。
3. intentionally publicなpackage facadeをsnapshot/contract testで確認する。
4. 単なるmoveならnon-import AST、serialized bytes/digest、golden behavior等でsemantic driftを可能な範囲で反証する。
5. 旧private pathや一時migration helperをfinal treeに残さない。

## Distribution source closure

構造変更では、working treeだけでなくGit HEADのproduction `.py` roster、sdist、direct wheel、sdistから再buildしたwheelの相対pathとSHA-256が一致することを検証する。`tests/architecture/distribution.py` は未追跡・ignoreされたsource、worktree差分、sourceの欠落・混入・改変、重複member、不正path、symlink sourceを拒否し、archiveを展開・実行しない。PPO normalizationの非Python runtime authorityである `trade_rl/evaluation/ppo_normalization_activation.json` は明示的なpackage-resource closureへ含め、checkout/sdist/wheel間のexact bytesとcanonical schemaを同じgateで検証する。

CIはbuilt wheelをcheckout外の新規venvへ非editable installし、isolated Pythonでpackage identity、public facade import、candidate/bootstrap CLI helpに加えて、installed wheelから実際のnormalization activation resourceを読み、そのSHA-256がcheckout authorityと一致することを確認する。通常のsource closureはPython source中心の配布契約であり、optional trainerの実学習、全platform動作、任意のnon-code resourceすべてを保証するものではない。normalization activation resourceは研究authorityであるためこの一般則への明示的な例外としてclosure対象にする。license/provenanceの恒久保持は別の既存gateも維持する。

## Optional allocation family ownership

`strategies/allocation.py` is a Tier-2 family entry point for `AllocationInputs`,
`AllocationContext`, `AllocationProposal` and `AfterCostTargetAllocator`. It
owns declared horizon/unit contracts and deterministic scalar optimization.
It imports no forecast/RL family, risk, simulation or evaluation owner; it reads
no dataset/account and fits no model. Generic artifact hashing may bind its
immutable input/proposal content.

`evaluation/allocation.py` is a Tier-2 composition entry point for
`propose_nonrl_target`, `execute_nonrl_proposal` and `NonRLExecutionResult`.
It builds account provenance from existing BookState/order owners, applies
canonical risk once and invokes simulation-owned target execution. It must
not use `execute_interval`'s compatibility cache or own cash/P&L updates.
`simulation/targets/execution.py` owns the opt-in quantity-HOLD cancellation
and one-bar execution path, without a dependency on strategy/evaluation.

The existing `trade_rl.strategies` and `trade_rl.evaluation` Tier-1 facades,
three-intent callers, forecast packet meanings and PPO observation/training
remain unchanged. `tests/architecture/test_allocation_boundary.py` guards
ownership, lower-layer direction and absence of a second ledger; mechanism
oracles live in strategy/evaluation tests. The current-close expected-return
contract is not supplied by a mean-log forecast packet. Wiring a separately
versioned expected-simple estimator or an RL consumer requires its own causal,
clock, economic and assurance verification. The separate direct-simple producer
and nonRL consumer below provide this bounded software connection; the remaining
RL and economic research contracts are still required.

## Net-profit declaration ownership

`evaluation/objectives/{contract.py,clock.py,binding.py}` はIssue #810の新規研究向けの事業目的、資本分母、終端純利益算術、regular金融時計と評価期間の一致bindingを所有する。Tier-2 facade `trade_rl.evaluation.objectives` は `CapitalContract`、`ObjectiveContract`、`FinancialClockContract`、`BoundObjectiveClock`、`net_equity_increment` をwrapperなしで公開する。Tier-1 `trade_rl.evaluation` の公開面は拡張しない。

このcapabilityはstandard library、`_validation`、canonical artifact hashingだけへ依存する。strategy、execution、Run/Study lifecycle、外部認証・ネットワークを呼ばず、第二の台帳や研究実行経路を持たない。下位strategyからevaluationへの逆依存を作らない。将来adapterは検証済みの値を下位constructorへ明示的に渡し、現在のPPO training objectiveやhistorical artifactを置換しない。

## Direct-simple producer and consumer ownership

`forecasts/simple_return.py` owns immutable direct labels, the distinct simple
model and marginal fit variance. `_ridge_math.py` owns only numeric weighted
Ridge operations shared with `ridge.py`; it owns no units, selection, schema or
execution. `simple_stream.py` owns separate versioned records/reader;
`simple_prequential.py` selects and fits each prefix once. The forecast facade
directly exports the six simple APIs from their owners. Tier-1 exports and old
log records retain their meanings.

`evaluation/forecast_allocation.py` exposes `HorizonCostEstimates`,
`propose_forecast_target` and `execute_forecast_proposal`. It validates exact
current snapshot/horizon/valuation, then invokes the existing allocation
composition. It owns no fit, ledger, risk formula or order transition. Forecast
owners do not import evaluation, simulation, risk or RL. Architecture tests guard
these directions and reject fit/accounting in admission; synthetic mechanism
tests cover direct labels, serialization, causality and canonical cash/quantity.

`simulation/accounting.BookState` owns the optional paired Dataset/index
processing clock and clone preservation. `simulation/stateful/execution.py`
checks and advances a declared clock after completed bars. The forecast consumer
requires a known matching clock without editing it or duplicating accounting.
Bootstrap values are caller declarations; legacy unmarked accounts and artifact
schemas retain their meanings. Continuation uses the returned book, order book
and next index together.

## Allocation PPO ownership

strategies/allocation_action.py owns detached four-action decisions/proposals.
evaluation/allocation_decision.py composes current forecast admission, selected
RL-feature availability and the shared final-risk/canonical transition.
evaluation/rl_allocation owns actual episode binding, the Gym adapter, explicit
SB3 fitting and sampled-source receipts. It owns no second PnL ledger.

strategies/rl/allocation_policy.py owns pure profile/recipe/observation encoding.
allocation_model and allocation_artifact own inference and write-once verified
policy loading; allocation_*_receipt, allocation_receipt_time,
allocation_recipe_validation and allocation_manifest own strict lower receipt
domains. Lower strategy owners do not import evaluation, fit or call networks.
SB3 fitting/loading is lazy; old PPO/A2C facades and Discrete3 remain unchanged.

MarketExecutor alone owns optional insolvency_valuation and its policy digest.
retain_debt wraps the resolved existing economics identity in a distinct schema;
the default floor_zero digest stays byte-identical. Source receipts distinguish
the declared episode envelope from actual action rows/counts and observed policy
or critic-bootstrap rows. Feature, forecast and cost receipts include bootstrap
inputs; execution receipts bind actual processing bars, profile-selected capacity
reference rows and the effective
calendar/bar duration. Explicit execution fields exclude unused global features.

## Allocation snapshot contract ownership

`strategies/allocation_snapshot.py` はimmutableなaccount/order evidenceのschemaと
structural clock / exact-quantity validation、canonical bytes / digestだけを所有する。
evaluation・simulation・risk・trainerへ依存せず、MarketExecutorやBookStateを再実装しない。
宣言mappingを使う独立テストとarchitecture import contractでこの境界を確認する。
既存strategy facadeとPPO v1のobservation / artifact契約は変更しない。
source factsを読むproducerとcomplete contextの検証は下記の上位observer ownerが担当する。
下位DTOにnumeric encodingやtraining consumerは含まれない。

## Independent account snapshot ownership

`strategies/allocation_snapshot.py` owns the immutable snapshot DTO, its closed
fact schemas, structural consistency and canonical serialization/digest. It
imports no evaluation, simulation, risk or learning code and supplies no numeric
policy encoding. Its source digest is a binding, not independent authenticity.

`evaluation/allocation_snapshot.py` owns source admission and selected-symbol
projection through the existing allocation context, BookState clock, native
order reader and observable tradability API. Canonical margin validation uses a
detached clone; simulation remains the sole ledger/order-transition owner.
Neither observer changes source account/order state or execution randomness.
One-slot vectors retain the global symbol index; full native account/history
identity remains digest-bound. Native completion tolerance stays owned by the
order module and is reused by the observer, not copied into the lower DTO.
The observer also reuses the native absolute minimum fill quantity for nonzero
active progress; it does not substitute the request-scaled completion threshold.
Existing default allocation/PPO behavior is unchanged.

## Allocation observation v2 declaration ownership

`strategies/rl/allocation_observation_v2.py` owns only the frozen typed schema,
ordered layout, detached payload and digest. Its public export is
`AllocationObservationSchema`; no numeric encoder is exported. Its direct
imports contain no NumPy, account/decision reader, evaluation, simulation, risk
or optional trainer. The existing `strategies.rl` package initializer retains
its current imports; this check does not claim transitive runtime independence.
The declaration introduces no schema dispatch or v1 recipe migration. Its peer
encoder consumes the same declaration; source admission and PPO integration
remain above these lower owners.

## Allocation observation v2 encoder ownership

`strategies/rl/allocation_observation_encoder_v2.py` exports only the pure
`encode_allocation_observation_v2` function. It reuses the schema's field order
and consumes AllocationAccountSnapshot/AllocationDecision without mutation.
Its direct imports exclude evaluation, simulation, risk and optional learners.
Runtime admission stays with the observer; canonical accounting/order rules are
not reimplemented here. Exact slot sorting is a representation choice, not
native ID-based execution priority. There is no v1 recipe change or dispatch.

## Allocation observation v2 runtime ownership

evaluation/rl_allocation/env composes native margin initialization, admitted
snapshot and lower encoder only for explicit v2. Its input_receipt recorder
observes actual SB3 callback inputs; training owns lazy collector construction.
strategies/rl/allocation_recipe_v2 and allocation_input_receipt own pure frozen
recipe reconstruction and receipt consistency. Existing manifest/training/model
validators dispatch the two supported versions; generic verified-file save/load
is reused. No lower owner imports runtime or optional learners; no second ledger,
Run/Study registration, default switch or new preprocessing layer is introduced.

## Allocation PPO training protocol declaration ownership

`strategies/rl/allocation_training_protocol.py` exports only
`AllocationPPOTrainingProtocol`, its closed payload/reader and content digest.
Direct dependencies are stdlib and existing stdlib canonical artifact helpers;
there is no numerical, account, risk, evaluation or optional learner import.
Existing package initializers are unchanged; transitive independence is not claimed.
Class identifiers are fixed data, never arbitrary import/constructor requests.
This pure owner adds no consumer; the separate explicit runtime owner is below.

## Allocation PPO protocol runtime ownership

evaluation/rl_allocation/training_protocol resolves the closed declaration into
known optional types and observes actual optimizer calls. training composes its
callback with the existing input recorder only for explicit protocol fits.
strategies/rl/allocation_protocol_receipt reconstructs strict v3 training data
and delegates common source/clock validation to existing receipt owners.
allocation_model reads actual settings/classes/architecture and stdlib installed
version metadata; it imports neither upper runtime nor optional learners.
allocation_manifest dispatches v3 only with recipe_v2; verified-file save/load
and recipe/env/encoder ownership stay unchanged. No new facade, Run/Study,
learner subclass, ledger, normalization or sampler is introduced.

## Allocation causal preprocessing ownership

`evaluation/rl_allocation/preprocessing.py` owns actual prefix row admission and
equal-symbol fitting, preserving the selected-feature dependency-scope guard.
`strategies/rl/allocation_preprocessing.py` owns the frozen closed declaration,
reader/digest and feature-only transform, composing existing numerical statistics.
Neither owner imports account/risk processing or optional learners; the lower
owner does not import the upper fitter. Explicit runtime admission composes them without a new facade,
online normalization or sampler; existing legacy preprocessing remains unchanged.

## Pure allocation observation v3 ownership

`strategies/rl/allocation_observation_v3.py` owns a detached observation payload
and feature-prefix encoder, composing existing v2 economics and frozen statistics.
`strategies/rl/allocation_recipe_v3.py` owns its pure recipe builder; the existing
recipe validator has an explicit v3 reconstruction lane and native root/tag dispatch.
Neither lower owner imports environments, fitting, risk/account execution or optional
learners. Explicit upper consumers compose them without a new public facade,
online fit or sampler.

## Frozen preprocessing runtime ownership

The existing upper preprocessing owner separates actual training-prefix validation
from inference application admission. Env owns source application and zero-terminal
bypass; training owns pre/post prefix envelopes and v4 dispatch. InputRecorder hashes
actual normalized tensors while TrainingSource retains raw meanings. The lower
preprocessing receipt owner checks closed source/declaration links and delegates
shared protocol mechanics, without importing Dataset/evaluation/optional learners.
Manifest/model extend explicit version/width/post-load guards; verified-file
save/load signatures and execution/account owners remain unchanged. No sampler or
Run/Study consumer is introduced.

## Stacked PR verification routing

`.github/workflows/ci.yml` routes PRs targeting `main` or `codex/**` through
Lean Core, PPO Runtime and Human Guide at the PR's exact head SHA. PR opened,
synchronize and reopened events run these jobs; PR edited runs them only when
the payload contains `changes.base`. Body/title edits skip the same software
and review jobs, while submitted/edited/dismissed review events remain eligible.
Metadata-only PR edits use a separate concurrency group and conditional job-name
expressions. Executed eligible jobs retain the existing verification names.
GitHub may display an unevaluated name expression for skipped jobs; such labels
are distinct from the exact verification names and do not prove verification ran.
Non-metadata events retain the existing concurrency group and cancellation policy.
Both generic and 4h review gates require all three software jobs to succeed;
empty, skipped, pending and failed results cannot satisfy that prerequisite.
Review events on an unsupported base still run the appropriate review gate,
which rejects the skipped software jobs instead of reporting a skipped gate.
The existing head-based choice of review checker remains unchanged. Routing
does not grant formal approval or relax the integration requirement that the
tested final head contains current `main`; live exact-head checks are still
required before merge. Push and one-shot research execution routing is unchanged.

## Allocation transition sidecar ownership

Five upper owners contain optional observation only: transition_facts detaches and
checks native economics, transition_arrays fixes tensor bytes, transition_trace
joins actual callback outputs to admitted chronological buffers,
transition_validation closes row/receipt links, and transition_trace_io publishes
verified external sidecar files without model deserialization. Env has one optional
pre-overwrite hook; fit owns attachment, cleanup and successful-final-policy
completion. Lower recipes/bundle/training/input owners remain unchanged. These
owners never execute another ledger, sample a policy, call buffer.get or alter RNG.


## Allocation chronological training schedule ownership

`strategies/rl/allocation_training_schedule.py` owns the immutable result-blind
window roster for Issue #810 allocation training. A window binds role
(`train` / `validation` / `held_out`), whole-Dataset identity as provenance,
symbol, complete episode indices/clocks and a digest of the actually consumed
training-source scope. The causal `window_id` excludes whole-Dataset lineage,
so an unused future suffix cannot rewrite an earlier window identity; the full
window payload and schedule still retain that lineage. Same-symbol clock
overlap is rejected even across different Dataset lineages, so relining or
overlapping crops cannot multiply the declared market interval. The schedule
fixes one causal training order, `cyclic_declared_order_v1`, and states that a
reset means a new independent
account while a PPO rollout boundary continues the current account. It imports
no evaluation, simulation, risk or optional learner backend and does not call a
Dataset, account or policy.

`evaluation/rl_allocation/training_schedule.py` is the upper runtime adapter.
It reconstructs each train-window identity from the existing
`allocation_training_source()` authority, requires exactly the declared train
roster, identical policy recipe / financial clock / spaces across children, and
cycles only after the active canonical `AllocationTradingEnv` reaches true
termination. Manual mid-episode reset and truncation-as-clean-reset fail closed.
The child environment remains the sole owner of risk, execution, accounting and
reward. Runtime usage reports reset and decision counts per declared window and
does not label symbol count, crop count or reset count as independent market
experience.

This lower schedule stage does not itself own learning or artifact publication.
The separate reviewed consumer below now fits PPO through this runtime while
leaving existing single-episode bundle identities unchanged. Run/Study
registration and continuous-account walk-forward remain outside both owners.


### Scheduled PPO fit consumer

`evaluation/rl_allocation/scheduled_training.py` is the first learner consumer
of the chronological schedule. It requires the explicit immutable PPO protocol,
uses the existing schedule Gym wrapper as the single SB3 environment, and leaves
every child `AllocationTradingEnv` as the owner of risk, execution, accounting
and reward. It does not construct a second ledger or source model.

The fit revalidates every declared source before and after learning, requires the
budget to cover every train window and to be an exact rollout multiple, and
records a detached training-only receipt. The receipt binds the schedule and
recipe digests, protocol and optimizer-call evidence, per-window true reset and
decision counts, and exact per-row decision reuse plus policy/critic observation
indices. A final DummyVecEnv post-terminal reset remains visible even when no
subsequent actor decision consumes it; reset count is therefore not described as
an independent-experience count.

Frozen v3 preprocessing is reconstructed once from its declared fit Dataset and
then checked for valid application at each scheduled child clock. This consumer
does not publish an allocation inference bundle, Run/Study artifact, winner
decision or deployment authorization. Existing single-episode bundle schemas and
`fit_allocation_ppo` remain unchanged.


## Allocation continuous-account handoff ownership

`evaluation/rl_allocation/continuation.py` owns the first bounded P5 handoff
capability for contiguous allocation OOS windows. It does not create another
ledger. A horizon-terminal `AllocationTradingEnv` exports a one-shot runtime
continuation containing a detached canonical `BookState`, the immutable
`OrderBookState`, the exact current account-context digest, and the same
`MarketExecutor` instance. Reusing that executor is intentional: stochastic
execution RNG, execution-rule caches and the declared execution policy continue
through the fold boundary instead of silently reseeding.

Resume requires the next window to start at the exact previous processing index,
use the same in-memory `MarketDataset`, account ID, symbol, initial-capital
denominator, execution policy and risk profile. The opening account/order digest
must equal the previous closing digest. Cash, exact quantities, peak equity,
drawdown, accumulated costs/carry, active GTC residuals, terminal order archive
and processing clock are therefore handed over without settlement or
reconstruction. Economic termination cannot be continued, and a continuation is
one-shot so a single execution RNG stream cannot branch into multiple claimed
continuations.

This runtime capability is deliberately not a serialized restart artifact and
does not yet own policy switching, fold construction or walk-forward selection.
The existing `robustness.walk_forward.stitching` continuous mode remains the
aggregation authority; actual allocation handoff digests can now satisfy its
opening/closing state-chain contract.


## Continuous allocation walk-forward runner ownership

`evaluation/rl_allocation/continuous_walk_forward.py` is the first bounded
consumer of the allocation continuation capability. It accepts an explicit
`WalkForwardFold` tuple, one fresh OOS `AllocationTradingEnv` per declared
`fold.test` range, and one caller-pinned policy declaration per fold. Preflight
requires the OOS environment ranges to equal the declared fold test ranges,
requires those test ranges and runtime windows to be contiguous, and validates
one account/Dataset/symbol/capital/execution/risk chain before the first reset.

Only the first fold uses the normal reset path. Every later fold resumes the
previous exact `AllocationAccountContinuation`, so policy switches do not reset
cash, quantities, orders, drawdown or the execution RNG stream. Each policy is
given the current observation and runtime recipe digest, and its returned action
must be one integer in `{0,1,2,3}` before it reaches the canonical allocation
environment.

The runner records actual canonical `execution.interval_net_return` values as
`ReturnKind.DECISION_STEP`, per-fold execution-counter deltas, real
opening/closing account state digests, the declared fold index, and the
caller-pinned policy digest. The resulting folds are passed to the existing
`StitchMode.CONTINUOUS_ACCOUNT` authority. Compounding the stitched simple
returns is required to match the actual continuous account equity path in
software oracles.

This owner does not train, deserialize or select policies and imports no
SB3/Torch learner backend. The policy digest is a caller declaration; artifact
byte verification must happen before this runner is called. `reset_seed` is
only the first Gym reset seed and does not replace the declared execution RNG or
a policy seed. Training/selection leakage, artifact admission and economic
winner selection remain higher-level Study responsibilities.


## Allocation continuous policy artifact admission ownership

`evaluation/rl_allocation/policy_admission.py` is the artifact-verification layer
immediately above the continuous OOS runner. One immutable declaration pins the
fold index, bundle root, expected canonical manifest digest and expected runtime
recipe digest. Admission first validates the complete fold/env/artifact roster,
then calls the existing `load_allocation_policy()` authority for **every** fold
before the first OOS reset. The loader remains responsible for the exact regular
bundle directory, canonical manifest bytes, pinned manifest digest, verified
private policy copy, policy-byte SHA-256 and recipe identity.

After loading, admission rechecks the returned manifest digest and recipe, then
wraps the real keyword-only `AllocationPPOPolicy.action(...,
runtime_recipe_digest=...)` call in the lower runner's backend-neutral action
contract. The lower runner therefore records the same verified manifest digest
as the fold policy identity; a caller cannot substitute an arbitrary digest at
this layer. If any later fold bundle is absent, malformed, tampered or bound to
another recipe, admission fails before any OOS account is initialized.

This layer does not train, select, authorize or rank policies and does not import
the training consumers. It is not the sealed-test access authority. A later
Study consumer must still bind each admitted artifact to result-blind selection
evidence and a one-shot sealed outer-test authorization before economic evidence
can be claimed.


The artifact-admission layer performs the complete continuous-runner preflight
before opening or deserializing any policy bundle. Fold chronology/ranges,
Dataset-instance identity, account/symbol/capital chain, execution/risk
identity, and per-fold recipe compatibility therefore fail closed before
artifact code is loaded. Bundle verification remains a second gate: all pinned
manifests/policy bytes must verify before the first OOS reset.


## Sealed allocation fold-policy authorization ownership

`evaluation/rl_allocation/sealed_policy_admission.py` binds an existing
`SealedTestAccessRecord` to one verified allocation policy artifact before a
continuous outer-OOS run. It does not select a configuration, create access
authority, train a learner or open final/unused data.

For every fold it reconstructs the canonical sealed-access record and requires
the registered experiment-plan digest, Dataset identity, fold index, exact
outer-test range and selected policy digest to match the declared fold,
allocation runtime and immutable artifact. A `selected_policy_digest=None`
record is not executable authorization. Access records must be unique.

After authorization validation, the lower artifact-admission owner performs the
full continuous-chain preflight and verifies/deserializes every pinned policy
bundle before the first OOS account reset. The admitted policy digests must then
still equal the sealed selected-policy identities before the continuous runner
may execute. Selection rules, ledger issuance, Study outcomes and final-test
authorization remain separate authorities.


The sealed allocation consumer also requires ledger membership, not only a
self-consistent access digest. The supplied canonical `SealedTestAccessRecord`
must compare equal to one of the immutable records exposed by the caller-supplied
`SealedTestLedgerProtocol`. This keeps authority content-addressed and permits
an equivalent reconstructed record after persistence, while a record absent from
the supplied ledger is rejected. Canonical digest reconstruction remains a
separate tamper check before that membership check.


Ledger membership authenticates the in-process issuer and the ledger also owns
one-shot runtime consumption. The sealed allocation consumer verifies every
policy artifact first, then atomically consumes the complete fold-access batch
immediately before the first OOS reset. Artifact verification failure therefore
leaves the access batch unspent; once OOS execution is admitted, any later
runtime failure leaves that batch consumed and a replay fails before artifact
deserialization. Partial batch consumption is forbidden.

This consumption ledger is still in-memory runtime authority. It is not a
durable cross-process tombstone, signed final-data permit or repository-global
execution lock. Persistence/restart semantics and final/unused-data permission
remain higher-level Study/outer-test responsibilities.


## Common allocation comparison evidence ownership

`evaluation/allocation_comparison.py` owns the pure Issue #810 P4 comparison
contract for the fixed candidate roster `nonrl / residual_ppo / direct_ppo`.
The contract binds one Dataset, business objective, financial clock, forecast
context, execution economics, risk profile, fold plan, account mode, capital
denominator, scenario roster, RL seed roster and maximum-drawdown guardrail.
It imports no strategy, simulator, learner, Study lifecycle or selection gate.

Evidence is admitted only as a complete matrix. The non-RL candidate has one row
per scenario; each RL candidate has exactly one row per declared seed/scenario.
Within a scenario every candidate/seed must refer to the same OOS source and
opening account state. One RL candidate/seed must retain the same policy digest
across base and all stress/diagnostic scenarios, preventing stress-specific
refits or policy substitution.

The pure summary reports median base after-cost terminal profit, incremental
base profit versus the common non-RL baseline, required-scenario validity,
required risk/execution status and optional diagnostic failures. It intentionally
contains no winner or selected-candidate field. Evidence production, economic
execution, preregistered selection and WINNER/NO_WINNER/INVALID remain separate
higher-level responsibilities.


## Allocation comparison evidence binding ownership

`evaluation/allocation_comparison_evidence.py` is the bounded adapter from an
already-completed continuous allocation OOS run into the pure P4 comparison
matrix. It does not rerun execution or construct a second account ledger.

The adapter projects the business objective and economic clock into comparison
identities that deliberately exclude fold window, scenario economics/risk,
candidate recipe and PPO-only rollout/GAE settings. The fold plan, scenario
context and RL fold-policy schedule have separate content-addressed identities.
Every runtime fold is rechecked against the comparison Dataset, forecast context,
candidate recipe, business objective, economic clock and declared OOS ranges.

A supplied `ContinuousAllocationWalkForwardResult` is not trusted by shape
alone. The adapter rebuilds continuous stitching from the fold results and
requires equality, requires each runtime environment to have completed its fold,
and recomputes the final canonical account-state digest from the actual
`BookState`. The stitched interval net returns are geometrically compounded and
must agree with final canonical equity. Maximum drawdown comes from the final
canonical `BookState.max_drawdown`, because interval-end returns cannot recover
intra-step drawdown exactly.

The produced ledger/execution digests are deterministic bindings over the
continuous runner's state chain, after-cost return series and execution
diagnostics. They are not an independent accounting/execution oracle. The
separate `validity_evidence_digest` must identify that higher-assurance
validation. Winner selection, Study mutation and final-data authorization remain
outside this adapter.


## Allocation cash/no-trade control ownership

`evaluation/allocation_cash_control.py` owns the explicit cash/no-trade reference
for Issue #810 comparison. It is **not** a fourth candidate family and does not
participate in candidate ranking. The control runs the same canonical continuous
allocation account with raw action 0 (`quantity_hold_current_v1`) from an
initially flat book, so it receives the actual declared cash-interest economics
instead of assuming zero return.

The control must remain flat for every fold, retain no active orders and produce
zero trade count, turnover, explicit execution cost, funding and borrow cost.
Its policy identity is seedless and scenario-result independent. The direct-PPO
recipe is used only as the runtime carrier needed by the common allocation
environment; the cash-control policy digest explicitly binds that carrier recipe
and the HOLD-current semantics.

Cash-control evidence reuses the continuous-comparison context validator and the
canonical account metrics from the actual runner. Its ledger/execution summary
digests are evidence bindings, not an independent replay oracle. Candidate
ranking remains limited to non-RL / residual PPO / direct PPO; a later
preregistered selection gate may require an eligible candidate to beat this
cash/no-trade reference before declaring a winner.


## Allocation preregistered selection ownership

`evaluation/allocation_selection.py` owns the pure development selection gate
for the Issue #810 common comparison. It imports the comparison contract only;
it does not execute candidates, load policies, issue sealed access, mutate a
Study or authorize unused/final data.

The frozen rule has two explicit non-negative base-scenario thresholds:
incremental after-cost terminal profit versus the non-RL allocator and versus
the cash/no-trade reference. Thresholds are strict; values within 1e-12 absolute
or relative tolerance are treated as equal and therefore do not pass merely
because of binary floating-point representation. Exact score ties use the fixed
complexity order `nonrl -> residual_ppo -> direct_ppo`.

All three candidate families must have valid/complete required evidence or the
decision is `INVALID`. Required cash-reference evidence must likewise be
valid, complete, unterminated and within the declared drawdown guardrail.
Required risk/execution failure for a candidate is an economic disqualification,
not an invalid Study. Optional diagnostic failures are reported but do not by
themselves remove eligibility.

The non-RL candidate is eligible only if it passes required risk/execution and
strictly beats cash by the declared margin. Residual/direct PPO must additionally
strictly beat the non-RL median base profit by the declared margin. Among
eligible candidates, the highest median base after-cost terminal profit wins.
If no candidate is eligible the pure outcome is `NO_WINNER`. This decision is
development evidence only; higher-level Study freeze/final-test authorization
must bind it separately.
