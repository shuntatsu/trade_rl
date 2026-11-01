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
│   ├── _timestamps.py
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
│       ├── derived_klines.py
│       └── dataset.py
│   └── market_store/{__init__.py,__main__.py,contracts.py,embargo.py,plan.py,checksum.py,parse.py,ingest.py,fetch.py,rules.py,store.py,export.py,cli.py,migrations/*.sql}
├── risk/
│   ├── inputs.py
│   ├── portfolio.py
│   ├── pretrade.py
│   └── emergency.py
├── simulation/
│   ├── accounting.py
│   ├── objectives/{__init__.py,contract.py,clock.py,binding.py}
│   ├── execution.py
│   ├── native_transition.py
│   ├── bar_path.py
│   ├── liquidity.py
│   ├── quantities.py
│   ├── depth.py
│   ├── orders/{model.py,admission.py,reconciliation.py}
│   ├── stateful/{runtime.py,execution.py,bar_lifecycle.py,order_transitions.py,symbol_fills.py,portfolio_session.py}
│   ├── targets/execution.py
│   └── diagnostics/{execution_stress.py,funding.py,runtime_performance.py,runtime_performance_io.py}
├── strategies/
│   ├── dataset_scope.py
│   ├── position_duration.py
│   ├── interface.py
│   ├── position_intent.py
│   ├── portfolio.py
│   ├── portfolio_observation.py
│   ├── portfolio_adapter.py
│   ├── portfolio_policy_codec.py
│   ├── decision_value.py
│   ├── decision_value_shallow.py
│   ├── controls.py
│   ├── carry.py
│   ├── rules/{trend.py,mean_reversion.py,channel_breakout.py}
│   ├── forecasts/{controller.py,supervised.py,ridge.py,ridge_artifact.py,lightgbm.py,prequential.py}
│   ├── allocation/{solver.py,cost_aware.py,buy_and_hold.py,costs.py}
│   └── rl/{intent.py,ppo.py,portfolio_env.py,portfolio_transform.py,portfolio_runtime_scenario.py,portfolio_scenario_runtime.py,portfolio_scenario_inference.py,portfolio_training.py,portfolio_inference_bundle.py,portfolio_episode_journal.py,portfolio_policy_sample_journal.py,a2c.py,ppo_normalization.py,ppo_artifact.py,a2c_artifact.py}
└── evaluation/
    ├── replay.py
    ├── paired_portfolio_action.py
    ├── paired_portfolio_label.py
    ├── decision_value_fit.py
    ├── decision_value_plan.py
    ├── decision_value_dev_study.py
    ├── portfolio_inference.py
    ├── bot.py
    ├── metrics.py
    ├── evidence.py
    ├── series.py
    ├── directional.py
    ├── carry.py
    ├── directional_candidates.py
    ├── directional_selection.py
    ├── directional_study.py
    ├── ppo_risk_study.py
    ├── dev_realdata_study/{__init__.py,__main__.py,protocol.py,metrics.py,runner.py,cli.py,dataset_build.py,trials.py,trial_sources.py,trial_study.py,trial_cli.py,trial_labels.py,phase3_registry.py,phase3_sources.py,phase3_study.py,phase3_cli.py}
    ├── forward_study/{__init__.py,__main__.py,registry.py,labels.py,seal.py,fs1.py,cli.py}
    ├── forward_acquisition/{__init__.py,__main__.py,plan.py,acquire.py,seal_record.py,cli.py}
    ├── ppo_feature_study.py
    ├── ppo_feature_checkpoint.py
    ├── ppo_normalization_replication.py
    ├── ppo_normalization_execution.py
    ├── ppo_normalization_activation.json
    ├── rl_family_comparison/{__init__.py,contract.py,cells.py,comparison.py}
    ├── paper/{__init__.py,store.py,account.py,engine.py,control.py,failure.py,lock.py,supervisor.py,operations.py,watchdog.py,attempt2.py,storage_probe.py,assessment.py,cli.py}
    ├── gates/{models.py,resolve.py}
    ├── comparison/{bootstrap.py,paired.py,seed_robustness.py,strategies.py}
    ├── robustness/
    │   ├── capacity.py
    │   ├── closed_trades.py
    │   ├── fold_metrics.py
    │   ├── perfect_information/{bound.py,solver.py}
    │   └── walk_forward/{capabilities.py,folds.py,sealed_test.py,stitching.py}
    ├── runs/{candidate.py,candidate_suite.py,config.py,execute.py,provenance.py,artifact.py,portfolio_family_comparison.py,portfolio_family_scenario_replay_v2.py,portfolio_family_scenario_attempt_v2.py,portfolio_family_scenario_result_artifact_v2.py,portfolio_family_attempt.py,portfolio_family_result_artifact.py,synthetic_value_attempt.py,synthetic_value_costaware_attempt.py}
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

`evaluation/residual_rl/` is the synthetic-only residual-PPO study over the cost-aware allocator (Issue #810). `market.py` generates deterministic multi-asset markets with an explicitly planted alpha (or none); `env.py` is a Gymnasium environment that reuses the real `strategies.allocation.cost_aware.CostAwareAllocationPolicy` read-only and owns a separate value ledger, a discrete trade-throttle residual and the explicit fee/spread charge; `protocol.py` freezes every outcome-relevant constant behind a source-bound digest; `evaluate.py` holds rollouts, metrics and the pre-declared decision rules; `train.py` imports stable-baselines3 and torch lazily (optional `train-sb3` extra); `study.py` and `cli.py` run the study and append every execution to a provenance JSON. It reads no Dataset, artifact store, network or live source, depends only on `strategies.allocation`, `strategies.portfolio`, `artifacts` and NumPy/Gymnasium at import time, is not re-exported from `trade_rl.evaluation`, and its labels (`SYNTHETIC_PASS` / `SYNTHETIC_FAIL`) are synthetic evidence only, never a profitability claim and never a clearance of the real-data fee-only learnability gate.

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
Fresh snapshot capture supplies a 64 KiB per-response body limit, while each of
the two exchange-info responses uses a 32 MiB limit. These operational budgets
apply only to those prospective network captures; exceeding one fails capture
without treating a truncated prefix as raw evidence. Transport callers that do
not supply a limit, including historical REST and Vision/cache reads, retain
their existing behavior. These body-byte limits are not endpoint size, peak
memory or total-runtime guarantees.
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
validating the expected protocol digest. `events()` retains its list contract;
`iter_events()` validates the complete chain before yielding from one read-only
SQLite snapshot. The engine replays that stream and keeps only each known key's
trusted sequence and digest, re-reading and validating the original row for an
idempotent retry. This is an evidence store, not another financial ledger.
`paper/account.py` composes the existing FundingCarryBot,
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
run, status, check, supervise and write-once evaluation commands. The account/engine
expose due announced funding for previously held exact quantities, including after
exits. `paper/failure.py` is the single exclusive-create failure-marker writer and
`paper/lock.py` the single collector lock plus a non-destructive probe; both are
import-light so the watchdog never loads the Binance stack. `paper/watchdog.py`
owns only operational liveness: the atomic collector
heartbeat, exit/signal/console failure hooks, the stateless stale check and the
child supervisor. It never opens the journal, control writer or market surfaces; it
reads the control database read-only through `control.incomplete_cycle_exists`
and creates failure markers through the one `failure.write_collection_failure`
writer (re-exported by `supervisor`). A held collector lock is contention, never a
failure. `paper/attempt2.py` composes the
existing lineage validator, `screen_plan`, `seal_paper_study`, FS1's authorization
pattern and the supervisor into plan/seal/launch gates for the successor attempt
with no network call (it loads the collector stack at import through
`seal_paper_study`, so the offline property is proved at call level). It imports
neither the collector nor `paper/storage_probe.py`. The probe is the one other
networked step: it runs the real `PaperCollector` for a few slots into a scratch root
whose research plan names no attempt lineage, so `attempt2 launch`, `cli evaluate`,
`cli run` and `cli supervise` refuse it (`attempt2 seal` only writes new roots).
Lineage v3 is sealable only through `attempt2`; `seal_paper_study` and `cli seal`
refuse it otherwise. These operations contain no production routing.

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
`portfolio_transform.py` owns an opt-in versioned Dict transform and Gym
observation wrapper for `SharedPortfolioGymEnv`. It binds the raw and
transformed layouts, Dataset/objective/clock/risk/execution context, and fit
scope. Local features reuse `PPOFeatureNormalizer`; global features and
per-symbol geometric mark-price references fit only on actor rows
`[start_index, agent_stop_index)`. Availability masks remain visible, account
and order notionals use fixed initial capital, and UTC timestamps are
subtracted as integers before conversion to floating point. The wrapper keeps
native actions, rewards, termination, and info unchanged; terminal observations
use fixed absorbing padding. If transforming a result fails after the wrapped
environment may have advanced, it records the failure and refuses another step
until reset. This preprocessing does not create free/reserved cash, train a
policy, refit on evaluation rows, or change legacy PPO defaults.
The same owner provides a distinct `PortfolioScenarioObservationRuntimeBindingV2`
and v2 transform entry point for explicitly admitted execution scenarios. The
v2 binding keeps the original fit stop and training execution-config digest
separate from the runtime actor stop and execution-policy digest; the existing
v1 transform payload and accepted binding type remain unchanged. The runtime
scenario wrapper rechecks its typed binding against the actual forward
environment before reset and each native step, and fails closed after a
post-step transform error.
`strategies/rl/portfolio_runtime_scenario.py` owns only the immutable, versioned
`PortfolioRuntimeExecutionScenarioV2` declaration. It binds caller-declared plan,
training-context, transform, and checkpoint SHA-256 anchors to exact typed base
and runtime `ExecutionCostConfig` policies and their canonical policy digests.
Version 2 permits changes only to fee, spread, impact, slippage, cost multiplier,
and order latency fields; borrow and capacity remain invariant. The declared
changed-field roster must exactly match the canonical per-field diff. Byte
verification compares against a separately retained typed declaration and an
external expected digest, without reconstructing untrusted JSON. The owner does
not load a model, validate Dataset/risk/profile/roster/capital/clock context,
run a scenario, or establish calibration, provenance, gate approval, or economic
performance. It is not re-exported by a package facade.
`strategies/rl/portfolio_scenario_runtime.py` owns the v2 runtime binding
construction, direct-owner source digest, and validating Gym observation
wrapper. It compares the base policy with the verified training context and
fitted transform, the runtime policy with the actual environment, and both
execution identities with the declared scenario. Its selected source digest
combines the existing selected source/runtime identity with the exact v2 owner
files; neither manifest claims complete transitive or in-memory code closure.
The binding reconstructs execution-policy and runtime/window checks from typed
inputs, but its preserved training-bound digest does not independently prove
that a training fit occurred or establish its provenance.
`strategies/rl/portfolio_scenario_inference.py` owns the explicit v2 bundle
loader and deterministic policy adapter. It reuses the strict bundle-byte,
training-context, source/runtime, checkpoint-copy, and loaded-policy checks,
then validates scenario and runtime context before checkpoint deserialization.
It shares the existing raw snapshot codec and native action proposal mapping.
Neither RL module changes the v1 bundle schema or loader signature, is exported
by a facade, wires the four-family comparator, or creates financial evidence.
`evaluation/runs/portfolio_family_scenario_replay_v2.py` is the separate
explicit evaluation owner for that composition. It checks the scenario and
current scenario-inference source pins before loading evaluation artifacts,
passes the declared runtime execution cost to the existing private four-family
preflight/native loop, and returns those four native outcomes together with the
actual scenario inference binding. The v2-only private boundary check repeats
the scenario/source pin and loaded policy environment validation before each
arm and completion observer; v1 leaves that hook unset. A failed check stops
before the next arm or observer. `portfolio_family_comparison.py` retains its
strict v1 signature and runtime-cost rejection while sharing that private loop.
The v2 replay entry is not re-exported by `evaluation.runs`. The separate
`portfolio_family_scenario_result_artifact_v2.py` owner validates the actual
scenario binding and native rows, and `portfolio_family_scenario_attempt_v2.py`
provides its one-shot local wrapper and strict attempt loader. The v2 artifact
has its own external identity and expected result digest, records before/after
provenance plus scenario-source pins, and preserves completed rows on failure;
it does not alter or bridge the v1 attempt/result schemas. These owners select
no scenario recipe and supply no economic evidence.
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

汎用のcanonical encoding、digest、atomic publication primitive、verified fileを持つ。`atomic_rename_directory`はstaging directoryの同一filesystem **no-replace** rename（Linux `renameat2(RENAME_NOREPLACE)`、macOS `renamex_np(RENAME_EXCL)`、Windows既存先拒否）を使い、Windowsの一時的なpermission errorだけをsource/target再確認付きで有界retryする。未対応環境はfail-closedとし、通常POSIX renameやcopy fallbackでatomic性と排他性を弱めない。market data、strategy、evaluation等のupper layerを知らない。

### `data`

`MarketDataset`、point-in-time contract/source、identity、bounded view、artifact codec/publication、dataset build、causal feature computationを持つ。strategy/evaluation/simulationへ依存しない。private `data/_timestamps.py` はtyped `datetime64` 値のexact integer normalizationを共有するinternal ownerであり、public facadeへexportしない。MarketDataset、RawMarketSeries、forecast cutoff/label validation、candidate typed configがこのhelperを使う。`data/features/numerics.py` はcanonical Dataset identityへ入るfeature計算のportable scalar/reduction semanticsを所有し、`core.py`、`cross_asset.py`、`builder.py` が共有する。モデル学習、simulation P&L、汎用evaluationの数値計算まではこのauthorityへ含めない。現行 `MarketBuildConfig` は `market_build_v3` / `portable_feature_numerics_v1` をbuild identityへbindし、historical `market_build_v2` artifactのreader互換はartifact contractとして維持する。`data/build/economics.py` は build-level `ExecutionEconomicsProfile` の単一ownerであり、`MarketBuildConfig` のfeature/build semanticsとは分離する。profile省略時はlegacy economic behaviorを維持し、明示profileは既存economic-semantics経路を通してimmutable Dataset fields/content identityへbindする。

`data/artifacts/codec.py` owns the verified Dataset reconstruction and manifest digest from one manifest read and its corresponding array bytes. `data/artifacts/publication.py` exposes `load_published_market_dataset_artifact` as an identity-first `(PublishedDatasetArtifact, MarketDataset)` pair, and `inspect_published_market_dataset_artifact` delegates to that owner. The existing `load_market_dataset_artifact` remains Dataset-only. The pair is bound to one verification pass; loading does not lock the directory or promise an OS-level atomic filesystem snapshot.

### `integrations`

外部venue/providerを内部data contractへ変換するadapter層。Binanceはtransport、cache、Vision archive、metadata、dataset assemblyを分離する。strategy/evaluationを知らない。`BinancePublicTransport`の既定値はnetwork-enabledの既存互換を維持し、bootstrapだけがsource freeze後に`allow_network=False`を明示してcache-only化する。

### `risk`

portfolio/pretrade/emergencyのhard safety・feasibilityを持つ。`PortfolioRiskModel` はcovariance投影が有限であることを要求し、実際の二次形式の項の絶対値合計から導いた丸め誤差限界を超える負の分散を拒否する。strategyのalpha判断やevaluationを所有しない。

### `simulation`

execution/accountingの経済正本と、order/stateful/target/diagnosticsを持つ。strategy/evaluationから独立することで、同じexecution semanticsを複数研究候補で共有できる。

`objectives/` は新規Study向けのpureな `CapitalContract`、`ObjectiveContract`、
`FinancialClockContract`、`BoundObjectiveClock` と固定資本の
`net_equity_increment` を所有する。値を受け取ってobjective identityを宣言するだけで、
ledger、replay、selection、study executionを実装しない。入力のequityは既存の
after-cost accounting ownerが供給する。objective subpackageはsimulation root facadeへ
再exportせず、validationとartifact identity hashing、および自身の契約型だけを使う。

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
`native_transition.py` owns the reusable numeric-target → explicit
`PreTradeRisk` → one-native-bar `MarketExecutor` transition. It returns the
existing risk target, execution result, and effective exact-quantity hold indices;
it creates no second book or ledger and imports neither strategy nor evaluation
code. Shared-cash replay and PPO paths that already apply configured pre-trade
risk use this owner. Those callers retain action mapping, cached target rebinding
against the pre-execution book, split adjustment, and observer/evidence capture.
The legacy PPO no-risk shortcut remains on its existing direct executor path,
and this seam does not make per-symbol PPO a joint portfolio learner.
`stateful/portfolio_session.py` owns an opt-in numeric shared-portfolio session:
it retains the returned book, executor, risk controller, and native index across
one-bar calls, while exposing only a detached book view. It delegates every step
to `native_transition.py`, reserves latency-aware terminal intervals for its
internal emergency flatten, and becomes unusable after a transition/observer
failure that may have advanced execution state. It imports no strategies or
evaluation code; the batch portfolio replay remains evaluation-owned and keeps
decision and ledger evidence collection there.
The shared-cash replay's explicit exact-quantity HOLD is carried through
`targets/execution.py` to reconciliation, which accepts it only when its target
weight exactly matches the current book weight; it cancels residual orders without
resizing the native quantity. Ordinary weight targets use the existing path.

### `strategies`

The PPO environment owns rebasing its cached quantity proposal after a processed
split; `evaluation/replay.py` owns the same transition for its single-symbol and
shared-cash callers. `simulation/targets/execution.py` supplies book mark prices
to `orders/reconciliation.py` for weight sizing while keeping trading close as
the order reference. With no held quantities, it resolves current dataset marks
before entry sizing so default initial book marks are harmless. The accounting
and execution owners remain unchanged.

small strategy interfaceとlogical intent、controls、rule、forecast、teacher-free RLを持ち、evaluationを知らない。`strategies/portfolio.py` はnumeric full-vector policy proposalと、tuple-backedな現在行・account・active-order・risk snapshotのdata-only contractを持ち、evaluation/simulation/riskのruntime objectをimportしない。`strategies/portfolio_observation.py` はsource-owned `MarketDataset`、現在の`BookState`、active-order tuple、risk config、decision windowからこのsnapshotを組み立てる。`strategies/portfolio_adapter.py` は固定roster・明示budgetでexact built-in `ConstantIntentStrategy`、`TrendIntentStrategy`、`RidgeForecastStrategy` をnumeric `PortfolioPolicy` proposalへ写すopt-in adapterと、明示cash zero-target policyを持つ。filled sideはsnapshotのexact quantityから導き、risk・pending-order reconciliation・fill・cost・accountingは共有portfolio session / native ownerに残す。legacy cached-target、minimum-hold、pending-aware intent、exact-quantity HOLD、任意strategy parityは提供しない。adapterはruntime/evaluationから独立し、strategy facadeへも再exportしない。`strategies/portfolio_policy_codec.py` はこのDTOとNumPy配列のraw-v1 Dict projection、strict joint action-code decoder、target proposal mappingを共有する純粋なstrategy-layer ownerであり、Gym/SB3/evaluation/simulation/riskをimportしない。`PortfolioInferencePolicy.decide_portfolio` はloaded bundleのsymbol/feature roster、decision window、native timestampを検証して同じcodecを使う。evaluation replayは既存のaccountingや実行経路を変えず、このbuilderを呼び出す。pending-order snapshotはintentのlimit・stop・submission-reference price factsもコピーする。3値すべてNoneのlegacy snapshotはconstructor互換だが、価格contextとしては不完全である。`strategies/allocation/` はcaller供給のforecast sourceに対する数値`PortfolioPolicy` seamを持つ。`solver.py` は`mu.w - gamma/2 w'Sigma w - sum c_i|w_i-w0_i|`をbox・gross制約の下でcyclic coordinate descent・active-set Newton加速・gross Lagrange乗数のbisectionで解く、NumPyだけに依存する純粋なsolverである（`trade_rl`内の他packageはimportしない）。解は許容誤差内の精度で、返す前にKKT最適性certificate（subgradient条件・box normal cone・gross相補性）を検証する。非収束・非有限・制約違反・certificate不合格は例外にして呼び出し側をfail-closedにする。`cost_aware.py` の`CostAwareAllocationPolicy`は固定decision schedule（既定は毎日00:00 UTC）でのみretargetし、decision時刻以前のmark価格だけからEWMA共分散を作り、tiny tradeはexact-quantity HOLDにし、solver/forecast/履歴不足の失敗時は全symbolをHOLDして注文を出さない。`buy_and_hold.py` は初回に等weightを取り以後exact quantityを保持する対照policyである。これらはaccountもP&L ledgerも持たず、risk・fill・cost・accountingは共有portfolio sessionに残る。`costs.py` だけが`simulation.execution.ExecutionCostConfig`を読んで一方向cost（impact・slippage・per-row overrideを含まないlower bound）を導出し、他のallocation moduleは`strategies.portfolio`以外のtrade_rl packageをimportしない。`trade_rl.strategies`のfacadeへ再exportしない。`dataset_scope.py` はdatasetに束縛されたfeature/symbol selection validationの単一ownerであり、forecastとRLのsibling familyが互いの内部実装へ依存せず共有する。`forecasts/supervised.py` はRidge / LightGBMが共有するcausal training rowsとfit-scope validationを所有し、両fit pathはestimator / solverより前にこれを通る。`fit_cutoff` / `label_end_times` の元unit、正のmultiplier、範囲、sub-nanosecond精度のexact integer validationは `data/_timestamps.py` が単一ownerとして提供する。fixed-duration unitとcalendar `Y` / `M`は値を正確にnsへ表せる場合だけ受け入れ、cast後の逆変換をoverflow判定の根拠にしない。Ridge / LightGBMのcausal rowsと`fit_cutoff`にはlosslessな`datetime64[ns]` validationを適用する。`position_duration.py` は実際のsigned quantityから保有episode ageを導き、minimum-hold中のintent制約を共通定義する。model自身やcandidate config自身の不変条件validationは各ownerに残す。

`strategies/rl/portfolio_env.py` はopt-in `SharedPortfolioGymEnv` adapterを提供する。固定shapeのjoint action/observationを、strategy-owned snapshot builderとsimulation-owned `SharedPortfolioSession`へ接続するが、`evaluation`には依存しない。shared risk・execution・accounting、native clock、reserved terminal settlementは既存ownerが引き続き処理する。これはGym adapterであり、それ自体はtrainerや学習済みjoint policyを提供せず、legacy PPO defaultも変更しない。RL facadeからは再exportしない。

The codec exports `ACCOUNT_FIELDS` for the four columns of the raw-v1 account
matrix. Its 21-column active-order matrix has one row per roster symbol, rejects
duplicate or out-of-roster orders, and represents each optional price with a
presence flag plus a zero-filled absent value. An all-`None` legacy price tuple
therefore remains representable, while consumers can still distinguish missing
price context. An exact-quantity `ACTION_HOLD` is accepted only if hard risk
leaves the current weight unchanged; an accepted HOLD preserves the current
signed filled quantity. Native reconciliation cancels any active remainder
before processing the next execution interval's fills. The action set has no
independent cancel action, and this wrapper does not add one.

`strategies/rl/portfolio_episode_journal.py` は明示的に包んだGym環境の外側で使うopt-in journalである。新しいrootを一度だけ確保し、source action/reward/native interval・terminal factsをeventごとのcanonical JSONへ記録してからtransitionを返す。observation、policy state、log-probabilityは記録せず、caller-declared context digestも独立検証済みとは扱わない。失敗・中断したepisodeは完了扱いにせず、状態が進んだ可能性のあるjournal error後はwrapperをpoisonしてstep/resetを続けない。trainer、rollout collector、episodeの学習再現性は提供しない。

The journal may wrap an observation-only Gym wrapper, but validates that the underlying environment is `SharedPortfolioGymEnv` and forwards the action object unchanged. Its manifest includes configured target weights; each step records the source-requested target weights and effective HOLD symbols. Those are target/hold facts, not fill quantities.

`strategies/rl/portfolio_policy_sample_journal.py` is a separate opt-in SB3 forward-hook recorder. It writes the actual policy forward-input tensors, sampled discrete action codes, joint log-probability, and critic value before SB3 advances the environment. Sample schema v2 adds the forward batch size and actual action/value/log-probability tensor dtypes and full batch shapes while retaining the existing recorded values. Attachment requires no pre-existing policy-local or global forward hooks and freezes both hook layouts and their dispatch flags; a later layout change poisons the journal before any sample is written. Constructor failure after hook registration removes only its owned hook and releases its policy reservation. The journal's context, transform, and policy-version strings are caller-declared labels, not Dataset or weight/checkpoint attestations. It does not capture optimizer updates, establish actor-to-native-transition joins, or change policy serialization; Torch and SB3 remain lazily imported only when the recorder is constructed.

`strategies/rl/portfolio_training.py` は明示的に呼び出す `fit_shared_portfolio_ppo` とその設定・結果型を所有する。これは1環境・CPUのopt-in PPO trainerで、actorのpolicy sampleをnative episode eventへ結び、rollout単位のcheckpoint・optimizer step呼出数・episode/join evidenceをwrite-once rootへ保存する。root予約前にrollout sizeとGAE lambdaが環境のbound financial clockに一致することを検証する。失敗時は利用可能なpartial evidenceを残し、完了扱いしない。failure receiptはhook登録後に観測したoptimizer post-step hook呼出数とSB3の報告するepoch counterを保持し、未初期化の場合はnullを記録する。target-KLはepoch途中で停止し得るため、このcounterは部分epochを含み得てoptimizer step数とは異なる。trainerは学習済みinference bundleやdeployment loaderを作らず、従来のper-symbol PPO入口・defaultにも接続しない。実行経路やソフトウェアテストは利益、候補選択、研究gateの通過を示さない。Torch/SB3はこのopt-in関数の呼出時にだけimportする。

`strategies/rl/portfolio_inference_bundle.py` はtrainerが残したexact completed attemptから、別schemaのopt-in inference bundleをpublish/loadする。publisherはtraining manifest・completion receipt・context・checkpoint bytesを照合し、loaderはtrustedな外部digestとforward runtime bindingをSB3 policyのload前に検証する。verified checkpointのdeserialization後、実際のobservation/action spacesに加え、宣言されたMultiInput policy class・MLP widths・activation・default features extractorも照合してから推論facadeを返す。これはinference architectureの確認であり、optimizer stateや全training hyperparameterをcheckpointから再構成しない。source/runtime manifestは明示したselected owner群の観測であり、この経路ではDataset module `data.market` と共有timestamp normalizer `data._timestamps` を含むが、完全なtransitive source closureや過去fitの独立証明ではない。bundleはlegacy PPO artifact/API/defaultへ接続せず、推定性能・research gate・deployment readinessを示さない。Torch/SB3はpublisherまたはloaderの実行時だけimportする。
このownerはv1のstrict loaderとscenario-v2 loaderが共有するprivate verified-input / checkpoint routinesも持つ。v1は全financial contextの一致を維持し、v2は別ownerでscenario-specific validationを済ませてから同じverified checkpoint loadへ進む。

`forecasts/controller.py` exposes `CostAwareForecastIntentController.decide_for_position` for direct calls and the cost-aware Ridge / LightGBM adapter path. Adapters call it only when `one_way_switch_cost` is configured and the observation includes `intent_gross_budget`; other observations keep the intent-only controller call. The controller stays independent of execution, risk and replay. Canonical replay supplies filled quantity/weight and an optional matching cached target, then applies its existing minimum-hold, hard-risk, execution and accounting stages.

`forecasts/ridge_artifact.py` owns write-once publication and verified loading for the existing immutable `RidgeForecastModel`. Its single canonical JSON payload binds declared dataset, fit scope, cutoff, model, source, and runtime metadata; loading requires both the external digest and caller-supplied typed binding. It uses the shared `trade_rl.artifacts` canonical encoding and atomic directory publication primitives. It does not fit a model or claim that declared metadata independently proves how a historical fit was produced. LightGBM persistence and evaluation remain outside this module.

`StrategyObservation.gross_position_return` と `current_position_quantity` はoptionalなexecution-derived inputである。`evaluation/replay.py` はexecutionのfill `OrderEvent.execution_price`と現行book markからsigned mark-to-average-fill returnを計算し、現時点の実約定quantityとともに `RegimeAdaptiveStrategy` へ渡す。strategy packageはfill ledgerやreplayへ依存せず、adaptive exit requestのlatchを公開する。`current_intent` は直近のeffective targetであり、未約定・部分約定後の実保有側とは異なることがあるため、adaptive latchはsigned filled quantityで管理し、数量が0になるまで維持する。replayはそのlatchがあるFLAT intentに限りminimum-hold constraintをbypassする。gross returnはentry後fee、funding、borrowを含まない。exit fillはtrigger後のeligible execution stepに発生し、gapやliquidityを含む経済保証ではない。

`StrategyObservation.intent_gross_budget` はcanonical single-symbol/shared-cash replayが渡すoptionalなper-symbol target budgetで、direct caller向けのdefaultは`None`である。`same_intent_target_weight`はbudgetがある時のoptional hintで、cached desired quantityを現在のmark/equityでweightへ換算する。cost-aware forecast adapterはbudgetがある時だけsigned `current_position_quantity`（`None`なら`current_weight`）の符号からactual sideを取り、candidateを重複なしの`[actual-side, actual-sideからのbaseline hysteresis proposal, FLAT]`で比較する。candidateのtarget exposure `a`は通常`intent * budget`だが、candidateがlogical `current_intent`に一致しhintがある場合はcached weightを使う。utilityは`U(a) = a*expm1(f) - c*abs(a - w)`で、`w`はactual weightであり、partial fillやpending residualをturnoverとして評価する。strict improvement時だけ置換するのでtieは先のcandidateを保つ。minimum-hold中とunlock decisionではhintを渡さず、既存constraintがquantity holdとnominal target resetを制御する。stable pairwise比較はactual/cached weightsをclipせず、near-zero `1 + c*k` gapだけexact binary-rational fallbackで保持する。任意の部分数量へ連続最適化せず、選択後のhard riskとexecutionを置き換えない。

### `evaluation`

lower layerを利用してReplay・metrics・gate・comparison・robustness・concrete runを構成する。

`evaluation/comparison/strategies.py` owns named independent-per-symbol comparisons and shared-cash factory comparisons. Shared-cash comparison gives each strategy arm one replay over the full symbol roster, reusing `evaluation/replay.py` and `evaluate_performance`; it does not fit models or select a winner.

`evaluation/replay.py` also owns `run_scripted_shared_cash_replay`, an opt-in
per-symbol action schedule that reuses the same shared risk, `MarketExecutor`, and
ledger path. Its no-decision marker preserves pending targets; exact-quantity
HOLD is accepted only when hard risk leaves the current book weight unchanged.

`run_shared_portfolio_replay` is a separate opt-in caller of that same replay
loop for a simultaneous numeric target vector. It builds its immutable actor
input through the existing current-row `StrategyObservation` owner, copies
account/order/risk facts into primitive values, and sends each complete vector
through the same native risk/execution transition. Numeric portfolio decisions
use their own result and ledger schema; the per-symbol intent API and its schemas
remain unchanged. It does not introduce another book, executor, or P&L loop.

`evaluation/portfolio_inference.py` is the opt-in filesystem consumer for a
published policy bundle. It loads a caller-bound paired Dataset artifact once,
uses an unreset Gym environment only to validate the bundle against that same
Dataset and runtime context, then passes that exact Dataset and unchanged
execution/risk configuration to `run_shared_portfolio_replay` once. The replay
creates the sole financial session and owns terminal settlement. The helper
does not introduce another ledger or change the legacy intent result schemas;
Gym/SB3 imports are local to the callable path.

`evaluation/runs/` の責務は一回の計算とimmutable Run evidenceである。

- `candidate_suite.py`: 5 candidates + 3 controlsのfit/replay構成。Ridge / LightGBMはRun configの正の整数`supervised_forecast_horizon_hours`（既定24、signed `datetime64[ns]`に収まる最大2,562,047時間）を両fitへ渡し、strategy名にも同じ時間数を使う。直接構築する`LeanCandidateConfig`は4つのthresholdでbuilt-in `bool`を拒否し、typed `fit_cutoff`もshared data helperでexact integer normalizationを行う。PPO・risk・execution設定はforecast horizonに依存しない。
- `config.py`: Run JSONの単一parse/resolution authority。候補Run configのforecast horizonをstrictに検証し、`fit_cutoff` / `evaluation_start` / `evaluation_stop_exclusive` の元ISO範囲・precisionもlosslessにns正規化してdataset-resolved specへbindする。typed `datetime64` はshared data helperでexact integer normalizationを行う。既にwrapされた `datetime64[ns]` から元時刻は復元できない。
- `execute.py`: resolved specから既存candidate suiteを一度実行するin-memory seam。
- `provenance.py`: implementation/runtime/research-context provenance生成。
- `artifact.py`: summary/raw returns/provenanceのpublication、verified load、semantic identity。`supports_shared_cash_result_schema`は、shared-cash evidenceを持つresult schemaとしてv7/v9だけを許可する。新規Runは正のforecast horizonを`lean_candidate_result_v8`（per-symbol）または`lean_candidate_result_v9`（shared-cash）へbindする。v9はv7の追加portfolio return seriesとsettlement / ledger evidenceを維持し、loaderがreturn / maximum drawdownをraw seriesから再計算する。旧result schemasは不変のまま読み込む。
- `candidate.py`: 上記を順番に呼ぶ薄いfilesystem CLI/facade。
- `ridge_artifact_comparison.py`: verified Datasetとcaller-bound frozen Ridge artifactで4つの固定shared-cash armを比較するfilesystem入口。既存paired Dataset loader、Ridge artifact loader、shared-cash comparison、replay/ledgerの所有境界を再利用し、fit、winner選択、result publicationは行わない。
- `ridge_comparison_artifact.py`: 上記比較の完全な4-arm結果を一つのcanonical JSONへwrite-once保存し、外部digestを必須とするstrict loaderで再検証する。既存のledger / metric / provenance ownerを再利用し、`BASE_BAR`・elapsed-years未指定・全arm共通annualization、flat no-flow cash/equity closure、funding-event合計を検証する。FLAT-only cash armはtrade/rebalance、closed/open-position、realized net/gross P&L diagnosticsもゼロでなければならない。interval-end metric drawdownはledger intrainterval drawdown以下であることを確認し、両者を同一視しない。ledger v2に独立記録されないcorporate settlement proceedsを含むuniversal fill-to-cash再構築は行わない。保存されたmarket profileは宣言データとして保持する。replay前後のprovenance採取、Dataset cadenceの再認証、fit、Study、winner選択は所有しない。
- `ridge_comparison_attempt.py`: 公開済み比較をroot単位でwrite-once予約し、正確な入力宣言とbefore/after provenanceを記録して一度だけ比較・結果publish・strict reloadを順に実行する。失敗をfailure recordへ残し、比較が返った後はpartial ledgerを保つ。global retry budget、fit、data取得、winner選択、approvalは所有しない。
- `portfolio_family_comparison.py`: caller-pinned Dataset、Ridge artifact、joint-PPO bundle、fixed roster/window/configをpreflightし、cash・fixed Trend・Ridge・joint PPOを順に既存のnative shared-portfolio replayへ渡すin-memory comparatorである。全armの検証・policy/risk構築を最初のreplay前に完了し、armごとにfresh policy/risk/sessionを使う一方、同一Datasetとimmutable execution/risk/profile config valuesを共有する。native early terminationは完了扱いへ変換せず、raw outcomeとwindow/termination/insolvency/terminal flagsを保持する。optional `on_family_completed` observerは各outcomeを次armの前に渡し、later-armまたはobserver failureを即時伝播する。ownerはattempt reservation、journal、artifact publication、fit、winner selection、research approvalを行わない。
- `portfolio_family_result_artifact.py`: caller-declared identityとfour-family resultsをcanonical, digest-bound filesへwrite-once保存し、started record、各native family row、result manifest、provenance、Dataset identityをstrict reloadで照合する。新規identityではoptionalな`ridge_one_way_switch_cost`をfloatへ正規化する。publisherはmanifest書込み前にstarted recordの正規identity・rootsと、failure/complete markerや余分なentryのないpre-result inventoryを検証する。各ledgerの`execution_policy_digest`はattempt identityの有効なpolicy digestと一致しなければならず、verified market profileがあればprofile digestとdefault rule stressを含む。v1保存結果は、他のidentity fieldがcanonical bytesで一致し、このfieldだけが同値の非負な正確なint表現なら、元のidentity digestとresult/started identityの一致を検証して読み込める。ledgerの独立した第二計算や、nested native eventの再解釈は行わない。
- `portfolio_family_attempt.py`: comparatorをone-shot local attempt rootへ接続する薄いwrapperである。各family outcomeを次arm開始前に保存し、strict result reloadの後にだけcomplete markerを書く。failure journalが書ける場合は失敗phaseと先に保存したfamily rowsを残し、journal自体の失敗では元例外を維持して二次書込errorを注記する。failure/completionが併存するrootはpublic loaderが拒否する。global retry budget、fit、winner選択、research approvalは所有しない。
- `portfolio_family_scenario_result_artifact_v2.py` と `portfolio_family_scenario_attempt_v2.py`: declared scenario用の別v2 identity/result schemaとone-shot local attempt/loadを所有する。actual replay binding、source/provenance records、native rowsをexternal identity/result digestに対してstrictに照合し、各arm rowを次armの前に保存する。v1 payloadやloaderへ接続せず、帳簿を再計算せず、scenario選択やeconomic resultも所有しない。

`trade_rl.evaluation.runs` はcandidate-run contract、execution、artifact inspection/publication、provenance construction、v1 fixed four-family comparison/attempt、result loader、および`supports_shared_cash_result_schema` queryのTier-2 public facadeである。`config.py`、`candidate_suite.py`、`execute.py`、`artifact.py`、`provenance.py`、`ridge_artifact_comparison.py`、`ridge_comparison_artifact.py`、`ridge_comparison_attempt.py`、`portfolio_family_comparison.py`、`portfolio_family_attempt.py`、`portfolio_family_result_artifact.py` は引き続き実装ownerであり、facadeはこれらをwrapperなしでre-exportするだけとする。`portfolio_family_scenario_replay_v2.py`、`portfolio_family_scenario_attempt_v2.py`、`portfolio_family_scenario_result_artifact_v2.py` はexplicit opt-in v2 ownersであり、Tier-2 facadeへ追加しない。`evaluation/experiments/` を含むproduction codeは `evaluation/runs/` の外からRun Coreを利用するときfacadeを経由し、package内部は循環を避けるためowner moduleを直接参照してよい。Tier-1 `trade_rl.evaluation` の公開面はこの規則によって拡大しない。candidate-runのpersisted schema互換契約はPython import pathとは独立して維持する。

`runs` はhigher-level experiment lifecycleを知らない。`evaluation/experiments/` はStudy/Experiment contract、append-only store、multi-seed EvidenceSet、analysis、controlled delta、lineage/budget/freeze workflowを所有する。`contracts/research.py` の `StudyResearchContext` / `ConsumedEvidence` はStudyをまたいで既知development evidenceが次の研究定義へ流入した事実をmachine-readableに表し、context-bound `StudyPlan` digestの一部となる。これはresult/selection oracleではなくprovenance authorityである。`contracts/study.py` がversioned `StudyProtocol` identityと、PPO holding-duration protocolの完全一致risk profileを所有し、`protocols.py` はv1 independent-account / v2 shared-cash result eligibilityとwinner orderingをpure selectorとして共有する。`analysis.py` はv3 seed-symbol comparisonとv4 combined shared-cash portfolio comparisonを所有し、v4はpersisted portfolio seriesから再計算されたreturn / drawdownを選定へ渡す。`codec.py` はpersisted JSONから既存contractへのfail-closed decodeとstable payload/identity変換を所有し、`inspection.py` はdisk graphからのread-only state reconstruction・tamper validation・`inspect_study`を所有する。`workflow.py` はmutation lock下のcommand orchestrationだけを所有し、各mutation前のdisk再構築と既存failure-injection seamを維持する。

`evaluation/experiments/bootstrap/` は次だけを所有する。

- `config.py`: strict `CanonicalM2BootstrapConfig` parse/normalization/preflightと単一seed-policy authority。historical v1-v5のpayload/digest/read semanticsを維持する。v2は明示的なbuild-level execution economics、v3はpreregistered final window、v4はさらに `StudyResearchContext` をbootstrap identityへbindし、新規final-eligible research lineの正本となる。v5はlegacy independent-account PPO holding selector、v6はnew shared-cash PPO holding selectorを別StudyPlan schemaへbindする。
- `binance.py`: exact exchange-info / Vision source freeze、raw-source roster、cache-only transport composition。
- `workflow.py`: source → canonical dataset → immutable StudyPlanをwhole-root stagingで構築し、manifest検証後に一回だけpublishする。v2ではbuildへ渡したexecution economicsと、生成/reloadしたDataset economic arraysおよびidentity-bound profileの一致もfail-closedで検証する。
- `cli.py`: `--config` / `--output` をparseしてworkflowを呼ぶだけのfilesystem adapter。
- `__init__.py`: intentionally narrow public facade。

`trade_rl.evaluation.experiments` から公開するbootstrap APIは `CanonicalM2BootstrapConfig`、`CanonicalM2BootstrapResult`、`bootstrap_canonical_m2_study`、`inspect_canonical_m2_bootstrap` の4つだけである。source-freeze private helperはpublic contractではない。

Bootstrapはpreparation-onlyであり、baseline、Controlled Experiment、winner freeze、sealed final-test authorizationを実行しない。`evaluation/runs -> evaluation/experiments` の逆依存を作らず、`integrations`から`evaluation`へ依存させず、`evaluation/experiments/bootstrap`からsealed final-test ownerへ依存させない。`evaluation/final_test` は逆向きのread-only consumerとして `evaluation/experiments` のinspection/contractsだけへ依存し、data/integrations/strategies/replay/runs/robustnessをimportしない。

## Private development study boundary

`evaluation/experiments/four_family_input_manifest.py` owns only a typed,
post-Dataset-freeze declaration for the fixed cash, fixed-trend, Ridge, and
joint-PPO roster. It reuses the existing four-family attempt identity and
records one base selection attempt plus one base evaluation attempt, with a
separate training window and fixed metadata caps. Its PPO declaration also
records the training-side weights, execution-cost, risk, and optional profile
values, and requires both attempt identities to match them. The transform field
is the canonical `PortfolioObservationTransform.digest`, not a file hash.
`verify_bytes` compares canonical serialized bytes against a separately retained
typed declaration; the module does not reconstruct untrusted plans, load
Dataset/model artifacts, verify eligibility, fit, replay, select, or authorize
execution. It is private and does not expand the experiments facade or the
canonical `StudyPlan` lifecycle.

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

## Prequential forecast packet boundary

`forecasts/prequential.py` owns causal prequential forecast packets for a downstream allocator. `PrequentialSchedule` fixes an expanding-window refit calendar (`first_fit_time`, `refit_interval_days`, optional `activation_delay_hours`); `PrequentialForecastSource` lazily fits one frozen vintage per scheduled cutoff through the existing `build_causal_forecast_training_set` and a small `PrequentialForecastFitter` protocol (`RidgeForecastFitter`; optional `LightGBMForecastFitter`), and serves `ForecastPacket(symbol, horizon_hours, decision_time, log point_forecast, vintage residual_variance, vintage_id, fit_time, fit_cutoff, max_training_label_available_at, digests)`. `ForecastPacket` enforces `max_training_label_available_at < fit_time <= forecast_available_at <= decision_time` at construction; the vintage residual variance is the sample-weighted in-sample squared residual of the training rows, so a caller can convert log to simple expectation with `expm1(mu + 0.5 * s2)`. The source builds the causal training set itself and hands a fitter only that frozen `CausalForecastTrainingSet` (never the `MarketDataset`), so a fitter cannot read rows after the cutoff; training residuals and every served point forecast go through the same `fitter.predict_rows` path and `model.predict` must agree with it. The module reads only the decision-row features and cutoff-bounded training rows, owns no account, ledger, or order logic, does not import evaluation, simulation or risk, and persists nothing. Unusable decision features yield `None` (caller treats it as no forecast); a non-dataset decision time, an unavailable decision row, or a vintage whose labels were not strictly earlier than its fit time raises.

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

`simulation/objectives` はdeclaration-onlyのlower-layer packageであり、
`trade_rl._validation`、`trade_rl.artifacts.hashing` と自身の型だけへ依存する。
これはcurrent PPO/replayがobjective contractを消費することを意味しない。

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
- `source-packet` はfull commit SHAのGit objectだけを入力とし、changed HEAD fileは`git cat-file blob`のraw bytesから deterministic TARへ格納する。patchとpath inventoryはsource object databaseをalternate参照するtemporary bare repositoryで描画し、Git versionをmanifestへ記録する。inherited Git environmentとglobal/system configを除き、`GIT_NO_LAZY_FETCH=1`でmissing objectをfail closedにする。両diff queryは`--ignore-submodules=none`を明示し、色・algorithm・indent heuristic・context・path quotingを固定する。attributesはempty treeとrenderer-owned empty `info/attributes` / `core.attributesFile`を使いsystem attributesを無効にする。patch byte identityはmanifest記載のGit versionとこのpolicyの範囲で扱い、異なるGit version間のbyte identityは主張しない。patchはraw Git transportでありcheckout-normalized形式ではない。symlink・gitlink・unsafe path（`.git` componentおよびWindowsの`git~1` aliasを含む）、case-insensitive file/directory prefix collision among emitted HEAD files、mutable ref、既存output、registered worktree内outputは拒否する。削除済みbase pathは出力TARに含まれないためprefix判定に含めない。生成packetはsource transport onlyであり、review・test・economic evidenceのauthorityを持たない。
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

## Forward study FS1 boundary

`evaluation/forward_study/` は Issue #810 の sealed forward study FS1 だけを所有する。`registry.py` はFS1-A/B/C + CASHのroster・scoring window・falsifier・validity guardの凍結値と `FROZEN_PARAMS_DIGEST`（JSON exportは `report/issue810-fs1-registry-20261030.json`）だけを持ち、dataset・clock・resultを読まない。`labels.py` は数値だけを受け取る純粋な `FORWARD_PASS` / `FORWARD_FAIL` / `FORWARD_INVALID` ruleで、`WINNER` を返さない。`seal.py` はscoring前提条件（scoring end date経過、Issue commentのauthorization reference、凍結params digest、checkout済みcode SHAとclean tree、dataset seal manifest、読み込んだdatasetのidentity・window・tradable割合）を全て集めて `Fs1Refusal` で拒否し、満たしたときだけ `ScoringPermit` を返す。`fs1.py` はpermitを必須とする唯一のscoring entrypointで、Phase 3 policy builderと `dev_realdata_study.runner` を変更せず再利用し、write-onceでresultを書く。`fs1.py` の公開 `run_fs1` はregistry・digest・clock・code identity・dataset loaderを引数に取らず、private factory `_environment`（frozen registry、実wall clock、`git`）だけがそれらを供給する（testsはこのfactoryをmonkeypatchする）。code identityは `git rev-parse --show-toplevel` が解決したtrade_rl repository（pinned root commit）の追跡済み `seal.py` であること、HEADがsealed SHAに一致すること、変更済み・未追跡fileがないことを要求する。scoringはscoring end + 24h settlement marginの後だけで、manifestはfetch時刻とpinned build-policy digestを持つ。`cli.py` (`python -m trade_rl.evaluation.forward_study`) は引数をそのまま渡すだけでseamを持たない。

このpackageはnetworkへ出ず（`integrations` を import しない）、データを取得せず、`final_test` authorization chain・paper/live・RLを使わず、`trade_rl.evaluation` の公開APIを拡張しない。`dev_realdata_study`・`strategies`・`simulation`・`data` はこのpackageへ依存しない。account・P&L ledger・fill modelは持たない。実データのscoringはscoring end date後にuserが取得とscoring runを別々に認可するまで実行しない。Contractの詳細は `architecture/research-assurance.md` の FS1 section、状態は `research/current-status.md`。

### Forward study FS1 acquisition tooling

`evaluation/forward_acquisition/` はFS1のdata acquisition・dataset build・seal manifest・seal-record textを一つの文書化されたcommand（`python -m trade_rl.evaluation.forward_acquisition acquire` / `seal-record`）として提供し、ad-hocなscriptやmonkeypatchでFS1 datasetを作らせない。`forward_study/` はnetworkを一切importしないため、acquisitionは別packageに置く。`plan.py` はFS1 span・archive URL・date range・build policy照合・出力layoutを持つ純粋moduleで `trade_rl.integrations` をimportしない。`acquire.py` の `run_acquire` は `--allow-network` が無ければplanだけを返し（network clientをimportせず、fileを作らず、development inputも読まない）、ある場合だけ（Issue #810 authorization URL、新規output root、settlement margin経過、clean checkout、developmentのidentity検証が揃ったときに）`trade_rl.integrations.binance` をlazy importしてfetchする。`run_acquire` はregistry・digest・clock・code identity・transportを引数に取らず、private factory `_run_acquire` / `_environment` だけがそれらを供給する（testsが置換する）。development期間のarchiveは再取得せず、published development Datasetのidentity（pinned `dataset_id`・artifact digest・内部にbindされた `raw_source_roster_digest`）に対してread-onlyで検証して再利用する。manifest（`manifest_bytes_from_dataset`）は最後に書き、receiptがbuilder commit・exact command・exchange snapshot・development reuse identityを持つ。`seal_record.py` はmanifestとreceiptを凍結registryと相互に検証してseal commentのMarkdownを返すだけで、投稿しない。このpackageはscoring・replay・strategyに依存せず、`forward_study` からは `registry` と `seal` だけをimportする。それ以外のmodule scope依存は `trade_rl.artifacts`、`trade_rl.data`（`data.artifacts.publication`・`data.market`）とnumpyで、`trade_rl.integrations.binance`・`trade_rl.data.build` は認可された経路（`acquire.py` の関数内）でだけlazy importする。output rootはcheckout外でなければならず、`forward_study`・`dev_realdata_study`・`strategies`・`simulation`・`data`・`integrations` から参照されない。手順の正本は `architecture/research-assurance.md` の *FS1 acquisition procedure*、状態は `research/current-status.md`。

## Market-data store runbook

`integrations/market_store/` is a reusable, generic PostgreSQL store of public Binance market data (Issue #810). It is an adapter beside `integrations/binance/`: it depends only on `data`, `artifacts` and `integrations.binance`, never on strategies, evaluation, simulation or risk, and it is not re-exported from `trade_rl.integrations`. It stores USD-M perpetual and spot 1h klines and USD-M funding rates for a configurable universe (default BTC, ETH, BNB, XRP, ADA, SOL against USDT), reuses the Vision URL planner (`plan_vision_kline_urls`, `vision_funding_url`) and the unchanged `build_binance_market_dataset`, and owns no strategy, execution or P&L semantics. `psycopg` is imported only by `store.py` and is the optional `store` extra (also in `dev`); `plan` and a dry-run `ingest` need neither PostgreSQL nor the extra, and only `fetch.py` imports a network client.

Schema (plain SQL, `migrations/NNNN_name.sql`, applied in order with a `schema_version` ledger that records each file's SHA-256 and refuses an edited or unknown applied migration): `instruments`, `exchange_info_snapshots` + `instrument_rule_snapshots` (current-snapshot rules valid from retrieval, not historical rules), `sources` (one row per archive: URL, kind, period, archive sha256, the `.CHECKSUM` line digest, the sha256 of the `.CHECKSUM` file, size, HTTP status, fetched_at), `ingestion_runs` (code commit, command line, counts, status, FS1 warm-up flag and authorization URL), `klines`, `funding_rates` and `ingestion_conflicts`. Every price, quantity and rate is exact `NUMERIC`; natural keys are primary keys; CHECK constraints enforce `high >= max(open, close)`, `low <= min(open, close)` and non-negative volume. `klines`, `funding_rates`, `sources` (once ingested) and the exchange snapshots are append-only (triggers reject UPDATE and DELETE). TRUNCATE is also blocked on every evidence table, and a 404 `sources` record may only gain its evidence, never change its identity. What is committed outside the per-archive transaction: before the run, `ingest --allow-network` snapshots `exchangeInfo` (`exchange_info_snapshots`, `instrument_rule_snapshots`, `instruments`, `run_id` NULL); during the run, `instruments` rows and 404 `missing` source markers are committed as they occur, plus the `ingestion_runs` audit row; none of these holds a price, a funding rate or a boundary state. Re-ingesting an archive is a no-op; a changed value for an existing key rolls the archive back, is written to `ingestion_conflicts`, fails the run and is never applied.

FS1 data-protection boundaries. The FS1 forward study controls two spans (constants pinned by tests against the FS1 registry `warmup_only_start` / `scoring_start` and against the SQL functions of the migration). (1) The **warm-up span** `[2026-08-01T00:00Z, 2026-11-01T00:00Z)` may, per *FS1 acquisition procedure* in `research-assurance.md`, be fetched only with a separate Issue #810 data-fetch authorization, so the store refuses it **by default**; a run may admit it only with `--allow-fs1-warmup-span` together with `--fs1-warmup-authorization-url` naming that Issue #810 comment (both or neither). The flag, the URL and the run are recorded in `ingestion_runs`, the database triggers admit warm-up rows only for that still-running authorized run, and the URL is an attestation of the user's authorization, not a verified one. The boundaries are module constants (`FS1_WARMUP_START`, `SEALED_WINDOW_START`), not parameters of any public constructor, planner, ingester or exporter, so no caller can move them. Even then FS1 scoring does not use the store: it re-acquires and re-verifies its own data through its sealed tool (`evaluation/forward_acquisition`), and store rows never feed FS1 scoring. (2) The **sealed window** `>= 2026-11-01T00:00Z` (scored once after 2027-05-02) is refused **unconditionally**: there is no flag, parameter, table or SQL function that lifts it, and no sealed or warm-up row, authorization or release state is ever stored by a failed or empty run. Lifting it after FS1 is scored needs a future reviewed migration (replacing `market_sealed_from()`) and pull request. Both spans are enforced at five layers, and the decision is always made from the period parsed out of the Vision URL itself (`archive_url.py`), never from caller-declared metadata: `ingest` refuses any `ArchiveSpec` whose kind, market, symbol, interval, cadence or period disagrees with its URL before a run starts or anything is requested; `VisionFetcher.fetch` independently refuses, before any network call, every URL that is not a recognised Vision archive or `.CHECKSUM` path with a period entirely before 2026-08-01 (a fetcher constructed with the warm-up authorization URL may also take periods entirely before 2026-11-01; none can take a URL touching the sealed window) and never follows redirects; the planner never plans an archive whose period passes a closed span (a straddling monthly archive is replaced by its in-bounds daily archives; a funding month that cannot be published inside the bound is reported and skipped) and a requested range end inside a closed span exits with code 4; every parsed row is checked before any write; and PostgreSQL triggers on `klines`, `funding_rates` and `sources` reject such rows even under a direct INSERT. By default `plan`/`ingest` therefore end at 2026-08-01. The database guard for the warm-up span is an accident guard, not an authorization boundary: it admits warm-up rows only to the backend session that started a still-`running` authorized run no older than 12 hours (a hard-killed run is closed by the next `begin_run`, and an old one stops admitting rows), run identity and authorization columns are immutable and a finished run cannot be reopened; but the session variable `trade_rl.fs1_warmup_run_id` is an ordinary custom setting that any login role may `SET`, any role that may INSERT into `ingestion_runs` can create an authorized-looking run, and PostgreSQL cannot verify that an Issue #810 comment exists. Keep the store's write role with trusted operators; the sealed window has no such path.

Export. `export_market_dataset` serves stored rows through the same transport interface as the Vision file path (Vision column layout from exact values, funding events with the Vision inclusive range, one stored raw `exchangeInfo` applied statically) so the unchanged builder, gap policy, 4h/1d aggregation and identity produce the `MarketDataset`. Digests of every row read, the exchange snapshot and an `fs1_warmup` record (whether any exported bar lies in the warm-up span, the last bar time and the authorizing run ids and Issue #810 URLs) enter the identity as `metadata_evidence` and the export receipt; the same store content and parameters give the same `dataset_id` (pin the rules snapshot with `--rules-snapshot-sha256`), and a test proves a store-backed build equals a file-backed build from the same archives.

Operating it (Windows, PowerShell; install Docker Desktop first, then `uv sync --extra store`):

```powershell
Copy-Item deploy/market-store/.env.example deploy/market-store/.env
# put a random password into MARKET_STORE_PASSWORD (never commit .env)
$pw = -join ((48..57 + 65..90 + 97..122) | Get-Random -Count 32 | ForEach-Object { [char]$_ })
(Get-Content deploy/market-store/.env) -replace 'CHANGE_ME_generate_a_random_password', $pw | Set-Content deploy/market-store/.env
docker compose -f deploy/market-store/compose.yml --env-file deploy/market-store/.env up -d
docker compose -f deploy/market-store/compose.yml ps    # wait for "healthy"
$env:TRADE_RL_MARKET_STORE_URL = "postgresql://market_store:$pw@127.0.0.1:55432/market_store"
uv run python -m trade_rl.integrations.market_store migrate
# default form: stops before the FS1 warm-up span (2026-08-01); --end is shown explicitly
uv run python -m trade_rl.integrations.market_store plan --start 2021-01-01 --end 2026-08-01   # offline
uv run python -m trade_rl.integrations.market_store ingest --start 2021-01-01 --end 2026-08-01  # dry run
uv run python -m trade_rl.integrations.market_store ingest --start 2021-01-01 --end 2026-08-01 --allow-network
# authorized form (only after the user's Issue #810 data-fetch authorization; never past 2026-11-01):
#   ... ingest --start 2026-08-01 --end 2026-10-30 --allow-network `
#       --allow-fs1-warmup-span --fs1-warmup-authorization-url https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-<id>
uv run python -m trade_rl.integrations.market_store status
uv run python -m trade_rl.integrations.market_store export --market usds_m --start 2021-01-01 --end 2026-08-01 --output-dir C:/data/exports/usds_m_2021_2026
```

The container binds only `127.0.0.1`, keeps data in the named volume `trade_rl_market_store_data` and reads credentials from the git-ignored `deploy/market-store/.env`; the connection URL is read only from `TRADE_RL_MARKET_STORE_URL`. Backup and restore: `docker compose -f deploy/market-store/compose.yml exec -T postgres pg_dump -U market_store -Fc market_store > market_store.dump`, then `pg_restore --clean --if-exists -d market_store` on a fresh container. Integration tests run against a real PostgreSQL only when `TRADE_RL_TEST_POSTGRES_URL` is set (each test uses its own throw-away schema); otherwise they skip. CI runs them in the `Market Store PostgreSQL` job with a `postgres:16.4` service container, fails if any of them was skipped, and that job is in the `needs` of the generic independent-review gate, so a red store job blocks the gate. FS1 scoring and acquisition never import the store (architecture test).

## Repository integration boundary

GitHub PR / CI / branch-protection・ruleset設定はRepository統合の安全性を管理するが、`trade_rl` runtime packageのauthorityではない。checked-in architecture testはPR/CI policy fileの契約を検証できるが、実際のbranch protection状態はGitHub側のread-backで別途確認する。

Integration invariant: tested PR head contains current `main`. merge直前のcurrent `main` commitがtested PR headのancestorであり、その同一PR HEADにpermanent CI successが存在することを統合証拠とする。`main` が進んだ後の古いPR-head Greenは再利用せず、current `main` を含む新HEADを再検証する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等の証拠としてよい。

このGit tree内のproseやarchitecture testだけでbranch protectionが有効とは判断しない。ruleset/protectionの設定変更後はGitHub stateをread-backし、required check、PR requirement、force-push/deletion、maintainer/admin bypassを確認する。管理surfaceが利用できない場合は未設定/未検証として扱う。

`Generic Independent Research Review` はhistorical job名を維持するが、Generic software PRではexact HEADの**model source review**を要求する。authenticated GitHub review APIの`COMMENTED`または`APPROVED`に単独marker/disposition/digest付きreceiptを必須とし、`software_source_review_v2`では`reviewer_model_id`・PASS・blocking finding 0・具体的reportをpublisher-attested metadataとして受理する。同じconfigured model（例: GPT-5.6 Sol）を許容し、provider-run provenance・diff digest・実装Task ID・model independenceは要求も主張もしない。旧`software_source_review_v1`は未検証job/diffを含むためPENDINGのまま廃止し、`independent_model_review_v1`は互換経路として残す。正本は [research assurance](research-assurance.md#generic-software-pr-review-receipt)。

同じexact HEADのactive submitted marked `### Disposition: BLOCKED` が一つでもあれば承認より優先し、後続の同一actorによる承認でも消去しない。保留・曖昧・dismissed・古いHEAD・不整合なreceiptはfail-closedに扱う。重大な指摘事項（Medium / High）が解消され全必須CIがGreenであれば、承認済みsoftwareレビューはIntegratorによるマージ・クローズの対象となる。このGeneric receiptは `Independent Research Review` の4h PPO smoke専用review/triggerやG4 economic execution、sealed PPO、unused/final one-shot authorization、live eligibilityを代替しない。

`tools/ppo_4h_gemini_review.py` は4h PPO smokeのresult-blind external-AI reviewをdefault-branch trust rootから実行する**repository-local reviewer transport**であり、`trade_rl` runtime packageやeconomic evaluatorの一部ではない。`.github/workflows/ppo-4h-gemini-review.yml` はcanonical PR comment requestを入口に、(1) targetをdataとして読むpacket生成job、(2) Gemini secretを持たずexact reviewed SHAへ固定command setを実行するCore / PPO Runtime / Human Guide verification jobs、(3) canonical packetだけを受け取るfresh secret-bearing review jobを分離する。review jobはtarget checkoutやtarget-generated executable artifactをauthorityとして受け取らず、trusted runner/workflow identity、request/tag identity、packet digest、trusted verification job identity、Gemini request/response identityをreviewer-run attestationへbindする。Gemini API key/model selectionはworkflow configurationであり、concrete model versionはprovenanceとして記録するがproduction trading/runtime schemaやvalidator allow-listへ固定しない。

## Development real-data study boundary

`evaluation/dev_realdata_study/` は Issue #810 P2 の development-only real-data walk-forward study だけを所有する。`protocol.py` は事前固定した全parameter・5 arm roster・3 scenario・selection ruleの定数とSHA-256 digestを持ち、実データ実行の前にcommitする。`runner.py` は `strategies/forecasts/prequential.py` の causal packet を log→simple return に写す adapter、`strategies/allocation/` の policy、armごと・scenarioごとに1回の連続 `run_shared_portfolio_replay`（terminal flat settle、ledger capture）、raw fill からの独立 ledger reconciliation、未来perturbation causality probeを束ねる。`metrics.py` は equity-path metrics、20日block bootstrap、`DEVELOPMENT_PROMISING` / `NO_WINNER` だけを返す選定ruleを持つ。このpackageは `WINNER` を出さず、unused-future窓を開かず、account・P&L ledger・fill modelを持たず、`trade_rl.evaluation` の公開APIを拡張しない。CLI (`python -m trade_rl.evaluation.dev_realdata_study`) は新規出力のみを書き、既存出力を上書きしない。`dataset_build.py` は同study用Datasetをlocalの Binance Vision archiveだけから作る **offline・opt-in** builder（`python -m trade_rl.evaluation.dev_realdata_study.dataset_build`）で、networkへ出ず、非空の出力先を拒否し、economic replayを実行しない。gap policy (`non_tradable_rows`: 欠損1h barはnon-tradable row・forward-fill mark・funding非計上、4h/1d barは完全な1h windowからaggregate、欠損windowは `available_at=range end` のplaceholderにして再帰的な上位clock featureを以降unavailable化) と入力rosterを `DATASET_BUILD_POLICY` とそのdigestとしてreceiptへ固定する。gap許容は `integrations/binance/dataset.py` の明示的な `BinanceGapPolicy` option（default `STRICT` = 従来どおり欠損barを拒否）と `integrations/binance/derived_klines.py` の1h→4h/1d aggregation transportが所有し、monkeypatchは使わない。

Phase 2 trial registry は同じpackage内の `trials.py`（T1-T8の事前固定spec・定数・実装判断、JSON exportは `report/issue810-phase2-trial-registry-20261029.json`）、`trial_sources.py`（`ExpectedReturnSource` registry、causalなTSMOM/XS-momentum/funding carry/blend source、T7のdrawdown overlay policy wrapper）、`trial_study.py`（armごとの隔離実行・FAILED_TO_RUN記録・13 arm Bonferroni・同一selection rule）、`trial_cli.py`、事後ラベル再計算の `trial_labels.py`（PR-C probe ruleとamended ruleの両ラベルをcommit済み証拠からcodeで計算）が所有する。PR-Cの `runner.py` を `arm_builder` / `build_allocator` / 汎用causality probeで拡張して再利用し、複製しない。同じくaccount・P&L ledgerは持たず、`WINNER` を出さない。 今後のreplayはcontrol両armの有効性・probe matrixをselection gateで検査する。経済結果の凍結を維持したまま、途中終了や非対応日時のbootstrap比較を除外理由とともに記録する。CLIは新規の非重複出力先だけを許可し、登録cut以外の因果性検証はdiagnostic-onlyと記録する。T7 overlayはinner allocatorの同時刻fail-closed HOLDを保持し、risk縮小の抑止件数を診断出力する。

Phase 3 roster（turnover aversion `kappa` とex-ante volatility target `VT12`、POST_HOC_EXPLORATORY・事前登録ではない）は `phase3_registry.py`（8 armの凍結roster・公開事前記録のverbatim本文とSHA-256・実装判断notes・JSON exportは `report/issue810-phase3-trial-registry-20261029.json`）、`phase3_sources.py`（T4×7 source、T2/T4 z-blend、vol-scaled passive comparatorのpolicy、arm builder）、`phase3_study.py`（armごとの隔離実行・causality probe/matrix・PR-C probe rule主ラベルと補助のamended rule・Phase 3 family 8とall-arms 21のBonferroni）、`phase3_cli.py`（新規出力のみ・全実行を数える`--execution-ledger`）が所有する。`kappa`（optimizerのL1 cost penalty倍率のみ。実行costは不変）とVT（解いたtargetをmin-trade rule前に`cap/sigma_p`倍へ縮小）は `strategies/allocation/cost_aware.py` のopt-in引数で、defaultでは従来のdecisionとbit-identical。label語彙は `POST_HOC_PROMISING` / `NO_WINNER` で、`WINNER` は出さない。

## Public API policy

Intentionally maintainedなpackage-level importは、内部private file移動より優先して安定させる。

例:

```python
from trade_rl.data import MarketDataset
from trade_rl.data import load_published_market_dataset_artifact
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
