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
│       ├── source_clock.py
│       └── dataset.py
│   ├── market_store/{__init__.py,__main__.py,contracts.py,embargo.py,plan.py,checksum.py,parse.py,ingest.py,fetch.py,rules.py,store.py,export.py,carry_source.py,cli.py,migrations/*.sql}
│   └── cross_asset/{__init__.py,__main__.py,catalog.py,parsers.py,pit.py,store.py,fetch.py,observations.py,features.py,normalization.py,loader.py,dataset.py,cli.py}
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
│   ├── rules/{trend.py,mean_reversion.py,channel_breakout.py,trend_stop.py}
│   ├── forecasts/{controller.py,supervised.py,ridge.py,ridge_artifact.py,lightgbm.py,prequential.py}
│   ├── allocation/{solver.py,cost_aware.py,buy_and_hold.py,costs.py}
│   ├── agy_anchor/{primitives.py,inputs.py,signals.py,targets.py,policy.py}
│   └── rl/{intent.py,ppo.py,portfolio_env.py,portfolio_transform.py,portfolio_runtime_scenario.py,portfolio_scenario_runtime.py,portfolio_scenario_inference.py,portfolio_training.py,portfolio_inference_bundle.py,portfolio_episode_journal.py,portfolio_policy_sample_journal.py,a2c.py,ppo_normalization.py,ppo_artifact.py,a2c_artifact.py}
└── evaluation/
    ├── replay.py
    ├── paired_portfolio_action.py
    ├── paired_portfolio_label.py
    ├── decision_value_fit.py
    ├── decision_value_plan.py
    ├── decision_value_feature_transform.py
    ├── decision_value_ledger_oracle.py
    ├── decision_value_reachable_labels.py
    ├── decision_value_margin.py
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
    ├── signal_measurement/{__init__.py,ic.py,leakage.py,multitest.py,payoff.py,profile.py,significance.py,targets.py}
    ├── trend_stop_study/{__init__.py,__main__.py,protocol.py,metrics.py,runner.py,data.py,cli.py}
    ├── forward_study/{__init__.py,__main__.py,registry.py,labels.py,seal.py,fs1.py,cli.py}
    ├── forward_acquisition/{__init__.py,__main__.py,plan.py,acquire.py,seal_record.py,cli.py}
    ├── signal_diagnosis/{__init__.py,protocol.py,verdict.py}
    ├── signal_diagnosis_run/{__init__.py,__main__.py,cli.py,fitting.py,leakage.py,measure.py,panel.py,runner.py}
    ├── xs_lowvol_funding/{__init__.py,protocol.py,universe.py,policy.py,data.py,decision.py,runner.py,candidates.json}
    ├── ppo_feature_study.py
    ├── ppo_feature_checkpoint.py
    ├── ppo_normalization_replication.py
    ├── ppo_normalization_execution.py
    ├── ppo_normalization_activation.json
    ├── paper/
    │   ├── {__init__.py,store.py,account.py,engine.py,control.py,failure.py,lock.py,collection_policy.py}
    │   ├── {supervisor.py,operations.py,watchdog.py,predecessor.py,attempt2.py,linux_service.py}
    │   └── {storage_probe.py,assessment.py,cli.py}
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

`evaluation/experiments/store.py` の `StudyStore.publish_json_once` はcanonical JSONを同じ親directoryのstagingへ書き出してflush/fsyncした後、排他的なhard linkで最終file名を一度だけ確保する。書込み後の存在確認から公開までの間に別processがfile（空fileを含む）を作っても置換せず、`FileExistsError`を既存の`InvalidExperimentStateError`へ変換する。競合時と未対応filesystemでの失敗時はstagingだけを清掃し、通常のPOSIX renameやcopyへfallbackしない。単一fileのno-replace公開を保証する境界であり、複数fileのtransaction、外部producerの認証、OS全種類でのhard-link可用性や経済結果の正しさまで証明しない。 hard linkが成功した時点でcanonical fileは公開済みとなる。以降のstaging削除に失敗した場合は`PublishedArtifactCleanupError`（`InvalidExperimentStateError`の派生）を返し、公開済み・後処理失敗・同名再公開禁止を明示する。callerはcanonical fileを確認し、再publishを試みない。hard link前の失敗とstaging削除失敗が重なった場合は元の例外を優先する。その際未削除のstagingが残り得るが、canonical fileを上書き・削除して自動修復しない。

PPO/A2Cの個別研究の固定設定・セル検証・判定oracleは、[外部研究リポジトリの研究文書](https://github.com/shuntatsu/trade_rl_research/blob/main/docs/README.md) に分離する。基盤はPPO/A2Cの学習adapter、runtime、独立したconstructor契約を引き続き所有し、研究packageへの逆依存は持たない。旧実装は元のGit commitに保全する。経済実験・認可条件・学習実装を変更しない。

The closed synthetic residual-PPO study over the cost-aware allocator (Issue #810) is archived at [the research archive for `evaluation/residual_rl/`](https://github.com/shuntatsu/trade_rl_research/blob/main/docs/residual-rl-archive.md) and its [frozen source, evidence and test manifest](https://github.com/shuntatsu/trade_rl_research/blob/main/migration/residual_rl_archive.json). Its historical separate value ledger and executable study are not part of the core distribution. The original protocol, provenance, and result JSON remain byte-identical in `report/`; `SYNTHETIC_FAIL` remains synthetic evidence only, never a profitability claim and never a clearance of the real-data fee-only learnability gate. The shared cost-aware allocator remains in the core.

## Provider evidence boundary

The separate paired-carry path keeps provider assembly in
`integrations/binance/carry.py`, deterministic monthly matched-quantity sizing
in `strategies/carry.py`, and replay composition/diagnostics in
`evaluation/carry.py`. It submits explicit quantities through canonical orders
and preserves GTC residuals; it does not convert a quantity hold to a newly sized
weight each hour. The existing BookState and stateful executor own all P&L.
Futures collateral checks are a projection of that same book excluding spot
assets, not an additional ledger. No live order connection is provided.

The multi-symbol carry development screen (Issue #810 Track B, spec
`docs/specs/funding-carry-multisymbol.md`) extends that path without replacing it.
`integrations/binance/carry.py` keeps `assemble_carry_dataset` (BTC/ETH only, content
identity byte-identical, pinned by a golden-identity test) and adds
`assemble_multisymbol_carry_dataset`, which reuses the same assembly body and
economics for a protocol-declared USDT universe of up to eleven pairs under its own
`funding_carry_multisymbol_development_v1` identity, plus `parse_carry_perp_rows`
(strict hourly perpetual rows) and `CarryCoverageError` (a documented absence only: an HTTP 404 on a planned archive or a
gap no declared halt explains, with its URL/status evidence; it is the only error that
may exclude a symbol). Server errors, transport failures, timeouts (bounded retry), an
oversize response (`RESPONSE_TOO_LARGE_STATUS`, not retried) and offline cache misses raise
the resumable `CarryOperationalHalt`; a checksum mismatch, a
malformed archive and duplicated, unordered or malformed rows raise `CarryIntegrityHalt`.
The runner leaves a `halt.json` marker for either (and for any unexpected error) and
writes no decision, so a halt can never become an exclusion or `INCONCLUSIVE_UNIVERSE`.
That outcome is terminal for the protocol: the runner refuses to reopen an output
directory holding it. The window end is exclusive for funding events as for klines.
`evaluation/carry.py` accepts any even leg count from two to twenty-two for the same
alternating spot/perpetual layout. `evaluation/carry_multisymbol/` owns the
pre-registered protocol constants and digest (`protocol.py`), the pure decision rule
(`decision.py`) and the output-directory runner (`runner.py`); the runner consumes a
`ScreenData` protocol, refuses any window but the registered one and writes every cell
before the decision. It imports nothing from the market-data store: the architecture
test that forbids any import of that package from outside it still holds. The store
package therefore owns the data side, `integrations/market_store/carry_source.py`
(`VisionArchiveSource`, `ArchiveDirFetcher`, `CarryScreenData`): checksum-verified
Vision archives without a database or Docker, or a populated store through the same
`CarryRowSource` protocol. `ArchiveDirFetcher` uses a distinct private staging file
per producer and a same-filesystem exclusive hard link when publishing cached
archive/checksum bytes; a raced existing cache file wins and is returned for
normal downstream checksum verification without overwriting. Pre-existing or
raced-in symlink cache entries are refused, rather than followed as cache hits.
Cache reads reuse the existing regular-file helper: supported POSIX
descriptor-relative no-follow opens reject a cache symlink swapped after a
cache-hit existence check, while the portable fallback refuses static linked
paths without claiming atomic protection against concurrent swaps. Any
already-linked ancestor in the absolute filesystem path (including above the
configured cache root), the cache root itself or a subdirectory is refused before
reads or writes, matching the cached-read no-follow boundary. On a successful
cache miss, the directory screen runs again after the injected fetch and after
parent creation, before staging bytes. This refuses directory links planted at either callback
boundary, but path-based checks are not atomic against an actively concurrent
directory replacement. Metadata inspection errors on cache directories, cached entries,
raced publication winners and private-stage cleanup are typed operational halts,
never an offline cache miss or missing-data exclusion. A stage cleanup failure
after an existing typed publication halt is retained as a diagnostic note,
not used to overwrite the primary publication failure. Filesystems without hard-link
support fail closed as operational halts; there is no ordinary rename, copy-overwrite or shared
`.part` fallback. These are cache publication and source integrity properties,
not economic evaluation or market-data authorization.

The wiring of the two sides is the repository-only command
`tools/carry_multisymbol_screen.py` (`plan`, `screen`), which sits outside the
production wheel; it requires a G4 authorization comment URL, the registered window and
a clean checkout, and reads nothing at or after 2026-08-01.

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
market summaries from raw bytes; consumption latency (`as_of` within
`MAX_CONSUME_LATENCY_MS` of capture completion) is separate from offline
historical verification and from the per-quote age bound at capture completion.
The worst-case venue-stamped mark age at valuation is `MAX_QUOTE_AGE_MS` +
`MAX_CONSUME_LATENCY_MS` = 15 s. `MAX_QUOTE_AGE_MS`, `MAX_QUOTE_FUTURE_SKEW_MS`,
`MAX_REQUEST_SECONDS` (one request), `MAX_CAPTURE_SPAN_SECONDS` and
`MAX_CONSUME_LATENCY_MS` are protocol parameters set from the source and engineering
reasoning, not from observed timing: a new paper attempt must pre-register them in its
research-assurance packet (attempt 3: `report/issue810-carry-attempt3-research-assurance-20261108.json`,
bound to these constants by test; its collection policy is implemented under
`report/issue810-carry-attempt3-implementation-research-assurance-20261108.json` and its
screen v3, evaluator audit and lineage v4 under
`report/issue810-carry-attempt3-seal-candidate-research-assurance-20261108.json`), pass G0-G2 and independent review, and never reuse an
earlier seal under changed tolerances.
`forward_rules.py` captures current public
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
`paper/collection_policy.py` is the pure, I/O-free home of the pre-registered
attempt-3 collection-failure policy (retry, skip and rules-hold constants, the
Class D/I/T classification of an acquisition failure from exception types and the
three skipped-slot budgets); `supervisor` applies it and `control` stores its
registry. It imports only `integrations.binance.forward` for the freshness error
type, and the watchdog never loads it. A transient failure is retried only inside
its slot before a capture is consumed; an unobtainable slot is skipped without a
command; a Class D/I failure or a budget breach writes the permanent marker.
`paper/control.py` records cycle start before acquisition and completion only
after account commits, preventing restart from hiding failed or unused captures,
and holds the tables `attempts` (one row per failed or unused capture directory)
and `skips` (one row per skipped slot) beside the cycle table. The transport
chains its final error to the HTTP or socket cause and a capture's freshness
rejection is `ForwardFreshnessError`, so the classification reads types, not
messages. These control records assign no financial meaning. `paper/operations.py` freezes
the ninety-day economic screen, runs sixty-second slots and exposes explicitly
chain-only operational status; `screen_plan` (v2) and `screen_plan_v3` share one
constructor of the economic fields, and v3 adds only the `operational_quality` clause
that restates the collection-policy constants. `paper/assessment.py` excludes a running
collector, replays the pinned event tip and all financial inputs, audits control/source
rosters, then measures and applies the fixed screen; the sealed plan's schema selects the
v2 audit (any failed capture or skipped slot rejects) or the v3 audit. `paper/collection_audit.py`
is the pure v3 audit: it reads the `attempts` and `skips` registry through a read-only
connection the caller passes, re-derives the skipped slots from the cycle roster, applies
the three budgets and the per-slot failed-capture bound, and builds the skip report (count
and distribution relative to the 00:00, 08:00 and 16:00 UTC settlements); it imports only
`collection_policy`. `paper/cli.py` exposes seal,
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
`paper/attempt3.py` is the same plan/seal/launch composition for the successor: it reads
the preserved attempt-2 root's identity with the v1 reader (read-only), builds lineage
v4 and the `screen_plan_v3`, and has the same offline/no-collector import boundary.
`paper/linux_service.py` is the Linux operating platform of that screen: the run layout,
the systemd user unit text and a read-only chrony/timesyncd clock report (allow-listed
read-only commands only); it imports no network, collector, journal or market surface,
writes a unit only on an explicit `--output-dir` and never changes host state. Its
successor packet is
`report/issue810-carry-attempt3-linux-platform-research-assurance-20261119.json`.
`paper/predecessor.py` holds attempt 1's recorded identity, the canonical lineage v3
and v4 records, the single lineage classifier (`lineage_structure_problems`) and the
read-only re-verification of a recorded predecessor root. `seal_collection`,
`seal_paper_study`, `cli seal`, `PaperEngine.initialize` and `PaperEngine(root)`
(on open), `validate_attempt_lineage`, `attempt2`, `attempt3`, `PaperCollector`,
`cli run` and `cli supervise` all use it; v3 also needs a module-private token held by
`attempt2` and v4 one held by `attempt3` at the seal functions (a caller contract), and
the collector re-reads an `available` root (v4: with its failure marker) on every entry.
A screen v3 plan is accepted only with lineage v4 and the reverse. This guards against accidental or non-canonical records through
the supported entry points; it is not authentication against a malicious local
operator. Among paper modules it imports only `account` at load time. These
operations contain no production routing.

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

small strategy interfaceとlogical intent、controls、rule、forecast、teacher-free RLを持ち、evaluationを知らない。`strategies/portfolio.py` はnumeric full-vector policy proposalと、tuple-backedな現在行・account・active-order・risk snapshotのdata-only contractを持ち、evaluation/simulation/riskのruntime objectをimportしない。`strategies/portfolio_observation.py` はsource-owned `MarketDataset`、現在の`BookState`、active-order tuple、risk config、decision windowからこのsnapshotを組み立てる。`strategies/portfolio_adapter.py` は固定roster・明示budgetでexact built-in `ConstantIntentStrategy`、`TrendIntentStrategy`、`RidgeForecastStrategy` をnumeric `PortfolioPolicy` proposalへ写すopt-in adapterと、明示cash zero-target policyを持つ。filled sideはsnapshotのexact quantityから導き、risk・pending-order reconciliation・fill・cost・accountingは共有portfolio session / native ownerに残す。legacy cached-target、minimum-hold、pending-aware intent、exact-quantity HOLD、任意strategy parityは提供しない。adapterはruntime/evaluationから独立し、strategy facadeへも再exportしない。`strategies/portfolio_policy_codec.py` はこのDTOとNumPy配列のraw-v1 Dict projection、strict joint action-code decoder、target proposal mappingを共有する純粋なstrategy-layer ownerであり、Gym/SB3/evaluation/simulation/riskをimportしない。`PortfolioInferencePolicy.decide_portfolio` はloaded bundleのsymbol/feature roster、decision window、native timestampを検証して同じcodecを使う。evaluation replayは既存のaccountingや実行経路を変えず、このbuilderを呼び出す。pending-order snapshotはintentのlimit・stop・submission-reference price factsもコピーする。3値すべてNoneのlegacy snapshotはconstructor互換だが、価格contextとしては不完全である。`strategies/allocation/` はcaller供給のforecast sourceに対する数値`PortfolioPolicy` seamを持つ。`solver.py` は`mu.w - gamma/2 w'Sigma w - sum c_i|w_i-w0_i|`をbox・gross制約の下でcyclic coordinate descent・active-set Newton加速・gross Lagrange乗数のbisectionで解く、NumPyだけに依存する純粋なsolverである（`trade_rl`内の他packageはimportしない）。解は許容誤差内の精度で、返す前にKKT最適性certificate（subgradient条件・box normal cone・gross相補性）を検証する。非収束・非有限・制約違反・certificate不合格は例外にして呼び出し側をfail-closedにする。`cost_aware.py` の`CostAwareAllocationPolicy`は固定decision schedule（既定は毎日00:00 UTC）でのみretargetし、decision時刻以前のmark価格だけからEWMA共分散を作り、tiny tradeはexact-quantity HOLDにし、solver失敗・forecastの不正な返却ベクトル・履歴不足の場合は全symbolをHOLDして注文を出さない（forecast source自体が投げた例外は現行protocolに従い伝播する）。有限なmark価格同士の除算・有限なリターンの二乗／重み付けでも浮動小数点overflowは起こり得るため、EWMA covarianceの導出結果が非有限なら拒否し、policyは`invalid_covariance`でexact-quantity HOLDしてforecast/solverに渡さない（通常の有限共分散と研究パラメータは不変）。`buy_and_hold.py` は初回に等weightを取り以後exact quantityを保持する対照policyである。これらはaccountもP&L ledgerも持たず、risk・fill・cost・accountingは共有portfolio sessionに残る。`costs.py` だけが`simulation.execution.ExecutionCostConfig`を読んで一方向cost（impact・slippage・per-row overrideを含まないlower bound）を導出し、他のallocation moduleは`strategies.portfolio`以外のtrade_rl packageをimportしない。`trade_rl.strategies`のfacadeへ再exportしない。`dataset_scope.py` はdatasetに束縛されたfeature/symbol selection validationの単一ownerであり、forecastとRLのsibling familyが互いの内部実装へ依存せず共有する。`forecasts/supervised.py` はRidge / LightGBMが共有するcausal training rowsとfit-scope validationを所有し、両fit pathはestimator / solverより前にこれを通る。`fit_cutoff` / `label_end_times` の元unit、正のmultiplier、範囲、sub-nanosecond精度のexact integer validationは `data/_timestamps.py` が単一ownerとして提供する。fixed-duration unitとcalendar `Y` / `M`は値を正確にnsへ表せる場合だけ受け入れ、cast後の逆変換をoverflow判定の根拠にしない。Ridge / LightGBMのcausal rowsと`fit_cutoff`にはlosslessな`datetime64[ns]` validationを適用する。`position_duration.py` は実際のsigned quantityから保有episode ageを導き、minimum-hold中のintent制約を共通定義する。model自身やcandidate config自身の不変条件validationは各ownerに残す。

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

`evaluation/terminal_settlement.py` owns the reusable observer-only terminal
dust classification and mark-net-of-exit-cost equity adjustment. It consumes
native shared-cash replay ledger evidence, Dataset venue rules and execution
costs, without changing any order, quantity, accounting transition, minimum
notional or study authorization. External research may import its two public
functions directly; the historical development runner retains the existing
names as aliases to the single shared implementation. Historical results and
source-bound review packets are not migrated or re-authorized by this API.

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
- `config.py`: Run JSONの単一parse/resolution authority。候補Run configのforecast horizonをstrictに検証し、`gross_budget`は有限な`(0, 1]`、`initial_capital`は有限な正値として結果生成前にfail-closedする。`fit_cutoff` / `evaluation_start` / `evaluation_stop_exclusive` の元ISO範囲・precisionもlosslessにns正規化してdataset-resolved specへbindする。typed `datetime64` はshared data helperでexact integer normalizationを行う。既にwrapされた `datetime64[ns]` から元時刻は復元できない。永続化する`ResolvedRunConfig`も同じbudget/capital境界を再検証し、Study/Experiment readerが実行不能な有限値をsemantic identityとして受け入れない。
- `execute.py`: resolved specから既存candidate suiteを一度実行するin-memory seam。
- `provenance.py`: implementation/runtime/research-context provenance生成。
- `artifact.py`: summary/raw returns/provenanceのpublication、verified load、semantic identity。`supports_shared_cash_result_schema`は、shared-cash evidenceを持てるresult schemaとしてv7/v9/v10を許可する。新規Runは正のforecast horizonと明示的な`forecast_switch_cost`（`null`または有限な非負値）を`lean_candidate_result_v10`へbindし、per-symbolとoptional shared-cashの両方を同じcurrent schemaで表現する。v10はrisk config・完全なevaluation-period coverage・training/replay suppression evidenceを保持し、shared-cash時はsettlement / ledger evidenceも必須とする。loaderはreturn / maximum drawdownをraw seriesから再計算する。historical v8/v9を含む旧result schemasは不変のまま読み込み、新fieldのbackfillを拒否する。
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

`forecasts/prequential.py` owns causal prequential forecast packets for a downstream allocator. `PrequentialSchedule` fixes an expanding-window refit calendar (`first_fit_time`, `refit_interval_days`, optional `activation_delay_hours`); `PrequentialForecastSource` lazily fits one frozen vintage per scheduled cutoff through the existing `build_causal_forecast_training_set` and a small `PrequentialForecastFitter` protocol (`RidgeForecastFitter`; optional `LightGBMForecastFitter`), and serves `ForecastPacket(symbol, horizon_hours, decision_time, log point_forecast, vintage residual_variance, vintage_id, fit_time, fit_cutoff, max_training_label_available_at, digests)`. `ForecastPacket` enforces `max_training_label_available_at < fit_time <= forecast_available_at <= decision_time` at construction; the vintage residual variance is the sample-weighted in-sample squared residual of the training rows, so a caller can convert log to simple expectation with `expm1(mu + 0.5 * s2)`. The source builds the causal training set itself and hands a fitter only that frozen `CausalForecastTrainingSet` (never the `MarketDataset`), so a fitter cannot read rows after the cutoff; training residuals and every served point forecast go through the same `fitter.predict_rows` path and `model.predict` must agree with it. The module exposes the immutable dataset symbol ordering through `PrequentialForecastSource.symbols`, reads only the decision-row features and cutoff-bounded training rows, owns no account, ledger, or order logic, does not import evaluation, simulation or risk, and persists nothing. The development-only `PrequentialSimpleReturnSource` adapter requires `snapshot.symbols` to match that source ordering exactly **before** requesting packets; reordered, missing, foreign or extra symbols fail closed without recording a decision or treating another asset's forecast as this asset's return. Each non-null response must be a validated `ForecastPacket` with the exact requested symbol, symbol index, decision timestamp and source horizon. Invalid packet metadata or overflowing conversion fails before any `calls`, `null_packets` or `history` change, without authenticating the independent market arrival or fit provenance. Unusable decision features yield `None` (caller treats it as no forecast); a non-dataset decision time, an unavailable decision row, or a vintage whose labels were not strictly earlier than its fit time raises.

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

`tools/issue810_portfolio_probability_probe.py` is opt-in repository-only tooling. It accepts an already-loaded PPO model, one transformed observation, and caller-declared symbol names; it lazily imports the pinned `stable-baselines3 2.3.2` and `torch 2.4.1` runtime only when invoked. It reports per-coordinate action probabilities but does not verify Dataset, transform, or checkpoint provenance. It does not load checkpoints, fit, train, replay, or provide economic or live-readiness authority. This helper stays outside the production wheel and `trade_rl` runtime.

`tools/issue810_joint_fee_only_fixture.py` is repository-only deterministic input tooling for one fixed synthetic joint-portfolio Dataset. `build_joint_fee_only_fixture()` returns an in-memory content-identified `MarketDataset` and a read-only recipe declaration bound with `MarketDataset.with_content_identity`. The Dataset has zero fee, spread, funding, and borrow arrays; its fixed volume and market-participation metadata describe only synthetic capacity. A later execution-cost configuration and experiment harness remain caller-owned. The helper does not load market artifacts, write files, create or step an environment, load or fit models, replay, or establish economic evidence or authorization. It remains outside the production wheel and `trade_rl` runtime.

### Agent Coordination Plane

`tools/agent_repo/coordination/` は複数AgentのTask分離・ownership・dependency・semantic collision・review/verification freshnessを扱うrepository toolingである。既存Control PlaneがRepositoryのsource-derived factを読むのに対し、Coordination Planeはそのfactを使って「誰が何を同時に実行できるか」を管理する。どちらも`trade_rl` runtime/domain authorityではなく、**production wheelへ混入させない**。

Coordinationの耐久契約は次である。

- **Task Packet** はtask id/revision、execution mode、dependency、write scope、capability、Acceptance Criteria/Test Oracle、base SHA、**Resource Key**を持ち、semantic contractをcanonical SHA-256 digestへbindする。
- stateは単一statusではなく `phase + condition`。phaseはplanned→ready→executing→review→verification→integration→completeを基本線とし、conditionはhealthy/blocked/stale/failed/conflictedを独立に表す。
- `read_only` Taskはimmutable snapshotを共有できるがwritable leaseを持たない。`write` Taskはsingle active ownerとisolated writable stateを必要とする。
- write ownershipは **lease epoch** をfencing tokenにし、reassignmentごとにnew epoch branch/worktreeを使う。expired leaseはbranch/PR/CI/artifact side effectをreconcileしてからresume/reassignする。
- Resource Key namespaceは最低限 `file:` / `authority:` / `identity:` / `schema:` / `workflow:` / `artifact:` / `side-effect:` を区別し、file-disjointでも同一identity/schema/side effectを変更するTaskをhard conflictとして表せる。 `file:` は大文字小文字を区別するrepository相対pathのprefix契約であり、backslash・空component・`.` componentを正規化して比較する。末尾の単独 `*` / `**` は保守的なsubtree指定、rootの `file:*` / `file:**` は全file resourceと競合する（両方read_onlyの既存例外を維持）。途中のwildcard、部分glob、絶対path、drive指定、parent traversal、空またはdot-onlyのpathはTask Packet生成時に拒否する。一般globやfilesystem alias解決ではなく、従来の有効なResource Key文字列・canonical digest・WriteScopeの契約は変更しない。以前受理されていた未対応patternは明示的な相対prefixへ修正してから再登録する。
- review/verification evidenceはTask revision + Task Contract digest + base + exact PR HEADにbindする。HEAD movement後の古いreview/CIをvalidにしない。integration evidenceはcurrent `main`にもbindする。
- durable coordination evidenceはIssue contract/status record、branch/commit、PR/review、CI/artifact、main commitに置き、ready-set/conflict graph/dashboardは再構築可能なderived stateとする。checked-in mutable global task ledgerは置かない。
- `python -m tools.agent_repo task ...` はnetwork-freeなparse/digest/readiness/status/dashboard操作だけを持つ。GitHub claim/comment/mergeのwrite authorityはこのCLIへ埋め込まない。

このlayerは複数Coordinator間のdistributed consensusを保証しない。v1のoperational invariantはactive Coordinatorが1つであることとし、Worker自身の完了主張やlabelだけをmerge authorityにしない。

## Forward study FS1 boundary

`evaluation/forward_study/` は Issue #810 の sealed forward study FS1 だけを所有する。`registry.py` はFS1-A/B/C + CASHのroster・scoring window・falsifier・validity guardの凍結値と `FROZEN_PARAMS_DIGEST`（JSON exportは `report/issue810-fs1-registry-20261030.json`）だけを持ち、dataset・clock・resultを読まない。`labels.py` は数値だけを受け取る純粋な `FORWARD_PASS` / `FORWARD_FAIL` / `FORWARD_INVALID` ruleで、`WINNER` を返さない。`seal.py` はscoring前提条件（scoring end date経過、Issue commentのauthorization reference、凍結params digest、checkout済みcode SHAとclean tree、dataset seal manifest、読み込んだdatasetのidentity・window・tradable割合）を全て集めて `Fs1Refusal` で拒否し、満たしたときだけ `ScoringPermit` を返す。`fs1.py` はpermitを必須とする唯一のscoring entrypointで、Phase 3 policy builderと `dev_realdata_study.runner` を変更せず再利用し、write-onceでresultを書く。`fs1.py` の公開 `run_fs1` はregistry・digest・clock・code identity・dataset loaderを引数に取らず、private factory `_environment`（frozen registry、実wall clock、`git`）だけがそれらを供給する（testsはこのfactoryをmonkeypatchする）。code identityは `git rev-parse --show-toplevel` が解決したtrade_rl repository（pinned root commit）の追跡済み `seal.py` であること、HEADがsealed SHAに一致すること、変更済み・未追跡fileがないことを要求する。scoringはscoring end + 24h settlement marginの後だけで、manifestはfetch時刻とpinned build-policy digestを持つ。`cli.py` (`python -m trade_rl.evaluation.forward_study`) は引数をそのまま渡すだけでseamを持たない。

このpackageはnetworkへ出ず（`integrations` を import しない）、データを取得せず、`final_test` authorization chain・paper/live・RLを使わず、`trade_rl.evaluation` の公開APIを拡張しない。`dev_realdata_study`・`strategies`・`simulation`・`data` はこのpackageへ依存しない。account・P&L ledger・fill modelは持たない。実データのscoringはscoring end date後にuserが取得とscoring runを別々に認可するまで実行しない。Contractの詳細は `architecture/research-assurance.md` の FS1 section、状態は `research/current-status.md`。

### Forward study FS1 acquisition tooling

`evaluation/forward_acquisition/` はFS1のdata acquisition・dataset build・seal manifest・seal-record textを一つの文書化されたcommand（`python -m trade_rl.evaluation.forward_acquisition acquire` / `seal-record`）として提供し、ad-hocなscriptやmonkeypatchでFS1 datasetを作らせない。`forward_study/` はnetworkを一切importしないため、acquisitionは別packageに置く。`plan.py` はFS1 span・archive URL・date range・build policy照合・出力layoutを持つ純粋moduleで `trade_rl.integrations` をimportしない。`acquire.py` の `run_acquire` は `--allow-network` が無ければplanだけを返し（network clientをimportせず、fileを作らず、development inputも読まない）、ある場合だけ（Issue #810 authorization URL、新規output root、settlement margin経過、clean checkout、developmentのidentity検証が揃ったときに）`trade_rl.integrations.binance` をlazy importしてfetchする。`run_acquire` はregistry・digest・clock・code identity・transportを引数に取らず、private factory `_run_acquire` / `_environment` だけがそれらを供給する（testsが置換する）。development期間のarchiveは再取得せず、published development Datasetのidentity（pinned `dataset_id`・artifact digest・内部にbindされた `raw_source_roster_digest`）に対してread-onlyで検証して再利用する。manifest（`manifest_bytes_from_dataset`）は最後に書き、receiptがbuilder commit・exact command・exchange snapshot・development reuse identityを持つ。`seal_record.py` はmanifestとreceiptを凍結registryと相互に検証してseal commentのMarkdownを返すだけで、投稿しない。このpackageはscoring・replay・strategyに依存せず、`forward_study` からは `registry` と `seal` だけをimportする。それ以外のmodule scope依存は `trade_rl.artifacts`、`trade_rl.data`（`data.artifacts.publication`・`data.market`）とnumpyで、`trade_rl.integrations.binance`・`trade_rl.data.build` は認可された経路（`acquire.py` の関数内）でだけlazy importする。output rootはcheckout外でなければならず、`forward_study`・`dev_realdata_study`・`strategies`・`simulation`・`data`・`integrations` から参照されない。手順の正本は `architecture/research-assurance.md` の *FS1 acquisition procedure*、状態は `research/current-status.md`。

## Signal diagnosis boundary (Issue #1146 SD0)

`evaluation/signal_diagnosis/` is the result-blind, **protocol-only** home of the signal diagnosis that precedes any new learner after #1025 (planned, not executed). `protocol.py` owns the frozen `signal_diagnosis_protocol_v4` (v1, v2 and v3 are not loadable and none produced a result): typed immutable dataclasses, canonical JSON via `trade_rl.artifacts.canonical`, a SHA-256 digest, `protocol_from_payload` / `load_signal_diagnosis_protocol(raw, expected_digest=...)` (digest over the raw bytes first, then strict duplicate-free finite canonical JSON, exact field sets at every level, and equality with the frozen reference of the declared version so a changed grid, threshold or window needs a new version), `reject_forbidden_window`, and the run-admission pair `SignalDiagnosisRunRequest` / `admit_run`. A cross-reference invariant makes a start-up partial year unrepresentable as a full stability year: every full year must start at or after the first usable fit and end inside the window, and the full and partial years together must be exactly the calendar years the out-of-sample span touches. `SignificanceSpec` freezes the decisive p-value procedure (the one-sided equal-weighted cosine Student-t test of a mean daily rank-IC increment above zero with the integer rule `125 nu^3 <= 8 n^2` for its cosine terms, a variance-estimate span of `n / nu` days that must cover the label overlap for every series length, p exactly 1.0 for fewer than four eligible days, and one Benjamini-Hochberg family of the 16 decision cells per target). `verdict.py` owns the pure section-5 rule `classify_signal_diagnosis` (INVALID / NO_SIGNAL / SIGNAL_NOT_TRADABLE / SIGNAL_CANDIDATE) over saved decision-cell evidence: a per-year `YearStabilityEvidence` roster (covered months and sign) and raw one-sided p-values with a `PValueProvenance` (procedure id and the eligible days behind each p-value) that must name the frozen procedure; a p-value below 1.0 for a series of fewer than four days, and a day count above the evaluation window's calendar days (`max_series_days`, 2038), are refused with a typed error. The verdict computes the BH adjustment itself over exactly the frozen family and has no field for an already adjusted p-value; it refuses foreign provenance. Every threshold is read from the protocol. The protocol payload exported as `report/issue1146-sd0-signal-diagnosis-protocol-v4-20261111.json` is the canonical bytes whose digest is `1f48b065b03bc38299a751bef89456b57d5afc6a7d8b917668b1b461f852d0b3`.

The package reads no dataset, market value or file, fits nothing, imports no network, process or filesystem module and owns no runner, CLI or `__main__` (the runner is the separate `signal_diagnosis_run` package); its only `trade_rl` imports are `trade_rl.artifacts.canonical`, `trade_rl.evaluation.forward_study.registry` (protocol) and its own `protocol` (verdict). The FS1 registry is the single evaluation-side owner of the warm-up start and sealed start; the market-data store pins the same instants for its own embargo and may not be imported from `evaluation`, so `tests/evaluation/test_signal_diagnosis_protocol.py` asserts the equality instead of copying the dates. Dataset id, artifact digest and arrays digest are pinned copies of the consumed development Dataset identity, cross-checked by tests against `dev_realdata_study`. The data is known-used and post hoc, so the protocol has no field for an edge, G4, G5 or live claim and the loader refuses one. `real_development` admission needs a separate lowercase SHA-256 authorization digest bound to the protocol digest; that digest records that an owner G4 lease exists and is not that lease. Only `signal_diagnosis_run` (the SD2 runner) imports it. `evaluation.__all__` does not export it. The G0-G2 contract is in `architecture/research-assurance.md` and the status in `research/current-status.md`.

## Signal diagnosis run boundary (Issue #1146 SD2)

`evaluation/signal_diagnosis_run/` is the **single-run runner** of the frozen `signal_diagnosis_protocol_v4`. It owns the filesystem and Git-process boundary that `signal_diagnosis` and `signal_measurement` deliberately lack: `panel.py` (the dense `SignalPanel` of a verified Dataset, the `DatasetIdentity` pins and `admit_panel`, which refuses any bar closing before the window start or after its end as FS1 warm-up, sealed or unused data, a wrong roster and an irregular grid), `fitting.py` (labels, `PrequentialSchedule` vintages, the array reproduction of the shared causal training-set builder, per-(symbol, feature) train-window standardise-and-clip, the purged forward-chaining inner CV from moments pinned equal to the shared Ridge fit, the shared Ridge and LightGBM fitters, the expanding-mean drift baseline), `measure.py` (decision-cell statistics with the SD1 tools, D4, D0/D1/D2/D5 reported layers and a vectorised grouped IC pinned equal to `daily_ic`), `leakage.py` (the future-perturbation probe of every maintained feature calculation on seeded synthetic bars and a label-copy screen of the Dataset features), `runner.py` (`run_signal_diagnosis`, the G4 authorization comment URL and its digest, code provenance, outputs that are written once) and `cli.py` / `__main__.py`.

The CLI reads and admits `manifest.json` before opening `arrays.npz`, and thus before the loader parses any array: `admit_artifact_manifest` (pure, in `panel.py`) requires the manifest to hash to its own `artifact_digest`, the pinned artifact, arrays, Dataset and feature-config digests, the roster, a continuous hourly grid and a bar count that fits the protocol window, and the CLI requires the pinned SHA-256 of the arrays bytes. A forbidden, malformed, foreign-roster or mismatched-pin manifest causes **zero arrays-file opens**; spies on the legacy `Path.open` and current descriptor-safe opener cover this stronger refusal boundary. The arrays file is opened once only after manifest admission, then its payload SHA-256 is checked against the pin and the Dataset is decoded from exactly those bytes with `data.artifacts.verified_bytes.load_dataset_from_verified_bytes` (no second path resolution, so a directory swapped after verification is never parsed). A payload-byte digest mismatch necessarily reads arrays once but never decodes them; spy and race-injection tests distinguish file reads from decoder calls. The initial manifest and arrays byte snapshots now use the existing `artifacts.verified_file.open_regular_binary`: on POSIX with `O_NOFOLLOW` and directory-fd support, the opened leaf and ancestor chain cannot be redirected by a concurrent symlink replacement before the read; `fstat` enforces regular-file type. The SD2 CLI **refuses** a platform missing POSIX descriptor-relative `O_NOFOLLOW`
and `O_DIRECTORY` support before any artifact file is opened; it does not use
the shared opener's weaker portable fallback. Both initial file snapshots are
bounded by technical budgets: manifest 4 MiB, arrays 1 GiB, `fstat` before
read, and a cumulative at-most-1-MiB chunk loop even under file growth.
An admitted manifest with wrong payload digest may still read bounded raw
array bytes once before rejection. This is no source-authenticity attestation,
and hardlinked/in-place-mutated regular files or swaps between the separate
manifest and arrays path resolutions are not excluded by no-follow alone. Admission then happens before any statistic: a real run needs a G4 authorization URL (`https://github.com/shuntatsu/trade_rl/{issues,pull}/<n>#issuecomment-<id>`; the runner cannot authenticate its author, the same limit as the trend-stop runner) whose digest is bound to the protocol digest, a clean Git working tree (a dirty tree is refused and the record names the HEAD, the `trade_rl/` tree id and the digest of the loaded modules), the frozen plan, the pinned protocol digest, the development Dataset identity (a mismatch is an INVALID verdict with no computation) and a panel inside the protocol window and roster. A synthetic run carries no URL, refuses the development identity and may only widen the refit interval or skip the reported-only layers. It is a **single run**: the protocol assigns D0, D1, D2 and part of D5 to card SD2 and D3, D4 to card SD3, but its decisive cells are family-by-model cells that need the fits and its budget is one run, so one invocation computes every layer. D4 is evaluated only for a cell that clears both Benjamini-Hochberg gates and the year rule (the protocol's `stop_when_upper_layer_stops`); every cell is saved, a failing cell is saved with its error and halts the run without a verdict, and the verdict comes only from `classify_signal_diagnosis`. Every seed (bootstrap 0, LightGBM `random_state` 0) is read from the protocol, the output holds no clock value and the same panel, protocol and code give byte-identical files.

The package imports numpy and the standard library, the SD0 and SD1 packages, `strategies.forecasts`, `data` (the path-free published-artifact byte decoder) and, in `leakage.py` only, `integrations.binance.dataset` and `data.features` for the maintained feature contract; it imports no network client, store, FS1 acquisition, final-test or forward-study module, and no other package imports it (`evaluation.__all__` does not export it). It reads no market value in the repository: a real run reads the published development Dataset passed on the command line and nothing else. LightGBM is the optional `forecast-gbm` extra; the CLI halts before reading data if it is missing. The G0-G2 contract and the author packet are in `architecture/research-assurance.md`, the run record and file layout in `specs/signal-diagnosis-run.md`.

## Market-data store runbook

`integrations/market_store/` is a reusable, generic PostgreSQL store of public Binance market data (Issue #810). It is an adapter beside `integrations/binance/`: it depends only on `data`, `artifacts` and `integrations.binance`, never on strategies, evaluation, simulation or risk, and it is not re-exported from `trade_rl.integrations`. It stores USD-M perpetual and spot 1h klines and USD-M funding rates for a configurable universe (default BTC, ETH, BNB, XRP, ADA, SOL against USDT), reuses the Vision URL planner (`plan_vision_kline_urls`, `vision_funding_url`) and the unchanged `build_binance_market_dataset`, and owns no strategy, execution or P&L semantics. `psycopg` is imported only by `store.py` and is the optional `store` extra (also in `dev`); `plan` and a dry-run `ingest` need neither PostgreSQL nor the extra, and only `fetch.py` imports a network client.

Schema (plain SQL, `migrations/NNNN_name.sql`, applied in order with a `schema_version` ledger that records each file's SHA-256 and refuses an edited or unknown applied migration): `instruments`, `exchange_info_snapshots` + `instrument_rule_snapshots` (current-snapshot rules valid from retrieval, not historical rules), `sources` (one row per archive: URL, kind, period, archive sha256, the `.CHECKSUM` line digest, the sha256 of the `.CHECKSUM` file, size, HTTP status, fetched_at), `ingestion_runs` (code commit, command line, counts, status, FS1 warm-up flag and authorization URL), `klines`, `funding_rates` and `ingestion_conflicts`. Every price, quantity and rate is exact `NUMERIC`; natural keys are primary keys; CHECK constraints enforce `high >= max(open, close)`, `low <= min(open, close)` and non-negative volume. `klines`, `funding_rates`, `sources` (once ingested) and the exchange snapshots are append-only (triggers reject UPDATE and DELETE). TRUNCATE is also blocked on every evidence table, and a 404 `sources` record may only gain its evidence, never change its identity. What is committed outside the per-archive transaction: before the run, `ingest --allow-network` snapshots `exchangeInfo` (`exchange_info_snapshots`, `instrument_rule_snapshots`, `instruments`, `run_id` NULL); during the run, `instruments` rows and 404 `missing` source markers are committed as they occur, plus the `ingestion_runs` audit row; none of these holds a price, a funding rate or a boundary state. Re-ingesting an archive is a no-op; a changed value for an existing key rolls the archive back, is written to `ingestion_conflicts`, fails the run and is never applied.

Known exchange placeholder (spot 1h, 2020-12-21 14:00 UTC). The public spot monthly archives for 2020-12 (verified for BTC, ETH, BNB, XRP, ADA and SOL against USDT) each contain one zero-volume placeholder row for the bar opening `1608559200000` (2020-12-21 14:00 UTC): open = high = low = close, volume, quote volume, taker volumes and trades all 0, and a `close_time` about 12.6 minutes **before** the open (about 13:47:20 UTC, different per symbol), which the generic close-after-open check rejects. `parse.py` recognises exactly that shape only for the current allowlist (`is_known_exchange_placeholder`: BTC/ETH/BNB/XRP/ADA/SOL/LTC/AVAX/DOGE/DOT/LINK against USDT, spot market, `1h`, that open time, a flat positive price, all volumes and trades zero, and a `close_time` strictly before the open by less than one bar) and nothing else. The row passes the embargo, period, alignment and ordering checks first, is then skipped rather than stored, and is returned in `ParsedKlines.placeholders`; ingestion counts it in `RunCounts.placeholder_rows_skipped` and lists it (URL, open/close time, price) in `IngestReport.skipped_placeholders`, so the ingest CLI output shows it. Any other malformed row, including the same shape for an unverified symbol or at another bar or market, a nonzero volume or trade count, a non-flat candle or a `close_time` that is not just before the open, still raises `ArchiveContentError` and aborts the archive before any write. The archive checksum, `sources.row_count` (rows actually stored, so one fewer than the archive's data rows) and the append-only tables are unchanged; the skip is not persisted in a database column (no migration), so the audit trail is the checksum-pinned archive plus the run report. The store therefore holds no bar for that hour: the export keeps the gap (nothing is filled) and the unchanged Dataset builder's gap policy decides what happens. Related but not skipped under the historical parser: rows with zero volume whose `close_time` falls after the open (for example the six spot 2021-02 archives' bar at 2021-02-11 03:00 UTC) passed the historical validation and were admitted as published. The new native-clock ingestion policy below refuses such rows when their native close does not match the exact interval endpoint.
The separate legacy/default file-backed `BinanceMarketDataSource` path in `integrations/binance/dataset.py` is **not row-identical** to the market store for this event: its `_parse_kline_rows` derives a close timestamp as `open_time + interval` without examining the archive `close_time` field, and currently retains the same zero-volume placeholder as a regular candle. The market store instead skips it and exposes the missing-hour gap. The characterized mismatch is pinned in `test_file_backed_reader_currently_retains_the_store_skipped_placeholder`; it is a known source-path difference, not evidence that either historical Dataset was retroactively corrected. Do not assert cross-path row/count parity here or silently change sealed/historical data. With the default `reconstructed_v1` source-clock policy, both file-backed `STRICT` and opt-in `NON_TRADABLE_ROWS` admit that archive row as present and tradable when surrounded by a complete synthetic grid; the latter reports zero missing bars. An authentic six-archive audit ([Issue #1153 evidence](https://github.com/shuntatsu/trade_rl/issues/1153#issuecomment-7157770114)) found four additional absent hours per archive (2020-12-21 15:00–17:00 and 2020-12-25 02:00 UTC): 740 source rows per 744 calendar hours, 739 admitted store rows, and legacy `STRICT` refuses the incomplete month. This explicit source-clock divergence is pinned in `test_file_backed_gap_policies_do_not_mask_store_skipped_placeholder` using identical input rows, not asserted as desired future policy. Full source-path harmonization still requires matching row-admission and source-clock rules in a versioned new Dataset identity rather than redefining historical evidence. The historical file-backed ingestion default and every published Dataset byte/ID remain unchanged. New opt-in `strict_source_close_v2` on `BinanceMarketDataSource` and `build_binance_market_dataset` validates the actual source `close_time` (`row[6]`) against `open_time + interval_ms - 1` before admission under both gap policies. A mismatch, including the Dec-2020 known placeholder, is refused rather than silently treated as tradable or skipped as a store placeholder. The `BaseIntervalAggregatingTransport` can validate each source 1h row before producing a higher timeframe; a v2 consumer refuses a legacy aggregator that never checked its underlying source. A successful v2 Dataset binds `binance_source_close_policy=strict_source_close_v2` into `metadata_evidence` and a new `dataset_id`, even when all valid values match v1. Contradictory caller metadata fails closed. In strict v2, the source-open and close-time fields are compared as *exact native integer timestamps* in the same unit: ordinary milliseconds or microseconds (including Binance SPOT public archives since 2025-01-01, and REST when that unit was explicitly requested). A one-microsecond deviation from the expected microsecond close, mixed units, or unsupported native precision is refused; the legacy default still uses the previous timestamp normalizer. The source's and aggregating transport's versioned policies are read-only after construction, so cached series cannot be re-labelled as strict v2. For non-1h requests, the v2 consumer accepts the exact maintained native `BinancePublicTransport` or exact directly supplied `BaseIntervalAggregatingTransport` that validated its raw components using v2 (not subclasses overriding `load_klines`); when that aggregator's base interval is not 1h, its inner source must itself be the exact maintained `BinancePublicTransport`, since an opaque 4h native-looking base may already have been synthesized from unchecked 1h inputs, and rejects unknown opaque wrappers that might synthesize a believable 4h/1d close-time before raw validation, even when the declared Dataset interval is itself 4h/1d. One-hour custom transports remain admissible for explicit raw-input testing, not as proof of original exchange bytes. A custom transport directly supplying its nominal native timeframe is still trusted to supply *raw* rows: the endpoint check alone does not prove that its returned bytes are unmodified venue evidence, nor that the bar was publicly available at the claimed timestamp. Authentic archive byte checks and parser-level row counts are established for those six archives; they are not a PostgreSQL export or historical Dataset certification. End-to-end database provenance and point-in-time parity across all supported source paths remain unproven; Issue #1153 remains open. The market-store exception is restricted to the source-level allowlist: the original six symbols plus LTC, AVAX, DOGE, DOT and LINK against USDT (the later result-blind coverage additions recorded in the source). Symbols outside that roster keep the ordinary malformed-close-time refusal. Existing published Dataset bytes are not rewritten.

New opt-in `strict_spot_placeholder_v3` is deliberately restricted to direct spot 1h Vision input (explicit `VISION` mode). It shares the store's exact documented allowlist and Decimal placeholder predicate, validates native clock units, in-range order, uniqueness, alignment, OHLC and volumes, and skips only the known placeholder with canonical millisecond fields. The maintained Vision transport skips only the exact ordered kline header. Other malformed native rows fail closed; arbitrary raw test transports do not attest exchange byte provenance. `STRICT` still refuses any incomplete clock; `NON_TRADABLE_ROWS` records the skipped placeholder as a missing base bar and adds `source_placeholders_skipped` to the gap report. Missing base rows remain non-tradable and have no fresh information or funding; previously observed features may carry only within the existing feature staleness bounds (two hours for default 1h OHLCV features). Nonempty spot funding events, REST/AUTO, non-spot, higher-timeframe/aggregation requests and overriding public-transport subclasses are refused. The new policy is bound into `metadata_evidence` and the new Dataset ID; the historical v1 and v2 admission behavior, market-store export identity, schema and frozen evidence remain unchanged. Synthetic cross-path tests compare the full Dataset identity arrays from admitted store-parser rows and v3 while requiring distinct policy-bound identities; the same checksum-backed synthetic fixture also exercises ingestion, store readback and export against the in-memory and real PostgreSQL test adapters. This does not certify the lineage of actual stored rows or authorize research.

New store ingestion uses `StoreKlineAdmission.STRICT_NATIVE_V2` (`store_native_close_v2`) for each newly parsed kline archive. It shares the file strict-v2 native-clock validator: the original ASCII integer open and close fields must represent the same millisecond or microsecond unit and the exact native interval endpoint (one native tick before the next open). Only the exact ordered 12-column Vision kline header is skipped under strict admission; a nonnumeric, signed or otherwise malformed first raw timestamp is validated as data, not silently discarded as metadata. Headerless archives remain supported. Sub-millisecond deviations, mixed units, unsupported precision and truncated regular bars are refused before any source or kline write; checksum verification and all existing embargo, period, alignment, ordering and Decimal validations remain in force. The documented 2020 spot placeholder is the one deliberately different admission outcome: the existing allowlist/shape checks still skip and report it, now also requiring the original canonical millisecond timestamp representation. No new zero-volume exception is added. Historical direct parser callers retain `normalized_close_after_open_v1` semantics for reproduction; new `run_ingestion` calls cannot select that legacy policy. The immutable ingestion-run request records `new_kline_admission=store_native_close_v2` for newly parsed archives, not an attestation of all stored rows. Already ingested sources (including identical-byte refetches) are skipped without changing their original run; identical rows from overlapping archives retain their original source association. The existing export receipt/Dataset identity and schema are unchanged and do not attest raw native-clock validation for historical or mixed-lineage stored rows. In particular, normalized millisecond equality cannot establish that a legacy microsecond source was originally valid. Future strict store-backed Dataset certification needs row-specific source provenance and a new explicitly versioned export identity; it cannot be inferred from the latest ingestion run. This software admission change does not re-ingest, rewrite, or validate frozen market data or authorize any research.

FS1 data-protection boundaries. The FS1 forward study controls two spans (constants pinned by tests against the FS1 registry `warmup_only_start` / `scoring_start` and against the SQL functions of the migration). (1) The **warm-up span** `[2026-08-01T00:00Z, 2026-11-01T00:00Z)` may, per *FS1 acquisition procedure* in `research-assurance.md`, be fetched only with a separate Issue #810 data-fetch authorization, so the store refuses it **by default**; a run may admit it only with `--allow-fs1-warmup-span` together with `--fs1-warmup-authorization-url` naming that Issue #810 comment (both or neither). The flag, the URL and the run are recorded in `ingestion_runs`, the database triggers admit warm-up rows only for that still-running authorized run, and the URL is an attestation of the user's authorization, not a verified one. The boundaries are module constants (`FS1_WARMUP_START`, `SEALED_WINDOW_START`), not parameters of any public constructor, planner, ingester or exporter, so no caller can move them. Even then FS1 scoring does not use the store: it re-acquires and re-verifies its own data through its sealed tool (`evaluation/forward_acquisition`), and store rows never feed FS1 scoring. (2) The **sealed window** `>= 2026-11-01T00:00Z` (scored once after 2027-05-02) is refused **unconditionally**: there is no flag, parameter, table or SQL function that lifts it, and no sealed or warm-up row, authorization or release state is ever stored by a failed or empty run. Lifting it after FS1 is scored needs a future reviewed migration (replacing `market_sealed_from()`) and pull request. Both spans are enforced at five layers, and the decision is always made from the period parsed out of the Vision URL itself (`archive_url.py`), never from caller-declared metadata: `ingest` refuses any `ArchiveSpec` whose kind, market, symbol, interval, cadence or period disagrees with its URL before a run starts or anything is requested; `VisionFetcher.fetch` independently refuses, before any network call, every URL that is not a recognised Vision archive or `.CHECKSUM` path with a period entirely before 2026-08-01 (a fetcher constructed with the warm-up authorization URL may also take periods entirely before 2026-11-01; none can take a URL touching the sealed window) and never follows redirects; the planner never plans an archive whose period passes a closed span (a straddling monthly archive is replaced by its in-bounds daily archives; a funding month that cannot be published inside the bound is reported and skipped) and a requested range end inside a closed span exits with code 4; every parsed row is checked before any write; and PostgreSQL triggers on `klines`, `funding_rates` and `sources` reject such rows even under a direct INSERT. By default `plan`/`ingest` therefore end at 2026-08-01. The database guard for the warm-up span is an accident guard, not an authorization boundary: it admits warm-up rows only to the backend session that started a still-`running` authorized run no older than 12 hours (a hard-killed run is closed by the next `begin_run`, and an old one stops admitting rows), run identity and authorization columns are immutable and a finished run cannot be reopened; but the session variable `trade_rl.fs1_warmup_run_id` is an ordinary custom setting that any login role may `SET`, any role that may INSERT into `ingestion_runs` can create an authorized-looking run, and PostgreSQL cannot verify that an Issue #810 comment exists. Keep the store's write role with trusted operators; the sealed window has no such path.

Export. The direct `BinanceMarketDataSource` and `build_binance_market_dataset` interfaces default to `STRICT` (including both requested time boundaries), whereas `export_market_dataset` and the market-store `export` CLI deliberately default to `NON_TRADABLE_ROWS`. Callers comparing the two paths must specify a matching gap policy; for the legacy file source the documented 2020 spot placeholder remains a source-clock interpretation difference. Direct spot 1h Vision v3 aligns admission for that exception without changing the existing export identity. `export_market_dataset` serves stored rows through the same transport interface as the Vision file path (Vision column layout from exact values, funding events with the Vision inclusive range, one stored raw `exchangeInfo` applied statically) so the unchanged builder, gap policy, 4h/1d aggregation and identity produce the `MarketDataset`. Digests of every row read, the exchange snapshot and an `fs1_warmup` record (whether any exported bar lies in the warm-up span, the last bar time and the authorizing run ids and Issue #810 URLs) enter the identity as `metadata_evidence` and the export receipt; the same store content and parameters give the same `dataset_id` (pin the rules snapshot with `--rules-snapshot-sha256`), and an existing test establishes a store-backed/file-backed match for its particular fixture with identical admitted archive rows. This is not a blanket same-archive parity guarantee: the legacy file path and store admit different rows for the documented Dec-2020 spot placeholder, and v3's narrower alignment is not a historical or all-timeframe export certification. Any stronger equivalence claim requires matching parsed rows and explicit gap-policy semantics.

The non-tradable base-bar policy has an additional common-clock boundary: the
generic `MarketDatasetBuilder` infers its regular clock from the earliest and
latest **observed** source bars. If every configured symbol is missing the
final requested interval, that clock would otherwise end early and silently
omit the unobserved last row. The Binance Dataset builder therefore refuses
any result whose first/last closed timestamps do not match the full requested
range; it does **not** invent a final price or extend historical Dataset IDs.
When only some symbols lack the last bar, the remaining symbol's close defines
the full clock and the missing slots stay non-tradable with the existing
causal feature masks. This fail-closed boundary does not resolve the separate
2020 spot source-close-time mismatch or authorize old Dataset reinterpretation.

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

The container binds only `127.0.0.1`, keeps data in the named volume `trade_rl_market_store_data` and reads credentials from the git-ignored `deploy/market-store/.env`; the connection URL is read only from `TRADE_RL_MARKET_STORE_URL`. Backup and restore: `docker compose -f deploy/market-store/compose.yml exec -T postgres pg_dump -U market_store -Fc market_store > market_store.dump`, then `pg_restore --clean --if-exists -d market_store` on a fresh container. Integration tests run against a real PostgreSQL only when `TRADE_RL_TEST_POSTGRES_URL` is set (each test uses its own throw-away schema); otherwise they skip. CI runs them in the `Market Store PostgreSQL` job with a `postgres:16.16-bookworm` service container, fails if any of them was skipped, and that job is in the `needs` of the generic independent-review gate, so a red store job blocks the gate. FS1 scoring and acquisition never import the store (architecture test).

The pinned Compose image and real-PostgreSQL CI service use `postgres:16.16-bookworm` (the 2026-11-12 PostgreSQL 16 maintenance release with an explicit Bookworm base). A green fresh-container CI run does **not** validate updating the existing `trade_rl_market_store_data` volume. Before applying this image to any populated local deployment, consult the [16.16 release notes](https://www.postgresql.org/docs/release/16.16/) (including the documented `REINDEX` cases and any earlier release notes applicable to the installed starting version), make and verify a backup, rehearse upgrade and recovery on a copy, and verify catalog/index health, migration-twice/status, row counts and checksums. PostgreSQL 16.x minor updates do not ordinarily need dump/restore, but CI does not replace a populated-volume rollback drill. This source change performs no service restart, populated-volume migration or actual production deployment.

## Cross-asset market context boundary

`integrations/cross_asset/` (Track D, software only) fetches key-less public daily series for world equity indices, US rates, FX/dollar, volatility/credit spreads and commodities, stamps every observation with the UTC instant at which it became knowable, and builds the versioned feature set `cross_asset_features_v1`. It reads no crypto price, computes no correlation, IC or return against crypto, and authorizes no economic experiment: whether this information predicts crypto after costs is decided only by the signal-diagnosis gate (Issue #1146), after review. It depends only on `artifacts`, `data` (timeframe and `MarketDataset`) and numpy; `tests/integrations/cross_asset/test_loader_cli.py` rejects imports of `integrations.binance`, `integrations.market_store`, strategies, evaluation and simulation, and any other production module importing the package. It is not re-exported from `trade_rl.integrations`.

### Sources (key-less, nothing behind a login)

| Source | URL pattern | Terms |
| --- | --- | --- |
| FRED CSV | `https://fred.stlouisfed.org/graph/fredgraph.csv?id=<ID>` | <https://fred.stlouisfed.org/legal/>: personal, non-commercial, educational use with attribution; series with a third-party copyright notice (S&P 500, NASDAQ, Nikkei, ICE BofA) are personal use only. Federal Reserve Board releases (H.15, H.10) are public domain (<https://www.federalreserve.gov/disclaimer.htm>). |
| Yahoo chart JSON | `https://query1.finance.yahoo.com/v8/finance/chart/<SYMBOL>?period1=631152000&period2=4102444800&interval=1d&events=history` | Unofficial, undocumented endpoint (the one yfinance uses). Terms: <https://legal.yahoo.com/us/en/yahoo/terms/otos/index.html>, not reviewed by the implementer; no redistribution or commercial licence was obtained. `range=max` silently degrades to monthly bars, so an explicit 1990..2100 period is required. |
| Stooq CSV | `https://stooq.com/q/d/l/?s=<SYMBOL>&i=d` | <https://stooq.com/>, terms not reviewed. On 2026-11-10 it served a JavaScript proof-of-work bot check instead of CSV. That is bot detection and is **not** bypassed: `fetch` probes the source once per run, records every Stooq series as `unavailable` and skips the rest. |

Raw data may carry third-party restrictions: it lives only under the git-ignored `data/cross_asset/` and is never committed. Using these inputs in a commercial trading system needs a separate terms review by the owner.

### Point-in-time rules

`pit.py` holds the pure rules (no clock, network or filesystem).

- **Stamp.** `knowable_at = max(session close, end of the publication business day)`. The session close is the venue's local close on the observation date converted to UTC through the IANA zone (daylight saving, including the different US/UK/EU change dates, is resolved by the database); the end of the publication day is 00:00 of the next calendar day in `America/New_York`. No intraday release time is assumed.
- **Publication calendar.** Weekends plus a deliberately conservative union of US federal holidays (observed on Friday/Monday when they fall on a weekend) and Good Friday. Columbus and Veterans Day count as closed although exchanges trade, which can only delay knowability. Ad-hoc closures are not modelled.
- **Lags** (`catalog.py`, never below one business day): FRED daily market series and Yahoo 1 business day; `SP500` 3 (observed three sessions behind on 2026-11-10); `DCOILWTICO`/`DCOILBRENTEU` 6 (EIA, observed about a week behind); H.10 series (`DTWEXBGS`, `DEX*`) are released weekly on Monday, so a value appears in the first Monday release strictly after its observation date (rolled to the next business day) plus a 1 business day buffer. `status` compares the configured lag with the latest published row of each fetch and flags a series that is behind its rule.
- **Session closes** used: New York 16:00 (equities, `NASDAQCOM`, `SP500`), 16:15 (VIX), 17:00 (rates, credit, FX and futures), 12:00 (H.10 noon rates); Tokyo 15:30; London 16:30; Berlin 17:30; Hong Kong 16:00; Shanghai 15:00. A later-than-actual close is conservative.
- **As-of join.** For a decision time `t` (a crypto bar close) a series contributes its latest row with `knowable_at <= t`; rows are only carried forward. Weekend and holiday bars therefore carry the last released value with growing staleness and nothing is back-filled. Staleness is the hours since the session close of the carried row and is exposed per feature; beyond the series' `max_staleness_hours` (10 days daily, 16 days weekly) the feature is unavailable (`available=False`, NaN) and is never extended.
- **In-progress bars.** Rows whose session had not closed at fetch time are dropped at load.
- **Missing values.** FRED `.`/empty and Yahoo null closes are dropped and counted (`missing_rows`); no value is interpolated or forward-filled inside a series.

### Vintages and storage

`fetch` writes exact bytes to `data/cross_asset/raw/<source>/<key>/<UTC stamp>_<sha8>.<ext>` and appends canonical JSON lines to `data/cross_asset/manifest.jsonl` (source URL, fetch/capture time, SHA-256, size, row count, first/last dates, status). The manifest writer refuses symlink/junction ancestors above or within the configured store root (also after creating the root) and symlinked leaf entries; supported POSIX append opens use `O_APPEND|O_CREAT|O_NOFOLLOW` and reject non-regular files before writing, while checked descriptor-based manifest reads reject linked files. Manifest append also requests nonblocking descriptor opening to avoid FIFO waits before the regular-file check, and refuses an existing inode with multiple hard links before modifying bytes, so an outside hardlink alias cannot be appended through the manifest path. These inode/link checks observe the state at open and do not prove atomic immunity against a concurrent hostile hardlink creation. The manifest reader rejects pre-existing FIFOs/devices before descriptor opening so a named pipe with no writer cannot block before its type is checked; the pre-open path type check is static and does not prove protection against a hostile swap after that check. Existing JSON schema, ordering and append-only receipt semantics are unchanged. As with raw publication, parent-path checks are static, not descriptor-atomic against an actively hostile concurrent directory replacement; portable platforms lacking `O_NOFOLLOW` retain only the static leaf check. New fetch/capture stamps retain **six UTC fractional digits**, including `.000000Z` for an exact second, and raw filenames include this precision. An identical refetch records `unchanged` and preserves the original vintage's `captured_at`; changed bytes create a new immutable vintage and revision summary. For raw files, `write_raw` publishes a uniquely staged, fsynced file through an exclusive same-filesystem hard link, not the shared atomic **replacement** primitive: if a second writer claims the same filename, only a matching full SHA-256 may be reused; distinct bytes with a colliding eight-hex filename prefix cause an explicit refusal without changing the first file. Existing raw targets are re-verified by the regular-file/no-follow descriptor digest helper during `write_raw` admission; direct `read_raw` and `verify_raw` now reject absolute/parent-traversing record paths and symlink/junction components (including the configured root, its ancestors and the raw leaf) through the checked regular-file descriptor helper. A matching digest alone never authorizes reading from a linked root. POSIX platforms with `dir_fd`/`O_NOFOLLOW` pin the ancestor chain while opening; other platforms retain static no-follow checks without claiming immunity to concurrent path swaps. Filesystems without exclusive hard-link support fail closed rather than using an overwrite fallback. For raw publication, write_raw checks that the entire existing absolute parent chain contains no symlink, junction or non-directory before path creation, and rechecks after mkdir before private staging; a preexisting raw/root/ancestor symlink and one planted at that creation boundary are refused before writing source bytes outside the requested store. This is a **static ancestor-path fence** plus exclusive-file write semantics, not descriptor-pinned immunity against an attacker who swaps a directory after the last check, nor complete power-loss durability. On POSIX filesystems where directory `fsync` explicitly reports `EINVAL`/`ENOTSUP`/`EOPNOTSUPP`, the staged file contents have been fsynced and the exclusive hard link already exists, but directory metadata durability cannot be asserted; unexpected directory I/O errors still propagate. A directory `fsync` failure occurs after the exclusive hard link may already have published bytes, so callers must check the existing raw file and digest on retry rather than interpreting the exception as an atomic rollback. Concurrent identical-byte claimants can both reuse the single published file after full-digest verification. `load_cross_asset_features(..., as_of=...)` selects vintages by parsed timezone-aware **microsecond-exact** timestamps, not lexicographically sorted/truncated seconds. Previously stored stamps without a decimal point were truncated by legacy code and are **not** proof of capture at the start of the second: `as_of` admits them conservatively only from the *next* whole second. This preserves historical raw files and manifests rather than retroactively inventing precision; legacy mixed-vintage order inside one second remains unknowable. The input provenance digest `cross_asset_input_provenance_v2` binds raw SHA-256, original captured timestamp and the conservative second-only handling rule; unlike the former digest, two same-byte captures at different times have distinct provenance. CLI `as_of`/build time evidence also retains microseconds. These changes are prospective code contracts, not a rewrite of existing artifacts or a claim of historical first-release vintages.

**Verified parse snapshot.** Cross-asset observations read each selected raw
file once, verify the SHA-256 of **those exact bytes** against the admitted
manifest entry, and pass the same immutable byte buffer to the source parser.
A verification read followed by an unverified second read is forbidden: a
source file changed between checks must not introduce a new value under the
old manifest digest. Missing or mismatched raw bytes halt the loader; source
verification does not upgrade latest-at-capture values to historical
first-release point-in-time evidence. On refetch, the old raw bytes are
likewise read once and checked against the full recorded SHA-256 *before*
being parsed for a revision summary or reused as an identical vintage.
An unreadable or tampered previous raw is recorded as
`revision={"unreadable_previous":true}`; even an equal incoming digest does
not allow an `unchanged` receipt against unverified old bytes. The new
capture is stored separately and no historical manifest or raw file is rewritten.
This ensures the compared previous values came from one verified snapshot,
but is not a claim of immunity against an arbitrary same-UID writer changing
the original pathname after the snapshot was checked.

### Revision risk and default membership

Each catalog series has `revision_risk` (`low`, `possible` or `unknown`) and a `revision_evidence` note. **Only `low` series produce columns by default**, per-series and composite alike: a `possible` or `unknown` series has no column in the frame unless it is named in `opt_in_series` (CLI `build --opt-in KEY,...`). The frame records the opt-in set and, in `column_sources`, the series behind every column.

**Vintage basis.** Timing is point-in-time, values are not: every frame uses the newest vintage at capture, never first releases (ALFRED first-release vintages are not loaded), so a revision fetched after a historical decision can appear at that decision. This applies to every `revision_risk` tier: `low` is a measured frequency, not first-release verification, and even catalog `low` examples may have historical value revisions. The dataset overlay therefore fails closed: it refuses columns without source provenance, refuses non-`low` sources that were not opted in, and **requires exact built-in Boolean `accept_revision_hindsight=True` for every selected latest-vintage overlay**, including `low` sources. This opt-in permits retrospective exploratory analysis only, not strict historical PIT eligibility or a trading research approval. Identity version `cross_asset_global_overlay_v3` records `vintage_basis=latest_vintage_at_capture_not_first_release`, `strict_point_in_time_verified=false`, every `latest_vintage_series`, `opt_in_series`, `accept_revision_hindsight` and the higher-risk `revision_hindsight_series`. A missing flag must never be treated as implicit consent simply because a series is `low`. An `as_of` snapshot taken long after the decision grid does not make a series first-release correct. Previous overlay schema identities are not rewritten.

A `MarketDataset` accepts **one** cross-asset overlay: callers select all desired `feature_names` in one call. A second overlay would retain the earlier `xa_*` feature columns but silently overwrite their source-vintage identity; this is rejected rather than accepted with missing provenance. The reserved `xa_` global-feature prefix also makes generic `MarketDataset.with_content_identity` and canonical identity rehydration reject payloads without `cross_asset_overlay`: a downstream transform such as `with_price_channels` **after** the overlay would otherwise retain retrospective feature bytes but erase the `strict_point_in_time_verified=false` flag while producing a newly verified Dataset ID. Perform causal price-feature augmentation **before** adding the exploratory cross-asset overlay. Existing native Dataset identities without `xa_*` columns are unaffected. The common Dataset identity boundary now requires the complete `cross_asset_global_overlay_v3` receipt, matching feature-column names, full lowercase source digests, `vintage_basis=latest_vintage_at_capture_not_first_release`, exact Boolean `strict_point_in_time_verified=false` and `accept_revision_hindsight=true`. Once a cross-asset Dataset has an identity, the generic `with_content_identity` API cannot rebind it, even with copied metadata; the explicit overlay constructor uses a zero-ID staging Dataset and a complete validated retrospective receipt for the initial identity; the generic method is not cryptographic source authentication. Existing identities are not rewritten. This is a fail-closed *schema/causal-admission invariant*, not cryptographic authentication of provider values or complete immunity against a malicious caller rebuilding all arrays and evidence.

Legacy manifest rows written before `captured_at` existed are resolved on read: duplicate `ok` rows and `unchanged` rechecks inherit the earliest supported capture time **only within the same series key, immutable raw path and SHA-256**. A matching raw path and digest for a *different series* cannot establish an origin. A legacy `unchanged` row without an origin belonging to that series fails closed (capture time unknown): refetch the series. Existing explicit capture timestamps are preserved. This reading policy does not rewrite the stored manifest or raw bytes.

`low` rests on a measurement, not on the nature of the series. ALFRED (`https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=<ID>&vintage_date=<date>`, key-less) serves the vintage published on a date; on 2026-11-10 the 2023-11-15 vintage was compared with the current values over the dates both contain:

| Series | Changed values | Label |
| --- | --- | --- |
| DGS2, T10Y2Y, DFII10, DEXUSEU, DEXUSUK, DEXCHUS, DEXUSAL, NIKKEI225 | 0 | low |
| DGS10 | 0 of 15454 (1 of 14484 against the 2020-01-02 vintage) | low |
| DEXJPUS | 0 of 13251 (0 of 12284 against 2020-01-02) | low |
| VIXCLS | 8 of 8544 (max 4.2%, last in 2019) | low |
| DTWEXBGS | 1707 of 4479 (max 0.5%, 2018-2023) | possible |
| NASDAQCOM | 996 of 13275 (max 1.4%, mostly 2018-2020) | possible |
| DCOILWTICO, DCOILBRENTEU | 0 changed, but 743 of 9539 and 890 of 9260 earlier values are now missing | possible |
| BAMLH0A0HYM2, BAMLC0A0CM | ICE licensing keeps a rolling three year window (the 2023-11-15 vintage held 4 observations); 0 changed in the 2024-06-03 and 2025-06-02 vintages | possible |
| SP500 | ALFRED returns 404: no vintage to compare | unknown |
| Yahoo and Stooq series | no vintage history exists; only repeated fetches (`status` counts revisions) can show it | unknown |

This is one comparison date, so `low` means "no revision observed", not a guarantee; re-measure with another vintage date before relying on a new series.

Continuous front-month futures (`GC=F`, `SI=F`, `HG=F`, `CL=F`, `BZ=F`) were removed from the catalog: their series jump on contract rolls and back-adjusting them would use future knowledge. No key-less spot gold, silver or copper exists (Yahoo `XAUUSD=X`, `XAGUSD=X`, `XCUUSD=X` answer 404), so oil comes only from FRED (`possible`, hence opt-in). Consequently the default frame has no commodity or equity-breadth composite.

### Yahoo bar dates

A Yahoo daily bar's timestamp is the **session open in the exchange's local time** (09:00 Berlin, 08:00 London, 09:00 Tokyo, 09:30 Hong Kong and Shanghai, 00:00 New York for the dollar index). `parse_yahoo_chart` takes its calendar date in `exchangeTimezoneName`, so the bar is attributed to its own session and never to an earlier one. A live sample spanning the US and EU DST changes is committed as timestamps only (`tests/integrations/cross_asset/fixtures/yahoo_bar_timestamps.json`, dummy closes) and `test_yahoo_bar_timestamps_map_to_their_own_session_never_an_earlier_one` asserts each bar falls after the previous day's close and before its mapped day's close.

### Features

`cross_asset_features_v1` (`features.py`; `definitions_digest()` is the SHA-256 of the full canonical definition: windows, member lists, the series catalog with revision labels, and the default-membership rule). Windows count **observations of the series**, not calendar days, and are trailing only. This module is unrelated to `trade_rl/data/features/cross_asset.py`, which computes BTC-relative features among the crypto symbols of one dataset.

- Per series: `ret|chg{1,5,20,60}` (log return for prices, difference for levels and spreads; NaN for non-positive prices), `rvol20` (sample std of the 1-observation change), `z252` (trailing z-score of log price or level), `sm20`/`sm60` (k-observation change over the trailing-252 std times sqrt(k), clipped at 5), `lvl` for levels and `abv60`/`abv200` (above the trailing moving average) for prices. Only default (`low`) and opted-in series have columns.
- Composites on the decision grid: `dollar_strength`, `risk_appetite`, `equity_momentum20`, `commodity_momentum20`, `commodity_momentum60`, `curve_slope`, `curve_slope_chg20`, `trend_breadth_60`, `trend_breadth_200`, `rotation_equity_minus_commodity`, `rotation_equity_minus_dollar`. Each is the mean of the signed `sm*` (or `abv*`) members that are eligible (default or opted in), present and fresh, with a fixed minimum member count; fewer members than the minimum makes the composite unavailable.
- **Default membership** (`revision_risk == low`): `dollar_strength` = `DEXJPUS`, `DEXCHUS` (+), `DEXUSEU`, `DEXUSUK`, `DEXUSAL` (-), min 3; `curve_slope` and `curve_slope_chg20` = `T10Y2Y`; `risk_appetite` (min 4) can only see `NIKKEI225` (+) and `VIXCLS` (-), `equity_momentum20` (min 3) only `NIKKEI225`, and breadth (min 5) only `NIKKEI225`, so those composites and both rotations are unavailable by default. Opt-in members: `DTWEXBGS`, `y_dxy` (dollar); `SP500`, `NASDAQCOM`, `y_gdaxi`, `y_ftse`, `y_hsi`, `y_ssec` (equity, risk, breadth); `BAMLH0A0HYM2`, `BAMLC0A0CM` (risk); `DCOILWTICO`, `DCOILBRENTEU` (commodity, breadth).
- Normalisation: `FeatureNormalizer.fit(frame, fit_start, fit_end)` uses only available rows with `fit_start <= t < fit_end`; a feature without enough fit data becomes unavailable instead of unscaled.

### CLI and hook

`python -m trade_rl.integrations.cross_asset fetch|build|status` (default root `data/cross_asset`). `build` aligns features to a regular UTC grid (`--timeframe 4h|1d`) and writes `built/cross_asset_features_v1/*.npz|json` for inspection. `load_cross_asset_features(root, decision_times, as_of=..., feature_names=...)` returns the aligned `FeatureFrame` for any UTC grid.

`build` and the loader take `--opt-in`/`opt_in_series` for non-default series.

The opt-in dataset hook is `with_cross_asset_global_features(dataset, frame, feature_names=..., accept_revision_hindsight=True)` in `dataset.py`; the acknowledgement is mandatory for even `low` series because no first-release vintages are loaded. `MarketDataset.global_features` already reaches the `SharedPortfolioGymEnv` observation (`global_features` plus the `global_feature_available` mask) and the portfolio transform fits its own scale on the training window, so no shared feature code changes. The hook returns a new dataset whose original global columns are byte-identical, with `xa_<name>` columns appended (an unavailable entry is 0.0 with `global_feature_available=False` and missing reason 2), and whose v3 identity binds the overlay (feature set, definitions digest, input digest, names, opt-in series, newest-vintage source roster, unverified strict PIT status, hindsight acknowledgement and higher-risk roster, normaliser digest). The acknowledgement is not research or economic execution authorization. Callers that do not use the hook get exactly the previous datasets, pinned by a golden dataset id test and a test that default import paths never load the package.

## Repository integration boundary

GitHub PR / CI / branch-protection・ruleset設定はRepository統合の安全性を管理するが、`trade_rl` runtime packageのauthorityではない。checked-in architecture testはPR/CI policy fileの契約を検証できるが、実際のbranch protection状態はGitHub側のread-backで別途確認する。

Integration invariant: tested PR head contains current `main`. merge直前のcurrent `main` commitがtested PR headのancestorであり、その同一PR HEADにpermanent CI successが存在することを統合証拠とする。`main` が進んだ後の古いPR-head Greenは再利用せず、current `main` を含む新HEADを再検証する。将来merge queueを採用する場合は、current target branchを含むmerge-group SHAのrequired checkを同等の証拠としてよい。

このGit tree内のproseやarchitecture testだけでbranch protectionが有効とは判断しない。ruleset/protectionの設定変更後はGitHub stateをread-backし、required check、PR requirement、force-push/deletion、maintainer/admin bypassを確認する。管理surfaceが利用できない場合は未設定/未検証として扱う。

`Generic Independent Research Review` はhistorical job名を維持するが、Generic software PRではexact HEADの**model source review**を要求する。authenticated GitHub review APIの`COMMENTED`または`APPROVED`に単独marker/disposition/digest付きreceiptを必須とし、`software_source_review_v2`では`reviewer_model_id`・PASS・blocking finding 0・具体的reportをpublisher-attested metadataとして受理する。同じconfigured model（例: GPT-5.6 Sol）を許容し、provider-run provenance・diff digest・実装Task ID・model independenceは要求も主張もしない。旧`software_source_review_v1`は未検証job/diffを含むためPENDINGのまま廃止し、`independent_model_review_v1`は互換経路として残す。正本は [research assurance](research-assurance.md#generic-software-pr-review-receipt)。

同じexact HEADのactive submitted marked `### Disposition: BLOCKED` が一つでもあれば承認より優先し、後続の同一actorによる承認でも消去しない。保留・曖昧・dismissed・古いHEAD・不整合なreceiptはfail-closedに扱う。重大な指摘事項（Medium / High）が解消され全必須CIがGreenであれば、承認済みsoftwareレビューはIntegratorによるマージ・クローズの対象となる。このGeneric receiptは `Independent Research Review` の4h PPO smoke専用review/triggerやG4 economic execution、sealed PPO、unused/final one-shot authorization、live eligibilityを代替しない。

`tools/ppo_4h_gemini_review.py` は4h PPO smokeのresult-blind external-AI reviewをdefault-branch trust rootから実行する**repository-local reviewer transport**であり、`trade_rl` runtime packageやeconomic evaluatorの一部ではない。`.github/workflows/ppo-4h-gemini-review.yml` はcanonical PR comment requestを入口に、(1) targetをdataとして読むpacket生成job、(2) Gemini secretを持たずexact reviewed SHAへ固定command setを実行するCore / PPO Runtime / Human Guide verification jobs、(3) canonical packetだけを受け取るfresh secret-bearing review jobを分離する。review jobはtarget checkoutやtarget-generated executable artifactをauthorityとして受け取らず、trusted runner/workflow identity、request/tag identity、packet digest、trusted verification job identity、Gemini request/response identityをreviewer-run attestationへbindする。Gemini API key/model selectionはworkflow configurationであり、concrete model versionはprovenanceとして記録するがproduction trading/runtime schemaやvalidator allow-listへ固定しない。

## Trend-with-stops study boundary

`strategies/rules/trend_stop.py` は因果的なDonchian breakoutとATR stopを、numericな`PortfolioPolicy`（`TrendStopPortfolioPolicy`）として提供する純粋なstrategy-layer ownerである。fixed rosterの`MarketDataset`の完了barだけからchannel・Wilder ATR・signalを導き、risk-sized target weightとexact-quantity HOLDを提案する。stopはdecision-layerのruleであり、注文・約定・cost・funding・hard risk・accountingは既存のshared portfolio sessionに残る（bar-extreme touch → next-open exit、`bar_extreme_touch_next_open_v1`）。evaluation・simulation・integrationsをimportせず、`trade_rl.strategies` facadeへ再exportしない。

`evaluation/trend_stop_study/` はこのruleのdevelopment-only multi-symbol studyだけを所有する。`protocol.py` は事前固定した全parameter・2 variant・2 timeframe・2 scenario・pass/fail基準のdigestと、G4 authorization URLの形式検査、2026-07-31を超える窓の拒否を持つ。`runner.py` は各cellを既存の`run_shared_portfolio_replay`で1回ずつ実行し、`metrics.py` がraw order eventから閉じたcycle・stop haircut・gateを再構成する。`data.py` はpublished Dataset artifact（Market Store export）とoffline Binance Vision fallbackを検証付きで読み、rosterは固定の11 symbol全てで、除外経路も縮小rosterもない（symbolが欠ける、またはVision cacheのarchiveにHTTP 404の記録があれば`INVALID`、404の記録がない欠落・破損・checksum不一致は結果を書かないresumableな停止）。`runner.py` はまた、実行したcode（Git HEAD、`trade_rl/`のtree id、dirty状態、読み込んだmoduleのdigest）を結果に記録し、dirty treeでは起動を拒否する。`WINNER`を出さず、unused-futureを開かず、account・ledger・fill modelを持たず、networkへ出ない。CLI (`python -m trade_rl.evaluation.trend_stop_study`) は新規出力のみを書き、G4 authorization URLなしでは開始しない。正本は[specs/trend-following-stops.md](../specs/trend-following-stops.md)、状態は`research/current-status.md`。

## Development real-data study boundary

`evaluation/dev_realdata_study/` は Issue #810 P2 の development-only real-data walk-forward study だけを所有する。`protocol.py` は事前固定した全parameter・5 arm roster・3 scenario・selection ruleの定数とSHA-256 digestを持ち、実データ実行の前にcommitする。`runner.py` は `strategies/forecasts/prequential.py` の causal packet を log→simple return に写す adapter、`strategies/allocation/` の policy、armごと・scenarioごとに1回の連続 `run_shared_portfolio_replay`（terminal flat settle、ledger capture）、raw fill からの独立 ledger reconciliation、未来perturbation causality probeを束ねる。`metrics.py` は equity-path metrics、20日block bootstrap、`DEVELOPMENT_PROMISING` / `NO_WINNER` だけを返す選定ruleを持つ。このpackageは `WINNER` を出さず、unused-future窓を開かず、account・P&L ledger・fill modelを持たず、`trade_rl.evaluation` の公開APIを拡張しない。CLI (`python -m trade_rl.evaluation.dev_realdata_study`) は新規出力のみを書き、既存出力を上書きしない。`dataset_build.py` は同study用Datasetをlocalの Binance Vision archiveだけから作る **offline・opt-in** builder（`python -m trade_rl.evaluation.dev_realdata_study.dataset_build`）で、networkへ出ず、非空の出力先を拒否し、economic replayを実行しない。gap policy (`non_tradable_rows`: 欠損1h barはnon-tradable row・forward-fill mark・funding非計上、4h/1d barは完全な1h windowからaggregate、欠損windowは `available_at=range end` のplaceholderにして再帰的な上位clock featureを以降unavailable化) と入力rosterを `DATASET_BUILD_POLICY` とそのdigestとしてreceiptへ固定する。gap許容は `integrations/binance/dataset.py` の明示的な `BinanceGapPolicy` option（default `STRICT` = 従来どおり欠損barを拒否）と `integrations/binance/derived_klines.py` の1h→4h/1d aggregation transportが所有し、monkeypatchは使わない。

Phase 2 trial registry は同じpackage内の `trials.py`（T1-T8の事前固定spec・定数・実装判断、JSON exportは `report/issue810-phase2-trial-registry-20261029.json`）、`trial_sources.py`（`ExpectedReturnSource` registry、causalなTSMOM/XS-momentum/funding carry/blend source、T7のdrawdown overlay policy wrapper）、`trial_study.py`（armごとの隔離実行・FAILED_TO_RUN記録・13 arm Bonferroni・同一selection rule）、`trial_cli.py`、事後ラベル再計算の `trial_labels.py`（PR-C probe ruleとamended ruleの両ラベルをcommit済み証拠からcodeで計算）が所有する。PR-Cの `runner.py` を `arm_builder` / `build_allocator` / 汎用causality probeで拡張して再利用し、複製しない。同じくaccount・P&L ledgerは持たず、`WINNER` を出さない。 今後のreplayはcontrol両armの有効性・probe matrixをselection gateで検査する。経済結果の凍結を維持したまま、途中終了や非対応日時のbootstrap比較を除外理由とともに記録する。CLIは新規の非重複出力先だけを許可し、登録cut以外の因果性検証はdiagnostic-onlyと記録する。T7 overlayはinner allocatorの同時刻fail-closed HOLDを保持し、risk縮小の抑止件数を診断出力する。

Phase 3 roster（turnover aversion `kappa` とex-ante volatility target `VT12`、POST_HOC_EXPLORATORY・事前登録ではない）は `phase3_registry.py`（8 armの凍結roster・公開事前記録のverbatim本文とSHA-256・実装判断notes・JSON exportは `report/issue810-phase3-trial-registry-20261029.json`）、`phase3_sources.py`（T4×7 source、T2/T4 z-blend、vol-scaled passive comparatorのpolicy、arm builder）、`phase3_study.py`（armごとの隔離実行・causality probe/matrix・PR-C probe rule主ラベルと補助のamended rule・Phase 3 family 8とall-arms 21のBonferroni）、`phase3_cli.py`（新規出力のみ・全実行を数える`--execution-ledger`）が所有する。`kappa`（optimizerのL1 cost penalty倍率のみ。実行costは不変）とVT（解いたtargetをmin-trade rule前に`cap/sigma_p`倍へ縮小）は `strategies/allocation/cost_aware.py` のopt-in引数で、defaultでは従来のdecisionとbit-identical。label語彙は `POST_HOC_PROMISING` / `NO_WINNER` で、`WINNER` は出さない。

## Signal measurement boundary (Issue #1146 SD1)

`evaluation/signal_measurement/` holds the **pure measurement tools** of the signal diagnosis (card SD1) and nothing else: `leakage.py` (a future-perturbation probe that rebuilds a feature after perturbing every input row after a cutoff and checks whether any earlier row changes—evidence of future dependence for a deterministic builder, not proof of causality if unchanged—plus a narrow affine/monotone label-copy screen that is not a substitute for the probe), `payoff.py` (the signed after-cost payoff `s*r - |s|*cost` with entry only when `|forecast| > k*cost`, the zero-cost drift-sign comparator, and the drift-excess payoff = signal payoff minus comparator on the same row and cost multiplier, which is **not** a beta-adjusted alpha; checked against a slow independent loop reference in the tests), `ic.py` (Spearman rank IC with average ranks, Pearson linear IC, per-UTC-day IC pooled over the day's rows with skipped days counted, and the daily rank-IC increment of a forecast over a baseline on identical rows), `targets.py` (forward log returns, the cross-sectional demean `r(i,t) - mean_j r(j,t)`, and a causal beta residual), `multitest.py` (Holm and Benjamini-Hochberg adjusted p-values where equality with the level counts as a rejection, and a 95% moving-block bootstrap interval that calls the unchanged `comparison.bootstrap.moving_block_mean_test`), `profile.py` (per-feature, per-period z-scores against the expanding window of strictly earlier periods) and `significance.py` (`ewc_mean_test`, the one-sided equal-weighted cosine Student-t test of a positive mean of a day series that the SD0 protocol freezes as the p-value of the Benjamini-Hochberg gate, its exact integer term rule `ewc_terms` and the Student-t tail; deterministic, no resampling, p exactly 1.0 for a non-positive mean, fewer than four days or a constant series, checked against a slow reference from the definition and calibrated on seeded synthetic series by `tools/issue1146_significance_calibration.py`).

The future-perturbation probe is a seeded **falsifier**, not proof that a feature is causal when no change is detected. It perturbs nonzero future input cells multiplicatively and zero-valued future cells additively; non-finite inputs and future-cell no-ops/overflow fail closed instead of silently certifying causality. These are synthetic software checks, not research/economic authorization.

The cross-sectional demean removes the same-time common level only; it estimates no coefficient and is **not beta removal**, because a market move that loads unequally on the symbols leaves `(beta_i - mean(beta)) * market` behind. The causal beta residual subtracts `beta_i * market` with `beta_i` fitted by least squares on rows whose label matured at or before the decision time, reports missing (`NaN`) instead of filling when there are too few rows, and is reported context rather than a decisive target.

The package imports numpy, the standard library and `comparison.bootstrap` only. A leaking feature (a label copy, a one-step lookahead, a centred average, a full-sample standardisation, a distant-future ratio) is flagged by the probe on synthetic prices while causal features are not. It reads no dataset, file or market value, fits no model, owns no runner, CLI or `__main__`, is not exported from `evaluation.__all__` and is imported only by the SD2 runner `signal_diagnosis_run`. It is verified on seeded **synthetic** markets with known truth (planted-signal recovery against the Gaussian closed form, a day-shuffled-future null demonstrating signal destruction under permutation with documented small-sample bootstrap dispersion, a common-level-only market whose demeaned IC is near zero beside an unequal-loading control that documents the residual, and a tail-only signal whose rank IC and linear IC diverge) and says nothing about real data. It does not import the SD0 protocol package and carries no threshold, grid or decision rule: callers supply block length, resamples, seed and levels. The one rule it holds is the integer cosine-term rule of `ewc_mean_test`, which a protocol test pins equal to the rule `SignificanceSpec` freezes. The SD2 runner that applies it to the consumed development Dataset needs a separate owner G4 lease; this package authorizes none of that.

## Core7 MTF Shadow-Anchor PPO boundary

`data/features/mtf6_signals.py` は `MarketDataset` の完了済み bar（`available_at <= t`）だけから 6 時間軸（15m/1h  proxy または causal external 入力 + 4h/1d/3d/1w）の `CORE7` / `ORTH6` 因子、All-Weather 75/25 アンカー、`vol_cushion_tail`、41 次元因果レジーム状態、6 次元機会状態、および固定因果 Non-RL レジームゲート（`compute_nonrl_regime_gates`）を構築する純粋なデータ層オーナーである。`SPOT_HALTS` 区間を非取引ギャップとしてマスクし前方補間せず、`>= 2026-08-01T00:00:00Z` を fail-closed に拒否し、strategies/evaluation/simulation/risk/integrations や PyTorch に依存しない。

`strategies/rl/shadow_anchor_mtf.py` は純粋 NumPy 決定論的推論の `ShadowAnchorMtfPolicy`（`PortfolioPolicy`）と `CashHoldPortfolioPolicy` を所有し、`anc_bidir_75_25`・`anc_regime_gate_nonrl`・`core7_sparse_ppo`・`orth6_cush_ppo` の 4 モードと交互 `:spot`/`:perp` ターゲット合成を提供する。`fit_stop_index <= start_index` を強制し、リバランス間隔外および deadband 未満時は `hold_current_quantity_symbols` を設定する（ただしドローダウン発動閾値到達時は HOLD を解除し、非リバランスバーでは直近リバランスの**リスク適用前**目標を再提案して `PreTradeRisk` の縮小が一度だけ掛かるようにする。縮小済み現在ウェイトを再投入して縮小を複利化しない。リスク縮小後目標が現在ウェイトから deadband 内なら逆像 `current / risk_scale` を提案して微小再約定を抑える）。`trade_rl.strategies` や `trade_rl.strategies.rl` の公開 `__all__` は拡張せず、`torch` / `stable_baselines3` / `gymnasium` を import しない。

`evaluation/core7_mtf_study/` は凍結プロトコル（`core7_mtf_shadow_anchor_ppo_v4`）・データ非依存判定（20% 最大ドローダウンは `STOP_FAIL_NO_RESCUE`）と Non-RL レジームゲート比較診断（`decision.py`）・因果性摂動プローブ（全決定バー・HOLD タプル・シナリオ遅延ごと）、約定ベース台帳再構成、fail-closed 認可（期待プロトコル digest 必須・Pre-FS1 禁輸は常時強制。`trade_rl` はネットワークを持たないため G4 URL は**形式検査のみ**で、`AuthorizationReceipt`（tools 層が取得・検証して渡す呼出側証明）が URL・コメント ID・プロトコル digest に束縛されている場合に限り証拠適格となり、URL のみは `UNAUTHENTICATED_NOT_EVIDENCE`、URL 無しは合成マーカー付きデータ限定で `SYNTHETIC_ONLY_NOT_EVIDENCE`））および `run_shared_portfolio_replay` 専用ランナー（`runner.py`）を所有する。第 2 のバックテスタや独自台帳・約定モデル・ネットワーク経路を持たず、`DELAY_1BAR` ストレスは `order_latency_bars=0` のまま `signal_lag_bars=1` で実信号遅延を適用し、`processing_bar_volume_capacity=False` により先行完了バーの 1% 出来高制限を維持する。正本仕様は [specs/core7-mtf-shadow-anchor-ppo.md](../specs/core7-mtf-shadow-anchor-ppo.md)。

`evaluation/core7_mtf_study/tierd.py` は Phase 1 Tier D の**データ組立のみ**を所有する純粋モジュールで、store・DB・ネットワーク・replay・収益計算を import しない。時間足 1h の `RawMarketSeries` ペアから、`protocol.ROSTER` 順の 4h・11 ペア・22 交互レッグ `MarketDataset`（`timestamps` は bar close）を組み立て、ロード時に `2026-08-01T00:00:00Z` を fail-closed に拒否する（`GuardedRowSource` は要求範囲を読込**前**に、返却行を読込**時**に検査し、最終 4h バーは close `2026-07-31T20:00Z` まで）。`SPOT_HALTS` は非取引（`tradable=False`・出来高 0）、ロスター欠落・余剰・クロック不一致・funding 欠落・未説明の欠損行は拒否し、データセット側コスト列は 0（`ExecutionCostConfig` と加算されるため）とする。store を読む配線は market-store boundary（`trade_rl` 内の他パッケージは store を import しない）のため `tools/core7_mtf_tierd.py`（`python -m tools.core7_mtf_tierd coverage | dry-check | run`）に置く。`dry-check` と `coverage` は `run_study` を呼ばず収益を計算しない。`run` は owner G4 認可 URL（既定値なし）・clean checkout・新規出力 directory を要求する。`--load-start` の既定は 2021-10-01（それ以前は登録されていない 2021 年の spot メンテナンス欠損 2021-08-13・2021-09-29 に当たり fail-closed になる）。プライマリソースに無い 1h 行は、`--daily-fallback`（既定オフ）で checksum 検証済み Vision **daily** archive からのみ補完し、補完した各時間の URL と SHA-256 を receipt に記録する（前方補間・REST は使わない）。spot レッグは `borrow_available=False`（spot ショート不可）、perp は True。`run` は `expected_protocol_digest=protocol.protocol_digest()` を渡し receipt に記録する。`run` は結果が `evidence_eligible` でなければ成功を報告せず（終了コード 4）、`tierd-run-receipt.json` に `evidence_eligible`・`outcome`・`underlying_outcome` を記録する。`run` は `--authorization-url` の owner G4 コメントを `tools/core7_g4_comment.py` が `gh api` （読取のみ・timeout 付き・token を引数や出力に出さない）で取得し、コメント id・`html_url`・issue 番号がURLと一致し、投稿者が owner（login `shuntatsu` かつ pin した数値 id `148511409`、type `User`）であり、本文が現行 protocol digest と読込む dataset_id を 64 桁の完全トークンとして引用することを、データ読込前（owner・digest）と読込後（dataset_id）に確認する。`run_study` の直前に再取得して本文 SHA-256 と `updated_at` が同一であることも要求し、`AuthorizationReceipt` を渡して run receipt に記録する。取得・検証の失敗は終了コード 5 で fail-closed とする。このドライバ自体は経済実験を認可しない。

## Low-turnover family screen boundary

`evaluation/lowturn_screen/` は凍結プロトコル `lowturn_family_screen_v1`（[specs/lowturn-family-screen.md](../specs/lowturn-family-screen.md)）を実装する一回限りのスクリーンで、`protocol.py`（定数・コスト/リスクは `core7_mtf_study.protocol` の値をそのまま再利用・`protocol_digest()`）、`signals.py`（`close` 配列だけからの因果的・ベクトル化された r30/r90/r180・vol60・XS スコア・SMA・availability。行 `t` は `timestamps[t]`（bar close）時点で完了した行 `<= t` のみを使い、`mtf6_signals` / `ShadowAnchorMtfPolicy` と同じく決定は行 `max(0, index - signal_lag_bars)` を読む）、`policies.py`（家族で媒介変数化した単一の `PortfolioPolicy`：`(index - segment_start) % 42` の週次リバランス、シンボル単位 0.02 デッドバンド、リバランス間は exact-quantity HOLD、`risk_scale` が下がったら最後のリバランス目標を再提案して risk が一度だけ縮小し、縮小後の一定 `risk_scale < 1` では `current / risk_scale` を提案して再縮小を避ける）、`decision.py`（データ非依存の `DEV_PASS` / `HOLDOUT_CONFIRMED` / ラベル規則と、報告専用で判定に使わない deflated Sharpe（5 trials）・暦年別リターン）、`runner.py`（`run_shared_portfolio_replay` を `settle_terminal_position=True` と `capture_ledger_evidence=True` で呼ぶ唯一の経路。台帳再構成・セル要約・未来摂動は core7 ランナーのものを再利用し、第 2 のバックテスタ・台帳・約定モデル・ネットワーク経路を持たない）を所有する。20% ドローダウン停止はエンジンが終了させず `risk_scale=0` で以降を平坦化し決済が走るため、台帳が一致し平坦で非 insolvent の停止セルは**有効な経済的 STOP**であり（INVALID ではない）、`DEV_PASS` / `HOLDOUT_CONFIRMED` の「停止なし」条件だけを満たさない。ラベルは `EXPLORATORY_CANDIDATE` が上限で、paper/live・FS1・unused/final データを認可しない。Tier D の取得・G4 コメント認証・clean tree・出力 write-once は `tools/lowturn_screen.py`（`python -m tools.lowturn_screen plan | run`）が `tools.core7_mtf_tierd` と同じ経路で担い、dataset id が凍結値でなければ拒否し、全セルと manifest と `decision.json` を書いてからラベルを出力する。

## Out-of-sample trend test boundary

`evaluation/trend_oos/` は、実行済みの旧 `trend_long_spot_oos_v1`（[歴史的仕様](../specs/trend-long-spot-oos.md)）と区別した提案中の **software-only** `trend_long_spot_oos_v2`（[現行仕様](../specs/trend-long-spot-oos-v2.md)）を実装する。旧BACKCASTを未使用OOSとして再利用せず、旧判定を変更しない。`protocol.py`（定数・コスト・リスク・ローリング条件は `lowturn_screen.protocol` から**import**し再入力せず、再利用する lowturn プロトコルの digest を固定して `verify_lowturn_binding()` が変更を検出する。ステージ窓は spec の表そのまま、`protocol_digest()`）、`data.py`（ステージ窓の 4h・11 ペア・22 レッグのデータ組立のみ。取引対象は spot で perp レッグは読み込まず全行 unavailable、未上場の spot 行は inactive・非 tradable・information unavailable で定数センチネル価格のみを持ち、価格を作らず前方補完しない。`StageRowSource` が FORWARD の読取上限（最終行・最新の完全に閉じたバー・正確な認可 URL）を読取前と返却行で強制し、Tier D / core7 の既定 embargo は変更しない）、`signals.py`（lowturn のヘルパを再利用し、未上場行を NaN にしてから lookback を作る）、`cash_policy.py`（変更しない `LowturnPolicy` を包む spot 現金口座ラッパー。全 arm で frozen scenario の最大設定約定コスト率 `r` を使い、現在gross `g0` と実効目標gross `g1` を `g1 <= max(0, (1-r*g0)/(1+r))` に制限して約定コストを現金予約する。binding時は全HOLDを外し、相対許容誤差なしで微小な実取引を認識する。台帳の累積過去手数料を信用枠として再利用せず、手数料だけによる負cashも `negative_cash_borrowing` としてINVALID。未モデル化の市場時刻変動・部分約定・追加venue手数料はreserveで保証せず旧現金残高だけでなく順序付きspot fillイベントでの途中負cashも監査しfail-closed（売却予定代金は先行する買いの原資に使わない））、`decision.py`（データ非依存の `OOS_SUPPORTED` / `OOS_NOT_SUPPORTED` / `INVALID` 規則、末端の dust を許容するセル有効性、判定に使わない暦年別リターンと stationary block bootstrap 下限）、`runner.py`（lowturn の方針を変更せず `run_shared_portfolio_replay` を `settle_terminal_position=True` と `capture_ledger_evidence=True` で呼ぶ唯一の経路。セル要約は `execution_cost=` を渡して lean-core の terminal dust を認識し、第 2 のバックテスタ・ネットワーク経路を持たない）を所有する。市場データ store・Vision アーカイブ・G4 コメント認証・clean tree・出力 write-once・`--final` の日付ゲート・BACKCAST 判定を要求する停止規則は `tools/trend_oos.py`（`python -m tools.trend_oos plan | coverage | halts-from-coverage | run`（`coverage` は信号・replay・収益を作らずに stage の spot アーカイブの欠損・fallback 補完・placeholder・checksum を `coverage.json` へ報告するだけで、欠損でも拒否しない。使えない月次アーカイブは検証済み日次アーカイブから再構成し、全シンボルに共通の欠損時間帯だけを coverage.json の SHA-256 と共に `protocol.DECLARED_HALTS` へ凍結して Tier D の `SPOT_HALTS` と同じ扱いにする。宣言外の欠損は run を拒否する））が担い、FORWARD の 2026-08-01 以降の読取は Issue #810 の正確な認可コメント URL に束縛した `ForwardEmbargoPolicy` だけを通る。ラベルの上限は BACKCAST のみで `OOS_SUPPORTED`、両ステージで `OOS_AND_FORWARD_SUPPORTED` であり、注文・live・FS1 採点を認可しない。

## Agy anchor port boundary

`strategies/agy_anchor/` is a numpy-only software port of the agy lab Non-RL anchor `ANC_BIDIR_75_25` (4H, klines + funding variant). It owns `primitives.py` (operation-for-operation numeric helpers), `inputs.py` (`AnchorInputs`: spot/perp close, quote and taker-buy quote volume, validity and raw funding events on a 4h bar-close clock, plus the count-based funding observations), `signals.py` (hybrid alphas E1-E4, N3/N4, trend/revert sleeves, `theta_trend` regime blend, weekly IV carry, bidirectional perp carry, the 0.75/0.25 blend and the tail volatility cushion), `targets.py` (the 4H target synthesis with `w_max_leg=0.55`, `gross_cap=2.5`, the daily stride and the 0.04 deadband) and `policy.py` (`AgyAnchorPolicy`, a `PortfolioPolicy` for the alternating 22-leg layout that reuses the exact-quantity HOLD / pre-image / non-compounding `risk_scale` contract of `ShadowAnchorMtfPolicy`). The package imports only numpy, `data.market` (inputs) and `strategies.portfolio` (policy); it has no I/O, network, randomness, torch/SB3, replay, simulation, risk or integrations dependency and is not re-exported from `strategies`. Hyperliquid, Binance open-interest/long-short metrics and the 15m jump are neutral in this variant. Every row depends only on rows and funding events up to its bar close (truncation test). `tests/fixtures/agy_anchor/` holds the parity fixture (known-used 2021 data; see its README): the port reproduces the lab with one disclosed correction, namely the lab round-7 `factor_scores[:, :, 9]` symbol/factor axis defect, which the port implements as the intended per-symbol N4. The port is software only: it authorizes no economic experiment, unused-data access or live order, and a green parity test says nothing about profitability. The Tier D assembly (`RawMarketSeries`, `tierd.py`) does not carry taker-buy volume; `from_market_dataset` takes it as an explicit `(T, legs)` argument, so no dataset code or `dataset_id` changes.

## Agy anchor out-of-sample test boundary

`evaluation/agy_anchor_oos/` implements the frozen protocol `agy_anchor_realistic_v1` ([specs/agy-anchor-realistic.md](../specs/agy-anchor-realistic.md), with Amendment 1). It reuses `trend_oos` and `lowturn_screen` by import and does not copy them. `protocol.py` holds the frozen constants (costs and risk are the Core7 MTF values, the anchor's constants are imported from `strategies/agy_anchor`), the stages DEV / BACKCAST_2020 / FORWARD, the 540-row stage warm-up and the 180-bar pair warm-up, and the cash-account rate `INITIAL_MARGIN_RATE = 0.4`, which is the engine's own `1 / max_leverage` (`simulation/execution.py:756`; the engine has no explicit perpetual initial-margin field). `protocol_digest()` binds the SHA-256 of the parity fixture, a digest of the port's source files and a computed digest of the harness source (`protocol`, `cash`, `decision`, `data`, `runner` and `tools/agy_anchor_oos.py`), so a change to any of them is a new digest. `data.py` assembles the 22-leg spot and perpetual stage dataset from hourly klines and funding (a market is unavailable before its listing, spot and perp separately; no price is invented) and carries the taker-buy quote volume of the same klines as a `(T, legs)` array beside the dataset, bound into the dataset identity by its SHA-256, because the Tier D assembler, which is bound by a research packet and is not touched, drops it. `build_anchor_inputs` gates a pair until both markets are listed plus the warm-up. `cash.py` scales every arm's target by the tighter of two bounds: spot long gross plus 0.4 times perpetual gross at most 1.0, and net signed exposure at most 0.97 (the engine books perps through cash like spot, `simulation/accounting.py:820`; Amendment 3), and reports which bound bound. `decision.py` is the pure data-free rule and the label table (`ANCHOR_SUPPORTED`, `ANCHOR_FORWARD_ONLY`, `ANCHOR_NOT_SUPPORTED`, `INVALID`; DEV never gates). `runner.py` calls `run_shared_portfolio_replay` once with settlement and ledger capture, passes `execution_cost=` to the core7 cell summary and probes causality per (arm, lag) over market rows and taker-buy volume. The package imports no network, store or ML module; only `tools/agy_anchor_oos.py` wires the market store, the FORWARD embargo policy (bound to the owner scope relay comment 7040247247 for BACKCAST_2020 and FORWARD), the run order (BACKCAST_2020 needs the DEV result, FORWARD final needs a BACKCAST_2020 decision of the same digest and is refused before 2027-05-02T00:00Z) and the G4 comment check. Passing these tests and CI authorizes no stage, no label and no order.

## Learning lab boundary

`evaluation/learning_lab/` is the common harness for the RL and ML overlays of the learning lab ([specs/learning-lab-registry.md](../specs/learning-lab-registry.md), first study [specs/anchor-rl-execution-dev.md](../specs/anchor-rl-execution-dev.md)). It is software plus synthetic tests; it carries no economic claim and loads no data. It sits under `evaluation/` beside `agy_anchor_oos/` because it reuses that package's cash-account scale and the ported anchor by import and is not re-exported from `trade_rl.evaluation` or imported by any other package. `splits.py` holds the frozen walk-forward windows on row timestamps (train 2022-01-01..2023-12-31, validate 2024-01-01..2025-06-30, test 2025-07-01..2026-07-31, half-open so a boundary row belongs to the later window) and `require_before_cutoff`, which refuses any row at or after 2026-08-01. `overlay_env.py` is the Gymnasium execution overlay: every `decision_every_bars` (6) bars the agent picks the trade fraction `f` of the gap between the current weights and the base target (discrete {0, 1/3, 2/3, 1} or continuous [0, 1]); `f = 1` executes the base proposal untouched and `f = 0` holds the exact quantities. Every native bar goes through `SharedPortfolioSession.advance` and terminal intervals through `settle_next_interval`, and the reward is `net_equity_increment` of the engine's book, so the engine is the only reward oracle and the package has no PnL calculator or second backtester. The base target is peeked on a deep copy of the base stack; the only domain-randomization and adversary seam is the per-episode `PerturbationSource` (cost multiplier, extra order latency, return shock), bounded by constants because the engine fixes its cost configuration per session. `baselines.py` builds the anchor behind the cash-account scale with a rebalance gate chosen by the overlay and runs the fixed strides 6, 18 and 42 and `CASH` through `run_shared_portfolio_replay`; an overlay with `f = 1` on every `S`-th decision reproduces the stride-`S` baseline bit for bit. `agents.py` wraps stable-baselines3 / sb3-contrib (PPO, A2C, SAC, QR-DQN) lazily, because `fit_shared_portfolio_ppo` is bound to `SharedPortfolioGymEnv`'s per-symbol action; checkpoints are written with their SHA-256 and loaded only on a match. `registry.py` is the append-only hash-chained JSONL trial registry (POSIX same-inode advisory shared/exclusive locks around verified reads and atomic logical appends; symlink/nonregular path refusal; no distributed writer authority) and the deflated Sharpe over all trials, `metrics.py` the net-return summary, `runner.py` the plan/train/evaluate orchestration (test only after a completed validate trial) and `synthetic.py` the seeded stand-in stage. Only `tools/learning_lab.py` is the real-data gate: `--data real` needs an owner G4 comment (`--authorization-url`) quoting `anchor_rl_execution_dev_v1` and the protocol digest, verified by `tools.core7_g4_comment` before anything is read, and even then no real-data loader is wired. The optional `learning-lab` extra pins `stable-baselines3`, `sb3-contrib`, `torch` and `lightgbm`; no module imports torch at import time.

## Cross-sectional low-volatility / funding out-of-sample test boundary

`evaluation/xs_lowvol_funding/` implements the frozen protocol `xs_lowvol_funding_v1` ([specs/xs-lowvol-funding.md](../specs/xs-lowvol-funding.md)): a weekly perp-only book that is long low-volatility, low-funding names and short high-volatility, high-funding names, tested on symbols no earlier study read. It reuses `trend_oos` (`OosCell`, the stationary bootstrap and the shared runner helpers), `lowturn_screen` (the deflated Sharpe) and the Core7 MTF cell summary by import and does not copy them. `protocol.py` holds every frozen constant, the cost table as `ExecutionCostConfig`, the risk config, the stage windows (DEV on the 11 Tier D pairs, DISJOINT on the frozen candidate list, FORWARD on the Vision listing at run time) and the SHA-256 of `candidates.json`; `protocol_digest()` binds that hash, the reused lowturn binding and a computed digest of the harness source (`protocol`, `universe`, `notices`, `policy`, `data`, `decision`, `runner` and `tools/xs_lowvol_funding.py`), so a change to any of them is a new digest. `universe.py` is the point-in-time monthly universe (90 days of complete 1h bars and nine funding events before the month, the 30 largest by trailing 30-day quote volume, flat below 12 eligible symbols). `policy.py` is `XsPolicy`, the only `PortfolioPolicy` of the package: weekly Monday decisions, the rank score, `floor(n/3)` inverse-volatility legs, the 20% shrunk-covariance volatility target, the per-name, gross and cash-account caps (margin `0.4 * gross <= 1.0`, net exposure `<= 0.97`), the exact-quantity HOLD and non-compounding `risk_scale` contract of the anchor policy, and the forced-close schedule, which `notices.py` derives only from recorded public Binance notices (Amendment 5): a held name's target is 0 from the first 4h bar opening at or after the earliest notice (released at most 90 days before the event) through the name's last available bar, an event without such a notice refuses the run before any replay, and the recorded file's SHA-256 is pinned in the protocol payload and verified before it is read. `data.py` assembles the perp-only 4h dataset for the legs that are ever universe members (an unavailable bar is inactive and untradable with a sentinel price; the first inactive bar opens at the last published close) and owns the stage read guard. `decision.py` is the pure data-free rule (INVALID, conditions C1-C6, the DISJOINT and FORWARD labels, the reported figures). `runner.py` calls `run_shared_portfolio_replay` once with settlement and ledger capture, passes `execution_cost=` to the core7 cell summary, books the forced-close adverse 5% as a cash debit on the equity path from the fill bar on, and probes causality per (arm, signal lag). A position still held at a name's last available bar is written off (Amendment 6): the engine settles it at the first inactive bar's open, the last published close, the shared fill tracker and (opt-in `engine_settlement`) the Core7 ledger reconciliation accept exactly that settlement, and the runner books the same 5% charge on the settled notional as a cash debit at the settlement bar and reports `write_off_events` and `write_off_count` per cell (never gating); a settlement the ledger cannot show exactly refuses with `UnsupportedWriteOffError` naming the symbol. The package imports no network, store or ML module; only `tools/xs_lowvol_funding.py` wires the market store, the FORWARD embargo policy (bound to the owner scope relay comment 7040247247 for DISJOINT and FORWARD), the run order (DISJOINT needs a DEV result without an invalid cell, FORWARD needs a DISJOINT decision labelled `XS_SUPPORTED` of the same digest and its final run is refused before 2027-05-08T00:00Z) and the G4 comment check that quotes the protocol id, the digest and the code commit. Passing these tests and CI authorizes no stage, no label and no order.

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

外部研究packageが使うtyped extension boundary (#1359) は、既存の次の
import先とsignatureを維持対象とする。別の実行engineや会計adapterは追加しない。

| import先 | 対象 |
|---|---|
| `trade_rl.data` / `trade_rl.data.build` | `MarketDataset` / `MarketDatasetBuilder` |
| `trade_rl.data.contracts` / `trade_rl.data.source` | `MarketBuildConfig`, `InstrumentContract`, `FeatureSpec`, `FeatureKind` / `MarketDataSource`, `RawMarketSeries` |
| `trade_rl.data.features` | `calculate_feature_events`（現行native-bar/availability契約） |
| `trade_rl.data.features.divergence` | `DivergenceConfig`, `DivergenceEvent`, `DivergenceSignals`, `detect_divergences` |
| `trade_rl.data.features.indicators.divergence_bank` | `build_divergence_bank`, `DIVERGENCE_INDICATOR_SPECS`, `DEFAULT_DIVERGENCE_INDICATORS` |
| `trade_rl.data.features.indicators.ichimoku_cloud` | `VisibleCloud`, `visible_ichimoku_cloud` |
| `trade_rl.strategies.portfolio` | `PortfolioPolicy`, `PortfolioDecisionSnapshot`, `PortfolioTargetProposal` |
| `trade_rl.risk` / `trade_rl.simulation` | `PreTradeRisk`, `PreTradeRiskConfig` / `ExecutionCostConfig` |
| `trade_rl.simulation` | `runtime_execution_policy_digest` (source-free runtime identity) |
| `trade_rl.risk.spot_cash` / `trade_rl.strategies.allocation.spot_cash` | `spot_cash_target_factor` / `SpotCashBudgetPolicy` |
| `trade_rl.risk.cash_projection` | `MixedCashProjection`, `project_mixed_cash_budget`, `snapshot_drawdown_scale` |
| `trade_rl.simulation.quantities` | `parse_quantity`（canonical rational string） |
| `trade_rl.evaluation.replay` | `run_shared_portfolio_replay`, `SharedCashPortfolioReplayResult`, `SharedCashPortfolioReplayLedgerEvidence` |
| `trade_rl.evaluation.equity_path` | `daily_equity`, `daily_returns`, `max_drawdown`, `per_year_returns`, `performance_summary`, `EquityPathSummary` |
| `trade_rl.evaluation.comparison.overlapping_blocks` | `moving_block_indices`, `bootstrap_excess`, `BootstrapInterval` |

Every executable study must use the common native execution and accounting
owner (MarketExecutor, BookState, SharedPortfolioSession / replay).
Study code may define targets, signals, experimental parameters, and
read-only ledger oracles; it must not maintain a second mutable cash,
position, funding, margin or P&L book. Legacy independent study books are
historical dependencies to retire after consumer and source-bound approval
closure, not templates for new code.

`spot_cash_target_factor` is a conservative long-only spot **decision-mark**
budget: reserve the supplied deterministic cost envelope on worst-case gross
turnover and reserve new buys against already available cash, without
crediting unexecuted sales. Both spot and mixed cash projections share the
native binary64-compatible scalar admission described below; reject invalid
inputs before narrowing. `SpotCashBudgetPolicy` wraps an all-spot policy
and leaves native risk/quantity HOLD and execution in charge. It rejects
shorts and active orders; it does not guarantee execution-time cash under
gaps, liquidity, fee overrides, or order-state changes. The separate opt-in
native strict spot cash admission rejects unaffordable actual fills under its
long-only, unlevered, deterministic execution contract (#1306).
The old source-bound trend-OOS wrapper is unchanged until reviewed migration.

The common decision-mark cash projection takes explicit sorted spot indices,
nonnegative spot weights, signed perpetual weights and caller-declared
margin/cash/net limits. It admits native binary64-compatible scalars only
(no booleans, scalar subclasses, arbitrary rationals, temporal values, wider floating-point
values or integers outside the exact +/-2**53 domain); validate before
narrowing to avoid masking negative spot weights or out-of-domain limits.
It returns immutable pre-cap margin and signed net
measures and their uniform scaling factors. The snapshot drawdown helper
uses the same native PreTradeRisk ramp and preserves the historical snapshot
peak-floor/clamp convention. Research owns the experimental constants and
report-only statistics, not a second balance, margin or risk calculation.
This is an opt-in projection for valid long-only spot / signed perpetual
targets, **not** a mixed-instrument fill-time collateral or venue-margin
guarantee. Existing source-bound AGY/XS/low-turnover implementations remain
unchanged until reviewed consumer migration.

`runtime_execution_policy_digest(cost, funding_price_rate_present=..., ...)`
computes the exact runtime execution-policy identity without loading a Dataset.
The caller must independently admit the configuration and explicitly bind the
presence (not nonzero content) of the optional event-price funding array, any
verified market-order-profile digest, rule stress, insolvency mode and strict
spot-cash mode. `MarketExecutor.execution_policy_digest` uses the same digest
composition, including wrapper order, and retains existing digests and cache
semantics. This is identity computation, not source attestation, profile/Dataset
validation, row-level cash admission or replay authorization. A result's own
unverified metadata is not an independent expected identity.

The explicit equity-path API samples observed UTC midnight rows from aligned,
strictly increasing int64 nanosecond timestamps and finite positive equity.
Equity input uses the native binary64-compatible domain: reject wider numeric
dtypes and integer marks above 2**53 before conversion, even if an individual
out-of-range integer could happen to be representable. Positive narrow integer
and binary64-compatible floating arrays remain supported. This is an input
precision guard, not arbitrary-precision statistics: return and drawdown
arithmetic retains the historical binary64 formulas.
It does not interpolate missing days or authenticate scoring-window coverage.
CAGR uses elapsed time and caller-specified days/year; daily Sharpe uses sample
standard deviation and the caller-specified annualization. Calendar-segment
returns retain the original inclusive next-year-boundary convention.
The overlapping-block API generates a common seeded non-circular resample
matrix for paired excess paths, then simple annualized mean percentile and
Bonferroni intervals. It is distinct from the existing circular
`moving_block_mean_test`; neither changes the other's semantics.
The new equity, bootstrap and divergence array boundaries require exact native
ndarrays with ordinary ndarray/memoryview ownership ending in owned storage or
exact raw bytes/bytearray; reject retained masks, subclasses and opaque owners
before conversion. Missingness erased by a caller cannot be reconstructed.
Nonfinite/bad inputs fail closed; the historical `ci95` mapping is available
only for alpha=0.05. Both APIs match the historical method for valid inputs,
are observer-only and own no dataset I/O, result selection, risk or accounting.
Existing source-bound historical modules are deliberately unchanged until
accepted migration/cutover; remove their legacy copies after consumer closure,
not by silently changing frozen study source bytes now.


The explicit divergence detector and visible cloud are native-row arithmetic,
not strategy, replay or economic authorization. A confirmed divergence is
emitted at its observation row, not its pivot; concept-family quorum and
deduplication are explicit. The oscillator bank reuses canonical feature
numerics and separates the optional additional oscillators. Missing values
remain NaN; no existing Dataset feature profile is changed. Visible Ichimoku
uses the span from the displaced past source row, not the existing unshifted
cloud-position feature. All periods are counts of native bars; its readiness
requires the entire longest dependency window through the visible row to be
tradable. Midpoints retain normal arithmetic and avoid sum overflow. These
primitives consume caller-supplied completed native observations: readiness
is not publication-time availability. The caller must use the canonical
native-clock and as-of alignment contracts, including missing-bar detection
and availability delays, before serving a decision. No new 3d/weekly Dataset
or execution-clock support follows merely from row arithmetic. Research
owns timeframe composition, strategy thresholds, rosters and trial contexts.
The original source-bound research branch is not modified by this API.


`trade_rl/py.typed` (PEP 561) はinline annotationの配布を明示する。
このmarkerは全private moduleを安定APIと宣言するものではなく、NumPy arrayのshape・金融・因果性のruntime契約を静的型検査だけで保証するものでもない。外部研究は検証済みcore commit/wheel digestにpinし、互換性変更をIssue/PRと再テストで扱う。optional trainerのimportや既存artifact/source identityは変更しない。

## Architecture change rule

Packageを追加・移動・削除するときは同じ変更で次を行う。

1. 責務と依存方向をこの文書へ反映する。
2. `tests/architecture/` でrequired/forbidden pathと依存を契約化する。
3. intentionally publicなpackage facadeをsnapshot/contract testで確認する。
4. 単なるmoveならnon-import AST、serialized bytes/digest、golden behavior等でsemantic driftを可能な範囲で反証する。
5. 旧private pathや一時migration helperをfinal treeに残さない。

## Distribution source closure

構造変更では、working treeだけでなくGit HEADのproduction `.py` roster、sdist、direct wheel、sdistから再buildしたwheelの相対pathとSHA-256が一致することを検証する。`tests/architecture/distribution.py` は未追跡・ignoreされたsource、worktree差分、sourceの欠落・混入・改変、重複member、不正path、symlink sourceを拒否し、archiveを展開・実行しない。PPO normalizationの非Python runtime authorityである `trade_rl/evaluation/ppo_normalization_activation.json` は明示的なpackage-resource closureへ含め、checkout/sdist/wheel間のexact bytesとcanonical schemaを同じgateで検証する。

CIはbuilt wheelをcheckout外の新規venvへ非editable installし、isolated Pythonでpackage identity、public facade import、candidate/bootstrap CLI helpに加えて、installed wheelから実際のnormalization activation resourceを読み、そのSHA-256がcheckout authorityと一致することを確認する。通常のsource closureはPython source中心の配布契約であり、optional trainerの実学習、全platform動作、任意のnon-code resourceすべてを保証するものではない。normalization activation resourceと `py.typed` はそれぞれ研究authorityと型付き配布契約のため、明示的なresource closure対象とする。Full CIではclean wheelの型情報を使い、checkout外のconsumerに対するmypy正例（正しい型のinfer）と負例（型不一致7件の期待診断）を検査する。core側のmypyが通るだけでは外部packageの型契約を満たしたとはみなさない。license/provenanceの恒久保持は別の既存gateも維持する。
