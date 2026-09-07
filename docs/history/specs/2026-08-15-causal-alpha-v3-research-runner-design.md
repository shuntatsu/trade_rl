# Causal Alpha V3 research runner design

## Status and objective

This specification closes the gap between the existing Causal Alpha V3 primitives and a reproducible real-data research workflow. The runner is research-only and must execute, in order, strict authored configuration, artifact/runtime identity closure, causal V3 fitting, a train-only signal gate, immutable candidate freeze, resumable production-environment selection replay, economic selection, exactly-once teacher holdout admission, and a research-only teacher package.

The maintained Universal U6 path remains unchanged. This work does not authorize production use and does not claim positive alpha or profitability.

## Non-goals

- Do not run or wire DAgger, BC, critic warm start, PPO, Lagrangian, or discounted PPO in this change.
- Do not change canonical U6 example configs, teacher kind, action mode, reward, admission thresholds, or hard-risk settings.
- Do not use validation symbols, test symbols, or teacher-admission holdouts for fit, signal gating, candidate freeze, or economic selection.
- Do not treat historical `unavailable_legacy` checkpoint diagnostics as resumable or promotable evidence.
- Do not introduce a parallel replay executor before the serial resume/identity contract is proven.

## Invariants

- Scalar reward remains pure net log growth.
- Execution cost is accounted for once by the production environment. V3 target cost is a decision hurdle only.
- The maintained one-decision execution delay is preserved.
- The hard `max_position_to_market_notional=0.02` portfolio-risk contract remains authoritative.
- V3 candidates are completely authored before the first signal evaluation. Candidate mutation after observing results is forbidden.
- Signal-gate contracts, economic-selection contracts, and the final holdout are chronologically disjoint.
- Every V3 research artifact carries a deterministic digest and is non-promotable.
- A holdout is evaluated only after one candidate is frozen and selected.
- Admission failure never produces a teacher package.
- Existing V3 primitive behavior is unchanged when new optional arguments are omitted.

## Architecture

### Strict configuration

`CausalAlphaV3ResearchConfig` uses schema `universal_causal_alpha_v3_research_config_v1`. It contains nested-selection counts, signal-gate thresholds, economic-selection thresholds, and at most eight authored candidates. Candidate semantic duplicates are rejected even when names differ. A candidate consists of one `CausalAlphaV3FitConfig` and one `CausalAlphaV3TargetConfig`.

### Nested chronological selection

For each train symbol, the existing `CausalAlphaEpisodePartition.selection_contracts` is divided into an earlier signal-gate prefix and a later economic-selection suffix. The existing latest holdout remains untouched. The runner fails before fitting if the declared signal prefix leaves fewer than the configured minimum economic contracts.

### Signal gate

At every signal contract start, `fit_causal_alpha_v3()` uses only labels whose realization index is strictly before the contract start. Predictions are evaluated only on rows whose 24h and 72h labels realize inside that same signal contract. To remove overlapping-label pseudo-replication, a deterministic greedy cohort accepts the next eligible decision only after the previous selected 72h label interval ends.

Each fit-config/symbol/episode scope records rank correlation, directional accuracy, top-minus-bottom realized-return spread, fit digest, forecast digest, and cohort indices. Gate aggregation operates on scope-level values using deterministic moving-block bootstrap. Candidate target variants sharing one fit config share the same signal-gate evidence.

### Candidate freeze

Only candidates whose fit config passes the signal gate enter an immutable freeze artifact. The freeze binds candidate digests and order, signal evidence digest, nested-partition digest, run-manifest digest, config digest, and V3 generator-code digest. Existing freeze evidence may be reused only when the expected artifact is byte-semantically identical.

### Production selection replay

For every frozen candidate and economic contract, the runner fits at `contract.start`, predicts the contract, computes production one-way execution costs and causal liquidity caps, compiles V3 targets, and evaluates the action path with `evaluate_episode_action_path_on_environment` against the same production environment implementation used by maintained causal selection.

`causal_alpha_v3_target_path()` receives an optional `actionable_mask`. A false row is an explicit `unactionable_hold`, cannot submit a target change, and preserves the prior target. Omitting the mask preserves current primitive behavior.

### Resumable immutable records

Completed replay scopes are persisted as one atomic JSON file per `(candidate_digest, symbol, episode_index)`. Every record binds run, freeze, candidate, contract, fit, forecast, and target-path identities. Resume first reconstructs the complete expected scope set and rejects unknown, corrupted, duplicated, or identity-drifted files. A completed valid scope is never replayed again.

A candidate may be irreversibly pruned when one completed scope proves a condition future scopes cannot repair: hard-risk violation, a symbol-episode net return below the maintained floor, or unexplained execution rejections above the allowed total. Mean-return and turnover conditions do not cause early pruning because future scopes can change them.

### Economic selection

A candidate is admissible only if it satisfies all authored thresholds for mean gross return, mean net return, symbol-episode lower-tail net return, turnover, unexplained execution rejections, positive-gross episode fraction, nonzero trades, and hard risk. Admissible candidates are ranked by lower-tail net return, mean net return, mean gross return, lower turnover, lower total execution cost, then deterministic candidate digest.

### Exactly-once admission

After selection, holdout targets are generated for the selected candidate only. Each symbol holdout result is written as a separate immutable atomic record before the aggregate admission decision. Resume validates existing admission records and evaluates only missing symbols, preventing a successfully written holdout result from being sampled again after process interruption.

The aggregate uses the maintained `evaluate_causal_alpha_teacher_admission` contract without threshold changes.

### Research teacher package

Admission success creates `UniversalCausalAlphaV3TeacherPackage`, containing selected candidate identity, signal/freeze/selection/admission evidence digests, per-symbol `EpisodeOracleBatch` targets, partition/sample identities, run/config/generator identities, and `research_only=true`, `promotion_eligible=false`. Canonical U6 consumers do not accept this package implicitly.

### CLI

`scripts/run_universal_causal_alpha_v3_research.py` loads one research JSON, one maintained `TrainingRunConfig`, and one `UniversalRuntimeFactoryContext`. It uses the existing artifact-verified Universal runtime factory to obtain train bindings, concrete environment factory, and instrument context provider, then invokes the V3 runner.

Terminal research outcomes use distinct exit codes: `0` admitted, `2` signal rejection, `3` economic-selection rejection, `4` teacher-admission rejection, and `1` invalid/corrupt/unexpected execution failure.

## Failure modes and required handling

- Config/runtime/dataset/feature/normalizer/source identity drift: reject before reuse or continuation.
- Duplicate candidate semantics: reject before evaluation.
- Future-label leakage: fit and diagnostic realization bounds are strict.
- Signal/economic/holdout scope overlap: reject partition construction.
- Label pseudo-replication: use non-overlapping 72h realization intervals for the signal gate.
- Partial record write: atomic file replacement; an absent destination means incomplete.
- Corrupt or unknown resume record: reject rather than ignore.
- Completed-scope duplicate execution: skip only after full identity and digest validation.
- Holdout early access: API/state transition rejects admission before selected evidence exists.
- Admission rerun after interruption: reuse only verified per-symbol records.
- Admission failure followed by package creation: prohibited by package constructor/runner state.
- Unavailable forecast row causing target mutation: actionable mask forces hold.
- Resource leak: every concrete environment is closed in `finally`.

## Test oracle

Correctness is observed through config rejection, exact contract partitions, fit knowledge cutoffs, non-overlapping cohort indices, bootstrap evidence, immutable digests, environment action history, realized performance metrics, atomic file state, resume replay counts, pruning state, selected-candidate identity, holdout factory/evaluation counts, package presence/absence, and canonical-config non-change.

## Required test layers

- Unit: config, contracts, non-overlap cohort, signal gate, selection ranking, record stores, actionable target mask.
- Integration: V3 target generation and replay against environment factories; exactly-once holdout admission.
- CLI/E2E: synthetic artifact-bound runner from JSON to terminal evidence.
- Contract/architecture: canonical U6 config/reward/risk/teacher invariants remain unchanged.
- Static: Ruff, format, Mypy, import architecture.
- Full regression: repository pytest, coverage ratchet, Linux/Windows compatibility, packaging/training image, exact-head GitHub Actions.

## Acceptance criteria

1. A strict authored config can be loaded and carries stable semantic identity.
2. Signal, economic selection, and holdout scopes are disjoint and artifact-bound.
3. V3 fit is executed on real sample blocks with a strict knowledge cutoff and a non-overlapping signal cohort.
4. Signal failure produces terminal non-promotable rejection and no economic replay.
5. Passing candidates are immutably frozen before any economic replay.
6. Production replay records are atomic, fully identity-bound, resumable, and never silently mixed across runs.
7. Economic selection enforces maintained lower-tail/hard-risk behavior and deterministic ranking.
8. Holdout admission is inaccessible before selection and exactly-once per persisted symbol record.
9. Admission failure cannot create a V3 teacher package; pass creates a research-only, non-promotable package.
10. The CLI can execute the complete deterministic research path without wiring downstream RL.
11. Canonical U6 configs, reward, hard market-notional risk, and execution delay are unchanged.
12. Targeted tests, static checks, full tests, coverage/architecture gates, and exact final-head CI pass before the PR is ready.
