# Causal Alpha V3 Signal Contract V2 Implementation Plan

> **For agentic workers:** use Superpowers execution/TDD/verification workflows. This plan records the quality contract and the observed RED/GREEN sequence; the exact final HEAD remains the completion authority.

**Goal:** Replace ambiguous V3 signal-scope semantics with an auditable V2 contract that separates raw records from independent chronological episodes, validates pooled chronology and label timing, binds runtime provenance, and resumes from strict run-bound signal leaf artifacts.

**Architecture:** Keep the existing V3 research stage ordering and clustered-bootstrap numerics. Version schemas where semantics change; make `(contract_start, contract_stop)` the independence key; validate shared chronology before fit; reject mixed fit evidence; and make persisted run-bound signal leaves the resume source of truth while recomputing derived aggregate evidence.

## Global constraints

- Do not change ridge fitting, forecast formulas, target-controller semantics, label formulas, economic selection, teacher admission, reward, risk, execution, DAgger, BC, critic warm start, PPO, or Lagrangian learning.
- Do not weaken signal-quality lower-CI thresholds or raw coverage.
- Same-episode symbol duplication must not increase independent evidence.
- V3 remains `research_only=true` and `promotion_eligible=false`.
- Old V1 authored signal semantics must not be silently reinterpreted as V2.
- All behavioral changes follow RED -> GREEN -> refactor and exact-final-HEAD verification.

---

## Task 1: Authored Signal Gate V2 semantics

**Contract**

- `CausalAlphaV3SignalGate.minimum_independent_episode_count: int`
- `CausalAlphaV3SignalGate.minimum_raw_scope_coverage: float`
- authored schema `universal_causal_alpha_v3_research_config_v2`
- `minimum_independent_episode_count <= signal_contract_count`

**Status**

- [x] V1 payload rejection test added and RED observed.
- [x] Impossible independent-episode threshold test added and RED observed.
- [x] Maintained example bound to `8 == 8` independent episodes and `1.0` raw coverage.
- [x] V2 fields/schema implemented without changing quality thresholds.

## Task 2: Signal leaf and aggregate Evidence V2

**Contract**

- leaf schema `causal_alpha_v3_signal_scope_v2`
- leaf persists `run_manifest_digest`, `contract_start`, `contract_stop`
- aggregate schema `causal_alpha_v3_signal_gate_evidence_v2`
- explicit raw and independent counts
- independence key `(contract_start, contract_stop)`
- one aggregate evidence object contains one run and one fit configuration
- one chronological cluster contains one fitted-model identity

**Status**

- [x] Strict leaf round-trip/schema regression added.
- [x] Equal local episode index with different intervals counts as separate chronological episodes.
- [x] Same-interval symbol duplication increases raw count but not independent count.
- [x] Explicit `independence_unit` / `aggregation_mode` / count evidence implemented.
- [x] Flat aggregate evaluator removed from current V3 path.
- [x] RED observed for mixed `fit_config_digest` and within-cluster `fit_digest` drift before production checks were added.
- [x] Fail-closed fit-consistency checks implemented.

## Task 3: Shared chronology and runtime provenance closure

**Contract**

- execution identity schema `causal_alpha_v3_execution_identity_v2`
- binds `shared_clock_digest`, `dependency_lock_digest`, `python_runtime_digest`
- pooled symbols must share timestamps, `(episode_index,start,stop)` schedule, `decision_bars`, and `signal_delay_decisions`
- shared-clock digest semantics are versioned when signal-delay timing becomes part of the identity

**Status**

- [x] Identity V2 strict round-trip and legacy rejection tests added.
- [x] Timestamp, episode-schedule, and decision-cadence drift tests added.
- [x] Cross-symbol signal-delay drift test added and RED observed.
- [x] Shared chronology digest binds canonical `datetime64[ns]` clock, schedule, cadence, and signal delay.
- [x] Dependency lock digest binds source-checkout `pyproject.toml` and `uv.lock`.
- [x] Python runtime digest binds implementation and exact major/minor/micro version.
- [x] Execution identity propagates the chronology/runtime changes into run identity.

## Task 4: Run-bound Signal leaf resume wiring

**Contract**

- every leaf and aggregate evidence is bound to the active `run_manifest_digest`
- artifact store rejects cross-run writes/loads and path/contract/schema/digest drift
- pipeline builds the complete expected leaf identity map once
- `load_signal_scope_metrics(expected=...)` is called once before fit iteration
- valid leaves are reused; only missing identities call the builder
- aggregate evidence is always recomputed from the validated complete leaf set

**Status**

- [x] Cross-run metric-mixing and cross-run store-write tests added.
- [x] Valid-leaf rerun test proves the builder must not be called for a persisted identity.
- [x] Partial-crash test requires exactly the missing leaf to be rebuilt.
- [x] Corrupt persisted leaf fails before builder invocation.
- [x] RED observed while pipeline still recomputed all leaves.
- [x] Run binding added to leaf/evidence serialization and strict loaders.
- [x] Store write/load run identity enforcement implemented.
- [x] Expected identity map and one-shot loader wired before fit iteration.
- [x] Reused and newly built leaves both receive run/fit/symbol/episode/interval/contract identity validation.
- [x] Aggregate fit evidence is recomputed on every run.

## Task 5: Architecture closure and final verification

**Scope**

- current runner exposes only the chronological clustered V2 evaluator
- no current code path may produce V2 Signal Evidence with a flat raw-record bootstrap
- active production code must not retain ambiguous V1 field semantics
- final diff must remain Signal Contract/provenance/resume only

**Checklist**

- [x] Architecture coverage confirms the clustered V2 evaluator is the current path.
- [x] Active `minimum_scope_count` / `minimum_scope_coverage` production references removed; remaining occurrences are historical/migration documentation only.
- [x] Parallel Teacher Admission work was detected on PR #407 and isolated; Signal Contract work continues on dedicated Draft PR #408.
- [x] Current main was merged normally into the isolated branch; no force-push or history rewrite was used.
- [ ] Run targeted V3 config/signal/runtime/pipeline tests on the exact final HEAD.
- [ ] Run Ruff, format, Mypy, import-linter, dead-code, compatibility, training-image/non-root probe, full pytest with branch coverage, critical branch coverage, package/uv identity, and PostgreSQL Catalog on the exact final HEAD.
- [ ] Re-read the exact final diff and verify no model/economic/reward/risk/execution/RL implementation drift.
- [ ] Perform falsification review against all 16 failure modes in the V2 design spec.
- [ ] Update Draft PR #408 with RED evidence, exact-final-HEAD verification, migration notes, and remaining statistical limitations.
- [ ] Mark PR #408 Ready only after every quality gate above is observed on the same final HEAD.

## Recorded RED evidence

- Config V2 RED: `4 failed / 3730 passed / 26 skipped`.
- Signal resume/run-binding RED: `5 failed / 3736 passed / 26 skipped`.
- Combined migration RED: `10 failed / 3743 passed / 26 skipped`; failures decomposed into stale V2 fixtures plus missing signal-delay/resume behaviors.
- Isolated fit-consistency RED on the Signal-only branch: `2 failed / 3753 passed / 26 skipped`, coverage `80.03%`; only mixed fit-config and cluster fit-digest tests failed while Static/Architecture/Compatibility/Training-image gates were Green.

The final GREEN evidence must be recorded from the exact final commit after documentation and production changes stop.
