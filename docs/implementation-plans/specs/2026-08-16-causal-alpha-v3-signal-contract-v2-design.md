# Causal Alpha V3 Signal Contract V2 Design

## Objective

Close the architectural ambiguity between raw `(symbol, episode)` signal records and statistically independent chronological episode evidence. A V3 signal artifact must state its independence unit, aggregation mode, raw coverage, independent episode count, chronology identity, and runtime provenance explicitly enough that a fresh process can audit how the gate was computed.

## Non-goals

- Do not change ridge fitting, forecast formulas, target controller semantics, label formulas, economic candidate selection, teacher admission, reward, risk, execution, DAgger, BC, critic warm start, PPO, or Lagrangian learning.
- Do not weaken rank-IC, top/bottom-spread, direction-accuracy, or raw coverage thresholds to obtain a pass.
- Do not treat same-episode symbol copies as independent evidence.
- Do not make V3 promotion eligible.

## Root cause

The architecture-hardening change correctly moved signal uncertainty from symbol-major raw records to chronological episode clusters, but the authored config and evidence schema retained the generic `scope` terminology. The current clustered evaluator therefore interprets `minimum_scope_count` as an independent episode count while `expected_scope_count` and `scope_coverage` remain raw record quantities. The legacy flat evaluator can also produce the same evidence schema with different statistical semantics.

This unit ambiguity caused the maintained `signal_contract_count=8` and `minimum_scope_count=24` configuration to become structurally impossible after clustering.

## Signal configuration contract

Bump the authored config schema to `universal_causal_alpha_v3_research_config_v2`.

Replace the ambiguous signal-gate fields:

- `minimum_scope_count` -> `minimum_independent_episode_count`
- `minimum_scope_coverage` -> `minimum_raw_scope_coverage`

The maintained research config uses:

- `signal_contract_count = 8`
- `minimum_independent_episode_count = 8`
- `minimum_raw_scope_coverage = 1.0`

`CausalAlphaV3ResearchConfig` must fail before any research computation when `minimum_independent_episode_count > signal_contract_count`.

The V1 authored config is not silently reinterpreted. A V1 file must fail with an unsupported-schema error under the V2 runner so old ambiguous semantics cannot be mistaken for the new contract.

## Chronology and cluster identity

`CausalAlphaV3SignalScopeMetric` becomes schema `causal_alpha_v3_signal_scope_v2` and persists the run and contract interval used for clustering:

- `run_manifest_digest`
- `contract_start`
- `contract_stop`

The leaf-level run binding is required because persisted signal leaves become resume source-of-truth artifacts. A leaf copied from another output root must not be reusable merely because its fit config, symbol, episode index, and contract digest happen to match. The artifact store validates every signal leaf's `run_manifest_digest` against the immutable run manifest before accepting it.

The independent cluster key is `(contract_start, contract_stop)`, not `episode_index` alone. `episode_index` remains part of the record identity and path for deterministic per-symbol storage, but it is no longer the statistical independence key.

Runtime preparation must establish a shared chronology before any pooled fit or signal evaluation:

1. all train-symbol datasets have identical timestamp arrays;
2. all train-symbol chronological partitions have identical `(start, stop)` episode schedules;
3. all train symbols use the same decision cadence;
4. all train symbols resolve the same episode horizon.

Any mismatch fails before fitting. This closes the hidden assumption required by integer knowledge cutoffs and cross-symbol episode aggregation.

The shared timestamp array is bound by `shared_clock_digest` in the V3 execution identity.

## Signal evidence V2

Bump aggregate signal evidence to `causal_alpha_v3_signal_gate_evidence_v2`.

The evidence explicitly persists:

- `raw_scope_count`
- `expected_raw_scope_count`
- `raw_scope_coverage`
- `independent_episode_count`
- `expected_independent_episode_count`
- `independence_unit = "chronological_episode"`
- `aggregation_mode = "cross_symbol_episode_mean"`
- rank-IC bootstrap evidence
- top/bottom realized spread bootstrap evidence
- direction-accuracy-excess bootstrap evidence
- gate digest, pass state, rejection reasons, and metric digests

The clustered evaluator accepts both expected counts explicitly. Rejection reasons use explicit names:

- `independent_episode_count`
- `raw_scope_coverage`
- `rank_ic_lower_ci`
- `top_bottom_spread_lower_ci`
- `direction_accuracy_excess_lower_ci`

The legacy flat evaluator must not be a current V3 execution path and must not produce Signal Gate Evidence V2. Current runner wiring exposes only the chronological clustered evaluator.

## Signal resume contract

Signal leaf records are strict, digest-validated, run-bound artifacts. V2 makes those leaves the resume source of truth.

After the run manifest is closed, the pipeline constructs the complete expected signal-record identity map for all representative fit configs, train symbols, and signal contracts, then loads valid existing leaves once.

For each expected scope:

- a valid persisted leaf with the exact current `run_manifest_digest` is reused without rebuilding the fit/forecast metric;
- a missing leaf is recomputed and atomically written;
- an unknown leaf, wrong run manifest, wrong contract digest, wrong path, duplicate identity, invalid schema, or digest mismatch fails closed.

Aggregate gate evidence is always recomputed from the validated leaf set. Derived aggregate evidence is not the resume source of truth. A process crash after one leaf is persisted but before aggregate signal evidence is written therefore resumes from the durable leaf set and computes only missing leaves.

## Runtime provenance

Bump `CausalAlphaV3ExecutionIdentity` to schema `causal_alpha_v3_execution_identity_v2` and bind:

- existing training contract, instrument context schema, source-tree digest, train symbols, and per-symbol runtime digests;
- `shared_clock_digest`;
- `dependency_lock_digest`, computed from the source-checkout `pyproject.toml` and `uv.lock` file digests;
- `python_runtime_digest`, computed from the Python implementation and exact major/minor/micro version.

A change to source, dependency lock, Python runtime, chronology, execution/risk settings, or existing runtime identity must change the execution identity and therefore prevent reuse of an old output root.

The container image identifier remains launcher-level provenance because the process cannot reliably infer its immutable image ID from inside every supported runtime. The artifact graph still binds code, lockfile, Python runtime, and market chronology directly.

## Responsibility boundaries

- `universal_causal_alpha_v3_config.py`: authored V2 config and cross-field validation only.
- `universal_causal_alpha_v3_signal.py`: partition/leaf/evidence data contracts and non-overlap cohort construction; no current flat aggregate evaluator.
- `universal_causal_alpha_v3_signal_v2.py`: chronological cluster aggregation, strict signal-leaf loader, and V2 gate evaluation.
- `universal_causal_alpha_v3_runtime.py`: runtime preparation, shared chronology validation, source/lock/Python provenance.
- `universal_causal_alpha_v3_identity.py`: execution/run identity contracts.
- `universal_causal_alpha_v3_pipeline.py`: stage sequencing and signal-leaf resume orchestration.
- `universal_causal_alpha_v3_artifact_store.py`: persistence and strict run-bound leaf loading; no statistical policy.

## Invariants

- Same-episode symbol duplication never increases independent evidence.
- Raw coverage still measures the complete expected `(fit, symbol, signal episode)` record graph.
- A signal leaf cannot be reused under a different run manifest.
- With aligned inputs and the same set of raw metrics, clustered rank/spread/direction bootstrap numeric results remain unchanged from the existing clustered evaluator.
- Signal failure still prevents candidate freeze, economic replay, admission, and teacher package generation.
- Candidate/model/controller outputs remain unchanged.
- Holdout/economic partition boundaries remain unchanged.
- V3 remains research-only and non-promotable.

## Failure modes and test oracles

1. **Ambiguous config semantics:** V1 authored config is rejected; V2 names encode the unit.
2. **Impossible gate:** `minimum_independent_episode_count > signal_contract_count` fails config construction.
3. **Cross-symbol clock drift:** differing timestamp arrays fail runtime preparation.
4. **Cross-symbol episode drift:** differing episode `(start, stop)` schedules fail runtime preparation.
5. **Wrong cluster key:** two records with equal local `episode_index` but different intervals create separate clusters and cannot masquerade as one chronological episode.
6. **Symbol duplication:** adding another symbol record to the same interval does not increment independent episode count.
7. **Incomplete raw evidence:** missing expected symbol/episode leaves reduce raw coverage and fail a 1.0 raw-coverage gate.
8. **Partial signal crash:** valid leaves survive and are reused; only missing leaves are recomputed.
9. **Corrupt/stale signal leaf:** strict loader fails rather than recomputing over or ignoring it.
10. **Cross-run leaf copy:** a signal leaf whose `run_manifest_digest` differs from the active run is rejected before reuse.
11. **Dependency drift:** changed lock digest changes execution identity and prevents output-root reuse.
12. **Python runtime drift:** changed Python runtime digest changes execution identity.
13. **Legacy evaluator leakage:** current runner/pipeline cannot call a flat raw-record bootstrap to produce V2 evidence.

## Required test layers

- Unit: config validation, metric/evidence schema validation, cluster identity, provenance digests.
- Contract/regression: symbol duplication, different interval with same episode index, strict run-bound leaf round-trip, legacy schema rejection.
- Integration: pipeline signal resume with persisted and missing leaves; corrupt/cross-run leaf rejection; runtime chronology mismatch.
- Architecture: runner uses only clustered V2 evaluator; signal statistics remain out of persistence layer; no model/economic/reward/risk changes.
- Static: Ruff, format, Mypy, import-linter, dead-code gate.
- Compatibility: Ubuntu and Windows existing suites.
- Build/runtime: training image and packaged non-root probe.
- Full suite and branch coverage on the exact final HEAD.

## Quality gate

Do not mark the PR ready unless all acceptance criteria and invariants are covered by observable tests, RED was observed for each new behavioral contract before production changes, targeted and full tests pass, static/architecture/compatibility/build checks pass on the exact final HEAD, final diff contains no unrelated learner/economic/reward/risk changes, and a falsification review finds no remaining unit ambiguity or resume bypass.
