# Causal Alpha V3 selection diagnostics hardening design

## Objective

Preserve the already-hardened Causal Alpha V3 research workflow while recovering the useful live-run observability that existed on the diverged `codex/causal-alpha-v3-real-checkpoint` line. Make economic-selection failures diagnosable at candidate × symbol × episode scope before changing predictor/controller behavior.

## Context

The diverged branch was recreated after the earlier cleanup and accumulated six commits implementing a JSONL/progress-oriented V3 runner. It was 125 commits behind maintained `main`, while maintained `main` already contained the stricter artifact-bound V3 runner with atomic per-scope records, execution identity closure, single-writer locking, signal leaf evidence, hardened admission, and reloadable teacher artifacts.

The branch history is therefore integrated as a second parent without adopting its stale tree. Its live-run checkpoint remains historical/non-promotable evidence. The maintained implementation continues to use atomic records as the source of truth.

## Non-goals

- Do not change canonical U6 reward, action mode, teacher, risk limits, execution delay, or admission ordering.
- Do not relax V3 signal, economic-selection, or admission gates.
- Do not tune against teacher-admission holdout, validation/test symbols, or sealed evaluation.
- Do not start DAgger, BC, critic warm start, PPO, or Lagrangian training.
- Do not claim the currently running legacy-branch Docker result is promotable evidence for maintained `main`.
- Do not add rolling-window models, regime adapters, or asymmetric long/short thresholds until candidate-level evidence identifies a root cause.

## Architecture

### 1. Immutable replay diagnostics leaf

Add a separate `CausalAlphaV3ReplayDiagnostics` artifact instead of changing `CausalAlphaV3ReplayMetric` schema. This preserves existing replay-resume compatibility and keeps economic-selection semantics unchanged.

Each diagnostics leaf is bound to:

- run manifest digest;
- freeze digest;
- candidate digest;
- symbol and episode index;
- contract digest;
- replay metric digest;
- fit / forecast / target-path digests.

It records only deterministic summaries already available from the target path:

- decision count;
- long / short / flat target counts;
- positive / negative / near-zero forecast counts;
- mean and maximum absolute target;
- mean expected return;
- mean and p90 uncertainty;
- mean absolute signal-to-uncertainty ratio;
- mean liquidity cap;
- mean chosen-vs-stay objective improvement;
- target reason counts.

No new replay or counterfactual simulation is performed.

### 2. Derived selection progress artifact

Write `selection/progress.json` after loading existing records and after every newly persisted replay. It is a derived monitoring artifact; atomic replay records remain authoritative.

Progress contains:

- expected/completed replay counts and completion fraction;
- fit-cache misses/hits;
- per-candidate completed scope count, irreversible rejection state, current mean gross/net, worst net, mean turnover, trade count;
- per-symbol completed scope count and descriptive gross/net/turnover aggregates;
- diagnostics coverage count;
- research-only / promotion-ineligible markers.

The progress artifact must never be consumed as selection/admission evidence.

### 3. Legacy live-run diagnosis boundary

The existing JSONL live run is retained only as historical evidence. The maintained path will not resume from that JSONL. Documentation must state that a legacy JSONL result can guide diagnosis, but a promotable V3 result must be rerun through the maintained atomic-record runner with current run/execution/code identities.

### 4. Root-cause decision rule

No model/controller change is authorized by this patch. After a maintained run, use diagnostics to decide the next experiment:

- gross negative with low turnover -> predictor/regime problem;
- gross positive but net negative -> execution/controller problem;
- tail failure with high uncertainty -> uncertainty calibration problem;
- systematic long/short imbalance -> asymmetric threshold experiment;
- horizon disagreement concentrated in failures -> horizon/rolling-window experiment.

Any such follow-up must use earlier selection data only and receive a separate TDD/quality contract.

## Invariants

- `CausalAlphaV3ReplayMetric` payload and digest semantics remain unchanged.
- Atomic replay records remain resume authority.
- Selection ranking order and all threshold values remain unchanged.
- Existing irreversible rejection semantics remain unchanged.
- Admission holdout remains unopened until selection succeeds.
- Diagnostics are research-only and `promotion_eligible=false`.
- Missing diagnostics never convert an invalid replay into a valid replay; corrupt diagnostics fail closed when explicitly loaded.

## Failure modes

- Diagnostics written with a different replay identity: reject.
- Duplicate/conflicting diagnostics for one replay: reject.
- Progress interrupted between replay write and progress refresh: safe; rebuild progress from authoritative replay records on restart.
- Legacy JSONL mixed with maintained atomic records: never merge as one evidence set.
- Candidate is pruned early: progress records the rejection based on persisted replay evidence; diagnostics are descriptive only.

## Test oracle

Correctness is observed through:

- exact digest/identity checks on diagnostics round trip;
- deterministic target-summary values for known arrays;
- progress reconstruction from a synthetic set of replay metrics;
- progress refresh after resume and after a new replay;
- selection winner/rejection unchanged with diagnostics enabled vs disabled;
- canonical U6 contract tests unchanged;
- architecture/static/full test suite and build checks.

## Required test layers

- Unit: diagnostics summary and strict schema validation.
- Contract: artifact identity and immutable write/reload.
- Integration: selection replay writes metric, diagnostics, and progress in correct order; resume rebuilds progress.
- Regression: existing selection ranking/gates and canonical U6 invariants.
- Static: Ruff, format, Mypy, import architecture, dead-code.
- Full: repository pytest/coverage and training image checks through CI.

## Quality gate

Complete only when targeted tests, related V3 workflow tests, full repository tests/coverage, static checks, architecture checks, training image/compatibility CI, and final exact-head CI succeed; final diff and branch/main relationship are reviewed; remaining empirical risks are documented.
