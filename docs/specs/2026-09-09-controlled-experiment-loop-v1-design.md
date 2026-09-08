Status: Active

# Controlled Experiment Loop v1 Design

## 1. Purpose

Trade RL already has a lean candidate runner that fits and independently replays the fixed candidate suite on a canonical dataset artifact and publishes immutable run evidence. It also has comparison, robustness, gate, and sealed outer-test primitives.

What is missing is the research unit that explains **why one run differs from another, whether the difference was actually controlled, what evidence was compared, what decision was made, and how a bounded sequence of development experiments reaches a frozen winner or an explicit no-winner outcome**.

Controlled Experiment Loop v1 adds that layer.

Its primary correctness property is not convenience. It is:

> An uncontrolled, tampered, provenance-mismatched, or selectively omitted development attempt cannot be recorded as a valid controlled experiment or selected through the winner lineage.

## 2. Scope decision

v1 ends at the development-study freeze boundary.

It manages:

1. a frozen development Study plan;
2. Study-wide implementation/runtime provenance;
3. immutable baseline run evidence;
4. sequential controlled Experiments;
5. candidate run evidence snapshots;
6. actual resolved baseline/candidate delta verification;
7. paired comparison evidence;
8. immutable Experiment decisions;
9. accepted-baseline lineage;
10. Study-level experiment budget;
11. final `WINNER` or `NO_WINNER` freeze.

It does **not** open unused future data or execute the final sealed test.

```text
Controlled Experiment Loop
    development Study
        -> Experiment 0001
        -> Experiment 0002
        -> ...
        -> WINNER / NO_WINNER freeze

================ sealed-data boundary ================

separate future subsystem
    frozen strategy artifact
        -> one-shot sealed unused-future authorization
        -> final evaluation / pre-registered stress
```

`trade_rl.evaluation.experiments` must not import or invoke the sealed outer-test authorization API.

## 3. Non-goals

v1 deliberately does not add:

- final unused-future execution;
- final stress orchestration;
- Ridge / LightGBM / PPO model serialization;
- a frozen trained-policy artifact;
- automatic hyperparameter search;
- automatic winner selection;
- distributed execution;
- database persistence;
- web UI or dashboard;
- experiment search service;
- new strategy families;
- new replay, accounting, risk, or execution semantics;
- code changes as an Experiment factor;
- dependency/runtime changes as an Experiment factor;
- LLM-authored autonomous research decisions.

The existing candidate runner remains the authority for fitting and replaying one candidate suite run. v1 may strengthen that runner's **evidence metadata** without changing fitting, replay, accounting, or strategy semantics.

## 4. Core terminology

### 4.1 Run

A **Run** is one computation performed by the existing candidate runner.

The current run artifact contains:

```text
summary.json
returns.npz
```

v1 adds one evidence-only companion file:

```text
provenance.json
```

A Run does not know why it was executed.

### 4.2 Experiment

An **Experiment** binds:

```text
one explicit hypothesis
+ one baseline Run fingerprint
+ one candidate Run fingerprint
+ one semantic controlled factor
+ verification of the actual resolved delta
+ comparison evidence
+ one immutable decision
```

An Experiment may be valid (`CONTROLLED`) or terminally invalid (`INVALID`). Invalid attempts remain part of Study history and consume the Study experiment budget.

### 4.3 Study

A **Study** is a bounded sequence of Experiments answering one research question under Study-level fixed controls and one fixed code/runtime provenance contract.

A Study owns:

- the development data/evaluation contract;
- implementation/runtime provenance;
- initial baseline;
- allowed factor types;
- maximum Experiment attempts;
- contiguous Experiment sequence;
- accepted-baseline lineage;
- final `WINNER` or `NO_WINNER` freeze.

## 5. Design principles

### 5.1 One Experiment changes one semantic research factor

The controlled unit is not "one JSON field". One semantic factor may map to multiple resolved fields.

For example, changing a feature set can change both `feature_names` and `feature_indices`, but it may not also change `fit_cutoff` or `gross_budget`.

### 5.2 Actual resolved evidence is authoritative

Requested config files are insufficient evidence. Controlled-delta verification compares the **resolved immutable run summaries**.

### 5.3 Code/runtime provenance is a Study-level fixed control

Two runs produced by different production source trees or different supported runtime environments are not a valid v1 controlled pair.

A code change, Python/runtime change, or relevant dependency-version change requires a new Study.

### 5.4 Study evidence is self-contained

External run paths are untrusted inputs to registration, not durable Study references.

Registration copies a verified byte-exact snapshot of the run evidence into a content-addressed Study evidence store. Later verification/comparison reads only that Study-owned snapshot.

### 5.5 Append-only artifacts, no mutable state file

There is no mutable `experiment.json` or `state.json` repeatedly rewritten.

State is derived from immutable artifact existence and digest references.

### 5.6 Fail closed

Unknown factors, undeclared deltas, provenance drift, mismatched dataset identity, tampered evidence, unsupported state transitions, broken lineage, sequence gaps, or post-freeze writes fail rather than being tolerated.

### 5.7 Existing research primitives remain authoritative

The subsystem reuses:

- canonical JSON and content digests from `trade_rl.artifacts`;
- verified regular-file boundaries from `trade_rl.artifacts`;
- current candidate-run computation from `trade_rl.evaluation.runs`;
- current paired-return comparison from `trade_rl.evaluation.comparison`.

It does not duplicate fitting, replay, P&L, or paired-bootstrap implementations.

## 6. Proposed package boundary

```text
trade_rl/evaluation/experiments/
├── __init__.py
├── errors.py
├── contracts/
│   ├── __init__.py
│   ├── study.py
│   ├── experiment.py
│   └── decision.py
├── artifacts/
│   ├── __init__.py
│   ├── evidence.py
│   ├── provenance.py
│   └── store.py
├── verification/
│   ├── __init__.py
│   ├── controlled_delta.py
│   └── run_evidence.py
└── workflow/
    ├── __init__.py
    ├── define.py
    ├── compare.py
    ├── decide.py
    └── freeze.py
```

Dependency direction:

```text
contracts
    ↓
artifacts
    ↓
verification
    ↓
workflow
    ↓
existing lower evaluation/artifact primitives
```

`contracts` must not depend on `workflow` or filesystem storage.

`experiments` must not depend on final sealed-test authorization.

## 7. Filesystem Study artifact

A Study is append-only and self-contained:

```text
<study-root>/
├── plan.json
├── baseline-evidence.json
├── evidence/
│   └── runs/
│       ├── <run-fingerprint-A>/
│       │   ├── manifest.json
│       │   ├── summary.json
│       │   ├── returns.npz
│       │   └── provenance.json
│       └── <run-fingerprint-B>/
│           └── ...
├── experiments/
│   ├── 0001/
│   │   ├── definition.json
│   │   ├── candidate-evidence.json
│   │   ├── verification.json
│   │   ├── comparison.json
│   │   └── decision.json
│   ├── 0002/
│   │   └── ...
│   └── ...
└── freeze.json
```

No published file is overwritten.

Experiment directory names are exactly four decimal digits from `0001` through `9999`.

Therefore v1 requires:

```text
1 <= max_experiments <= 9999
```

The sequence must be contiguous from `0001` through the highest attempt.

## 8. Canonical identity

All contract identities use the existing canonical JSON conversion and `content_digest()` SHA-256 authority.

### 8.1 Identity rules

- Study plan identity is the digest of canonical `StudyPlan` content.
- Experiment identity is the digest of canonical immutable `ExperimentDefinition` content.
- Verification, comparison, decision, and freeze identities are content digests of their canonical payloads.
- Time metadata is not inserted into Study-plan or Experiment-definition identity merely to make an object unique.
- Filesystem paths are never research identity.
- Run-file evidence uses SHA-256 + exact byte size.

### 8.2 Run fingerprint

A Run fingerprint is the canonical digest of the immutable evidence identity payload, including at least:

```text
summary SHA-256 + size
returns SHA-256 + size
provenance SHA-256 + size
resolved dataset identity
resolved dataset artifact identity
resolved candidate configuration identity
resolved evaluation scope identity
implementation digest
runtime environment digest
```

The input source path used during registration is **not persisted as research identity**.

Experiment definitions and lineage bind to the Run fingerprint, not an external path.

## 9. Run provenance prerequisite

Controlled Experiment Loop cannot prove a comparison is controlled if code/runtime provenance is absent. Therefore v1 requires a small evidence-schema extension to the candidate runner.

The candidate runner continues to publish `summary.json` and `returns.npz` with their existing semantic contents and additionally writes:

```text
provenance.json
```

### 9.1 Provenance payload

`provenance.json` contains at least:

```text
schema_version
implementation_digest
runtime_environment_digest
implementation manifest
runtime environment manifest
```

### 9.2 Implementation digest

v1 deliberately uses a conservative source-tree identity.

The implementation manifest is derived from the importable `trade_rl` production Python source tree:

1. enumerate all runtime `.py` files under the installed/imported `trade_rl` package root;
2. exclude `__pycache__`, `.pyc`, temporary/generated files, tests, docs, and filesystem absolute paths;
3. record normalized relative POSIX path + SHA-256 of exact source bytes;
4. sort by relative path;
5. compute a canonical content digest over a versioned manifest.

Any production Python source change therefore changes `implementation_digest` and requires a new Study.

This is intentionally stricter than attempting to guess which source file influenced a Run.

### 9.3 Runtime environment digest

The runtime environment manifest binds at least:

- Python implementation and full version;
- operating-system family/release;
- machine architecture;
- `trade-rl` package version;
- NumPy version;
- Gymnasium version;
- LightGBM version when used by the fixed candidate suite;
- Stable-Baselines3 version when used;
- Torch version when used;
- other direct runtime package versions that the implementation plan confirms materially participate in candidate-run computation.

The exact package list is fixed in implementation code/tests, not discovered ad hoc per Run.

The canonical digest of this manifest is `runtime_environment_digest`.

### 9.4 Study provenance rule

`create_study()` freezes the expected implementation/runtime digests before baseline registration.

Every baseline/candidate Run registered into that Study must contain identical provenance digests.

A mismatch is not a valid Experiment factor in v1; registration or verification fails closed and the researcher must start a new Study.

## 10. Evidence snapshot registration

External Run artifacts are untrusted inputs.

`register_run_evidence()`:

1. opens `summary.json`, `returns.npz`, and `provenance.json` through the existing regular-file/no-final-symlink boundary;
2. computes exact digest + size from the opened snapshots;
3. validates supported summary/provenance schema;
4. validates Study provenance compatibility;
5. constructs the Run fingerprint;
6. stages byte-exact copies under `evidence/runs/<run-fingerprint>/`;
7. writes a canonical `manifest.json` binding filenames, sizes, digests, resolved identities, and provenance digests;
8. atomically publishes the completed evidence directory.

Unsafe deserialization of NPZ evidence must occur only from a verified private/snapshotted file with `allow_pickle=False`.

If the exact Run fingerprint already exists, registration is idempotent only after verifying the stored manifest/files are byte-identical to the requested Run fingerprint. Otherwise it fails as an integrity conflict.

The Study never relies on the original external path after successful registration.

## 11. StudyPlan contract

`StudyPlan` is immutable and created before development iteration.

It contains at least:

```text
schema_version
research_question
dataset_id
dataset_artifact_digest
evaluation symbol contract
fit_cutoff
evaluation_start
evaluation_stop_exclusive
initial_capital
execution contract identifier
PPO seed policy
implementation_digest
runtime_environment_digest
max_experiments
allowed_factor_types
initial baseline requested-config identity
```

Study-level hard-fixed fields in v1:

- dataset identity;
- dataset artifact identity;
- evaluation symbol set;
- development evaluation window;
- fit cutoff;
- execution contract;
- initial capital;
- PPO seed policy;
- implementation digest;
- runtime environment digest.

Changing any of them means starting a new Study.

`fit_symbol_scope` may be an Experiment factor only if the Study explicitly permits it; the evaluation scope still remains fixed.

## 12. Controlled factor registry

v1 uses typed controlled factors, not arbitrary free-form path whitelists.

Supported factors:

```text
FEATURE_SET
RULE_SIGNAL
RULE_THRESHOLDS
FORECAST_THRESHOLDS
FIT_SYMBOL_SCOPE
PPO_TRAINING_BUDGET
GROSS_BUDGET
```

A Study may allow only a subset.

Each factor defines:

1. which resolved fields may differ;
2. which resolved fields must remain equal;
3. which strategies may be affected;
4. which strategies must remain exactly unchanged at raw-return level when the current architecture makes that invariant valid.

Unknown factor types are rejected.

### 12.1 Initial semantic delta map

The implementation plan must derive exact JSON paths from the current resolved candidate-run summary schema.

Conceptually:

| Factor | Allowed semantic delta |
|---|---|
| `FEATURE_SET` | selected feature names/indices only |
| `RULE_SIGNAL` | rule signal name/index only |
| `RULE_THRESHOLDS` | rule entry/exit threshold pair only |
| `FORECAST_THRESHOLDS` | forecast entry/exit threshold pair only |
| `FIT_SYMBOL_SCOPE` | fit symbol names/indices only |
| `PPO_TRAINING_BUDGET` | PPO total timesteps only |
| `GROSS_BUDGET` | gross budget fields currently bound by candidate Run evidence |

If the current summary schema cannot represent a proposed factor unambiguously, that factor must not be implemented until the evidence schema is deliberately improved.

### 12.2 Non-empty actual factor change

A candidate that resolves identically to the baseline is not a successful controlled Experiment merely because a factor was declared.

A valid `CONTROLLED` verification requires at least one actual resolved delta within the declared factor's allowed field set.

No-op candidate definitions become terminal `INVALID` once trustworthy evidence proves the declared factor produced no resolved change.

## 13. ExperimentDefinition contract

An Experiment is defined before candidate evidence is observed.

`ExperimentDefinition` contains at least:

```text
schema_version
study_digest
sequence
hypothesis
baseline_run_fingerprint
factor type
candidate requested configuration
```

Requirements:

- sequence is exactly the next Study attempt number;
- hypothesis is non-empty;
- factor is permitted by StudyPlan;
- baseline fingerprint exists and is lineage-eligible;
- candidate requested config is complete enough for the existing runner;
- Study is not frozen;
- Study budget remains.

The definition is immutable once published.

## 14. Derived Experiment state

No mutable status artifact exists.

```text
definition.json only
    = DEFINED

+ candidate-evidence.json
    = EVIDENCE_BOUND

+ verification.json with CONTROLLED
    = VERIFIED

+ comparison.json
    = COMPARED

+ decision.json
    = DECIDED
```

A `verification.json` with `INVALID` is terminal:

```text
INVALID
```

An INVALID Experiment has no comparison or decision artifact.

No workflow operation may skip a required prior stage.

## 15. Baseline and candidate evidence bindings

`baseline-evidence.json` binds the Study to the initial Run fingerprint.

Each `candidate-evidence.json` binds one Experiment to exactly one registered candidate Run fingerprint.

Binding validates:

- the Run fingerprint exists in the Study evidence store;
- its manifest/files pass digest + size re-verification;
- its dataset/evaluation/provenance matches StudyPlan;
- a candidate binding is not already published for the Experiment.

Binding does not claim the candidate is controlled.

## 16. Actual controlled-delta verification

`verify_experiment()` compares the resolved baseline and candidate summaries from the **Study-owned evidence snapshots**.

Checks include at least:

```text
run evidence integrity
study identity binding
implementation provenance equality
runtime provenance equality
dataset identity equality
dataset artifact identity equality
evaluation symbol roster equality
evaluation window equality
fit cutoff equality
other Study-fixed controls equality
strategy roster equality
non-empty declared-factor actual delta only
unaffected-strategy raw return invariance where required
```

### 16.1 CONTROLLED

If all required checks pass, publish `verification.json` with status `CONTROLLED`.

### 16.2 INVALID

If trustworthy evidence exists and proves the controlled contract was violated, publish terminal `verification.json` with status `INVALID` and explicit failed checks.

Examples:

- `FEATURE_SET` also changed `fit_cutoff`;
- dataset identity drifted;
- implementation/runtime provenance changed;
- evaluation window drifted;
- strategy roster changed;
- declared factor produced no resolved change;
- an unaffected strategy return series changed.

INVALID consumes one Study attempt.

### 16.3 Operation failure versus INVALID result

Use an exception and publish no verification artifact when trustworthy evidence cannot be established, for example:

- evidence file missing;
- digest/size mismatch;
- malformed unsupported artifact;
- unsafe snapshot failure;
- filesystem publication failure.

Use INVALID only when trustworthy registered evidence exists and proves the Experiment contract was violated.

## 17. Unaffected-strategy falsification oracle

The factor registry identifies strategies that should be invariant to a factor under the current architecture.

When a strategy is declared unaffected, baseline/candidate raw return arrays and any required immutable replay evidence must match exactly.

A mismatch is evidence of hidden-condition drift or semantic regression and makes the Experiment INVALID.

The implementation plan must derive the exact affected/unaffected sets from current code/data flow and contract tests. They may not be guessed from strategy names.

## 18. Comparison contract

Only a CONTROLLED Experiment can be compared.

Comparison reuses the existing return-series and paired-comparison primitives.

Evidence is stored for every expected:

```text
symbol × strategy
```

At minimum `ExperimentComparison` records:

- baseline/candidate performance metrics;
- total-return delta;
- Sharpe delta;
- maximum-drawdown delta;
- turnover delta;
- total execution-cost delta;
- funding P&L delta;
- borrow-cost delta;
- paired excess total return;
- paired excess log return;
- moving-block bootstrap confidence interval;
- bootstrap p-value;
- block size;
- baseline/candidate Run fingerprints;
- verification digest.

The complete expected symbol × strategy Cartesian product is required. Missing, duplicate, or extra cells fail closed.

A statistical threshold does not automatically determine the research decision.

## 19. Experiment decision contract

Experiment decisions are exactly:

```text
ACCEPT_CANDIDATE
KEEP_BASELINE
INCONCLUSIVE
```

There is no mutable `RETEST` decision. A retest is a new Experiment and consumes another attempt.

`ExperimentDecision` contains at least:

```text
schema_version
study_digest
experiment_digest
verification_digest
comparison_digest
decision
rationale
decided_by
decided_at
```

`rationale` and `decided_by` are non-empty. `decided_at` is timezone-aware.

The decision artifact is immutable and published once.

The subsystem does not automatically decide from p-values or one aggregate metric.

## 20. Accepted-baseline lineage

A later Experiment baseline is eligible only if it is:

1. the initial Study baseline; or
2. a candidate Run from a prior `ACCEPT_CANDIDATE` decision reachable from the initial baseline.

Candidates from `KEEP_BASELINE`, `INCONCLUSIVE`, or `INVALID` Experiments are not lineage eligible.

This prevents rejected candidates from being silently resurrected later as the winner path.

## 21. Experiment budget and sequence integrity

Every Experiment definition consumes one Study attempt, including INVALID attempts.

Rules:

- sequence is contiguous from 1;
- directories are exactly `0001` through `9999`;
- existing sequences cannot be overwritten;
- a deleted/gapped sequence makes the Study inconsistent and prevents freeze;
- public API cannot renumber a sequence;
- `max_experiments` is fixed in StudyPlan;
- defining `max_experiments + 1` raises `ExperimentBudgetExceededError`.

The budget bounds adaptive reuse of the same development window. Individually controlled Experiments do not eliminate Study-level development overfitting.

## 22. Study freeze

A Study freeze is terminal and immutable.

Allowed outcomes:

```text
WINNER
NO_WINNER
```

### 22.1 WINNER

Requires:

```text
selected_run_fingerprint
selected_strategy
rationale
```

The selected Run must be reachable through ACCEPT lineage and the selected strategy must exist in that Run.

### 22.2 NO_WINNER

Must not contain a selected Run or selected strategy.

### 22.3 Freeze preconditions

Before `freeze.json` publication verify:

- StudyPlan integrity;
- Study provenance contract;
- baseline evidence integrity;
- all content-addressed Run evidence integrity;
- contiguous Experiment sequence;
- attempts <= budget;
- every Experiment terminal (`DECIDED` or `INVALID`);
- all digest/fingerprint references resolve;
- selected winner Run, if any, is ACCEPT-lineage eligible;
- selected strategy exists;
- no invalid/rejected candidate was resurrected;
- no existing freeze artifact.

After freeze, all mutating Study operations reject with `StudyFrozenError`.

## 23. StudyStore responsibility

`StudyStore` is the only stateful helper class proposed for v1.

It owns filesystem mechanics only:

- validated path resolution;
- exclusive creation;
- verified snapshot copying;
- content-addressed evidence placement;
- staging;
- atomic publication;
- verified reads;
- artifact existence checks;
- failed staging cleanup.

It must not own research decisions, factor semantics, comparison policy, or mutable Study state.

## 24. Public API

High-level package surface:

```text
create_study
register_run_evidence
define_experiment
bind_candidate_evidence
verify_experiment
compare_experiment
decide_experiment
freeze_study
inspect_study
```

Public immutable contracts:

```text
StudyPlan
ControlledFactor
ExperimentDefinition
RunEvidence
ControlledVerification
ExperimentComparison
ExperimentDecision
StudyFreeze
StudySnapshot
```

Private codecs, provenance builders, field maps, filesystem helpers, and adapters are not package-root exports by default.

No public API mutates fields on an in-memory Study object.

## 25. Error model

All subsystem-specific operational errors derive from:

```text
ControlledExperimentError
```

Required subclasses:

```text
ContractViolationError
ArtifactIntegrityError
InvalidExperimentStateError
UncontrolledDeltaError
ExperimentBudgetExceededError
StudyFrozenError
```

Semantics:

- `ContractViolationError`: invalid contract input or unsupported factor.
- `ArtifactIntegrityError`: digest, size, symlink, malformed, missing, provenance, or evidence-store conflict failure.
- `InvalidExperimentStateError`: operation attempted before required prior artifact or after a terminal state.
- `UncontrolledDeltaError`: domain-level undeclared/no-op factor delta; workflow verification normally materializes trustworthy violations as terminal INVALID evidence.
- `ExperimentBudgetExceededError`: attempt exceeds frozen Study budget.
- `StudyFrozenError`: mutation attempted after freeze.

The implementation must preserve the distinction between evidence proving an Experiment INVALID and an operation failing before trustworthy verification is possible.

## 26. Integrity boundary

Research evidence is untrusted until registered into the Study-owned content-addressed store.

The implementation must account for:

- symlink substitution;
- file replacement during registration;
- size/digest mismatch;
- malformed JSON;
- malformed/unsafe NPZ;
- missing NPZ keys;
- duplicate/missing symbol-strategy cells;
- source/runtime provenance drift;
- partial filesystem publication;
- overwrite attempts;
- evidence fingerprint collision/conflict;
- sequence directory collision;
- post-freeze mutation.

No final-test dataset accessor, sealed-test ledger call, or unused-future opening capability belongs in this package.

## 27. Quality contract

### Objective

Create a development-only Controlled Experiment Loop that can prove what changed between baseline and candidate Runs, prove they share the Study's code/runtime/data/evaluation controls, retain self-contained immutable evidence, compare complete symbol-level results, record decisions, enforce bounded accepted lineage, and freeze a winner/no-winner outcome without opening final unused data.

### Non-goals

Section 3 is binding and must not expand silently during implementation.

### Invariants

1. Existing candidate fitting/replay semantics remain unchanged.
2. Existing `MarketExecutor + BookState` economic authority remains unchanged.
3. Study hard-fixed data/evaluation/provenance controls cannot drift.
4. One Experiment changes one declared semantic factor only.
5. The declared factor must produce a non-empty actual resolved delta.
6. Actual resolved evidence, not requested config alone, determines control validity.
7. Registered evidence is byte-exact, Study-owned, content-addressed, and immutable.
8. INVALID attempts remain visible and consume budget.
9. Experiment definitions and terminal artifacts are immutable.
10. Rejected/inconclusive/invalid candidates are not winner-lineage eligible.
11. Study freeze is terminal.
12. Experiment Loop cannot open final unused data.

### Failure modes

At minimum:

- malformed Study plan;
- unsupported/duplicate factor settings;
- wrong next sequence;
- budget exhaustion;
- missing baseline;
- broken baseline lineage;
- source evidence tampering during registration;
- Study evidence tampering after registration;
- symlink evidence;
- implementation digest drift;
- runtime environment drift;
- dataset drift;
- evaluation-window drift;
- fit-cutoff drift;
- symbol roster drift;
- strategy roster drift;
- undeclared resolved delta;
- no-op declared factor;
- unaffected-strategy return drift;
- incomplete/duplicate/extra symbol × strategy comparison;
- compare-before-verify;
- decide-before-compare;
- second decision publication;
- open Experiment at freeze;
- sequence gap at freeze;
- rejected candidate selected as winner;
- invalid WINNER/NO_WINNER field combination;
- mutation after freeze;
- partial publication / cleanup failure.

### Risk

High. A false-positive validity result can make research conclusions depend on an uncontrolled or adaptively cherry-picked development comparison. Integrity, provenance, lineage, and state-transition violations are fail-closed.

### Test Oracle

Correctness is observed through:

- canonical content digest equality;
- source/runtime provenance digest equality;
- exact file SHA-256 and byte size;
- byte equality between verified input snapshot and Study evidence copy;
- filesystem state transitions;
- immutable/exclusive publication;
- resolved baseline/candidate delta classification;
- unaffected-strategy raw-return equality;
- complete symbol × strategy comparison coverage;
- deterministic paired-comparison evidence under fixed seed;
- Experiment sequence and budget accounting;
- accepted-baseline graph reachability;
- freeze precondition reconstruction from disk;
- static absence of final-test dependency.

### Required Test Layers

- architecture/static dependency tests;
- contract unit tests;
- provenance unit/property tests;
- property tests for canonical identity and allowed/unallowed delta partitioning;
- artifact integration tests with real temporary files;
- filesystem race/failure/cleanup tests;
- existing candidate-run compatibility/regression tests;
- paired comparison tests;
- workflow/state-transition tests;
- lineage/freeze tests;
- tamper/symlink/security regression tests;
- Ruff;
- format check;
- Mypy;
- full repository pytest suite;
- package identity check;
- exact-head GitHub CI;
- requirements-first falsification review.

## 28. TDD strategy

Implementation uses RED -> GREEN -> Refactor.

The first production change must be preceded by failing contract/architecture tests for:

1. required `evaluation/experiments` boundary;
2. forbidden dependency on sealed final-test authorization;
3. StudyPlan/factor/budget invariants;
4. candidate-run provenance evidence requirement;
5. content-addressed evidence snapshot binding;
6. actual controlled-delta classification;
7. state-transition ordering;
8. accepted-baseline lineage;
9. Study freeze rules.

Tests are not weakened to accommodate shortcuts.

## 29. Acceptance criteria

v1 is specification-complete only when all are true:

1. `trade_rl.evaluation.experiments` exists as a distinct responsibility boundary.
2. Experiment code has no dependency on sealed final-test authorization.
3. StudyPlan has deterministic canonical identity and immutable publication.
4. StudyPlan freezes implementation and runtime environment digests before baseline registration.
5. Candidate runner publishes evidence-only provenance without changing fitting/replay/accounting semantics.
6. Run registration verifies and snapshots summary/returns/provenance bytes into the Study evidence store.
7. Run fingerprint is independent of external source path.
8. Symlink/tampered/missing/malformed evidence is rejected fail closed.
9. Evidence-store reuse is allowed only for an identical verified Run fingerprint.
10. ExperimentDefinition is immutable and bound to Study, sequence, hypothesis, factor, and baseline Run fingerprint.
11. Exactly one registered semantic factor governs allowed resolved delta.
12. Verification compares actual resolved registered evidence rather than requested config only.
13. Study-fixed data/evaluation/provenance controls cannot drift inside a valid Experiment.
14. Undeclared delta cannot produce CONTROLLED verification.
15. A declared factor that produces no resolved change cannot produce CONTROLLED verification.
16. Unaffected-strategy drift is detected wherever the factor contract declares invariance.
17. Trustworthy uncontrolled evidence produces terminal INVALID history rather than disappearing.
18. INVALID attempts consume Study budget.
19. Experiment sequence is contiguous, four-digit, and cannot be overwritten/reused.
20. `max_experiments` is fixed in `[1, 9999]` and cannot increase after creation.
21. Only CONTROLLED Experiments can produce comparison evidence.
22. Comparison covers the complete expected symbol × strategy matrix with no missing/duplicate/extra cell.
23. Comparison reuses existing paired-return/statistical primitives.
24. ExperimentDecision is immutable and bound to verification + comparison identities.
25. Decisions are limited to ACCEPT_CANDIDATE / KEEP_BASELINE / INCONCLUSIVE.
26. Only the initial baseline or ACCEPT_CANDIDATE lineage can serve as future baseline.
27. Rejected/inconclusive/invalid candidates cannot be selected through winner lineage.
28. Study freeze validates all terminal attempts, Run evidence, provenance, lineage, and references before publication.
29. WINNER requires a valid accepted Run + existing strategy; NO_WINNER forbids them.
30. Freeze makes the Study terminal against further mutation.
31. Existing candidate runner computation semantics and existing evaluation package APIs remain compatible.
32. No final-unused-data or final stress execution is introduced.
33. Current architecture/research docs are updated if the implemented evidence/public boundary changes them.
34. Targeted tests, full tests, Ruff, format, Mypy, package identity, and exact-head CI pass.
35. Final diff contains no temporary migration/debug workflow or generated output.
36. Independent/falsification review finds no unresolved Critical or High contract violation.

## 30. Follow-up sub-projects

### 30.1 Frozen Strategy Artifact

The candidate runner records configuration/development evidence but does not persist the fitted Ridge/LightGBM/PPO object as the reusable trained-policy identity.

A later sub-project must define a safe, versioned, immutable frozen-strategy artifact before claiming that the exact fitted winner can be carried unchanged into unused-future evaluation.

### 30.2 Sealed final evaluation

After a Study is frozen and a frozen-strategy artifact exists, a separate design cycle may bind Study/frozen-strategy identity to the existing one-shot sealed-test authorization and pre-registered stress protocol.

That subsystem is intentionally outside v1.

## 31. Resolved design decisions

The following choices are normative for v1 unless this Active spec is explicitly revised before implementation:

- development-only scope through Study freeze;
- Study / Experiment / Run as separate concepts;
- one semantic controlled factor per Experiment;
- typed factor registry;
- append-only immutable state artifacts;
- Study-owned content-addressed Run evidence snapshots;
- candidate-run `provenance.json` evidence companion;
- full production Python source-tree digest as conservative implementation identity;
- fixed canonical runtime-environment digest;
- code/runtime changes require a new Study;
- actual resolved summary delta as authority;
- non-empty actual factor delta required;
- raw unaffected-strategy evidence as adversarial oracle where valid;
- INVALID attempts retained and budgeted;
- `1 <= max_experiments <= 9999`;
- contiguous four-digit sequence numbers;
- accepted-baseline lineage;
- no automatic winner selection;
- exactly three Experiment decisions;
- WINNER / NO_WINNER Study outcomes;
- no final-test access from the package;
- frozen trained-strategy serialization deferred to a separate sub-project.
