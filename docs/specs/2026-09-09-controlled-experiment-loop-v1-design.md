Status: Active

# Controlled Experiment Loop v1 Design

## 1. Purpose

Trade RL already has a lean candidate runner that fits and independently replays the fixed candidate suite on a canonical dataset artifact and publishes immutable run evidence. It also has comparison, robustness, gate, and sealed outer-test primitives.

What is missing is the research unit that explains **why one run differs from another, whether the difference was actually controlled, what evidence was compared, what decision was made, and how a bounded sequence of development experiments reaches a frozen winner or an explicit no-winner outcome**.

Controlled Experiment Loop v1 adds that layer.

Its primary correctness property is:

> An uncontrolled, tampered, provenance-mismatched, selectively omitted, or lineage-invalid development attempt cannot be recorded as a valid controlled experiment or selected as the Study winner.

## 2. Scope

v1 ends at the development-study freeze boundary.

It manages:

1. a frozen development Study plan;
2. Study-wide code/runtime provenance;
3. a pre-resolved initial baseline configuration;
4. immutable baseline run evidence;
5. sequential controlled Experiments;
6. candidate run evidence snapshots;
7. actual resolved baseline/candidate delta verification;
8. paired comparison evidence;
9. immutable Experiment decisions;
10. accepted-baseline lineage;
11. Study-level experiment budget;
12. final `WINNER` or `NO_WINNER` freeze.

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

`trade_rl.evaluation.experiments` must not import or invoke sealed outer-test authorization.

## 3. Non-goals

v1 does not add:

- final unused-future execution;
- final stress orchestration;
- fitted Ridge / LightGBM / PPO serialization;
- a frozen trained-policy artifact;
- automatic hyperparameter search;
- automatic winner selection;
- code changes as an Experiment factor;
- dependency/runtime changes as an Experiment factor;
- distributed execution;
- database persistence;
- web UI/dashboard;
- new strategy families;
- new replay/accounting/risk/execution semantics;
- autonomous LLM research decisions.

The existing candidate runner remains the authority for fitting and replaying one suite Run. v1 may strengthen its **evidence metadata** without changing computation semantics.

## 4. Core terminology

### Run

A **Run** is one computation performed by the current candidate runner.

The current artifact has:

```text
summary.json
returns.npz
```

v1 adds an evidence-only companion:

```text
provenance.json
```

A Run does not encode the research reason for executing it.

### Experiment

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

A trustworthy attempt is either `CONTROLLED` or terminal `INVALID`. INVALID attempts remain in Study history and consume budget.

### Study

A **Study** is a bounded sequence of Experiments answering one research question under one fixed development data/evaluation/provenance contract.

## 5. Design principles

1. **One semantic factor per Experiment.** A factor may map to multiple resolved fields but unrelated factors cannot be mixed.
2. **Resolved evidence is authoritative.** Requested JSON alone cannot prove control.
3. **Code/runtime provenance is fixed for the Study.** A production-code or relevant runtime change requires a new Study.
4. **Study evidence is self-contained.** External Run paths are untrusted registration inputs, not durable references.
5. **Append-only state.** No mutable `state.json` or rewritten Experiment artifact.
6. **INVALID remains visible.** A trustworthy failed-control attempt cannot disappear from budget/history.
7. **Fail closed.** Unknown factor, provenance drift, undeclared delta, evidence tampering, broken lineage, sequence gap, concurrency conflict, or post-freeze write is not tolerated.
8. **Existing lower authorities remain authoritative.** Do not reimplement candidate fitting, replay, accounting, or paired bootstrap.

## 6. Package boundary

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

`contracts` does not depend on workflow/filesystem storage. `experiments` does not depend on final sealed-test authorization.

## 7. Study filesystem artifact

```text
<study-root>/
├── plan.json
├── baseline-evidence.json
├── evidence/
│   └── runs/
│       └── <run-fingerprint>/
│           ├── manifest.json
│           ├── summary.json
│           ├── returns.npz
│           └── provenance.json
├── experiments/
│   ├── 0001/
│   │   ├── definition.json
│   │   ├── candidate-evidence.json
│   │   ├── verification.json
│   │   ├── comparison.json
│   │   └── decision.json
│   └── ...
└── freeze.json
```

Published files are never overwritten.

Experiment names are exactly four decimal digits. v1 therefore enforces:

```text
1 <= max_experiments <= 9999
```

Sequences are contiguous from `0001` to the highest attempt.

## 8. Identity

All contract identities use existing canonical JSON + `content_digest()`.

Rules:

- Study identity = canonical StudyPlan digest.
- Experiment identity = canonical ExperimentDefinition digest.
- Verification/comparison/decision/freeze identities = canonical content digests.
- Filesystem paths are never research identity.
- Run file evidence binds SHA-256 + exact byte size.
- timestamps are not added to Study/Experiment identity just to create uniqueness.

### Run fingerprint

The Run fingerprint digests an immutable identity payload containing at least:

```text
summary SHA-256 + size
returns SHA-256 + size
provenance SHA-256 + size
resolved dataset identity
resolved dataset artifact identity
resolved candidate configuration identity
resolved evaluation scope identity
strategy roster identity
implementation digest
runtime environment digest
```

External source paths are excluded. Experiment lineage binds Run fingerprints.

## 9. Candidate-run provenance prerequisite

Without generation-time provenance, two results produced by different code/runtime environments cannot be proven controlled.

The candidate runner therefore keeps `summary.json` / `returns.npz` semantics and additionally publishes `provenance.json`.

### Implementation digest

Use a conservative production source-tree digest:

1. enumerate runtime `.py` files under the imported `trade_rl` package root;
2. exclude `__pycache__`, bytecode, tests, docs, absolute paths, temporary/generated debris;
3. record normalized relative POSIX path + SHA-256 of exact source bytes;
4. sort by relative path;
5. canonical-digest a versioned manifest.

Any production Python source change changes the digest and requires a new Study. This strictness is intentional.

### Runtime environment digest

A versioned canonical manifest binds at least:

- Python implementation/full version;
- OS family/release;
- machine architecture;
- `trade-rl` version;
- NumPy;
- Gymnasium;
- LightGBM;
- Stable-Baselines3;
- Torch;
- any additional direct runtime package the implementation review proves participates materially in the fixed suite.

The package list is fixed in code/tests, not discovered differently per Run.

`create_study()` freezes expected implementation/runtime digests before baseline registration. Every registered Run must match. Code/runtime changes are not Experiment factors in v1.

## 10. Study-owned Run evidence registration

External Run directories are untrusted inputs.

`register_run_evidence()`:

1. verifies `summary.json`, `returns.npz`, `provenance.json` as non-symlink regular files;
2. reads a stable snapshot and computes exact digest/size;
3. validates supported summary/provenance schemas;
4. validates Study provenance compatibility;
5. validates resolved dataset/evaluation compatibility;
6. builds the Run fingerprint;
7. stages byte-exact copies under `evidence/runs/<run-fingerprint>/`;
8. writes canonical `manifest.json`;
9. atomically publishes the complete evidence directory.

Unsafe NPZ reading happens only from a verified Study-owned/private snapshot and uses `allow_pickle=False`.

If an identical Run fingerprint already exists, registration is idempotent only after re-verifying all stored bytes/manifest. A conflicting existing directory is an integrity failure.

After successful registration the Study never depends on the original external Run path.

## 11. StudyPlan contract

StudyPlan is created before baseline results are admitted into the Study.

Creation inputs include a canonical dataset artifact and the initial baseline run configuration. Study creation uses the **same configuration-resolution authority as the candidate runner** to validate/resolve names, timestamps, scopes, and numeric fields without fitting models.

If necessary, implementation may extract the current private run-config resolution into one shared lower helper; it must not create two independent resolution implementations.

StudyPlan contains at least:

```text
schema_version
research_question
dataset_id
dataset_artifact_digest
evaluation symbol set
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
initial_baseline_resolved_config
candidate_strategy_names
control_strategy_names
```

The initial baseline Run later registered must resolve exactly to `initial_baseline_resolved_config` and the Study fixed controls.

### Study-fixed fields

- dataset identity/artifact identity;
- evaluation symbol set/window;
- fit cutoff;
- execution contract;
- initial capital;
- PPO seed policy;
- implementation digest;
- runtime environment digest;
- strategy/control roster contract.

Changing them requires a new Study.

`fit_symbol_scope` can be an Experiment factor only when explicitly allowed; evaluation scope remains fixed.

### Candidate/control roster

v1 preserves the current research distinction:

```text
candidates:
  trend
  mean_reversion
  ridge24
  lightgbm24
  ppo

controls:
  cash
  constant_long
  constant_short
```

The implementation plan must verify this roster against the current candidate-suite authority before coding. Strategy roster drift inside one Study is invalid.

## 12. Controlled factor registry

Supported semantic factors:

```text
FEATURE_SET
RULE_SIGNAL
RULE_THRESHOLDS
FORECAST_THRESHOLDS
FIT_SYMBOL_SCOPE
PPO_TRAINING_BUDGET
GROSS_BUDGET
```

A Study permits only an explicit subset.

Each factor defines:

1. resolved fields allowed to differ;
2. Study/factor fields required equal;
3. strategies permitted to be affected;
4. strategies required exactly invariant at raw-return level where the current code/dataflow makes this a valid oracle.

Unknown factors fail closed.

The implementation plan derives exact JSON paths from the current resolved run summary.

Conceptual mapping:

| Factor | Allowed semantic delta |
|---|---|
| `FEATURE_SET` | feature names/indices |
| `RULE_SIGNAL` | rule signal name/index |
| `RULE_THRESHOLDS` | rule entry/exit threshold pair |
| `FORECAST_THRESHOLDS` | forecast entry/exit threshold pair |
| `FIT_SYMBOL_SCOPE` | fit symbol names/indices |
| `PPO_TRAINING_BUDGET` | PPO total timesteps |
| `GROSS_BUDGET` | currently evidence-bound gross budget fields |

If current evidence cannot express a factor unambiguously, that factor is not implemented until the evidence schema is intentionally improved.

A declared factor must produce a **non-empty actual resolved delta**. A no-op candidate becomes terminal INVALID once trustworthy evidence proves no allowed field actually changed.

## 13. ExperimentDefinition

Published before candidate evidence is admitted.

Contains at least:

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

- exact next sequence;
- non-empty hypothesis;
- Study-permitted factor;
- existing lineage-eligible baseline;
- candidate config valid enough for current runner;
- Study not frozen;
- budget remains.

Definition publication is immutable.

## 14. Derived Experiment state

No mutable status file.

```text
definition.json                              -> DEFINED
+ candidate-evidence.json                   -> EVIDENCE_BOUND
+ verification.json(status=CONTROLLED)      -> VERIFIED
+ comparison.json                           -> COMPARED
+ decision.json                             -> DECIDED

verification.json(status=INVALID)           -> INVALID (terminal)
```

INVALID has no comparison/decision artifact. Workflow stages cannot be skipped.

## 15. Baseline/candidate binding

`baseline-evidence.json` binds the Study to its initial Run fingerprint.

The initial baseline must match StudyPlan's resolved baseline config, fixed controls, candidate/control roster, and provenance.

Each `candidate-evidence.json` binds one Experiment to one registered Run fingerprint after evidence-store re-verification.

Binding does not yet claim the candidate is controlled.

## 16. Controlled-delta verification

`verify_experiment()` compares **Study-owned registered evidence**.

Checks include:

```text
run evidence integrity
Study identity binding
implementation/runtime provenance equality
dataset/artifact identity equality
evaluation symbol roster/window equality
fit cutoff equality
other Study-fixed controls equality
strategy/control roster equality
non-empty declared-factor actual delta only
unaffected-strategy raw-return invariance where required
```

### CONTROLLED

All mandatory checks pass -> publish immutable `verification.json` with `CONTROLLED`.

### INVALID

Trustworthy evidence proves a controlled-contract violation -> publish terminal `INVALID` with explicit failed checks.

Examples:

- undeclared `fit_cutoff` change;
- provenance drift;
- dataset/window drift;
- strategy roster drift;
- declared factor is a no-op;
- supposedly unaffected strategy returns changed.

INVALID consumes the attempt.

### Operation failure vs INVALID

Do **not** publish INVALID when trustworthy evidence cannot be established (missing file, digest mismatch, malformed unsupported artifact, unsafe snapshot failure, filesystem publication failure). Raise an operational integrity/state error and leave verification unpublished.

## 17. Unaffected-strategy falsification oracle

For factors where the current architecture proves some strategies independent of the changed factor, their raw return arrays must match exactly baseline vs candidate.

Mismatch implies hidden-condition drift or semantic regression -> INVALID.

The implementation plan derives exact affected/unaffected sets from current code/dataflow and contract tests. Names alone are insufficient evidence.

## 18. Comparison

Only CONTROLLED Experiments can compare.

Reuse existing return-series/paired-comparison primitives.

For every expected `symbol × strategy` cell record at least:

- baseline/candidate metrics;
- total-return delta;
- Sharpe delta;
- max-drawdown delta;
- turnover delta;
- execution-cost delta;
- funding delta;
- borrow delta;
- paired excess total/log return;
- moving-block bootstrap CI/p-value/block size;
- baseline/candidate Run fingerprints;
- verification digest.

Missing, duplicate, or extra symbol×strategy cells fail closed.

No p-value or aggregate metric automatically determines the decision.

## 19. Experiment decision

Exactly:

```text
ACCEPT_CANDIDATE
KEEP_BASELINE
INCONCLUSIVE
```

A retest is a new Experiment and budget attempt.

Decision binds:

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

`rationale`/`decided_by` are non-empty; `decided_at` is timezone-aware. Decision is immutable and published once.

## 20. Accepted-baseline lineage

A later baseline is eligible only if it is:

1. the initial Study baseline; or
2. a candidate Run from a prior `ACCEPT_CANDIDATE` decision reachable from the initial baseline.

KEEP_BASELINE, INCONCLUSIVE, and INVALID candidates never become lineage eligible.

## 21. Budget and sequence

Every Experiment definition consumes one attempt, including eventual INVALID attempts.

- contiguous sequence starting at 1;
- four-digit directory names;
- no overwrite/reuse/renumber;
- gaps make Study inconsistent and block freeze;
- `max_experiments` fixed in `[1, 9999]`;
- attempt `max_experiments + 1` raises `ExperimentBudgetExceededError`.

This bounds adaptive reuse of one development window; individual control does not eliminate Study-level overfitting.

## 22. Concurrency model

v1 is **single-writer, multi-reader** at the Study mutation boundary.

Every mutating public operation (`create/register/define/bind/verify/compare/decide/freeze`) is serialized by a StudyStore exclusive filesystem mutation lock. The exact cross-platform mechanism is an implementation-plan concern, but it must be process-safe for supported platforms and release automatically on process termination rather than relying on a permanently stale sentinel.

Readers inspect immutable published artifacts without taking the mutation lock when safe.

Concurrent mutation must never allow:

- two Experiments to claim the same sequence;
- a candidate binding after freeze;
- freeze to publish while another mutation is partially staged;
- overwrite of an already published artifact.

Race tests are required.

## 23. Study freeze

Terminal outcomes:

```text
WINNER
NO_WINNER
```

### WINNER

Requires:

```text
selected_run_fingerprint
selected_strategy
rationale
```

The Run must be ACCEPT-lineage reachable. `selected_strategy` must belong to `candidate_strategy_names`, **never** `control_strategy_names`.

If controls dominate and no research candidate is supportable, the correct outcome is `NO_WINNER`, not a control labeled as winner.

### NO_WINNER

Forbids selected Run/strategy fields.

### Freeze reconstruction

Before publication verify from disk:

- StudyPlan/provenance integrity;
- baseline/evidence store integrity;
- contiguous sequence and budget;
- every Experiment terminal DECIDED or INVALID;
- all digest/fingerprint references resolve;
- selected winner Run is ACCEPT-lineage eligible;
- selected strategy is an existing candidate, not a control;
- no rejected/invalid candidate resurrection;
- no existing freeze.

After freeze every mutating operation raises `StudyFrozenError`.

## 24. StudyStore

The only stateful helper class in v1. Filesystem mechanics only:

- path validation;
- exclusive mutation lock;
- regular-file verification;
- verified snapshot copy;
- content-addressed placement;
- staging/atomic publication;
- immutable reads;
- existence/integrity checks;
- failed-staging cleanup.

It does not own research decisions, factor semantics, statistics, or mutable in-memory Study state.

## 25. Public API

High-level surface:

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

Immutable public contracts:

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

Private codecs/provenance builders/field maps/store internals are not package-root exports by default.

## 26. Error model

```text
ControlledExperimentError
├── ContractViolationError
├── ArtifactIntegrityError
├── InvalidExperimentStateError
├── UncontrolledDeltaError
├── ExperimentBudgetExceededError
└── StudyFrozenError
```

- `ContractViolationError`: malformed/unsupported contract input.
- `ArtifactIntegrityError`: digest/size/symlink/malformed/provenance/evidence-store conflict.
- `InvalidExperimentStateError`: wrong workflow state or concurrent/state conflict.
- `UncontrolledDeltaError`: domain-level undeclared/no-op delta; workflow normally materializes trustworthy violations as INVALID.
- `ExperimentBudgetExceededError`: Study attempt budget exhausted.
- `StudyFrozenError`: mutation after freeze.

Trustworthy INVALID evidence is distinct from an operation that failed before verification could be trusted.

## 27. Quality contract

### Objective

Create a development-only Controlled Experiment Loop that proves what changed between Runs, proves Study-wide code/runtime/data/evaluation controls, retains self-contained immutable evidence, compares complete symbol-level outcomes, records decisions, enforces bounded accepted lineage, and freezes winner/no-winner without opening final unused data.

### Invariants

1. Existing fitting/replay/accounting semantics remain unchanged.
2. `MarketExecutor + BookState` remains economic authority.
3. Study data/evaluation/provenance controls cannot drift.
4. Initial baseline Run matches pre-resolved Study baseline config.
5. One Experiment changes one declared semantic factor.
6. Declared factor produces non-empty actual resolved delta.
7. Study-owned evidence is byte-exact, content-addressed, immutable.
8. INVALID remains visible and budgeted.
9. Definitions/terminal artifacts are immutable.
10. Rejected/inconclusive/invalid candidates are not winner-lineage eligible.
11. Control strategies cannot be Study winner.
12. Study freeze is terminal.
13. Final unused data is unreachable from this package.

### Failure modes

At minimum:

- malformed plan/baseline config;
- baseline Run does not match planned resolved config;
- unsupported factor;
- wrong/concurrent sequence;
- budget exhaustion;
- broken lineage;
- source or Study evidence tampering;
- symlink evidence;
- code/runtime provenance drift;
- dataset/window/fit-cutoff drift;
- symbol or strategy roster drift;
- undeclared/no-op delta;
- unaffected-strategy return drift;
- incomplete/duplicate/extra comparison cells;
- compare-before-verify;
- decide-before-compare;
- second decision;
- race between mutation/freeze;
- open Experiment at freeze;
- sequence gap;
- rejected candidate or control selected as winner;
- invalid WINNER/NO_WINNER fields;
- mutation after freeze;
- partial publication/cleanup failure.

### Risk

High. False-positive control can produce invalid research conclusions. Integrity, provenance, lineage, concurrency, and state violations fail closed.

### Test Oracle

- canonical digest equality;
- provenance digest equality;
- exact file digest/size and snapshot byte equality;
- filesystem publication state;
- resolved delta classification;
- unaffected-strategy raw-return equality;
- complete symbol×strategy comparison coverage;
- deterministic paired evidence under fixed seed;
- sequence/budget accounting;
- lineage reachability;
- freeze reconstruction;
- static absence of final-test dependency;
- concurrent mutation exclusion.

### Required Test Layers

- architecture/static dependency tests;
- contract unit tests;
- provenance unit/property tests;
- property tests for identity and delta partitioning;
- real temporary-filesystem artifact integration tests;
- race/failure/cleanup tests;
- candidate-run compatibility/regression tests;
- paired comparison tests;
- workflow/state tests;
- lineage/freeze tests;
- tamper/symlink/security regression tests;
- Ruff / format / Mypy;
- full pytest suite;
- package identity;
- exact-head GitHub CI;
- requirements-first falsification review.

## 28. TDD strategy

RED before production changes for at least:

1. `evaluation/experiments` boundary;
2. forbidden sealed-final-test dependency;
3. StudyPlan/baseline/factor/budget invariants;
4. candidate-run provenance evidence;
5. content-addressed evidence registration;
6. controlled-delta classification;
7. state ordering;
8. lineage;
9. concurrency/freeze rules.

No assertion weakening to reach Green.

## 29. Acceptance criteria

v1 is complete only when all are true:

1. `trade_rl.evaluation.experiments` is a distinct boundary.
2. It cannot depend on sealed final-test authorization.
3. StudyPlan identity/publication is deterministic and immutable.
4. Study creation pre-resolves and freezes the initial baseline config via shared run-config resolution.
5. Study freezes implementation/runtime digests before baseline registration.
6. Candidate runner emits evidence-only provenance without changing computation semantics.
7. Run registration verifies and snapshots summary/returns/provenance into Study storage.
8. Run fingerprint is independent of external path.
9. Symlink/tampered/missing/malformed evidence fails closed.
10. Existing evidence fingerprint reuse requires exact verified identity.
11. Initial baseline evidence matches planned resolved config and fixed controls.
12. ExperimentDefinition is immutable and binds Study/sequence/hypothesis/factor/baseline fingerprint.
13. Exactly one semantic factor governs allowed delta.
14. Verification uses resolved registered evidence, not requested config alone.
15. Study-fixed controls/provenance cannot drift in CONTROLLED Experiment.
16. Undeclared or no-op delta cannot become CONTROLLED.
17. Unaffected-strategy drift is detected where contractually valid.
18. Trustworthy uncontrolled attempts become terminal INVALID.
19. INVALID attempts consume budget.
20. Sequence is contiguous/four-digit/non-overwritable.
21. `max_experiments` is immutable in `[1,9999]`.
22. Only CONTROLLED Experiment can compare.
23. Comparison has exact complete symbol×strategy matrix and reuses existing paired primitives.
24. Decision is immutable, evidence-bound, and one of ACCEPT_CANDIDATE / KEEP_BASELINE / INCONCLUSIVE.
25. Only initial baseline or ACCEPT lineage can serve as later baseline.
26. Rejected/inconclusive/invalid candidates cannot enter winner lineage.
27. Mutations are process-serialized so sequence/freeze races cannot violate append-only state.
28. Freeze reconstructs and verifies all terminal attempts/evidence/provenance/lineage.
29. WINNER requires accepted Run + existing non-control candidate strategy.
30. A control cannot be labeled Study winner; control dominance yields NO_WINNER unless another candidate is independently justified.
31. NO_WINNER forbids selected Run/strategy.
32. Freeze prevents subsequent mutation.
33. Existing candidate computation semantics/evaluation public APIs remain compatible.
34. No final-unused-data or final stress execution is introduced.
35. Current architecture/research docs are updated for implemented evidence/public-boundary changes.
36. Targeted/full tests, Ruff, format, Mypy, package identity, exact-head CI all pass.
37. Final diff contains no temporary/debug/generated debris.
38. Independent/falsification review finds no unresolved Critical/High violation.

## 30. Follow-up sub-projects

### Frozen Strategy Artifact

A later design must persist the exact fitted Ridge/LightGBM/PPO winner identity before claiming the same trained policy is carried unchanged into unused-future evaluation.

### Sealed final evaluation

After Study freeze + frozen-strategy artifact, a separate cycle may bind them to existing one-shot sealed-test authorization and pre-registered stress. This is outside v1.

## 31. Resolved design decisions

Normative choices for v1:

- development-only through Study freeze;
- Study / Experiment / Run separated;
- one semantic factor per Experiment;
- typed factor registry;
- Study creation pre-resolves initial baseline config;
- append-only immutable artifacts;
- self-contained content-addressed Run snapshots;
- candidate-run `provenance.json`;
- conservative full production source-tree implementation digest;
- fixed runtime-environment digest;
- code/runtime changes require new Study;
- actual resolved delta authority;
- non-empty actual factor delta;
- unaffected-strategy adversarial oracle where valid;
- INVALID retained/budgeted;
- `[1,9999]` bounded attempts;
- contiguous four-digit sequence;
- single-writer mutation serialization;
- accepted-baseline lineage;
- exactly three Experiment decisions;
- controls are benchmarks, never Study winner;
- WINNER / NO_WINNER outcomes;
- no automatic winner selection;
- no final-test access;
- frozen trained-strategy serialization deferred.
