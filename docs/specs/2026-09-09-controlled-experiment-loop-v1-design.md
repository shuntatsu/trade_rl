Status: Active

# Controlled Experiment Loop v1 Design

## 1. Purpose

Trade RL has a lean candidate runner that can fit and independently replay the fixed candidate suite on one canonical dataset artifact and publish immutable `summary.json` / `returns.npz` evidence. It also has comparison, robustness, gate, and sealed outer-test primitives.

What is missing is the research unit that explains **why one run differs from another, whether the difference was actually controlled, what evidence was compared, what decision was made, and how a sequence of development experiments reaches a frozen winner or an explicit no-winner outcome**.

Controlled Experiment Loop v1 adds that layer.

The system exists to make development iteration auditable and fail closed. Its primary correctness property is not convenience; it is that an uncontrolled or tampered experiment cannot be recorded as a valid controlled experiment.

## 2. Scope decision

v1 ends at the development-study freeze boundary.

It manages:

1. a frozen development Study plan;
2. immutable baseline run evidence;
3. sequential controlled Experiments;
4. candidate run evidence binding;
5. actual resolved baseline/candidate delta verification;
6. paired comparison evidence;
7. immutable Experiment decisions;
8. accepted-baseline lineage;
9. Study-level experiment budget;
10. final `WINNER` or `NO_WINNER` freeze.

It does **not** open unused future data or execute the final sealed test.

The intended boundary is:

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
- LLM-authored autonomous research decisions.

The existing candidate runner remains the authority for fitting and replaying one candidate suite run.

## 4. Core terminology

### 4.1 Run

A **Run** is one computation performed by the existing candidate runner. It produces immutable result evidence, currently:

```text
summary.json
returns.npz
```

A Run does not know why it was executed.

### 4.2 Experiment

An **Experiment** binds:

```text
one explicit hypothesis
+ one baseline Run evidence identity
+ one candidate Run evidence identity
+ one semantic controlled factor
+ verification of the actual resolved delta
+ comparison evidence
+ one immutable decision
```

An Experiment may be valid (`CONTROLLED`) or terminally invalid (`INVALID`). Invalid attempts remain part of the Study history and consume the Study experiment budget.

### 4.3 Study

A **Study** is a bounded sequence of Experiments answering one research question under Study-level fixed controls.

A Study owns:

- the fixed development data/evaluation contract;
- initial baseline;
- allowed factor types;
- maximum number of Experiment attempts;
- Experiment sequence;
- accepted-baseline lineage;
- final `WINNER` or `NO_WINNER` freeze.

## 5. Design principles

### 5.1 One Experiment changes one semantic research factor

The controlled unit is not "one JSON field". One semantic factor may legitimately map to multiple resolved fields.

For example, changing a feature set can change both `feature_names` and `feature_indices`, but it may not also change `fit_cutoff` or `gross_budget`.

### 5.2 Actual resolved evidence is authoritative

Requested config files are insufficient evidence. Controlled-delta verification compares the **resolved immutable run summaries**, not only user-requested JSON.

### 5.3 Append-only artifacts, no mutable state file

There is no mutable `experiment.json` or `state.json` that is repeatedly rewritten.

State is derived from which immutable artifacts exist and which digests they bind.

### 5.4 Fail closed

Unknown factors, undeclared deltas, mismatched dataset identity, tampered evidence, unsupported state transitions, broken lineage, sequence gaps, or post-freeze writes fail rather than being silently tolerated.

### 5.5 Existing research primitives remain authoritative

The subsystem reuses:

- canonical JSON and content digests from `trade_rl.artifacts`;
- verified regular-file reads and SHA-256 checks from `trade_rl.artifacts`;
- current candidate-run artifacts from `trade_rl.evaluation.runs`;
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

A Study is represented as an append-only directory:

```text
<study-root>/
├── plan.json
├── baseline-evidence.json
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

No file in this structure is overwritten after publication.

The experiment directories are zero-padded positive sequence numbers. The sequence must be contiguous from `0001` through the highest attempt.

## 8. Canonical identity

All contract identities use the existing canonical JSON conversion and `content_digest()` SHA-256 authority.

### 8.1 Identity rules

- Study plan identity is the digest of the canonical `StudyPlan` payload.
- Experiment identity is the digest of the canonical immutable `ExperimentDefinition` payload.
- Verification, comparison, decision, and freeze identities are the canonical content digests of their payloads.
- Timestamp metadata must not be inserted into Study-plan or Experiment-definition identity merely to make an artifact unique.
- Human-readable filesystem paths are locators, not identity.
- File evidence identity is SHA-256 + byte size of the actual file snapshot.

### 8.2 Run evidence

A Run is bound by a `RunEvidence` record containing at least:

```text
summary SHA-256
summary size bytes
returns SHA-256
returns size bytes
run fingerprint / evidence digest
dataset identity from resolved summary
resolved candidate configuration identity
resolved evaluation scope identity
optional source locator
```

The optional source path is non-authoritative and excluded from evidence identity. Moving an identical Run artifact to another filesystem location does not change its research identity.

Before using a referenced result file, the implementation must use the existing verified regular-file boundary. Symlinks, size mismatch, digest mismatch, missing files, or malformed evidence fail closed.

## 9. StudyPlan contract

`StudyPlan` is immutable and created before development iteration begins.

It contains at least:

```text
schema_version
research_question
dataset_id
dataset_artifact_digest
symbol set / evaluation symbol contract
fit_cutoff
evaluation_start
evaluation_stop_exclusive
initial_capital
execution contract identifier
PPO seed policy
max_experiments
allowed_factor_types
initial baseline configuration identity
```

`max_experiments` must be a positive bounded integer and cannot be increased after Study creation.

The following are Study-level hard-fixed fields in v1:

- dataset identity;
- dataset artifact identity;
- evaluation symbol set;
- development evaluation window;
- execution contract;
- initial capital;
- PPO seed policy.

Changing any of these means starting a new Study, not another Experiment within the same Study.

`fit_cutoff` is also Study-fixed in v1. `fit_symbol_scope` may be an Experiment factor only if the Study explicitly permits that factor; the dataset and evaluation scope still remain fixed.

## 10. Controlled factor registry

v1 uses typed controlled factors rather than arbitrary free-form JSON paths.

Supported factor types:

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

Each factor type defines:

1. which resolved fields may differ;
2. which resolved fields must remain equal;
3. which strategies are expected to be affected;
4. which strategies must remain exactly unchanged at raw-return evidence level when applicable.

The registry is versioned as part of the implementation contract. Unknown factor types are rejected.

### 10.1 Initial allowed resolved deltas

The implementation plan must derive the exact field map from the current resolved candidate-run summary schema rather than inventing parallel field names.

Conceptually:

| Factor | Allowed semantic delta |
|---|---|
| `FEATURE_SET` | selected feature names/indices only |
| `RULE_SIGNAL` | rule signal name/index only |
| `RULE_THRESHOLDS` | rule entry/exit threshold pair only |
| `FORECAST_THRESHOLDS` | forecast entry/exit threshold pair only |
| `FIT_SYMBOL_SCOPE` | fit symbol names/indices only |
| `PPO_TRAINING_BUDGET` | PPO total timesteps only |
| `GROSS_BUDGET` | evaluation/training gross budget only where the current run schema binds it |

If the actual current summary schema cannot represent a proposed factor unambiguously, that factor must not be implemented until the schema contract is improved deliberately.

## 11. ExperimentDefinition contract

An Experiment is defined before candidate evidence is observed.

`ExperimentDefinition` contains at least:

```text
schema_version
study_digest
sequence
hypothesis
baseline_run_evidence_digest
factor type
candidate requested configuration
```

Requirements:

- sequence is positive and equals the next available Study attempt number;
- hypothesis is non-empty;
- factor type is permitted by the Study;
- baseline evidence exists and is lineage-eligible;
- candidate configuration is complete enough to run the existing candidate runner;
- the Study must not already be frozen;
- the budget must not already be exhausted.

The definition is immutable once published.

## 12. Derived Experiment state

There is no mutable status artifact.

State is derived as follows:

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

A verification artifact with `INVALID` is terminal:

```text
INVALID
```

An INVALID Experiment has no comparison or decision artifact.

No workflow operation may skip the required previous stage.

## 13. Baseline and candidate run evidence

The Study begins with one registered immutable baseline Run evidence record.

Each Experiment later registers exactly one candidate Run evidence record.

Registration verifies:

- referenced files are regular files, not symlinks;
- expected byte sizes match;
- SHA-256 digests match;
- `summary.json` is valid supported JSON;
- `returns.npz` is the exact digest-bound file later used for comparison;
- resolved summary dataset identity matches the Study;
- resolved evaluation scope is parseable and compatible with the Study contract.

Registration does not yet claim the candidate is controlled; it only binds immutable evidence.

## 14. Actual controlled-delta verification

`verify_experiment()` compares the **resolved baseline summary** and **resolved candidate summary**.

Verification checks include at least:

```text
run artifact integrity
study identity binding
dataset identity equality
dataset artifact identity equality
evaluation symbol roster equality
evaluation window equality
fixed Study controls equality
strategy roster equality
declared-factor actual delta only
unaffected-strategy raw return invariance where required
```

### 14.1 Controlled outcome

If every required check passes, publish `verification.json` with status `CONTROLLED`.

### 14.2 Invalid outcome

If the candidate evidence is readable and attributable to the Experiment but violates the controlled contract, publish terminal `verification.json` with status `INVALID` and explicit failed checks.

Examples:

- `FEATURE_SET` Experiment also changed `fit_cutoff`;
- dataset identity drifted;
- evaluation window drifted;
- strategy roster changed;
- an unaffected control strategy return series changed.

An INVALID attempt still consumes one Study Experiment number and budget unit.

### 14.3 Operation failure versus INVALID result

Not every error becomes an INVALID artifact.

Use an exception and publish nothing for cases where the verification operation cannot establish a trustworthy evidence snapshot, for example:

- evidence file disappeared;
- digest verification failed;
- unsupported/malformed artifact cannot be interpreted;
- filesystem publication itself failed.

Use terminal `INVALID` only when trustworthy evidence exists and proves the controlled-experiment contract was violated.

## 15. Unaffected-strategy falsification oracle

The factor registry identifies strategies that must not be affected by a factor.

For example, under `FEATURE_SET`, the forecast/RL candidates may change, but simple rule strategies and controls are expected to be independent of selected model features under the current architecture.

When a factor marks a strategy as unaffected, its raw return series and relevant immutable replay evidence must match exactly between baseline and candidate.

A mismatch is treated as evidence of hidden-condition drift or semantic regression and makes the Experiment INVALID.

The implementation plan must verify the current candidate-run architecture before assigning the exact affected/unaffected sets. No set may be guessed solely from strategy names.

## 16. Comparison contract

Only a `CONTROLLED` Experiment can be compared.

Comparison reuses the existing return-series and paired-comparison primitives.

Evidence is stored for every compatible:

```text
symbol × strategy
```

At minimum, `ExperimentComparison` records:

- baseline and candidate performance metrics;
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
- referenced baseline/candidate evidence digests;
- verification digest.

The comparison must cover the complete expected symbol × strategy Cartesian product. Missing cells fail closed.

A statistical threshold does not automatically determine the research decision.

## 17. Experiment decision contract

Experiment-level decisions are exactly:

```text
ACCEPT_CANDIDATE
KEEP_BASELINE
INCONCLUSIVE
```

There is no mutable `RETEST` decision. A retest is a new Experiment with a new sequence number and immutable definition.

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

`decided_at` must be timezone-aware.

The decision artifact is immutable and may be published only once.

The subsystem does not automatically decide from p-values or one aggregate metric.

## 18. Accepted-baseline lineage

An Experiment definition must bind to a baseline Run evidence digest.

A Run is eligible to serve as a later Experiment baseline only if it is:

1. the initial Study baseline; or
2. a candidate Run from a prior `ACCEPT_CANDIDATE` decision reachable from the initial baseline.

Candidates from `KEEP_BASELINE`, `INCONCLUSIVE`, or `INVALID` Experiments are not baseline-lineage eligible.

This prevents a rejected candidate from being silently resurrected later as the selected winner path.

## 19. Experiment budget and sequence integrity

Every Experiment definition consumes one Study attempt, including INVALID attempts.

Rules:

- experiment directories are contiguous from sequence 1;
- existing sequences cannot be overwritten;
- a deleted sequence produces an invalid Study state and prevents freeze;
- a sequence cannot be renumbered through the public API;
- `max_experiments` cannot be changed after Study creation;
- defining attempt `max_experiments + 1` raises `ExperimentBudgetExceededError`.

The budget exists to bound adaptive reuse of the same development window. Individually controlled Experiments do not eliminate Study-level development overfitting.

## 20. Study freeze

A Study freeze is terminal and immutable.

Allowed outcomes:

```text
WINNER
NO_WINNER
```

### 20.1 WINNER

Requires:

```text
selected_run_evidence_digest
selected_strategy
rationale
```

The selected Run must be reachable through valid ACCEPT lineage.

The selected strategy must exist in that Run evidence.

### 20.2 NO_WINNER

Must not contain a selected Run or selected strategy.

### 20.3 Freeze preconditions

Before publishing `freeze.json`, verify:

- Study plan integrity;
- baseline evidence integrity;
- contiguous Experiment sequence;
- total attempts <= budget;
- every Experiment is terminal (`DECIDED` or `INVALID`);
- all artifact digest references resolve consistently;
- selected winner Run, if any, is ACCEPT-lineage eligible;
- selected strategy exists;
- no invalid/rejected candidate was resurrected;
- Study has no existing freeze artifact.

After freeze, all mutating Study operations reject with `StudyFrozenError`.

## 21. Filesystem store responsibility

`StudyStore` is the only stateful helper class proposed for v1.

It owns filesystem mechanics only:

- validated path resolution;
- exclusive creation;
- staging;
- atomic publication;
- verified reads;
- artifact existence checks;
- cleanup of failed staging work.

It must not own research decisions, factor semantics, comparison policy, or mutable Study state.

Workflow functions remain explicit pure/domain operations around immutable contracts and the store.

## 22. Public API

The intended high-level public surface under `trade_rl.evaluation.experiments` is:

```text
create_study
register_run_evidence
define_experiment
verify_experiment
compare_experiment
decide_experiment
freeze_study
inspect_study
```

Public immutable contract types:

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

Private codecs, field maps, filesystem helpers, and comparison adapters are not package-root exports by default.

No public API mutates fields on an in-memory Study object.

## 23. Error model

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
- `ArtifactIntegrityError`: digest, size, symlink, malformed, or missing trustworthy evidence failure.
- `InvalidExperimentStateError`: operation attempted before required prior artifact or after terminal state.
- `UncontrolledDeltaError`: domain helper result for an undeclared resolved delta when used outside publication workflow; workflow-level verification normally materializes this as terminal INVALID evidence rather than discarding the attempt.
- `ExperimentBudgetExceededError`: attempt exceeds frozen Study budget.
- `StudyFrozenError`: mutation attempted after freeze.

The implementation must preserve the distinction between evidence proving an Experiment INVALID and an operation failing before trustworthy verification is possible.

## 24. Security and integrity boundary

Research evidence is untrusted until verified.

The implementation must account for:

- symlink substitution;
- file replacement after initial inspection;
- size/digest mismatch;
- malformed JSON;
- missing NPZ keys;
- duplicate/missing symbol-strategy cells;
- partial filesystem publication;
- overwrites;
- sequence directory collision;
- stale evidence locators;
- post-freeze mutation.

Unsafe deserialization must operate only on a verified private snapshot where required by the existing artifact-safety contract.

No final-test dataset accessor, sealed-test ledger call, or unused-future opening capability belongs in this package.

## 25. Quality contract

### Objective

Create a development-only Controlled Experiment Loop in which the repository can prove what changed between baseline and candidate Runs, bind immutable evidence, compare complete symbol-level results, record decisions, enforce bounded accepted lineage, and freeze a winner/no-winner outcome without opening final unused data.

### Non-goals

The Non-goals in section 3 are binding and must not be expanded silently during implementation.

### Invariants

1. Existing candidate-run fitting/replay semantics remain unchanged.
2. Existing `MarketExecutor + BookState` economic authority remains unchanged.
3. Study hard-fixed controls cannot drift between Experiments.
4. One Experiment changes one declared semantic factor only.
5. Actual resolved run evidence, not requested config alone, determines control validity.
6. INVALID attempts remain visible and consume budget.
7. Experiment definitions and all terminal artifacts are immutable.
8. A rejected/inconclusive/invalid candidate is not winner-lineage eligible.
9. Study freeze is terminal.
10. Experiment Loop cannot open final unused data.

### Failure modes

At minimum:

- malformed Study plan;
- duplicate/empty factor settings;
- unsupported factor;
- wrong next sequence;
- budget exhaustion;
- missing baseline;
- broken baseline lineage;
- tampered summary or returns file;
- symlink evidence;
- dataset drift;
- evaluation-window drift;
- symbol roster drift;
- strategy roster drift;
- undeclared resolved delta;
- unaffected-strategy return drift;
- incomplete symbol × strategy comparison;
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

High. A false-positive validity result can cause research conclusions to be based on an uncontrolled or adaptively cherry-picked development comparison. The subsystem therefore treats integrity, lineage, and state-transition violations as fail-closed conditions.

### Test Oracle

Correctness is observed through:

- canonical content digest equality;
- exact file SHA-256 and byte size;
- filesystem state transitions;
- immutable/exclusive publication;
- resolved baseline/candidate delta classification;
- unaffected-strategy raw-return equality;
- complete symbol × strategy comparison coverage;
- deterministic paired-comparison evidence under fixed seed;
- Experiment sequence and budget accounting;
- accepted-baseline graph reachability;
- freeze precondition reconstruction from disk;
- absence of final-test dependency.

### Required Test Layers

- architecture/static dependency tests;
- contract unit tests;
- property tests for canonical identity and delta partitioning;
- artifact integration tests with real temporary files;
- filesystem failure/cleanup tests;
- existing candidate-run compatibility/regression tests;
- paired comparison tests;
- workflow/state-transition tests;
- lineage and freeze tests;
- tamper/symlink/security regression tests;
- Ruff;
- format check;
- Mypy;
- full repository pytest suite;
- package identity check;
- exact-head GitHub CI;
- requirements-first falsification review.

## 26. TDD strategy

Implementation uses RED -> GREEN -> Refactor.

The first production change must be preceded by failing contract/architecture tests for:

1. required `evaluation/experiments` package boundary;
2. forbidden dependency on sealed final-test authorization;
3. StudyPlan/factor/budget invariants;
4. immutable evidence binding;
5. actual controlled-delta classification;
6. state-transition ordering;
7. accepted-baseline lineage;
8. Study freeze rules.

Tests are not weakened to accommodate an implementation shortcut.

## 27. Acceptance criteria

v1 is specification-complete only when all are true:

1. `trade_rl.evaluation.experiments` exists as a distinct responsibility boundary.
2. Experiment code has no dependency on sealed final-test authorization.
3. Study plan has deterministic canonical identity and immutable publication.
4. Run evidence binds actual summary/returns bytes by SHA-256 and size.
5. Symlink/tampered/missing evidence is rejected fail closed.
6. Experiment definition is immutable and bound to Study, sequence, hypothesis, factor, and baseline evidence.
7. Exactly one registered semantic factor governs allowed resolved delta.
8. Verification compares actual resolved run evidence rather than requested config only.
9. Study-fixed controls cannot drift inside a valid Experiment.
10. Undeclared delta cannot produce `CONTROLLED` verification.
11. Unaffected-strategy drift is detected wherever the factor contract declares invariance.
12. Trustworthy uncontrolled evidence produces terminal INVALID history rather than disappearing.
13. INVALID attempts consume Study budget.
14. Experiment sequence is contiguous and cannot be overwritten/reused.
15. Study budget cannot be increased after creation.
16. Only CONTROLLED Experiments can produce comparison evidence.
17. Comparison covers the complete expected symbol × strategy matrix.
18. Comparison reuses existing paired-return/statistical primitives rather than reimplementing them.
19. Experiment decision is immutable and bound to verification + comparison identities.
20. Experiment decisions are limited to ACCEPT_CANDIDATE / KEEP_BASELINE / INCONCLUSIVE.
21. Only the initial baseline or ACCEPT_CANDIDATE lineage can be used as future baseline.
22. Rejected/inconclusive/invalid candidates cannot be selected through winner lineage.
23. Study freeze validates all terminal attempts and references before publication.
24. WINNER requires a valid accepted Run + strategy; NO_WINNER forbids them.
25. Freeze makes the Study terminal against further mutation.
26. Existing candidate runner semantics and existing evaluation public APIs remain compatible.
27. No final-unused-data or final stress execution is introduced.
28. Targeted tests, full tests, Ruff, format, Mypy, package identity, and exact-head CI pass.
29. Final diff contains no temporary migration/debug workflow or generated output.
30. Independent/falsification review finds no unresolved Critical or High contract violation.

## 28. Follow-up sub-projects

### 28.1 Frozen Strategy Artifact

The current candidate runner records configuration and development evidence but does not persist the fitted Ridge/LightGBM/PPO strategy object as the final reusable trained-policy identity.

A later sub-project must define a safe, versioned, immutable frozen-strategy artifact before claiming that the exact trained winner can be carried unchanged into the unused-future evaluation.

### 28.2 Sealed final evaluation

After a Study is frozen and a frozen-strategy artifact exists, a separate design cycle may bind the Study/frozen strategy identity to the existing one-shot sealed-test authorization and pre-registered stress protocol.

That later subsystem is intentionally outside Controlled Experiment Loop v1.

## 29. Resolved design decisions

The following choices are final for v1 unless the spec is explicitly revised before implementation:

- development-only scope through Study freeze;
- Study / Experiment / Run as separate concepts;
- one semantic controlled factor per Experiment;
- typed factor registry, not arbitrary path whitelist as the primary contract;
- append-only immutable artifacts, no mutable state file;
- actual resolved summary delta as authority;
- raw unaffected-strategy evidence as an adversarial oracle where valid;
- INVALID attempts retained and budgeted;
- bounded Study experiment count;
- contiguous sequence numbers;
- accepted-baseline lineage;
- no automatic winner selection;
- exactly three Experiment decisions;
- WINNER / NO_WINNER Study outcomes;
- no final-test access from the package;
- frozen trained-strategy serialization deferred to a separate sub-project.
