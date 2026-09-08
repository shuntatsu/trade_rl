Status: Active

# Controlled Experiment Loop v1 — Review Amendment

This amendment is normative together with `2026-09-09-controlled-experiment-loop-v1-design.md` until implementation completes and the durable contract is folded into current architecture/research documentation.

## Review conclusion

The Study / Experiment / Run split, append-only evidence model, semantic factor registry, bounded lineage, and development/final-data boundary are retained.

The implementation MUST additionally apply the corrections below. These close review findings that would otherwise permit dependency inversion, post-definition substitution, retrospective Run attachment, or an unfinishable Study after an operationally failed attempt.

## A1. Provenance generation belongs to the Run layer

The candidate runner is a lower-level evidence producer. It MUST NOT import `trade_rl.evaluation.experiments` in order to create `provenance.json`.

Generation-time provenance therefore lives at:

```text
trade_rl/evaluation/runs/provenance.py
```

The experiments package may parse, validate, fingerprint, and snapshot that provenance, but it does not own the generator used by `runs/candidate.py`.

The effective package shape is therefore:

```text
trade_rl/evaluation/runs/
  candidate.py
  candidate_suite.py
  config.py
  provenance.py

trade_rl/evaluation/experiments/
  ...
  artifacts/
    evidence.py
    store.py
```

There is no required `experiments/artifacts/provenance.py` in v1.

### Required architecture invariant

No module under `trade_rl.evaluation.runs` may import `trade_rl.evaluation.experiments`.

## A2. Candidate-run provenance must be stable across the computation

`run_candidate_artifact()` MUST capture implementation/runtime provenance before the candidate computation and capture it again before publication.

Publication is allowed only when the pre-run and post-run implementation/runtime digests are identical.

If source/runtime provenance changes while a Run is executing, the Run is not published.

This protects against a long-running fit/replay being attributed to source bytes that were only observed after computation finished.

## A3. Trustworthy Run registration is separate from Study compatibility

`register_run_evidence()` is an integrity operation, not a controlled-experiment decision.

It MUST:

- verify regular-file/symlink rules;
- verify exact digest/size while taking the Study-owned snapshot;
- parse supported summary/provenance schemas;
- derive semantic identities and Run fingerprint;
- atomically publish self-contained Run evidence.

It MUST NOT reject an otherwise trustworthy candidate Run merely because its dataset, evaluation window, implementation digest, runtime digest, or other controlled field differs from the Study.

Those differences are domain evidence for `verify_experiment()` and must be materialized as terminal `INVALID` when the evidence is trustworthy.

### Initial baseline exception

The initial baseline is stricter. A separate `bind_baseline_evidence()` operation validates that the registered Run exactly matches the StudyPlan baseline config, fixed controls, roster, dataset identity, and provenance before publishing `baseline-evidence.json`.

### Candidate binding

`bind_candidate_evidence()` only binds a registered trustworthy Run fingerprint to the already-defined Experiment. It does not prematurely classify controlledness.

## A4. ExperimentDefinition freezes the resolved candidate configuration

Pre-registration is ineffective if the Definition stores one requested config but verification accepts a different Run merely because both differences belong to the same semantic factor.

Therefore `define_experiment()` MUST use the same shared run-config resolution authority as Study creation and the candidate runner.

`ExperimentDefinition` MUST bind both:

```text
candidate_requested_config_digest
candidate_resolved_config
```

The raw requested config may be retained as canonical JSON text for audit, but `candidate_resolved_config` is normative.

`verify_experiment()` MUST require the registered candidate Run's resolved config to equal `ExperimentDefinition.candidate_resolved_config` exactly before classifying the baseline/candidate delta.

A candidate Run produced from a different threshold/value/feature set than the frozen Definition is terminal `INVALID`, even when the difference would otherwise be legal for the declared factor.

This prevents post-definition tuning within the allowed factor after observing results.

## A5. Shared run-config resolution is a lower authority

The current private config parsing/resolution in `evaluation/runs/candidate.py` is extracted into:

```text
trade_rl/evaluation/runs/config.py
```

The candidate runner, `create_study()`, and `define_experiment()` MUST all use this one authority.

Do not maintain parallel name/index/timestamp validation logic in `experiments`.

The resolved contract covers all current input degrees of freedom, including the evaluation fields that are not nested under `summary.json["candidate_config"]`:

```text
signal name/index
feature names/indices
fit symbol names/indices
fit cutoff
rule thresholds
forecast thresholds
PPO total timesteps/seed
evaluation start/stop
gross budget
initial capital
execution overlay identifier
```

## A6. Factor effects derived from the current candidate-suite dataflow

For the current suite, the implementation plan uses these affected sets unless tests/source inspection prove the production dataflow changed before implementation:

| Factor | Strategies allowed to change | Strategies required raw-return invariant |
|---|---|---|
| `FEATURE_SET` | `ridge24`, `lightgbm24`, `ppo` | `cash`, `constant_long`, `constant_short`, `trend`, `mean_reversion` |
| `RULE_SIGNAL` | `trend`, `mean_reversion` | controls, `ridge24`, `lightgbm24`, `ppo` |
| `RULE_THRESHOLDS` | `trend`, `mean_reversion` | controls, `ridge24`, `lightgbm24`, `ppo` |
| `FORECAST_THRESHOLDS` | `ridge24`, `lightgbm24` | controls, rules, `ppo` |
| `FIT_SYMBOL_SCOPE` | `ridge24`, `lightgbm24`, `ppo` | controls, rules |
| `PPO_TRAINING_BUDGET` | `ppo` | controls, rules, `ridge24`, `lightgbm24` |
| `GROSS_BUDGET` | all directional candidates and `constant_long`/`constant_short`; PPO training also changes | `cash` only |

`GROSS_BUDGET` is intentionally broad because the current candidate suite passes `gross_budget` both to PPO fitting and to common replay.

## A7. Study-owned evidence scope

"Self-contained Study evidence" means self-contained evidence needed to inspect, verify, compare, decide, and freeze the development Study.

v1 does not copy the full canonical dataset artifact into the Study and therefore does not claim that the Study directory alone can re-run fitting from scratch.

Exact rerun inputs remain identified by dataset identity/artifact digest plus external retained dataset artifact.

## A8. Public API correction

The v1 high-level surface is:

```text
create_study
register_run_evidence
bind_baseline_evidence
define_experiment
bind_candidate_evidence
record_experiment_failure
verify_experiment
compare_experiment
decide_experiment
freeze_study
inspect_study
```

`bind_baseline_evidence` is explicit rather than being hidden inside generic Run registration.

## A9. Run provenance binds the research context

A Study must not accept a favorable Run that was generated independently and only attached after its results were known.

Candidate-run provenance therefore contains an optional evidence-only field:

```text
research_context_digest
```

`run_candidate_artifact()` accepts an optional `research_context_digest` argument and CLI option. When provided it must be a lowercase SHA-256 digest and is included in `provenance.json`; it does not affect fitting/replay computation.

Controlled Study usage requires:

- initial baseline Run provenance `research_context_digest == StudyPlan digest`;
- candidate Run provenance `research_context_digest == ExperimentDefinition digest`.

`bind_baseline_evidence()` rejects a baseline without the exact Study digest context.

`verify_experiment()` marks a bound candidate terminal `INVALID` when its context digest does not equal the ExperimentDefinition digest.

Standalone candidate Runs outside Controlled Experiment Loop may omit the context digest for backward-compatible research use, but such Runs cannot be bound as a Controlled Study baseline/candidate.

This is an audit/preregistration boundary, not a claim that software can prevent a researcher from performing unrelated external exploratory Runs before the Study begins.

## A10. Operationally failed attempts are terminal and budgeted

A defined Experiment can fail before trustworthy candidate evidence exists (for example OOM, missing optional runtime dependency, or an execution failure). Leaving such an Experiment permanently open makes the Study impossible to freeze and encourages deleting/restarting the Study.

v1 therefore adds a terminal non-research state:

```text
FAILED
```

A `failure.json` artifact contains at least:

```text
schema_version
study_digest
experiment_digest
reason
recorded_by
recorded_at
```

Rules:

- `reason` and `recorded_by` are non-empty;
- `recorded_at` is timezone-aware;
- FAILED consumes the already-defined Experiment attempt;
- FAILED has no comparison/decision and is never lineage eligible;
- `record_experiment_failure()` is allowed only before `candidate-evidence.json` exists;
- once candidate evidence is bound, the Experiment must proceed through verification rather than being converted to FAILED;
- FAILED is immutable and terminal;
- Study freeze accepts terminal `DECIDED`, `INVALID`, or `FAILED` attempts.

This records operational failures without misclassifying them as controlled-domain INVALID evidence and without silently erasing the attempt from the Study budget.

## Additional acceptance criteria

Implementation is not complete unless all are true:

1. `evaluation.runs` has no dependency on `evaluation.experiments`.
2. pre/post provenance drift prevents candidate Run publication.
3. trustworthy but Study-incompatible candidate evidence can be registered and later becomes `INVALID` through verification.
4. baseline binding rejects a registered Run that does not exactly match the pre-resolved Study baseline contract.
5. Definition freezes a resolved candidate config and candidate evidence must match it exactly.
6. same-factor-but-different-value post-definition substitution becomes `INVALID`.
7. the affected/unaffected factor table above is covered by contract tests against current suite semantics.
8. controlled baseline/candidate bindings require exact Study/Experiment research context digests.
9. an Experiment can be terminally FAILED only before candidate evidence binding, remains budgeted, and cannot enter lineage.
10. freeze reconstructs FAILED attempts as terminal history rather than requiring Study deletion/restart.
