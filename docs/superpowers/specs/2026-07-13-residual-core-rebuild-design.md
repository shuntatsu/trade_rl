# Residual Core Rebuild Design

## Status

Approved for implementation on 2026-07-13.

Production remains **NO-GO**. This redesign changes software structure and evidence handling; it does not upgrade any research result into production authorization.

## Decision

Replace the mixed-generation `mars_lite` codebase with one authoritative `trade_rl` package focused on baseline-anchored residual reinforcement learning.

Compatibility with the legacy Mars environment, direct-action PPO path, legacy CLI scripts, legacy artifact shapes, and legacy tests is intentionally not preserved. Git history is the compatibility archive.

## Goals

1. Make responsibility boundaries visible from the directory tree.
2. Keep domain and evaluation logic independent from CLI, filesystem, and serving frameworks.
3. Represent datasets, signals, policies, evaluations, selections, and releases as separate typed artifacts.
4. Make invalid states unrepresentable or fail closed at validation boundaries.
5. Keep baseline-only fallback explicit and distinguish it from a selected PPO policy.
6. Keep research completion, candidate selection, and production eligibility as separate states.
7. Use one implementation for metrics, dataset identity, execution accounting, artifact hashing, and gate decisions.
8. Enforce dependency direction in CI.

## Non-goals

- Changing PPO libraries solely for architectural novelty.
- Claiming profitability or production readiness.
- Preserving old command names or import paths.
- Supporting the rejected direct-action policy as a maintained comparison mode.
- Building a distributed registry in this change.

## Package structure

```text
trade_rl/
  domain/
    datasets.py
    signals.py
    policies.py
    evaluation.py
    selection.py
    releases.py
  data/
    ingestion/
    storage/
    features/
    alignment/
    validation/
  strategies/
    trend/
    baselines/
  signals/
    training/
    inference/
    validation/
    gates/
  rl/
    environment/
    observations/
    actions/
    rewards/
    training/
    checkpoints/
    ensemble/
  simulation/
    execution/
    costs/
    funding/
    accounting/
  evaluation/
    metrics/
    comparisons/
    bootstrap/
    walk_forward/
    gates/
  artifacts/
    codec.py
    hashing.py
    store.py
    validators.py
  workflows/
    train_signal.py
    train_residual.py
    walk_forward.py
    evaluate.py
    publish.py
  serving/
    bundle/
    loader/
    runtime/
    registry/
    audit/
  config/
    schemas.py
    loaders.py
    presets/
  cli/
    app.py
    commands/
```

Directories may be introduced only when they contain a real responsibility. Empty architecture theater is forbidden.

## Dependency direction

Allowed dependency direction:

```text
cli -> workflows -> application components -> domain
serving -> artifacts + domain + inference components
artifacts -> domain
rl/signals/strategies/simulation/evaluation/data -> domain

domain -> standard library only
```

Forbidden examples:

- `domain` importing pandas, Stable-Baselines3, FastAPI, filesystem code, or CLI code.
- `evaluation` importing workflows.
- training code importing serving runtime.
- artifact schemas importing orchestration functions.
- CLI namespaces being passed into domain or training code.

## Typed run model

A run contains separate immutable records:

- `DatasetManifest`
- `SignalArtifactManifest`
- `PolicyEnsembleManifest`
- `EvaluationReport`
- `SelectionDecision`
- `ReleaseManifest`

Each record has a schema version and canonical digest. Cross-artifact references use digests, not mutable paths alone.

### Required invariants

- A `baseline_only` selection has no active policy digest.
- A `residual_policy` selection has a non-empty ensemble with exactly the declared number of members.
- A disabled or rejected signal cannot be referenced as an enabled alpha source.
- A production release cannot be created from a failed mandatory gate.
- Dataset identities must agree across signal, policy, evaluation, and selection artifacts.
- Selection evidence must predate sealed holdout evaluation.
- A research run can complete even when release creation is blocked.

## Run directory

```text
run/
  dataset/manifest.json
  signal/manifest.json
  policy/ensemble.json
  policy/member-000/...
  evaluation/development.json
  evaluation/holdout.json
  evaluation/stress-cost2x.json
  evaluation/baselines.json
  selection/decision.json
  release/deployment-manifest.json
  report/summary.md
```

Files that do not apply are absent. They are not represented by misleading null-filled pseudo-artifacts.

## Current real-data result migration

The 2026-07-13 result is classified as:

```text
ResearchRun: COMPLETED
SignalArtifact: REJECTED
ResidualPolicyCandidate: NOT_SELECTED
BaselineFallback: SELECTED_FOR_ANALYSIS
ProductionRelease: BLOCKED
```

The supplied legacy report and manifests become migration fixtures. The migration test must prove that:

- configuration A is represented as a baseline identity candidate;
- no PPO policy is advertised as selected;
- the failed signal gate disables residual alpha;
- the failed final gate blocks release generation;
- holdout profit is retained as evidence but does not override failed gates;
- the legacy feature metadata is not mislabeled as a PPO ensemble artifact.

## CLI

One installed command is authoritative:

```text
trade-rl data ...
trade-rl signal ...
trade-rl train ...
trade-rl walk-forward ...
trade-rl evaluate ...
trade-rl registry ...
trade-rl serve ...
```

The CLI converts arguments into typed configuration objects and calls workflows. Workflows never receive `argparse.Namespace` objects.

## Evaluation

All return, Sharpe, Sortino, drawdown, turnover, cost, funding, and paired-excess calculations live under `trade_rl.evaluation`.

Annualization is explicit in the input series metadata. Base-bar and decision-step returns cannot be mixed silently.

Walk-forward orchestration is split into:

- pure fold construction and validation;
- fold-local training;
- checkpoint selection;
- configuration selection;
- outer-OOS evaluation;
- stitched chronological aggregation;
- artifact publication.

No single module owns all stages.

## Artifact and publication behavior

Artifact writes use staging directories, canonical JSON, SHA-256 content digests, validation before publication, and atomic pointer replacement.

Failed runs are isolated and cannot overwrite the last successful result.

Research artifacts cannot be registered as production releases. Baseline-only serving remains possible only as an explicitly typed research or safety fallback bundle; it does not imply production eligibility.

## Deletion policy

Delete rather than deprecate:

- legacy Mars environments and multi-timeframe wrappers;
- direct-action PPO environments and training paths;
- direct-only BC, PBT, walk-forward, and candidate selection;
- `action_mode="direct"` branches;
- old training and evaluation scripts;
- one-off diagnostic, sweep, and ablation scripts that are not supported product interfaces;
- `legacy_tests`;
- obsolete documentation and compatibility adapters.

Shared concepts are reimplemented behind the new interfaces. Old files are not retained merely because they contain reusable lines.

## Quality gates

CI must run:

- Ruff with unused imports and unused variables enabled;
- mypy on the new package without disabling core type errors;
- Import Linter contracts for dependency direction;
- pytest for unit, contract, migration, and integration tests;
- coverage over workflows, evaluation, artifacts, and core RL modules;
- dead-code reporting using Vulture as an advisory report initially;
- property-based tests using Hypothesis for fold boundaries, artifact invariants, identity parity, and stitched OOS ordering.

## Rollout

The rebuild is delivered on one feature branch, but commits remain reviewable:

1. freeze design and implementation plan;
2. add typed domain and migration tests;
3. add artifact and evaluation foundations;
4. add residual training and walk-forward components;
5. add workflow and CLI boundaries;
6. migrate serving and registry contracts;
7. remove legacy code and scripts;
8. harden CI and documentation;
9. verify on GitHub Actions and open a draft pull request.

The branch is not merged until all mandatory checks pass and the final diff contains no maintained legacy execution path.
