# Default Target-Weight Full-Research Workflow Design

## Decision

The maintained full-research `develop` phase will use
`examples/binance-multitimeframe/walk-forward-target-weight-constrained-growth.json`
when no explicit training template is supplied.

That workflow compares exactly three single-symbol target-weight profiles:

1. `training-target-weight-growth-ppo.json` — gamma-one net-growth PPO control;
2. `training-target-weight-constrained-growth.json` — gamma-one constrained-growth Lagrangian PPO candidate;
3. `training-target-weight-constrained-growth-discounted.json` — 168-hour discounted constrained-growth ablation.

`training-full.json` remains readable and runnable only through an explicit
`--training-template training-full.json` selection. It is not part of the
implicit default candidate catalog.

## Context

The maintained Binance documentation already defines the three target-weight
growth profiles as the comparable objective family and identifies
`training-full.json` as a legacy mixed-shaping comparison. The phase runner,
however, currently initializes its default workflow from `walk-forward-full.json`,
which contains only the legacy `training-full.json` candidate. This makes the
runtime default disagree with the documented research contract.

## Architecture

The runner will expose one private module-level `Path` constant for the default
workflow template and use that constant at the beginning of the workflow
selection branch. The existing explicit-template branch remains unchanged: it
loads the maintained target-weight workflow as the evaluation envelope, replaces
the candidate list with the explicitly requested training profile, and writes a
new generation-scoped materialized configuration.

No configuration file is renamed or rewritten. Existing references and historical
artifacts therefore remain resolvable, while new default generations use the
three-profile catalog.

## Data Flow

### Default invocation

```text
run_full_research_state.py --phase develop
  -> default workflow Path
  -> walk-forward-target-weight-constrained-growth.json
  -> materialize three run_file profiles with packaged Git provenance
  -> six-fold walk-forward comparison
  -> selection proposal
```

### Explicit legacy invocation

```text
run_full_research_state.py --phase develop \
  --training-template training-full.json
  -> validate explicit template
  -> load target-weight workflow envelope
  -> replace candidates with the explicit training-full payload
  -> materialize one generation-scoped candidate
```

## Compatibility and Isolation

- Existing generations, Docker images, checkpoints, manifests, and telemetry are immutable.
- The currently running generation is not stopped, rewritten, resumed, or migrated.
- The change affects only generations built from a new source commit and image.
- `training_run_config_v4`, market dataset identity, checkpoint identity, and artifact schemas do not change.
- Resume compatibility remains governed by the existing configuration and architecture identity checks.
- Production status remains `NO-GO`.

## Failure Behavior

- A missing or malformed default workflow continues to fail closed during configuration materialization.
- An explicit template outside the maintained example directory remains rejected.
- An explicit template containing resume checkpoints remains rejected for selected-final training.
- No fallback to `training-full.json` is introduced.

## Testing Strategy

A focused regression test will load the runner without executing `main()` and
assert that:

- the default workflow constant resolves to
  `walk-forward-target-weight-constrained-growth.json`;
- the default catalog contains the three expected `run_file` profiles in order;
- `training-full.json` is absent from the default catalog;
- `training-full.json` remains accepted by the existing explicit-template path.

The test is committed first and must fail against the current runner. The minimal
constant and wiring change is then added, followed by focused tests, related
example/profile tests, static checks, and the repository's exact-head CI.

## TDD Evidence

RED head `6dafdbfe87a2bc7ea8e21cf108d65e9af02eae69` completed all compatibility,
training-image, frontend, formatting, typing, architecture, dead-code, and smoke
checks before full pytest reported exactly the two new default-workflow contract
failures. The remaining repository result was 3,177 passed and 26 skipped with
80.65% total branch-aware coverage.

## Documentation

The Docker full-training runbook and maintained single-symbol documentation will
identify the three-profile walk-forward file as the default and classify
`training-full.json` as an explicit legacy mixed-shaping comparison.

## Non-Goals

- Changing any reward coefficient or target-weight profile payload;
- changing PPO, Lagrangian PPO, BC, risk, execution, or selection-gate logic;
- reducing folds, seeds, timesteps, or model capacity;
- starting or stopping a training generation;
- merging the independent telemetry-semantics pull request;
- authorizing live order submission or production deployment.
