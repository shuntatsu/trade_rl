# Training Capability Audit Operations Boundary — Compatibility Amendment

## Status

This amendment updates the quality contract for the training-capability audit operations-boundary change after the real hierarchical-sequence behavior-cloning integration test exposed a pre-existing audit-fixture incompatibility.

It supersedes only the conflicting clauses in:

- `docs/implementation-plans/specs/2026-08-18-training-capability-audit-operations-design.md`; and
- `docs/implementation-plans/plans/2026-08-18-training-capability-audit-operations.md`

that require the hierarchical sequence audit probe to remain numerically/configurationally identical to the old script implementation.

All report, CLI, workflow-command, ownership, serialization, digest, algorithm-matrix, resume, replay, export, residual-control, production-training, serving, research, risk, reward, execution, and promotion contracts remain unchanged unless explicitly stated below.

## Why the contract changed

The original extraction plan assumed the existing sequence audit fixture was still a valid positive control and therefore treated exact sequence-probe configuration preservation as an invariant.

The strengthened integration test disproved that assumption.

At commit `ff5b0d994223177a928836c401fe3662924b08bd`, with the old sequence fixture using `episode_bars=8`, standard CI run `32229984475` failed only the real hierarchical behavior-cloning sequence integration test while static analysis, architecture checks, compatibility jobs, and the training image remained healthy.

Observed failure evidence included:

- teacher target approximately `+0.4`;
- learned policy target approximately `-0.113`;
- direction agreement `0.0`;
- active target RMSE approximately `0.513`;
- only `5` complete causal holdout episodes;
- after-cost cash-baseline regret failure; and
- causal Oracle-regret admission failure.

This demonstrated that preserving `episode_bars=8` would preserve an invalid audit positive control rather than preserve a meaningful capability contract.

After changing only the audit fixture to `episode_bars=2`, teacher-event support became dense enough for the same hierarchical head to reconstruct the intended direction and target closely. With the maintained loss weights and patience restored, the observed integration evidence included:

- teacher target approximately `+0.4`;
- learned policy target approximately `+0.38168`;
- direction agreement `1.0`;
- active target RMSE approximately `0.01837`;
- `23` complete causal holdout episodes;
- after-cost cash baseline PASS; and
- non-collapse evidence PASS.

The only remaining failure was a tiny positive hindsight-Oracle regret against the inherited legacy default threshold `0.0`.

Repository-maintained hierarchical behavior-cloning profiles and behavior-cloning gate tests explicitly use `behavior_cloning_max_causal_holdout_regret=0.2`. Requiring exactly `0.0` would make the audit demand effectively exact hindsight-Oracle replication from an approximate neural policy even when direction, target reconstruction, non-collapse, and after-cost cash-baseline checks pass.

The audit therefore now authors the maintained admission threshold explicitly rather than inheriting the legacy default.

## Revised objective

Move the full training-capability audit under `trade_rl.operations`, preserve its external report/CLI/workflow contract, and keep the real hierarchical sequence probe valid under the repository's maintained behavior-cloning admission contract.

## Revised non-goals

The following remain outside this PR:

- no change to PPO, SAC, TD3, or TQC audit hyperparameters;
- no change to production training profiles or defaults;
- no change to general behavior-cloning gate implementation;
- no change to `trade_rl.learning.rollout_evaluation` or its tolerance semantics;
- no change to production research, reward, risk, execution, selection, serving, or promotion behavior;
- no change to `full_training_capability_audit_v1` fields or digest construction;
- no change to audit output-root replacement, persisted JSON bytes, or script stdout contract;
- no change to `.github/workflows/full-training-capability-audit.yml` or its invocation command; and
- no weakening, skipping, deletion, or masking of the real sequence integration test.

## Authorized audit-only compatibility corrections

Exactly two sequence-audit configuration corrections are authorized by this amendment:

1. `ResidualMarketEnvConfig.episode_bars` changes from `8` to `2` for the synthetic structured-sequence audit fixture.
   - `episode_hours=2.0` remains unchanged.
   - `decision_hours=0.25` remains unchanged.
   - `decision_every=1` remains unchanged.
   - This is an audit-fixture support-density correction, not a production environment change.

2. `ResidualTrainingConfig.behavior_cloning_max_causal_holdout_regret` is explicitly set to `0.2` for the structured-sequence audit probe.
   - The generic configuration default is not changed.
   - Maintained production/example profiles are not changed.
   - Reconstruction, non-collapse, cash-baseline, and lower-confidence-bound gates remain required.

No other exploratory hyperparameter trial is part of the final contract. In particular, the attempted longer behavior-cloning patience and increased target-loss weight were rejected and reverted after producing worse behavior.

## Acceptance criteria added by this amendment

In addition to the original acceptance criteria:

1. A lightweight contract test must assert that the sequence audit uses `episode_bars=2` while retaining `episode_hours=2.0` and `decision_hours=0.25`.
2. A lightweight contract test must assert that the sequence audit explicitly authors `behavior_cloning_max_causal_holdout_regret=0.2`.
3. The real hierarchical sequence integration test must execute actual behavior cloning and require:
   - `active_target_rmse` gate PASS;
   - after-cost cash-baseline regret gate PASS;
   - causal regret upper-confidence-bound gate PASS;
   - regret threshold exactly `0.2`; and
   - non-empty behavior-cloning sample support.
4. General rollout-evaluation implementation and tests must match `main`; the audit correction must not leak into shared evaluation semantics.
5. All original static, architecture, compatibility, coverage, packaging, and exact-final-HEAD CI gates remain required.
6. The manually dispatched `Full training capability audit` workflow remains required on the same exact final HEAD before the PR is Ready or merged.

## Invariants

- External CLI and report contracts remain identical.
- `full_training_capability_audit_v1` remains the schema authority.
- The workflow command remains `uv run python scripts/run_training_capability_audit.py --output var/training-capability-audit`.
- The audit remains a diagnostic positive-control implementation probe, not evidence of profitability or production readiness.
- Production training defaults and maintained research workflows do not inherit these audit-only fixture values.
- The regret threshold does not replace reconstruction quality: target reconstruction and economic/non-collapse gates are independently asserted.
- A future change that causes the sequence policy to flip direction, collapse, fail the cash baseline, or exceed the maintained regret threshold must fail closed.

## Failure modes and Test Oracle

The amendment specifically guards against:

- sparse episode segmentation producing insufficient or misleading teacher-event support;
- a wrong-sign target head still appearing superficially trained;
- all-HOLD/all-TRADE/constant-action collapse;
- a policy that trades but loses to cash after costs;
- a policy whose causal regret exceeds the maintained admission floor;
- accidentally reverting to the legacy `0.0` regret default;
- accidentally changing shared rollout-evaluation semantics to make the audit pass; and
- treating standard CI as a substitute for the real workflow-dispatch audit.

Correctness is observed from the authored config values, generated behavior-cloning gate artifacts, causal holdout artifacts, full repository tests, static/architecture gates, compatibility jobs, package/image checks, and the independent Full training capability audit workflow.

## TDD / falsification evidence

Two independent RED phases are preserved as evidence:

- Ownership RED: historical pre-production commit `471eb43d6a913c04118acc81294ff4ba32dfc7b2`, CI run `32131478755`, failed because the package-owned operations boundary did not yet exist.
- Sequence admission RED: commit `956ce6cd9eea04f078bc51a74baad1fab76b82b6`, CI run `32270028766`, produced exactly two intended failures: the audit still authored regret `0.0` instead of `0.2`, and the real sequence integration failed the same `0.0` regret admission check.

The `episode_bars=2` correction is backed by the earlier `episode_bars=8` integration failure at `ff5b0d994223177a928836c401fe3662924b08bd` / run `32229984475`; it is not justified merely because the final implementation passes.

## Quality gate

This amendment does not declare the work complete. Completion still requires, on one exact final HEAD:

- final diff limited to the audit ownership change, its audit-only compatibility corrections, tests, and contract documentation;
- targeted and real sequence tests Green;
- full pytest and branch coverage Green;
- critical branch coverage ratchet Green;
- Ruff, format, Mypy, Import Linter, and dead-code checks Green;
- Ubuntu and Windows compatibility Green;
- training image, image identity, and packaged non-root runtime probe Green;
- package/module/CLI/uv identity Green;
- unresolved review threads absent;
- standard CI Green; and
- `Full training capability audit` workflow-dispatch Green with its independent schema/status assertions and artifact upload.

Until the workflow-dispatch gate is observed on the exact final HEAD, the PR remains Draft and is not complete.