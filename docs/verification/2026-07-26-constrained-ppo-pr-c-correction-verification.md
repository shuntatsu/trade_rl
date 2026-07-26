# Corrected Lagrangian PPO Verification Record

## Scope

This record verifies the corrected PR C Lagrangian PPO contract implemented on branch `agent/pr191-stability-correction-v2` for PR #193.

The correction preserves the maintained all-cost net-log-growth reward and leaves ordinary `ppo` and PR B `cost_critic_ppo` behavior unchanged. It corrects the constrained actor, completed-episode estimator, dual scheduling, canonical-action probe, diagnostics, and checkpoint identity.

This verification does **not** establish production budgets, production safety, sealed-test superiority, or permission to promote a trained policy. Those remain separate evaluation and governance decisions.

## Verified implementation contract

The verified implementation has the following normative behavior:

- The actor composes raw advantages as `A_reward - A_cost @ lambda`.
- Reward and individual cost advantages are not independently normalized for optimization.
- Only the final combined actor-advantage vector receives the pinned SB3 Torch normalization.
- A zero multiplier vector retains the ordinary PPO actor path.
- Multipliers are frozen for the complete rollout and every minibatch and epoch derived from it.
- Elapsed time and explicit completion semantics are stored in rollout order.
- Economic termination and maintained time-limit completion contribute completed episodes.
- Shadow/external truncations are censored, clear the affected environment state, and do not enter the numerator or denominator.
- Constraint estimates retain observations across warmup, interval, and minimum-support skips.
- Rare event constraints require their configured completed-episode support before a dual update.
- EMA decay is denominator-aware and the dual update remains an integral projected update.
- Lower-bound state and upper-cap saturation are reported separately; `saturated` means upper-cap saturation only.
- Target-weight zero action is recorded as cash; residual zero action is recorded as baseline.
- Canonical-action budget violations create warning evidence but do not reject training solely because an estimate exceeds a budget.
- Raw effective-penalty diagnostics use `raw_cost_advantages * frozen_multipliers`.
- Standardized cost advantages are retained only as separately named correlation diagnostics.
- Probe payload, Lagrangian schema, completion semantics, state schema versions, and corrected actor semantics are bound into checkpoint identity.

## Targeted correction suite

Verified branch head: `acbfdb38982d2029f5542ce7866209b5c2ecc755`

Workflow:

- Name: `PR C Correction Targeted Verification`
- Run ID: `30209703694`
- Job ID: `89813766436`
- Conclusion: success
- Exit code: `0`
- Result: `154 passed in 10.27s`

The workflow checked out `github.head_ref`, so the verified branch head is the run metadata head above. The `GITHUB_SHA` printed inside a pull-request workflow is GitHub's synthetic pull-request merge ref and is not used as the branch-head identity in this record.

The targeted command covered:

- raw advantage composition and final-only actor normalization;
- time-aware episode aggregation and censoring;
- pooled dual scheduling, minimum support, EMA, and boundaries;
- canonical-action probe semantics and fail-closed malformed inputs;
- raw penalty diagnostics and deterministic evidence;
- rollout storage alignment;
- actor parity and correction integration;
- checkpoint continuation and evidence persistence;
- SB3 backend construction and probe integration;
- Cost Critic diagnostics/evidence regressions.

## Full CI-equivalent verification

Verified branch head: `acbfdb38982d2029f5542ce7866209b5c2ecc755`

Main CI:

- Workflow: `CI`
- Run ID: `30209703690`
- Conclusion: success

Completed successful checks:

- workflow security;
- Ruff;
- Ruff format check;
- Mypy;
- import architecture;
- dead-code report;
- recovery and structured-serving smoke;
- full tests and coverage;
- critical branch coverage;
- CLI smoke;
- Ubuntu compatibility;
- Windows compatibility;
- complete training-image build, identity recording, and packaged non-root runtime probe.

Full pytest result:

- `1820 passed, 2 skipped, 11 warnings in 89.00s`
- required coverage threshold: `80.0%`
- total coverage: `85.49%`
- covered lines: `23007 / 25713`
- branch coverage: `73.43%` (`6231 / 8486` branches)

Platform and image jobs:

- Ubuntu compatibility job ID: `89813796092` — success
- Windows compatibility job ID: `89813796111` — success
- Training image job ID: `89813796096` — success
- Rebuilt Core job ID: `89813796100` — success

The repository's PostgreSQL Catalog workflow also completed successfully on the same head:

- Run ID: `30209703688`

## Controlled behavioral evidence

| Contract | Verification evidence |
|---|---|
| Zero-multiplier ordinary-PPO policy and optimizer parity | Actor correction and actor-advantage integration tests in the 154-test targeted suite passed. |
| Raw cost-unit conversion invariance | Raw-composition and actor-correction tests passed with reciprocal multiplier scaling. |
| Frozen multipliers across every minibatch and epoch | Lagrangian PPO integration tests passed with rollout-frozen multiplier assertions. |
| Irregular-time drawdown and turnover aggregation | Completed-episode tests passed for 6-hour/18-hour weighted aggregation, including drawdown `0.175`, margin deficit `0.07`, and turnover `3.5`. |
| Shadow censoring excluded from estimator support | Episode accumulator tests passed with censored transitions clearing state and contributing no numerator or denominator. |
| Warmup observations retained | Dual-controller tests passed with observations accumulated before the first eligible update. |
| Twenty-episode rare-event support | Configuration and dual-controller tests passed for the canonical rare-event minimum support of 20 completed episodes. |
| Denominator-aware EMA | Dual-controller tests passed for `beta_effective = ema_beta ** consumed_denominator`. |
| Lower bound versus upper cap | Dual and stability diagnostics tests passed with independent lower-bound reporting and upper-cap-only saturation. |
| Checkpoint continuation equality | Checkpoint round-trip tests passed for dual state, accumulator state, frozen multipliers, Cost Critic state, optimizer state, probe identity, and next-step continuation. |
| Cash versus baseline probe semantics | Canonical-action probe tests passed for target-weight cash and residual baseline semantics. |
| Warning without rejection | SB3 backend probe tests passed while continuing training with violated probe budgets recorded as warning evidence. |
| Ordinary PPO and Cost Critic PPO regressions | Full repository suite, platform compatibility, and training-image checks passed. |

## Superseded documents

The following older PR C plans now carry a prominent superseded notice and are non-normative where they conflict with the correction specification and plan:

- `docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-lagrangian.md`
- `docs/superpowers/plans/2026-07-26-constrained-ppo-pr-c-stability-addendum.md`

Normative correction documents:

- `docs/superpowers/specs/2026-07-26-pr-c-lagrangian-stability-correction.md`
- `docs/superpowers/plans/2026-07-26-pr-c-lagrangian-stability-correction.md`

## Decision

The corrected PR C implementation is verified for its stated research and engineering scope on the exact implementation head recorded above. The PR should remain draft until final documentation cleanup receives its own exact-head CI result and the PR metadata is updated. Production budget selection, deployment safety, and strategy promotion remain out of scope.
