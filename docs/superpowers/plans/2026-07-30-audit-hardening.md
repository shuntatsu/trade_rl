# Post-audit Hardening Implementation Plan

**Goal:** Make execution promotion, margin configuration, artifact publication, and model-loading boundaries fail closed against the audited defects.

**Architecture:** Preserve existing module boundaries, reject unsupported semantics instead of approximating them, bind execution evidence to the complete execution configuration, and deserialize mutable filesystem artifacts only from private verified copies.

**Tech stack:** Python 3.12, dataclasses, NumPy, pytest, Stable-Baselines3, canonical JSON, and SHA-256 artifact contracts.

## Global constraints

- Keep selected-final training fail closed.
- Reject legacy execution-promotion evidence rather than silently migrating it.
- Do not present proportional account-wide collateral allocation as multi-asset isolated margin.
- Do not weaken checkpoint architecture, dataset, environment, seed, timestep, or training identity checks.
- Do not introduce a new runtime dependency.

## Task 1: Execution configuration boundary

- [x] Add RED tests for tail multipliers below one, economic identity drift, and unsupported multi-asset isolated margin.
- [x] Require `tail_slippage_multiplier >= 1.0` whenever tail probability is non-zero.
- [x] Expand `execution_policy_v2` identity to include complete mechanics and economic settings.
- [x] Reject multi-asset isolated margin while retaining the semantically equivalent single-asset case.
- [x] Pass targeted simulation tests.

## Task 2: Execution promotion evidence v2

- [x] Add a RED test proving zero order events could be promoted under the old contract.
- [x] Advance the schema to `execution_promotion_evidence_v2`.
- [x] Require at least one order event in addition to `complete_order_evidence`.
- [x] Bind promotion evidence to the complete `execution_policy_v2` digest.
- [x] Update maintained promotion fixtures and tests.
- [x] Pass targeted promotion tests.

## Task 3: Artifact publication failure handling

- [x] Add RED tests for fixed temporary-name reuse and pointer-write partial publication.
- [x] Use process-unique, exclusively created temporary pointer files.
- [x] Roll a published run back to staging when the `latest.json` update fails.
- [x] Increase generated run-ID time resolution from seconds to microseconds.
- [x] Preserve atomic last-writer-wins semantics for `latest.json` across distinct successful runs.
- [x] Pass targeted ArtifactStore tests.

## Task 4: Checkpoint and replay deserialization boundary

- [x] Add RED tests for symlinked checkpoint manifests and policies.
- [x] Require regular, non-symlink checkpoint and replay files.
- [x] Open source files with no-follow semantics where the platform supports them.
- [x] Copy opened content into a private temporary directory and verify digest/size again.
- [x] Load SB3 checkpoints and pickle replay buffers only from the private verified copy.
- [x] Keep private-copy behavior tested on Windows; keep symlink-specific tests POSIX-only.
- [x] Pass targeted checkpoint, resume, and transfer tests.

## Task 5: Dead and misleading API cleanup

- [x] Confirm `ExecutionCostConfig.rate_per_turnover` has no production consumer.
- [x] Confirm `MarketExecutor._capacity_notional` is preserved only by direct private-helper tests.
- [x] Remove both APIs and obsolete tests/imports.
- [x] Retain public market-notional behavior through the maintained dataset path.

## Task 6: Verification and publication

- [x] Reproduce the audited failures before implementation.
- [x] Pass the targeted 71-test GREEN suite after implementation.
- [x] Pass targeted Ruff and Mypy checks before the implementation commit.
- [x] Compare the implementation commit against `main` and remove temporary scripts/workflows.
- [x] Open PR #301 with the design, regression tests, implementation, and compatibility impact.
- [ ] Pass the repository's complete pull-request CI and PostgreSQL workflow on the final branch head.
- [ ] Inspect final CI diagnostics and merge only after every required check is green.
