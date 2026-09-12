# Execution Economics Build Contract Implementation Plan

Status: Active

**Goal:** Repair #481 by adding one explicit build-level execution-economics authority, preserve historical/legacy identities, introduce Canonical M2 bootstrap config v2, and regenerate a real-cost research-assumption baseline without mutating the superseded zero-cost Study.

**Spec:** `docs/specs/2026-09-12-execution-economics-build-contract.md`

## Task 1 — Profile contract and strict parser (TDD)

- [ ] Add RED tests for exact profile payload, strict unknown-field rejection, finite/non-negative validation, participation range, fee model mutual exclusion, and borrow availability/rate consistency.
- [ ] Verify RED reaches pytest after Ruff/Format/Mypy and fails only because the profile authority does not exist.
- [ ] Add `trade_rl/data/build/economics.py` with `ExecutionEconomicsProfile` and shared strict payload parser/serializer.
- [ ] Export only the capability-level contract needed by build/bootstrap callers; do not add a repository-wide registry.
- [ ] Run focused tests and static checks.

## Task 2 — Market JSON request and Builder plumbing (TDD)

- [ ] Add RED tests for optional `execution_economics` in `load_market_build_request()` and strict nested-key rejection.
- [ ] Add `execution_economics: ExecutionEconomicsProfile | None` to `MarketDatasetBuildRequest`, separate from `MarketBuildConfig`.
- [ ] Add optional `execution_economics` parameter to `MarketDatasetBuilder.build()`.
- [ ] Pass the profile through the one existing `build_market_economic_semantics()` call.
- [ ] Bind explicit profile canonical payload to dataset identity metadata only when a profile is supplied.
- [ ] Prove profile omission preserves legacy behavior/content identity.
- [ ] Prove economics-only changes modify dataset ID while preserving feature_config_digest and normalization_digest.

## Task 3 — Artifact and executor integration (TDD)

- [ ] Add artifact round-trip assertions for fee/maker/taker/spread/participation/borrow arrays and dataset ID.
- [ ] Add deterministic one-trade zero-overlay regression: configured dataset fee/spread produces exact non-zero total cost according to the existing executor formula.
- [ ] Add explicit no-double-count regression for generic vs maker/taker fee models.
- [ ] Confirm no executor formula or overlay defaults are changed.

## Task 4 — Binance build plumbing (TDD)

- [ ] Add `execution_economics` optional argument to `build_binance_market_dataset()`.
- [ ] Pass it unchanged into `MarketDatasetBuilder.build()`.
- [ ] Add focused integration test proving Binance build output carries profile economics without modifying feature preset/metadata responsibilities.
- [ ] Preserve public API surface except the function parameter; no new integration-owned economics type.

## Task 5 — Canonical M2 bootstrap v1/v2 evolution (TDD)

- [ ] Pin existing v1 fixture/payload/digest/read behavior before changing production.
- [ ] Add v2 tests: `execution_economics` required, exact strict payload, profile digest binding, missing profile rejected.
- [ ] Update `CanonicalM2BootstrapConfig` to support v1 (profile absent) and v2 (profile required) explicitly.
- [ ] Keep v1 serializer/reader byte-semantically compatible for existing payloads.
- [ ] Pass v2 profile into `build_binance_market_dataset()`.
- [ ] Verify inspected/published bootstrap reload retains the v2 config/profile and dataset economics.

## Task 6 — Falsification and full repository gate

- [ ] Mutate/drop profile after parsing and verify tests fail.
- [ ] Mutate builder so profile does not affect identity and verify tests fail.
- [ ] Allow generic+maker/taker simultaneous positive fees and verify tests fail.
- [ ] Revert participation plumbing to 1.0 and verify tests fail.
- [ ] Remove v2 required-profile check and verify tests fail.
- [ ] Run Ruff, Format, production Mypy, architecture-tooling Mypy, full pytest, build, sdist/direct-wheel/rebuilt-wheel source closure, clean install, CLI and package identity.
- [ ] Re-read final diff from Acceptance Criteria and current main, not from implementation intent.

## Task 7 — Canonical M2 repair artifact

- [ ] Construct a new `canonical_m2_bootstrap_config_v2` using the documented `canonical_m2_research_assumption_v1` profile: generic fee 0.0005, spread 0.0002, participation 0.05, borrow rate 0.
- [ ] Label the numbers explicitly as reproducible research assumptions, not historical/account-specific Binance truth.
- [ ] Bootstrap a new immutable dataset/Study lineage from frozen/approved source evidence.
- [ ] Run the new baseline EvidenceSet.
- [ ] Verify dataset ID, artifact digest, Study digest and baseline EvidenceSet fingerprint differ from the superseded zero-cost lineage.
- [ ] Inspect baseline diagnostics and require non-zero `total_cost` for trading strategy observations that actually trade.
- [ ] Verify superseded Study SHA/digest/artifacts remain unchanged and inspectable.

## Task 8 — Durable docs and cleanup

- [ ] Update `docs/research/current-status.md`: M1 complete; previous Canonical M2 is zero-trading-cost diagnostic; repaired real-cost-assumption lineage and exact current state.
- [ ] Update existing architecture doc(s) with only durable responsibility/compatibility rules; do not duplicate full field lists unnecessarily.
- [ ] Remove this completed plan and matching active spec from current tree after implementation/research artifact verification.
- [ ] Restore `docs/README.md` to “no Active spec/plan” when complete.
- [ ] Final exact-head CI after cleanup and current-main synchronization.

## Quality Gate

Completion requires all spec Acceptance Criteria, preserved v1/legacy identity evidence, explicit no-double-count semantics, new canonical baseline evidence with observed non-zero trading cost, immutable old lineage, exact-head CI, final topology/diff review, falsification review, and documented residual realism limits.
