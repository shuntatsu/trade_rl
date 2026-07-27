# Dataset Catalog Reconciliation Implementation Plan

> **For agentic workers:** Use superpowers:test-driven-development and verification-before-completion for every task.

**Goal:** Decouple immutable dataset publication from optional PostgreSQL registration and provide a retryable reconciliation workflow and CLI.

**Architecture:** Keep `trade_rl.data` filesystem-only. Reconstruct a validated `PublishedDatasetArtifact` from an existing artifact directory, compose registration at workflow level, and expose an explicit catalog reconciliation command.

## Task 1: RED publication boundary contracts

**Files:**
- Create: `tests/data/test_dataset_catalog_boundary.py`
- Modify later: `trade_rl/data/artifact.py`

- [ ] Publish a canonical dataset while `TRADE_RL_DATABASE_URL` is set and patch catalog construction to fail if consulted.
- [ ] Assert publication succeeds and the artifact loads correctly.
- [ ] Add an architecture assertion that Python files under `trade_rl/data` contain no `trade_rl.catalog` imports.
- [ ] Run the focused tests and confirm they fail because publication still invokes catalog registration.

## Task 2: RED retryable reconciliation contracts

**Files:**
- Create: `tests/workflows/test_dataset_catalog_reconciliation.py`
- Modify later: `trade_rl/data/artifact.py`
- Create later: `trade_rl/workflows/dataset_catalog_reconciliation.py`

- [ ] Publish one dataset artifact.
- [ ] Use a fake catalog whose first `register()` call fails and second call succeeds.
- [ ] Assert the artifact remains valid after failure and the second call uses the same immutable digest and location.
- [ ] Assert reconstructed registration metadata and dataset identity are exact.
- [ ] Run the focused tests and confirm failure because the reconciliation workflow does not exist.

## Task 3: Implement filesystem-only publication and artifact inspection

**Files:**
- Modify: `trade_rl/data/artifact.py`
- Modify: `trade_rl/data/__init__.py`

- [ ] Remove catalog imports and implicit registration from `publish_market_dataset_artifact()`.
- [ ] Add `inspect_published_market_dataset_artifact(root)` that validates the complete artifact, reads the canonical manifest digest, and returns `PublishedDatasetArtifact`.
- [ ] Keep the directory contract exactly `manifest.json` and `arrays.npz`.
- [ ] Run focused data tests.

## Task 4: Implement reconciliation workflow

**Files:**
- Create: `trade_rl/workflows/dataset_catalog_reconciliation.py`
- Test: `tests/workflows/test_dataset_catalog_reconciliation.py`

- [ ] Implement `reconcile_market_dataset_catalog(artifact_root, catalog)`.
- [ ] Load the dataset and typed published identity.
- [ ] Build registration with `market_dataset_registration()`.
- [ ] Call `catalog.register()` without migration or environment lookup.
- [ ] Run focused workflow and catalog tests.

## Task 5: Add CLI reconciliation command

**Files:**
- Modify: `trade_rl/cli/catalog.py`
- Modify: `tests/cli/test_catalog_commands.py`

- [ ] Add `catalog reconcile-market-dataset --artifact-root PATH`.
- [ ] Resolve catalog using the existing explicit database URL contract.
- [ ] Invoke the workflow and print the normal artifact record schema.
- [ ] Add CLI tests for success and failure propagation.

## Task 6: Exact-head verification

- [ ] Run full pytest and coverage.
- [ ] Run Ruff, format, Mypy, Import Linter, dead-code, critical coverage, CLI smoke, Windows, Ubuntu, training image, and PostgreSQL Catalog workflows.
- [ ] Confirm final diff contains no temporary source-export or updater files.
- [ ] Record RED and GREEN evidence in the PR.
- [ ] Squash merge the exact verified head into main; production remains `NO-GO`.
