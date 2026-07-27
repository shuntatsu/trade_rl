# Dataset Publication and Catalog Reconciliation Design

## Problem

`publish_market_dataset_artifact()` currently performs two different operations behind one API:

1. atomically publishes an immutable dataset directory to the filesystem;
2. if `TRADE_RL_DATABASE_URL` is present, registers the published artifact in PostgreSQL.

The filesystem rename completes before catalog registration. If PostgreSQL registration then fails, the caller receives an exception even though the immutable artifact already exists. Retrying the same publication fails with `FileExistsError`, so the apparent failure cannot be recovered by repeating the original operation.

The data layer also imports the catalog service directly, coupling canonical artifact persistence to optional infrastructure and ambient environment.

## Decision

Make filesystem publication and catalog synchronization separate, explicit operations.

`publish_market_dataset_artifact()` will only validate, stage, atomically rename, and return `PublishedDatasetArtifact`. It will never inspect `TRADE_RL_DATABASE_URL` or contact PostgreSQL.

Add a workflow-level reconciliation operation:

```python
reconcile_market_dataset_catalog(
    artifact_root: Path,
    catalog: ArtifactCatalog,
) -> ArtifactRecord
```

The workflow will:

1. load and fully validate the existing immutable dataset artifact;
2. reconstruct its typed `PublishedDatasetArtifact` identity from the canonical manifest;
3. build the existing `ArtifactRegistration`;
4. call the catalog's idempotent `register()` method.

A failed catalog attempt leaves the filesystem artifact unchanged and can be retried with the same artifact path. No republishing is required.

## CLI

Add:

```text
trade-rl catalog reconcile-market-dataset --artifact-root PATH
```

The command resolves PostgreSQL through the existing `--database-url` / `TRADE_RL_DATABASE_URL` mechanism and runs the reconciliation workflow. Catalog migration remains a separate explicit command.

## Dependency Direction

- `trade_rl.data` owns filesystem publication, loading, and artifact inspection only.
- `trade_rl.catalog` owns catalog contracts and adapters.
- `trade_rl.workflows.dataset_catalog_reconciliation` composes data and catalog.
- `trade_rl.cli.catalog` exposes the explicit operational command.

The data package must not import `trade_rl.catalog`.

## Compatibility

`publish_market_dataset_artifact()` keeps its signature and return type. Existing callers that only need the artifact continue to work, but no longer receive an exception from optional catalog infrastructure.

`register_artifact_if_configured()` remains available for unrelated legacy callers during this PR; the dataset publication path stops using it.

## Evidence and Error Handling

- Filesystem publication errors remain exceptions and leave no final destination.
- Catalog reconciliation errors remain exceptions, but the exception occurs in a clearly named catalog operation after publication has already been acknowledged.
- Reconciliation is safely repeatable because PostgreSQL registration is keyed by immutable artifact identity.
- The artifact directory remains exactly `manifest.json` and `arrays.npz`; no mutable sidecar is added.

## Testing

Tests must prove:

1. publication never constructs a catalog even when the database environment is set;
2. catalog failure does not change or remove the published artifact;
3. the same artifact can be reconciled successfully after an earlier failed attempt;
4. reconciliation reconstructs the exact artifact digest, dataset ID, location, and metadata;
5. the CLI invokes reconciliation with the explicitly resolved catalog;
6. an architecture test prevents `trade_rl.data` from importing `trade_rl.catalog`.
