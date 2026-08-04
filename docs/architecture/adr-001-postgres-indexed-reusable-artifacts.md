# ADR-001: PostgreSQL-indexed reusable training artifacts

## Status

Accepted

## Context

Full walk-forward training repeatedly produces expensive deterministic artifacts,
especially fold normalizers and Oracle teacher datasets. Teacher arrays are about
200 MB per identity and market datasets are multiple GB, while lookups are low-rate
and local to one training host. PostgreSQL already provides the durable artifact
catalog and the training-data Docker volume already provides persistent binary
storage.

## Decision

Use PostgreSQL as the authoritative reusable-artifact index. Store the canonical
cache identity, content digest, schema version, dataset dependency, status, size,
and durable location in `catalog_artifacts`. Keep large immutable payloads as
content-addressed compressed files on `trade-rl-training-data`.

An artifact is reusable only when its SQL record is `ready`, its location remains
beneath the configured cache root, the file exists, and the artifact loader verifies
all declared content and identity digests. A missing SQL row falls back to the
deterministic filesystem key and backfills the catalog after successful validation.

The first indexed/reused artifact kinds are:

- Oracle episode and rollout Teacher datasets
- Fold observation normalizers

## Trade-offs

- PostgreSQL stores pointers and audit metadata rather than multi-GB BYTEA values.
  Database backups alone therefore do not contain the binary payloads; the named
  Docker volume must also be backed up.
- A catalog outage fails runs that explicitly enable durable reuse. This is accepted
  because the same PostgreSQL instance is already required for the maintained market
  dataset and sealed-test ledger.
- Cache identities are schema-versioned. Implementations must bump the identity
  schema when numerical semantics change.

## Consequences

- Equivalent generations can discover and reuse artifacts without recomputation.
- Digest conflicts fail closed instead of silently replacing evidence.
- SQL remains queryable and compact while large arrays retain efficient NPZ access.
- New reusable artifact types should use `ReusableArtifactIndex` instead of adding
  ad-hoc caches.
