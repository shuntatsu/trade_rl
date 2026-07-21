# PostgreSQL Artifact Catalog Design

## Goal

Run a local PostgreSQL service with Docker Compose and use it as the authoritative searchable catalog for reusable research artifacts. Large numerical arrays and model files remain in immutable filesystem artifacts; PostgreSQL stores identity, provenance, dependency edges, locations, lifecycle state, and cache lookup keys.

## Scope

The first implementation provides:

- a pinned PostgreSQL Docker Compose service with a persistent named volume and health check;
- versioned SQL migrations applied by the Python package;
- a framework-independent artifact catalog contract;
- a PostgreSQL adapter using psycopg 3;
- idempotent registration and exact cache-key lookup;
- dependency edges between source and derived artifacts;
- CLI commands for migration, health, registration, lookup, and listing;
- optional automatic registration of published market-dataset artifacts when a database URL is configured;
- unit tests plus a real PostgreSQL integration test in GitHub Actions.

This change does not move NumPy arrays, checkpoints, or models into PostgreSQL. It does not yet persist SequencePolicyPlane, fold normalizers, or Oracle teacher payloads; it creates the catalog foundation and registers canonical market datasets. Those artifact types can be connected incrementally without schema changes.

## Architecture

### Docker service

`compose.yaml` defines PostgreSQL 16 with:

- database `trade_rl`;
- application user `trade_rl`;
- credentials supplied from environment variables with local-development defaults;
- host port configurable through `TRADE_RL_POSTGRES_PORT`;
- a named volume for `/var/lib/postgresql/data`;
- `pg_isready` health checking;
- no application container and no public-network assumptions.

`.env.example` documents the development DSN. Real credentials remain untracked.

### Catalog package

A new `trade_rl.catalog` package has clear boundaries:

- `contracts.py`: immutable artifact records, artifact kinds, statuses, queries, and repository protocol. It is standard-library only.
- `postgres.py`: psycopg implementation and connection management. Importing the package remains possible without the optional PostgreSQL dependency; attempting to connect without psycopg raises a focused installation error.
- `migrations.py`: ordered migration discovery, advisory locking, checksum verification, and transactional application.
- `service.py`: environment-based optional registration helpers used by publication workflows.

The PostgreSQL adapter is below workflows and above no domain logic. Data and artifact code may call the small service seam, but catalog contracts do not import market-data or RL modules.

### Database model

`catalog_schema_migrations`

- migration version;
- immutable migration checksum;
- application timestamp.

`catalog_artifacts`

- `artifact_digest` primary key;
- `artifact_kind`;
- `schema_version`;
- optional `dataset_id`;
- `cache_key_digest`;
- canonical JSON `cache_key`;
- canonical JSON `metadata`;
- filesystem or object-store `location`;
- byte size;
- lifecycle status;
- creation and last-seen timestamps.

A uniqueness constraint on `(artifact_kind, cache_key_digest)` enforces one authoritative artifact for an exact reusable computation identity. Re-registering the same digest and payload is idempotent. Conflicting identities fail closed.

`catalog_artifact_dependencies`

- parent artifact digest;
- child artifact digest;
- dependency role;
- composite primary key and cascading foreign keys.

Indexes support artifact-kind, dataset, status, creation-time, and cache-key lookup.

### Cache identity

Callers provide a canonical cache-key mapping. The application computes its SHA-256 digest from canonical JSON. Examples:

- market dataset: source evidence, symbols, interval, range, feature-config digest, normalization digest, metadata mode;
- future sequence plane: dataset ID, layout digest, normalizer digest;
- future fold normalizer: dataset ID, train range, sequence layout digest;
- future Oracle teacher: dataset ID, train range, environment digest, action-spec digest, teacher-config digest.

PostgreSQL never decides whether two computations are equivalent; it only enforces the exact identity supplied by the relevant domain module.

## Publication flow

Canonical filesystem publication remains atomic and authoritative for bytes:

1. write and validate the artifact in staging;
2. atomically publish the directory;
3. if `TRADE_RL_DATABASE_URL` is set, register the published artifact in PostgreSQL;
4. if catalog registration fails, leave the valid filesystem artifact intact and return a clear error so the caller can retry registration idempotently.

The initial automatic hook is limited to market datasets. It records the dataset ID, artifact digest, schema version, path, array file size, and identity payload metadata.

## CLI

Add `trade-rl catalog` commands:

- `migrate` — apply pending migrations;
- `health` — connect and verify migration state;
- `register` — register a generic artifact from explicit JSON cache-key and metadata payloads;
- `find` — exact artifact-kind plus cache-key lookup;
- `list` — recent artifacts with optional kind, dataset, and status filters.

The DSN comes from `--database-url` or `TRADE_RL_DATABASE_URL`. JSON output follows existing CLI conventions and never prints passwords.

## Error handling

- Migration checksums are immutable; a changed applied migration fails.
- Migration application uses a PostgreSQL advisory lock to prevent races.
- Registration validates SHA-256 digests, non-empty kinds/schema/location, non-negative sizes, and canonical JSON values.
- Duplicate cache keys with different artifact digests fail rather than overwrite.
- Database connection errors are surfaced with no fallback to an implicit local database.
- Catalog use is optional unless a catalog command is invoked or the environment variable is configured.

## Testing

- Contract tests for validation and deterministic cache-key digests.
- Repository tests for idempotent registration, conflicting registration, lookup, filtering, dependencies, and migration checksum behavior.
- CLI tests using a fake repository seam.
- Publication-hook tests proving no database import or connection occurs without configuration.
- A GitHub Actions PostgreSQL service test applies migrations and executes the real psycopg integration suite.
- Existing full CI remains authoritative for lint, typing, architecture boundaries, and regression coverage.

## Operational commands

Start the database:

```bash
docker compose up -d postgres
docker compose ps
docker compose exec postgres pg_isready -U trade_rl -d trade_rl
```

Install and migrate:

```bash
uv sync --extra dev --extra postgres
export TRADE_RL_DATABASE_URL=postgresql://trade_rl:trade_rl@localhost:5432/trade_rl
uv run trade-rl catalog migrate
uv run trade-rl catalog health
```

Stop while retaining data:

```bash
docker compose down
```

Delete the local database volume only when explicitly desired:

```bash
docker compose down -v
```
