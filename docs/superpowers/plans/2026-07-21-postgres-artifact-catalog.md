# PostgreSQL Artifact Catalog Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Docker Compose PostgreSQL service and a typed, migration-backed artifact catalog that indexes immutable research artifacts without storing large numerical payloads in the database.

**Architecture:** Keep filesystem artifacts authoritative for bytes. Add framework-independent catalog contracts, a psycopg PostgreSQL adapter, transactional migrations, CLI operations, and an optional market-dataset publication hook keyed by exact content identity.

**Tech Stack:** Python 3.12, PostgreSQL 16, Docker Compose, psycopg 3, SQL, pytest, GitHub Actions.

## Global Constraints

- PostgreSQL stores metadata, identity, dependencies, locations, and lifecycle state only.
- NumPy arrays, checkpoints, models, and teacher payloads remain immutable filesystem artifacts.
- Database use remains optional unless a catalog command is invoked or `TRADE_RL_DATABASE_URL` is configured.
- All cache lookups use canonical JSON and SHA-256 digests.
- Duplicate cache keys with different artifact digests fail closed.
- Applied migration checksums are immutable.
- No password may appear in CLI JSON output.
- Existing artifact publication remains atomic and valid even if catalog registration later fails.

---

### Task 1: Docker PostgreSQL development service

**Files:**
- Create: `compose.yaml`
- Create: `.env.example`
- Modify: `.gitignore`
- Modify: `README.ja.md`
- Test: `tests/ops/test_postgres_compose_contract.py`

**Interfaces:**
- Produces: service `postgres`, database/user `trade_rl`, environment variable `TRADE_RL_DATABASE_URL`.

- [ ] Write a failing contract test that parses `compose.yaml` with PyYAML and asserts PostgreSQL 16, persistent volume, health check, configurable port, and environment-backed credentials.
- [ ] Run `pytest tests/ops/test_postgres_compose_contract.py -q` and verify failure because the compose file is absent.
- [ ] Add the compose service, local environment example, ignore rules, and startup documentation.
- [ ] Run the focused test and verify it passes.

### Task 2: Catalog contracts and deterministic cache identities

**Files:**
- Create: `trade_rl/catalog/__init__.py`
- Create: `trade_rl/catalog/contracts.py`
- Test: `tests/catalog/test_contracts.py`
- Modify: `.importlinter`

**Interfaces:**
- Produces: `ArtifactKind`, `ArtifactStatus`, `ArtifactRegistration`, `ArtifactRecord`, `ArtifactQuery`, `ArtifactCatalog`, and `cache_key_digest(mapping) -> str`.

- [ ] Write failing tests for SHA-256 validation, canonical cache-key order independence, metadata JSON validation, non-negative size, and immutable record fields.
- [ ] Run `pytest tests/catalog/test_contracts.py -q` and verify import failure.
- [ ] Implement standard-library-only contracts and protocol.
- [ ] Add an import-linter contract preventing `trade_rl.catalog.contracts` from importing psycopg, NumPy, Torch, SB3, workflows, or integrations.
- [ ] Run the focused tests and import-linter.

### Task 3: Transactional PostgreSQL migrations

**Files:**
- Create: `trade_rl/catalog/sql/0001_artifact_catalog.sql`
- Create: `trade_rl/catalog/sql/__init__.py`
- Create: `trade_rl/catalog/migrations.py`
- Test: `tests/catalog/test_migrations.py`

**Interfaces:**
- Produces: `Migration`, `load_migrations()`, and `apply_migrations(connection) -> tuple[int, ...]`.

- [ ] Write failing tests using a recording connection for ordered discovery, checksum verification, advisory locking, rollback on failure, and no-op reapplication.
- [ ] Run the focused test and verify failure.
- [ ] Implement migration loading with `importlib.resources`, SHA-256 checksums, transaction use, and advisory lock `hashtext('trade_rl_catalog_migrations_v1')`.
- [ ] Add tables `catalog_schema_migrations`, `catalog_artifacts`, and `catalog_artifact_dependencies` with exact constraints and indexes from the spec.
- [ ] Run focused tests and static checks.

### Task 4: psycopg repository adapter

**Files:**
- Create: `trade_rl/catalog/postgres.py`
- Modify: `pyproject.toml`
- Test: `tests/catalog/test_postgres_repository.py`

**Interfaces:**
- Produces: `PostgresArtifactCatalog(database_url)`, `.migrate()`, `.health()`, `.register()`, `.find()`, `.list()`, and `.add_dependency()`.

- [ ] Write failing repository tests against a fake DB-API connection for parameterized SQL, idempotent registration, cache-key conflict detection, exact lookup, filters, dependency insert, and safe optional dependency errors.
- [ ] Run the focused tests and verify failure.
- [ ] Add optional dependency group `postgres = ["psycopg[binary]>=3.2,<4"]` and mypy ignore for `psycopg.*`.
- [ ] Implement the adapter with explicit transactions and JSONB parameters; never interpolate caller values into SQL.
- [ ] Run focused tests, Ruff, and Mypy.

### Task 5: CLI catalog operations

**Files:**
- Create: `trade_rl/cli/catalog.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/cli/test_catalog_commands.py`

**Interfaces:**
- Produces: `trade-rl catalog migrate|health|register|find|list`.

- [ ] Write failing CLI tests with a repository factory seam and JSON output assertions.
- [ ] Run focused tests and verify the `catalog` group is missing.
- [ ] Implement DSN resolution from `--database-url` then `TRADE_RL_DATABASE_URL`, rejecting absent configuration.
- [ ] Implement commands without exposing the DSN or password in output.
- [ ] Run focused and existing CLI tests.

### Task 6: Optional market-dataset registration hook

**Files:**
- Create: `trade_rl/catalog/service.py`
- Modify: `trade_rl/data/artifact.py`
- Modify: `trade_rl/cli/app.py`
- Test: `tests/catalog/test_service.py`
- Test: `tests/data/test_market_artifact_catalog.py`

**Interfaces:**
- Produces: `register_artifact_if_configured(registration) -> ArtifactRecord | None` and `market_dataset_registration(published, dataset) -> ArtifactRegistration`.

- [ ] Write failing tests proving no psycopg import/connection without configuration, exact dataset cache identity, idempotent registration, and filesystem publication preceding DB registration.
- [ ] Run focused tests and verify failure.
- [ ] Implement the environment-gated service seam and market-dataset registration after atomic publication.
- [ ] Ensure registration errors do not delete or corrupt the published filesystem artifact.
- [ ] Run focused tests and the complete data artifact suite.

### Task 7: Real PostgreSQL integration and CI

**Files:**
- Create: `tests/catalog/test_postgres_integration.py`
- Create: `.github/workflows/postgres-catalog.yml`
- Modify: `README.ja.md`

**Interfaces:**
- Consumes: Docker/PostgreSQL service and public catalog API.
- Produces: verified migration/register/find/list/dependency behavior against PostgreSQL 16.

- [ ] Write an integration test marked `postgres` that skips without `TRADE_RL_TEST_DATABASE_URL`.
- [ ] Add the marker to pytest configuration.
- [ ] Add a PR workflow with PostgreSQL 16 service, `uv sync --extra dev --extra postgres`, migration execution, and the integration test.
- [ ] Run the workflow and inspect failures.
- [ ] Fix only demonstrated failures, then rerun.

### Task 8: Full verification and publication

**Files:**
- Remove any temporary source-export workflow.
- Review all changed files.

**Interfaces:**
- Produces: a clean PR and merge-ready implementation.

- [ ] Run focused catalog, CLI, data artifact, and compose tests.
- [ ] Run `ruff check .`, `ruff format --check .`, `mypy trade_rl`, `lint-imports`, and full `pytest` with coverage.
- [ ] Run Docker Compose config validation and the PostgreSQL integration workflow.
- [ ] Confirm no credentials, generated databases, volumes, temporary workflows, or binary artifacts are committed.
- [ ] Update PR summary with exact verification evidence and merge after all required checks succeed.
