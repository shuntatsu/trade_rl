# Trade RL

**English** | [日本語](README.ja.md)

Trade RL is a portfolio reinforcement-learning research and deployment codebase for producing risk-constrained target weights across multiple instruments.

## Status

**Production: NO-GO.**

The repository contains a separated offline Control Plane and authenticated, read-only Serving Plane. Code and CI alone do not authorize live trading. Production remains blocked until the unchecked items in [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) have attached operational evidence.

No return, Sharpe ratio, benchmark result, or synthetic experiment in this repository is a promise of future profitability.

## Architecture

```text
Control Plane
  data -> quality gates -> training -> walk-forward/holdout evaluation
       -> ServingBundle candidate -> evidence -> immutable registry
       -> approved atomic activation

Serving Plane
  authenticated account state + cached market snapshot
       -> shared observation builder -> policy -> DecisionPipeline
       -> guardrails -> pre-trade risk verdict -> read-only response
```

The only normative architecture document is [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Setup

Python 3.12 and `uv` are recommended.

```bash
uv sync --all-extras --dev
uv run ruff check .
uv run ruff format --check .
uv run mypy mars_lite
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

## Control Plane

Run the complete validation and training pipeline:

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N
```

A successful run constructs and registers an immutable candidate bundle. It does **not** activate the candidate. Activation is reserved for the deployment workflow after evidence and environment approval.

Registry operations are available through:

```bash
uv run python scripts/manage_registry.py --registry-dir output/model_registry list
uv run python scripts/manage_registry.py --registry-dir output/model_registry show-active
```

## Serving Plane

Required environment variables:

```text
TRADE_RL_SERVING_TOKEN      required bearer token
TRADE_RL_REGISTRY_DIR       registry directory
TRADE_RL_AUDIT_DB           SQLite audit/replay database
TRADE_RL_DATA_DIR           market-data directory
TRADE_RL_ALLOWED_ORIGINS    comma-separated allowlist
TRADE_RL_HOST               default 127.0.0.1
TRADE_RL_PORT               default 8001
```

Start serving:

```bash
uv run python scripts/run_server.py
```

Exposed routes:

- `GET /health`
- `GET /ready`
- `POST /api/signal/latest` with `Authorization: Bearer ...`

The Serving Plane contains no training, model deletion, promotion, rollback, or registry-mutation routes.

## Documentation

- [Japanese documentation / 日本語ドキュメント](docs/ja/README.md)
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — current system architecture
- [`docs/MODEL_LIFECYCLE.md`](docs/MODEL_LIFECYCLE.md) — candidate, evidence, registry, activation, rollback
- [`docs/OPERATIONS.md`](docs/OPERATIONS.md) — deployment and incident procedures
- [`docs/SECURITY.md`](docs/SECURITY.md) — trust boundaries and threats
- [`docs/TESTING.md`](docs/TESTING.md) — tests and acceptance gates
- [`docs/PRODUCTION_READINESS.md`](docs/PRODUCTION_READINESS.md) — GO/NO-GO checklist
- [`docs/DECISIONS.md`](docs/DECISIONS.md) — architectural decisions
- [`docs/RESEARCH_HISTORY.md`](docs/RESEARCH_HISTORY.md) — non-authoritative research history

Approved specifications and implementation plans are retained under `docs/superpowers/`.