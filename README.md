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
  data -> quality gates -> sealed development/holdout split
       -> training -> walk-forward and holdout evaluation
       -> release eligibility + mandatory risk policy
       -> immutable ServingBundle candidate -> evidence -> registry
       -> approved atomic activation -> live served-identity verification

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

Before a release-capable run, copy and edit [`config/release-risk.example.json`](config/release-risk.example.json). `symbol_liquidity_caps` must exactly cover the resolved bundle symbols.

Run the complete validation and training pipeline:

```bash
uv run python scripts/run_pipeline.py \
  --source postgres \
  --git-sha "$(git rev-parse HEAD)" \
  --model-version model-YYYYMMDD-N \
  --risk-config config/release-risk.example.json
```

A release-capable run requires a non-empty sealed holdout, passing mandatory gates, and a complete risk policy. `--force`, `--skip-p0`, `--skip-wf`, or `--skip-gate` makes a run research-only: it may produce reports, but it cannot construct or register a candidate. Use `--no-register` for intentional research runs.

A successful eligible run constructs and registers an immutable candidate bundle. It does **not** activate the candidate. Activation is reserved for the deployment workflow after evidence and environment approval.

Registry operations are available through:

```bash
uv run python scripts/manage_registry.py --registry-dir output/model_registry list
uv run python scripts/manage_registry.py --registry-dir output/model_registry show-active
```

## Serving Plane

Required environment variables:

```text
TRADE_RL_SERVING_TOKEN      required bearer token
TRADE_RL_RELEASE_GIT_SHA    exact 40-character SHA of the running release
TRADE_RL_REGISTRY_DIR       registry directory
TRADE_RL_AUDIT_DB           SQLite audit/replay database
TRADE_RL_DATA_DIR           market-data directory
TRADE_RL_ALLOWED_ORIGINS    comma-separated allowlist
TRADE_RL_HOST               default 127.0.0.1
TRADE_RL_PORT               default 8001
```

Start serving:

```bash
export TRADE_RL_RELEASE_GIT_SHA="$(git rev-parse HEAD)"
uv run python scripts/run_server.py
```

Exposed routes:

- `GET /health`
- `GET /ready`
- `POST /api/signal/latest` with `Authorization: Bearer ...`

`/ready` reports the active model version, bundle digest, and running release Git SHA. Strict Production serving rejects an active bundle built from a different Git SHA and preserves the previous healthy in-memory bundle.

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

## Local validation

P0 uses the release candidate's resolved `horizon` and `decision_every`. The explicit `--p0-days` option changes only the synthetic sample duration; it does not replace candidate timing.

Serving snapshots are content-addressed: the snapshot identity hashes the ordered schema, selected timestamps, feature values, global values, and close history. The CSV provider uses only a completed bar and measures age from that bar's close time for `15m`, `1h`, `4h`, and `1d` data.

Run the exchange-free local drill with:

```bash
uv run python scripts/run_local_gameday.py
```

The legacy dashboard server is development-only and requires `TRADE_RL_ENABLE_LEGACY_METRICS_SERVER=1` unless an intentional development caller opts in directly. The filesystem Registry is a single-node local implementation; passing this drill does not establish multi-node or testnet readiness.

