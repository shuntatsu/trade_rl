# Testing

## Required CI gates

Every proposed merge must pass on the exact PR head:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy mars_lite
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

A previous successful run does not validate a newer commit.

## Test layers

### Unit

- ServingBundle digest, path, file-set, and schema validation;
- immutable Registry registration and atomic active pointer;
- observation construction with real current weights;
- account-state and pending-order validation;
- guardrail turnover and loss/drawdown calculations;
- pre-trade worst-case exposure;
- bearer authentication and replay store.

### Integration

- candidate construction, registration, activation, and served identity;
- corrupted candidate preserves the prior healthy runtime;
- current positions are present before policy prediction;
- feature normalization, feature mask, symbol order, and global-feature order match the bundle;
- read-only Serving routes expose no destructive operation;
- deployment gate runs before Registry activation;
- rollback returns the served identity to a known-good version.

### Adversarial

- manifest tampering and path traversal;
- non-finite metrics and threshold override attempts;
- concurrent Registry operations;
- partial disk-copy and active-pointer failures;
- pending-order one-sided fill scenarios;
- request replay and conflicting request payloads;
- stale, NaN, and all-zero market data.

### Research and strategy regression

P0, walk-forward, replay simulation, baseline, synthetic-data, and training tests remain useful for detecting algorithmic regressions. Their success does not establish live profitability.

## Test replacement policy

A test may be deleted when its underlying contract has been removed, but equivalent safety coverage must be added for the new contract. Tests must not preserve obsolete behavior merely to keep an old implementation alive.

## Slow tests

The full suite includes CPU-intensive learning and evaluation cases. Focused tests may be used during development, but merge status is determined only by the complete CI suite and coverage gate.

## Evidence in PRs

PR descriptions must identify:

- exact head SHA;
- CI run ID;
- lint, format, mypy, pytest, and coverage results;
- known untested external integrations;
- remaining Production blockers.
