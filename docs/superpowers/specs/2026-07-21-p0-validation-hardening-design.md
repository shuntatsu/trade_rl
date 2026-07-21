# P0 Validation Hardening Design

## Goal

Strengthen the research trust boundary by making sealed outer-test access persistent, proving training-serving observation parity from real environment state, enforcing historical metadata promotion rules, and adding independent accounting and execution-sensitivity verification.

## Scope

1. Run and record repository-wide static checks and tests against one exact commit.
2. Add a persistent sealed-test ledger backed by the artifact catalog, with an in-memory implementation retained only for unit isolation.
3. Add training-serving parity tests using observations captured after real environment steps, covering raw and normalized observations, feature ordering, masks, staleness, book state, pending target, previous action, policy action, and ensemble action.
4. Add a promotion gate that rejects frozen or conservative static Binance metadata for selected-final/release evidence unless explicitly backed by verified historical signed metadata.
5. Add an independent hand-calculation oracle for a two-asset synthetic market without calling production accounting or execution helpers.
6. Add deterministic execution-sensitivity evaluation over fees, spread, slippage, capacity, signal delay, limit-fill policy, and tradability delay.
7. Record multi-seed and unused-period statistics including per-seed return, median, worst seed, drawdown, turnover, baseline difference, and bootstrap confidence interval.

## Architecture

### Persistent sealed-test ledger

Introduce a narrow ledger protocol consumed by `ConcreteFoldRunner`. The PostgreSQL implementation stores one immutable authorization record keyed by `(experiment_plan_digest, dataset_id, fold_index)`. The uniqueness constraint is the enforcement mechanism across processes and machines. The existing in-memory ledger remains a test double but is not used by maintained orchestration when a catalog is configured.

### Serving parity

Expose one canonical observation reconstruction boundary shared by training and serving. Tests run a real environment for several transitions, capture the resulting market and portfolio state, reconstruct the serving observation, and compare every structured component and final action. Flat serving remains supported but must be produced from the canonical structured observation rather than arbitrary caller-owned ordering.

### Historical metadata promotion gate

Promotion policy is phase-aware. Development runs may use `frozen_snapshot` or `conservative_static`. Selected-final, confirmation, serving packaging, and release promotion require `historical_signed` metadata evidence with matching dataset identity and rule-history digest.

### Independent accounting oracle

The oracle lives under tests and reimplements cash, quantity, mark-to-market, fees, funding, split, delisting recovery, margin, PnL, and reward equations directly from documented formulas. It must not import production accounting or execution functions.

### Sensitivity and multi-seed evidence

A deterministic evaluator runs the same selected candidate and baseline under a Cartesian set of execution assumptions. Results are serialized with exact configuration digests. Multi-seed evidence is aggregated without choosing the best seed.

## Error handling

All trust-boundary checks fail closed. Duplicate sealed-test access, metadata-evidence mismatch, observation component mismatch, non-finite accounting, and incomplete sensitivity matrices are errors, not warnings.

## Verification

The branch must pass `ruff check .`, `ruff format --check .`, `mypy .`, and `pytest -q`. GitHub Actions evidence must bind commit SHA, workflow run IDs, source-tree digest, dependency lock digest, and training-image digest. PostgreSQL integration, Serving E2E, and training-image build must run on the same PR head.
