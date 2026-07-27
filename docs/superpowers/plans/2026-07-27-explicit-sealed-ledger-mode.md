# Explicit Sealed-Test Ledger Mode Implementation Plan

> **For agentic workers:** Use superpowers:test-driven-development and verification-before-completion for every task.

**Goal:** Make outer-test ledger semantics an explicit, identity-bound configuration choice and require durable PostgreSQL for maintained formal walk-forward runs.

**Architecture:** Add a small enum and field to `MarketWalkForwardConfig`, resolve the ledger through a pure mode-directed factory, record the mode in published evidence, and update maintained example configurations. Keep the fold runner and ledger implementations unchanged.

## Task 1: RED configuration and factory contracts

**Files:**
- Create: `tests/workflows/test_explicit_sealed_ledger_mode.py`
- Modify later: `trade_rl/workflows/market_walk_forward_config.py`
- Modify later: `trade_rl/workflows/market_walk_forward.py`

- [ ] Add tests proving local mode ignores `TRADE_RL_DATABASE_URL`.
- [ ] Add a test proving durable mode fails without a database URL.
- [ ] Add a test proving durable mode constructs PostgreSQL resources without calling migration.
- [ ] Add a test proving mode changes `digest_payload()` and experiment-plan digest.
- [ ] Run the focused file and confirm failures are caused only by the missing explicit mode contract.

## Task 2: Implement explicit mode

**Files:**
- Modify: `trade_rl/workflows/market_walk_forward_config.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`

- [ ] Add `SealedTestLedgerMode(StrEnum)` with `LOCAL_EXPLORATORY` and `DURABLE_POSTGRES`.
- [ ] Add `sealed_test_ledger_mode` to `MarketWalkForwardConfig` and its canonical digest.
- [ ] Parse the JSON field with strict enum validation and legacy default `local_exploratory`.
- [ ] Replace the ambient `_sealed_test_ledger()` behavior with `_sealed_test_ledger(mode)`.
- [ ] Never call `catalog.migrate()` in the walk-forward execution path.
- [ ] Pass the explicitly selected ledger into `ConcreteFoldRunner`.
- [ ] Record mode, durability, and evidence tier in `walk-forward.json`.
- [ ] Run focused workflow/config/PostgreSQL tests.

## Task 3: Bind maintained configurations

**Files:**
- Modify: `examples/binance/walk-forward-smoke.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-growth-optimal.json`
- Modify relevant example contract tests.

- [ ] Set smoke to `local_exploratory`.
- [ ] Set maintained full and growth-optimal profiles to `durable_postgres`.
- [ ] Add contract assertions so future edits cannot silently remove or downgrade the mode.

## Task 4: Exact-head verification

- [ ] Run full pytest and coverage.
- [ ] Run Ruff, format, Mypy, Import Linter, dead-code, critical coverage, CLI smoke, Windows, Ubuntu, training image, and PostgreSQL Catalog workflows.
- [ ] Confirm final diff contains no temporary workflow or test-export helper.
- [ ] Record RED and GREEN evidence in the PR before merge.
