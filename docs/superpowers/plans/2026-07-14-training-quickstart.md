# Training Quickstart Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a copy-paste training quickstart that generates a valid demo market dataset, runs the maintained `trade-rl train run` command, and explains how to replace the demo with real data.

**Architecture:** Keep the existing training pipeline unchanged. Add one deterministic example dataset builder, one minimal validated training configuration, a root `START.md`, and a focused integration test that executes the builder and parses the configuration through production code.

**Tech Stack:** Python 3.12, NumPy, `trade_rl.data.MarketDataset`, `write_market_dataset_files`, JSON, pytest, Stable-Baselines3 CLI.

## Global Constraints

- The authoritative training entry point remains `trade-rl train run --config CONFIG --dataset DATASET --output STORE`.
- Example training must use the real artifact writer and real `TrainingRunConfig` parser.
- The example must remain CPU-compatible and small enough for a first smoke run.
- Documentation must state that training success does not authorize production trading.
- No synthetic performance or profitability claims.

---

### Task 1: Deterministic demo dataset builder

**Files:**
- Create: `examples/quickstart/create_demo_dataset.py`
- Test: `tests/examples/test_training_quickstart.py`

**Interfaces:**
- Consumes: `trade_rl.data.MarketDataset`, `trade_rl.data.write_market_dataset_files`
- Produces: `build_demo_dataset(n_bars: int = 1024) -> MarketDataset` and CLI output directory containing `manifest.json` plus `arrays.npz`

- [ ] Create a deterministic one-symbol, hourly, continuous market dataset with causal momentum/volatility features and realistic OHLCV shapes.
- [ ] Add argument parsing for `--output` and `--bars`.
- [ ] Write the artifact through `write_market_dataset_files` and print a compact JSON result.
- [ ] Add tests for deterministic identity, successful artifact round-trip, and expected symbols/features.

### Task 2: Minimal maintained training configuration

**Files:**
- Create: `examples/quickstart/training.json`
- Test: `tests/examples/test_training_quickstart.py`

**Interfaces:**
- Consumes: `TrainingRunConfig.from_json(Path)`
- Produces: a CPU-friendly PPO configuration with no alpha/factor artifacts and a three-dimensional dynamic residual action

- [ ] Configure 512 requested timesteps, 64-step PPO rollouts, batch size 32, one seed, 1-hour decisions, 168-hour episodes, time-series trend, and explicit initial capital.
- [ ] Keep exports disabled for the first-run path.
- [ ] Test that production parsing succeeds and the action layout is exactly `fast_tilt`, `slow_tilt`, `risk_tilt`.

### Task 3: Root quickstart documentation

**Files:**
- Create: `START.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: the example builder and config from Tasks 1–2
- Produces: a three-command first run and a real-data migration guide

- [ ] Document prerequisites and installation.
- [ ] Provide exact commands to generate data, train, and inspect `latest.json` and the published run directory.
- [ ] Explain output files, rerunning with a unique `--run-id`, GPU selection, larger training, and real dataset artifact requirements.
- [ ] Document common failures and the `failed/<run-id>` isolation behavior.
- [ ] Link `START.md` prominently from `README.md`.

### Task 4: Verification

**Files:**
- Test: `tests/examples/test_training_quickstart.py`

**Interfaces:**
- Produces: evidence that documentation commands reference real paths and example assets are production-parseable

- [ ] Run `uv run ruff check .`.
- [ ] Run `uv run ruff format --check .`.
- [ ] Run `uv run mypy trade_rl`.
- [ ] Run `uv run lint-imports`.
- [ ] Run `uv run pytest tests/examples/test_training_quickstart.py -q`.
- [ ] Run the standard full pytest command with branch coverage.
- [ ] Run `uv run trade-rl --version`.
