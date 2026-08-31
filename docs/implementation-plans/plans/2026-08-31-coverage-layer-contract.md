# Coverage Layer Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore the configured 80% core coverage gate without weakening it by separating exact-optional NautilusTrader coverage into its existing isolated capability layer and adding a dedicated Nautilus coverage ratchet.

**Architecture:** Keep `Rebuilt Core` responsible for code executable under its declared dependency set and exclude only `trade_rl/integrations/nautilus/**`, which requires the exact optional `nautilus_trader` runtime. Measure that adapter package separately inside `Nautilus Capability`, preserving its current one-test-group-per-process topology and using coverage.py subprocess/multiprocessing support so native kernel tests remain isolated. No production or trading behavior changes.

**Tech Stack:** Python 3.12, pytest, coverage.py 7.15.x, GitHub Actions, TOML/INI coverage configuration.

**Spec:** Existing `pyproject.toml` coverage gate, `.github/workflows/ci.yml`, and `.github/workflows/nautilus-capability.yml` contracts plus the 2026-08-31 exact-head coverage diagnostics.

## Baseline Evidence

- Exact core coverage before the split: `79.18290882942891%`.
- The same artifact excluding only `trade_rl/integrations/nautilus/**`: `80.09797033943163%`.
- Isolated Nautilus diagnostic run `33359438890`: combined `81.32361189007291%`, statements `87.52751283932501%`, branches `61.19047619047619%`.
- Dedicated Nautilus combined-coverage ratchet: `81.3%`.

## Global Constraints

- Keep `[tool.coverage.report].fail_under = 80` unchanged.
- Keep all existing critical-coverage file/group thresholds unchanged.
- Do not modify trading, simulation, reward, policy, data, or research behavior.
- Do not run all Nautilus capability tests in one pytest process; the native runtime aborts under that topology.
- Exclude only the exact optional Nautilus adapter package from the core coverage source set.
- Add a dedicated Nautilus coverage measurement and fail-closed ratchet before considering the scope split complete.
- Keep all existing Nautilus capability tests and assertions; coverage instrumentation must not replace or weaken them.

---

### Task 1: Lock the two-layer coverage contract with a failing architecture test

**Files:**
- Create: `tests/architecture/test_coverage_layer_contract.py`
- Modify later: `pyproject.toml`
- Create later: `.coveragerc.nautilus`
- Modify later: `.github/workflows/nautilus-capability.yml`

**Interfaces:**
- Consumes: repository TOML/config/workflow text.
- Produces: exact assertions for core omit scope, unchanged 80% threshold, dedicated Nautilus config, and isolated coverage workflow wiring.

- [ ] **Step 1: Write the failing test**

Assert that:

```python
core = config["tool"]["coverage"]["run"]
assert core["omit"] == ["trade_rl/integrations/nautilus/*"]
assert config["tool"]["coverage"]["report"]["fail_under"] == 80
assert (ROOT / ".coveragerc.nautilus").is_file()
```

Also assert the dedicated config measures `trade_rl/integrations/nautilus`, enables branch/parallel coverage, and the maintained Nautilus workflow invokes coverage without collapsing its separate pytest commands.

- [ ] **Step 2: Run RED verification**

Run:

```bash
uv run pytest -q tests/architecture/test_coverage_layer_contract.py
```

Expected: FAIL because the core omit and dedicated Nautilus coverage config/workflow do not yet exist.

### Task 2: Split core and Nautilus coverage responsibilities

**Files:**
- Modify: `pyproject.toml`
- Create: `.coveragerc.nautilus`
- Modify: `.github/workflows/nautilus-capability.yml`
- Test: `tests/architecture/test_coverage_layer_contract.py`

**Interfaces:**
- Core coverage: unchanged global threshold, omits only `trade_rl/integrations/nautilus/*`.
- Nautilus coverage: source `trade_rl/integrations/nautilus`, branch coverage, parallel data, subprocess + multiprocessing support.

- [ ] **Step 1: Add the exact core omit**

Add only:

```toml
omit = ["trade_rl/integrations/nautilus/*"]
```

under `[tool.coverage.run]`. Do not change `fail_under` or critical coverage tables.

- [ ] **Step 2: Add dedicated coverage config**

Create `.coveragerc.nautilus` with source restricted to the Nautilus adapter package, branch coverage enabled, parallel data enabled, `multiprocessing,thread` concurrency, `sigterm = True`, and `patch = subprocess,_exit` so child processes save evidence.

- [ ] **Step 3: Instrument the existing capability workflow without changing test grouping**

Keep each existing pytest invocation separate. Start each invocation under the dedicated coverage config, combine/report after all capability tests, upload `nautilus-coverage.json`, and enforce the measured baseline ratchet established by the diagnostic run.

- [ ] **Step 4: Verify GREEN locally/targeted**

Run:

```bash
uv run pytest -q tests/architecture/test_coverage_layer_contract.py
uv run ruff check .
uv run ruff format --check .
```

Expected: PASS.

### Task 3: Verify both quality layers on one exact head

**Files:** no production changes expected.

**Interfaces:**
- Core oracle: full pytest + global combined coverage >= 80%.
- Nautilus oracle: existing capability tests + dedicated branch coverage ratchet.

- [ ] **Step 1: Run exact-head core quality**

Expected: full suite passes, global coverage is at least 80%, critical coverage remains green, Mypy/import-linter/package identity remain green.

- [ ] **Step 2: Run exact-head Nautilus capability**

Expected: all existing capability probes remain green under isolated process topology and dedicated Nautilus coverage meets or exceeds the recorded baseline.

- [ ] **Step 3: Falsification review**

Confirm no broader omit pattern was introduced, no Nautilus tests were removed/skipped, no fail-under value was reduced, no coverage failure is hidden by `continue-on-error`, and the dedicated report is generated from the same PR head.

- [ ] **Step 4: Review final diff and residual risk**

Expected: only coverage/test/workflow/docs changes; no production source changes. Record that coverage proves execution of paths, not economic correctness or production readiness.
