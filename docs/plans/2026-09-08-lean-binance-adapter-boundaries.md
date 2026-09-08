# Lean Binance Adapter Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Replace the flat ~60 KB Binance integration and its two adjacent helper modules with a dependency-acyclic `trade_rl.integrations.binance` package whose folders reflect transport, Vision, cache, metadata, and dataset responsibilities without changing observable behavior.

**Architecture:** Introduce an IO-free `types.py` as clarified by the Binance boundary amendment. Split behavior through exhaustive top-level-definition classification; no top-level function/class from the old adapter may be silently dropped. Preserve package-level public APIs, exact URL/query/retry/cache/parsing semantics, and dataset identity. Remove old flat modules rather than forwarding them.

**Tech Stack:** Python 3.12 stdlib HTTP/ZIP/CSV, NumPy, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md`, Amendment 1, and `docs/specs/2026-09-08-binance-adapter-boundary-amendment.md`.

## Global Constraints

- Base exact verified Phase 2A head: `343c8af5dfcea8a472ccc44ef2b617ea7f4d0c68`.
- Preserve every symbol intentionally exported by the old `trade_rl.integrations.binance.__all__` through the new package `__init__.py`.
- Preserve current `trade_rl.integrations` package-level exports, including `FrozenBinanceExchangeInfoTransport`.
- `types.py` performs no network/filesystem IO and imports no other Binance submodule.
- `cache.py` must not import concrete `BinancePublicTransport`; use a protocol.
- `metadata.py` must not import concrete transport implementation; frozen metadata wrapper uses a protocol.
- No request URL/query/retry/backoff/cache-evidence/archive parsing/funding alignment/metadata/dataset semantic change.
- No compatibility files at `integrations/binance.py`, `binance_cache.py`, or `frozen_binance_metadata.py` in final tree.
- Full same-head CI and requirements-first falsification are required.

---

## Task 1: Exhaustive definition inventory and RED layout contracts

**Files:**
- Create: `docs/plans/2026-09-08-binance-definition-inventory.md`
- Create: `tests/architecture/test_lean_binance_layout.py`
- Modify: `tests/integrations/*` imports only as RED requires

- [ ] **Step 1: Enumerate every old top-level definition with AST**

Run on exact base:

```python
import ast
from pathlib import Path

for source in (
    Path("trade_rl/integrations/binance.py"),
    Path("trade_rl/integrations/binance_cache.py"),
    Path("trade_rl/integrations/frozen_binance_metadata.py"),
):
    tree = ast.parse(source.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            print(source, node.lineno, node.end_lineno, type(node).__name__, node.name)
```

Write every listed definition into the inventory with exactly one owner: `types`, `vision`, `cache`, `metadata`, `transport`, `dataset`, or DELETE with evidence. No production migration may start until classification is exhaustive.

- [ ] **Step 2: Write RED physical-layout tests**

Require:

```text
integrations/binance/__init__.py
integrations/binance/types.py
integrations/binance/vision.py
integrations/binance/cache.py
integrations/binance/metadata.py
integrations/binance/transport.py
integrations/binance/dataset.py
```

and absence of:

```text
integrations/binance.py
integrations/binance_cache.py
integrations/frozen_binance_metadata.py
```

- [ ] **Step 3: Write dependency-cycle tests**

AST tests must reject:

```text
types -> any trade_rl.integrations.binance.*
cache -> transport
metadata -> transport
vision -> transport/cache/metadata/dataset
transport -> dataset
dataset -> no upward caller dependency
```

Allowed direction is the amendment's dependency graph.

- [ ] **Step 4: Write public-export regression**

Record the exact old `binance.py.__all__` set and assert the new package exposes the same names. Separately preserve the current `trade_rl.integrations` exports.

- [ ] **Step 5: Run RED and commit tests/inventory only**

Expected: layout/import collection fails because flat modules still exist.

---

## Task 2: Create `types.py` and split Vision pure functions

**Files:**
- Create: `trade_rl/integrations/binance/types.py`
- Create: `trade_rl/integrations/binance/vision.py`
- Test: URL planning, funding URL, archive parsing/epoch normalization fixture tests

- [ ] **Step 1: Move shared enums/errors to `types.py`**
- [ ] **Step 2: Move interval/Vision constants, UTC/time helpers, URL planners and archive parsing to `vision.py`**
- [ ] **Step 3: Run existing URL/parser tests against new owners**
- [ ] **Step 4: Commit only when old/new behavior matches fixture outputs**

---

## Task 3: Split cache ownership without transport cycle

**Files:**
- Create: `trade_rl/integrations/binance/cache.py`
- Delete later: `trade_rl/integrations/binance_cache.py`

- [ ] **Step 1: Move low-level cache path/evidence validation helpers**
- [ ] **Step 2: Move cache plan/report/inspect/require/sync functions**
- [ ] **Step 3: Replace concrete transport type dependency with `_VisionArchiveTransport` protocol only**
- [ ] **Step 4: Run cache corruption/missing/empty/download tests**

---

## Task 4: Split metadata and frozen snapshot ownership

**Files:**
- Create: `trade_rl/integrations/binance/metadata.py`
- Delete later: `trade_rl/integrations/frozen_binance_metadata.py`

- [ ] **Step 1: Move exchange-info snapshot and instrument metadata types/parsers**
- [ ] **Step 2: Move `FrozenBinanceExchangeInfoTransport` using a local protocol, not concrete transport import**
- [ ] **Step 3: Run metadata, effective execution-rule, tamper/digest/frozen-cache tests**

---

## Task 5: Split bounded public transport

**Files:**
- Create: `trade_rl/integrations/binance/transport.py`

- [ ] **Step 1: Move `BinancePublicTransport` with exact constructor defaults**
- [ ] **Step 2: Keep REST endpoints/query construction/retry/backoff byte-equivalent in behavior**
- [ ] **Step 3: Import Vision planning/parsing and low-level cache verification through lower modules**
- [ ] **Step 4: Run transport retry/error/cache/REST-vs-Vision tests**

---

## Task 6: Split dataset/source assembly and delete old flat modules

**Files:**
- Create: `trade_rl/integrations/binance/dataset.py`
- Create: `trade_rl/integrations/binance/__init__.py`
- Delete: `trade_rl/integrations/binance.py`
- Delete: `trade_rl/integrations/binance_cache.py`
- Delete: `trade_rl/integrations/frozen_binance_metadata.py`
- Modify: `trade_rl/integrations/__init__.py`
- Modify all tests/internal imports

- [ ] **Step 1: Move kline/funding conversion, `BinanceMarketDataSource`, feature presets, and `build_binance_market_dataset`**
- [ ] **Step 2: Build package facade matching old maintained `binance.py.__all__`**
- [ ] **Step 3: Rewrite repository imports to real owners or package facade**
- [ ] **Step 4: Delete old flat modules; no shim**
- [ ] **Step 5: Require inventory classification coverage and no stale import paths**

---

## Task 7: Full falsification and exact-head quality gate

- [ ] Run `ruff check trade_rl tests`.
- [ ] Run `ruff format --check trade_rl tests`.
- [ ] Run `mypy trade_rl`.
- [ ] Run full `pytest -q tests`.
- [ ] Verify package identity.
- [ ] Compare exact Phase 2A base to final head; production diff outside `integrations/binance*` must be import-only.
- [ ] Confirm old flat modules and every temporary migration helper are absent from exact final tree.
- [ ] Falsify retry count/backoff, cache evidence/tamper detection, Vision URL order, REST query fields, archive parsing, funding alignment, exchange metadata, frozen metadata, dataset identity, and package public exports.
- [ ] Require normal same-head GitHub Actions success and record exact counts.
- [ ] Keep Draft; do not merge without explicit user authorization.
