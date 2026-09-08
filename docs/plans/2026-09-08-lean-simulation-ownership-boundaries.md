# Lean Simulation Ownership Boundaries Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize the maintained simulation package into explicit orders, stateful, targets, and diagnostics owners without changing any execution, order, fill, funding, liquidation, or accounting semantics.

**Architecture:** Keep the economic primitives `accounting.py`, `execution.py`, `bar_path.py`, and `liquidity.py` at the simulation root. Move the remaining 14 responsibility-specific modules into four subpackages by filesystem rename, rewrite only import ownership, preserve the exact root `trade_rl.simulation.__all__`, and delete every old private flat path without compatibility shims. Behavioral preservation is verified with the existing simulation suite plus normalized AST equivalence against the exact Phase 3A base.

**Tech Stack:** Python 3.12, pytest, Ruff, Mypy, GitHub Actions, Python `ast`.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md` section 9.7 and the target package layout in section 8.

## Global Constraints

- Logical base is exact Phase 3A head `8a3e1eb3c92d86053ee6305d0116a9218379810c` from Draft PR #442.
- Keep `MarketExecutor` plus `BookState` as the single economic accounting authority.
- Do not change strategy thresholds, model behavior, training budget, fit/evaluation symbol scope, risk limits, or profitability status.
- Do not change fee, spread, impact, funding, borrow, liquidation, order-admission, reconciliation, fill, or accounting ordering semantics.
- Do not keep private compatibility shims for the retired flat simulation module paths.
- Preserve the exact maintained package-level `trade_rl.simulation.__all__` contract.
- `trade_rl.simulation` may depend on lower data/risk contracts but must not depend on `trade_rl.strategies` or `trade_rl.evaluation`.
- One-shot migration/verification helpers must not survive in the final tree.
- The pull request remains Draft and must not be merged without explicit user authorization.

## Quality Contract

### Objective

Make simulation ownership visible from the filesystem while proving that the move is semantically zero for economic execution/accounting behavior.

### Non-goals

- No algorithm rewrite.
- No new order type, fill rule, funding rule, performance metric, or target-exposure behavior.
- No public API expansion beyond package files required for ownership.
- No evaluation reorganization; that is Phase 4.
- No Controlled Experiment Loop implementation.

### Acceptance Criteria

1. Root KEEP files remain at their exact paths: `accounting.py`, `execution.py`, `bar_path.py`, `liquidity.py`.
2. The 14 MOVE files have exactly one new owner under `orders/`, `stateful/`, `targets/`, or `diagnostics/`.
3. All 14 old flat private paths are absent; no forwarding shim remains.
4. Root `trade_rl.simulation.__all__` remains exactly:
   `BookState`, `EconomicTerminationReason`, `ExecutionCostConfig`, `ExecutionEnvironmentStress`, `ExecutionResult`, `MarketExecutor`.
5. Existing package-level imports keep resolving to the same implementation objects after path migration.
6. No production module under `trade_rl/simulation` imports `trade_rl.strategies` or `trade_rl.evaluation`.
7. Golden-ledger, independent-accounting, order, stateful, funding, target-exposure, execution, and liquidity tests retain their existing assertions and pass.
8. A requirements-first AST falsification check proves that moved modules and any changed KEEP/root-facade modules have identical non-import AST after normalizing only authorized old-to-new import owner paths.
9. Ruff, Format, Mypy, full `tests/`, package identity, and normal PR CI pass on the same final HEAD.
10. No temporary migration or falsification workflow/script remains in that final HEAD.

### Invariants

- `BookState` mutations and termination semantics are unchanged.
- `MarketExecutor` sequencing, fee/spread/impact calculations, funding/borrow/liquidation behavior, and quantity semantics are unchanged.
- Order state transitions, admission decisions, reconciliation, active remainder expiration, and terminal reasons are unchanged.
- Stateful bar lifecycle and per-symbol fill ordering are unchanged.
- Target execution and exposure-controller outputs are unchanged.
- Runtime-performance/funding evidence bytes and identity remain governed by their existing tests.
- Root package exports are stable even though private old module paths intentionally break.

### Failure Modes

- Import cycles introduced between the four new subpackages.
- A stale private import silently keeps an old module path alive.
- A file is omitted from the move inventory.
- Ruff import reordering is mistaken for an algorithm change, or vice versa.
- Root KEEP modules are accidentally moved or edited beyond import ownership.
- Order/fill/accounting event ordering changes despite broad test Green.
- `ExecutionEnvironmentStress` disappears from the root facade.
- A temporary migration workflow or script is committed as permanent source.

### Risk

High correctness risk because simulation owns money-state transitions and execution/accounting evidence. The implementation method therefore prioritizes mechanical rename, existing behavioral oracles, independent ledger/property tests, and AST equivalence over manual refactoring.

### Test Oracle

Required observable evidence:

- `tests/simulation/test_golden_ledger_fixtures.py`;
- `tests/simulation/test_independent_accounting_oracle.py`;
- `tests/simulation/test_accounting_*.py`;
- `tests/simulation/test_execution_*.py`;
- `tests/simulation/test_orders.py`;
- `tests/simulation/test_order_admission*.py`;
- `tests/simulation/test_order_reconciliation.py`;
- `tests/simulation/test_stateful_*.py`;
- `tests/simulation/test_funding_*.py`;
- `tests/simulation/test_target_exposure_controller.py`;
- `tests/simulation/test_liquidity*.py`;
- architecture layout/public-export/dependency tests;
- normalized AST comparison against `8a3e1eb3c92d86053ee6305d0116a9218379810c`;
- full `uv run pytest -q tests`;
- Ruff, Format, Mypy, package identity, normal PR CI.

### Required Test Layers

- Architecture/static import contracts.
- Unit/contract simulation tests.
- Stateful integration-style execution/accounting tests.
- Golden/independent accounting regression oracles.
- Property tests through the full test tree.
- Static analysis: Ruff, Ruff format, Mypy.
- Repository-level full test suite.

### Quality Gate

Do not mark Phase 3B verified unless all Acceptance Criteria are evidenced on one final HEAD, the final diff has no one-shot helpers or old shims, the AST falsification review finds no unauthorized non-import semantic difference, and remaining limitations are documented.

---

## File Classification

Every current production file under `trade_rl/simulation` on exact Phase 3A base is classified exactly once.

### KEEP

| Current path | Disposition | Reason |
|---|---|---|
| `trade_rl/simulation/__init__.py` | KEEP, import-owner update only | maintained package facade |
| `trade_rl/simulation/accounting.py` | KEEP | core economic/accounting primitive |
| `trade_rl/simulation/execution.py` | KEEP, import-owner update if required | core execution primitive |
| `trade_rl/simulation/bar_path.py` | KEEP, import-owner update if required | core bar execution primitive |
| `trade_rl/simulation/liquidity.py` | KEEP | core liquidity primitive |

### MOVE

| Current path | New owner |
|---|---|
| `trade_rl/simulation/orders.py` | `trade_rl/simulation/orders/model.py` |
| `trade_rl/simulation/order_admission.py` | `trade_rl/simulation/orders/admission.py` |
| `trade_rl/simulation/order_reconciliation.py` | `trade_rl/simulation/orders/reconciliation.py` |
| `trade_rl/simulation/stateful_runtime.py` | `trade_rl/simulation/stateful/runtime.py` |
| `trade_rl/simulation/stateful_execution.py` | `trade_rl/simulation/stateful/execution.py` |
| `trade_rl/simulation/stateful_bar_lifecycle.py` | `trade_rl/simulation/stateful/bar_lifecycle.py` |
| `trade_rl/simulation/stateful_order_transitions.py` | `trade_rl/simulation/stateful/order_transitions.py` |
| `trade_rl/simulation/stateful_symbol_fills.py` | `trade_rl/simulation/stateful/symbol_fills.py` |
| `trade_rl/simulation/target_execution.py` | `trade_rl/simulation/targets/execution.py` |
| `trade_rl/simulation/target_exposure_controller.py` | `trade_rl/simulation/targets/exposure_controller.py` |
| `trade_rl/simulation/execution_stress.py` | `trade_rl/simulation/diagnostics/execution_stress.py` |
| `trade_rl/simulation/funding_evidence.py` | `trade_rl/simulation/diagnostics/funding.py` |
| `trade_rl/simulation/runtime_performance.py` | `trade_rl/simulation/diagnostics/runtime_performance.py` |
| `trade_rl/simulation/runtime_performance_io.py` | `trade_rl/simulation/diagnostics/runtime_performance_io.py` |

Classification total: `5 KEEP + 14 MOVE = 19` current production files. There is no DELETE-by-omission case.

## Authorized Import Owner Map

Only these private module-owner rewrites are authorized by Phase 3B:

```python
IMPORT_OWNER_MAP = {
    "trade_rl.simulation.orders": "trade_rl.simulation.orders.model",
    "trade_rl.simulation.order_admission": "trade_rl.simulation.orders.admission",
    "trade_rl.simulation.order_reconciliation": "trade_rl.simulation.orders.reconciliation",
    "trade_rl.simulation.stateful_runtime": "trade_rl.simulation.stateful.runtime",
    "trade_rl.simulation.stateful_execution": "trade_rl.simulation.stateful.execution",
    "trade_rl.simulation.stateful_bar_lifecycle": "trade_rl.simulation.stateful.bar_lifecycle",
    "trade_rl.simulation.stateful_order_transitions": "trade_rl.simulation.stateful.order_transitions",
    "trade_rl.simulation.stateful_symbol_fills": "trade_rl.simulation.stateful.symbol_fills",
    "trade_rl.simulation.target_execution": "trade_rl.simulation.targets.execution",
    "trade_rl.simulation.target_exposure_controller": "trade_rl.simulation.targets.exposure_controller",
    "trade_rl.simulation.execution_stress": "trade_rl.simulation.diagnostics.execution_stress",
    "trade_rl.simulation.funding_evidence": "trade_rl.simulation.diagnostics.funding",
    "trade_rl.simulation.runtime_performance": "trade_rl.simulation.diagnostics.runtime_performance",
    "trade_rl.simulation.runtime_performance_io": "trade_rl.simulation.diagnostics.runtime_performance_io",
}
```

No other production semantic rewrite is authorized.

---

### Task 1: Add Executable Phase 3B Architecture Contracts

**Files:**
- Create: `tests/architecture/test_lean_simulation_layout.py`
- Create: `docs/plans/2026-09-08-lean-simulation-ownership-boundaries.md` (this plan)

**Interfaces:**
- Consumes: exact Phase 3A simulation tree and current root `trade_rl.simulation.__all__`.
- Produces: failing structural contract that identifies only missing Phase 3B ownership boundaries.

- [ ] **Step 1: Add the target layout contract**

The test must assert all 19 classified paths have the intended KEEP/MOVE disposition, all four new subpackage `__init__.py` files exist, and all 14 old flat paths are absent.

- [ ] **Step 2: Add the root public export contract**

Assert:

```python
EXPECTED_PUBLIC = {
    "BookState",
    "EconomicTerminationReason",
    "ExecutionCostConfig",
    "ExecutionEnvironmentStress",
    "ExecutionResult",
    "MarketExecutor",
}
assert set(simulation.__all__) == EXPECTED_PUBLIC
for name in EXPECTED_PUBLIC:
    assert getattr(simulation, name) is not None
```

- [ ] **Step 3: Add dependency-direction and stale-import contracts**

Parse every production `trade_rl/simulation/**/*.py` file with `ast`; reject imports whose exact module is one of the 14 retired flat owners, and reject any import beginning with `trade_rl.strategies` or `trade_rl.evaluation`.

- [ ] **Step 4: Run the architecture test on the old tree**

Run:

```bash
uv run pytest -q tests/architecture/test_lean_simulation_layout.py
```

Expected RED: new subpackages/files are absent and old flat paths are present. Existing public API and upward-dependency assertions should remain Green.

- [ ] **Step 5: Run static checks plus full tests on the RED head**

Run:

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Record the exact RED failure count and verify failures come only from the new layout contract before production changes.

---

### Task 2: Mechanically Relocate the Four Simulation Responsibility Families

**Files:**
- Create/move: `trade_rl/simulation/orders/{__init__.py,model.py,admission.py,reconciliation.py}`
- Create/move: `trade_rl/simulation/stateful/{__init__.py,runtime.py,execution.py,bar_lifecycle.py,order_transitions.py,symbol_fills.py}`
- Create/move: `trade_rl/simulation/targets/{__init__.py,execution.py,exposure_controller.py}`
- Create/move: `trade_rl/simulation/diagnostics/{__init__.py,execution_stress.py,funding.py,runtime_performance.py,runtime_performance_io.py}`
- Modify: repository Python imports that use one of the 14 old private module owners.
- Modify only as needed for imports: `trade_rl/simulation/__init__.py`, root KEEP modules, evaluation/strategy/tests that directly import old simulation private modules.
- Delete: all 14 old flat MOVE paths.

**Interfaces:**
- Consumes: exact old module bodies and the Authorized Import Owner Map above.
- Produces: same classes/functions/constants under new private owners; exact package-level public API remains unchanged.

- [ ] **Step 1: Establish a pre-move simulation baseline**

Run:

```bash
uv run pytest -q tests/simulation
```

The exact pass count is recorded as the pre-move behavioral baseline.

- [ ] **Step 2: Move source files by filesystem rename**

Use direct renames for all 14 MOVE rows. Do not copy/reimplement class or function bodies.

- [ ] **Step 3: Rewrite exact import owners repo-wide**

Apply only `IMPORT_OWNER_MAP`. Then use AST import inspection to prove no exact old owner remains in `trade_rl/**/*.py` or `tests/**/*.py`.

- [ ] **Step 4: Add minimal subpackage markers**

Each new subpackage `__init__.py` contains only a responsibility docstring unless a concrete internal import need is independently proven. Do not create a second public facade or speculative re-export layer.

- [ ] **Step 5: Update root simulation facade import ownership**

Change only the owner for `ExecutionEnvironmentStress`:

```python
from trade_rl.simulation.diagnostics.execution_stress import ExecutionEnvironmentStress
```

Keep the six-name `__all__` exactly unchanged.

- [ ] **Step 6: Run focused simulation and architecture tests**

Run:

```bash
uv run pytest -q tests/simulation tests/architecture/test_lean_simulation_layout.py
```

Expected GREEN with no assertion deletion or weakening.

---

### Task 3: Verify Economic and Ordering Semantics Against Independent Oracles

**Files:**
- No production code changes expected.
- Tests remain unchanged except import-owner updates forced by private path removal.

**Interfaces:**
- Consumes: relocated simulation package.
- Produces: evidence that the move did not alter money-state/event ordering behavior.

- [ ] **Step 1: Run golden and independent accounting oracles explicitly**

```bash
uv run pytest -q \
  tests/simulation/test_golden_ledger_fixtures.py \
  tests/simulation/test_independent_accounting_oracle.py \
  tests/simulation/test_accounting_branch_ratchet.py \
  tests/simulation/test_accounting_precision.py \
  tests/simulation/test_accounting_v2.py \
  tests/simulation/test_accounting_validation.py
```

- [ ] **Step 2: Run order/stateful/funding/target suites explicitly**

```bash
uv run pytest -q \
  tests/simulation/test_orders.py \
  tests/simulation/test_order_admission.py \
  tests/simulation/test_order_admission_partial.py \
  tests/simulation/test_order_reconciliation.py \
  tests/simulation/test_stateful_active_remainder_expiry.py \
  tests/simulation/test_stateful_execution.py \
  tests/simulation/test_stateful_execution_adapter.py \
  tests/simulation/test_stateful_execution_characterization.py \
  tests/simulation/test_stateful_execution_services.py \
  tests/simulation/test_stateful_terminal_event_reason.py \
  tests/simulation/test_funding_boundary_evidence.py \
  tests/simulation/test_funding_evidence_artifact.py \
  tests/simulation/test_funding_mark_price.py \
  tests/simulation/test_target_exposure_controller.py
```

- [ ] **Step 3: Treat any behavioral failure as a migration bug**

Do not change expected outputs to make the move Green. Diagnose import/identity/ordering changes and restore semantic equivalence.

---

### Task 4: Falsify Semantic Drift with Normalized AST Comparison

**Files:**
- Temporary verification workflow/script only; it must be deleted before final HEAD.

**Interfaces:**
- Consumes: exact Phase 3A base and relocated Phase 3B production modules.
- Produces: independent structural evidence beyond test Green.

- [ ] **Step 1: Fetch exact Phase 3A base**

Use Git to read source from `8a3e1eb3c92d86053ee6305d0116a9218379810c` without checking it out over the working tree.

- [ ] **Step 2: Compare all 14 moved modules**

For each mapping, normalize old AST import modules using `IMPORT_OWNER_MAP`. Require:

```python
sorted(old_import_nodes) == sorted(new_import_nodes)
old_non_import_top_level_ast == new_non_import_top_level_ast
```

Import ordering alone may differ after Ruff; import membership may not.

- [ ] **Step 3: Compare changed KEEP/facade/outside-production modules**

For every production file changed only because it imported a retired owner, apply the same normalized-import/exact-non-import AST oracle. Any non-import difference is a failure requiring investigation.

- [ ] **Step 4: Verify root public API and retired-path absence**

Require exact root `__all__`, no old files, no stale exact imports, no one-shot helper files.

- [ ] **Step 5: Delete the falsification helper**

The evidence is recorded in the PR; the one-shot workflow/script is removed from the final tree.

---

### Task 5: Final Exact-Head Repository Verification

**Files:**
- No new production changes expected.
- Update PR body with actual measured evidence only.

**Interfaces:**
- Consumes: final Phase 3B tree after all helpers are removed.
- Produces: merge-ready evidence, while PR remains Draft until explicitly authorized.

- [ ] **Step 1: Run static analysis**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
```

- [ ] **Step 2: Run full repository tests**

```bash
uv run pytest -q tests
```

- [ ] **Step 3: Check package identity**

```bash
expected="$(uv run python -c 'from importlib.metadata import version; print(version("trade-rl"))')"
module="$(uv run python -c 'import trade_rl; print(trade_rl.__version__)')"
test "$module" = "$expected"
```

- [ ] **Step 4: Audit final diff against exact Phase 3A base**

Confirm the production change is limited to mechanical moves/import ownership plus minimal package markers, with no unplanned economic/algorithmic file content change.

- [ ] **Step 5: Verify normal PR CI on the same final HEAD**

Do not reuse a bot-produced or prior SHA's Green result. The normal PR CI for the final helper-free HEAD must be `success`.

- [ ] **Step 6: Requirements-first final review**

Re-evaluate the Acceptance Criteria, Invariants, Failure Modes, actual diff, tests, and CI without assuming the migration implementation was correct. Record remaining limitations: tests/AST checks establish preserved repository behavior but do not establish profitability or live-market suitability.
