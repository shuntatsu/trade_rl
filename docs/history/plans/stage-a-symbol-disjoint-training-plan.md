# Stage A Symbol-Disjoint Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ensure Stage A shared-policy training uses only combinations of symbol-disjoint train symbols and cannot leak validation/test symbol identity into training.

**Architecture:** Derive a dedicated triplet schedule from `SymbolDisjointManifest`, generalize the existing resumable plan to variable cycle sizes, and make the maintained Binance command accept only the new manifest. Existing cursor, transfer, PostgreSQL dataset, and checkpoint contracts remain unchanged.

**Tech Stack:** Python 3.12, frozen dataclasses, canonical JSON/content digests, pytest, existing Stage A workflow modules.

## Global Constraints

- No validation or test symbol may appear in a train stage.
- The maintained Stage A operator command must reject the legacy all-symbol triplet manifest.
- Existing legacy manifest tests and plans remain readable for non-Stage-A compatibility.
- Plan stages per cycle are derived from the bound manifest and must repeat exactly.
- No reward, PPO, execution, or serving behavior changes.

---

### Task 1: Symbol-disjoint triplet manifest

**Files:**
- Create: `trade_rl/workflows/symbol_disjoint_triplet_manifest.py`
- Create: `tests/workflows/test_symbol_disjoint_triplet_manifest.py`

- [ ] Write tests for 84/1/1 counts, split subset closure, balanced train appearances, deterministic ordering, JSON round trip, and tamper rejection.
- [ ] Run the focused test and verify import failure.
- [ ] Implement the immutable manifest and exact JSON loader/writer.
- [ ] Re-run focused tests and static checks.
- [ ] Commit.

### Task 2: Variable-size resumable plan

**Files:**
- Modify: `trade_rl/workflows/symbol_triplet_training_cursor.py`
- Modify: `tests/workflows/test_symbol_triplet_training_cursor.py`

- [ ] Write a failing test proving a 9-symbol train manifest creates 84 stages per cycle.
- [ ] Replace the hard-coded 319-stage validation with a manifest-derived cycle size.
- [ ] Keep existing 319-stage legacy tests green.
- [ ] Run cursor, orchestrator, runner, and command tests.
- [ ] Commit.

### Task 3: Fail-closed Stage A command

**Files:**
- Modify: `trade_rl/workflows/binance_symbol_triplet_stage_command.py`
- Modify: `tests/workflows/test_binance_symbol_triplet_stage_command.py`

- [ ] Write failing tests that reject the legacy manifest and load the symbol-disjoint manifest.
- [ ] Switch the command loader to the new manifest type.
- [ ] Verify a completed plan returns before Binance metadata and PostgreSQL access.
- [ ] Run focused and complete static checks.
- [ ] Commit.

### Task 4: Full verification

- [ ] Run all affected local tests.
- [ ] Open a draft PR with the discovered leak and RED/GREEN evidence.
- [ ] Require full CI, Ubuntu/Windows compatibility, Training image, and PostgreSQL Catalog on one exact head.
