# Base Hygiene Oracle Repair Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore trustworthy cross-platform and V10 execution test oracles without changing production behavior.

**Architecture:** Keep this repair test-only. Normalize the authored V4 configuration through Python text-mode newline handling before hashing, and align V10 risk-reduction assertions with the existing execution-eligibility contract introduced after the V11 research run. Keep repository-wide formatting cleanup and U0 comparator changes out of this branch.

**Tech Stack:** Python 3.12, pytest, NumPy, GitHub Actions, Ruff.

**Spec:** Existing V10 execution contract in `trade_rl/learning/causal_alpha_v10_hierarchy.py` and V11 consolidated research report.

## Global Constraints

- Do not change production code or economic behavior.
- Do not change V9/V10/V11 research artifacts or run identity.
- Preserve executable partial risk reduction when the cap satisfies entry/no-trade constraints.
- Require explicit reduce-only flattening when a partial risk reduction cannot execute under the current contract.
- Make the V4 authored-config identity independent of CRLF versus LF checkout policy only; preserve sensitivity to all other text changes.
- Keep repository-wide Ruff formatting cleanup in a separate change.

---

### Task 1: Cross-platform V4 authored-config oracle

**Files:**
- Modify: `tests/architecture/test_causal_alpha_v5_boundaries.py`

**Interfaces:**
- Consumes: `examples/binance/universal-causal-alpha-v4-research.json`
- Produces: newline-stable SHA-256 assertion over UTF-8 text content

- [x] **Step 1: Establish failing evidence**

Ubuntu CI at `dd0c4b7` reports the raw-byte digest `a5369a00...` while the test expects the CRLF digest `560e08e0...`; Windows passes the same test.

- [x] **Step 2: Correct the oracle**

Read the file with `Path.read_text(encoding="utf-8")`, hash the resulting normalized text bytes, and assert the normalized digest `a5369a00d3862fec96ba2ffb74d2b745c3829a0801464fac5052536c34be2c51`.

- [ ] **Step 3: Verify on Linux and Windows CI**

Expected: the architecture compatibility test passes on both platforms.

### Task 2: Non-executable V10 hard-risk reduction oracle

**Files:**
- Modify: `tests/learning/test_causal_alpha_v10_closed_loop.py`

**Interfaces:**
- Consumes: `CausalAlphaV10ExecutionContract`, `_partial_risk_reduction_executable`
- Produces: assertions that distinguish executable partial reduction from fail-closed flattening

- [x] **Step 1: Establish failing evidence**

Full-suite evidence at the current U0 head, which does not modify V10 production code, shows stale expectations for a `0.04` cap under `entry_threshold=0.10` and `no_trade_band=0.05`.

- [x] **Step 2: Correct non-executable assertions**

Require a requested target of `0.0`, `risk_projection` path reason, `risk_cap_flatten` hierarchy reason, and `reduce_only=True` when the cap cannot execute.

- [x] **Step 3: Preserve executable partial-reduction coverage**

Keep the existing `0.05` cap case with `entry_threshold=0.05` and `no_trade_band=0.05` asserting a partial target of `0.05`.

- [ ] **Step 4: Run focused V10 tests**

Expected: `tests/learning/test_causal_alpha_v10_closed_loop.py` passes.

### Task 3: Micro-risk reduce-only oracle

**Files:**
- Modify: `tests/learning/test_causal_alpha_v10_reduce_only.py`

**Interfaces:**
- Consumes: V10 risk-cap projection and no-trade-band behavior
- Produces: explicit fail-closed flatten assertion for sub-band micro reductions

- [x] **Step 1: Establish failing evidence**

The stale test expects `0.1004 -> 0.10`; current production contract rejects the `0.0004` delta because it is below the `0.05` no-trade band.

- [x] **Step 2: Correct the assertion**

Require target `0.0`, hierarchy reason `risk_cap_flatten`, and `reduce_only=True`.

- [ ] **Step 3: Run focused reduce-only tests**

Expected: `tests/learning/test_causal_alpha_v10_reduce_only.py` passes.

### Task 4: Verification and review

**Files:**
- No production changes expected.

**Interfaces:**
- Consumes: final branch diff and CI evidence
- Produces: reviewable Draft PR

- [ ] **Step 1: Verify diff scope**

Expected: only the three test files plus this plan document differ from `main`.

- [ ] **Step 2: Run compatibility and relevant CI**

Expected: Linux/Windows compatibility no longer disagree on the V4 authored example; focused V10 regressions pass.

- [ ] **Step 3: Record remaining repository-wide failures separately**

Expected: existing Ruff format drift remains explicitly out of scope and is not hidden by weakening checks.
