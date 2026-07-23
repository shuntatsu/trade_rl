# Paper Reconciliation CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:test-driven-development and superpowers:verification-before-completion task by task.

**Goal:** Produce immutable paper reconciliation evidence from a strict normalized request and require an explicit reconciliation path in the public Serving package CLI.

**Architecture:** `trade_rl.cli.extended` owns request parsing and machine-readable command results. `trade_rl.evaluation.paper_reconciliation` remains the sole authority for validation, derived pass state, hashing, and immutable artifact writing. The Serving package Python API remains backward compatible while the CLI requires an explicit path.

**Tech Stack:** Python 3.12, argparse, JSON, pytest, Ruff, MyPy, GitHub Actions.

## Constraints

- Production remains `NO-GO`.
- Direct exchange routing remains `NOT_IMPLEMENTED`.
- The reconciliation command accepts normalized measurements only; no venue adapter or network access.
- No signing key is accepted by the reconciliation command.
- `passed` is derived, not accepted from the request.
- A valid failed report is written as evidence; promotion still rejects it.

### Task 1: Define the reconciliation-create CLI contract

**Files:**
- Modify: `tests/cli/test_offline_approval_cli.py`
- Modify: `trade_rl/cli/extended.py`

- [ ] Add a failing test for `trade-rl reconciliation create` using a complete `paper_reconciliation_request_v1` request.
- [ ] Assert immutable artifact loading, derived `passed`, exact result JSON, and `production_status: NO-GO`.
- [ ] Add a failing test proving that an extra caller-supplied `passed` field is rejected and no artifact is written.
- [ ] Confirm RED because the command is unsupported.
- [ ] Add the parser, strict request-field closure, type helpers, command handler, and dispatcher entry.
- [ ] Confirm focused GREEN.

### Task 2: Require the explicit artifact path in Serving CLI

**Files:**
- Modify: `tests/cli/test_artifact_commands.py` or the nearest existing Serving package CLI tests
- Modify: `trade_rl/cli/extended.py`

- [ ] Add a failing parser/forwarding test for required `--paper-reconciliation`.
- [ ] Assert the exact `Path` reaches `package_selected_training_run(paper_reconciliation_path=...)`.
- [ ] Add the required parser option and forwarding argument.
- [ ] Retain the Python package API sibling fallback for direct callers.
- [ ] Confirm focused GREEN.

### Task 3: Document, verify, and publish

**Files:**
- Modify: `README.md`
- Modify: `README.ja.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md` only if the operational status wording changes materially.

- [ ] Add the reconciliation-create command before confirmation creation in the documented offline evidence flow.
- [ ] State that input measurements are externally normalized and are not independently proven by the CLI.
- [ ] Run Ruff, formatter, MyPy, Import Linter, complete pytest with branch coverage, Studio checks, Ubuntu/Windows compatibility, training image, and PostgreSQL Catalog at the exact PR head.
- [ ] Record RED/GREEN SHAs, run IDs, test count, coverage, and artifact digests.
- [ ] Mark ready and squash merge only after every required check succeeds.
