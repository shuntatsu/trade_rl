# Documentation Readability Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the accumulated documentation history with a concise, current, internally consistent documentation set centered on one Japanese README.

**Architecture:** Keep user-facing documents small and responsibility-specific. Remove completed plans and audit reports from the working tree because Git history preserves them, then enforce the maintained document set with a repository test and link checker.

**Tech Stack:** Markdown, Python 3.12, pytest, GitHub Actions, Import Linter.

## Global Constraints

- `README.md` is the sole repository-level overview.
- Production status remains `NO-GO`; documentation must not claim profitability or direct exchange execution.
- The maintained training schema is `training_run_config_v2`.
- The maintained observation encoders are `flat_mlp`, `asset_set`, and `hierarchical_sequence_v2`.
- Historical plan and audit content is removed from the working tree, not rewritten as current behavior.
- Internal links must resolve from their containing document.

---

### Task 1: Replace the documentation contract test

**Files:**
- Modify: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: current schema constants, `.importlinter`, maintained Markdown paths.
- Produces: one test module that rejects stale documentation layout and broken links.

- [ ] Remove `README.ja.md` and dated verification files from `MAINTAINED_DOCUMENTS`.
- [ ] Add `docs/README.md` and `docs/CONFIGURATION.md`.
- [ ] Assert the old historical directories and `docs/MULTITIMEFRAME_RESEARCH.md` are absent.
- [ ] Assert `training_run_config_v2`, the three encoder values, separated attention settings, architecture identity, TensorBoard diagnostics, and structured export are documented.
- [ ] Scan all maintained Markdown files for unresolved relative links.
- [ ] Add readability bounds for README headings and line count.
- [ ] Run `uv run pytest tests/test_current_documentation_contract.py -q` and confirm RED before document changes.

### Task 2: Rewrite repository entry points

**Files:**
- Rewrite: `README.md`
- Rewrite: `START.md`
- Delete: `README.ja.md`

**Interfaces:**
- Consumes: current CLI commands and example configuration paths.
- Produces: a concise project overview and executable first-run guide.

- [ ] Rewrite README around status, quickstart, capability map, architecture summary, evidence boundaries, and document map.
- [ ] Keep detailed subsystem explanations out of README and link to focused docs.
- [ ] Rewrite START around installation, dataset creation, training, artifact inspection, Studio, configuration v2, resume, and common failures.
- [ ] Delete README.ja.md.

### Task 3: Create the maintained documentation map and configuration reference

**Files:**
- Create: `docs/README.md`
- Create: `docs/CONFIGURATION.md`

**Interfaces:**
- Consumes: `ResidualTrainingConfig`, `TrainingRunConfig`, structured policy/export contracts.
- Produces: a stable navigation page and current v2 configuration reference.

- [ ] Document the maintained document set and where historical material can be found.
- [ ] Document `training_run_config_v2` and explicit legacy rejection.
- [ ] Document encoder selection and inactive-setting fail-closed behavior.
- [ ] Document timeframe and asset attention settings separately.
- [ ] Document BC/PPO-family identity sharing, checkpoints, diagnostics, export, and serving.

### Task 4: Rewrite current architecture and status docs

**Files:**
- Rewrite: `docs/ARCHITECTURE.md`
- Rewrite: `docs/RESEARCH_STATUS.md`
- Update: `docs/BINANCE.md`
- Update: `docs/operations/docker-gpu-full-training.md`
- Update: `studio/README.md`

**Interfaces:**
- Consumes: current module boundaries, hierarchical sequence implementation, release gates, and runtime contracts.
- Produces: focused architecture, research, data, GPU, and Studio documents.

- [ ] Keep the exact Import Linter layer order in architecture documentation.
- [ ] Replace the old sequence description with TCN → cross-timeframe → cross-asset attention.
- [ ] Explain architecture digest and fail-closed checkpoint/serving behavior.
- [ ] Separate implemented capabilities from empirical evidence and operational authorization.
- [ ] Cross-link focused operational documents without duplicating README.

### Task 5: Remove historical documentation clutter

**Files:**
- Delete: `.superpowers/sdd/task-8-report.md`
- Delete: `docs/MULTITIMEFRAME_RESEARCH.md`
- Delete: dated files under `docs/audits`, `docs/reviews`, `docs/plans`, and `docs/verification`
- Delete: historical files under `docs/superpowers/plans`, `docs/superpowers/specs`, and `docs/superpowers/status`, preserving only this design and implementation plan
- Delete: `.github/workflows/docs-inventory.yml`

**Interfaces:**
- Consumes: the documentation inventory artifact.
- Produces: a small working-tree documentation surface with history retained by Git.

- [ ] Delete the identified historical files in one deterministic cleanup operation.
- [ ] Verify no maintained document links to a deleted path.

### Task 6: Verify and publish

**Files:**
- Test: `tests/test_current_documentation_contract.py`
- Validate: all changed Markdown and repository CI.

**Interfaces:**
- Consumes: completed documentation tree.
- Produces: merge-ready PR with passing documentation and repository checks.

- [ ] Run the documentation contract test.
- [ ] Run Ruff, formatting check, MyPy, Import Linter, and full pytest.
- [ ] Run the Markdown inventory/link workflow or equivalent final check.
- [ ] Confirm no `training_run_config_v1`, `sequence_encoder`, or `asset_set_encoder` appears as an active maintained setting.
- [ ] Remove temporary workflow files.
- [ ] Update PR description with the final document set and deletion count.
