# Documentation Refresh Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reorganize and refresh Trade RL documentation so maintained docs describe only current contracts, historical design material is clearly separated, and documentation tests fail closed on stale paths, missing runbooks, and identity drift.

**Architecture:** Keep the existing top-level documentation set as the maintained source of truth, but reduce responsibility overlap: repository `README.md` is the entry point, `docs/README.md` is the router, reference documents each own one normative area, `docs/operations/` contains only current executable runbooks, and historical design/verification material lives under `docs/implementation-plans/`. Strengthen `tests/test_current_documentation_contract.py` before the moves/rewrites so the refresh is driven by observable contracts rather than prose review alone.

**Tech Stack:** Markdown, Python 3.12, pytest, pathlib, regular expressions, Ruff, MyPy, Import Linter, GitHub Actions.

## Global Constraints

- Production status remains `NO-GO` unless independently changed by actual authorization/evidence.
- No profitability claim is introduced.
- Direct exchange order routing remains documented as unimplemented.
- Historical artifacts are never rewritten to imply new identity semantics.
- Maintained one-symbol and Universal shared-policy contracts remain distinct.
- Causal and sealed-evaluation boundaries are not weakened.
- Licensing provenance remains traceable.
- Historical documentation with unique provenance value is moved, not silently deleted.
- Do not change production logic solely to make documentation easier to write.
- Do not merge to `main` as part of this work.
- Required current targets must fail closed; do not use `if path.exists()` or `if path.is_file()` to make a missing required document/path silently pass.

---

### Task 1: Make the documentation contract test the new information architecture

**Files:**
- Modify: `tests/test_current_documentation_contract.py`
- Test: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: current repository paths, `tests.architecture.import_linter_config.configured_layers()`, source constants such as `OBSERVATION_SCHEMA` and `SERVING_BUNDLE_SCHEMA`.
- Produces: fail-closed documentation oracles used by all later tasks.

- [ ] **Step 1: Expand the maintained-document inventory and define current operations explicitly**

Add all maintained top-level reference documents to `MAINTAINED_DOCUMENTS`, including:

```python
ROOT / "docs" / "UNIVERSAL_TRAINING.md"
ROOT / "docs" / "REWARD_OBJECTIVE.md"
ROOT / "docs" / "EXECUTION_ROBUSTNESS.md"
ROOT / "docs" / "NAUTILUS_MIGRATION.md"
ROOT / "docs" / "LICENSING.md"
ROOT / "docs" / "LICENSING_PROVENANCE.md"
```

Define the only maintained operational runbooks:

```python
CURRENT_OPERATION_RUNBOOKS = (
    ROOT / "docs" / "operations" / "causal-scenario-c3-execution.md",
    ROOT / "docs" / "operations" / "docker-gpu-full-training.md",
)
```

Use `CURRENT_OPERATION_RUNBOOKS` inside `MAINTAINED_DOCUMENTS` rather than duplicating the path list.

- [ ] **Step 2: Write a failing operations/history separation test**

Add:

```python
def test_operations_directory_contains_only_current_runbooks() -> None:
    actual = tuple(sorted((ROOT / "docs" / "operations").glob("*.md"), key=str))
    assert actual == tuple(sorted(CURRENT_OPERATION_RUNBOOKS, key=str))
```

Run:

```bash
uv run pytest tests/test_current_documentation_contract.py::test_operations_directory_contains_only_current_runbooks -q
```

Expected: FAIL because historical Stage A design/plan and PR verification documents are still in `docs/operations/`.

- [ ] **Step 3: Stop using the repository README as a schema/architecture duplicate**

Change `test_current_schema_contracts_are_documented()` so `OBSERVATION_SCHEMA` and `SERVING_BUNDLE_SCHEMA` are required in `docs/ARCHITECTURE.md`, not in root `README.md`. Keep configuration-specific names in `CONFIGURATION.md` and one-symbol semantics in `SINGLE_SYMBOL.md`.

Add an entry-point ownership assertion:

```python
def test_readme_does_not_duplicate_reference_internals() -> None:
    readme = _text(ROOT / "README.md")
    for reference_only_term in (
        "Gated Cross-Timeframe Attention",
        "CanonicalStructuredPolicyLoader",
        "sb3_policy_identity_v4",
        "structured_policy_export_v2",
    ):
        assert reference_only_term not in readme
```

Run:

```bash
uv run pytest tests/test_current_documentation_contract.py::test_readme_does_not_duplicate_reference_internals -q
```

Expected: FAIL against the pre-refresh README.

- [ ] **Step 4: Add current-path and licensing oracles**

Add:

```python
def test_current_documentation_targets_exist() -> None:
    required = (
        ROOT / "docker" / "Dockerfile.training",
        ROOT / "docker" / "compose.training.yaml",
        ROOT / "docker" / "compose.universal-training.yaml",
        ROOT / "LICENSE",
        ROOT / "LICENSES" / "THIRD_PARTY_NOTICES.md",
    )
    missing = [path.relative_to(ROOT).as_posix() for path in required if not path.is_file()]
    assert missing == []


def test_readme_uses_current_third_party_notice_path() -> None:
    readme = _text(ROOT / "README.md")
    assert "LICENSES/THIRD_PARTY_NOTICES.md" in readme
    assert "`THIRD_PARTY_NOTICES.md`" not in readme
```

Run the second test and confirm it fails against the pre-refresh README.

- [ ] **Step 5: Add maintained-doc history and Universal resume-identity oracles**

Add:

```python
def test_maintained_reference_docs_do_not_depend_on_transient_pr_numbers() -> None:
    transient = re.compile(r"\bPR #\d+\b")
    offenders = [
        path.relative_to(ROOT).as_posix()
        for path in MAINTAINED_DOCUMENTS
        if transient.search(_text(path))
    ]
    assert offenders == []


def test_universal_training_documents_checkpoint_generator_identity() -> None:
    universal = _text(ROOT / "docs" / "UNIVERSAL_TRAINING.md")
    for phrase in (
        "generator_code_digest",
        "grid_digest",
        "causal_alpha_selection_checkpoint_metric_v2",
        "Fail closed",
    ):
        assert phrase.lower() in universal.lower()
```

Run each test separately. Expected: at least the transient PR test fails on current `ARCHITECTURE.md`, and the generator identity test fails until `UNIVERSAL_TRAINING.md` is refreshed.

- [ ] **Step 6: Keep link resolution fail closed**

Retain `test_internal_markdown_links_resolve()` over all repository Markdown. Do not add existence guards around required files. This test is expected to catch stale links after history moves until references are updated.

- [ ] **Step 7: Commit the RED contract tests**

```bash
git add tests/test_current_documentation_contract.py
git commit -m "test: define refreshed documentation contracts"
```

The commit may intentionally be red until Tasks 2-5 complete; record the failing test names in the PR/work log rather than weakening assertions.

---

### Task 2: Move historical operations material into the history area

**Files:**
- Move to `docs/implementation-plans/specs/`:
  - `docs/operations/stage-a-evaluation-dataset-manifest-design.md`
  - `docs/operations/stage-a-production-evaluator-design.md`
  - `docs/operations/stage-a-symbol-disjoint-training-design.md`
  - `docs/operations/stage-a-zero-shot-evaluation-design.md`
  - `docs/operations/stage-a-zero-shot-orchestrator-design.md`
- Move to `docs/implementation-plans/plans/`:
  - `docs/operations/audit-hardening-verification.md`
  - `docs/operations/evidence-stage-transaction-hardening-verification.md`
  - `docs/operations/stage-a-evaluation-dataset-manifest-implementation-plan.md`
  - `docs/operations/stage-a-postgres-sealed-test-ledger-implementation-plan.md`
  - `docs/operations/stage-a-production-evaluator-plan.md`
  - `docs/operations/stage-a-sb3-evaluation-environment-implementation-plan.md`
  - `docs/operations/stage-a-symbol-disjoint-training-plan.md`
  - `docs/operations/stage-a-zero-shot-evaluation-plan.md`
  - `docs/operations/stage-a-zero-shot-orchestrator-implementation-plan.md`
- Modify references in any Markdown files that link to moved documents.
- Test: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: the history classification established by the approved design.
- Produces: `docs/operations/` containing exactly two maintained runbooks and preserved historical documents under `implementation-plans/`.

- [ ] **Step 1: Copy each design file verbatim into `specs/`**

Keep the original filename and content. Do not rewrite old claims into present tense; these documents are provenance records.

- [ ] **Step 2: Copy each implementation/verification file verbatim into `plans/`**

Keep the original filename and content, including historical PR references. The transient-PR prohibition applies to maintained documents, not historical material.

- [ ] **Step 3: Update links that reference the old `docs/operations/...` locations**

Search repository Markdown for each moved filename. Change only link destinations; do not rewrite historical content to pretend it was authored at the new path.

- [ ] **Step 4: Delete the fourteen old `docs/operations/` copies**

After confirming destination files exist and have identical content, remove the originals.

- [ ] **Step 5: Run the directory and link contracts**

```bash
uv run pytest \
  tests/test_current_documentation_contract.py::test_operations_directory_contains_only_current_runbooks \
  tests/test_current_documentation_contract.py::test_internal_markdown_links_resolve -q
```

Expected: PASS after all link destinations are fixed.

- [ ] **Step 6: Commit the history relocation**

```bash
git add docs/operations docs/implementation-plans
git commit -m "docs: separate runbooks from implementation history"
```

---

### Task 3: Refresh the repository entry point and documentation router

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Test: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: current status from `RESEARCH_STATUS.md`, current quickstart from `START.md`, current repository layout.
- Produces: a bounded repository entry point and an authoritative documentation routing page.

- [ ] **Step 1: Rewrite root README around five questions**

Keep only material needed to answer:

1. what Trade RL is;
2. what is maintained today;
3. what is not production-ready;
4. the shortest execution path;
5. where to read next.

Preserve these explicit boundaries:

```text
Production status: NO-GO
Profitability claim: none
Direct exchange routing: not implemented
one maintained run = one instrument = one target-weight action
Universal shared-policy research is not simultaneous portfolio allocation
```

Keep the quickstart commands and repository map. Replace architecture/encoder/export details with links to their primary owner documents. Keep `LICENSES/THIRD_PARTY_NOTICES.md` as the actual third-party notice path.

- [ ] **Step 2: Refresh `docs/README.md` as the router**

List every maintained top-level document by user goal, state that `operations/` contains only current executable runbooks, and state that `implementation-plans/` contains history. Do not index the moved Stage A design/plan files as current operations.

- [ ] **Step 3: Run entry-point ownership and path tests**

```bash
uv run pytest \
  tests/test_current_documentation_contract.py::test_readme_is_a_bounded_entry_point \
  tests/test_current_documentation_contract.py::test_readme_does_not_duplicate_reference_internals \
  tests/test_current_documentation_contract.py::test_readme_uses_current_third_party_notice_path \
  tests/test_current_documentation_contract.py::test_internal_markdown_links_resolve -q
```

Expected: PASS.

- [ ] **Step 4: Commit the entry-point refresh**

```bash
git add README.md docs/README.md
git commit -m "docs: refresh repository entry points"
```

---

### Task 4: Refresh maintained architecture, research, and Universal training references

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Inspect and modify only if stale: `docs/CONFIGURATION.md`, `docs/SINGLE_SYMBOL.md`, `docs/REWARD_OBJECTIVE.md`, `docs/EXECUTION_ROBUSTNESS.md`, `docs/MULTITIMEFRAME_RESEARCH.md`, `docs/BINANCE.md`, `docs/NAUTILUS_MIGRATION.md`, `docs/LICENSING.md`, `docs/LICENSING_PROVENANCE.md`
- Test: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: current source contracts, current config/example names, current Import Linter layers, current causal-alpha v2 checkpoint implementation.
- Produces: maintained references that describe current behavior without development chronology.

- [ ] **Step 1: Remove transient implementation history from `ARCHITECTURE.md`**

Delete normative dependence on PR numbers such as the PR #191/#193 chronology. Preserve the actual current constrained-PPO, runner-classification, offline-signing, and serving boundaries as present-tense architecture statements.

- [ ] **Step 2: Recheck architecture ownership against code**

Keep the Import Linter order exactly equal to `configured_layers()`. Keep the current observation/action/reward, policy identity, structured export, serving bundle, artifact/catalog, execution, training/evaluation, and privileged runner boundaries. Do not move empirical status claims into architecture.

- [ ] **Step 3: Make `RESEARCH_STATUS.md` an explicit status ledger**

Preserve and sharpen distinctions among software implemented, CI verified, empirical evaluation incomplete, profitability not claimed, and production authorization `NO-GO`. Do not claim final empirical admission for Universal causal alpha or Stage A unless independent evidence exists.

- [ ] **Step 4: Document causal-alpha checkpoint resume identity in `UNIVERSAL_TRAINING.md`**

Add a subsection near candidate selection/checkpointing that states the v2 checkpoint row schema is `causal_alpha_selection_checkpoint_metric_v2` and binds both:

```text
grid_digest
generator_code_digest
```

Explain that resume recomputes/validates the expected generator-code digest and fails closed on a mismatch before accepting persisted replay metrics. Make clear this prevents replay reuse across a generator implementation change even when the candidate grid is unchanged.

- [ ] **Step 5: Audit remaining maintained references only for factual drift**

For each inspected file, verify current schema names, CLI/script/config paths, Docker locations, license locations, and maintained-vs-legacy wording. Edit only confirmed drift; avoid stylistic rewrites where the contract is already current.

- [ ] **Step 6: Run maintained-reference contract tests**

```bash
uv run pytest \
  tests/test_current_documentation_contract.py::test_current_schema_contracts_are_documented \
  tests/test_current_documentation_contract.py::test_architecture_layer_order_matches_import_linter \
  tests/test_current_documentation_contract.py::test_research_status_has_timeless_heading_and_explicit_stage_boundaries \
  tests/test_current_documentation_contract.py::test_maintained_reference_docs_do_not_depend_on_transient_pr_numbers \
  tests/test_current_documentation_contract.py::test_universal_training_documents_checkpoint_generator_identity \
  tests/test_current_documentation_contract.py::test_internal_markdown_links_resolve -q
```

Expected: PASS.

- [ ] **Step 7: Commit maintained-reference changes**

```bash
git add docs README.md
git commit -m "docs: align maintained references with current contracts"
```

---

### Task 5: Falsification review the refreshed documentation and tests

**Files:**
- Modify only if a defect is found: documentation and documentation tests touched above.
- Test: `tests/test_current_documentation_contract.py`
- Test: `tests/architecture/test_repository_layout.py`
- Test: `tests/test_license_contract.py`

**Interfaces:**
- Consumes: completed documentation layout and contract tests.
- Produces: evidence that the tests detect stale/missing targets rather than merely executing.

- [ ] **Step 1: Search maintained docs for stale path patterns**

```bash
rg -n 'docker/docker/|`THIRD_PARTY_NOTICES\.md`|docs/operations/.*-(design|plan|implementation-plan)\.md' \
  README.md START.md docs/*.md docs/operations/*.md
```

Expected: no stale current-path matches. Historical `implementation-plans/` is excluded from this current-doc scan.

- [ ] **Step 2: Search maintained docs for transient PR/commit authority**

```bash
rg -n '\bPR #[0-9]+\b|[0-9a-f]{40}' README.md START.md docs/*.md docs/operations/*.md
```

Expected: no transient PR/commit identifier used as current normative authority. If a hash is a domain/content digest example rather than a Git commit, inspect it rather than deleting mechanically.

- [ ] **Step 3: Search documentation tests for silent required-target skips**

```bash
rg -n 'if .*\.(exists|is_file)\(' tests/test_current_documentation_contract.py tests/architecture/test_repository_layout.py
```

Expected: no guard that silently ignores a required current file. Filtering optional recursive inventories is acceptable only where absence is not itself part of the contract; inspect every match.

- [ ] **Step 4: Re-run targeted documentation/layout/license tests**

```bash
uv run pytest -q \
  tests/test_current_documentation_contract.py \
  tests/architecture/test_repository_layout.py \
  tests/test_license_contract.py
```

Expected: PASS.

- [ ] **Step 5: Self-review the complete diff**

Verify that no security, provenance, migration, sealed-evaluation, causal, licensing, or `NO-GO` boundary was removed. Verify that every moved historical file still exists at its destination and that `docs/operations/` contains only the two current runbooks.

- [ ] **Step 6: Commit any falsification-review corrections**

If defects were found, commit only the corrective diff:

```bash
git add README.md START.md docs tests
git commit -m "docs: fix refresh review findings"
```

If no defect was found, do not create an empty commit.

---

### Task 6: Run repository-wide quality gates on one final HEAD

**Files:**
- No intended source changes; fix only evidence-backed failures caused by this work.

**Interfaces:**
- Consumes: final documentation/test tree.
- Produces: same-head verification evidence for PR #401.

- [ ] **Step 1: Run formatting and static checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
```

Expected: all PASS.

- [ ] **Step 2: Run the full Python suite with the repository's maintained coverage command**

Use the same pytest/coverage command configured by the Core CI workflow. Expected: all tests PASS and all configured total/critical coverage gates remain satisfied.

- [ ] **Step 3: Run frontend checks only if frontend files or referenced frontend commands changed**

If `frontend/README.md`, frontend config, or a frontend command in maintained docs changed, run:

```bash
npm test --prefix frontend -- --run
npm run typecheck --prefix frontend
npm run build --prefix frontend
npm run check:layout --prefix frontend
```

Otherwise rely on the unchanged frontend path but still require the final Core CI workflow to pass.

- [ ] **Step 4: Verify Git diff and repository state**

```bash
git diff --check
git status --short
git log -1 --oneline
```

Expected: no whitespace errors, no accidental generated/local artifacts, and the recorded HEAD equals the head used for CI.

- [ ] **Step 5: Require final-head GitHub workflows**

On the exact final HEAD, require successful relevant workflows including Core CI, Nautilus Capability, and PostgreSQL Catalog. Inspect failed job logs rather than rerunning blindly. Do not reuse success from an earlier commit as evidence for the final head.

- [ ] **Step 6: Perform final independent/falsification review**

Reconstruct the acceptance criteria from `docs/implementation-plans/specs/2026-08-14-documentation-refresh-design.md` and review the final diff without assuming implementation decisions are correct. Explicitly verify:

- current-vs-history placement;
- link/path resolution;
- root README responsibility;
- no transient PR authority in maintained docs;
- generator-code checkpoint identity documentation;
- research/production status distinctions;
- retained security/provenance/licensing/migration boundaries;
- fail-closed documentation tests.

- [ ] **Step 7: Update PR #401 metadata and leave `main` untouched**

Record the exact final HEAD, test counts, static-check results, workflow status, documentation restructuring, remaining warnings/risks, and anything not empirically verified. Mark the PR Ready only after all quality gates pass. Do not merge to `main` without explicit user permission.
