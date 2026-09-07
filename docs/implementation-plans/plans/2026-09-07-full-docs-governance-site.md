# Full Documentation Governance and Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a typed documentation governance system shared by humans, agents, CI, and a strict Zensical/GitHub Pages site.

**Architecture:** Preserve current maintained documentation content while physically consolidating historical material under `docs/history/`. Add inherited YAML metadata, deterministic validation/manifest/site generation tooling, path-based Agent context routing, and exact-head CI/Page publishing. Generated outputs stay uncommitted and raw history stays outside strict site input.

**Tech Stack:** Python 3.12, PyYAML 6.0.3, pytest, Zensical 0.0.59, GitHub Actions, GitHub Pages.

**Spec:** `docs/implementation-plans/specs/2026-09-07-full-docs-governance-site-design.md`

## Global Constraints

- Do not change RL/runtime/research semantics to make docs easier to organize.
- Preserve `NO-GO`, causal/sealed-evaluation, licensing, provenance, and direct-routing boundaries.
- Historical content must remain historical and non-authoritative.
- Generated `.docs-build/` and `site/` outputs are not committed.
- External Actions are immutable-SHA pinned.
- Zensical is pinned to `0.0.59` for reproducible site validation.

---

### Task 1: Documentation governance RED contracts

**Files:**
- Create: `tests/docs/test_docs_governance.py`
- Create: `tests/docs/test_docs_falsification.py`

**Interfaces:**
- Consumes: repository current docs/source tree.
- Produces: expected API `build_manifest`, `validate_repository`, `context_for_path`, `stage_site` from `scripts.docs._core`.

- [x] Write integration RED contract for lifecycle separation, reward routing, manifest generation, and site staging.
- [ ] Add falsification fixtures for duplicate canonical ownership, missing metadata, historical authority escalation, unresolved related-code patterns, and invalid section/type boundaries.
- [ ] Run targeted tests on the branch and preserve the expected missing-tooling failure as RED evidence.

### Task 2: Metadata and validation engine

**Files:**
- Create: `scripts/docs/__init__.py`
- Create: `scripts/docs/_core.py`
- Create: `scripts/docs/validate.py`
- Create: `docs/.meta.yml`
- Create: `docs/operations/.meta.yml`
- Create: `docs/performance/.meta.yml`
- Create: `docs/history/.meta.yml`

**Interfaces:**
- Consumes: inherited `.meta.yml`, optional Markdown front matter, repository paths.
- Produces: deterministic document records and `validate_repository(root) -> tuple[str, ...]`.

- [ ] Implement deterministic YAML loading/front-matter merge.
- [ ] Validate required enums/fields and current/history lifecycle boundary.
- [ ] Validate unique canonical `source_of_truth_for` ownership.
- [ ] Validate `related_code` glob resolution and section/type contracts.
- [ ] Validate current relative links without treating raw history as strict current content.
- [ ] Run RED→GREEN targeted tests.

### Task 3: Agent routing and repository instructions

**Files:**
- Create: `AGENTS.md`
- Modify: `docs/AGENTS.md`
- Create: `scripts/docs/context.py`
- Test: `tests/docs/test_docs_governance.py`

**Interfaces:**
- Consumes: manifest current records and `related_code` globs.
- Produces: deterministic canonical-first context records and CLI output.

- [ ] Route exact source paths to canonical/supporting docs ordered by authority and site order.
- [ ] Add topic lookup fallback.
- [ ] Document source-vs-doc inconsistency handling and history non-authority rules.
- [ ] Verify important mappings including rewards, configuration, Universal workflows, execution, evaluation, serving, operations, and licensing.

### Task 4: Consolidate historical documentation boundary

**Files:**
- Move: `docs/implementation-plans/specs/**` → `docs/history/specs/**`
- Move: `docs/implementation-plans/plans/**` → `docs/history/plans/**`
- Move: `docs/implementation-plans/evidence/**` → `docs/history/evidence/**`
- Move: root legacy `docs/implementation-plans/*.md` → appropriate `docs/history/{plans,evidence}/`
- Move: `docs/implementation/**` → `docs/history/legacy/**`
- Move: `docs/architecture/**` → `docs/history/decisions/**` or `docs/history/legacy-design/**`
- Create: `docs/history/README.md`

**Interfaces:**
- Consumes: exact existing blobs/content.
- Produces: one physical history root with lifecycle normalization.

- [ ] Preserve historical file content unless a path-only compatibility note is required.
- [ ] Remove obsolete parallel historical roots from the final tree.
- [ ] Ensure current docs/tests never consume historical content as current authority.
- [ ] Verify no current operation runbook is moved into history.

### Task 5: Automatic manifest, indexes, and strict site staging

**Files:**
- Create: `scripts/docs/generate.py`
- Extend: `scripts/docs/_core.py`
- Create: `mkdocs.yml` as a human-readable bootstrap pointing to generated build usage, or generate `.docs-build/mkdocs.generated.yml` as the build authority.
- Modify: `docs/README.md`

**Interfaces:**
- Consumes: validated document records.
- Produces: `.docs-build/manifest.json`, `.docs-build/site-src/**`, `.docs-build/mkdocs.generated.yml`.

- [ ] Generate deterministic manifest sorted by lifecycle/authority/path.
- [ ] Generate homepage and section indexes from metadata.
- [ ] Generate topic and authority indexes.
- [ ] Generate history catalog without copying raw history into strict site source.
- [ ] Rewrite current source-relative links to staged-site-relative links.
- [ ] Generate deterministic Zensical navigation configuration.
- [ ] Verify repeated generation produces byte-identical outputs.

### Task 6: Repository entry points and compatibility

**Files:**
- Modify: `README.md`
- Modify: `START.md` only if needed for accurate entry-point links.
- Modify: current maintained docs only where paths/ownership references changed.
- Modify: `tests/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: new governance lifecycle and generated index model.
- Produces: stable human/Agent entry points without duplicate normative text.

- [ ] Point root README and Agent instructions at the documentation portal/governance commands.
- [ ] Update documentation contract tests to assert the new boundary rather than fixed obsolete paths.
- [ ] Keep existing schema/version/NO-GO assertions fail-closed.
- [ ] Do not weaken assertions merely because files moved.

### Task 7: CI site gate and GitHub Pages publication

**Files:**
- Modify: `.github/workflows/ci.yml`
- Create: `.github/workflows/docs-pages.yml`
- Modify: `.github/check_workflow_security.py` only if Pages write-permission policy requires a narrowly-scoped documented exception.
- Test: workflow-security/documentation tests.

**Interfaces:**
- Consumes: exact checked-out SHA, generated site configuration/source.
- Produces: PR docs gate and `main` Pages artifact/deployment.

- [ ] Add PR quality steps: docs validation, generation, strict Zensical 0.0.59 build.
- [ ] Use immutable official Action SHAs for checkout/configure-pages/upload-pages-artifact/deploy-pages.
- [ ] Keep build job read-only; give deploy job only `pages: write` and `id-token: write` required for Pages.
- [ ] Restrict deployment to `main` push and GitHub Pages environment.
- [ ] Update workflow security validator to distinguish standard GitHub Pages deployment from privileged custom/self-hosted workflows if necessary.
- [ ] Verify exact-head workflow security.

### Task 8: Falsification, regression, and completion gate

**Files:**
- Modify: tests/tooling only for defects discovered by review.

**Interfaces:**
- Consumes: final branch diff and exact HEAD.
- Produces: objective completion evidence.

- [ ] Run targeted docs tests and falsification tests.
- [ ] Run docs validator/generator twice and compare outputs.
- [ ] Run strict Zensical build.
- [ ] Run workflow security, Ruff, format, mypy, import-linter, full pytest, frontend checks if affected, and required build checks.
- [ ] Review final diff for runtime changes, broken authority boundaries, duplicate normative text, stale old paths, and generated files.
- [ ] Independently reconstruct acceptance criteria and try to falsify each one.
- [ ] Confirm final PR HEAD and same-head CI/required workflow results.
- [ ] Report separately whether GitHub Pages repository setting/environment permits deployment; do not treat site build success as proof of successful public publication.
