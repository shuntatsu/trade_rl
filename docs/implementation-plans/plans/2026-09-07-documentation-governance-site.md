# Documentation Governance and Site Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the current documentation layout with a machine-validated, agent-routable, Zensical-published documentation system without changing Trade RL runtime behavior.

**Architecture:** YAML metadata and inherited `.meta.yml` files define lifecycle, type, authority, topics, canonical ownership, and code routing. Small Python tools under `scripts/docs/` validate and generate deterministic manifests/context; Zensical 0.0.59 consumes the same Markdown for the site. Current docs move into intent-based directories; old architecture/implementation material moves unchanged into `docs/history/`.

**Tech Stack:** Python 3.12, PyYAML 6.0.3 already in `dev`, pytest, Zensical 0.0.59 via pinned `uvx`, GitHub Actions/Pages.

**Spec:** `docs/implementation-plans/specs/2026-09-07-documentation-governance-site-design.md`

## Global Constraints

- No RL/runtime/research semantics changes.
- Zensical remains documentation tooling and is not added to application runtime dependencies.
- Historical files are preserved as provenance and are not current authority.
- Existing production `NO-GO`, profitability, causal, sealed-evaluation, licensing, and direct-exchange-routing boundaries remain explicit.
- All GitHub Actions references are immutable 40-character commit SHAs.
- No completion claim before exact-head verification.

---

### Task 1: RED governance contract

**Files:**
- Modify: `tests/test_current_documentation_contract.py`
- Create: `tests/docs/test_documentation_governance.py`

**Interfaces:**
- Consumes: current repository tree.
- Produces: failing contracts for `AGENTS.md`, target directories, metadata tooling, canonical uniqueness, history separation, context routing, Zensical config/workflow.

- [ ] Add tests that require `AGENTS.md`, `mkdocs.yml`, `scripts/docs/`, and the target directory layout.
- [ ] Add tests that reject the old `docs/implementation/`, `docs/implementation-plans/`, and historical `docs/architecture/` roots.
- [ ] Add tests that require current docs at their new paths and require `START.md` to be a compatibility entry point.
- [ ] Add metadata/routing tests against wished-for APIs `scripts.docs.metadata`, `scripts.docs.manifest`, and `scripts.docs.context`.
- [ ] Commit RED tests before implementation and run PR CI to prove failure is caused by the missing governance system.

### Task 2: Metadata model and validation

**Files:**
- Create: `scripts/docs/__init__.py`
- Create: `scripts/docs/model.py`
- Create: `scripts/docs/metadata.py`
- Create: `scripts/docs/validate.py`
- Test: `tests/docs/test_documentation_governance.py`

**Interfaces:**
- `load_effective_metadata(path: Path, docs_root: Path) -> DocumentMetadata`
- `discover_documents(docs_root: Path) -> tuple[DocumentMetadata, ...]`
- `validate_documents(root: Path) -> tuple[str, ...]`

- [ ] Implement front-matter parsing and ancestor `.meta.yml` merge.
- [ ] Validate required enums and non-empty topics for current pages.
- [ ] Enforce folder/type/lifecycle/authority compatibility.
- [ ] Enforce unique current canonical `source_of_truth_for` keys.
- [ ] Enforce current-link resolution and current `related_code` root existence while excluding historical freshness checks.
- [ ] Re-run focused tests until GREEN.

### Task 3: Manifest, generated indexes, and agent context

**Files:**
- Create: `scripts/docs/manifest.py`
- Create: `scripts/docs/generate.py`
- Create: `scripts/docs/context.py`
- Test: `tests/docs/test_documentation_governance.py`

**Interfaces:**
- `build_manifest(root: Path) -> dict[str, object]`
- `write_build_outputs(root: Path, output_root: Path) -> None`
- `documents_for_path(root: Path, changed_path: str) -> tuple[DocumentMetadata, ...]`
- CLI: `python scripts/docs/context.py --path PATH` and `--topic TOPIC`

- [ ] Generate deterministically sorted JSON with no timestamps or environment-dependent values.
- [ ] Route source paths to canonical before supporting docs and exclude history unless `--include-history` is supplied.
- [ ] Generate topic/authority indexes into `.docs-build/` only.
- [ ] Add falsification tests for prefix ambiguity, duplicate ownership, empty topic, and history leakage.

### Task 4: Physical documentation migration and metadata

**Files:**
- Create/Move current docs under `docs/getting-started`, `docs/guides`, `docs/reference`, `docs/research`, `docs/operations`, `docs/performance`, `docs/legal`.
- Move historical trees under `docs/history`.
- Create `.meta.yml` and section `index.md` files.
- Modify: `README.md`, `START.md`, `frontend/README.md` only where navigation links require it.

**Interfaces:**
- Current paths exactly match the design spec.
- Historical blob contents remain unchanged except the newly approved design/plan are moved with the specs/plans trees.

- [ ] Move canonical files with preserved semantic content and update their relative current links.
- [ ] Move old `docs/architecture/` to `docs/history/architecture/`.
- [ ] Move specs/plans/evidence and legacy implementation trees into history.
- [ ] Add inherited metadata for every section and page-level canonical ownership/routing metadata for current authority docs.
- [ ] Rewrite `docs/index.md` as the human/agent portal.
- [ ] Replace root `START.md` with a small compatibility entry point to `docs/getting-started/quickstart.md`.
- [ ] Update repository README links to new canonical paths.
- [ ] Run current-link/orphan/authority tests.

### Task 5: Agent bootstrap contract

**Files:**
- Create: `AGENTS.md`
- Test: `tests/docs/test_documentation_governance.py`

**Interfaces:**
- Agent bootstrap points to `docs/index.md` and `scripts/docs/context.py`.
- Priority: executable contract > current canonical docs > current guides/runbooks > evidence > history.

- [ ] Document required reading, update triggers, folder routing, authority rules, and completion checks.
- [ ] Explicitly forbid history as current runtime authority.
- [ ] Map architecture/config/reward/execution/universal/research/license changes to canonical docs.
- [ ] Verify contract test rejects missing bootstrap requirements.

### Task 6: Zensical site

**Files:**
- Create: `mkdocs.yml`
- Create: `docs/stylesheets/extra.css`
- Modify section index/front matter as required for site rendering.

**Interfaces:**
- Build command: `uvx --from zensical==0.0.59 zensical build --strict`.
- Site source: `docs/`; output: `site/`.

- [ ] Configure Material-compatible modern theme, Japanese language, repository links, search, meta, tags, navigation indexes, strict link validation, light/dark palettes, and readable history warning styling.
- [ ] Keep configuration within features supported by Zensical 0.0.59.
- [ ] Ensure generated/build-only paths are ignored and not authority.
- [ ] Verify strict build in CI.

### Task 7: CI and GitHub Pages

**Files:**
- Create: `.github/workflows/docs.yml`
- Modify: `.github/workflows/ci.yml`
- Modify: `.gitignore`
- Test: existing `.github/check_workflow_security.py` plus docs contracts.

**Interfaces:**
- Pull requests: validate metadata/generator/context + strict Zensical build, no deployment.
- Main: same build then deploy via GitHub Pages environment.

- [ ] Pin `actions/checkout`, `astral-sh/setup-uv`, `actions/configure-pages`, `actions/upload-pages-artifact`, and `actions/deploy-pages` by SHA.
- [ ] Give build `contents: read`; deploy only `contents: read`, `pages: write`, `id-token: write`.
- [ ] Add docs tooling/tests to core CI so docs correctness does not depend solely on Pages workflow.
- [ ] Run workflow-security validation.

### Task 8: Full verification and falsification review

**Files:**
- Review complete diff only; no planned feature additions.

**Interfaces:**
- Quality gate from the design spec.

- [ ] Run targeted docs tests.
- [ ] Run Ruff, format, Mypy, workflow security, and import architecture checks.
- [ ] Run full pytest/coverage and relevant compatibility checks through GitHub CI on exact HEAD.
- [ ] Run strict Zensical build on exact HEAD.
- [ ] Compare branch with `main` and prove no `trade_rl/` runtime file changed.
- [ ] Search diff for old current paths, duplicate canonical ownership, mutable actions, generated site output, history treated as authority, and weakened safety/status claims.
- [ ] Inspect same-head workflow runs and only then make the PR ready for review.
