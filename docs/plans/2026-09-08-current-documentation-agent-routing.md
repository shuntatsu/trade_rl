# Current Documentation and Agent Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: use `superpowers:test-driven-development` for the documentation contract and `superpowers:verification-before-completion` before any completion claim.

**Goal:** Replace the completed redesign/planning document pile with a small current-only documentation system that gives humans and agents one clear entry point, explicit ownership/update rules, and no working-tree history archive.

**Architecture:** Keep root `README.md` as the short project landing page and add root `AGENTS.md` as a thin agent bootstrap. Make `docs/README.md` the documentation portal, `docs/AGENTS.md` the detailed agent/update contract, `docs/architecture/{lean-core,package-boundaries}.md` the current architecture authorities, and `docs/research/current-status.md` the current research-state authority. Once durable current content is preserved, remove the completed redesign document and all completed specs/plans; Git history remains the archive.

**Tech Stack:** Markdown, Python 3.12 contract tests, pytest, Ruff, Mypy, GitHub Actions.

**Spec:** `docs/specs/2026-09-08-lean-package-boundaries-design.md` section 11, implemented after verified Phase 4A exact head `21efa93ddc6073fbd31c7000461801a699797254`.

## Global Constraints

- No production Python behavior change.
- No evaluation/strategy/data/risk/simulation contract change.
- No Controlled Experiment Loop implementation.
- Do not create `docs/history/` or `docs/archive/`.
- Git history is the archive for completed designs/plans.
- `LICENSE`, `LICENSES/LICENSING.md`, `LICENSES/PROVENANCE.md`, SPDX license texts, and third-party notices are permanent compliance material and must remain untouched.
- Current docs must not claim profitability or Production authorization that does not exist.
- Current docs must use the maintained candidate CLI path `python -m trade_rl.evaluation.runs.candidate`.
- Current architecture docs must describe the final Phase 1–4A package tree, not transitional old private paths.
- Root `AGENTS.md` is a short bootstrap only; detailed docs governance lives in `docs/AGENTS.md` to avoid duplicated policy.
- Active future specs/plans may be created under `docs/specs/` and `docs/plans/`, but completed artifacts are removed after durable content is reflected into current authority docs.

---

## Quality Contract

### Objective

Make it possible for a new maintainer or agent to answer three questions from the filesystem alone:

1. Where do I start reading?
2. Which document owns this kind of information?
3. What must I update when I change code, architecture, research status, CLI usage, or documentation layout?

### Non-goals

- Preserve completed implementation chronology in the working tree.
- Keep compatibility copies of the 2026-09-08 redesign/spec/plan documents.
- Create a documentation generator, website, or UI.
- Re-litigate the already implemented package-boundary design.
- Change research results or claim a winner.

### Acceptance Criteria

1. Root contains `AGENTS.md` and `README.md`.
2. Final `docs/` contains exactly:
   - `README.md`
   - `AGENTS.md`
   - `architecture/lean-core.md`
   - `architecture/package-boundaries.md`
   - `research/current-status.md`
3. `docs/trade_rl_lean_redesign_20260908.md` is absent after its current normative content is decomposed.
4. All completed `docs/plans/*.md` and `docs/specs/*.md` are absent; no history/archive replacement is created.
5. Root README routes to `docs/README.md` and `docs/research/current-status.md`, and uses the current candidate CLI.
6. Root `AGENTS.md` routes agents to `docs/README.md` and `docs/AGENTS.md` without duplicating the detailed policy.
7. `docs/README.md` provides one route for architecture, package/dependency rules, research status, agent rules, and permanent licensing/provenance material.
8. `docs/AGENTS.md` defines source-of-truth handling, documentation folder semantics, update triggers, naming/retention rules, and pre-finish verification.
9. Current architecture docs describe the final lean package owners and dependency direction established through Phases 1–4A.
10. Research status preserves the current observable state:
    - M1 lean core complete;
    - M2 comparison infrastructure complete;
    - real-data development comparison not run yet;
    - M3 not started;
    - profitability claim none;
    - Production/live routing not authorized.
11. Current docs retain the 5 candidates + 3 controls and the universal fit/evaluation-scope contract.
12. All relative Markdown links in current repository Markdown resolve.
13. Current docs contain no transient PR number or 40-character commit SHA as normative architecture/status text.
14. Permanent licensing/provenance files remain byte-identical to the exact Phase 4A base.
15. Final exact HEAD passes documentation contract tests plus normal repository CI.

### Invariants

- Code/tests remain executable evidence; docs do not override an observed source/test mismatch silently.
- `MarketExecutor + BookState` remain documented as the single execution/accounting authority.
- Fit symbols and evaluation symbols remain documented as separate scopes.
- The same frozen universal strategy is documented as independently replayed per evaluation symbol.
- Aggregate P&L is not documented as the sole source of truth.
- A correct `no winner` result remains an allowed research outcome.
- No current doc turns historical implementation detail into a supported private API.

### Failure Modes

- Deleting the redesign doc before preserving its current research/economic contracts.
- Duplicating the same normative architecture across several docs and allowing drift.
- Leaving an orphan current doc that is not reachable from `docs/README.md`.
- Keeping completed plans/specs beside current docs and causing agents to treat them as current authority.
- A root agent file and docs agent file diverging because both contain full policy copies.
- Markdown links point to removed redesign/spec/plan files.
- README still treats the removed redesign file as the canonical design.
- Current docs use a retired CLI/private module path.
- License/provenance material is deleted as “old docs.”
- Current status is accidentally upgraded to profitability/Production readiness.

### Test Oracle

Observe:

- exact final docs filesystem shape;
- root/portal/agent routing links;
- relative-link resolution;
- required current-status statements;
- required package/dependency owners;
- current candidate CLI;
- absence of completed/history docs;
- absence of transient PR/SHA identifiers in current authority docs;
- byte identity of compliance files;
- full repository CI.

### Required Test Layers

- Documentation architecture contract test under `tests/architecture/`.
- Relative-link contract test across repository Markdown.
- Static analysis/format for the new Python test.
- Full repository regression suite, Mypy, Ruff, package identity.
- Exact-head normal PR CI.

### Quality Gate

Do not call Phase 4B complete until every Acceptance Criterion is checked on the final helper-free HEAD, current content preservation is self-reviewed against the old redesign/spec, all completed artifacts are removed, all links resolve, compliance files are unchanged, and normal CI is Green on that exact HEAD.

---

## Final Documentation Tree

```text
README.md
AGENTS.md
LICENSE
LICENSES/

docs/
├── README.md
├── AGENTS.md
├── architecture/
│   ├── lean-core.md
│   └── package-boundaries.md
└── research/
    └── current-status.md
```

`docs/specs/` and `docs/plans/` are not retained as empty tracked directories. Future active work may recreate them as needed. `docs/history/` and `docs/archive/` are forbidden.

## Authority and Ownership Model

### `README.md`

Short project landing page only:

- one-sentence purpose;
- concise current status;
- candidate set;
- development-run command;
- links to docs portal/current research status/licensing.

### Root `AGENTS.md`

Thin bootstrap:

- read `docs/README.md` first;
- read `docs/AGENTS.md` before modifying docs/architecture/research contracts;
- inspect source/tests rather than treating old plans/history as current behavior;
- detailed rules are not duplicated here.

### `docs/README.md`

Human + agent portal:

- “understand runtime architecture” → `architecture/lean-core.md`;
- “package ownership/dependencies/public API” → `architecture/package-boundaries.md`;
- “what is actually validated now” → `research/current-status.md`;
- “agent/documentation rules” → `AGENTS.md`;
- “licensing/provenance” → permanent `../LICENSES/*` material.

### `docs/AGENTS.md`

Detailed documentation governance:

- current evidence/source-of-truth model;
- folder semantics;
- one-authority-per-concept rule;
- update triggers;
- naming rules (`lowercase-kebab-case.md` for new docs);
- temporary `specs/` and `plans/` lifecycle;
- no history/archive;
- required link/status/path verification before completion.

### `docs/architecture/lean-core.md`

Current runtime/research architecture only:

- lean package responsibilities;
- data → strategy/risk → simulation → evaluation flow;
- strategy intent/quantity-hold contract;
- Strategy vs Risk/Execution responsibility split;
- causality/data rules;
- execution/accounting single-authority rules;
- independent per-symbol replay principle.

### `docs/architecture/package-boundaries.md`

Filesystem and dependency contract:

- final top-level/subpackage ownership established by Phases 1–4A;
- allowed dependency direction;
- package-level public API policy;
- private path policy/no forwarding shims;
- `evaluation/runs` vs future `evaluation/experiments` separation;
- active spec/plan docs are never runtime authority.

### `docs/research/current-status.md`

Current research evidence state only:

- M1/M2/M3 status;
- 5 candidates + 3 controls;
- universal model/policy contract and separate fit/evaluation scopes;
- development evidence protocol/artifact outputs;
- winner/no-winner decision principles;
- no profitability claim;
- no Production/live routing authorization.

---

## Completed Artifact Classification

All documents present on the exact Phase 4A base under `docs/plans/` and `docs/specs/` are completed cleanup artifacts once Phase 4B copies their durable current content into the new authority docs.

### DELETE after durable-content preservation

`docs/plans/`:

- `2026-09-08-binance-definition-inventory.md`
- `2026-09-08-lean-binance-adapter-boundaries.md`
- `2026-09-08-lean-data-lifecycle-boundaries.md`
- `2026-09-08-lean-evaluation-ownership-boundaries.md`
- `2026-09-08-lean-package-boundaries-foundation.md`
- `2026-09-08-lean-package-boundaries.md`
- `2026-09-08-lean-package-file-inventory.md`
- `2026-09-08-lean-simulation-ownership-boundaries.md`
- `2026-09-08-lean-strategy-family-boundaries.md`
- this Phase 4B plan itself after successful completion.

`docs/specs/`:

- `2026-09-08-binance-adapter-boundary-amendment.md`
- `2026-09-08-lean-package-boundaries-amendment-1.md`
- `2026-09-08-lean-package-boundaries-design.md`

Root docs:

- `docs/trade_rl_lean_redesign_20260908.md`

No replacement archive directory is created.

---

## Task 1: Establish RED documentation contract

**Files:**
- Create: `tests/architecture/test_current_documentation_contract.py`

**Interfaces:**
- Consumes: final target docs tree and current observed research/package contracts.
- Produces: executable requirements for docs shape, routing, links, status, compliance retention, and absence of completed/history artifacts.

- [ ] **Step 1: Write the failing contract test**

The test must require:

- exact final docs paths;
- root `AGENTS.md`;
- no `docs/plans`, `docs/specs`, `docs/history`, `docs/archive`, or old redesign file;
- root/docs routing links;
- current CLI path and absence of old candidate CLI;
- exact current research-state statements;
- final package-owner names;
- no PR-number/full-SHA normative references in current authority docs;
- relative Markdown link resolution;
- exact SHA-256 of permanent licensing/provenance files captured from the exact Phase 4A base.

- [ ] **Step 2: Run RED verification**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Expected: static checks Green. Documentation contract fails because root/docs entry files and authority directories do not yet exist and completed plans/specs/redesign remain. Unrelated existing tests remain Green.

- [ ] **Step 3: Commit RED contract**

```bash
git add tests/architecture/test_current_documentation_contract.py
 git commit -m "test: require current-only documentation routing"
```

---

## Task 2: Build current authority docs before deleting old sources

**Files:**
- Create: `AGENTS.md`
- Create: `docs/README.md`
- Create: `docs/AGENTS.md`
- Create: `docs/architecture/lean-core.md`
- Create: `docs/architecture/package-boundaries.md`
- Create: `docs/research/current-status.md`
- Modify: `README.md`

**Interfaces:**
- Consumes: exact Phase 4A source tree, root README, redesign doc, final package-boundary spec/amendments.
- Produces: the durable current documentation authorities that replace those completed artifacts.

- [ ] **Step 1: Write all current authority docs**

Do not copy chronology or PR/commit references. Rewrite current-state contracts only.

- [ ] **Step 2: Update root README routing**

Replace the old redesign-doc authority link with `docs/README.md` and `docs/research/current-status.md`. Preserve the maintained development-run command and current status without inflating claims.

- [ ] **Step 3: Run documentation contract before deletion**

Expected: current-doc content/routing portions should become Green; retention-layout tests remain RED because completed artifacts still exist.

---

## Task 3: Remove completed documentation artifacts

**Files:**
- Delete every file listed under Completed Artifact Classification.
- Delete the Phase 4B plan itself only after Tasks 1–2 content has been durably reflected and pre-deletion checks pass.

**Interfaces:**
- Consumes: durable current authority docs.
- Produces: current-only working tree with Git history as the archive.

- [ ] **Step 1: Verify no current authority links to a file scheduled for deletion**

Use repository Markdown link scanning before deletion.

- [ ] **Step 2: Delete completed redesign/plans/specs**

Do not move them to another folder.

- [ ] **Step 3: Verify final docs tree is exact**

The only `docs/` files are the five current authority files listed in the final tree.

---

## Task 4: Full verification and falsification

- [ ] **Step 1: Run documentation contract**

```bash
uv run pytest -q tests/architecture/test_current_documentation_contract.py
```

Expected: all pass.

- [ ] **Step 2: Run static checks and full tests**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run pytest -q tests
```

Expected: all pass.

- [ ] **Step 3: Package identity**

Require installed metadata version == `trade_rl.__version__`.

- [ ] **Step 4: Falsification review**

Explicitly try to find:

- an orphan current doc not routed from `docs/README.md`;
- a broken relative link;
- an old private module/CLI path in current docs;
- completed plan/spec/redesign text surviving;
- a current contract present only in deleted docs and missing from new authorities;
- duplicated normative contract with inconsistent values;
- transient PR/SHA references;
- accidentally changed/deleted licensing/provenance files;
- a claim that M2 real-data comparison/profitability/Production readiness is complete when it is not.

- [ ] **Step 5: Normal PR CI on exact final helper-free HEAD**

Do not reuse earlier CI results. Record exact SHA and observed Ruff/Format/Mypy/test/package-identity results.

---

## Self-Review

- Spec section 11 is covered: current-only docs, no history/archive, completed plan/spec deletion, licensing retention.
- User agent-routing requirement is covered by root `AGENTS.md` bootstrap plus detailed `docs/AGENTS.md` governance without duplicated full policy.
- Current research content from the redesign doc is assigned to `lean-core.md` or `current-status.md` before deletion.
- Final package ownership from Phases 1–4A is assigned to `package-boundaries.md`.
- The plan explicitly prevents deletion-only “cleanup” from erasing normative current content.
- The plan does not implement the Controlled Experiment Loop or change production code.
