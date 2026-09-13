# Human Guide Markdown-first Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Status: Active

**Goal:** Replace the visualization-first Human Guide with a Markdown-first technical document whose core explanation is readable without interaction while preserving exact-SHA source traceability, search, accessibility, and deployment contracts.

**Architecture:** Guide pages become `pages/<id>.md` plus `meta/<id>.json`. TypeScript pairs raw Markdown with machine metadata and renders the article first; code references move to a collapsed appendix. Manifest v2 owns grouped navigation and reading order; source/code freshness remains fail-closed in Python tooling.

**Tech Stack:** React 19, TypeScript 6, Vite 8 raw imports, `react-markdown`, `remark-gfm`, Vitest, Testing Library, Playwright, Python content-contract tooling.

**Spec:** `docs/specs/2026-09-13-guide-markdown-first-design.md`

## Global Constraints

- `trade_rl/**` runtime semantics must not change.
- Manifest schema is `document-guide-v2`.
- Page roles are exactly `overview | detail | status | reference`.
- `overview` and `status` require `code_references=[]`; `reference` requires at least one code reference.
- No MDX, raw HTML execution, Mermaid, custom Markdown directives, or arbitrary JavaScript in content.
- Markdown renderer is `react-markdown + remark-gfm`; unsafe link protocols are rejected.
- Source links remain pinned to `GUIDE_SOURCE_REV` exact SHA.
- Full AST index and compact runtime symbol index remain fail-closed.
- Production JavaScript chunk budget remains 500,000 bytes; do not raise the limit.
- 320px document-level horizontal overflow remains forbidden.
- Overview/status must not mount implementation references.
- Replay/PPO primary sequences must be present in the initial DOM without click/tap.

---

### Task 1: Introduce the document content contract

**Files:**
- Modify: `guide/content/manifest.json`
- Create: `guide/content/pages/*.md`
- Create: `guide/content/meta/*.json`
- Modify: `guide/src/content/schema.ts`
- Modify: `guide/src/content/loadTopics.ts`
- Create: `guide/src/content/markdown.ts`
- Modify: `guide/tools/code_symbols.py`
- Modify: `guide/tools/content_contract.py`
- Modify: `guide/package.json`
- Test: `guide/tests/content.test.ts`
- Test: `tests/architecture/test_human_guide_runtime_code_symbols.py`
- Add targeted architecture tests for Markdown/meta pairing and role contracts.

**Interfaces:**
- `GuidePageRole = "overview" | "detail" | "status" | "reference"`.
- `GuideTopic` exposes `id`, `title`, `nav_label`, `summary`, `role`, `keywords`, `source_sections`, `code_references`, `markdown`, and parsed `headings`; it no longer exposes primary `sections` or `visualization`.
- `GuideManifest` exposes `groups` and `reading_order`.
- `loadTopics()` pairs eager raw Markdown imports from `content/pages/*.md` with JSON metadata from `content/meta/*.json` and rejects missing/orphan/id-mismatched pairs.
- `extractMarkdownText(markdown)` and `extractMarkdownHeadings(markdown)` are deterministic utilities used by search and tests.

- [ ] **Step 1: Write failing TypeScript contract tests**

Add tests that require `document-guide-v2`, grouped navigation, full Markdown/meta pairing, role constraints, deterministic headings, and duplicate normalized heading rejection.

- [ ] **Step 2: Write failing Python architecture tests**

Require `code_symbols.py` to discover code references from `content/meta`, and require content checks to reject missing/orphan Markdown/meta, invalid roles, overview/status code references, reference pages without code references, and duplicate ATX heading slugs.

- [ ] **Step 3: Run RED**

Run:

```bash
uv run pytest -q tests/architecture/test_human_guide_runtime_code_symbols.py tests/architecture -k 'guide or docs_layout'
cd guide && npm test -- --run tests/content.test.ts
```

Expected: semantic failures because v2 manifest/pages/meta/role/headings do not exist yet. Import/path setup failures do not count as RED.

- [ ] **Step 4: Add dependencies and minimal v2 loader/schema**

Add pinned `react-markdown` and `remark-gfm`, implement the v2 manifest/meta parser, raw Markdown loader, deterministic heading extraction/slugging, and role validation.

- [ ] **Step 5: Update Python content/code contracts**

Move topic discovery from `content/topics/*.json` to `content/meta/*.json`, preserving source-section fingerprint refresh/check behavior and code digest/kind/variable/test fail-closed validation.

- [ ] **Step 6: Create all eight page/meta pairs**

Migrate metadata without changing research facts. `overview` and `research-status` have empty code references; `code-map` remains reference metadata with code references.

- [ ] **Step 7: Run GREEN**

Run targeted TypeScript and Python contract tests, then `npm run source-check` and runtime symbol-index tests.

---

### Task 2: Render safe Markdown first and move implementation details to an appendix

**Files:**
- Create: `guide/src/components/MarkdownArticle.tsx`
- Create: `guide/src/components/ImplementationReferenceAppendix.tsx`
- Refactor/reuse: `guide/src/components/CodeInspector.tsx` source-link/detail primitives as needed; do not retain the old side-by-side primary inspector.
- Modify: `guide/src/app/App.tsx`
- Modify: `guide/src/app/useHashRoute.ts`
- Modify: `guide/src/components/SourceLinks.tsx`
- Modify: `guide/src/styles/index.css` (or current main stylesheet)
- Test: create `guide/tests/markdown-article.test.tsx`
- Test: update `guide/tests/interactions.test.tsx`
- Test: update `guide/tests/routing.test.ts`

**Interfaces:**
- `MarkdownArticle({ topic })` renders headings, paragraphs, lists, GFM tables, code, blockquotes, and safe links.
- `ImplementationReferenceAppendix({ topic, openSymbol? })` renders native `<details>` sections at article end, all collapsed unless `openSymbol` matches exactly one reference.
- Route canonical selections are `heading` or `symbol`; legacy `step` parses only through an explicit compatibility map and otherwise falls back to page root.

- [ ] **Step 1: Write failing renderer tests**

Assert Markdown renders heading/list/table/fenced text and raw HTML is displayed/ignored rather than executed. Assert `javascript:`/`data:` links are not emitted as navigable unsafe links.

- [ ] **Step 2: Write failing appendix tests**

Assert detail/reference appendix is after article content, initially collapsed, keyboard/native-details accessible, and only a symbol deep link opens its matching item. Assert overview/status never mount it.

- [ ] **Step 3: Write failing routing tests**

Require `heading` and `symbol` formatting/parsing, legacy step compatibility for known replay/PPO steps, and unknown step fallback to root.

- [ ] **Step 4: Run RED**

Run the three targeted Vitest files and confirm failures are due to missing Markdown renderer/appendix/new route behavior.

- [ ] **Step 5: Implement minimal renderer/appendix/routing**

Use `react-markdown` with `remark-gfm`, no `rehype-raw`; explicit safe link URI handling; native `<details>`; exact-SHA source links from existing symbol metadata.

- [ ] **Step 6: Run GREEN and refactor**

Run targeted tests, then full Vitest. Remove duplicated CodeInspector-only primary-layout state while retaining reusable metadata/source rendering.

---

### Task 3: Make search document-aware without making interaction mandatory

**Files:**
- Modify: `guide/src/content/search.ts`
- Modify: `guide/src/components/SearchPalette.tsx`
- Modify: `guide/src/app/App.tsx`
- Test: `guide/tests/search.test.ts`
- Test: `guide/tests/interactions.test.tsx`

**Interfaces:**
- Search result kinds become `page | heading | symbol | variable`.
- Page/heading results navigate to topic/heading; symbol/variable results navigate to topic/symbol and open exactly one appendix item.
- Search indexes Markdown body/headings including inline/fenced code, plus metadata aliases/symbols/variables/source paths.

- [ ] **Step 1: Write failing search tests**

Require Markdown heading/body search, `desired_quantity` Japanese/Python alias parity, heading result routing, and symbol deep-link behavior.

- [ ] **Step 2: Run RED**

Run `npm test -- --run tests/search.test.ts tests/interactions.test.tsx` and confirm the old topic/code search model fails the new assertions.

- [ ] **Step 3: Implement document-aware search**

Use Markdown extraction utilities; preserve deterministic ranking and Japanese normalization; remove sequence-step search dependence.

- [ ] **Step 4: Run GREEN**

Run targeted tests and full Vitest.

---

### Task 4: Migrate navigation and all eight pages to reading-first content

**Files:**
- Modify: `guide/src/components/Sidebar.tsx`
- Modify: `guide/src/components/MobileNav.tsx`
- Modify: `guide/src/components/AppShell.tsx`
- Modify: all `guide/content/pages/*.md`
- Modify: all `guide/content/meta/*.json`
- Modify: `guide/src/app/App.tsx`
- Test: `guide/tests/content.test.ts`
- Test: `guide/tests/interactions.test.tsx`
- Replace obsolete visualization-specific unit tests with reading-first document tests.

**Interfaces:**
- Navigation receives manifest groups, not a flat list only.
- Next-reading uses `reading_order`; `code-map` is skipped.
- Overview structure: purpose → compact static core flow → verified/insufficient/not-proven table → reading map.
- Replay structure includes all semantic steps in static text and step headings.
- PPO structure includes Observation v2 five-part table and static action→risk→execution→reward flow.
- Data/execution/experiment pages expose assumptions/invariants/failure states without clicks.
- Research status is prose/table only.
- Code map is a static reference/ownership table.

- [ ] **Step 1: Add failing reading-order/navigation tests**

Require four navigation groups and reading order that excludes `code-map`.

- [ ] **Step 2: Add failing page-content tests**

Require overview proven/unproven content, replay semantic sequence labels, PPO five observation segments and reward flow, and absence of primary visualization controls.

- [ ] **Step 3: Run RED**

Run targeted content/interactions tests.

- [ ] **Step 4: Migrate content and grouped navigation**

Translate existing JSON semantics into Markdown without strengthening research claims. Keep code references only in metadata appendix.

- [ ] **Step 5: Run GREEN**

Run full Vitest plus `npm run source-check`.

---

### Task 5: Remove dead visualization-first implementation and enforce responsive document styling

**Files:**
- Delete dead files under `guide/src/visualizations/` once no longer referenced.
- Delete/replace obsolete `architecture-flow`, `sequence-diagram`, `ppo-sequence`, `code-map`, and graph-layout tests when their behavior is replaced by document-contract tests.
- Modify: `guide/src/styles/index.css`
- Modify: `guide/README.md`
- Modify: `guide/package.json` / lockfile as needed.
- Test: architecture/runtime symbol/bundle budget tests.

**Interfaces:**
- No `VisualizationRenderer` import in application path.
- Text/code blocks may scroll locally; `.main-content`/document root may not overflow horizontally.
- No production dependency remains solely for removed visualization code.

- [ ] **Step 1: Add/adjust failing regression tests**

Assert App has no primary visualization stage, runtime bundle still only contains referenced code symbols, and package/build contract still enforces 500,000 bytes.

- [ ] **Step 2: Remove dead visualization code and stale CSS**

Delete only after replacement tests are Green; keep reusable non-visual components.

- [ ] **Step 3: Update Guide README**

Document Markdown/meta authority split, grouped roles, appendix behavior, source contracts, and build budget.

- [ ] **Step 4: Run full static/unit/build verification**

```bash
uv run pytest -q tests
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
cd guide && npm run check
```

Expected: all Green and production JS chunk <= 500,000 bytes with no oversized warning.

---

### Task 6: Replace browser/public oracles and complete the branch

**Files:**
- Modify: `guide/e2e/guide.spec.ts`
- Modify: `guide/e2e/accessibility.spec.ts`
- Modify: `guide/e2e/public-smoke.spec.ts`
- Modify/remove obsolete visualization screenshot expectations under `guide/e2e/visual.spec.ts` as appropriate.
- Modify: PR #521 title/body after implementation evidence exists.
- Remove active spec/plan from current tree only at final completion after durable contracts are promoted to `guide/README.md`/tests.

**Interfaces / Test Oracle:**
- Desktop and 320px mobile overview are readable without interaction.
- Replay initial DOM exposes observation→strategy→risk→execution→book order.
- PPO initial DOM exposes all five observation parts and action→risk→execution→reward.
- Appendix starts collapsed and opens by keyboard.
- `desired_quantity` search opens the replay implementation reference.
- Exact-SHA source link is correct.
- Serious/critical axe violations are zero.
- Document-level horizontal overflow is zero.
- Public smoke checks the same reading-first path after Pages deployment.

- [ ] **Step 1: Write failing E2E expectations against the old UI**

Run the relevant Playwright tests and confirm reading-first assertions fail for semantic reasons.

- [ ] **Step 2: Complete CSS/copy/test migration until local E2E is Green**

Run `npm run e2e`; expected project-specific skips only, zero failures.

- [ ] **Step 3: Final diff self-review and falsification**

Check acceptance criteria, source/research fact parity, dead code, unsafe links, routing compatibility, bundle size, no `trade_rl/**` runtime semantic changes, no generated/dist files, and whether any current test could pass a visualization-first regression.

- [ ] **Step 4: Exact-head CI**

Require Lean Core and Human Guide jobs Green on the final PR HEAD only.

- [ ] **Step 5: Finish branch**

Update PR #521 to implementation status, remove active spec/plan after durable contract promotion, re-run exact-head CI, and merge only after final evidence is current.

- [ ] **Step 6: Post-merge verification**

Require exact main push CI Green, `Deploy Human Guide` build/deploy Green, and public smoke Green against `https://shuntatsu.github.io/trade_rl/` with `GUIDE_SOURCE_REV` equal to the merged main SHA.
