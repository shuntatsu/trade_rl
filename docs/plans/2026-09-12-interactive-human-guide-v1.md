# Interactive Human Guide v1 Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a React/Vite/Tailwind human-facing interactive guide whose structured explanations stay mechanically traceable to the authoritative Trade RL docs and are pleasant to use in both low-contrast light and dark themes.

**Architecture:** Keep `docs/` authoritative and place the non-authoritative frontend in root `guide/`. Human explanations live in JSON and are rendered by generic React visualization families; topic-specific research truth is not embedded in JSX. A Python source-contract checker fingerprints exact authoritative Markdown sections so docs changes make the Guide stale until a human reviews the affected topic. Node tooling is isolated to `guide/`, while existing Python Lean Core CI remains unchanged in scope.

**Tech Stack:** Node.js 24 LTS, React 19.3, TypeScript, Vite 8 current supported line, Tailwind CSS 4.3 with `@tailwindcss/vite`, Vitest, React Testing Library, Playwright Chromium, Python 3.12 standard library.

**Spec:** `docs/specs/2026-09-12-interactive-human-guide-v1.md`

## Global Constraints

- `guide/` is explanatory and non-authoritative.
- Current technical/research authorities remain `docs/architecture/*` and `docs/research/current-status.md`.
- Node packages live only under `guide/package.json` / `guide/package-lock.json`.
- No `trade_rl/**` runtime/public API/artifact contract changes.
- Explanation copy and visualization models are authored in `guide/content/topics/*.json`, not scattered through React components.
- Every topic binds to at least one exact authoritative Markdown section using reviewed SHA-256 fingerprints.
- Missing/changed source section, malformed content, broken visualization references, duplicate IDs, TypeScript errors, browser interaction regressions, or failed production build must fail closed.
- Existing Python Lean Core CI is not weakened or made conditional on Guide success.
- Light and dark themes share one information architecture and one content source.
- Approved visual direction: current information density, slightly lower contrast/saturation, cool neutral surfaces, clear interactive states, readability and visualization above decoration.
- Normal body text remains WCAG AA >= 4.5:1 even with the softer contrast direction.
- Main UX works from 320px through desktop and does not rely on hover.
- Verified baseline UI must not imply profitability, winner selection, or Production authorization.
- Completed Active spec/plan are removed before final Ready state; durable maintenance rules move to `docs/README.md` / `docs/AGENTS.md`.

## File Structure

```text
guide/
  README.md
  package.json
  package-lock.json
  .nvmrc
  index.html
  vite.config.ts
  vitest.config.ts
  playwright.config.ts
  tsconfig.json
  tsconfig.app.json
  tsconfig.node.json
  src/
    main.tsx
    app/
      App.tsx
      useHashRoute.ts
      useTheme.ts
    components/
      AppShell.tsx
      Sidebar.tsx
      MobileNav.tsx
      SearchPalette.tsx
      SourceLinks.tsx
      TopicHeader.tsx
      DetailPanel.tsx
    visualizations/
      ArchitectureFlow.tsx
      FlowStepper.tsx
      ObservationVector.tsx
      EconomicsAuthorityPath.tsx
      ExperimentLoop.tsx
      ResearchStatusBoard.tsx
    content/
      schema.ts
      loadTopics.ts
      search.ts
    styles/
      app.css
    test/
      setup.ts
  content/
    manifest.json
    topics/
      overview.json
      data-flow.json
      architecture.json
      execution-economics.json
      ppo-observation-v2.json
      experiment-loop.json
      research-status.json
  tools/
    content_contract.py
  tests/
    content.test.ts
    routing.test.tsx
    interactions.test.tsx
    theme.test.tsx
  e2e/
    guide.spec.ts
    accessibility.spec.ts
    visual.spec.ts

tests/architecture/test_human_guide.py
README.md
docs/README.md
docs/AGENTS.md
.gitignore
.github/workflows/ci.yml
```

---

### Task 1: Establish the Guide boundary and frontend workspace RED

**Files:**
- Create: `tests/architecture/test_human_guide.py`
- Create: `guide/README.md`
- Create: `guide/package.json`
- Create: `guide/.nvmrc`
- Modify: `.gitignore`

**Interfaces:**
- Consumes current-only docs policy from `docs/README.md` / `docs/AGENTS.md`.
- Produces permanent assertions that the Guide is outside authoritative `docs/`, has its own Node workspace, and cannot add production Python ownership.

- [ ] **Step 1: Write the failing architecture contract.**

Create `tests/architecture/test_human_guide.py` with assertions equivalent to:

```python
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GUIDE = ROOT / "guide"


def test_human_guide_is_non_authoritative_and_isolated() -> None:
    assert (GUIDE / "README.md").is_file()
    assert (GUIDE / "package.json").is_file()
    text = (GUIDE / "README.md").read_text(encoding="utf-8")
    assert "non-authoritative" in text
    assert "docs/architecture" in text
    assert "docs/research/current-status.md" in text
    assert not (ROOT / "docs" / "history").exists()
    assert not (ROOT / "docs" / "archive").exists()
```

Also assert the root Python package metadata does not acquire React/Vite/Tailwind dependencies and `guide/dist`, `guide/node_modules`, `guide/playwright-report`, `guide/test-results`, `guide/coverage` are ignored.

- [ ] **Step 2: Run the focused test and observe RED.**

```bash
uv run pytest -q tests/architecture/test_human_guide.py
```

Expected: failures for missing Guide workspace files/ignore entries.

- [ ] **Step 3: Add the minimal workspace manifest and non-authority README.**

`guide/README.md` must state verbatim concepts:

```text
This guide is non-authoritative.
Technical truth lives in docs/architecture/* and docs/research/current-status.md.
Human-facing content lives in guide/content/topics/*.json.
```

`guide/.nvmrc` pins the chosen Node 24 LTS patch. `package.json` sets `private: true` and `engines.node` to Node 24.x.

- [ ] **Step 4: Add only generated-output ignore rules.**

```text
guide/node_modules/
guide/dist/
guide/coverage/
guide/playwright-report/
guide/test-results/
```

- [ ] **Step 5: Re-run the focused test.**

Expected: boundary/ignore assertions Green; runtime/content assertions intentionally remain absent until later tasks.

- [ ] **Step 6: Commit the boundary checkpoint.**

```bash
git add .gitignore guide/README.md guide/package.json guide/.nvmrc tests/architecture/test_human_guide.py
git commit -m "test: define interactive human guide boundary"
```

---

### Task 2: Install and pin React/Vite/Tailwind/TypeScript test tooling

**Files:**
- Modify: `guide/package.json`
- Create: `guide/package-lock.json`
- Create: `guide/tsconfig.json`
- Create: `guide/tsconfig.app.json`
- Create: `guide/tsconfig.node.json`
- Create: `guide/vite.config.ts`
- Create: `guide/vitest.config.ts`
- Create: `guide/playwright.config.ts`
- Create: `guide/src/test/setup.ts`

**Interfaces:**
- Produces npm scripts: `dev`, `lint`, `typecheck`, `test`, `build`, `e2e`, `check`.
- Tailwind is integrated through `@tailwindcss/vite`, not PostCSS.

- [ ] **Step 1: Install exact core runtime versions and lock them.**

Use Node 24 LTS and create a lockfile with:

```bash
npm --prefix guide install --save-exact react@19.3.0 react-dom@19.3.0
npm --prefix guide install --save-dev --save-exact \
  vite@8 \
  @vitejs/plugin-react \
  typescript \
  tailwindcss@4.3 \
  @tailwindcss/vite@4.3 \
  vitest \
  jsdom \
  @testing-library/react \
  @testing-library/jest-dom \
  @testing-library/user-event \
  @playwright/test \
  eslint \
  @eslint/js \
  typescript-eslint \
  eslint-plugin-react-hooks \
  eslint-plugin-react-refresh
```

`package-lock.json` is the reproducibility authority for exact transitive versions.

- [ ] **Step 2: Add scripts.**

```json
{
  "scripts": {
    "dev": "vite",
    "lint": "eslint src tests e2e vite.config.ts vitest.config.ts playwright.config.ts",
    "typecheck": "tsc -b --pretty false",
    "test": "vitest run",
    "build": "tsc -b && vite build",
    "e2e": "playwright test",
    "check": "npm run lint && npm run typecheck && npm run test && npm run build"
  }
}
```

- [ ] **Step 3: Configure Vite + Tailwind.**

`vite.config.ts` uses:

```ts
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";
import { defineConfig } from "vite";

export default defineConfig({
  base: "./",
  plugins: [react(), tailwindcss()],
});
```

`base: "./"` keeps the production bundle portable for future static hosting.

- [ ] **Step 4: Configure Vitest and Playwright.**

Vitest uses `jsdom`, setup file, and coverage only as a signal. Playwright starts `npm run dev -- --host 127.0.0.1` on a fixed local port, uses Chromium, and defines desktop plus 320px projects.

- [ ] **Step 5: Run empty-toolchain verification.**

```bash
npm --prefix guide ci
npm --prefix guide run typecheck
npm --prefix guide run lint
```

Expected: Green with no application source yet beyond setup/config.

- [ ] **Step 6: Commit.**

```bash
git add guide/package.json guide/package-lock.json guide/*.json guide/*.ts guide/src/test/setup.ts
git commit -m "build: add guide React Vite Tailwind toolchain"
```

---

### Task 3: Implement authoritative-doc freshness checking before UI work

**Files:**
- Create: `guide/tools/content_contract.py`
- Create: `guide/content/manifest.json`
- Create: `guide/content/topics/overview.json`
- Extend: `tests/architecture/test_human_guide.py`

**Interfaces:**
- `extract_markdown_section(text: str, heading: str) -> str`
- `section_sha256(text: str) -> str`
- CLI: `--check` and `--refresh TOPIC_ID [TOPIC_ID ...]`
- Topics use `source_sections[] = {path, heading, sha256}`.

- [ ] **Step 1: Write RED unit/architecture tests for Markdown section extraction.**

Cover exact ATX headings, nested headings, CRLF normalization, duplicate heading rejection, missing heading rejection, and same/higher-level section termination.

- [ ] **Step 2: Add RED tests for topic contract failures.**

Reject:

```text
missing source file
missing source heading
empty source_sections
duplicate topic id
manifest/topic mismatch
stale sha256
unknown visualization kind
broken node/edge/step references
```

- [ ] **Step 3: Run RED.**

```bash
uv run pytest -q tests/architecture/test_human_guide.py
```

- [ ] **Step 4: Implement `content_contract.py` with only stdlib.**

Digest rule:

```python
hashlib.sha256(section.replace("\r\n", "\n").encode("utf-8")).hexdigest()
```

`--check` never mutates files. `--refresh` requires explicit topic IDs and rewrites only their source digests, preventing a blanket command from silently approving every stale topic.

- [ ] **Step 5: Create `manifest.json` and `overview.json` bound to real headings.**

Read the referenced current sections before running refresh.

- [ ] **Step 6: Prove stale detection.**

A test modifies a copied authoritative section without touching topic JSON and verifies `--check` fails with the topic/source heading named in the error.

- [ ] **Step 7: Run focused quality checks.**

```bash
uv run ruff check guide/tools tests/architecture/test_human_guide.py
uv run ruff format --check guide/tools tests/architecture/test_human_guide.py
uv run pytest -q tests/architecture/test_human_guide.py
uv run python guide/tools/content_contract.py --check
```

- [ ] **Step 8: Commit.**

```bash
git add guide/content guide/tools tests/architecture/test_human_guide.py
git commit -m "feat: bind guide topics to authoritative docs"
```

---

### Task 4: Build the typed React shell and approved low-contrast theme system

**Files:**
- Create: `guide/index.html`
- Create: `guide/src/main.tsx`
- Create: `guide/src/app/App.tsx`
- Create: `guide/src/app/useHashRoute.ts`
- Create: `guide/src/app/useTheme.ts`
- Create: `guide/src/content/schema.ts`
- Create: `guide/src/content/loadTopics.ts`
- Create: `guide/src/styles/app.css`
- Create: `guide/tests/content.test.ts`
- Create: `guide/tests/theme.test.tsx`

**Interfaces:**
- `parseGuideManifest(value: unknown): GuideManifest`
- `parseTopic(value: unknown): GuideTopic`
- `loadTopics(): GuideTopic[]`
- `useHashRoute(validIds: readonly string[]): [string, (id: string) => void]`
- `useTheme(): {theme, resolvedTheme, setTheme}` where explicit preference is `light | dark | system`.

- [ ] **Step 1: Write failing typed-content tests.**

Assert all topic/manifest objects fail closed on unknown visualization kinds, missing required fields, duplicate IDs, and broken references.

- [ ] **Step 2: Write failing theme tests.**

Test priority:

```text
saved explicit preference > OS prefers-color-scheme > light fallback
```

Storage exceptions must not break rendering.

- [ ] **Step 3: Implement content types/runtime guards.**

Do not use `as GuideTopic` on raw JSON without validation.

- [ ] **Step 4: Implement hash routing without React Router.**

The hook listens to `hashchange`, normalizes unknown hashes to home, and never keeps a competing second route authority in component state.

- [ ] **Step 5: Implement Tailwind theme tokens.**

`src/styles/app.css` starts with:

```css
@import "tailwindcss";
@custom-variant dark (&:where(.dark, .dark *));
```

Use CSS custom properties for the approved palette.

Light:

```text
bg #f5f7fa
surface #f9fafb
surface-2 #f1f4f8
text #334155
muted #64748b
border #dbe3ec
accent #6f8faa
```

Dark:

```text
bg #17202b
surface #1d2835
surface-2 #243140
text #bec9d8
muted #95a5b8
border #334254
accent #7899b6
```

Do not use pure black/white for primary surfaces/text. Persistent category colors use restrained low-saturation teal/violet/amber/rose; accent is reserved mainly for focus/selection/actions.

- [ ] **Step 6: Add a token-level WCAG contrast test.**

Implement relative luminance in Vitest and assert body text/background and body text/surface pairs are >= 4.5:1 in both themes. If the approved token approximation fails, adjust tokens rather than weakening the test.

- [ ] **Step 7: Implement the smallest renderable `App` with theme toggle and home route.**

- [ ] **Step 8: Run targeted frontend checks.**

```bash
npm --prefix guide run lint
npm --prefix guide run typecheck
npm --prefix guide run test
npm --prefix guide run build
```

- [ ] **Step 9: Commit.**

```bash
git add guide
git commit -m "feat: add typed interactive guide shell"
```

---

### Task 5: Implement navigation, search, responsive shell, and progressive disclosure

**Files:**
- Create: `guide/src/components/AppShell.tsx`
- Create: `guide/src/components/Sidebar.tsx`
- Create: `guide/src/components/MobileNav.tsx`
- Create: `guide/src/components/SearchPalette.tsx`
- Create: `guide/src/components/TopicHeader.tsx`
- Create: `guide/src/components/SourceLinks.tsx`
- Create: `guide/src/components/DetailPanel.tsx`
- Create: `guide/src/content/search.ts`
- Create: `guide/tests/routing.test.tsx`
- Extend: `guide/tests/interactions.test.tsx`

**Interfaces:**
- `searchTopics(topics, query)` searches title/summary/keywords/section headings.
- Topic selection writes URL hash; browser back/forward restores the selected topic.
- Search result selection uses the same route authority.

- [ ] **Step 1: Add RED tests for route/search behavior.**

Cover unknown hash fallback, direct deep link, back/forward, empty query, case-insensitive keyword hit, and deterministic result ordering.

- [ ] **Step 2: Implement desktop shell matching the approved concept.**

Layout priorities:

```text
left topic rail
central visualization/content
right contextual status/source rail when space allows
```

Avoid excessive card nesting. Use open whitespace and one purposeful panel boundary around the main visualization.

- [ ] **Step 3: Implement `Ctrl/Cmd+K` search palette.**

Do not intercept shortcut while another editable element owns focus. Search remains fully usable by click/tap.

- [ ] **Step 4: Implement 320px mobile reflow.**

Desktop sidebar becomes a compact accessible navigation control. Selected detail moves below the visualization rather than being hidden off-canvas.

- [ ] **Step 5: Implement progressive disclosure.**

Home shows the overall map + research status + three learning entrances. Deep technical copy remains collapsed until requested.

- [ ] **Step 6: Run tests/build.**

```bash
npm --prefix guide run check
```

- [ ] **Step 7: Commit.**

```bash
git add guide/src guide/tests
git commit -m "feat: add guide navigation search and responsive UX"
```

---

### Task 6: Implement the six interactive visualization families and all seven topics

**Files:**
- Create: `guide/src/visualizations/ArchitectureFlow.tsx`
- Create: `guide/src/visualizations/FlowStepper.tsx`
- Create: `guide/src/visualizations/ObservationVector.tsx`
- Create: `guide/src/visualizations/EconomicsAuthorityPath.tsx`
- Create: `guide/src/visualizations/ExperimentLoop.tsx`
- Create: `guide/src/visualizations/ResearchStatusBoard.tsx`
- Create: remaining six `guide/content/topics/*.json`
- Extend: `guide/tests/content.test.ts`
- Extend: `guide/tests/interactions.test.tsx`

**Interfaces:**
- Components consume declarative JSON models only.
- Selection state is local UI state; source truth remains content JSON.
- Every selectable visualization item is keyboard accessible and exposes selected state via ARIA.

- [ ] **Step 1: Add RED content tests requiring exact seven topic IDs.**

```text
overview
data-flow
architecture
execution-economics
ppo-observation-v2
experiment-loop
research-status
```

- [ ] **Step 2: Author each topic from current authoritative sections.**

Each topic has `source_sections`, human copy, and declarative visualization data. Do not include transient PR numbers or 40-char commit SHAs in durable human copy.

- [ ] **Step 3: Implement `ArchitectureFlow`.**

Selecting a node persists a detail panel with:

```text
Responsibility
Consumes
Produces
Does not own
Source docs
```

- [ ] **Step 4: Implement `FlowStepper` for Data Flow.**

Steps:

```text
source -> freeze -> build -> dataset artifact -> strategy -> replay -> evidence
```

Previous/Next buttons and direct step selection work by keyboard/touch.

- [ ] **Step 5: Implement `ObservationVector`.**

Segments are exactly:

```text
local values
availability / finite mask
normalized staleness
current intent
current weight
```

Show symbol ID and dataset-global aggregates explicitly in an "excluded from v2" area.

- [ ] **Step 6: Implement `EconomicsAuthorityPath`.**

Path:

```text
ExecutionEconomicsProfile
-> MarketDatasetBuilder
-> immutable Dataset economic arrays
-> zero-overlay executor
-> total_cost diagnostics
```

Generic fee and maker/taker models are shown as mutually exclusive authority choices; UI states the no-double-charge rule.

- [ ] **Step 7: Implement `ExperimentLoop`.**

States:

```text
baseline -> preregister -> execute -> verify -> decide -> freeze
```

Rejected/failed/invalid branches remain visible as semantic outcomes rather than hidden errors.

- [ ] **Step 8: Implement `ResearchStatusBoard`.**

Use four statuses with icon + text:

```text
verified
pending
not claimed
limitation
```

The Study 004 baseline can be shown as verified while profitability/winner/Production remain separate `not claimed` states.

- [ ] **Step 9: Refresh only reviewed source fingerprints.**

```bash
uv run python guide/tools/content_contract.py --refresh \
  overview data-flow architecture execution-economics \
  ppo-observation-v2 experiment-loop research-status
uv run python guide/tools/content_contract.py --check
```

- [ ] **Step 10: Run targeted frontend tests/build.**

```bash
npm --prefix guide run check
```

- [ ] **Step 11: Commit.**

```bash
git add guide
git commit -m "feat: add interactive Trade RL visual explanations"
```

---

### Task 7: Add real-browser UX, accessibility, dark-mode, and responsive verification

**Files:**
- Create/extend: `guide/e2e/guide.spec.ts`
- Create/extend: `guide/e2e/accessibility.spec.ts`
- Create/extend: `guide/e2e/visual.spec.ts`
- Modify as failures demand: React/CSS files only

**Interfaces:**
- Browser verification is the Test Oracle for interaction/responsive behavior.
- Visual screenshots are QA artifacts, not checked-in authority.

- [ ] **Step 1: Write Playwright tests before final visual polish.**

Required browser assertions:

```text
Home renders and all seven topics are reachable.
Architecture node selection updates persistent detail.
Data-flow stepper reaches first/last state and clamps bounds.
PPO segment selection updates explanation.
Economics path exposes no-double-charge rule.
Experiment loop exposes verify/decide/freeze states.
Theme toggle persists and system mode still works.
Ctrl/Cmd+K search selects a result.
Browser back/forward restores topics.
```

- [ ] **Step 2: Add desktop light/dark viewport coverage.**

Verify at a stable desktop viewport close to the approved concept proportions. Capture screenshots for review but do not commit temporary screenshots.

- [ ] **Step 3: Add 320px viewport checks.**

Assert:

```ts
expect(await page.evaluate(() => document.documentElement.scrollWidth <= document.documentElement.clientWidth)).toBe(true);
```

Primary navigation, visualization selection, detail, search, and theme remain usable.

- [ ] **Step 4: Add keyboard-only path.**

Use Tab/Enter/Space/Escape only to navigate topic, diagram selection, search, theme, and close transient UI.

- [ ] **Step 5: Add reduced-motion test.**

With Playwright `reducedMotion: "reduce"`, no information or required state transition disappears.

- [ ] **Step 6: Run Playwright and fix actual browser defects rather than weakening assertions.**

```bash
npx --prefix guide playwright install chromium
npm --prefix guide run e2e
```

- [ ] **Step 7: Fidelity review against the approved visual concept.**

Inspect at minimum:

```text
information hierarchy
soft light palette
soft blue-charcoal dark palette
sidebar/main/right-rail balance
system flow readability
card/surface restraint
text density
interactive selected states
mobile collapse
```

Use implementation screenshots and the approved concept side-by-side; fix material mismatches.

- [ ] **Step 8: Commit.**

```bash
git add guide/src guide/e2e
git commit -m "test: verify guide UX themes and responsive behavior"
```

---

### Task 8: Integrate repository routing, CI, lifecycle cleanup, and final verification

**Files:**
- Modify: `README.md`
- Modify: `docs/README.md`
- Modify: `docs/AGENTS.md`
- Modify: `.github/workflows/ci.yml`
- Extend: `tests/architecture/test_human_guide.py`
- Delete after durable rules land:
  - `docs/specs/2026-09-12-interactive-human-guide-v1.md`
  - `docs/plans/2026-09-12-interactive-human-guide-v1.md`

**Interfaces:**
- Root README gives humans a Guide entrance.
- `docs/README.md` says Guide is explanatory/non-authoritative.
- `docs/AGENTS.md` adds the update trigger: when an authoritative section used by Guide changes, update/check the matching topic.
- CI adds an independent `Human Guide` job; existing `Lean Core` job remains intact.

- [ ] **Step 1: Add RED routing/lifecycle assertions.**

Require root README link to `guide/README.md`, docs README authority warning, docs AGENTS update rule, and final absence of inactive spec/plan.

- [ ] **Step 2: Update routing docs.**

Durable wording must distinguish:

```text
Authoritative docs = machine/reviewer contract.
Interactive Guide = human explanation derived from those docs.
```

- [ ] **Step 3: Add `Human Guide` CI job.**

Use immutable action SHAs. Job outline:

```yaml
human-guide:
  runs-on: ubuntu-latest
  steps:
    - checkout exact PR/head SHA
    - setup Node 24 LTS
    - setup Python/uv only for source fingerprint checker
    - npm --prefix guide ci
    - uv run python guide/tools/content_contract.py --check
    - npm --prefix guide run lint
    - npm --prefix guide run typecheck
    - npm --prefix guide run test
    - npm --prefix guide run build
    - install Playwright Chromium
    - npm --prefix guide run e2e
```

Do not add `needs: human-guide` to Lean Core or vice versa; both report independently.

- [ ] **Step 4: Run all local/static gates.**

```bash
uv run python guide/tools/content_contract.py --check
npm --prefix guide ci
npm --prefix guide run lint
npm --prefix guide run typecheck
npm --prefix guide run test
npm --prefix guide run build
npm --prefix guide run e2e
uv run pytest -q tests/architecture/test_human_guide.py
```

- [ ] **Step 5: Run the repository Python final gate unchanged.**

```bash
uv run ruff check trade_rl tests guide/tools
uv run ruff format --check trade_rl tests guide/tools
uv run mypy trade_rl
uv run mypy tests/architecture/imports.py tests/architecture/distribution.py
uv run pytest -q tests
uv build
```

Also perform the existing distribution source-closure, sdist rebuild, clean non-editable install/public import/CLI smoke, and package identity checks.

- [ ] **Step 6: Self-review final diff and falsify update safety.**

Manually test these wrong implementations and confirm tests kill them:

```text
change a bound docs heading without updating Guide -> source checker RED
remove a source heading -> source checker RED
hand-author malformed topic reference -> content test RED
hide detail behind hover only -> Playwright keyboard/touch RED
make dark body text too dim -> contrast test RED
hardcode research claim into JSX instead of topic JSON -> architecture/source scan RED
commit dist/report/screenshot -> hygiene test/status review RED
```

- [ ] **Step 7: Promote durable maintenance rules, then remove Active design/plan.**

After `docs/README.md` / `docs/AGENTS.md` contain the permanent Guide contract:

```bash
git rm docs/specs/2026-09-12-interactive-human-guide-v1.md
git rm docs/plans/2026-09-12-interactive-human-guide-v1.md
```

Re-run current-doc layout tests to prove no completed Active artifact remains.

- [ ] **Step 8: Push/open PR and require exact-head CI Green.**

Final PR must remain unmerged until:

```text
Lean Core = success
Human Guide = success
no unresolved review thread
current main contained in tested head
final diff contains no temporary artifact
```

- [ ] **Step 9: Final independent review.**

Review original Acceptance Criteria against source/content/tests/browser evidence rather than treating CI Green alone as completion.

---

## Completion Oracle

The feature is complete only when all of the following are directly observed on one final HEAD:

- Guide is reachable from the repository root but clearly non-authoritative.
- All seven topics are interactive and sourced from structured JSON.
- A changed bound source-doc section causes a deterministic stale-Guide failure.
- Light and dark modes both match the approved low-contrast visual direction without violating AA body-text contrast.
- Desktop and 320px browser paths work by touch/mouse and keyboard.
- Search, deep links, back/forward, visualization selection, and theme persistence work in Playwright.
- Vite production build is Green and generated `dist/` is not committed.
- Existing Python tests/build/distribution/package identity remain Green.
- `trade_rl/**` production source remains untouched unless a separately justified requirement appears; such a discovery requires re-review rather than scope creep.
- Active spec/plan are removed after their durable rules are promoted.
- Exact final-head GitHub Actions `Lean Core` and `Human Guide` jobs are both Green.
