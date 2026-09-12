# Interactive Human Guide Pages v1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the existing non-authoritative Interactive Guide through GitHub Pages only after a successful same-repository `main` push CI, then verify the deployed public site without changing Trade RL runtime/research semantics.

**Architecture:** Keep `.github/workflows/ci.yml` as the quality gate. Add a separate `workflow_run` deployment workflow that binds checkout to the successful CI run's exact `head_sha`, builds only `guide/`, uploads only `guide/dist/`, deploys through the official GitHub Pages actions, and runs a public Chromium smoke against the deployed URL. GitHub Pages source enablement remains an external one-time repository setting and is never inferred from Git-tree state.

**Tech Stack:** GitHub Actions, GitHub Pages, Node 24.21.0, React 19.3, Vite 8.3, Playwright 1.63, Python/pytest architecture contract tests.

**Spec:** `docs/specs/2026-09-13-interactive-guide-pages-v1.md`

## Global Constraints

- `docs/` remains the technical/research authority; `guide/` remains non-authoritative.
- No `trade_rl/**` runtime/public API/research/artifact semantics change.
- No `gh-pages` branch, PAT, deploy key, analytics, backend, custom domain, or database.
- Deployment eligibility requires successful same-repository `main` **push** CI and exact `workflow_run.head_sha` checkout.
- PR/fork/failed/cancelled CI must never receive Pages write deployment.
- Publish only `guide/dist/`.
- Keep current Lean Core and Human Guide CI coverage unchanged.
- GitHub Pages official actions are pinned to exact commits: `actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d` (v6.0.0), `actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9` (v5.0.0), `actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346` (v5.0.1).
- Reuse existing repository pins for checkout and Node setup: `actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5`, `actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020`.
- Pages enablement must be read back from GitHub state; workflow presence alone is not publication proof.
- After successful deployment, the public browser smoke must verify root render, hash navigation/search, theme toggle, source link, and 320px no-overflow.

---

### Task 1: Repair the over-scoped Coordination docs oracle

**Files:**
- Modify: `tests/architecture/test_agent_coordination_boundary.py`
- Verify: `docs/README.md`

**Interfaces:**
- Consumes: current-only docs policy and the active Pages spec/plan.
- Produces: a Coordination-specific completion oracle that allows unrelated legitimate Active work.

- [ ] **Step 1: Reproduce the existing RED**

Run:

```bash
uv run pytest -q tests/architecture/test_agent_coordination_boundary.py::test_completed_coordination_ephemeral_docs_are_removed
```

Expected: FAIL because the test globally requires `現在Activeなspec / planはない` while the approved Pages spec/plan are legitimately Active.

- [ ] **Step 2: Apply the narrow oracle fix**

Change the end of `test_completed_coordination_ephemeral_docs_are_removed` from:

```python
index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
assert "現在Activeなspec / planはない" in index
for path in completed:
    assert path.name not in index
```

to:

```python
index = (ROOT / "docs" / "README.md").read_text(encoding="utf-8")
for path in completed:
    assert path.name not in index
```

Do not alter the assertions that the completed Coordination Plane spec/plan files are absent.

- [ ] **Step 3: Verify GREEN**

Run:

```bash
uv run pytest -q tests/architecture/test_agent_coordination_boundary.py
```

Expected: all tests in that file PASS.

- [ ] **Step 4: Commit**

```bash
git add tests/architecture/test_agent_coordination_boundary.py
git commit -m "test: scope coordination docs cleanup oracle"
```

---

### Task 2: Freeze the Pages deployment contract with a failing architecture test

**Files:**
- Create: `tests/architecture/test_guide_pages_deployment.py`
- Expected-missing implementation: `.github/workflows/deploy-guide.yml`
- Expected-missing public smoke: `guide/playwright.public.config.ts`, `guide/e2e/public-smoke.spec.ts`

**Interfaces:**
- Consumes: exact action pins and workflow security contract from the spec.
- Produces: static fail-closed tests for deployment trigger, exact SHA binding, permissions, artifact scope, Pages action pins, and public-smoke wiring.

- [ ] **Step 1: Add the RED contract test**

Create `tests/architecture/test_guide_pages_deployment.py`:

```python
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "deploy-guide.yml"


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_pages_deployment_is_bound_to_successful_same_repo_main_push_ci() -> None:
    text = _workflow()
    for required in (
        "workflow_run:",
        'workflows: ["CI"]',
        "types: [completed]",
        "github.event.workflow_run.conclusion == 'success'",
        "github.event.workflow_run.event == 'push'",
        "github.event.workflow_run.head_branch == 'main'",
        "github.event.workflow_run.head_repository.full_name == github.repository",
        "ref: ${{ github.event.workflow_run.head_sha }}",
    ):
        assert required in text


def test_pages_deployment_has_minimal_permissions_and_exact_pins() -> None:
    text = _workflow()
    for required in (
        "permissions: {}",
        "contents: read",
        "pages: write",
        "id-token: write",
        "actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5",
        "actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020",
        "actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d",
        "actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9",
        "actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346",
    ):
        assert required in text


def test_pages_artifact_and_public_smoke_are_guide_only() -> None:
    text = _workflow()
    assert "path: guide/dist" in text
    assert "npm run source-check" in text
    assert "npm run build" in text
    assert "GUIDE_PUBLIC_URL:" in text
    assert "npm run e2e:public" in text
    assert (ROOT / "guide" / "playwright.public.config.ts").is_file()
    assert (ROOT / "guide" / "e2e" / "public-smoke.spec.ts").is_file()
```

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run pytest -q tests/architecture/test_guide_pages_deployment.py
```

Expected: FAIL because `deploy-guide.yml` and public-smoke files do not exist.

- [ ] **Step 3: Commit the RED test only**

```bash
git add tests/architecture/test_guide_pages_deployment.py
git commit -m "test: define Guide Pages deployment contract"
```

---

### Task 3: Implement the exact-SHA Pages workflow and public-site smoke

**Files:**
- Create: `.github/workflows/deploy-guide.yml`
- Create: `guide/playwright.public.config.ts`
- Create: `guide/e2e/public-smoke.spec.ts`
- Modify: `guide/package.json`

**Interfaces:**
- Consumes: successful `CI` workflow-run payload, `workflow_run.head_sha`, existing Guide build/source-check scripts.
- Produces: Pages artifact, deployment `page_url`, and `npm run e2e:public` public-site oracle.

- [ ] **Step 1: Add the deployment workflow**

Create `.github/workflows/deploy-guide.yml` with this structure:

```yaml
name: Deploy Human Guide

on:
  workflow_run:
    workflows: ["CI"]
    types: [completed]

permissions: {}

concurrency:
  group: pages
  cancel-in-progress: false

jobs:
  build:
    if: >-
      github.event.workflow_run.conclusion == 'success' &&
      github.event.workflow_run.event == 'push' &&
      github.event.workflow_run.head_branch == 'main' &&
      github.event.workflow_run.head_repository.full_name == github.repository
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Checkout verified main SHA
        uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
          persist-credentials: false
      - name: Set up Node
        uses: actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020
        with:
          node-version: "24.21.0"
          cache: npm
          cache-dependency-path: guide/package-lock.json
      - name: Install Guide dependencies
        working-directory: guide
        run: npm ci
      - name: Verify Guide source binding
        working-directory: guide
        run: npm run source-check
      - name: Build Guide
        working-directory: guide
        run: npm run build
      - name: Upload Pages artifact
        uses: actions/upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9
        with:
          path: guide/dist

  deploy:
    needs: build
    runs-on: ubuntu-latest
    permissions:
      contents: read
      pages: write
      id-token: write
    environment:
      name: github-pages
      url: ${{ steps.deployment.outputs.page_url }}
    outputs:
      page_url: ${{ steps.deployment.outputs.page_url }}
    steps:
      - name: Configure Pages metadata
        uses: actions/configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d
      - name: Deploy Pages artifact
        id: deployment
        uses: actions/deploy-pages@368f82528645a54fb793d4d04e342629a3f51346

  smoke:
    needs: deploy
    runs-on: ubuntu-latest
    permissions:
      contents: read
    steps:
      - name: Checkout deployed SHA
        uses: actions/checkout@34e114876b0b11c390a56381ad16ebd13914f8d5
        with:
          ref: ${{ github.event.workflow_run.head_sha }}
          persist-credentials: false
      - name: Set up Node
        uses: actions/setup-node@49933ea5288caeca8642d1e84afbd3f7d6820020
        with:
          node-version: "24.21.0"
          cache: npm
          cache-dependency-path: guide/package-lock.json
      - name: Install Guide dependencies
        working-directory: guide
        run: npm ci
      - name: Install Chromium
        working-directory: guide
        run: npx playwright install --with-deps chromium
      - name: Smoke public Guide
        working-directory: guide
        env:
          GUIDE_PUBLIC_URL: ${{ needs.deploy.outputs.page_url }}
        run: npm run e2e:public
```

`configure-pages` must use default `enablement: false`; do not add PAT or administration secret.

- [ ] **Step 2: Add public Playwright config**

Create `guide/playwright.public.config.ts`:

```ts
import { defineConfig, devices } from "@playwright/test";

const baseURL = process.env.GUIDE_PUBLIC_URL;
if (!baseURL) {
  throw new Error("GUIDE_PUBLIC_URL is required for public Guide smoke tests");
}

export default defineConfig({
  testDir: "./e2e",
  testMatch: "public-smoke.spec.ts",
  fullyParallel: true,
  forbidOnly: true,
  retries: 1,
  reporter: "list",
  use: {
    baseURL,
    trace: "retain-on-failure",
    screenshot: "only-on-failure",
  },
  projects: [
    {
      name: "public-desktop",
      use: { ...devices["Desktop Chrome"], viewport: { width: 1440, height: 1000 } },
    },
    {
      name: "public-mobile-320",
      use: { ...devices["Desktop Chrome"], viewport: { width: 320, height: 800 } },
    },
  ],
});
```

No `webServer` is allowed in this config.

- [ ] **Step 3: Add the public critical-journey smoke**

Create `guide/e2e/public-smoke.spec.ts`:

```ts
import { expect, test } from "@playwright/test";


test("published Guide serves the critical human journey", async ({ page }) => {
  const response = await page.goto("/#overview");
  expect(response?.ok()).toBeTruthy();
  await expect(
    page.getByRole("heading", {
      level: 1,
      name: "実データで動く、検証可能なトレーディングRLシステム",
    }),
  ).toBeVisible();

  await page.keyboard.press("Control+K");
  const search = page.getByRole("textbox", { name: "ガイドを検索" });
  await search.fill("手数料");
  await page.getByRole("button", { name: /実行コスト/ }).click();
  await expect(page).toHaveURL(/#execution-economics$/);

  await page.getByRole("button", { name: "ダークモードへ" }).click();
  await expect(page.locator("html")).toHaveClass(/dark/);

  const source = page.getByRole("link", { name: /current-status\.md/ }).first();
  await expect(source).toHaveAttribute("href", /github\.com\/shuntatsu\/trade_rl\/blob\/main\/docs\//);
});


test("published Guide has no 320px horizontal overflow", async ({ page }, testInfo) => {
  test.skip(testInfo.project.name !== "public-mobile-320", "mobile project only");
  await page.goto("/#ppo-observation-v2");
  const sizes = await page.evaluate(() => ({
    scrollWidth: document.documentElement.scrollWidth,
    clientWidth: document.documentElement.clientWidth,
  }));
  expect(sizes.scrollWidth).toBeLessThanOrEqual(sizes.clientWidth + 1);
});
```

- [ ] **Step 4: Wire the package script**

Add to `guide/package.json` scripts:

```json
"e2e:public": "playwright test --config playwright.public.config.ts"
```

Do not add it to `npm run check`; it requires a real deployed URL and belongs only to post-deploy verification.

- [ ] **Step 5: Run targeted GREEN checks**

Run:

```bash
uv run pytest -q tests/architecture/test_guide_pages_deployment.py
cd guide && npm run check
```

Expected: both PASS. Do not try to fake `GUIDE_PUBLIC_URL` with the local Vite server as proof of publication.

- [ ] **Step 6: Commit**

```bash
git add .github/workflows/deploy-guide.yml guide/playwright.public.config.ts guide/e2e/public-smoke.spec.ts guide/package.json tests/architecture/test_guide_pages_deployment.py
git commit -m "feat: deploy Interactive Guide with GitHub Pages"
```

---

### Task 4: Document the deployment lifecycle without claiming publication early

**Files:**
- Modify: `guide/README.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/README.md`
- Keep Active until post-deploy completion: `docs/specs/2026-09-13-interactive-guide-pages-v1.md`, `docs/plans/2026-09-13-interactive-guide-pages-v1.md`

**Interfaces:**
- Consumes: deployment workflow contract from Task 3.
- Produces: durable maintenance guidance while external Pages enablement is pending.

- [ ] **Step 1: Add durable Guide deployment instructions**

In `guide/README.md`, add a `## GitHub Pages deployment` section that states:

```text
- ci.yml remains the quality gate.
- deploy-guide.yml only deploys a successful same-repository main push CI head SHA.
- Pages source must be GitHub Actions in repository Settings.
- Generated dist/ remains untracked.
- Public-smoke uses GUIDE_PUBLIC_URL and does not replace local E2E.
```

Do not state that the public URL is live until the post-merge oracle proves it.

- [ ] **Step 2: Add Agent routing**

In `docs/AGENTS.md`, add one update-matrix row:

```markdown
| Guide Pages deployment / public URL | `.github/workflows/deploy-guide.yml`, `guide/README.md`, Pages state read-back |
```

- [ ] **Step 3: Keep docs index truthful**

`docs/README.md` must list both Active Pages artifacts while external verification remains pending:

```markdown
- `specs/2026-09-13-interactive-guide-pages-v1.md`
- `plans/2026-09-13-interactive-guide-pages-v1.md`
```

It must not say `現在Activeなspec / planはない` until final cleanup.

- [ ] **Step 4: Run docs/architecture checks**

Run:

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py tests/architecture/test_agent_coordination_boundary.py tests/architecture/test_guide_pages_deployment.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add guide/README.md docs/AGENTS.md docs/README.md docs/specs/2026-09-13-interactive-guide-pages-v1.md docs/plans/2026-09-13-interactive-guide-pages-v1.md
git commit -m "docs: define Guide Pages deployment lifecycle"
```

---

### Task 5: Finalize the implementation PR to the external-enablement gate

**Files:**
- Review all implementation diff.
- No main merge in this task.

**Interfaces:**
- Consumes: Tasks 1-4.
- Produces: a Ready-for-review implementation PR whose only remaining external blocker is Pages source enablement and later merge permission.

- [ ] **Step 1: Synchronize current main if it moved**

Before final CI, compare current `main` to the implementation HEAD. If #499 or any other PR has landed, non-force merge current main into the feature branch, resolve overlapping docs/tests by preserving both contracts, and rerun source freshness. Never reuse pre-sync Green evidence.

- [ ] **Step 2: Run the full permanent quality gate**

Required exact-head commands/oracles:

```bash
uv run ruff check trade_rl tests tools
uv run ruff format --check trade_rl tests tools
uv run mypy trade_rl
uv run mypy tools/agent_repo tests/architecture/distribution.py
uv run pytest -q tests
uv build
cd guide && npm run check && npm run e2e
```

Then require the permanent GitHub Actions `CI` workflow on the same final HEAD to complete successfully.

- [ ] **Step 3: Review the final diff and security contract**

Verify:

```text
- no trade_rl/** semantic changes
- no PAT/deploy key/secret additions
- no gh-pages branch
- deploy condition includes success + push + main + same repository
- checkout binds exact workflow_run.head_sha
- Pages write/id-token permissions exist only on deploy job
- artifact path is guide/dist
- Pages action SHAs match the pinned stable releases
- public smoke has no local webServer fallback
- Active spec/plan remain because publication is not yet externally verified
```

- [ ] **Step 4: Mark the implementation PR ready only if exact-head CI is Green**

Record current main SHA, final PR HEAD SHA, CI run ID, test counts, unresolved review threads, and the external Pages-setting blocker.

---

### Task 6: Enable Pages, merge, prove publication, then clean Active docs

**Files / external state:**
- External GitHub state: Settings → Pages → Source: GitHub Actions
- Post-merge: `main` CI + `deploy-guide.yml`
- Cleanup follow-up: remove Active spec/plan and update `guide/README.md`, `docs/README.md`, optional root `README.md` public Guide link.

**Interfaces:**
- Consumes: Ready implementation PR plus explicit user merge permission and Pages source enablement.
- Produces: a publicly reachable Guide and a current-only durable docs tree.

- [ ] **Step 1: Read back Pages state before claiming publication**

After the user enables `Settings → Pages → Source: GitHub Actions`, fetch repository metadata/API state. Require Pages to report enabled before treating deployment failure as a code defect.

- [ ] **Step 2: Merge only with explicit user permission**

Immediately before merge, re-read current main, PR head, exact-head CI, mergeability and review threads. Merge with `expected_head_sha` only if the tested PR head still contains current main.

- [ ] **Step 3: Verify post-merge main CI and deployment**

Require:

```text
main merge SHA -> CI success
that CI exact head SHA -> deploy-guide build success
Pages deploy -> success
smoke -> success
```

Record the deployment run ID, deployment URL and artifact/deployment evidence.

- [ ] **Step 4: Verify public URL independently**

Require `https://shuntatsu.github.io/trade_rl/` to return HTTP success and the workflow public smoke to pass root/hash/theme/source/mobile assertions.

- [ ] **Step 5: Create the current-only cleanup follow-up**

Remove:

```text
docs/specs/2026-09-13-interactive-guide-pages-v1.md
docs/plans/2026-09-13-interactive-guide-pages-v1.md
```

Update durable docs so `docs/README.md` again states no Active spec/plan. Add the public Guide URL to `guide/README.md` and root `README.md` only now, after public reachability is proven.

Run full docs/Guide tests and exact-head CI on the cleanup PR; merge that cleanup only with explicit user permission.

## Plan Self-Review

- Spec coverage: trigger security, exact SHA binding, action pinning, permissions, artifact scope, one-time Pages setting, public browser smoke, post-merge oracle and current-only cleanup are all assigned to Tasks 2-6.
- Failure-mode coverage: Pages disabled, stale source fingerprint, CI failure/cancellation, deployment failure and main movement are fail-closed and keep the last published site intact.
- Type/interface consistency: `GUIDE_PUBLIC_URL` is the only public-smoke input; deployment `page_url` is the workflow output passed to it.
- No placeholders/TODOs remain.
- #499 overlap is explicitly handled by final-main synchronization; the narrow Coordination test fix must converge rather than be duplicated if #499 lands first.
