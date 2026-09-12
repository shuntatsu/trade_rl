# Interactive Human Guide Pages v1 Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Publish the existing non-authoritative Interactive Guide through GitHub Pages only after a successful same-repository `main` push CI, then verify the deployed site without changing Trade RL runtime or research semantics.

**Architecture:** Keep `.github/workflows/ci.yml` as the quality gate. Add a separate `workflow_run` Pages workflow bound to the successful CI run's exact `head_sha`, publish only `guide/dist/`, deploy with official GitHub Pages actions pinned to immutable SHAs, then run a public Chromium smoke against the deployment URL. Pages source enablement remains external GitHub state.

**Tech Stack:** GitHub Actions, GitHub Pages, Node 24.21.0, React 19.3, Vite 8.3, Playwright 1.63, pytest architecture contract tests.

**Spec:** `docs/specs/2026-09-13-interactive-guide-pages-v1.md`

## Global Constraints

- `docs/` remains authoritative; `guide/` remains explanatory only.
- No `trade_rl/**` runtime/public API/research/artifact semantic changes.
- No `gh-pages` branch, PAT, deploy key, analytics, backend, database, or custom domain.
- Deployment requires successful same-repository `main` **push** CI and exact `workflow_run.head_sha` checkout.
- PR/fork/failed/cancelled CI must never deploy.
- Publish only `guide/dist/`.
- Keep existing Lean Core and Human Guide gates intact.
- Pin Pages actions exactly: `configure-pages@45bfe0192ca1faeb007ade9deae92b16b8254a0d`, `upload-pages-artifact@fc324d3547104276b827a68afc52ff2a11cc49c9`, `deploy-pages@368f82528645a54fb793d4d04e342629a3f51346`.
- Reuse existing checkout/setup-node pins.
- Publication is not complete until Pages state, deploy run, URL reachability and public browser smoke are verified.

---

### Task 1: Scope the Coordination cleanup oracle

**Files:**
- Modify: `tests/architecture/test_agent_coordination_boundary.py`

- [ ] Reproduce the existing failure caused by the global `現在Activeなspec / planはない` assertion.
- [ ] Remove only that global assertion; keep checks that the completed Coordination spec/plan files are absent and unindexed.
- [ ] Run `uv run pytest -q tests/architecture/test_agent_coordination_boundary.py` and require PASS.

### Task 2: Define the Pages deployment contract before implementation

**Files:**
- Create: `tests/architecture/test_guide_pages_deployment.py`

- [ ] Assert `workflow_run` watches workflow `CI` and `completed`.
- [ ] Assert eligibility includes `conclusion == success`, `event == push`, `head_branch == main`, same repository, exact `workflow_run.head_sha` checkout.
- [ ] Assert workflow-level `permissions: {}` and job-local `contents: read`, `pages: write`, `id-token: write`.
- [ ] Assert exact immutable action SHAs and `guide/dist` artifact path.
- [ ] Assert public-smoke wiring exists.
- [ ] Run the test before implementation and require RED because the workflow/smoke files do not yet exist.

### Task 3: Implement Pages deployment and public smoke

**Files:**
- Create: `.github/workflows/deploy-guide.yml`
- Create: `guide/playwright.public.config.ts`
- Create: `guide/e2e/public-smoke.spec.ts`
- Modify: `guide/package.json`

- [ ] Add a `workflow_run` workflow named `Deploy Human Guide`.
- [ ] Build job: exact verified SHA checkout, Node 24.21.0, `npm ci`, `npm run source-check`, `npm run build`, upload only `guide/dist`.
- [ ] Deploy job: `github-pages` environment, `configure-pages` with default `enablement=false`, then `deploy-pages`; grant Pages/OIDC permissions only here.
- [ ] Smoke job: exact deployed SHA checkout, install Node/Chromium, set `GUIDE_PUBLIC_URL` from deployment `page_url`, run `npm run e2e:public`.
- [ ] Public Playwright config must require `GUIDE_PUBLIC_URL`, use no `webServer`, and cover desktop + 320px Chromium.
- [ ] Public smoke must verify HTTP/root rendering, search/hash navigation, theme toggle, GitHub source link and mobile no-overflow.
- [ ] Add `"e2e:public": "playwright test --config playwright.public.config.ts"` without adding it to local `npm run check`.
- [ ] Run `uv run pytest -q tests/architecture/test_guide_pages_deployment.py` and `cd guide && npm run check`; require PASS.

### Task 4: Document durable deployment maintenance

**Files:**
- Modify: `guide/README.md`
- Modify: `docs/AGENTS.md`
- Modify: `docs/README.md`

- [ ] Document that `ci.yml` remains the quality gate and `deploy-guide.yml` only deploys verified same-repository main-push CI SHAs.
- [ ] Document one-time Settings → Pages → Source: GitHub Actions requirement.
- [ ] Document that `dist/` remains generated/untracked and public smoke does not replace local E2E.
- [ ] Add Agent update-matrix routing for Pages deployment/public URL.
- [ ] Keep the Active spec and plan indexed until external publication verification is complete.
- [ ] Run current docs + Coordination + Pages deployment architecture tests; require PASS.

### Task 5: Finalize implementation PR to the external Pages-setting gate

- [ ] If `main` moved, non-force synchronize and rerun all evidence on the new HEAD; never reuse stale Green.
- [ ] Run full permanent gates: Ruff, format, production Mypy, repository-tooling Mypy, full pytest, build/distribution/clean-install, `npm run check`, existing Guide E2E, then exact-head GitHub Actions CI.
- [ ] Review final diff for no `trade_rl/**` semantic changes, no secrets/PAT, no `gh-pages` branch, exact action pins, exact-SHA binding, minimal permissions, `guide/dist` artifact scope and no public-smoke local-server fallback.
- [ ] Keep PR Draft until exact-head CI is Green and the only remaining blocker is external Pages enablement.

### Task 6: Enable Pages, merge, verify public deployment, then clean Active docs

- [ ] User enables Settings → Pages → Build and deployment → Source: GitHub Actions because the available connector has no Pages-setting write surface.
- [ ] Read back Pages state; do not claim publication from workflow files alone.
- [ ] Merge only with explicit user permission, after rechecking current main, PR HEAD, mergeability, reviews and exact-head CI.
- [ ] Require merged-main CI success, then `Deploy Human Guide` success for that exact merge SHA.
- [ ] Require public smoke success and HTTP reachability at `https://shuntatsu.github.io/trade_rl/`.
- [ ] After publication proof, remove this Active spec/plan in a cleanup follow-up, restore `docs/README.md` to no Active work, and add the proven public URL to durable Guide/root documentation.

## Plan Self-Review

- Spec coverage includes trigger security, exact SHA binding, permissions, action pinning, artifact scope, external Pages state, public browser smoke, post-merge oracle and current-only cleanup.
- Pages-disabled, stale source fingerprint, failed/cancelled CI, deploy outage and moving-main failure modes remain fail-closed.
- #499 overlap is handled by mandatory final-main synchronization; the narrow Coordination test fix converges if #499 lands first.
- No placeholders or TODOs remain.
