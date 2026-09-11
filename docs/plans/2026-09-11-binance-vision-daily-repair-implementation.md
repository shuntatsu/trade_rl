# Binance Vision Daily Repair Implementation Plan

Status: Active

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve the canonical M2 preregistration while repairing incomplete official Binance Vision monthly kline archives with explicitly frozen official daily archives.

**Architecture:** Keep the existing `vision-plan.json` as the primary pre-registered source plan. Canonical bootstrap adds a separate `vision-resolution.json` derived only from primary timestamp coverage, uses a bootstrap-specific frozen transport to merge primary + daily repair rows, and binds the resolution digest into bootstrap manifest v2 while retaining manifest v1 inspection compatibility.

**Tech Stack:** Python 3.12, NumPy, stdlib JSON/ZIP/CSV, pytest, Ruff, Mypy, uv, GitHub Actions.

**Spec:** `docs/specs/2026-09-11-binance-vision-daily-repair-design.md`

## Global Constraints

- Do not change the canonical M2 bootstrap config, symbol roster, periods, timeframes, features, thresholds, PPO budget, seeds, or experiment budget.
- Do not interpolate, forward-fill, or synthesize missing OHLC rows.
- Do not use REST kline data as repair evidence.
- Keep funding behavior unchanged.
- Keep ordinary `BinancePublicTransport` AUTO/VISION behavior unchanged.
- Derive repairs only from expected-vs-observed timestamps; never inspect price/return/P&L to choose repair sources.
- Do not hard-code XRP, 2022, or the observed missing dates in production code.
- Preserve `canonical_m2_vision_plan_v1` semantics.
- New bootstrap writer emits manifest v2; manifest v1 reader compatibility remains.
- After source synchronization, dataset build and inspection remain network-free.
- Active spec/plan are removed from current docs tree after durable contracts are promoted.

## Quality Contract

**Objective:** Make canonical source freeze deterministic and reproducible when an official monthly kline archive is internally incomplete but official daily archives contain the missing timestamps.

**Non-goals:** No strategy/simulation/risk changes, no general transport fallback, no research-condition changes, no final unused-future evaluation.

**Acceptance Criteria:**

1. Generic synthetic monthly gaps resolve to exact official daily URLs.
2. Complete primary series produce no repair URLs or repair downloads.
3. Primary + repair merge yields exactly one row per expected timestamp.
4. Dataset-semantic overlap disagreement fails closed.
5. Missing/incomplete daily repair fails closed.
6. Persisted resolution tamper, repair-byte tamper, sidecar tamper, and roster mismatch are rejected offline.
7. New artifacts use manifest v2 with `vision_resolution_digest`; v1 artifacts remain inspectable.
8. Public transport behavior and public package APIs remain unchanged.
9. Final exact HEAD and merge HEAD pass the repository hardened CI.
10. The unchanged canonical M2 config can be rerun; pristine bootstrap inspection must succeed before baseline executes.

**Invariants:** Primary plan/config digests remain stable; primary rows cannot be overwritten; repair URL set is deterministic; raw source roster contains every used primary/repair archive; inspection never uses network.

**Failure Modes:** Missing repair archive, still-missing repair timestamp, primary/repair semantic conflict, wrong daily URL, non-official URL, resolution tamper, raw-byte/sidecar tamper, v1/v2 schema confusion, hidden network fallback, study-specific hard-coding.

**Risk:** High for research provenance and reproducibility; medium for package runtime because changes are restricted to canonical bootstrap composition and manifest inspection.

**Test Oracle:** Exact timestamp coverage, exact deterministic resolution payload, semantic overlap comparison, raw-byte SHA/size evidence, manifest/content digests, network-call absence, unchanged public transport tests, full repository CI.

**Required Test Layers:** Unit, integration, regression, tamper/falsification, Ruff, Format, Mypy, architecture tooling, full pytest, build, distribution closure, clean-install smoke, package identity, real-data canonical bootstrap rerun.

**Quality Gate:** All Acceptance Criteria and failure-mode tests pass on exact feature HEAD and merge HEAD; post-merge real-data bootstrap succeeds with unchanged preregistration before baseline is allowed to run.

---

### Task 1: RED — generic primary-gap resolution contract

**Files:**
- Create: `tests/evaluation/experiments/bootstrap/test_binance_repair.py`
- Read: `trade_rl/evaluation/experiments/bootstrap/binance.py`
- Read: `trade_rl/integrations/binance/vision.py`
- Read: `trade_rl/integrations/binance/cache.py`

**Interfaces:**
- Consumes: existing `_freeze_binance_source`, `_inspect_frozen_binance_source`, `vision_cache_path`, official Vision URL builders.
- Produces test contract for future internal resolution helpers and frozen composite behavior.

- [ ] **Step 1: Add a reusable synthetic archive fixture with arbitrary gap dates**

Create a test fixture covering a non-study-specific symbol such as `TESTUSDT`, a one-month 1h range, and a missing UTC day chosen inside the range. The fake live transport must return a monthly archive lacking that day and a daily archive containing all bars for that day. The fixture records every requested URL and supplies matching synthetic exchange metadata.

- [ ] **Step 2: Add the primary RED behavior**

Assert that a future repaired freeze requests exactly the missing day's official daily URL, preserves `vision-plan.json` as the primary monthly plan, writes `vision-resolution.json`, and returns a composite transport whose merged clock is complete.

Run:

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap/test_binance_repair.py
```

Expected RED: fail because the current freeze accepts the incomplete primary archive without a resolution artifact or repair request, and the current composite transport later exposes the incomplete clock.

- [ ] **Step 3: Add no-repair and genericity RED cases**

Add a complete-primary case that asserts no daily URL is requested. Add a second gap on a different arbitrary day/timeframe so an implementation hard-coded to the observed XRP dates cannot pass.

- [ ] **Step 4: Add merge failure RED cases**

Add tests where a daily repair still omits one required timestamp and where overlapping primary/daily rows disagree in one dataset-semantic field. Both must fail closed.

- [ ] **Step 5: Commit RED only**

```bash
git add tests/evaluation/experiments/bootstrap/test_binance_repair.py
git commit -m "test: define Binance Vision daily repair contract"
```

Do not modify production source in this Task.

---

### Task 2: GREEN — source resolution and cache-only repair transport

**Files:**
- Modify: `trade_rl/evaluation/experiments/bootstrap/binance.py`
- Modify only if needed for parsing reuse: `trade_rl/integrations/binance/vision.py`
- Test: `tests/evaluation/experiments/bootstrap/test_binance_repair.py`
- Regression: `tests/evaluation/experiments/bootstrap/test_binance.py`
- Regression: `tests/integrations/test_binance.py`

**Interfaces:**
- Consumes: primary `BinanceVisionCachePlan`, cached archive sidecars, `plan_vision_kline_urls`, `vision_kline_url`, interval milliseconds, config symbol/timeframe ordering.
- Produces: internal deterministic resolution payload, resolution digest, and bootstrap-specific cache-only merged kline loader.

- [ ] **Step 1: Introduce internal resolution types/helpers**

In `bootstrap/binance.py`, add small internal value objects/helpers whose effective contract is:

```python
@dataclass(frozen=True, slots=True)
class _VisionRepair:
    symbol: str
    timeframe: str
    missing_open_ms: tuple[int, ...]
    daily_urls: tuple[str, ...]


def _resolve_vision_repairs(
    config: CanonicalM2BootstrapConfig,
    *,
    primary_plan: BinanceVisionCachePlan,
    cache_root: Path,
) -> tuple[_VisionRepair, ...]: ...
```

For each config symbol/timeframe, derive that series' primary kline URLs with the existing `plan_vision_kline_urls` authority rather than guessing by string parsing the mixed kline/funding plan. Read only validated cached primary kline bytes, compute the expected native clock, reject duplicate/backwards/out-of-range primary timestamps, and derive ordered daily URLs from missing UTC days.

- [ ] **Step 2: Persist canonical resolution payload**

Add `_vision_resolution_payload(...)` with schema `canonical_m2_vision_resolution_v1` and exact ordered fields from the spec. Its `primary_plan_digest` must equal `content_digest(_vision_plan_payload(config))`.

- [ ] **Step 3: Sync only resolved daily URLs**

After existing primary `sync_binance_vision_cache`, call existing `sync_binance_vision_urls` only for the ordered unique repair URLs. Do not broaden the primary plan or mutate `vision-plan.json`.

- [ ] **Step 4: Add bootstrap-specific merged kline loading**

Extend `_FrozenDatasetTransport` with the resolution mapping. Its `load_klines` must read primary cached rows through the existing cache-only transport, read repair daily archives only from cache, compare overlaps by normalized open timestamp + open/high/low/close/quote-volume semantics, retain primary rows, fill only missing expected timestamps, sort by open timestamp, and require exact complete native cadence before returning rows.

Do not change `BinancePublicTransport.load_klines` public behavior.

- [ ] **Step 5: Update existing bootstrap fake archives to remain valid complete-primary fixtures**

`tests/evaluation/experiments/bootstrap/test_binance.py::_payload_for_url` currently emits only two kline rows per monthly archive because the old freeze validated byte presence but not clock completeness. Change the test fixture—not the production acceptance rule—so normal no-repair tests provide a complete synthetic native clock for each planned month/timeframe. Keep existing assertions that a complete primary source performs no extra repair downloads.

- [ ] **Step 6: Bind repair bytes into source roster**

Build `raw_source_roster` from the ordered union `primary_plan.urls + repair_urls`. Keep roster item keys exactly `url / sha256 / size_bytes`.

- [ ] **Step 7: Make offline inspection recompute, not trust, resolution**

`_inspect_frozen_binance_source` must recompute missing timestamps/repair URLs from primary cached bytes, compare exact payload to saved `vision-resolution.json`, validate all repair bytes/sidecars, and instantiate the same network-cut merged transport.

- [ ] **Step 8: Extend `FrozenBinanceSource`**

Add:

```python
vision_resolution_payload: dict[str, object] | None
vision_resolution_digest: str | None
```

Newly frozen sources always populate them. Legacy primary-only inspection may use `None` only when explicitly requested by the v1 manifest path in Task 3.

- [ ] **Step 9: Verify targeted GREEN**

Run:

```bash
uv run pytest -q \
  tests/evaluation/experiments/bootstrap/test_binance_repair.py \
  tests/evaluation/experiments/bootstrap/test_binance.py \
  tests/integrations/test_binance.py
uv run ruff check trade_rl/evaluation/experiments/bootstrap/binance.py tests/evaluation/experiments/bootstrap/test_binance_repair.py
uv run ruff format --check trade_rl/evaluation/experiments/bootstrap/binance.py tests/evaluation/experiments/bootstrap/test_binance_repair.py
uv run mypy trade_rl/evaluation/experiments/bootstrap trade_rl/integrations/binance
```

Expected: all targeted tests/checks pass.

- [ ] **Step 10: Commit**

```bash
git add trade_rl/evaluation/experiments/bootstrap/binance.py trade_rl/integrations/binance/vision.py tests/evaluation/experiments/bootstrap/test_binance_repair.py tests/evaluation/experiments/bootstrap/test_binance.py tests/integrations/test_binance.py
git commit -m "fix: resolve incomplete Vision months with frozen daily evidence"
```

Stage only files actually changed.

---

### Task 3: RED/GREEN — bootstrap manifest v2 and v1 reader compatibility

**Files:**
- Modify: `trade_rl/evaluation/experiments/bootstrap/workflow.py`
- Modify: `tests/evaluation/experiments/bootstrap/test_workflow.py`
- Modify: `tests/evaluation/experiments/bootstrap/test_digest_validation.py`
- Modify: `tests/evaluation/experiments/bootstrap/test_tamper.py`

**Interfaces:**
- Consumes: `FrozenBinanceSource.vision_resolution_digest`, existing manifest/body digest contract.
- Produces: v2 writer plus explicit v1/v2 reader dispatch.

- [ ] **Step 1: Write RED tests for v2 writer**

Assert new bootstrap manifests have `schema_version == "canonical_m2_bootstrap_manifest_v2"`, contain lowercase SHA-256 `vision_resolution_digest`, and include it in `bootstrap_digest` calculation.

- [ ] **Step 2: Write RED v1 compatibility test**

Construct or transform a valid no-repair synthetic bootstrap into the exact legacy shape: manifest v1, no `vision_resolution_digest`, no `source/vision-resolution.json`, raw roster limited to primary plan URLs. Assert `inspect_canonical_m2_bootstrap` still accepts it without network.

- [ ] **Step 3: Write RED schema-confusion tests**

Reject v1 manifest + resolution file, v2 manifest without resolution file/digest, v2 resolution digest mismatch, and v2 roster missing a repair URL.

- [ ] **Step 4: Implement explicit schema dispatch**

Keep constants for v1 and v2 schemas/key sets. New `_manifest_body` writes v2 and requires non-`None` resolution digest. `_inspect_manifest` accepts only the two exact schemas and validates their exact key sets.

- [ ] **Step 5: Couple source inspection mode to manifest schema without network**

Adjust inspection ordering or parameters so v1 invokes legacy primary-only source inspection and v2 requires/recomputes resolution. Do not infer v1/v2 solely from file presence.

- [ ] **Step 6: Run targeted manifest/tamper suite**

```bash
uv run pytest -q tests/evaluation/experiments/bootstrap
uv run ruff check trade_rl/evaluation/experiments/bootstrap tests/evaluation/experiments/bootstrap
uv run ruff format --check trade_rl/evaluation/experiments/bootstrap tests/evaluation/experiments/bootstrap
uv run mypy trade_rl/evaluation/experiments/bootstrap
```

Expected: all bootstrap tests pass.

- [ ] **Step 7: Commit**

```bash
git add trade_rl/evaluation/experiments/bootstrap/workflow.py tests/evaluation/experiments/bootstrap
git commit -m "feat: bind Vision repair resolution into bootstrap manifest v2"
```

---

### Task 4: Falsification and regression hardening

**Files:**
- Modify: `tests/evaluation/experiments/bootstrap/test_binance_repair.py`
- Modify if needed: `tests/evaluation/experiments/bootstrap/test_tamper.py`
- No production changes unless a test exposes a real defect.

**Interfaces:**
- Consumes: completed Tasks 2–3.
- Produces: evidence that the tests detect plausible incorrect implementations.

- [ ] **Step 1: Add arbitrary-gap parameterization**

Parameterize at least two symbols, two native intervals, and multiple missing-day positions using synthetic archive fixtures. The expected repair URLs are computed from fixture dates, not copied from production output.

- [ ] **Step 2: Add request-boundary assertion**

Assert complete-primary cases request exactly the primary plan URLs, while incomplete cases request exactly primary URLs followed by the minimum ordered repair URL set; no REST kline URL may occur.

- [ ] **Step 3: Add tamper assertions**

Mutate `missing_open_ms`, repair URL order, repair raw bytes, repair sidecar digest, and raw roster membership independently; each must fail inspection.

- [ ] **Step 4: Add explicit offline assertion**

After freeze, monkeypatch `urllib.request.urlopen` to fail and verify source inspection plus resolved kline load still succeeds from cache.

- [ ] **Step 5: Run mutation/falsification probes**

Temporarily test at least these incorrect variants without committing them: ignore resolution digest, ignore overlap conflict, omit repair URL from roster, always request one daily URL even when primary complete. Confirm each is killed by a maintained test, then restore production source.

- [ ] **Step 6: Commit only maintained tests if changed**

```bash
git add tests/evaluation/experiments/bootstrap
git commit -m "test: harden frozen Vision repair evidence"
```

---

### Task 5: Durable documentation and active-doc cleanup

**Files:**
- Modify: `docs/architecture/controlled-experiment-loop.md`
- Modify: `docs/README.md`
- Delete: `docs/specs/2026-09-11-binance-vision-daily-repair-design.md`
- Delete: `docs/plans/2026-09-11-binance-vision-daily-repair-implementation.md`

**Interfaces:**
- Consumes: implemented source/manifest contracts.
- Produces: current-only durable documentation.

- [ ] **Step 1: Document durable source-freeze contract**

Add the primary-plan + explicit-resolution + daily-repair + v1/v2 manifest rules to `controlled-experiment-loop.md`, including that repair choice is timestamp-only and inspection is network-free.

- [ ] **Step 2: Remove Active spec/plan from the current tree**

Delete both ephemeral docs after durable text is present.

- [ ] **Step 3: Restore docs index**

Change `docs/README.md` back to `現在Activeなspec/planはない。` unless another concurrent active doc merged into the current base; if so preserve that unrelated active entry rather than deleting it.

- [ ] **Step 4: Verify docs contracts**

```bash
uv run pytest -q tests/architecture/test_current_docs_layout.py
```

Expected: pass, with no stale completed spec/plan retained.

- [ ] **Step 5: Commit**

```bash
git add docs tests/architecture/test_current_docs_layout.py
git commit -m "docs: record resolved Binance source evidence contract"
```

Stage the architecture test only if it actually changes.

---

### Task 6: Final feature verification, review, and integration

**Files:**
- Review all feature diff against current `main`.

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces: mergeable PR evidence.

- [ ] **Step 1: Reconcile current `main` before final CI**

Re-read open PRs and main HEAD. If unrelated PRs such as simulation SHA authority or Agent control-plane have merged, merge current main into this feature branch without force-push/history rewrite and rerun affected checks.

- [ ] **Step 2: Run exact final hardened verification**

```bash
uv run ruff check trade_rl tests
uv run ruff format --check trade_rl tests
uv run mypy trade_rl
uv run mypy tests/architecture/imports.py tests/architecture/distribution.py
uv run pytest -q tests
uv build
uv run python -m tests.architecture.distribution dist/*.tar.gz dist/*.whl
```

Also rebuild a wheel from the sdist, run the existing clean non-editable install/public-import/CLI smoke, and verify package identity exactly as `.github/workflows/ci.yml` does.

- [ ] **Step 3: Self-review and independent/falsification review**

Review requirement compliance, source authority, manifest compatibility, tamper handling, network cut, public API, diff scope, dead helpers, and test assertion quality. Because no independent subagent executor is available in this session, reconstruct the review from the original spec and observable diff rather than reusing implementation conclusions.

- [ ] **Step 4: Update PR evidence and merge only exact verified head**

Mark Ready only after final CI passes. Merge using expected head SHA. Then verify the merge commit itself with the `main` push CI before proceeding to research execution.

---

### Task 7: Resume canonical M2 on the unchanged preregistration

**Files/refs:**
- Branch: `research/m2-canonical-study-001`
- Frozen config: `studies/m2/canonical-study-001/bootstrap.json`
- Do not edit the config.

**Interfaces:**
- Consumes: merged production fix on `main`.
- Produces: immutable real-data bootstrap evidence, then baseline EvidenceSet only after bootstrap inspection passes.

- [ ] **Step 1: Merge new main into the research branch without rewriting preregistration history**

Preserve the original preregistration commit as ancestry. Do not rebase or force-push.

- [ ] **Step 2: Reassert preregistration identity before data execution**

Require raw bootstrap SHA-256:

```text
9c75849672fe56004261b540c84cdbb003a8821b94dd2f80f8bbe3e8a69f2eb4
```

and parsed config digest:

```text
71b72031d33b50468774e6bc355206725d6a85ea2a5680fa8e5c8b9c385aee5b
```

- [ ] **Step 3: Run full pre-execution repository gate**

Require Ruff, Format, Mypy, architecture tooling, full pytest, build, distribution closure, clean install, and package identity on the exact evidence tree before research execution.

- [ ] **Step 4: Execute canonical bootstrap**

Run the unchanged bootstrap config. Require persisted `vision-resolution.json`, manifest v2, primary+repair raw source roster, and successful network-free `inspect_canonical_m2_bootstrap`.

- [ ] **Step 5: Verify repair evidence before looking at research outcomes**

Confirm repair decisions correspond only to timestamp gaps, raw source hashes/sizes are recorded, and source resolution reproduces offline. Do not change the config based on repaired prices or strategy outcomes.

- [ ] **Step 6: Execute baseline only after pristine bootstrap success**

Copy the pristine `study/`, run `run_baseline`, reconstruct with `inspect_study`, and persist an index binding execution Git SHA, main base SHA, preregistration SHA, config/bootstrap/dataset/study digests, baseline evidence fingerprint, source-preflight run ID, and metadata-probe run ID.

- [ ] **Step 7: Record current research status without claiming a winner**

Update the study README / `docs/research/current-status.md` only with observed bootstrap/baseline execution status and immutable evidence identifiers. Do not claim profitability, winner, or Production authorization until the controlled development comparison is complete.
