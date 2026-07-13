# Residual Walk-Forward Architecture Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the residual nested Walk-Forward path dependency-correct, atomic, leak-resistant, provenance-bound, and correctly aggregated without changing legacy direct-weight Walk-Forward behavior.

**Architecture:** Keep fold/statistical primitives in `mars_lite.eval`, move orchestration to `mars_lite.pipeline`, make `residual_candidates` the single A/B/C/D implementation, split checkpoint validation from configuration selection, publish artifacts transactionally, and compute stitched OOS metrics from base-bar return series.

**Tech Stack:** Python 3.12, NumPy, dataclasses, Stable-Baselines3 PPO, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- `release_eligible` remains `false`.
- Residual Registry registration remains fail-closed.
- Legacy direct Walk-Forward dispatch and output names remain unchanged.
- No `mars_lite.eval` module may import `mars_lite.pipeline`.
- At least two residual folds must complete before publishing success.
- All JSON writes use `allow_nan=False`.
- `argparse.Namespace` is not mutated after configuration resolution.
- The direct 3-fold × 3-seed evidence remains diagnostic only: average returns and DSR do not authorize production.

---

### Task 1: Pure fold planning and stitched OOS statistics

**Files:**
- Modify: `mars_lite/eval/residual_walk_forward.py`
- Modify: `mars_lite/eval/relative_evaluation.py`
- Test: `tests/test_residual_walk_forward.py`
- Create: `tests/test_residual_walk_forward_architecture.py`

**Interfaces:**
- Produces `ResidualFoldSpec`, `RelativeFoldSeries`, `build_residual_fold_specs`, `stitch_relative_fold_results`, and `summarize_residual_folds`.
- `mars_lite.eval.residual_walk_forward` imports no `mars_lite.pipeline` modules.

- [ ] **Step 1: Write failing four-segment fold tests**

```python
spec = build_residual_fold_specs(
    n_bars=4_000,
    n_folds=3,
    purge_bars=24,
    horizon=12,
)[0][0]
assert spec.policy_train_end < spec.checkpoint_validation_start
assert spec.checkpoint_validation_end < spec.configuration_selection_start
assert spec.configuration_selection_end < spec.outer_test_start
assert spec.purge_bars == 24
```

- [ ] **Step 2: Write failing dependency-direction test**

```python
source = Path("mars_lite/eval/residual_walk_forward.py").read_text()
tree = ast.parse(source)
for node in ast.walk(tree):
    if isinstance(node, ast.ImportFrom):
        assert not (node.module or "").startswith("mars_lite.pipeline")
```

- [ ] **Step 3: Write failing stitched-series test**

Construct two non-overlapping fold return arrays and assert stitched total return equals `np.expm1(np.log1p(all_returns).sum())`, total bars equal the concatenated length, and bootstrap uses the concatenated hybrid-minus-shadow series.

- [ ] **Step 4: Run tests and confirm RED**

```bash
uv run pytest -q tests/test_residual_walk_forward.py tests/test_residual_walk_forward_architecture.py
```

- [ ] **Step 5: Implement pure dataclasses and functions**

```python
@dataclass(frozen=True)
class ResidualFoldSpec:
    fold: int
    policy_train_start: int
    policy_train_end: int
    checkpoint_validation_start: int
    checkpoint_validation_end: int
    configuration_selection_start: int
    configuration_selection_end: int
    outer_test_start: int
    outer_test_end: int
    purge_bars: int

@dataclass(frozen=True)
class RelativeFoldSeries:
    fold: int
    hybrid_returns: np.ndarray
    shadow_returns: np.ndarray
    hybrid_trades: int
    shadow_trades: int
    hybrid_turnover: float
    shadow_turnover: float
    hybrid_cost: float
    shadow_cost: float
```

Use 70/15/15 development proportions separated by effective purge and return skipped-fold diagnostics.

- [ ] **Step 6: Expose base-bar returns from relative evaluation**

Add `include_return_series: bool = False` to `evaluate_relative_agent`. When true, return JSON-safe `hybrid_base_bar_returns` and `shadow_base_bar_returns`; leave the default payload unchanged.

- [ ] **Step 7: Run tests and confirm GREEN**

- [ ] **Step 8: Commit**

```bash
git add mars_lite/eval/residual_walk_forward.py mars_lite/eval/relative_evaluation.py tests/test_residual_walk_forward.py tests/test_residual_walk_forward_architecture.py
git commit -m "refactor: make residual walk-forward evaluation pure"
```

---

### Task 2: Authoritative candidate training and separate validation roles

**Files:**
- Modify: `mars_lite/pipeline/residual_candidates.py`
- Modify: `mars_lite/pipeline/residual_pipeline.py`
- Test: `tests/test_residual_candidate_training.py`
- Create: `tests/test_residual_pipeline_shared_candidates.py`

**Interfaces:**

```python
train_select_residual_candidates(
    *,
    args,
    train_fs,
    checkpoint_val_fs,
    selection_fs,
    trend_family,
    alpha,
    env_kwargs,
    output,
) -> ResidualCandidateSelection
```

`ResidualCandidateSelection` adds `selected_model_digest: str`.

- [ ] **Step 1: Write failing dataset-role tests**

Monkeypatch `train_ppo` and `evaluate_relative_agent`. Assert `train_ppo(..., val_fs=checkpoint_val_fs)` and every A/B/C/D matrix evaluation receives `selection_fs`.

- [ ] **Step 2: Write failing shared-implementation test**

Monkeypatch `mars_lite.pipeline.residual_candidates.train_select_residual_candidates` and assert `run_baseline_residual()` calls it once.

- [ ] **Step 3: Run tests and confirm RED**

```bash
uv run pytest -q tests/test_residual_candidate_training.py tests/test_residual_pipeline_shared_candidates.py
```

- [ ] **Step 4: Implement model artifact digest**

```python
def digest_model_artifact(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    digest = hashlib.sha256()
    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        digest.update(child.relative_to(path).as_posix().encode())
        digest.update(hashlib.sha256(child.read_bytes()).digest())
    return digest.hexdigest()
```

A uses `identity:base_trend_v2`.

- [ ] **Step 5: Replace duplicated single-split A/B/C/D logic**

Create non-overlapping checkpoint-validation and configuration-selection windows in `run_baseline_residual()` and call the public candidate function.

- [ ] **Step 6: Run tests and confirm GREEN**

- [ ] **Step 7: Commit**

```bash
git add mars_lite/pipeline/residual_candidates.py mars_lite/pipeline/residual_pipeline.py tests/test_residual_candidate_training.py tests/test_residual_pipeline_shared_candidates.py
git commit -m "refactor: centralize residual candidate selection"
```

---

### Task 3: Immutable resolved configuration

**Files:**
- Create: `mars_lite/pipeline/residual_wf_config.py`
- Create: `tests/test_residual_wf_config.py`

**Interfaces:**

```python
@dataclass(frozen=True)
class ResidualWalkForwardConfig:
    @classmethod
    def from_args(cls, args, *, dataset_identity: str) -> "ResidualWalkForwardConfig": ...
    def to_dict(self) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing requested/effective-value tests**

Assert the object stores requested/effective decision interval, ensemble size, purge, timeframe, annualization, horizon, signal model, fee profile, dataset identity, and optional Git SHA. Assert the input Namespace remains unchanged.

- [ ] **Step 2: Run tests and confirm RED**

- [ ] **Step 3: Implement frozen config and validation**

Resolve `bars_per_year` from `TF_TO_MINUTES`, reject non-positive decision intervals/folds/seeds, and expose a strict serializable dictionary.

- [ ] **Step 4: Run tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add mars_lite/pipeline/residual_wf_config.py tests/test_residual_wf_config.py
git commit -m "feat: add immutable residual walk-forward config"
```

---

### Task 4: Pipeline orchestration and transactional publication

**Files:**
- Create: `mars_lite/pipeline/residual_walk_forward.py`
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/run_baseline_residual.py`
- Test: `tests/test_residual_walk_forward_runner.py`
- Test: `tests/test_residual_walk_forward_cli.py`
- Create: `tests/test_residual_walk_forward_atomicity.py`

**Interfaces:**
- Produces `run_residual_walk_forward(args, output_dir) -> dict[str, object]` from the pipeline module.
- Produces one completed run under `residual_wf_runs/<run_id>` and an atomic top-level report.

- [ ] **Step 1: Write failing import/dispatch tests**

Assert CLI imports `run_residual_walk_forward` from `mars_lite.pipeline.residual_walk_forward`, while direct `--phase wf` remains unchanged.

- [ ] **Step 2: Write failing atomicity tests**

Use a temporary output containing an existing successful report. Force fold 1 to raise. Assert the old report remains byte-identical, no new fold files appear in the old successful run, and the failed run is isolated under `failed/<run_id>`.

- [ ] **Step 3: Write failing minimum-fold test**

Stub fold planning to return zero or one executable fold and assert `RuntimeError` before publication.

- [ ] **Step 4: Run tests and confirm RED**

- [ ] **Step 5: Implement orchestration**

Create staging with `tempfile.mkdtemp(prefix="residual-wf-", dir=output / ".staging")`. Write all fold artifacts there. Validate at least two completed folds, strict JSON serialization, model digests, and expected fold reports. Atomically rename staging to `residual_wf_runs/<run_id>`, then replace the top-level report through `os.replace()`.

On failure, move staging to `failed/<run_id>` and re-raise.

- [ ] **Step 6: Delete orchestration from eval module**

Leave only pure planning/statistical helpers in `mars_lite/eval/residual_walk_forward.py`.

- [ ] **Step 7: Run tests and confirm GREEN**

- [ ] **Step 8: Commit**

```bash
git add mars_lite/pipeline/residual_walk_forward.py mars_lite/eval/residual_walk_forward.py scripts/run_pipeline.py scripts/run_baseline_residual.py tests/test_residual_walk_forward_runner.py tests/test_residual_walk_forward_cli.py tests/test_residual_walk_forward_atomicity.py
git commit -m "feat: publish residual walk-forward runs atomically"
```

---

### Task 5: Provenance, stitched reporting, and annualization parity

**Files:**
- Modify: `mars_lite/pipeline/residual_walk_forward.py`
- Modify: `mars_lite/learning/baselines.py`
- Modify: `mars_lite/eval/walk_forward.py`
- Test: `tests/test_residual_walk_forward_fold.py`
- Test: `tests/test_residual_walk_forward_runner.py`
- Create: `tests/test_baseline_annualization.py`
- Create: `tests/test_baseline_rebalance_phase.py`

**Interfaces:**
- All 1x/2x OOS reports carry the same `selected_model_digest`.
- Baseline simulation accepts `bars_per_year` and absolute start/rebalance phase.

- [ ] **Step 1: Write failing same-digest test**

Assert both cost scenarios contain the selected digest returned by the candidate selection and reject mismatched digests.

- [ ] **Step 2: Write failing annualization tests**

For identical return arrays, assert the 4h Sharpe uses `sqrt(2190)` and 1d uses `sqrt(365)`, not `sqrt(8760)`.

- [ ] **Step 3: Write failing rebalance-phase test**

Evaluate the same absolute timestamps through two contextual slices and assert baseline target weights are identical at matching timestamps.

- [ ] **Step 4: Run tests and confirm RED**

- [ ] **Step 5: Thread `bars_per_year` and absolute phase**

Add optional `bars_per_year` to baseline result conversion and DSR. Pass effective values from residual config. Ensure strategies using periodic rebalance derive it from absolute timestamp/index rather than slice-relative `t % N`.

- [ ] **Step 6: Build stitched aggregate from fold series**

Use `include_return_series=True`, remove raw arrays from public fold reports after constructing `RelativeFoldSeries`, and place the combined metrics under `summary["stitched_oos"]`.

- [ ] **Step 7: Run tests and confirm GREEN**

- [ ] **Step 8: Commit**

```bash
git add mars_lite/pipeline/residual_walk_forward.py mars_lite/learning/baselines.py mars_lite/eval/walk_forward.py tests/test_residual_walk_forward_fold.py tests/test_residual_walk_forward_runner.py tests/test_baseline_annualization.py tests/test_baseline_rebalance_phase.py
git commit -m "fix: bind residual reports to honest OOS metrics"
```

---

### Task 6: Normative architecture and operating documentation

**Files:**
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/BASELINE_RESIDUAL_RL.md`
- Modify: `docs/ja/BASELINE_RESIDUAL_RL.md`
- Modify: `docs/superpowers/specs/2026-07-13-residual-walk-forward-design.md`
- Test: `tests/test_documentation_contract.py`

- [ ] **Step 1: Write failing documentation contract assertions**

Require the normative architecture to mention residual Control Plane orchestration, pure eval primitives, atomic staging/publication, at least two folds, stitched OOS, and research-only registration boundary.

- [ ] **Step 2: Run tests and confirm RED**

- [ ] **Step 3: Update English and Japanese documentation**

Document the new output layout, four-segment fold, prior-success behavior after failure, model digest, and direct 3×3 evidence as a diagnostic failure of the legacy direct path rather than evidence about residual performance.

- [ ] **Step 4: Run tests and confirm GREEN**

- [ ] **Step 5: Commit**

```bash
git add docs/ARCHITECTURE.md docs/BASELINE_RESIDUAL_RL.md docs/ja/BASELINE_RESIDUAL_RL.md docs/superpowers/specs/2026-07-13-residual-walk-forward-design.md tests/test_documentation_contract.py
git commit -m "docs: define hardened residual walk-forward architecture"
```

---

### Task 7: Full verification and PR readiness

**Files:**
- Modify: `.github/workflows/ci.yml` only if new focused tests are not already included.
- Modify: PR #12 body.

- [ ] **Step 1: Run focused contracts**

```bash
uv run pytest -q \
  tests/test_residual_walk_forward.py \
  tests/test_residual_walk_forward_architecture.py \
  tests/test_residual_candidate_training.py \
  tests/test_residual_pipeline_shared_candidates.py \
  tests/test_residual_wf_config.py \
  tests/test_residual_walk_forward_atomicity.py \
  tests/test_residual_walk_forward_fold.py \
  tests/test_residual_walk_forward_runner.py \
  tests/test_residual_walk_forward_cli.py \
  tests/test_baseline_annualization.py \
  tests/test_baseline_rebalance_phase.py
```

Expected: all pass.

- [ ] **Step 2: Run static checks**

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy mars_lite
```

Expected: all pass.

- [ ] **Step 3: Run complete suite and coverage**

```bash
uv run pytest --cov=mars_lite --cov-fail-under=70 tests/
```

Expected: all pass, coverage at least 70%.

- [ ] **Step 4: Verify branch hygiene**

Confirm no temporary patch scripts, CI write permissions, generated model files, output directories, or `.staging` data are committed.

- [ ] **Step 5: Update PR body and mark ready**

Record exact CI evidence and keep Production status `NO-GO`.
