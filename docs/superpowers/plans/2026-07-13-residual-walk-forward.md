# Residual Walk-Forward Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a research-only nested expanding Walk-Forward path that trains and selects baseline-anchored residual configurations A/B/C/D inside each fold and evaluates the frozen selection on a chronologically later OOS slice at 1x and 2x costs.

**Architecture:** A new `mars_lite.eval.residual_walk_forward` module owns chronological fold construction, per-fold orchestration, deterministic reporting, and cross-fold summaries. Candidate training and A/B/C/D selection are extracted into a reusable `train_select_residual_candidates()` function in `mars_lite.pipeline.residual_pipeline`, so the existing single-split workflow and the new Walk-Forward use identical candidate semantics. CLI scripts dispatch residual `phase=wf` to the new runner while leaving legacy direct Walk-Forward unchanged.

**Tech Stack:** Python 3.12, NumPy, Stable-Baselines3 PPO, Gymnasium, pytest, Ruff, mypy, GitHub Actions.

## Global Constraints

- The workflow is research-only and must never register or activate a model.
- The authoritative output is `residual_walk_forward.json` with `mode=baseline_residual_walk_forward_v1` and `action_schema=baseline_residual_v1`.
- Each completed fold selects A, B, or D using inner validation only; C remains diagnostic and is never selected.
- The same trained selected policy and frozen alpha artifact must be reused for 1x and 2x outer OOS evaluation.
- Outer OOS scored ranges must be chronological and non-overlapping, with `max(configured_purge, horizon, 24)` purge bars.
- Trend history context is allowed before validation/test scoring but must not contribute to returns.
- Unexpected fold exceptions fail the complete run; only declared minimum-size conditions may skip a fold.
- Legacy direct `phase_wf()` behavior and output names remain unchanged.
- JSON serialization must reject NaN and infinity.

---

### Task 1: Chronological fold planner and report summary

**Files:**
- Create: `mars_lite/eval/residual_walk_forward.py`
- Create: `tests/test_residual_walk_forward.py`

**Interfaces:**
- Produces: `ResidualFoldSpec`, `build_residual_fold_specs(n_bars: int, n_folds: int, purge_bars: int, horizon: int) -> tuple[list[ResidualFoldSpec], list[dict[str, object]]]`
- Produces: `summarize_residual_folds(folds: list[dict[str, object]], requested_folds: int, skipped_folds: list[dict[str, object]]) -> dict[str, object]`
- Produces: `save_residual_walk_forward_report(path: Path, payload: dict[str, object]) -> None`

- [ ] **Step 1: Write failing tests for chronological fold boundaries**

```python
from mars_lite.eval.residual_walk_forward import build_residual_fold_specs


def test_fold_specs_are_expanding_and_oos_ranges_do_not_overlap():
    specs, skipped = build_residual_fold_specs(
        n_bars=1_000, n_folds=3, purge_bars=24, horizon=12
    )
    assert not skipped
    assert [spec.outer_train_start for spec in specs] == [0, 0, 0]
    assert all(spec.outer_test_start >= spec.outer_train_end + 24 for spec in specs)
    assert all(
        left.outer_test_end <= right.outer_test_start
        for left, right in zip(specs, specs[1:])
    )
    assert all(spec.inner_train_end < spec.inner_validation_start for spec in specs)
```

- [ ] **Step 2: Run the boundary test and verify RED**

Run: `uv run pytest -q tests/test_residual_walk_forward.py::test_fold_specs_are_expanding_and_oos_ranges_do_not_overlap`

Expected: FAIL because `mars_lite.eval.residual_walk_forward` does not exist.

- [ ] **Step 3: Implement the immutable fold specification and planner**

```python
@dataclass(frozen=True)
class ResidualFoldSpec:
    fold: int
    outer_train_start: int
    outer_train_end: int
    inner_train_start: int
    inner_train_end: int
    inner_validation_start: int
    inner_validation_end: int
    outer_test_start: int
    outer_test_end: int
    purge_bars: int


def build_residual_fold_specs(...):
    effective_purge = max(int(purge_bars), int(horizon), 24)
    edges = np.linspace(int(n_bars * 0.4), n_bars, n_folds + 1).astype(int)
    # Build expanding folds and append a declared skip record when train<200,
    # validation<100, or test<50.
```

The planner must validate positive inputs, include exact absolute indices in skip records, and never silently clamp an invalid range into a completed fold.

- [ ] **Step 4: Add failing summary tests**

```python
def test_summary_counts_selection_activity_and_zero_trade_warnings():
    folds = [
        {
            "selected_configuration": "A",
            "alpha_enabled": False,
            "selected_seed_fallbacks": [],
            "outer_oos": {
                "relative_1x": {
                    "hybrid": {"total_return": 0.1, "n_trades": 4},
                    "shadow": {"total_return": 0.1, "n_trades": 4},
                    "paired": {"excess_log_return": 0.0},
                },
                "relative_2x": {
                    "hybrid": {"total_return": 0.08, "n_trades": 4},
                    "shadow": {"total_return": 0.08, "n_trades": 4},
                    "paired": {"excess_log_return": 0.0},
                },
            },
            "split": {"outer_test_scored_bars": 100},
        },
        {
            "selected_configuration": "B",
            "alpha_enabled": False,
            "selected_seed_fallbacks": [True, False],
            "outer_oos": {
                "relative_1x": {
                    "hybrid": {"total_return": 0.02, "n_trades": 0},
                    "shadow": {"total_return": 0.01, "n_trades": 0},
                    "paired": {"excess_log_return": 0.01},
                },
                "relative_2x": {
                    "hybrid": {"total_return": 0.0, "n_trades": 0},
                    "shadow": {"total_return": 0.0, "n_trades": 0},
                    "paired": {"excess_log_return": 0.0},
                },
            },
            "split": {"outer_test_scored_bars": 120},
        },
    ]
    summary = summarize_residual_folds(folds, requested_folds=2, skipped_folds=[])
    assert summary["selection_counts"] == {"A": 1, "B": 1, "D": 0}
    assert summary["shadow_zero_trade_folds"] == 1
    assert summary["hybrid_zero_trade_folds"] == 1
    assert summary["selected_member_fallback_count"] == 1
    assert summary["total_scored_oos_bars"] == 220
```

- [ ] **Step 5: Implement deterministic summary and strict JSON save**

Use `statistics.mean`, `statistics.median`, and explicit float conversion. `save_residual_walk_forward_report()` must call `json.dumps(..., allow_nan=False, sort_keys=True, indent=2)` and append one newline.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest -q tests/test_residual_walk_forward.py`

Expected: PASS.

Commit: `feat: add residual walk-forward fold planner`

---

### Task 2: Extract shared residual candidate training and selection

**Files:**
- Modify: `mars_lite/pipeline/residual_pipeline.py`
- Modify: `tests/test_residual_matrix.py`
- Create: `tests/test_residual_candidate_training.py`

**Interfaces:**
- Produces: `ResidualCandidateSelection` dataclass.
- Produces: `train_select_residual_candidates(*, args, train_fs, val_fs, trend_family, alpha, env_kwargs, output) -> ResidualCandidateSelection`
- Consumes existing: `select_residual_configuration()`, `_train_residual_ensemble()`, `_evaluation_kwargs()`, `IdentityResidualAgent`, `FixedResidualAgent`.

- [ ] **Step 1: Write a failing dependency-injected candidate-selection test**

Monkeypatch `_train_residual_ensemble` and `evaluate_relative_agent` so no PPO training occurs. Assert that:

```python
result = train_select_residual_candidates(...)
assert set(result.development_results) == {"A", "B", "C", "D"}
assert result.selected_configuration == "B"
assert result.selected_alpha_enabled is False
assert result.selected_agent is fake_b_agent
```

Also assert that when `alpha.enabled` is false, C/D are absent and only A/B can be selected.

- [ ] **Step 2: Run the tests and verify RED**

Run: `uv run pytest -q tests/test_residual_candidate_training.py`

Expected: FAIL because `ResidualCandidateSelection` and `train_select_residual_candidates()` do not exist.

- [ ] **Step 3: Add the shared dataclass and function**

```python
@dataclass(frozen=True)
class ResidualCandidateSelection:
    development_results: dict[str, dict[str, Any]]
    development_cost2x_results: dict[str, dict[str, Any]]
    selection: dict[str, Any]
    selected_configuration: str
    selected_agent: object
    selected_policies: tuple[object, ...]
    selected_model_path: Path | None
    selected_alpha_enabled: bool
```

The function must build A/B and optional C/D exactly once, run the existing selector, and resolve the selected agent without accessing outer-test data.

- [ ] **Step 4: Replace the duplicated candidate block in `run_baseline_residual()`**

Replace the current A/B/C/D construction and selected-agent `if` chain with one call to `train_select_residual_candidates()`. Keep the existing report field names and values identical by reading fields from the returned dataclass.

- [ ] **Step 5: Run candidate and existing matrix tests**

Run: `uv run pytest -q tests/test_residual_candidate_training.py tests/test_residual_matrix.py tests/test_relative_evaluation.py`

Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `refactor: share residual candidate selection`

---

### Task 3: Implement one nested residual Walk-Forward fold

**Files:**
- Modify: `mars_lite/eval/residual_walk_forward.py`
- Modify: `tests/test_residual_walk_forward.py`

**Interfaces:**
- Produces: `run_residual_fold(*, fs, spec, args, output_dir, dependencies=None) -> dict[str, object]`
- Consumes: `train_select_residual_candidates()`, `FrozenResidualAlpha.fit()`, `walk_forward_ic()`, `evaluate_residual_alpha_gate()`, `evaluate_relative_agent()`, `with_history_context()`, `run_all_baselines()`.

- [ ] **Step 1: Write a failing orchestration-order test using stubs**

Record calls in a list and assert this order:

```python
assert calls == [
    "leak_test",
    "alpha_gate",
    "alpha_fit",
    "candidate_selection",
    "outer_eval_1x",
    "outer_eval_2x",
    "baselines_1x",
    "baselines_2x",
]
```

The fake candidate selector must receive only inner train/validation FeatureSets. The fake outer evaluator must receive the exact same `selected_agent` object for 1x and 2x.

- [ ] **Step 2: Run the test and verify RED**

Run: `uv run pytest -q tests/test_residual_walk_forward.py::test_run_residual_fold_freezes_selection_before_outer_oos`

Expected: FAIL because `run_residual_fold()` does not exist.

- [ ] **Step 3: Implement fold execution**

The implementation must:

1. slice inner training by absolute indices;
2. run leak self-test and raise on failure;
3. fit alpha only on inner training;
4. construct validation/test context using `history_bars=max(lookbacks)+rebalance_every`;
5. call `train_select_residual_candidates()` into `residual_wf/fold_<k>/`;
6. evaluate the selected object twice without retraining;
7. evaluate diagnostic baselines at 1x and 2x using the scored test start marker;
8. save the fold alpha artifact and `fold_report.json` using strict JSON;
9. include exact split indices, context counts, development matrices, selection, fallback flags, activity metrics, and model/artifact identities.

- [ ] **Step 4: Add activity and context tests**

Use a deterministic trending fixture with `IdentityResidualAgent` and the real residual environment. Assert `shadow.n_trades > 0`, `outer_test_scored_bars` excludes context, and the report preserves both hybrid and shadow trade counts.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest -q tests/test_residual_walk_forward.py tests/test_contextual_evaluation.py tests/test_baseline_residual_env.py`

Expected: PASS.

Commit: `feat: execute nested residual walk-forward folds`

---

### Task 4: Implement full runner and authoritative report

**Files:**
- Modify: `mars_lite/eval/residual_walk_forward.py`
- Create: `tests/test_residual_walk_forward_runner.py`

**Interfaces:**
- Produces: `run_residual_walk_forward(args, output_dir: str | Path) -> dict[str, object]`

- [ ] **Step 1: Write a failing runner test**

Monkeypatch dataset construction and `run_residual_fold()`. Assert that the runner:

```python
report = run_residual_walk_forward(args, tmp_path)
assert report["mode"] == "baseline_residual_walk_forward_v1"
assert report["action_schema"] == "baseline_residual_v1"
assert report["release_eligible"] is False
assert (tmp_path / "residual_walk_forward.json").is_file()
assert not (tmp_path / "walk_forward_cost1x.json").exists()
```

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest -q tests/test_residual_walk_forward_runner.py`

Expected: FAIL because the top-level runner does not exist.

- [ ] **Step 3: Implement the runner**

Set residual defaults on the args object (`action_mode`, `min_trade_delta`, `lambda_turnover`), build the FeatureSet once, build fold specs, execute valid folds sequentially, summarize them, and write exactly one authoritative top-level JSON report.

The top-level `config` must include requested folds, completed folds, purge, horizon, decision interval, ensemble size, seeds, run tier, source bar count, and split ratios. `release_blocker` must state that sealed residual release evidence is incomplete.

- [ ] **Step 4: Add fail-closed tests**

Assert that leak-test failure, non-finite payloads, and unexpected fold exceptions abort the run and do not produce a successful top-level report.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest -q tests/test_residual_walk_forward.py tests/test_residual_walk_forward_runner.py`

Expected: PASS.

Commit: `feat: add residual walk-forward runner`

---

### Task 5: Wire CLI dispatch without changing direct Walk-Forward

**Files:**
- Modify: `scripts/run_pipeline.py`
- Modify: `scripts/run_baseline_residual.py`
- Modify: `tests/test_residual_release_boundary.py`
- Create: `tests/test_residual_walk_forward_cli.py`

**Interfaces:**
- Consumes: `run_residual_walk_forward(args, output_dir)`.

- [ ] **Step 1: Write failing dispatch tests**

Patch both residual runner functions and assert:

```python
# run_pipeline.py
--action-mode baseline-residual --phase wf --no-register
# calls run_residual_walk_forward exactly once and never run_baseline_residual.

# run_baseline_residual.py
--phase wf
# calls run_residual_walk_forward exactly once.
```

Also assert `action_mode=direct, phase=wf` still calls the legacy production/direct path.

- [ ] **Step 2: Run and verify RED**

Run: `uv run pytest -q tests/test_residual_walk_forward_cli.py`

Expected: FAIL because both scripts dispatch all residual phases to the single-split runner.

- [ ] **Step 3: Implement explicit phase dispatch**

```python
if args.action_mode == "baseline-residual":
    if args.phase == "wf":
        run_residual_walk_forward(args, Path(args.output))
    else:
        run_baseline_residual(args, Path(args.output))
    return 0
```

Apply the equivalent dispatch in `scripts/run_baseline_residual.py`. Do not change direct mode.

- [ ] **Step 4: Preserve registration fail-closed behavior**

Keep `validate_residual_invocation()` before dispatch in the control-plane script. Add a test showing that omitting `--no-register` still fails before the WF runner is invoked.

- [ ] **Step 5: Run focused tests and commit**

Run: `uv run pytest -q tests/test_residual_walk_forward_cli.py tests/test_residual_release_boundary.py tests/test_production_pipeline.py`

Expected: PASS.

Commit: `feat: dispatch residual walk-forward from cli`

---

### Task 6: Documentation, CI focus set, and complete verification

**Files:**
- Modify: `docs/BASELINE_RESIDUAL_RL.md`
- Modify: `docs/ja/BASELINE_RESIDUAL_RL.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `tests/test_documentation_contract.py` only when a documented command is contract-tested.

**Interfaces:**
- No new runtime interfaces.

- [ ] **Step 1: Document the exact research command and output distinction**

Document that legacy direct WF produces `walk_forward_cost1x.json` / `walk_forward_cost2x.json`, while residual WF produces `residual_walk_forward.json`. Include the exact command from the specification and explain hybrid-vs-shadow trade counts.

- [ ] **Step 2: Add focused residual WF tests to CI**

Add `tests/test_residual_walk_forward.py`, `tests/test_residual_walk_forward_runner.py`, and `tests/test_residual_walk_forward_cli.py` to the existing `Run critical residual contracts` step.

- [ ] **Step 3: Run static and focused verification**

Run:

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy mars_lite
uv run pytest -q \
  tests/test_residual_walk_forward.py \
  tests/test_residual_walk_forward_runner.py \
  tests/test_residual_walk_forward_cli.py \
  tests/test_residual_candidate_training.py \
  tests/test_residual_matrix.py \
  tests/test_residual_release_boundary.py
```

Expected: all commands PASS.

- [ ] **Step 4: Run the complete repository test suite**

Run: `uv run pytest --cov=mars_lite --cov-fail-under=70 tests/`

Expected: PASS with coverage at least 70%.

- [ ] **Step 5: Verify branch diff and research boundary**

Confirm no Registry write path was introduced, the direct Walk-Forward implementation is unchanged, and the authoritative residual output contains `release_eligible: false`.

- [ ] **Step 6: Commit**

Commit: `docs: document residual walk-forward workflow`
