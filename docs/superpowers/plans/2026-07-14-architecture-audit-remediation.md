# Architecture Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Resolve every issue recorded in the post-merge architecture audit while preserving public compatibility and research-only safety status.

**Architecture:** Keep `ResidualMarketEnv` and the CLI as stable facades, move cohesive logic behind typed helpers and integration protocols, and bind every new artifact or checkpoint through existing content-addressed identities. Implement behavior changes test-first and preserve old dataset API return types through warning wrappers.

**Tech Stack:** Python 3.12, NumPy, Gymnasium, Stable-Baselines3, sb3-contrib, pytest, Hypothesis, Mypy, Ruff, Import Linter, pytest-cov.

## Global Constraints

- Production status remains `NO-GO`.
- Default baseline window and minimum history are exactly 720 hours.
- Default baseline tolerance is exactly 0.015 and is never prorated.
- Sealed outer-test data cannot affect training, checkpointing or selection.
- Public imports remain compatible for one release.
- Dataset, environment, action, signal, normalizer and checkpoint identities fail closed.
- No Stable-Baselines3 or sb3-contrib imports are permitted from `trade_rl.workflows`.

---

### Task 1: Fixed-window reward pre-roll

**Files:**
- Modify: `trade_rl/rl/rewards.py`
- Modify: `trade_rl/rl/environment.py`
- Create: `trade_rl/rl/episode.py`
- Modify: `tests/test_rewards.py`
- Modify: `tests/test_environment.py`
- Create: `tests/test_episode.py`

**Interfaces:**
- Produces: `RewardTracker.reset(..., hybrid_history: Sequence[float] = (), shadow_history: Sequence[float] = ()) -> None`
- Produces: `EpisodeRange(start: int, stop: int, reward_start: int)` and `resolve_episode_range(...) -> EpisodeRange`

- [ ] **Step 1: Write failing reward tests**

```python
def test_default_baseline_penalty_requires_complete_window() -> None:
    config = RewardConfig()
    tracker = RewardTracker(config=config, decision_hours=4.0)
    assert tracker.baseline_window_steps == 180
    assert tracker.baseline_minimum_history_steps == 180
    for _ in range(179):
        result = tracker.step(
            hybrid_log_return=-0.01,
            shadow_log_return=0.0,
            hybrid_drawdown=0.0,
            shadow_drawdown=0.0,
        )
        assert result.baseline_underperformance_penalty == 0.0


def test_baseline_tolerance_is_not_prorated() -> None:
    tracker = RewardTracker(config=RewardConfig(), decision_hours=4.0)
    tracker.reset(hybrid_history=(-0.001,) * 90, shadow_history=(0.0,) * 90)
    assert tracker.last_context_after.baseline_tolerance == pytest.approx(0.015)
```

- [ ] **Step 2: Run reward tests and verify failure**

Run: `uv run pytest tests/test_rewards.py -q`
Expected: failures showing minimum history is 42 steps or tolerance is prorated.

- [ ] **Step 3: Implement complete-window defaults and seeded reset**

Set both reward configuration defaults to 720 hours. Replace proportional tolerance calculations with `self.config.baseline_tolerance`. Validate equal, finite seed histories, truncate to `baseline_window_steps`, and initialize the tracker without emitting reward.

- [ ] **Step 4: Write failing episode pre-roll tests**

```python
def test_episode_range_reserves_reward_preroll() -> None:
    value = resolve_episode_range(
        requested_start=200,
        episode_bars=60,
        reward_preroll_bars=180,
        dataset_bars=500,
    )
    assert value.start == 20
    assert value.reward_start == 200
    assert value.stop == 260


def test_episode_range_rejects_missing_preroll() -> None:
    with pytest.raises(ValueError, match="pre-roll"):
        resolve_episode_range(
            requested_start=100,
            episode_bars=60,
            reward_preroll_bars=180,
            dataset_bars=500,
        )
```

- [ ] **Step 5: Implement episode helper and environment seeding**

`ResidualMarketEnv.reset()` uses `resolve_episode_range`, replays hybrid and shadow baseline returns only across `[start, reward_start)`, seeds `RewardTracker`, then exposes the first rewarded observation at `reward_start`. Explicit restored states must include compatible reward history or fail closed.

- [ ] **Step 6: Run focused tests and commit**

Run: `uv run pytest tests/test_rewards.py tests/test_episode.py tests/test_environment.py -q`
Expected: PASS.

Commit: `feat: enforce fixed reward window with causal preroll`

---

### Task 2: Alpha and factor artifact adapters

**Files:**
- Create: `trade_rl/artifacts/signals.py`
- Create: `trade_rl/integrations/signal_artifacts.py`
- Modify: `trade_rl/cli/config.py`
- Modify: `trade_rl/cli/app.py`
- Modify: `trade_rl/workflows/training.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/serving/bundle.py`
- Create: `tests/test_signal_artifacts.py`
- Modify: `tests/test_cli.py`
- Modify: `tests/test_market_walk_forward.py`

**Interfaces:**
- Produces: `SignalArtifactManifest`, `LoadedAlphaArtifact`, `LoadedFactorArtifact`
- Produces: `load_alpha_artifact(path, *, dataset_id, evaluation_start) -> LoadedAlphaArtifact`
- Produces: `load_factor_artifact(path, *, dataset_id, evaluation_start, expected_names) -> LoadedFactorArtifact`

- [ ] **Step 1: Write failing manifest and causal-range tests**

```python
def test_alpha_artifact_rejects_fit_range_touching_evaluation(tmp_path: Path) -> None:
    write_alpha_fixture(tmp_path, fit_stop=100, evaluation_start=100)
    with pytest.raises(ValueError, match="strictly before"):
        load_alpha_artifact(tmp_path, dataset_id=DATASET_ID, evaluation_start=100)


def test_factor_artifact_requires_exact_factor_names(tmp_path: Path) -> None:
    write_factor_fixture(tmp_path, names=("value", "carry"))
    with pytest.raises(ValueError, match="factor names"):
        load_factor_artifact(
            tmp_path,
            dataset_id=DATASET_ID,
            evaluation_start=200,
            expected_names=("carry", "value"),
        )
```

- [ ] **Step 2: Implement canonical signal manifests and providers**

Use canonical JSON, deterministic NPZ, SHA-256 allow-lists, exact shape checks, finite-value checks, dataset identity and half-open fit ranges. Providers implement the existing environment `AlphaProvider` and `FactorBasisProvider` protocols.

- [ ] **Step 3: Write failing CLI configuration tests**

```python
def test_training_config_requires_alpha_artifact_when_action_enables_alpha() -> None:
    raw = training_mapping(action={"alpha_enabled": True})
    with pytest.raises(ValueError, match="alpha artifact"):
        TrainingRunConfig.from_mapping(raw)
```

- [ ] **Step 4: Extend run configuration and workflow wiring**

Add optional `alpha_artifact: Path | None` and `factor_artifact: Path | None`. Remove unconditional rejection of alpha/factor actions. Require exact artifact presence and dimensions, pass providers and digests into each environment, and include paths/digests in run and serving manifests.

- [ ] **Step 5: Add fold-local validation tests and commit**

Run: `uv run pytest tests/test_signal_artifacts.py tests/test_cli.py tests/test_market_walk_forward.py -q`
Expected: PASS.

Commit: `feat: wire causal alpha and factor artifacts into training`

---

### Task 3: Framework-neutral checkpoint loading

**Files:**
- Create: `trade_rl/workflows/checkpoints.py`
- Create: `trade_rl/integrations/checkpoints.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Modify: `trade_rl/integrations/__init__.py`
- Modify: `.importlinter`
- Create: `tests/test_checkpoint_integration.py`
- Modify: `tests/test_architecture.py`

**Interfaces:**
- Produces: `PolicyCheckpoint` dataclass and `PolicyCheckpointLoader` protocol
- Produces: `StableBaselines3CheckpointLoader.load(checkpoint: PolicyCheckpoint) -> PredictivePolicy`

- [ ] **Step 1: Write an architecture test that forbids framework imports**

```python
def test_workflows_do_not_import_sb3() -> None:
    source = Path("trade_rl/workflows/market_walk_forward.py").read_text()
    assert "stable_baselines3" not in source
    assert "sb3_contrib" not in source
```

- [ ] **Step 2: Write loader behavior tests**

Test all four algorithm names, unsupported algorithms, missing files and non-finite predictions using monkeypatched model classes in the integration module.

- [ ] **Step 3: Implement protocol and adapter**

Move `_load_model` behavior into `trade_rl.integrations.checkpoints`. Workflows receive a loader dependency and operate only on the protocol. Extend Import Linter to explicitly forbid framework imports from all workflow modules.

- [ ] **Step 4: Run architecture and integration tests and commit**

Run: `uv run pytest tests/test_checkpoint_integration.py tests/test_architecture.py tests/test_market_walk_forward.py -q && uv run lint-imports`
Expected: PASS.

Commit: `refactor: isolate model checkpoints behind integration protocol`

---

### Task 4: Canonical dataset artifact API

**Files:**
- Modify: `trade_rl/data/artifact_codec.py`
- Modify: `trade_rl/data/artifact.py`
- Modify: `trade_rl/data/artifacts.py`
- Modify: `trade_rl/data/__init__.py`
- Modify: `tests/test_market_artifact.py`
- Modify: `tests/test_dataset_artifacts.py`

**Interfaces:**
- Produces: `DatasetArtifactFiles`, `PublishedDatasetArtifact`
- Produces: `write_market_dataset_files`, `publish_market_dataset_artifact`, `load_market_dataset_artifact`

- [ ] **Step 1: Write failing canonical-result tests**

```python
def test_write_market_dataset_files_returns_typed_result(tmp_path: Path) -> None:
    result = write_market_dataset_files(tmp_path, dataset_fixture())
    assert result.manifest_path == tmp_path / "manifest.json"
    assert result.arrays_path == tmp_path / "arrays.npz"
    assert len(result.artifact_digest) == 64


def test_legacy_writer_warns_and_preserves_return_type(tmp_path: Path) -> None:
    with pytest.warns(DeprecationWarning):
        legacy = data.artifact.write_market_dataset_artifact(tmp_path, dataset_fixture())
    assert isinstance(legacy, str)
```

- [ ] **Step 2: Implement typed canonical API**

Keep codec serialization in one place. Atomic publication writes to a sibling staging directory, validates by reloading, renames exclusively and returns `PublishedDatasetArtifact`.

- [ ] **Step 3: Add compatibility wrappers**

Both old module functions emit `DeprecationWarning(stacklevel=2)` and preserve their historical return values. Export only the canonical names from `trade_rl.data` documentation.

- [ ] **Step 4: Run artifact tests and commit**

Run: `uv run pytest tests/test_market_artifact.py tests/test_dataset_artifacts.py -q`
Expected: PASS.

Commit: `refactor: unify market dataset artifact publication api`

---

### Task 5: Intermediate training checkpoints

**Files:**
- Modify: `trade_rl/rl/training.py`
- Create: `trade_rl/rl/checkpointing.py`
- Modify: `trade_rl/workflows/training.py`
- Modify: `trade_rl/workflows/checkpoints.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Create: `tests/test_checkpointing.py`
- Modify: `tests/test_training_pipeline.py`
- Modify: `tests/test_market_walk_forward.py`

**Interfaces:**
- Produces: `CheckpointConfig(interval_steps: int, max_checkpoints: int)`
- Produces: `CheckpointManifest`
- Produces: `build_checkpoint_callback(...) -> BaseCallback`

- [ ] **Step 1: Write failing callback tests**

Use a fake model/callback clock to assert checkpoints are staged at exact observed timesteps, bounded by `max_checkpoints`, and never published when model serialization fails.

- [ ] **Step 2: Extend training configuration**

Add validated `checkpoint_interval_steps >= 0` and `max_checkpoints >= 1`. Resolve a maintained default interval from total timesteps when omitted; explicit zero disables intermediate checkpoints.

- [ ] **Step 3: Implement atomic checkpoint manifests**

Each checkpoint directory contains `policy.zip` and `checkpoint.json` binding algorithm, seed, requested/observed timesteps, model digest, environment digest and training configuration digest.

- [ ] **Step 4: Select only on checkpoint-validation range**

Walk-forward enumerates checkpoint manifests, evaluates each on the checkpoint-validation view, chooses deterministically by configured metric and tie-breakers, and records the selected checkpoint digest. No sealed-test adapter is constructed until selection is complete.

- [ ] **Step 5: Run checkpoint and walk-forward tests and commit**

Run: `uv run pytest tests/test_checkpointing.py tests/test_training_pipeline.py tests/test_market_walk_forward.py -q`
Expected: PASS.

Commit: `feat: add causal intermediate checkpoint selection`

---

### Task 6: Focused environment and walk-forward modules

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/episode.py`
- Create: `trade_rl/rl/transition.py`
- Modify: `trade_rl/workflows/market_walk_forward.py`
- Create: `trade_rl/workflows/walk_forward_evaluation.py`
- Modify: `tests/test_environment.py`
- Modify: `tests/test_market_walk_forward.py`

**Interfaces:**
- Produces: pure transition helpers for terminal classification and execution result summarization
- Produces: `evaluate_checkpoint_on_view(...) -> CheckpointEvaluation`

- [ ] **Step 1: Add characterization tests**

Capture current reset modes, liquidation semantics, action diagnostics and fold evaluation outputs before moving code. Compare complete dataclass results and observation bytes.

- [ ] **Step 2: Extract pure helpers without changing behavior**

Move episode range/pre-roll logic to `episode.py`, economic terminal classification and transition summaries to `transition.py`, and range-scoped policy evaluation to `walk_forward_evaluation.py`. Keep re-exports or facade methods where external tests import old names.

- [ ] **Step 3: Enforce size and dependency tests**

Add tests that `environment.py` and `market_walk_forward.py` no longer contain the extracted helper definitions and that new modules import only lower layers.

- [ ] **Step 4: Run characterization and architecture tests and commit**

Run: `uv run pytest tests/test_environment.py tests/test_market_walk_forward.py tests/test_architecture.py -q && uv run lint-imports`
Expected: PASS.

Commit: `refactor: split episode transition and walk-forward evaluation logic`

---

### Task 7: Documentation, typing and critical coverage gates

**Files:**
- Modify: `README.md`
- Modify: `docs/ARCHITECTURE.md`
- Modify: `docs/RESEARCH_STATUS.md`
- Modify: `.github/workflows/ci.yml`
- Modify: `pyproject.toml`
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/simulation/execution.py`
- Modify: `trade_rl/strategies/trend.py`
- Create: `scripts/check_critical_coverage.py`
- Create: `tests/test_critical_coverage_config.py`

**Interfaces:**
- Produces: `scripts/check_critical_coverage.py coverage.json pyproject.toml -> exit status`

- [ ] **Step 1: Add schema consistency test**

```python
def test_maintained_docs_reference_reward_v4() -> None:
    for path in (Path("README.md"), Path("docs/ARCHITECTURE.md"), Path("docs/RESEARCH_STATUS.md")):
        text = path.read_text()
        assert "reward schema v3" not in text.lower()
        assert "reward schema v4" in text.lower()
```

- [ ] **Step 2: Remove broad index suppressions**

Delete file-wide `disable-error-code="index"` directives from modified modules. Introduce typed local arrays, explicit integer indices and narrow line-level ignores only where NumPy stubs cannot express a proven shape.

- [ ] **Step 3: Implement critical coverage gate**

Generate JSON coverage with contexts/branches. The script aggregates configured critical paths, requires 90 percent branch coverage for accounting, pretrade, rewards, gates, artifacts and serving, and applies a documented non-regression threshold to execution. Fail with a per-module table.

- [ ] **Step 4: Update CI**

Run full tests once with `--cov-report=json:coverage.json`, then invoke the critical coverage script. Keep global `fail_under = 80`.

- [ ] **Step 5: Add targeted missing tests until the gate passes**

Cover economic termination, margin/cost exhaustion, risk override ordering, reward pre-roll, artifact corruption and serving identity rejection branches.

- [ ] **Step 6: Run docs, typing and coverage checks and commit**

Run: `uv run ruff check . && uv run ruff format --check --diff . && uv run mypy trade_rl && uv run pytest --cov=trade_rl --cov-branch --cov-report=json:coverage.json && uv run python scripts/check_critical_coverage.py coverage.json pyproject.toml`
Expected: PASS.

Commit: `test: enforce critical financial module coverage`

---

### Task 8: Full verification, review and integration

**Files:**
- Review: all changed files
- Update: `docs/superpowers/plans/2026-07-14-architecture-audit-remediation.md` checkboxes

- [ ] **Step 1: Run the complete local verification suite**

```bash
uv sync --extra dev --extra export
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy --no-incremental trade_rl
uv run lint-imports
uv run vulture trade_rl tests --min-confidence 100
uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing --cov-report=json:coverage.json
uv run python scripts/check_critical_coverage.py coverage.json pyproject.toml
uv run trade-rl --version
```

Expected: every command exits zero.

- [ ] **Step 2: Verify repository invariants**

Search for maintained reward-v3 documentation, direct workflow SB3 imports, broad index suppressions, duplicate undocumented dataset writers and temporary integration workflows. Expected: none outside explicit compatibility tests or changelog history.

- [ ] **Step 3: Open a draft PR and inspect the complete diff**

Confirm no generated model, dataset, coverage, cache or temporary patch artifacts are committed.

- [ ] **Step 4: Run GitHub Actions and fix every failure**

Require exact-head success for Ruff, format, Mypy, Import Linter, dead-code, full tests/coverage, critical coverage and CLI smoke.

- [ ] **Step 5: Mark ready, merge with expected head SHA and delete the branch**

Commit: `chore: complete architecture audit remediation`
