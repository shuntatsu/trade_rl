# Universal Training Launch and Monitoring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Launch the immutable full Universal U6 training generation, continuously inspect intermediate/reward evidence, repair truthful failures, and obtain verified final artifacts for all three algorithms and seeds.

**Architecture:** Wire existing append-only transition telemetry and reward-component TensorBoard logging into every SB3 member, add a generation heartbeat and evidence summarizer, and launch through a dedicated Docker Compose file bound to source/lock/runtime-manifest identities. Each failed attempt remains immutable; fixes create a new commit, image, and generation.

**Tech Stack:** Python 3.12+, Stable-Baselines3, TensorBoard event accumulator, Docker Compose, NVIDIA Container Toolkit/CUDA, PostgreSQL 16, pytest, PowerShell orchestration.

## Global Constraints

- Canonical configs: `universal-u6-ppo.json`, `universal-u6-lagrangian.json`, and `universal-u6-discounted.json`.
- Full settings per algorithm: 524288 timesteps, seeds 0/1/2, 8 environments, 128 rollout steps, batch size 256, CUDA, 8 retained checkpoints at 32768-step intervals.
- Architecture: `u_medium_direct`; baseline: `supervised_allocator`; folds: 0 and 1.
- The full run must not use smoke overrides, fewer seeds, fewer timesteps, fewer environments, or CPU fallback.
- Every image is bound to a clean Git commit, source-tree digest, lockfile digest, and runtime-manifest digest.
- The existing `trade_rl_db` and all historical containers/volumes/runs remain intact.
- Completion is software success with `research_success=false`; no profitability or sealed research-success claim is allowed.
- A run may resume only from a checkpoint whose environment, training-config, cache, dataset, normalizer, and runtime-manifest identities all match.

---

### Task 1: Wire Durable Telemetry and Heartbeats into SB3 Training

**Files:**
- Create: `trade_rl/rl/training_heartbeat.py`
- Modify: `trade_rl/integrations/sb3_training.py:1063-1090`
- Modify: `trade_rl/rl/training_telemetry.py:470-552`
- Test: `tests/rl/test_training_heartbeat.py`
- Test: `tests/integrations/test_sb3_training.py`
- Test: `tests/integrations/test_training_telemetry.py`

**Interfaces:**
- Produces: `build_training_heartbeat_callback(path, *, seed, algorithm, identity) -> BaseCallback`.
- Changes: enabled TensorBoard training receives `CheckpointCallback`,
  `TensorBoardMetricsCallback`, `TrainingTelemetryCallback`, and
  `TrainingHeartbeatCallback` in one `CallbackList`.
- Produces per member: `telemetry.jsonl`, indexed telemetry sidecars,
  `training-heartbeat.json`, and TensorBoard event files.
- Test helpers use a `FakeLogger` with `name_to_value` and reuse the existing fake
  SB3 model/environment factory; `capture_backend_learn_callbacks` returns the
  actual callback passed to the fake model's `learn` method.

- [ ] **Step 1: Write failing callback-wiring and heartbeat tests**

```python
def test_heartbeat_is_atomic_finite_and_identity_bound(tmp_path: Path) -> None:
    callback = build_training_heartbeat_callback(
        tmp_path / "training-heartbeat.json",
        seed=1,
        algorithm="ppo",
        identity={"runtime_manifest_digest": "a" * 64},
    )
    callback.model = SimpleNamespace(num_timesteps=2048, logger=FakeLogger())
    callback._on_training_start()
    callback._on_rollout_end()
    payload = json.loads((tmp_path / "training-heartbeat.json").read_text())
    assert payload["phase"] == "training"
    assert payload["global_step"] == 2048
    assert payload["runtime_manifest_digest"] == "a" * 64
    assert payload["seed"] == 1


def test_sb3_backend_wires_checkpoint_metrics_telemetry_and_heartbeat(monkeypatch, tmp_path: Path) -> None:
    callbacks = capture_backend_learn_callbacks(monkeypatch, tmp_path)
    names = {type(item).__name__ for item in callbacks.callbacks}
    assert names == {
        "CheckpointCallback",
        "TensorBoardMetricsCallback",
        "TrainingTelemetryCallback",
        "TrainingHeartbeatCallback",
    }
```

Extend telemetry integration tests to assert the callback writes
`reward_total_scaled`, portfolio/baseline value, drawdown, interval cost, and
the concrete routed symbol for each sampled environment.

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/rl/test_training_heartbeat.py tests/integrations/test_sb3_training.py tests/integrations/test_training_telemetry.py -q
```

Expected: heartbeat import fails and backend callback-name assertion fails.

- [ ] **Step 3: Implement the heartbeat and callback list**

```python
def build_training_heartbeat_callback(
    path: Path,
    *,
    seed: int,
    algorithm: str,
    identity: Mapping[str, str],
) -> Any:
    from stable_baselines3.common.callbacks import BaseCallback

    class TrainingHeartbeatCallback(BaseCallback):
        def _write(self, phase: str) -> None:
            values = getattr(self.model.logger, "name_to_value", {})
            scalars = {
                str(key): float(value)
                for key, value in values.items()
                if isinstance(value, (int, float, np.number)) and np.isfinite(value)
            }
            atomic_write_bytes(
                path,
                canonical_json_bytes(
                    {
                        "schema_version": "training_heartbeat_v1",
                        "updated_at": datetime.now(UTC).isoformat(),
                        "phase": phase,
                        "algorithm": algorithm,
                        "seed": seed,
                        "global_step": int(self.model.num_timesteps),
                        "scalars": scalars,
                        **dict(identity),
                    }
                ),
            )

        def _on_training_start(self) -> None:
            self._write("training")

        def _on_step(self) -> bool:
            return True

        def _on_rollout_end(self) -> None:
            self._write("training")

        def _on_training_end(self) -> None:
            self._write("completed")

    return TrainingHeartbeatCallback(verbose=0)
```

In `StableBaselines3Backend.train`, build all enabled callbacks and always pass
one `CallbackList` when more than one exists:

```python
callbacks = [checkpoint_callback]
if metrics_callback is not None:
    callbacks.append(metrics_callback)
    callbacks.append(
        build_training_telemetry_callback(
            path=output_path.parent / "telemetry.jsonl",
            seed=seed,
            sample_every=config.tensorboard_log_interval,
        )
    )
callbacks.append(
    build_training_heartbeat_callback(
        output_path.parent / "training-heartbeat.json",
        seed=seed,
        algorithm=config.algorithm,
        identity=heartbeat_identity,
    )
)
callback = callbacks[0] if len(callbacks) == 1 else CallbackList(callbacks)
```

Build `heartbeat_identity` without breaking non-Universal SB3 callers:

```python
heartbeat_identity = {
    "environment_digest": str(identity["environment_digest"]),
    "training_config_digest": content_digest(config.digest_payload()),
}
runtime_manifest_digest = os.environ.get(
    "TRADE_RL_RUNTIME_MANIFEST_DIGEST", ""
).strip()
if runtime_manifest_digest:
    heartbeat_identity["runtime_manifest_digest"] = require_sha256(
        runtime_manifest_digest,
        field="TRADE_RL_RUNTIME_MANIFEST_DIGEST",
    )
```

The full Universal Compose service always supplies this environment variable;
legacy and non-Universal training remains valid without it.

Keep the existing fail-closed behavior for non-finite TensorBoard values. The
append-only telemetry writer must close on training end and flush on every
rollout boundary.

- [ ] **Step 4: Run focused tests and validation**

```powershell
uv run pytest tests/rl/test_training_heartbeat.py tests/rl/test_tensorboard_logging.py tests/integrations/test_sb3_training.py tests/integrations/test_training_telemetry.py tests/telemetry -q
uv run ruff check trade_rl/rl/training_heartbeat.py trade_rl/rl/training_telemetry.py trade_rl/integrations/sb3_training.py tests/rl/test_training_heartbeat.py
uv run mypy trade_rl/rl/training_heartbeat.py trade_rl/rl/training_telemetry.py trade_rl/integrations/sb3_training.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/rl/training_heartbeat.py trade_rl/rl/training_telemetry.py trade_rl/integrations/sb3_training.py tests/rl/test_training_heartbeat.py tests/integrations/test_sb3_training.py tests/integrations/test_training_telemetry.py
git commit -m "feat: persist universal training progress"
```

### Task 2: Reward and Runtime Evidence Monitor

**Files:**
- Create: `trade_rl/operations/universal_training_monitor.py`
- Create: `scripts/monitor_universal_training.py`
- Test: `tests/operations/test_universal_training_monitor.py`
- Test: `tests/scripts/test_monitor_universal_training.py`

**Interfaces:**
- Produces: `inspect_universal_training_generation(root, *, now) -> UniversalTrainingSnapshot`.
- CLI reads Docker state, heartbeat, telemetry, checkpoints, and TensorBoard;
  writes `monitor-snapshot.json` and `reward-trends.json` atomically.
- Exit codes: 0 healthy/completed, 2 warning, 3 failed/stale/non-finite, 4 incomplete evidence.
- Test helpers write real TensorBoard scalar events, telemetry JSONL, heartbeat,
  and checkpoint manifests under `tmp_path`; mutation changes exactly one of
  heartbeat time, scalar finiteness, captured log text, or checkpoint presence.
  `fixture_now()` is fixed UTC time one minute after the healthy heartbeat.

- [ ] **Step 1: Write failing trend and health-classification tests**

```python
def test_monitor_reports_reward_components_and_training_health(tmp_path: Path) -> None:
    generation = completed_member_fixture(
        tmp_path,
        rewards=[-0.4, -0.2, 0.1, 0.3],
        growth=[-0.01, 0.0, 0.01, 0.02],
        drawdown=[0.20, 0.18, 0.15, 0.10],
    )
    snapshot = inspect_universal_training_generation(generation, now=fixture_now())
    member = snapshot.members[0]
    assert member.reward_total.direction == "improving"
    assert member.reward_growth.direction == "improving"
    assert member.drawdown.direction == "improving"
    assert member.nonfinite_count == 0
    assert snapshot.status == "healthy"


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (("stale_heartbeat", "stale"), ("nan_event", "non-finite"), ("oom", "OOM"), ("missing_checkpoint", "checkpoint")),
)
def test_monitor_fails_closed_on_runtime_evidence_defects(tmp_path: Path, mutation: str, reason: str) -> None:
    generation = mutated_member_fixture(tmp_path, mutation)
    snapshot = inspect_universal_training_generation(generation, now=fixture_now())
    assert snapshot.status == "failed"
    assert any(reason.lower() in item.lower() for item in snapshot.findings)
```

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/operations/test_universal_training_monitor.py tests/scripts/test_monitor_universal_training.py -q
```

Expected: imports fail for monitor module and script.

- [ ] **Step 3: Implement evidence parsing and deterministic trend summaries**

Read TensorBoard scalars using
`tensorboard.backend.event_processing.event_accumulator.EventAccumulator`.
Require these project tags where applicable:

```python
REWARD_TAGS = (
    "trade_rl/reward_mean",
    "trade_rl/reward_growth_raw_mean",
    "trade_rl/reward_absolute_component_mean",
    "trade_rl/reward_excess_component_mean",
    "trade_rl/reward_baseline_penalty_weighted_mean",
    "trade_rl/reward_drawdown_penalty_weighted_mean",
    "trade_rl/reward_projection_penalty_weighted_mean",
    "trade_rl/reward_terminal_penalty_weighted_mean",
    "trade_rl/reward_margin_penalty_weighted_mean",
    "trade_rl/reward_total_raw_mean",
    "trade_rl/portfolio_value_mean",
    "trade_rl/baseline_portfolio_value_mean",
    "trade_rl/drawdown_mean",
    "trade_rl/interval_cost_mean",
    "trade_rl/rolling_growth_gap_mean",
)
TRAIN_TAGS = (
    "train/approx_kl",
    "train/explained_variance",
    "train/policy_gradient_loss",
    "train/value_loss",
    "train/entropy_loss",
    "train/std",
)
```

Summarize each scalar in non-overlapping first/last 20% windows with count,
minimum, maximum, mean, median, slope per 100k steps, and direction. Parse
`telemetry.jsonl` for per-symbol counts/reward/PV/drawdown/cost, checkpoint
manifests for age and identity, heartbeat for last step and staleness, container
logs for `OOM`, `CUDA out of memory`, tracebacks, NaN, and Inf, and `docker
inspect` for exit/OOM status. Mark stale after two expected checkpoint intervals
or 30 minutes without heartbeat advancement, whichever is longer.

The CLI accepts `--generation-root`, `--container`, and `--output-root`; it never
mutates or stops the container.

- [ ] **Step 4: Run focused tests and validation**

```powershell
uv run pytest tests/operations/test_universal_training_monitor.py tests/scripts/test_monitor_universal_training.py tests/rl/test_tensorboard_logging.py tests/telemetry -q
uv run ruff check trade_rl/operations/universal_training_monitor.py scripts/monitor_universal_training.py tests/operations/test_universal_training_monitor.py tests/scripts/test_monitor_universal_training.py
uv run mypy trade_rl/operations/universal_training_monitor.py scripts/monitor_universal_training.py
```

Expected: all pass.

- [ ] **Step 5: Commit**

```powershell
git add trade_rl/operations/universal_training_monitor.py scripts/monitor_universal_training.py tests/operations/test_universal_training_monitor.py tests/scripts/test_monitor_universal_training.py
git commit -m "feat: monitor universal reward trends"
```

### Task 3: Immutable Docker Generation Launcher

**Files:**
- Create: `compose.universal-training.yaml`
- Create: `scripts/run_universal_training_generation.py`
- Modify: `Dockerfile.training:43-58`
- Test: `tests/scripts/test_run_universal_training_generation.py`
- Modify: `tests/examples/test_docker_training_assets.py`
- Modify: `tests/test_training_compose_contract.py`

**Interfaces:**
- Produces: one preflight command and one detached full-training container per
  generation; prints container name, image, generation root, and all digests.
- Uses external Docker network `trade_rl_default` and database host
  `trade_rl_db:5432`.
- Test helpers replace the command runner with an ordered call recorder and stub
  Git/digest functions with fixed valid values; `mark_git_dirty` changes only the
  porcelain-status result.

- [ ] **Step 1: Write failing provenance and compose-contract tests**

```python
def test_launcher_builds_clean_digest_bound_image_and_generation(monkeypatch, tmp_path: Path) -> None:
    calls = capture_commands(monkeypatch)
    result = launch_generation(project_root=tmp_path, generation="universal-u6-20260812T120000Z")
    build = next(call for call in calls if call[:2] == ("docker", "build"))
    assert "TRADE_RL_GIT_COMMIT" in " ".join(build)
    assert "TRADE_RL_SOURCE_TREE_DIGEST" in " ".join(build)
    assert "TRADE_RL_LOCKFILE_DIGEST" in " ".join(build)
    assert "TRADE_RL_RUNTIME_MANIFEST_DIGEST" in " ".join(build)
    assert result.container_name == "trade-rl-universal-u6-20260812T120000Z"


def test_launcher_rejects_dirty_tree_and_existing_generation(monkeypatch, tmp_path: Path) -> None:
    mark_git_dirty(monkeypatch)
    with pytest.raises(RuntimeError, match="clean Git tree"):
        launch_generation(project_root=tmp_path, generation="generation-a")
```

Compose tests assert `gpus: all`, the external database network, named training
volume, a read-only nested bind from
`${TRADE_RL_UNIVERSAL_ARTIFACT_ROOT:-./artifacts/universal}` to
`/workspace/var/universal`, no source-code bind mount, restart policy `no`, and
the exact canonical three config paths. Relative artifact paths in the runtime
manifest resolve from `/workspace/var/universal` exactly as they do from the host
manifest directory.

- [ ] **Step 2: Run focused tests and confirm RED**

```powershell
uv run pytest tests/scripts/test_run_universal_training_generation.py tests/examples/test_docker_training_assets.py tests/test_training_compose_contract.py -q
```

Expected: launcher/compose imports or asset assertions fail.

- [ ] **Step 3: Implement provenance-bound build, preflight, and detached launch**

Add `COPY --chown=trainer:trainer scripts ./scripts` to the training image and a
build arg/label for `TRADE_RL_RUNTIME_MANIFEST_DIGEST`. The compose trainer
command is:

```yaml
command:
  - python
  - scripts/run_universal_full_research.py
  - --selected-architecture
  - u_medium_direct
  - --ppo-config
  - examples/binance-multitimeframe/universal-u6-ppo.json
  - --lagrangian-config
  - examples/binance-multitimeframe/universal-u6-lagrangian.json
  - --discounted-config
  - examples/binance-multitimeframe/universal-u6-discounted.json
  - --runtime-factory
  - trade_rl.workflows.binance_universal_runtime:build_runtime
  - --runtime-manifest
  - /workspace/var/universal/runtime-manifest.json
  - --postgres-url
  - postgresql://trade_rl:trade_rl@trade_rl_db:5432/trade_rl
  - --frozen-metadata-root
  - /workspace/var/cache/frozen-metadata/usds-m
  - --baseline
  - supervised_allocator
  - --fold
  - "0"
  - --fold
  - "1"
  - --output-root
  - /workspace/var/runs/${TRADE_RL_RUN_GENERATION}
```

The service also sets
`TRADE_RL_RUNTIME_MANIFEST_DIGEST=${TRADE_RL_RUNTIME_MANIFEST_DIGEST:?required}`
and the launcher passes the digest obtained by loading the manifest, so full-run
heartbeats are closed over the same preflight identity.

The launcher validates a clean tree, computes `git rev-parse HEAD`,
`source_tree_digest(project_root)`, SHA-256 of `uv.lock`, and the loaded runtime
manifest digest. It builds the image, runs preflight in a one-off container,
re-loads and compares the manifest, then starts a uniquely named detached trainer
with `docker compose run --detach --name`. It refuses an existing container or
run directory and writes `launch-manifest.json` before training.

- [ ] **Step 4: Run focused tests, config validation, and CUDA smoke**

```powershell
uv run pytest tests/scripts/test_run_universal_training_generation.py tests/examples/test_docker_training_assets.py tests/test_training_compose_contract.py -q
uv run ruff check scripts/run_universal_training_generation.py tests/scripts/test_run_universal_training_generation.py
uv run mypy scripts/run_universal_training_generation.py
docker compose -f compose.universal-training.yaml config --quiet
docker run --rm --gpus all nvidia/cuda:12.1.1-base-ubuntu22.04 nvidia-smi
```

Expected: tests/static/config checks pass and the CUDA container lists the GPU.

- [ ] **Step 5: Commit**

```powershell
git add compose.universal-training.yaml scripts/run_universal_training_generation.py Dockerfile.training tests/scripts/test_run_universal_training_generation.py tests/examples/test_docker_training_assets.py tests/test_training_compose_contract.py
git commit -m "feat: launch immutable universal training generations"
```

### Task 4: Smoke, Full Launch, Supervision, Repair, and Completion Audit

**Files:**
- Runtime artifacts: `artifacts/universal/runtime-manifest.json`
- Runtime artifacts: Docker volume `/workspace/var/runs/<generation>`
- Evidence: `artifacts/universal/monitor/<generation>/monitor-snapshot.json`
- Evidence: `artifacts/universal/monitor/<generation>/reward-trends.json`

**Interfaces:**
- Consumes: all prior plans and canonical U6 configs.
- Produces: final policies/checkpoints/telemetry for algorithms × seeds,
  `universal-training.json` per algorithm, and
  `universal-full-research-training.json` for the complete comparison.

- [ ] **Step 1: Run the complete software validation gate serially**

```powershell
uv run pytest -q
uv run ruff check trade_rl scripts tests
uv run mypy trade_rl scripts
git status --short
```

Expected: all tests, lint, and types pass; tree is clean. If full coverage is a
repository gate, run its configured coverage command and require the configured
threshold before building the image.

- [ ] **Step 2: Run named CPU and CUDA smoke generations**

Create temporary smoke copies of the three configs with one seed, one environment,
three PPO updates, and `device=cpu` for CPU; repeat with `device=cuda`. Launch them
with generation IDs ending `-cpu-smoke` and `-cuda-smoke`. Verify:

```powershell
uv run python scripts/monitor_universal_training.py --generation-root artifacts/universal/smoke/cuda --container trade-rl-universal-cuda-smoke --output-root artifacts/universal/monitor/cuda-smoke
```

Expected: exit 0, three algorithm manifests, one policy each, advancing
heartbeats, non-empty TensorBoard reward tags, non-empty append-only telemetry,
and no OOM/NaN/Inf. Smoke artifacts are explicitly excluded from completion.

- [ ] **Step 3: Launch the canonical full generation**

```powershell
$generation = "universal-u6-full-$(Get-Date -AsUTC -Format 'yyyyMMddTHHmmssZ')"
uv run python scripts/run_universal_training_generation.py --generation $generation --compose-file compose.universal-training.yaml --runtime-manifest artifacts/universal/runtime-manifest.json
```

Expected: prints a unique running container, immutable image identity, generation
root, runtime-manifest digest, source digest, and lockfile digest. Record this
JSON as the authoritative launch evidence.

- [ ] **Step 4: Monitor at every rollout/checkpoint boundary until terminal**

Run the monitor repeatedly without stopping a healthy container:

```powershell
uv run python scripts/monitor_universal_training.py --generation-root "artifacts/universal/runs/$generation" --container "trade-rl-$generation" --output-root "artifacts/universal/monitor/$generation"
docker inspect "trade-rl-$generation" --format '{{json .State}}'
docker logs --tail 400 "trade-rl-$generation"
nvidia-smi
```

At each snapshot inspect total reward, every reward component, policy vs baseline
portfolio value, rolling-growth gap, drawdown, costs, constraint metrics,
approximate KL, explained variance, policy/value loss, entropy, action standard
deviation, per-symbol sampling, throughput, memory, and checkpoint age. Do not
declare health from container state alone.

- [ ] **Step 5: Apply the evidence-preserving repair loop for any failure**

Before changing code, copy the failed generation's launch manifest, last monitor
snapshot, logs, Docker state, and checkpoint manifests into its evidence
directory. Reproduce the smallest truthful failure, invoke
`superpowers:systematic-debugging`, add a failing regression test, implement the
root-cause fix under TDD, rerun the complete relevant gate, commit, rebuild a new
image, and launch a new generation ID. Never overwrite or delete the failed run.
Resume only when checkpoint identity verification passes; otherwise restart that
algorithm/seed.

- [ ] **Step 6: Audit completion against every required member and artifact**

Require all nine members:

```python
expected = {
    (algorithm, seed)
    for algorithm in ("ppo", "lagrangian", "discounted")
    for seed in (0, 1, 2)
}
assert observed_members == expected
assert all(member.actual_timesteps >= 524_288 for member in members)
assert all(member.policy_digest and member.final_checkpoint_digest for member in members)
assert all(member.telemetry_records > 0 and member.reward_tag_count > 0 for member in members)
assert final_manifest["selected_architecture"] == "u_medium_direct"
assert final_manifest["research_success"] is False
```

Re-hash every policy, checkpoint, dataset, normalizer, runtime manifest, image
identity, and final manifest. Compare algorithm non-training surfaces and require
only the authored algorithm/gamma differences. Confirm the final container exit
code is 0 and `OOMKilled=false`. Produce a concise handoff summary of intermediate
data quality and reward trends, including negative or inconclusive findings.

## Plan Completion Gate

The goal is not complete until the full-generation audit—not a smoke run—proves
all nine members and the comparison manifest. If the process remains running,
continue monitoring. If it fails, execute the repair loop. Mark completion only
after current files, Docker state, hashes, telemetry, and final manifests jointly
prove the requested end state.
