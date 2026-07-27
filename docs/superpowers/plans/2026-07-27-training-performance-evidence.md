# Training Performance Evidence Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add deterministic, per-member training performance evidence that separates rollout collection, optimizer updates, environment stepping, policy feature extraction, sequence reconstruction, host-to-device tensor materialization, total wall time, throughput, and CUDA allocator peaks without changing the learned objective or model outputs.

**Architecture:** A package-level recorder temporarily wraps the maintained SB3 model, policy, and vector-environment call boundaries only while `model.learn()` executes. Sequence rollout materialization reports through a process-local `ContextVar`, so the compact buffer remains reusable outside training and no recorder becomes part of checkpoint state. Each trained member writes `training-performance.json`; the existing run manifest includes the file through the normal artifact closure.

**Tech Stack:** Python 3.12, PyTorch 2.3.1, Stable-Baselines3 2.3.2, Gymnasium, NumPy, pytest, canonical JSON artifacts.

## Global Constraints

- Preserve ordinary PPO, Cost Critic PPO, Lagrangian PPO, SAC, TD3, and TQC optimization semantics.
- Do not synchronize CUDA inside per-step or per-minibatch timers; synchronize only once before measurement and once before final CUDA memory capture.
- Treat component timers as nested observations: environment, feature, reconstruction, and tensor-conversion time may overlap rollout or optimization time and must not be summed into a fake residual.
- Restore every temporarily wrapped callable even when learning raises.
- Do not serialize recorder objects or temporary wrappers into checkpoints.
- Bind the evidence payload with a content digest and finite, non-negative validation.
- Continue to record existing external `nvidia-smi` smoke metrics; add allocator evidence rather than replacing it.
- Production remains `NO-GO`.

---

### Task 1: Add the deterministic performance recorder and artifact

**Files:**
- Create: `trade_rl/rl/training_performance.py`
- Test: `tests/rl/test_training_performance_evidence.py`

**Interfaces:**
- Produces: `TrainingPerformanceEvidence`, `TrainingPerformanceRecorder`, `activate_training_performance`, `measure_sequence_reconstruction`, `measure_sequence_tensor_conversion`, and `write_training_performance_evidence`.
- `TrainingPerformanceRecorder.start(torch_module: object, device: object) -> None` resets counters and CUDA peak statistics.
- `TrainingPerformanceRecorder.instrument_model(model: object) -> ContextManager[None]` temporarily times callable `collect_rollouts`, algorithm `train`, policy `extract_features`, and vector-environment `step` boundaries when present.
- `TrainingPerformanceRecorder.finish(torch_module: object, device: object, requested_environment_steps: int, observed_environment_steps: int) -> TrainingPerformanceEvidence` freezes validated evidence.

- [ ] **Step 1: Write RED tests for exact timer accumulation and restoration**

Use an injected monotonic fake clock. The fake model advances the clock by two seconds in `collect_rollouts`, three in algorithm `train`, four in `policy.extract_features`, and five in `env.step`. Explicit sequence contexts advance it by six and seven seconds. Assert exact seconds, one call per boundary, exact requested/observed steps, throughput, CPU-null CUDA fields, and restoration of original bound callables.

- [ ] **Step 2: Write RED tests for failure restoration and canonical persistence**

Raise inside a wrapped model method, assert all original callables are restored, then build a successful evidence object, write it, parse JSON, recompute `content_digest` after removing `digest`, and assert equality. Reject negative steps, non-finite durations, a second `start()`, and `finish()` before `start()`.

- [ ] **Step 3: Implement the minimal recorder**

Implement a mutable recorder with private metric totals and counts, a single start/finish lifecycle, `ContextVar` activation, and callable restoration in `finally`. Resolve CUDA through `torch_module.device(device).type`; call `cuda.synchronize`, `reset_peak_memory_stats`, `max_memory_allocated`, and `max_memory_reserved` only for a CUDA device.

- [ ] **Step 4: Run focused tests**

Run: `uv run pytest tests/rl/test_training_performance_evidence.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

Commit: `feat: add deterministic training performance evidence`.

### Task 2: Instrument sequence materialization and the authoritative SB3 backend

**Files:**
- Modify: `trade_rl/integrations/compact_rollout_buffer.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/integrations/test_compact_rollout_performance.py`
- Test: `tests/integrations/test_sb3_training_performance.py`

**Interfaces:**
- Consumes: Task 1 recorder/context functions.
- Produces: `members/member-NNN/training-performance.json` for every member that executes at least one new environment step.

- [ ] **Step 1: Write RED compact-buffer timing test**

Construct an `IndexBackedDictRolloutBuffer` through `object.__new__`, provide decision indices, a fake reconstructor, and a fake `to_torch`. Under an active recorder, assert one reconstruction call, one tensor-conversion call, exact injected durations, and unchanged returned values.

- [ ] **Step 2: Write RED backend artifact test**

Use the existing fake-SB3 pattern with a typed training probe. Make fake `learn()` invoke `collect_rollouts`, algorithm `train`, policy `extract_features`, and environment `step`, then advance `num_timesteps`. Assert the member performance artifact exists, reports the observed step count, records each call family, uses `device_type="cpu"`, has null CUDA peaks, and has a valid digest.

- [ ] **Step 3: Add compact-buffer timing seams**

Wrap only `reconstructor.reconstruct(decision_indices)` with `measure_sequence_reconstruction()` and only NumPy-to-Torch materialization with `measure_sequence_tensor_conversion()`. Preserve the existing one-materialization-per-rollout cache.

- [ ] **Step 4: Add SB3 learning instrumentation**

Immediately before `model.learn()`, capture `model.num_timesteps`, start a recorder, activate it, and instrument the model. After learning, finish with requested remaining timesteps and the observed timestep delta, then write `training-performance.json`. Do not create the file when resume has zero remaining work.

- [ ] **Step 5: Run focused integration tests**

Run: `uv run pytest tests/integrations/test_compact_rollout_performance.py tests/integrations/test_sb3_training_performance.py tests/integrations/test_sb3_training.py tests/integrations/test_sb3_cost_critic_backend.py tests/integrations/test_sb3_lagrangian_backend.py -q`
Expected: PASS.

- [ ] **Step 6: Commit**

Commit: `feat: record SB3 phase and sequence timings`.

### Task 3: Surface allocator evidence in the maintained CUDA smoke

**Files:**
- Modify: `examples/binance-multitimeframe/run_gpu_training_smoke.py`
- Modify: `tests/examples/test_run_gpu_training_smoke.py`

**Interfaces:**
- Consumes: per-member `training-performance.json` from Task 2.
- Produces: `performance.training_artifact` and `resume.performance.training_artifact` inside `gpu-training-smoke.json`.

- [ ] **Step 1: Write RED loader validation tests**

Add `_load_training_performance(member_root: Path) -> dict[str, object]`. Test a valid schema/digest payload and reject missing files, wrong schema, digest mismatch, non-positive observed steps, and non-positive wall time.

- [ ] **Step 2: Load both original and resumed member evidence**

Read each member performance artifact, include it beside the existing external wall/throughput/`nvidia-smi` metrics, and bump the smoke schema from `gpu_sequence_target_oracle_bc_training_smoke_v5` to `gpu_sequence_target_oracle_bc_training_smoke_v6`.

- [ ] **Step 3: Run smoke-asset tests**

Run: `uv run pytest tests/examples/test_run_gpu_training_smoke.py tests/examples/test_docker_training_assets.py tests/examples/test_binance_multitimeframe_full_assets.py -q`
Expected: PASS.

- [ ] **Step 4: Commit**

Commit: `feat: surface allocator evidence in GPU smoke`.

### Task 4: Exact-head verification

**Files:**
- Create after verification: `docs/verification/2026-07-27-training-performance-evidence.md`

- [ ] **Step 1: Run static gates**

Run:

```bash
uv run ruff check trade_rl/rl/training_performance.py trade_rl/integrations/compact_rollout_buffer.py trade_rl/integrations/sb3_training.py examples/binance-multitimeframe/run_gpu_training_smoke.py tests/rl/test_training_performance_evidence.py tests/integrations/test_compact_rollout_performance.py tests/integrations/test_sb3_training_performance.py tests/examples/test_run_gpu_training_smoke.py
uv run ruff format --check trade_rl/rl/training_performance.py trade_rl/integrations/compact_rollout_buffer.py trade_rl/integrations/sb3_training.py examples/binance-multitimeframe/run_gpu_training_smoke.py tests/rl/test_training_performance_evidence.py tests/integrations/test_compact_rollout_performance.py tests/integrations/test_sb3_training_performance.py tests/examples/test_run_gpu_training_smoke.py
uv run mypy trade_rl/rl/training_performance.py trade_rl/integrations/compact_rollout_buffer.py trade_rl/integrations/sb3_training.py
```

Expected: all commands PASS.

- [ ] **Step 2: Run the complete repository gates**

Run:

```bash
uv run pytest -q
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
uv run lint-imports
```

Expected: all commands PASS with no reduction of the maintained coverage gates.

- [ ] **Step 3: Record honest verification**

The verification document must state that CPU CI validates schema, timing seams, restoration, artifact closure, and compatibility. It must not claim representative 4070 Ti SUPER throughput or CUDA peak values until a real GPU run populates them.

- [ ] **Step 4: Commit**

Commit: `docs: verify training performance evidence`.
