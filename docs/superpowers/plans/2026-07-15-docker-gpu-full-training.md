# Docker GPU Full Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the complete three-asset multi-timeframe research pipeline from a GPU-enabled Docker service whose persistent data and artifacts live in a named volume, and fail unless the sealed evaluation meets the approved return and drawdown thresholds.

**Architecture:** Add vector-environment width to the canonical training configuration and let the SB3 backend use subprocess environments while keeping seeds sequential. Package the maintained CUDA preset in Docker Compose, write CUDA and evaluation evidence into the volume-backed work root, and verify the final research gate before completion.

**Tech Stack:** Python 3.12, Stable-Baselines3 2.3.2, PyTorch 2.3.1, Gymnasium 0.29.1, Docker Compose, NVIDIA Container Toolkit, pytest, Ruff, Mypy.

## Global Constraints

- Dataset indicators remain exactly 96 causal features across `15m`, `1h`, `4h`, and `1d`.
- The policy observation width, not the raw indicator count, drives network sizing.
- Full training uses seeds `0`, `1`, and `2`; walk-forward uses two sealed folds.
- Docker named volume `trade-rl-training-data` is mounted at `/workspace/var`.
- Maintained Docker training requires CUDA and must not silently fall back to CPU.
- Final gate: sealed cost-adjusted return `> 0`, Trend-baseline uplift `>= 0`, maximum drawdown `<= 0.20`.
- Production status remains `NO-GO`.

---

### Task 1: Canonical vector-environment training configuration

**Files:**
- Modify: `trade_rl/rl/training.py`
- Test: `tests/rl/test_ensemble_training.py`
- Test: `tests/rl/test_algorithm_configs.py`

**Interfaces:**
- Produces: `ResidualTrainingConfig.n_envs: int` and digest key `n_envs`.
- Consumes: existing PPO `n_steps`, `batch_size`, and canonical digest machinery.

- [ ] **Step 1: Write failing validation and digest tests**

```python
def test_vector_environment_width_is_validated_and_digested() -> None:
    config = _config(n_envs=4, n_steps=8, batch_size=8)
    assert config.n_envs == 4
    assert config.digest_payload()["n_envs"] == 4
    with pytest.raises(ValueError, match="n_envs must be a positive integer"):
        _config(n_envs=0)

def test_ppo_batch_size_divides_vector_rollout() -> None:
    ResidualTrainingConfig(
        timesteps=32,
        gamma=0.99,
        seeds=(0,),
        n_steps=4,
        n_envs=2,
        batch_size=8,
    )
```

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/rl/test_ensemble_training.py tests/rl/test_algorithm_configs.py -q`
Expected: failure because `n_envs` is not accepted or emitted.

- [ ] **Step 3: Implement the canonical field**

Add `n_envs: int = 1`, validate it with the other positive integers, change PPO divisibility to `(n_steps * n_envs) % batch_size == 0`, round PPO timesteps to multiples of `n_steps * n_envs`, and include `n_envs` in `digest_payload()`.

- [ ] **Step 4: Run GREEN and validation**

Run: `uv run pytest tests/rl/test_ensemble_training.py tests/rl/test_algorithm_configs.py -q`
Expected: all selected tests pass.

Run: `uv run ruff check trade_rl/rl/training.py tests/rl/test_ensemble_training.py tests/rl/test_algorithm_configs.py`
Expected: exit 0.

### Task 2: SB3 subprocess rollout environments

**Files:**
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/integrations/test_sb3_training.py`

**Interfaces:**
- Consumes: `ResidualTrainingConfig.n_envs` and the existing environment factory.
- Produces: one direct environment for `n_envs == 1`, otherwise an SB3 `SubprocVecEnv` built from independent factory calls.

- [ ] **Step 1: Write failing environment-construction tests**

Create tests around a small `_build_training_environment(factory, n_envs)` seam. Assert one factory call and an unwrapped environment for width one; assert a vector environment with `num_envs == 2` for width two and close it in `finally`.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/integrations/test_sb3_training.py -q`
Expected: import failure for `_build_training_environment`.

- [ ] **Step 3: Implement the minimal builder and identity probe**

Use `DummyVecEnv` only when required by SB3 internals and `SubprocVecEnv` for `n_envs > 1`. Probe and validate a direct environment before constructing subprocess workers. Ensure every created environment is closed exactly once and preserve asset-set layout metadata from the probe.

- [ ] **Step 4: Run GREEN and focused lint/type checks**

Run: `uv run pytest tests/integrations/test_sb3_training.py tests/workflows/test_training_run.py -q`
Expected: all selected tests pass.

Run: `uv run ruff check trade_rl/integrations/sb3_training.py tests/integrations/test_sb3_training.py`
Expected: exit 0.

Run: `uv run mypy trade_rl/integrations/sb3_training.py trade_rl/rl/training.py`
Expected: exit 0.

### Task 3: CUDA-sized maintained presets

**Files:**
- Modify: `examples/binance-multitimeframe/training-full.json`
- Modify: `examples/binance-multitimeframe/walk-forward-full.json`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: maintained full configs with `device="cuda"`, `n_envs=4`, `policy_net_arch=[256,256]`, and 128-dimensional embeddings.

- [ ] **Step 1: Change tests first**

Assert both full training and fold-local candidate configurations use CUDA, four environments, `(256, 256)`, and 128-dimensional asset/global embeddings.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/examples/test_binance_multitimeframe_full_assets.py -q`
Expected: assertions show the old CPU/one-environment/128x128 preset.

- [ ] **Step 3: Update both JSON presets**

Set the exact values required by the tests without changing timesteps, seeds, folds, costs, or risk limits.

- [ ] **Step 4: Run GREEN**

Run: `uv run pytest tests/examples/test_binance_multitimeframe_full_assets.py tests/workflows/test_training_run_config.py -q`
Expected: all selected tests pass.

### Task 4: Docker named-volume and CUDA preflight contract

**Files:**
- Create: `Dockerfile.training`
- Create: `compose.training.yaml`
- Create: `.dockerignore`
- Create: `examples/binance-multitimeframe/training_cuda_preflight.py`
- Create: `examples/binance-multitimeframe/run_gpu_training_smoke.py`
- Create: `tests/examples/test_training_cuda_preflight.py`
- Create: `tests/examples/test_run_gpu_training_smoke.py`
- Create: `tests/examples/test_docker_training_assets.py`

**Interfaces:**
- Produces: Compose service `trainer`, named volume `trade-rl-training-data`, work root `/workspace/var/binance-multitimeframe-full`, and JSON CUDA preflight evidence.

- [ ] **Step 1: Write failing Docker-asset and preflight tests**

The Docker test must assert `gpus: all`, the named-volume mount, and the maintained runner command. The preflight test injects a torch-like probe and asserts that missing CUDA raises `RuntimeError`, while a visible device writes device name, capability, and total memory to JSON. The smoke-script test calls its configuration builder and asserts CUDA, four vector environments, `[256, 256]`, 128-dimensional embeddings, and the caller-supplied timestep count.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/examples/test_training_cuda_preflight.py tests/examples/test_run_gpu_training_smoke.py tests/examples/test_docker_training_assets.py -q`
Expected: failures because files and preflight module do not exist.

- [ ] **Step 3: Implement the preflight and container assets**

The image installs Python 3.12 and locked project dependencies, uses an unprivileged runtime user, and invokes preflight before the full runner. Compose mounts only the named volume for runtime data and requests all GPUs. `.dockerignore` excludes `.git`, `.venv`, `.worktrees`, caches, and `var`. The smoke script builds the repository's deterministic tiny market dataset, materializes the approved CUDA/vector/network configuration, executes one seed through the authoritative training workflow, and writes the resulting device, timestep, environment-width, policy, and checkpoint evidence.

- [ ] **Step 4: Run GREEN and build validation**

Run: `uv run pytest tests/examples/test_training_cuda_preflight.py tests/examples/test_run_gpu_training_smoke.py tests/examples/test_docker_training_assets.py -q`
Expected: all selected tests pass.

Run: `docker compose -f compose.training.yaml config`
Expected: exit 0 with resolved GPU and volume declarations.

### Task 5: Machine-readable profitability gate

**Files:**
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Create: `trade_rl/evaluation/research_gate.py`
- Create: `tests/evaluation/test_research_gate.py`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Produces: `ResearchReturnGate`, `evaluate_research_return_gate(...)`, and `research-gate.json` under the volume-backed work root.
- Consumes: sealed walk-forward aggregate evidence already published by the authoritative workflow.

- [ ] **Step 1: Write failing pure gate tests**

Cover passing metrics, zero/negative net return, negative baseline uplift, drawdown above 0.20, non-finite values, and exact boundary behavior.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/evaluation/test_research_gate.py -q`
Expected: import failure for the new gate module.

- [ ] **Step 3: Implement the pure gate and runner integration**

The gate returns structured thresholds, observed values, per-condition booleans, and `passed`. The runner reads `walk-forward.json`: it uses `selected_independent_summary.mean_fold_return` as selected net return, subtracts `baseline_independent_summary.mean_fold_return` for uplift, and computes the maximum independently reset fold drawdown from every `folds[*].selected_returns` sequence. It writes `research-gate.json`, adds it to `summary.json`, and exits non-zero after preserving artifacts when `passed` is false.

- [ ] **Step 4: Run GREEN and focused validation**

Run: `uv run pytest tests/evaluation/test_research_gate.py tests/examples/test_binance_multitimeframe_full_assets.py -q`
Expected: all selected tests pass.

Run: `uv run ruff check trade_rl/evaluation/research_gate.py examples/binance-multitimeframe/run_full_research.py tests/evaluation/test_research_gate.py`
Expected: exit 0.

### Task 6: Documentation and complete verification

**Files:**
- Modify: `README.ja.md`
- Modify: `README.md`
- Create: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Produces: exact build, start, status, log, artifact-copy, resume/retry, and cleanup commands.

- [ ] **Step 1: Document the maintained command and evidence paths**

Document `docker compose -f compose.training.yaml build trainer`, `docker compose -f compose.training.yaml run --rm trainer`, volume inspection, and artifact extraction without claiming profitability.

- [ ] **Step 2: Run repository gates**

Run: `uv run ruff check .`
Expected: exit 0.

Run: `uv run ruff format --check .`
Expected: exit 0.

Run: `uv run mypy trade_rl`
Expected: exit 0.

Run: `uv run lint-imports`
Expected: exit 0.

Run: `uv run pytest --cov=trade_rl --cov-branch`
Expected: at least 80% total coverage and zero failures.

Run: `git diff --check`
Expected: exit 0.

### Task 7: Verify canonical nested checkpoints

**Files:**
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`

**Interfaces:**
- Consumes: canonical `checkpoint_manifests(member / "checkpoints")` artifacts with `step-*/checkpoint.json` and `step-*/policy.zip`.
- Produces: full-run verification that accepts valid nested checkpoints and rejects missing or invalid checkpoint artifacts.

- [ ] **Step 1: Write a failing nested-checkpoint regression**

Build a three-member training directory in the test with non-empty run, ensemble, environment, policy files and canonical checkpoints published through `publish_checkpoint`. Call `_verify_training(root)` and require it to return normally.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/examples/test_binance_multitimeframe_full_assets.py -q`
Expected: failure `member 0 has no retained checkpoints` because the runner incorrectly searches `checkpoints/*.zip`.

- [ ] **Step 3: Use the canonical checkpoint loader**

Replace the flat glob with `checkpoint_manifests(member / "checkpoints")`. This validates both manifest identity and nested policy files rather than merely finding a filename.

- [ ] **Step 4: Run GREEN and focused validation**

Run: `uv run pytest tests/examples/test_binance_multitimeframe_full_assets.py tests/rl/test_checkpointing.py -q`
Expected: all selected tests pass.

Run: `uv run ruff check examples/binance-multitimeframe/run_full_research.py tests/examples/test_binance_multitimeframe_full_assets.py`
Expected: exit 0.

### Task 8: Preserve packaged-source Git provenance

**Files:**
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `tests/workflows/test_training_run_config.py`
- Modify: `examples/binance-multitimeframe/run_full_research.py`
- Modify: `examples/binance-multitimeframe/run_gpu_training_smoke.py`
- Modify: `tests/examples/test_binance_multitimeframe_full_assets.py`
- Modify: `tests/examples/test_run_gpu_training_smoke.py`
- Modify: `Dockerfile.training`
- Modify: `compose.training.yaml`
- Modify: `tests/examples/test_docker_training_assets.py`
- Modify: `docs/operations/docker-gpu-full-training.md`

**Interfaces:**
- Produces: `TrainingRunConfig.git_dirty: bool | None`, build-time `TRADE_RL_GIT_COMMIT`, and build-time `TRADE_RL_GIT_DIRTY`.
- Consumes: exact 40-character lowercase Git commit and boolean dirty state from the host checkout used as Docker build context.

- [ ] **Step 1: Write failing provenance tests**

Assert training config parses/digests `git_dirty`; executing with explicit `git_commit` and `git_dirty=false` works outside a Git checkout; both maintained runners inject the two required environment values; Docker build requires and exports both values.

- [ ] **Step 2: Run RED**

Run: `uv run pytest tests/workflows/test_training_run_config.py tests/examples/test_run_gpu_training_smoke.py tests/examples/test_binance_multitimeframe_full_assets.py tests/examples/test_docker_training_assets.py -q`
Expected: failures because dirty provenance and build arguments are absent.

- [ ] **Step 3: Implement fail-closed packaged provenance**

Add `git_dirty` to parsing and digest identity, pass it to `capture_runtime_provenance`, and require both environment values when the runners materialize Docker configs. Add Docker build arguments and environment variables. Document PowerShell commands that set the commit from `git rev-parse HEAD` and dirty state from `git status --porcelain` before build.

- [ ] **Step 4: Run GREEN and focused validation**

Run the RED command again, plus Ruff, Mypy, Compose config with explicit provenance variables, and the full test suite.

### Task 9: Container GPU smoke and full research execution

**Files:**
- Runtime artifacts only: Docker volume `trade-rl-training-data`.

**Interfaces:**
- Produces: CUDA evidence, deterministic dataset pair, three full member policies and checkpoints, two sealed folds, `research-gate.json`, and `summary.json`.

- [ ] **Step 1: Build and verify CUDA**

Run: `docker compose -f compose.training.yaml build trainer`
Expected: image builds successfully.

Run: `docker compose -f compose.training.yaml run --rm --entrypoint uv trainer run python examples/binance-multitimeframe/training_cuda_preflight.py --output /workspace/var/cuda-preflight.json`
Expected: JSON reports CUDA available and RTX 4050 device memory.

- [ ] **Step 2: Run a bounded vectorized GPU smoke**

Run: `docker compose -f compose.training.yaml run --rm --entrypoint uv trainer run python examples/binance-multitimeframe/run_gpu_training_smoke.py --work-root /workspace/var/gpu-smoke --timesteps 8192 --n-envs 4`
Expected: `/workspace/var/gpu-smoke/smoke-summary.json` reports `resolved_device` beginning with `cuda`, `n_envs` equal to 4, 8,192 observed timesteps, and a published policy checkpoint.

- [ ] **Step 3: Run the complete pipeline**

Run: `docker compose -f compose.training.yaml run --rm trainer`
Expected: exit 0 only when data build, three-seed full training, two-fold walk-forward, and the research gate all finish successfully.

- [ ] **Step 4: Audit final evidence**

Inspect `/workspace/var/binance-multitimeframe-full/summary.json`, `research-gate.json`, member manifests, checkpoint manifests, fold evidence, dataset IDs, CUDA evidence, and logs. If the gate fails, diagnose action collapse, turnover, entropy, KL, value loss, and seed dispersion; change only evidence-supported hyperparameters and repeat from a new run generation.
