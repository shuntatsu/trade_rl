# Causal Alpha V3 Manual Launcher Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an owner-only, immutable, detached-container control path for fresh Causal Alpha V3 Signal Contract V2 real-data generations with `start`, `status`, `collect`, and `stop` operations.

**Architecture:** Keep scientific V3 code unchanged. A small Python lifecycle controller owns provenance, trusted paths, Docker state, outcome classification, and retention; a dedicated Compose file owns the fixed V3 command and durable mounts; a `workflow_dispatch` workflow exposes only generation and operation to the repository owner on `main`.

**Tech Stack:** Python 3.12, Docker/Compose, GitHub Actions, existing V3 research CLI/runtime manifest contracts, pytest, Ruff, Mypy, import-linter.

## Global Constraints

- Branch starts directly from `main`; do not depend on PR #410 or #411.
- Fixed research config: `examples/binance/universal-causal-alpha-v3-research.json`.
- Fixed run config: `examples/binance-multitimeframe/universal-u6-ppo.json`.
- Runtime artifact root comes only from trusted environment/repository configuration.
- Persistent output root in container: `/workspace/var/runs/<generation>`.
- Existing V3 Signal/Selection/Admission numerical semantics and U6 Reward/Risk/Execution/Training semantics are non-goals and must not change.
- Scientific outcomes `0/2/3/4` must be represented separately from launcher execution validity.
- Source run data is never removed merely because collection succeeds.
- Privileged workflow remains owner-only, `main`-only, read-only permissions, immutable action SHAs, no PR trigger.

---

### Task 1: Lifecycle controller contracts and fail-closed state transitions

**Files:**
- Create: `scripts/control_causal_alpha_v3_research_generation.py`
- Create: `tests/scripts/test_control_causal_alpha_v3_research_generation.py`

**Interfaces:**
- Produces `CausalAlphaV3Launch` immutable launch evidence with `to_payload()`.
- Produces `classify_research_outcome(exit_code: int, *, operator_stopped: bool = False) -> tuple[str, str]` returning `(execution_status, research_outcome)`.
- Produces `start_generation(...)`, `status_generation(...)`, `collect_generation(...)`, `stop_generation(...)`.
- CLI: `--operation start|status|collect|stop --generation NAME` plus trusted-path options/defaults sourced from environment; arbitrary dispatch paths are not part of the public workflow.

- [ ] **Step 1: Write RED tests for generation grammar, research outcome classification, and start refusal**

```python
@pytest.mark.parametrize("value", ("../x", "x/y", "x y", "", "."))
def test_generation_rejects_unsafe_segments(value: str) -> None:
    with pytest.raises(ValueError, match="generation"):
        validate_generation(value)

@pytest.mark.parametrize(
    ("code", "execution", "outcome"),
    ((0, "completed", "admitted"), (2, "completed", "signal_rejected"),
     (3, "completed", "selection_rejected"), (4, "completed", "admission_rejected"),
     (1, "failed", "unavailable"), (137, "failed", "unavailable")),
)
def test_research_exit_code_classification(code, execution, outcome) -> None:
    assert classify_research_outcome(code) == (execution, outcome)


def test_start_rejects_existing_state_container_or_output(monkeypatch, tmp_path: Path) -> None:
    fixture = launcher_fixture(monkeypatch, tmp_path)
    fixture.state_dir.mkdir(parents=True)
    with pytest.raises(FileExistsError):
        start_generation(**fixture.start_kwargs)
```

- [ ] **Step 2: Push tests only and verify valid RED**

Run in CI through the repository's existing full workflow. Expected functional RED: import/collection failure because the controller module does not exist. Static-only formatting failures do not count as RED and must be corrected without production implementation.

- [ ] **Step 3: Implement immutable launch evidence and trusted provenance resolution**

Implementation requirements:

```python
_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$")

@dataclass(frozen=True, slots=True)
class CausalAlphaV3Launch:
    generation: str
    container_name: str
    image: str
    image_id: str
    git_commit: str
    source_tree_digest: str
    lockfile_digest: str
    runtime_manifest_digest: str
    research_config_digest: str
    run_config_digest: str
    output_path: str
```

Use existing `source_tree_digest`, `load_universal_runtime_manifest`, SHA-256 file hashing, canonical JSON, and atomic writes. Require clean Git status. Build image identity from exact commit/runtime digest and validate resulting image labels before start.

- [ ] **Step 4: Implement `start` with ordered preflight and detached container creation**

Required observable order:

1. validate generation and paths;
2. reject dirty checkout;
3. resolve exact HEAD and digests;
4. reject existing state/container/output;
5. require external `trade-rl-training-data` volume and `trade_rl_default` network;
6. build/validate training-runtime image;
7. run a no-write Compose preflight that loads `/workspace/var/universal/runtime-manifest.json` and checks expected digest;
8. write immutable launch manifest atomically;
9. `docker compose ... run --detach --name <container> research`.

- [ ] **Step 5: Implement read-only `status` and terminal-only `collect`**

`status` may call Docker inspect/log metadata but may not invoke stop/rm/cp. `collect` rejects running containers, copies the source generation directory and container logs to a retained directory outside the Docker volume, writes `research-result.json`, and preserves the source container/output.

- [ ] **Step 6: Implement `stop` as operator action, not scientific rejection**

Stop only the identity-matching active container, then retain partial output/logs. Result must be `execution_status=operator_stopped`, `research_outcome=unavailable` regardless of Docker's stop exit code.

- [ ] **Step 7: Run focused tests and commit Task 1**

Run:

```bash
uv run pytest tests/scripts/test_control_causal_alpha_v3_research_generation.py -q
uv run ruff check scripts/control_causal_alpha_v3_research_generation.py tests/scripts/test_control_causal_alpha_v3_research_generation.py
uv run ruff format --check scripts/control_causal_alpha_v3_research_generation.py tests/scripts/test_control_causal_alpha_v3_research_generation.py
uv run mypy scripts/control_causal_alpha_v3_research_generation.py
```

Expected: all pass.

---

### Task 2: Dedicated V3 Compose contract

**Files:**
- Create: `docker/compose.causal-alpha-v3-research.yaml`
- Create: `tests/examples/test_causal_alpha_v3_research_compose.py`

**Interfaces:**
- Service name: `research`.
- Fixed command invokes `scripts/run_universal_causal_alpha_v3_research.py` with the authored V2 config, U6 PPO run config, maintained runtime factory, mounted runtime manifest, frozen metadata root, and generation output root.
- Uses external volume `trade-rl-training-data`, trusted artifact root bind read-only, external network `trade_rl_default`, and `restart: "no"`.

- [ ] **Step 1: Write RED Compose contract tests**

Tests parse YAML and assert exact service command tokens, `gpus: all`, persistent `/workspace/var`, read-only `/workspace/var/universal`, external network, no checkout/source bind, and no variable config path for research/run configs.

- [ ] **Step 2: Verify RED because Compose file is absent**

Run:

```bash
uv run pytest tests/examples/test_causal_alpha_v3_research_compose.py -q
```

- [ ] **Step 3: Implement minimal Compose file**

The image/build args mirror existing `compose.universal-training.yaml`, but the command is V3-specific and the output root is `/workspace/var/runs/${TRADE_RL_RUN_GENERATION:?required}`. No scientific thresholds or config values are duplicated into Compose.

- [ ] **Step 4: Run Compose contract + existing Docker asset regressions and commit**

```bash
uv run pytest tests/examples/test_causal_alpha_v3_research_compose.py tests/examples/test_docker_training_assets.py tests/test_training_compose_contract.py -q
```

---

### Task 3: Owner-only GitHub Actions control surface

**Files:**
- Create: `.github/workflows/causal-alpha-v3-research.yml`
- Create: `tests/architecture/test_causal_alpha_v3_research_workflow.py`

**Interfaces:**
- `workflow_dispatch` inputs: `operation` choice `start|status|collect|stop`; `generation` string.
- Job runner: `[self-hosted, linux, x64, gpu, nvidia]`.
- Environment: `gpu-full-training`.
- Guard includes `github.actor == github.repository_owner` and `github.ref == 'refs/heads/main'`.
- Permissions: `contents: read` only.
- Checkout is immutable SHA and exact `${{ github.sha }}`, `persist-credentials: false`.

- [ ] **Step 1: Write RED workflow structure/security tests**

Tests assert no `pull_request`/`pull_request_target`, only the two dispatch inputs, no arbitrary path/config input, exact runner/environment/guard, pinned Actions, and invocation of the lifecycle controller. `collect` and `stop` must upload retained evidence with `if: always()` or an equivalent failure-safe condition.

- [ ] **Step 2: Verify RED with workflow absent**

```bash
uv run pytest tests/architecture/test_causal_alpha_v3_research_workflow.py tests/architecture/test_workflow_security.py -q
```

- [ ] **Step 3: Implement workflow**

Resolve trusted roots only from `${{ vars.TRADE_RL_UNIVERSAL_ARTIFACT_ROOT }}` and optional `${{ vars.TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT }}`. Do not interpolate dispatch input into shell except as a quoted environment value consumed by Python argument parsing.

Use a concurrency group scoped to the V3 control plane so two lifecycle mutations cannot overlap. For `start`, optionally require `nvidia-smi` and a container CUDA probe before launch; `status/collect/stop` must not rebuild scientific state.

- [ ] **Step 4: Run workflow security/architecture tests and commit**

```bash
uv run pytest tests/architecture/test_causal_alpha_v3_research_workflow.py tests/architecture/test_workflow_security.py tests/architecture/test_gpu_workflow_consolidation.py -q
uv run python .github/check_workflow_security.py .
```

---

### Task 4: Documentation, falsification, exact-head verification, and PR readiness

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Modify: `docs/README.md` only if an operations link is required by current docs conventions.
- Keep design and plan documents already on branch.

**Interfaces:**
- Document dispatch semantics, trusted variables, generation immutability, exit/outcome split, collection behavior, and the fact that real GPU/data prerequisites remain locally/private-runner verified.

- [ ] **Step 1: Add documentation contract tests if existing docs tests do not cover the launcher**

Assert docs contain workflow path/name, `start/status/collect/stop`, scientific exit-code meanings, and explicit statement that launcher success is not profitability/Production GO.

- [ ] **Step 2: Falsification review from original acceptance criteria**

Add RED regressions before fixes for any reproduced defect. Explicitly attempt:

- `../` and shell-metacharacter generation names;
- trusted root symlink/path escape if applicable;
- dirty-tree launch;
- stale/mismatched launch manifest;
- foreign container with expected name but wrong labels/image;
- collection while running;
- scientific exit 2/3/4 being treated as infrastructure failure;
- operator stop being treated as scientific rejection;
- retention failure followed by accidental source removal;
- status path causing mutation;
- workflow dispatch from PR/non-main/non-owner;
- arbitrary config/path injection via workflow inputs.

- [ ] **Step 3: Self-review final diff**

Confirm no changes under existing V3 numerical/gate/selection/admission modules, RL reward/risk/execution modules, or U6 authored configs. Confirm no debug code, temporary workflows, secrets, generated run artifacts, or state files are committed.

- [ ] **Step 4: Run exact-head full quality gate**

Required evidence:

```text
Targeted launcher/controller/compose/workflow tests
Ruff
Format
Mypy
Import architecture
Dead-code check
Full pytest + coverage ratchet
Critical branch coverage
Ubuntu compatibility
Windows compatibility
Training image/package identity
GitHub Actions exact final HEAD
```

PostgreSQL Catalog is required only if final changed paths trigger that workflow; otherwise record it as path-filter non-applicable rather than claiming success.

- [ ] **Step 5: Create/update Draft PR and mark Ready only after exact-final-head verification**

PR body must include What, Why, acceptance criteria, design decisions, RED/GREEN evidence, falsification findings, exact-head CI, unverified real-run prerequisites, and residual risks. Do not merge into `main` without explicit user authorization.
