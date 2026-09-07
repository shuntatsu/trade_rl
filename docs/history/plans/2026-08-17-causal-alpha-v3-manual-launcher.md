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
```

- [ ] **Step 2: Push tests only and verify valid RED**

Run in CI through the repository's existing full workflow. Expected functional RED: import/collection failure because the controller module does not exist. Static-only formatting failures do not count as RED and must be corrected without production implementation.

- [ ] **Step 3: Implement immutable launch evidence and trusted provenance resolution**

Use an immutable `CausalAlphaV3Launch` dataclass carrying generation, container/image identities, exact Git/source/lock/runtime/config digests, and fixed output path. Use existing `source_tree_digest`, `load_universal_runtime_manifest`, SHA-256 file hashing, canonical JSON, and atomic writes. Require clean Git status.

- [ ] **Step 4: Implement `start` with ordered preflight and detached container creation**

Required observable order: validate generation/paths; reject dirty checkout; resolve exact identities; reject existing state/container; require external volume/network; build/validate image; preflight mounted runtime manifest and absent output root; atomically write launch state; start named detached `research` service.

- [ ] **Step 5: Implement read-only `status` and terminal-only `collect`**

`status` may inspect only. `collect` rejects a running container, copies source generation output and logs into a retained directory, writes `research-result.json`, and never removes source run/container/volume.

- [ ] **Step 6: Implement `stop` as operator action, not scientific rejection**

Stop only the identity-matching active container, retain partial output/logs, and force `execution_status=operator_stopped`, `research_outcome=unavailable`.

- [ ] **Step 7: Run focused tests and commit Task 1**

```bash
uv run pytest tests/scripts/test_control_causal_alpha_v3_research_generation.py -q
uv run ruff check scripts/control_causal_alpha_v3_research_generation.py tests/scripts/test_control_causal_alpha_v3_research_generation.py
uv run ruff format --check scripts/control_causal_alpha_v3_research_generation.py tests/scripts/test_control_causal_alpha_v3_research_generation.py
uv run mypy scripts/control_causal_alpha_v3_research_generation.py
```

---

### Task 2: Dedicated V3 Compose contract

**Files:**
- Create: `docker/compose.causal-alpha-v3-research.yaml`
- Create: `tests/examples/test_causal_alpha_v3_research_compose.py`

**Interfaces:** service `research`; fixed V3 CLI command; external `trade-rl-training-data`; trusted read-only `/workspace/var/universal`; external `trade_rl_default`; `restart: "no"`.

- [ ] Write RED YAML contract tests for exact command, GPU, mounts, network, and no variable config paths.
- [ ] Verify RED because Compose file is absent.
- [ ] Implement minimal Compose using the maintained training-runtime image/build args.
- [ ] Run focused Compose and existing Docker asset regressions.

---

### Task 3: Owner-only GitHub Actions control surface

**Files:**
- Create: `.github/workflows/causal-alpha-v3-research.yml`
- Create: `tests/architecture/test_causal_alpha_v3_research_workflow.py`

**Interfaces:** `workflow_dispatch` only; inputs `operation` and `generation`; privileged GPU labels; `gpu-full-training`; owner/main guard; `contents: read`; exact SHA checkout with credentials disabled.

- [ ] Write RED workflow/security tests proving no PR trigger, no arbitrary path/config input, immutable actions, lifecycle controller invocation, and failure-safe evidence upload.
- [ ] Verify RED with workflow absent.
- [ ] Implement workflow using trusted repository variables only for infrastructure roots.
- [ ] Run workflow security, architecture, and GPU-workflow consolidation checks.

---

### Task 4: Documentation, falsification, exact-head verification, and PR readiness

**Files:**
- Modify: `docs/UNIVERSAL_TRAINING.md`
- Keep design and plan under `docs/implementation-plans/`.

- [ ] Add/extend documentation contract tests for lifecycle operations, exit semantics, and NO-GO boundary.
- [ ] Falsify path injection, dirty/stale identity, foreign container, premature collect, research-rejection misclassification, operator-stop misclassification, status mutation, retention/source deletion, and workflow input/actor/ref restrictions; add RED regression before any fix.
- [ ] Self-review final diff for scope, secrets, debug/temp files, and unchanged scientific modules/configs.
- [ ] Run targeted tests, Ruff, format, Mypy, import architecture, vulture, full pytest/coverage, critical coverage, Ubuntu/Windows compatibility, training image/package identity, and exact-head GitHub Actions.
- [ ] Update Draft PR with RED/GREEN/falsification/exact-head evidence and mark Ready only after final quality gate. Do not merge without explicit user authorization.
