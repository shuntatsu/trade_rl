# SB3 Training Boundary Split Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extract runtime, vector-environment, and behavior-cloning helper implementations from `trade_rl.integrations.sb3_training` while preserving `StableBaselines3Backend`, all current behavior, and compatibility imports.

**Architecture:** Add three focused integration modules and import their private helper symbols back into `sb3_training.py` without wrappers. Lock ownership, reverse-dependency, compatibility-identity, and coordinator-size contracts with an AST-based architecture test. Do not change algorithms, serialized identities, error messages, worker defaults, or training data flow.

**Tech Stack:** Python 3.12, Gymnasium, NumPy, Stable-Baselines3 adapters, Pytest, Ruff, MyPy, Import Linter, GitHub Actions.

## Global Constraints

- Base the work on exact `main` commit `b0664257b7353605342247e62caba27e6965ca0c`.
- Do not modify the files changed by Draft PR #385.
- `StableBaselines3Backend` stays in `trade_rl/integrations/sb3_training.py`.
- Existing underscore-prefixed imports from `sb3_training.py` remain valid and resolve to the canonical moved objects.
- New modules must not import `trade_rl.integrations.sb3_training`.
- Preserve current exception types, exception messages, environment-variable defaults, `spawn` process mode, artifact schemas, and output payloads.
- Final `sb3_training.py` size must be below 61,440 bytes.
- No new DTO, protocol, checkpoint schema, compatibility layer, or dependency is introduced.

---

## File map

**Create**

- `trade_rl/integrations/sb3_runtime.py`: runtime/resource policy and optional Oracle accelerator resolution.
- `trade_rl/integrations/sb3_environment.py`: training-info filtering, vector-environment assembly, and structured-export reset normalization.
- `trade_rl/integrations/sb3_behavior_cloning.py`: coordinator-independent BC identity, configuration, label, gate, and policy-candidate helpers.
- `tests/architecture/test_sb3_training_boundaries.py`: ownership and compatibility ratchets.

**Modify**

- `trade_rl/integrations/sb3_training.py`: remove moved definitions and import canonical symbols under their existing names.

**Verify without modifying unless a real defect is found**

- `tests/integrations/test_sequence_runtime_acceleration.py`
- `tests/integrations/test_action_head_bc_routing.py`
- `tests/integrations/test_parallel_sequence_subprocess_smoke.py`
- `tests/integrations/test_sb3_training.py`
- `tests/integrations/test_sb3_training_transfer.py`
- `tests/integrations/test_sb3_lagrangian_probe_backend.py`
- `tests/integrations/test_sb3_cost_critic_backend.py`
- `tests/integrations/test_sb3_lagrangian_backend.py`
- `tests/architecture/test_ownership_boundaries.py`

---

### Task 1: Add the RED ownership contract

**Files:**
- Create: `tests/architecture/test_sb3_training_boundaries.py`

**Interfaces:**
- Consumes: repository paths from `tests.architecture.repository_paths`.
- Produces: an exact list of helper names owned by each new module and a 60 KiB coordinator size ratchet.

- [ ] **Step 1: Write the failing architecture test**

Create a test with these exact ownership sets:

```python
RUNTIME_HELPERS = {
    "_lagrangian_probe_worker_count",
    "_oracle_solver_config",
    "_oracle_accelerator_backend",
    "_teacher_worker_count",
    "_oracle_episode_sampling_config",
    "_configure_torch_cuda_runtime",
    "_configure_sequence_runtime",
}

ENVIRONMENT_HELPERS = {
    "_HEAVY_TRAINING_INFO_KEYS",
    "_compact_training_info",
    "_TrainingInfoFilter",
    "_filtered_training_environment",
    "_build_training_environment",
    "_effective_vector_environment_kind",
    "_compact_filtered_training_environment",
    "_build_parallel_sequence_training_environment",
    "_reset_observation_for_export",
}

BEHAVIOR_CLONING_HELPERS = {
    "_teacher_cache_key",
    "_TeacherIdentity",
    "_behavior_cloning_quality",
    "_resolve_behavior_cloning_seed",
    "_save_behavior_cloning_policy_candidate",
    "_restore_member_seed_after_behavior_cloning",
    "_required_hierarchical_config",
    "_teacher_change_labels",
    "_uses_hierarchical_actor_head",
    "_hierarchical_teacher_labels",
    "_hierarchical_behavior_cloning_config",
    "_behavior_cloning_gate_thresholds",
    "_evaluate_hierarchical_behavior_cloning_gate",
    "_enforce_behavior_cloning_gates",
}
```

Implement helpers that parse each module with `ast`, collect top-level definitions/assignments, and collect import targets. Add tests that require:

```python
assert RUNTIME_HELPERS <= _defined_names(runtime_path)
assert ENVIRONMENT_HELPERS <= _defined_names(environment_path)
assert BEHAVIOR_CLONING_HELPERS <= _defined_names(bc_path)
assert not (RUNTIME_HELPERS | ENVIRONMENT_HELPERS | BEHAVIOR_CLONING_HELPERS) & _defined_names(training_path)
assert training_path.stat().st_size < 61_440
```

Import all four modules and require, for every helper name:

```python
assert getattr(sb3_training, name) is getattr(owner_module, name)
```

Parse the three new modules and reject imports equal to or prefixed by `trade_rl.integrations.sb3_training`.

- [ ] **Step 2: Run the focused test and capture RED**

Run:

```bash
uv run pytest tests/architecture/test_sb3_training_boundaries.py -q
```

Expected: collection or assertion failure because the three owner modules do not yet exist and the definitions remain in `sb3_training.py`.

- [ ] **Step 3: Commit the RED test**

```bash
git add tests/architecture/test_sb3_training_boundaries.py
git commit -m "test: lock SB3 training ownership boundaries"
```

---

### Task 2: Extract runtime and resource policy

**Files:**
- Create: `trade_rl/integrations/sb3_runtime.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/architecture/test_sb3_training_boundaries.py`
- Verify: `tests/integrations/test_sequence_runtime_acceleration.py`

**Interfaces:**
- Produces the exact seven `RUNTIME_HELPERS` symbols listed in Task 1.
- `sb3_training.py` imports those symbols directly; no wrappers or aliases through assignment.

- [ ] **Step 1: Move the exact runtime implementations**

Create `sb3_runtime.py` with the existing implementations unchanged. Use these imports:

```python
from __future__ import annotations

import os
from typing import Any, cast

import numpy as np

from trade_rl.learning.episode_oracle_bc import (
    OracleEpisodeSamplingConfig,
    oracle_episode_sampling_config,
)
from trade_rl.learning.oracle_bellman_contracts import (
    CompileMode,
    OracleSolverConfig,
    SolverSelection,
)
from trade_rl.learning.oracle_solver import OracleBatchBackend
from trade_rl.rl.training import ResidualTrainingConfig
from trade_rl.rl.training_modes import CudaRuntimeMode
```

Keep the lazy import of `solve_torch_cuda_oracle_batch` inside `_oracle_accelerator_backend`.

- [ ] **Step 2: Replace coordinator definitions with direct imports**

Add this import block to `sb3_training.py`:

```python
from trade_rl.integrations.sb3_runtime import (
    _configure_sequence_runtime,
    _configure_torch_cuda_runtime,
    _lagrangian_probe_worker_count,
    _oracle_accelerator_backend,
    _oracle_episode_sampling_config,
    _oracle_solver_config,
    _teacher_worker_count,
)
```

Delete only the seven moved definitions. Do not modify their call sites.

- [ ] **Step 3: Run runtime and architecture tests**

```bash
uv run pytest \
  tests/architecture/test_sb3_training_boundaries.py \
  tests/integrations/test_sequence_runtime_acceleration.py \
  tests/integrations/test_sb3_lagrangian_probe_backend.py \
  -q
```

Expected: runtime behavioral tests pass; the architecture test remains RED only for the two not-yet-created owner modules and the final size constraint.

- [ ] **Step 4: Run static checks for touched files**

```bash
uv run ruff check --fix trade_rl/integrations/sb3_runtime.py trade_rl/integrations/sb3_training.py
uv run ruff format trade_rl/integrations/sb3_runtime.py trade_rl/integrations/sb3_training.py
uv run mypy trade_rl/integrations/sb3_runtime.py trade_rl/integrations/sb3_training.py
```

- [ ] **Step 5: Commit the runtime extraction**

```bash
git add trade_rl/integrations/sb3_runtime.py trade_rl/integrations/sb3_training.py
git commit -m "refactor: extract SB3 runtime policy"
```

---

### Task 3: Extract vector-environment ownership

**Files:**
- Create: `trade_rl/integrations/sb3_environment.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/architecture/test_sb3_training_boundaries.py`
- Verify: `tests/integrations/test_parallel_sequence_subprocess_smoke.py`

**Interfaces:**
- Produces the exact nine `ENVIRONMENT_HELPERS` symbols listed in Task 1.
- Preserves `SubprocVecEnv(..., start_method="spawn")` and cleanup on `ParallelSequenceVecEnv` construction failure.

- [ ] **Step 1: Move info filtering and environment assembly**

Create `sb3_environment.py` with the existing constant, wrapper class, and functions unchanged. Use imports equivalent to:

```python
from __future__ import annotations

from collections.abc import Callable, Mapping
from functools import partial
from typing import Any

import gymnasium as gym
import numpy as np

from trade_rl.rl.training import ResidualTrainingConfig
```

Retain lazy imports for Stable-Baselines3 vector environments, `sequence_policy_plane_materialization`, and `ParallelSequenceVecEnv`.

- [ ] **Step 2: Import the canonical environment helpers into the coordinator**

Add:

```python
from trade_rl.integrations.sb3_environment import (
    _TrainingInfoFilter,
    _build_parallel_sequence_training_environment,
    _build_training_environment,
    _compact_filtered_training_environment,
    _compact_training_info,
    _effective_vector_environment_kind,
    _filtered_training_environment,
    _HEAVY_TRAINING_INFO_KEYS,
    _reset_observation_for_export,
)
```

Delete the moved definitions and constant from `sb3_training.py`. Do not alter backend call sites.

- [ ] **Step 3: Run environment-focused tests**

```bash
uv run pytest \
  tests/architecture/test_sb3_training_boundaries.py \
  tests/integrations/test_parallel_sequence_subprocess_smoke.py \
  tests/integrations/test_sb3_training_performance.py \
  tests/integrations/test_sb3_constraint_info.py \
  -q
```

Expected: behavioral tests pass; the architecture test remains RED only for the BC module and possibly the size ratchet.

- [ ] **Step 4: Run static checks and commit**

```bash
uv run ruff check --fix trade_rl/integrations/sb3_environment.py trade_rl/integrations/sb3_training.py
uv run ruff format trade_rl/integrations/sb3_environment.py trade_rl/integrations/sb3_training.py
uv run mypy trade_rl/integrations/sb3_environment.py trade_rl/integrations/sb3_training.py
git add trade_rl/integrations/sb3_environment.py trade_rl/integrations/sb3_training.py
git commit -m "refactor: extract SB3 environment assembly"
```

---

### Task 4: Extract behavior-cloning helper ownership

**Files:**
- Create: `trade_rl/integrations/sb3_behavior_cloning.py`
- Modify: `trade_rl/integrations/sb3_training.py`
- Test: `tests/architecture/test_sb3_training_boundaries.py`
- Verify: BC integration tests.

**Interfaces:**
- Produces the exact fourteen `BEHAVIOR_CLONING_HELPERS` symbols listed in Task 1.
- Uses existing `BehaviorCloningConfig`, gate DTOs, teacher labels, content/file digests, and policy persistence contracts.

- [ ] **Step 1: Move the exact BC helper implementations**

Create `sb3_behavior_cloning.py` with imports equivalent to:

```python
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import file_digest
from trade_rl.learning import (
    BehaviorCloningConfig,
    BehaviorCloningGateEvaluation,
    BehaviorCloningGateThresholds,
    SupervisedPolicyDataset,
    evaluate_behavior_cloning_gates,
)
from trade_rl.learning.hierarchical_teacher_labels import (
    HierarchicalTeacherLabels,
    build_hierarchical_teacher_labels,
)
from trade_rl.rl.checkpointing import save_policy_without_runtime_state
from trade_rl.rl.training import ResidualTrainingConfig
```

Copy the existing implementations without changing numeric formulas, thresholds, messages, or serialization paths.

- [ ] **Step 2: Import moved BC helpers into `sb3_training.py`**

Add a direct import block containing all fourteen names. Delete their definitions from the coordinator without changing call sites.

- [ ] **Step 3: Run BC and architecture tests**

```bash
uv run pytest \
  tests/architecture/test_sb3_training_boundaries.py \
  tests/integrations/test_action_head_bc_routing.py \
  tests/integrations/test_behavior_cloning_parallelism.py \
  tests/integrations/test_behavior_cloning_split_identity.py \
  tests/integrations/test_sb3_training.py \
  tests/learning/test_episode_teacher_integration.py \
  -q
```

If either named BC test file does not exist on the exact head, replace it with the closest existing test returned by repository search; do not create a compatibility filename solely for the command.

Expected: all selected tests pass, including the ownership test and the 60 KiB size ratchet.

- [ ] **Step 4: Run static checks and commit**

```bash
uv run ruff check --fix \
  trade_rl/integrations/sb3_behavior_cloning.py \
  trade_rl/integrations/sb3_training.py
uv run ruff format \
  trade_rl/integrations/sb3_behavior_cloning.py \
  trade_rl/integrations/sb3_training.py
uv run mypy \
  trade_rl/integrations/sb3_behavior_cloning.py \
  trade_rl/integrations/sb3_training.py
git add trade_rl/integrations/sb3_behavior_cloning.py trade_rl/integrations/sb3_training.py
git commit -m "refactor: extract SB3 behavior cloning helpers"
```

---

### Task 5: Coordinator cleanup and focused regression gate

**Files:**
- Modify only if required: `trade_rl/integrations/sb3_training.py`
- Verify all touched files and focused suites.

**Interfaces:**
- Produces a coordinator that owns orchestration and `StableBaselines3Backend`, with direct imports for all extracted helpers.

- [ ] **Step 1: Remove unused imports and format**

```bash
uv run ruff check --fix \
  trade_rl/integrations/sb3_training.py \
  trade_rl/integrations/sb3_runtime.py \
  trade_rl/integrations/sb3_environment.py \
  trade_rl/integrations/sb3_behavior_cloning.py \
  tests/architecture/test_sb3_training_boundaries.py
uv run ruff format \
  trade_rl/integrations/sb3_training.py \
  trade_rl/integrations/sb3_runtime.py \
  trade_rl/integrations/sb3_environment.py \
  trade_rl/integrations/sb3_behavior_cloning.py \
  tests/architecture/test_sb3_training_boundaries.py
```

- [ ] **Step 2: Run the complete focused regression set**

```bash
uv run pytest \
  tests/architecture/test_sb3_training_boundaries.py \
  tests/architecture/test_ownership_boundaries.py \
  tests/integrations/test_sequence_runtime_acceleration.py \
  tests/integrations/test_action_head_bc_routing.py \
  tests/integrations/test_parallel_sequence_subprocess_smoke.py \
  tests/integrations/test_sb3_training.py \
  tests/integrations/test_sb3_training_transfer.py \
  tests/integrations/test_sb3_lagrangian_probe_backend.py \
  tests/integrations/test_sb3_cost_critic_backend.py \
  tests/integrations/test_sb3_lagrangian_backend.py \
  -q
```

- [ ] **Step 3: Run architecture and type gates**

```bash
uv run mypy trade_rl tests/architecture/test_sb3_training_boundaries.py
uv run lint-imports
uv run vulture trade_rl --min-confidence 100
```

- [ ] **Step 4: Inspect the structural diff**

Require all of the following:

```text
- no algorithmic expression changed;
- no error string changed;
- no backend method signature changed;
- no call site was redirected through a wrapper;
- no new module imports sb3_training;
- sb3_training remains below 61,440 bytes;
- PR #385 files are untouched.
```

- [ ] **Step 5: Commit cleanup if the previous tasks did not already leave a clean tree**

```bash
git add trade_rl/integrations tests/architecture/test_sb3_training_boundaries.py
git commit -m "refactor: finalize SB3 training boundaries"
```

Skip this commit when there is no diff.

---

### Task 6: Final exact-head verification and PR

**Files:**
- No production changes unless verification finds a real regression.
- Create a Draft PR from `agent/split-sb3-training-boundaries` to `main`.

- [ ] **Step 1: Run repository verification on one head**

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy trade_rl
uv run lint-imports
uv run vulture trade_rl --min-confidence 100
uv run pytest --cov=trade_rl --cov-report=term-missing -q
npm test --prefix frontend -- --run
npm run typecheck --prefix frontend
npm run build --prefix frontend
```

- [ ] **Step 2: Open a Draft PR**

The PR body must include:

```text
What: extract runtime, environment, and BC helper ownership from sb3_training.
Why: reduce same-layer responsibility concentration without changing behavior.
Design: direct compatibility re-exports; no wrappers; coordinator retains backend orchestration.
Non-goals: algorithms, schemas, worker defaults, checkpoints, environment semantics, PR #385 scope.
Tests: exact commands and results.
Risks: private compatibility names remain temporarily; later phases may extract checkpoint/teacher orchestration.
```

- [ ] **Step 3: Require exact-head CI**

Require all normal CI jobs on the final PR head:

- Rebuilt Core;
- Windows compatibility;
- Ubuntu compatibility;
- Training image and non-root probe;
- frontend tests/typecheck/build/layout;
- Ruff, format, MyPy, Import Linter, Vulture;
- recovery/structured-serving smoke;
- full pytest, coverage, and critical branch coverage;
- package and lock identity.

The PostgreSQL specialist workflow is not expected to trigger because this PR does not touch its path-filtered files.

- [ ] **Step 4: Final self-review**

Read the complete diff as a reviewer. Check responsibility boundaries, imports, exception preservation, worker cleanup, private compatibility identity, dead code, and accidental changes. Fix findings and repeat focused plus full verification on the new final head.

- [ ] **Step 5: Mark Ready only after exact-head success**

Do not merge without explicit owner authorization. Production remains **NO-GO**; this is an architecture-maintenance change only.
