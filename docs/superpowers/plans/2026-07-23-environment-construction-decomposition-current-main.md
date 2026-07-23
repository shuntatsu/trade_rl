# Environment Construction Decomposition — Current Main Plan

> **Required workflow:** execute task-by-task with TDD and exact-head verification.

**Goal:** Complete the remaining `AUD-RL-001` remediation from current main without replacing PR #114's canonical observation-contract extraction.

**Base:** `cc1aac077c73c1ec5304236c0a9471b5bea9b106`

## Constraints

- Preserve the complete public constructor signature.
- Keep `EnvironmentObservationContractBuilder` unchanged.
- Preserve all digests, schemas, validation messages, reset/step behavior, and mutable-state ownership.
- Keep `ResidualMarketEnv.__init__` at or below 180 source lines.
- Do not introduce exchange routing or production authorization.

## Task 1 — Capture current-main RED

Create:

- `tests/architecture/test_environment_construction_decomposition.py`
- `tests/architecture/test_environment_constructor_public_api.py`

Require the three owner modules/classes, frozen dataclasses, explicit delegation, constructor span, low-level-symbol absence, and exact constructor parameter contract.

Expected RED:

- `environment_dependencies.py`, `environment_assembly.py`, and `environment_state.py` do not exist;
- the current constructor spans 321 lines.

Record exact head, workflow run, test counts, passing unrelated jobs, artifact IDs, and digests.

## Task 2 — Extract dependency resolution

Create:

- `trade_rl/rl/environment_dependencies.py`
- focused dependency tests in `tests/rl/test_environment_construction_services.py`

Move existing constructor branches without changing order or messages:

- trend/resolver reconciliation;
- alpha/factor identity and shape validation;
- provider minimum indices;
- risk provider fallback and identity;
- leverage/gross constraints;
- action-spec construction/validation;
- episode/decision timing;
- reward tracker and preroll.

Integrate through one immutable request/result pair.

## Task 3 — Preserve PR #114 observation ownership

Do not modify `trade_rl/rl/environment_observation_contract.py`.

In the bounded constructor, call the existing:

```python
EnvironmentObservationContractBuilder(...).build(
    minimum_start_index=dependencies.minimum_start_index
)
```

Keep its 100.0% branch-coverage threshold and existing characterization tests.

## Task 4 — Extract service assembly

Create:

- `trade_rl/rl/environment_assembly.py`
- focused assembly tests.

Assemble the existing emergency monitor, hybrid/shadow executors, sampler, and maintained execution/observation/decision/risk/reward/info/termination services. Return them in an immutable result. Do not wrap or reimplement step-time behavior.

Update the older step-decomposition architecture test so it verifies service ownership in the assembly module and continued delegation from the facade.

## Task 5 — Extract initial mutable state

Create:

- `trade_rl/rl/environment_state.py`
- focused state tests.

Create fresh books, order books, arrays, diagnostics, episode metadata, reward cache, and reset state. Return values only; explicitly assign each field in `ResidualMarketEnv`.

## Task 6 — Add full characterization

Create:

- `tests/rl/test_environment_construction_characterization.py`

Require the unchanged canonical payload SHA-256:

`9d6540b3e3d3616bbb41caff036c6ef37228af56506adb030229aead86b11de1`

This covers identities, spaces, timing, initial state, seeded reset, one seeded step, and post-step state.

## Task 7 — Coverage and full verification

Measure full-suite branch coverage for the three new modules and add a separate aggregate critical-coverage group. Preserve the existing 100.0% observation-builder threshold and every other threshold.

Run:

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q --cov=trade_rl --cov-branch --cov-report=json:coverage.json
uv run python .github/check_critical_coverage.py coverage.json pyproject.toml
```

Require exact-head success for:

- Rebuilt Core;
- Ubuntu compatibility;
- Windows compatibility;
- training image and non-root runtime;
- PostgreSQL Catalog.

## Task 8 — Verification and audit closeout

Create a verification document recording current-main RED, focused GREEN, baseline characterization, full counts/coverage, artifacts, and review findings.

After the verified implementation PR is squash-merged, update `docs/verification/2026-07-23-architecture-audit-closeout.md` in a separate docs-only PR and mark `AUD-RL-001` resolved. Retain production `NO-GO` and external paper/live/exchange limitations.