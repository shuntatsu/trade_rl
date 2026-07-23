# Environment Construction Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove construction-time policy concentration from `ResidualMarketEnv.__init__` while preserving every public environment contract and mutable-state ownership boundary.

**Architecture:** Four stateless construction services consume immutable request dataclasses and return immutable result dataclasses. `ResidualMarketEnv` applies those results, computes its unchanged digest, and remains the sole owner of episode-varying mutable state.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium, Pytest, Ruff, Mypy, existing GitHub Actions and critical branch-coverage tooling.

## Global Constraints

- Preserve the complete public `ResidualMarketEnv(...)` signature.
- Preserve all existing validation messages, schema versions, digests, spaces, reset/step results, and mutable-state ownership.
- Do not introduce dependencies, direct exchange routing, production authorization, or profitability claims.
- `ResidualMarketEnv.__init__` must be at most 180 source lines.
- Construction services must retain no global, persisted, or cross-environment mutable state.

---

### Task 1: Capture the architecture RED

**Files:**
- Create: `tests/architecture/test_environment_construction_decomposition.py`

**Interfaces:**
- Consumes: current `ResidualMarketEnv` source.
- Produces: permanent ownership and constructor-span contract.

- [ ] **Step 1: Write the failing architecture test**

Require these modules/classes:

```python
REQUIRED = {
    "trade_rl.rl.environment_dependencies": "EnvironmentDependencyResolver",
    "trade_rl.rl.environment_observation_contract": "EnvironmentObservationContractFactory",
    "trade_rl.rl.environment_assembly": "EnvironmentServiceAssembler",
    "trade_rl.rl.environment_state": "EnvironmentInitialStateFactory",
}
```

Parse `ResidualMarketEnv.__init__` with `inspect.getsourcelines()` and require a span of at most 180 lines. Require delegation names `resolve`, `build`, `assemble`, and `create`, and forbid direct constructor calls for `ObservationBuilder`, `SequenceObservationBuilder`, `MarketExecutor`, `EpisodeContractSampler`, and `BookState.zero` in the constructor source.

- [ ] **Step 2: Run the architecture test**

Run:

```bash
uv run pytest -q tests/architecture/test_environment_construction_decomposition.py
```

Expected: failure because the four modules do not exist and the constructor exceeds 180 lines.

- [ ] **Step 3: Record RED evidence**

Record the exact head, workflow run, failed assertions, passing unrelated jobs, Pytest count, artifact ID, and digest in the Draft PR body or a temporary RED PR. Do not merge a RED-only branch.

---

### Task 2: Extract dependency resolution

**Files:**
- Create: `trade_rl/rl/environment_dependencies.py`
- Create: `tests/rl/test_environment_dependencies.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentDependencyRequest: ...

@dataclass(frozen=True, slots=True)
class EnvironmentDependencies:
    trend_strategy: TrendStrategy
    market_input_resolver: MarketInputResolver | None
    alpha_enabled: bool
    alpha_artifact_digest: str | None
    alpha_contract: AlphaContract
    static_factor_basis: np.ndarray | None
    factor_basis_provider: object | None
    factor_artifact_digest: str | None
    action_spec: ActionSpec
    action_names: tuple[str, ...]
    composer: BaselineResidualComposer
    pre_trade_risk: PreTradeRisk
    portfolio_risk: PortfolioRiskModel
    portfolio_risk_inputs_provider: PortfolioRiskInputsProvider | None
    config: ResidualMarketEnvConfig
    reward_tracker: RewardTracker
    nominal_episode_bars: int
    nominal_decision_bars: int
    resolved_decision_hours: float
    minimum_start_index: int

class EnvironmentDependencyResolver:
    @staticmethod
    def resolve(request: EnvironmentDependencyRequest) -> EnvironmentDependencies: ...
```

- [ ] **Step 1: Add focused failing tests**

Cover trend/resolver mismatch, missing enabled alpha provider, artifact digest resolution, static factor shape/finite validation, provider minimum index, advanced risk-provider fallback, leverage/gross validation, target-weight symbol count, decision/episode timing, reward preroll, and successful immutable output.

- [ ] **Step 2: Run focused tests and verify failure**

```bash
uv run pytest -q tests/rl/test_environment_dependencies.py
```

Expected: import failure for the new module.

- [ ] **Step 3: Implement the resolver**

Move the existing constructor logic and private digest/factor helpers without changing branch order or exception text. Copy static factor arrays on input and return; validate SHA-256 identities through the existing `require_sha256` helper.

- [ ] **Step 4: Replace dependency logic in the facade**

Construct one `EnvironmentDependencyRequest`, call `EnvironmentDependencyResolver.resolve()`, and explicitly assign every returned field. Keep `AlphaProvider` and `FactorBasisProvider` declarations in `environment.py`.

- [ ] **Step 5: Run focused and existing identity/timing tests**

```bash
uv run pytest -q \
  tests/rl/test_environment_dependencies.py \
  tests/rl/test_environment_identity.py \
  tests/rl/test_environment_time_config.py \
  tests/rl/test_signal_artifact_environment.py \
  tests/rl/test_target_weight_action.py
```

Expected: all pass.

---

### Task 3: Extract observation-contract construction

**Files:**
- Create: `trade_rl/rl/environment_observation_contract.py`
- Create: `tests/rl/test_environment_observation_contract.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces:

```python
@dataclass(frozen=True, slots=True)
class EnvironmentObservationContractRequest: ...

@dataclass(frozen=True, slots=True)
class EnvironmentObservationContract:
    observation_builder: ObservationBuilder
    layout: ObservationLayout
    asset_active_column: int
    sequence_observation_builder: SequenceObservationBuilder | None
    sequence_policy_plane: SequencePolicyPlane | None
    sequence_layout_metadata: dict[str, object] | None
    observation_schema: str
    observation_contract_digest: str
    observation_space: spaces.Space[object]
    action_space: spaces.Box[np.ndarray]
    minimum_start_index: int

class EnvironmentObservationContractFactory:
    @staticmethod
    def build(request: EnvironmentObservationContractRequest) -> EnvironmentObservationContract: ...
```

- [ ] **Step 1: Add flat and structured RED tests**

Cover normalizer size, dataset, schema, schema digest, action digest, artifact digest, passthrough indices, sequence dataset/schema validation, sequence minimum index, keys/shapes/dtypes, and stable contract digest.

- [ ] **Step 2: Implement the factory by moving existing code exactly**

Preserve component-key sorting, dtype string generation, feature/window metadata, float16/uint8 sequence spaces, flat Box bounds, and action-space bounds.

- [ ] **Step 3: Integrate and run observation suites**

```bash
uv run pytest -q \
  tests/rl/test_environment_observation_contract.py \
  tests/rl/test_sequence_environment_config.py \
  tests/rl/test_sequence_environment.py \
  tests/rl/test_observation_parity.py \
  tests/serving/test_observation_parity.py
```

Use the exact existing filenames returned by repository search if a listed suite name differs.

---

### Task 4: Extract maintained service assembly

**Files:**
- Create: `trade_rl/rl/environment_assembly.py`
- Create: `tests/rl/test_environment_assembly.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces immutable `EnvironmentServiceAssembly` containing the emergency monitor, hybrid/shadow executors, sampler, and the existing execution, observation, decision, risk, reward, info, and termination services.

- [ ] **Step 1: Add service ownership tests**

Assert separate hybrid/shadow executor instances with equal execution-policy digests, exact existing service classes, and request-field propagation for initial capital, timing, risk providers, and signal delay.

- [ ] **Step 2: Implement stateless `EnvironmentServiceAssembler.assemble()`**

Instantiate the existing maintained collaborators only. Do not add wrappers around their step-time methods.

- [ ] **Step 3: Integrate and run environment decomposition tests**

```bash
uv run pytest -q \
  tests/rl/test_environment_assembly.py \
  tests/architecture/test_environment_decomposition.py \
  tests/rl/test_environment_timing.py \
  tests/rl/test_pending_order_environment.py \
  tests/rl/test_emergency_drawdown.py
```

---

### Task 5: Extract deterministic initial mutable state

**Files:**
- Create: `trade_rl/rl/environment_state.py`
- Create: `tests/rl/test_environment_state.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Produces `EnvironmentInitialState` with initial indices, books, episode seed/hours/mode, previous action, pending targets, order books, position age, execution state, diagnostics, reward-history cache, and `has_reset`.

- [ ] **Step 1: Add initial-state tests**

Assert cloned but distinct books, correct contract multipliers, empty order books, zero arrays with exact dtypes/shapes, independent mutable containers, and unchanged seed/hour defaults.

- [ ] **Step 2: Implement `EnvironmentInitialStateFactory.create()`**

Create values only; do not retain them on the factory and do not mutate an environment instance.

- [ ] **Step 3: Apply the returned fields explicitly in the facade**

Avoid `self.__dict__.update()` or reflection. Each mutable field remains visibly owned by `ResidualMarketEnv`.

- [ ] **Step 4: Run reset and stateful replay tests**

```bash
uv run pytest -q \
  tests/rl/test_environment_state.py \
  tests/rl/test_environment_reset.py \
  tests/rl/test_environment_initial_state.py \
  tests/rl/test_environment_replay.py
```

Resolve exact filenames by repository search where necessary.

---

### Task 6: Characterize and bound the facade

**Files:**
- Modify: `tests/architecture/test_environment_construction_decomposition.py`
- Create: `tests/rl/test_environment_construction_characterization.py`
- Modify: `trade_rl/rl/environment.py`

**Interfaces:**
- Consumes all four construction services.
- Produces the final constructor orchestration and permanent parity contract.

- [ ] **Step 1: Capture pre-refactor deterministic payload**

From unchanged `main`, record a canonical payload containing digests, timing, action names, flat/structured space signatures, initial state, seeded reset result, seeded step result, books, pending targets, order books, and diagnostics. Store only the canonical digest and key explicit assertions in the permanent test.

- [ ] **Step 2: Complete facade integration**

Keep the constructor at or below 180 source lines. Compute `_environment_digest` after dependency, observation, and service fields are assigned and before initial state is used by reset/step.

- [ ] **Step 3: Run characterization and architecture tests**

```bash
uv run pytest -q \
  tests/architecture/test_environment_construction_decomposition.py \
  tests/rl/test_environment_construction_characterization.py
```

Expected: exact parity and architecture GREEN.

- [ ] **Step 4: Run the complete environment and serving-parity regression set**

Use repository search to enumerate all tests importing `ResidualMarketEnv`, then run those suites plus simulation stateful replay and serving observation parity.

---

### Task 7: Coverage, full verification, and closeout

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-23-environment-construction-decomposition.md`
- Modify after merge in a separate docs-only PR: `docs/verification/2026-07-23-architecture-audit-closeout.md`

- [ ] **Step 1: Measure construction-module branch coverage**

Run full Pytest with JSON coverage, sum covered/total branches for the four new modules, and set a new `environment_construction` group minimum by truncating the exact observed percentage to one decimal place. Do not lower any existing threshold.

- [ ] **Step 2: Run local/static verification where available**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

- [ ] **Step 3: Run exact-head GitHub Actions**

Require success for Rebuilt Core, Ubuntu compatibility, Windows compatibility, training image, and PostgreSQL Catalog. Record test count, total/branch coverage, construction-group coverage, run IDs, artifact IDs, and digests.

- [ ] **Step 4: Review the final diff**

Confirm there is no temporary workflow, patch payload, source export, generated status file, duplicate implementation, schema change, or direct-exchange behavior.

- [ ] **Step 5: Complete and merge the PR**

Update the PR body with RED/GREEN evidence, mark ready, and squash merge with the verified exact head SHA.

- [ ] **Step 6: Update the audit closeout independently**

Submit a docs-only PR that marks `AUD-RL-001` resolved with the merged PR, constructor span, module ownership, parity evidence, and exact-head verification. Retain production `NO-GO` and the external paper/live/exchange limitations.