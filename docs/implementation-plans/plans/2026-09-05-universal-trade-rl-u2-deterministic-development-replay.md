# Universal Trade RL U2 Deterministic Development Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a synthetic-only, fail-closed U2 Development replay session that reuses the frozen U1 environment exactly, evaluates one canonical scope under candidate/cash/+1/-1 variants, and emits content-addressed raw net-economic evidence without opening real Development or Admission artifacts.

**Architecture:** A high-level replay session owns canonical scope reconstruction and Task 7B verified common-view datasets exactly once. Each replay call resolves one canonical scope, creates a fresh verified `UniversalTradeEnvironment`, resets it with the candidate training seed at the scope-derived start index, drives exactly the frozen 720-hour U1 horizon, and records immutable raw evidence. Gross metrics and Selection decisions remain outside this plan.

**Tech Stack:** Python 3.12, dataclasses, NumPy, Gymnasium/U1 environment, existing U0/U1/U2 workflow contracts, pytest, Ruff, MyPy, import-linter, GitHub Actions.

**Spec:**
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-deterministic-development-replay-design.md`
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-development-replay-seed-amendment.md`
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-evaluation-tile-outcome-semantics-amendment.md`
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-development-access-amendment.md`
- `docs/implementation-plans/specs/2026-09-05-universal-trade-rl-u2-evaluation-common-index-amendment.md`

## Global Constraints

- Real U2 training remains `NO-GO`.
- Real Development numeric evaluation is forbidden in this task; tests use synthetic artifacts only.
- Admission remains sealed and must never be opened by this task.
- Production remains `NO-GO`.
- Candidate inference is always `deterministic=True`.
- Evaluation seed is exactly the candidate training seed and must be one of `(0, 1, 2)`.
- Candidate/cash/constant-long/constant-short paired evidence uses the same evaluation seed, scope, dataset, U1 runtime, and paired checkpoint identity.
- U1 observation/action/reward/Risk/Execution/Accounting semantics are not modified.
- U1 episode semantics remain fixed 720 h, cash-only reset, external truncation, no terminal liquidation.
- `evaluation_stop_bar_index` is an exclusive metadata boundary; normal U1 runtime end is `outcome_stop_bar_index_exclusive - 1`.
- `BookState.returns_history` is treated as a simple-return series. Do not rename or reinterpret it as log return.
- Task 7C-1 does not define or claim gross return/gross wealth.
- Generic `walk_forward_evaluation.evaluate_range_evidence()` is not used because it reconstructs a different runtime and liquidates on end.

---

## File Structure

- Create `trade_rl/workflows/universal_trade_rl_u2_replay.py`
  - one responsibility: canonical Development replay session, policy-variant action selection, actual U1 replay, immutable raw evidence.
- Create `tests/workflows/test_universal_trade_rl_u2_replay.py`
  - pure contract/identity/fail-before-I/O tests using lightweight fakes where possible.
- Create `tests/integrations/test_universal_trade_rl_u2_replay.py`
  - actual synthetic `UniversalTradeEnvironment` replay and economic/runtime oracles.
- Modify `.github/workflows/universal-trade-rl-u2-contracts.yml`
  - add replay unit/integration tests to the maintained U2 focused gate.
- Do not change `trade_rl/rl/environment.py`, `trade_rl/rl/universal_trade_environment.py`, generic walk-forward evaluation, or generic accounting/economics unless an independently reproduced defect proves it necessary.

---

### Task 1: Replay identity, variant, and canonical-closure gate

**Files:**
- Create: `trade_rl/workflows/universal_trade_rl_u2_replay.py`
- Create: `tests/workflows/test_universal_trade_rl_u2_replay.py`
- Modify: `.github/workflows/universal-trade-rl-u2-contracts.yml`

**Interfaces:**
- Consumes:
  - `UniversalTradeRLUniverseManifest`
  - `UniversalTradeRLU2TimePartition`
  - `UniversalTradeRLU2Contract`
  - `UniversalTradeRLU2DevelopmentScopeClosure`
  - `build_universal_trade_rl_u2_development_scope_closure(...)`
  - `load_universal_trade_rl_u2_development_evaluation_datasets(...)`
- Produces:
  - `UniversalTradeRLU2ReplayVariant`
  - `UniversalTradeRLU2ReplayRequest`
  - `UniversalTradeRLU2DevelopmentReplaySession`
  - `build_universal_trade_rl_u2_development_replay_session(...)`

- [ ] **Step 1: Write the canonical-closure RED tests**

Add tests that construct a valid synthetic U2 manifest/partition/contract/closure using the existing Task 7A fixture pattern, then supply a closure with one canonical scope removed. The source loader is a spy that raises if called.

```python
def test_u2_replay_rejects_incomplete_supplied_closure_before_numeric_loading() -> None:
    fixture = _fixture()
    canonical = _closure(fixture)
    incomplete = replace(canonical, scopes=canonical.scopes[:-1], digest="")
    loader_calls: list[object] = []

    def loader(locator: object):
        loader_calls.append(locator)
        raise AssertionError("numeric loader must not be called")

    with pytest.raises(ValueError, match="canonical|closure|scope"):
        _module().build_universal_trade_rl_u2_development_replay_session(
            manifest=fixture.manifest,
            time_partition=fixture.partition,
            u2_contract=fixture.contract,
            u1_contract=fixture.u1_contract,
            policy_contract=fixture.policy_contract,
            normalizer=fixture.normalizer,
            supplied_scope_closure=incomplete,
            artifact_locators=fixture.locators,
            source_loader=loader,
            environment_factory=fixture.environment_factory,
        )

    assert loader_calls == []
```

Also add request/variant validation:

```python
with pytest.raises(ValueError, match="seed"):
    UniversalTradeRLU2ReplayRequest(
        scope_digest=scope.digest,
        policy_variant=UniversalTradeRLU2ReplayVariant.CASH,
        evaluation_seed=99,
        paired_candidate_checkpoint_digest="a" * 64,
    )
```

- [ ] **Step 2: Run the focused RED**

Run through the maintained U2 focused workflow. Expected result: existing U2 tests pass and the new replay test fails only because `trade_rl.workflows.universal_trade_rl_u2_replay` or its public API is not implemented. Format/import failures do not count as the behavioral RED and must be fixed before proceeding.

- [ ] **Step 3: Implement variant/request validation and pre-I/O canonical closure comparison**

Use exact public shapes:

```python
class UniversalTradeRLU2ReplayVariant(str, Enum):
    CANDIDATE = "candidate"
    CASH = "cash"
    CONSTANT_LONG = "constant_long"
    CONSTANT_SHORT = "constant_short"


@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayRequest:
    scope_digest: str
    policy_variant: UniversalTradeRLU2ReplayVariant
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str
```

The session builder must reconstruct the canonical closure before invoking Task 7B loading:

```python
canonical = build_universal_trade_rl_u2_development_scope_closure(
    manifest=manifest,
    time_partition=time_partition,
    u2_contract=u2_contract,
)
if supplied_scope_closure != canonical:
    raise ValueError("U2 Development replay supplied closure is not canonical")
```

Validate `evaluation_seed in u2_contract.training_seeds` when requests are replayed; no separate evaluation RNG parameter exists.

- [ ] **Step 4: Run Task 1 tests and confirm GREEN**

Expected: closure-drift failures occur before `source_loader`; valid session construction proceeds to the existing Task 7B loader boundary.

- [ ] **Step 5: Commit Task 1**

Commit message:

```text
feat: add U2 deterministic replay identity gate
```

---

### Task 2: Verified dataset ownership and exact scope lookup

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_replay.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_replay.py`

**Interfaces:**
- Consumes Task 7B:
  - `load_universal_trade_rl_u2_development_evaluation_datasets(...) -> dict[str, MarketDataset]`
- Produces session properties:
  - `scope_closure_digest: str`
  - `evaluation_dataset_ids: tuple[tuple[str, str], ...]`
  - `scope(scope_digest: str) -> UniversalTradeRLU2EvaluationScope`

- [ ] **Step 1: Write RED tests for exact scope and dataset closure**

Require unknown scope digest rejection, duplicated scope identity rejection through canonical comparison, and exact scope-dataset identity:

```python
with pytest.raises(ValueError, match="scope"):
    session.scope("f" * 64)

scope = canonical.scopes[0]
assert session.scope(scope.digest) == scope
assert session.evaluation_dataset_ids == tuple(
    (symbol, datasets[symbol].dataset_id) for symbol in expected_symbol_order
)
```

Add a test whose Task 7B loader returns a dataset whose ID differs from `scope.evaluation_dataset_digest`; session construction must fail closed.

- [ ] **Step 2: Run focused RED**

Expected: failures identify missing session scope/dataset ownership behavior only.

- [ ] **Step 3: Implement one-time Task 7B loading and exact scope map**

The session constructor loads Train+Development common-view datasets exactly once after metadata validation. It stores datasets by symbol and builds a unique scope-digest map. Revalidate:

```python
if dataset.dataset_id != scope.evaluation_dataset_digest:
    raise ValueError("U2 replay dataset identity mismatch")
if dataset.symbols != (scope.concrete_symbol,):
    raise ValueError("U2 replay dataset symbol mismatch")
```

Do not copy or mutate the immutable `MarketDataset` arrays.

- [ ] **Step 4: Run Task 2 tests and confirm GREEN**

- [ ] **Step 5: Commit Task 2**

Commit message:

```text
feat: bind U2 replay session to canonical datasets
```

---

### Task 3: Fresh verified U1 environment and exact episode alignment

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_replay.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_replay.py`
- Create: `tests/integrations/test_universal_trade_rl_u2_replay.py`

**Interfaces:**
- Consumes:
  - `Callable[[MarketDataset], UniversalTradeEnvironment]`
  - `require_universal_trade_rl_u1_environment_contract(...)`
- Produces private maintained boundary:
  - `_create_verified_environment(scope) -> UniversalTradeEnvironment`
  - `_reset_scope_environment(environment, scope, evaluation_seed)`

- [ ] **Step 1: Write RED tests for environment reuse and reset indices**

Use `tests.rl.universal_trade_test_support.make_u1_wrapper` and synthetic `MarketDataset` common views.

Assert a valid scope reset returns:

```python
observation, info = env.reset(
    seed=request.evaluation_seed,
    options={"start_idx": scope.evaluation_start_bar_index},
)
assert info["start_index"] == scope.evaluation_start_bar_index
assert info["end_index"] == scope.outcome_stop_bar_index_exclusive - 1
```

Create an environment factory that returns the same mutable U1 wrapper twice for two variants. The second replay must raise before stepping it.

- [ ] **Step 2: Run focused RED**

Expected: replay lacks verified fresh-environment/reset handling.

- [ ] **Step 3: Implement U1 contract checks and weak identity isolation**

For every new environment:

```python
require_universal_trade_rl_u1_environment_contract(
    contract=u1_contract,
    environment=environment,
)
if environment.contract.digest != policy_contract.digest:
    raise ValueError("U2 replay policy contract mismatch")
if environment.sequence_normalizer is not normalizer:
    # identity equality is also checked; exact frozen object reuse is preferred in this session.
    if environment.sequence_normalizer is None or environment.sequence_normalizer.digest != normalizer.digest:
        raise ValueError("U2 replay normalizer generation mismatch")
if environment.dataset.dataset_id != scope.evaluation_dataset_digest:
    raise ValueError("U2 replay environment dataset identity mismatch")
```

Track issued mutable environments with weak references so reusing the same environment/base environment across variants fails closed without retaining closed environments unnecessarily.

Reset only with:

```python
environment.reset(
    seed=request.evaluation_seed,
    options={"start_idx": scope.evaluation_start_bar_index},
)
```

Do not pass `episode_hours`, `episode_bars`, `initial_book`, or a non-cash mode.

Verify runtime `end_index == scope.outcome_stop_bar_index_exclusive - 1` through the base environment after reset.

- [ ] **Step 4: Run unit + integration tests and confirm GREEN**

- [ ] **Step 5: Commit Task 3**

Commit message:

```text
feat: align U2 Development replay with frozen U1 episodes
```

---

### Task 4: Deterministic policy variants and exact 2880-step replay

**Files:**
- Modify: `trade_rl/workflows/universal_trade_rl_u2_replay.py`
- Modify: `tests/workflows/test_universal_trade_rl_u2_replay.py`
- Modify: `tests/integrations/test_universal_trade_rl_u2_replay.py`

**Interfaces:**
- Produces:
  - `UniversalTradeRLU2ReplayEvidence`
  - `UniversalTradeRLU2DevelopmentReplaySession.replay(request, *, model=None) -> UniversalTradeRLU2ReplayEvidence`

- [ ] **Step 1: Write RED tests for action selection**

Candidate model spy:

```python
class ModelSpy:
    def __init__(self) -> None:
        self.deterministic_flags: list[bool] = []

    def predict(self, observation, deterministic: bool):
        self.deterministic_flags.append(deterministic)
        return np.asarray([0.25], dtype=np.float32), None
```

Require all candidate calls to use `deterministic=True`. Require baselines to use exact normalized scalar actions `0.0`, `+1.0`, `-1.0` and never call a model.

Reject:
- candidate without model;
- non-candidate with model;
- non-finite/wrong-shape/out-of-range candidate action through the maintained strict U1 action parser/environment.

- [ ] **Step 2: Write RED integration test for exact full tile**

With an actual synthetic `UniversalTradeEnvironment`, replay one scope and assert:

```python
assert evidence.observed_decision_count == scope.decision_count == 2880
assert evidence.runtime_start_bar_index == scope.evaluation_start_bar_index
assert evidence.runtime_end_bar_index == scope.outcome_stop_bar_index_exclusive - 1
assert evidence.final_current_bar_index == evidence.runtime_end_bar_index
assert evidence.terminated is False
assert evidence.truncated is True
assert evidence.normal_completion is True
assert evidence.terminal_accounting_mode == "mark_to_market"
assert evidence.terminal_liquidation_cost == 0.0
```

- [ ] **Step 3: Implement policy action resolution and replay loop**

Use exact fixed baseline actions:

```python
_BASELINE_ACTIONS = {
    UniversalTradeRLU2ReplayVariant.CASH: np.asarray([0.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_LONG: np.asarray([1.0], dtype=np.float32),
    UniversalTradeRLU2ReplayVariant.CONSTANT_SHORT: np.asarray([-1.0], dtype=np.float32),
}
```

Loop until `terminated or truncated`; never manually liquidate. Count decisions and retain the final `info`.

Normal completion is true only when all normative conditions match. If an economic termination occurs early, return evidence with `normal_completion=False`; do not convert it to an exception unless the environment contract itself is invalid.

Infrastructure exceptions propagate.

- [ ] **Step 4: Implement immutable evidence identity**

Use a frozen dataclass with at least:

```python
@dataclass(frozen=True, slots=True)
class UniversalTradeRLU2ReplayEvidence:
    scope_closure_digest: str
    scope_digest: str
    universe_manifest_digest: str
    u1_contract_digest: str
    u2_contract_digest: str
    source_dataset_digest: str
    evaluation_dataset_digest: str
    concrete_symbol: str
    symbol_role: str
    cell: str
    source_window: str
    tile_index: int
    policy_variant: str
    evaluation_seed: int
    paired_candidate_checkpoint_digest: str
    runtime_start_bar_index: int
    runtime_end_bar_index: int
    final_current_bar_index: int
    observed_decision_count: int
    normal_completion: bool
    terminated: bool
    truncated: bool
    termination_reason: str | None
    terminal_accounting_mode: str
    initial_capital: float
    final_net_portfolio_value: float
    net_wealth_ratio: float
    net_simple_returns: tuple[float, ...]
    maximum_drawdown: float
    turnover_total: float
    total_execution_cost: float
    funding_pnl: float
    borrow_cost: float
    trade_count: int
    rebalance_count: int
    normalized_action_trace: tuple[float, ...]
    realized_exposure_trace: tuple[float, ...]
    digest: str = ""
```

Include later-required raw execution/risk counters exposed by maintained step/terminal info where they have stable semantics; do not invent counters that are not observable.

Digest every evidence field except `digest` using `content_digest` and validate supplied digest on reconstruction.

- [ ] **Step 5: Reconcile net accounting exactly**

`BookState.returns_history` is a simple-return series. Require:

```python
wealth_from_returns = math.prod(1.0 + value for value in evidence.net_simple_returns)
assert math.isclose(
    wealth_from_returns,
    evidence.net_wealth_ratio,
    rel_tol=0.0,
    abs_tol=1e-10,
)
```

Do not call this a gross series and do not add costs back.

- [ ] **Step 6: Run Task 4 focused/integration tests and confirm GREEN**

- [ ] **Step 7: Commit Task 4**

Commit message:

```text
feat: replay deterministic U2 Development scopes
```

---

### Task 5: Falsification of pairing, termination, and evidence identity

**Files:**
- Modify: `tests/workflows/test_universal_trade_rl_u2_replay.py`
- Modify: `tests/integrations/test_universal_trade_rl_u2_replay.py`
- Modify production only when a test demonstrates a real gap.

**Interfaces:** Existing Task 1-4 public API only.

- [ ] **Step 1: Add pairing/seed falsification tests**

Prove:
- seed outside `(0,1,2)` rejected;
- reset receives exact request seed;
- changing only seed changes evidence digest;
- changing only paired checkpoint digest changes evidence digest;
- all four variants on one pair carry identical scope/dataset/U1/U2/seed/checkpoint identity.

- [ ] **Step 2: Add canonical closure/scope falsification tests**

Prove before numeric access:
- one removed tile -> reject, loader call 0;
- one substituted scope digest -> reject, loader call 0;
- Admission scope or locator -> reject, loader call 0;
- wrong common-view identity -> reject.

- [ ] **Step 3: Add runtime/economic falsification tests**

Prove:
- factory returns wrong U1 risk/execution/runtime generation -> reject before replay;
- factory reuses mutable environment/base environment -> reject;
- malformed candidate action -> reject;
- normal full tile cannot report `terminated=True`;
- terminal liquidation on normal time limit is rejected as contract drift;
- synthetic early economic termination produces explicit non-normal evidence with termination reason rather than fake normal completion.

- [ ] **Step 4: Add evidence tampering tests**

Use `dataclasses.replace(evidence, ...)` with `digest=""` for a legitimate new identity and with an old digest for tamper detection. Mutations to returns, seed, checkpoint, final wealth, dataset identity, or policy variant must either produce a different digest or fail validation.

- [ ] **Step 5: Run falsification tests and fix only reproduced defects**

Do not weaken assertions or skip failing cases. If a test assumption is wrong, justify any test correction from the frozen spec/U1 implementation before changing it.

- [ ] **Step 6: Commit Task 5**

Commit message:

```text
test: falsify U2 deterministic Development replay
```

---

### Task 6: Focused CI, self-review, independent/falsification review, and exact-head verification

**Files:**
- Modify: `.github/workflows/universal-trade-rl-u2-contracts.yml`
- Review all Task 7C-1 changed files.

**Interfaces:** No new production interface.

- [ ] **Step 1: Keep replay tests in the maintained U2 focused gate**

The focused workflow must format/check and run:

```text
tests/workflows/test_universal_trade_rl_u2_replay.py
tests/integrations/test_universal_trade_rl_u2_replay.py
```

alongside existing U2 source/FIT/time-partition/environment/training and generic router regressions.

- [ ] **Step 2: Run targeted verification on exact head**

Require success for:
- U2 focused unit/integration/falsification tests;
- actual U1 replay integration;
- existing SB3 timeout/bootstrap regressions;
- generic routed environment regressions.

- [ ] **Step 3: Run repository static gates**

Require exact-head success for:

```text
Ruff
Ruff format check
Mypy
Import architecture
Dead-code/static checks required by CI
```

- [ ] **Step 4: Perform self-review from the original spec**

Review the complete diff for:
- no generic U1 economic mutation;
- no generic walk-forward evaluator reuse;
- no terminal liquidation path;
- exact `O_start-1 -> O_stop-1` alignment;
- exact 2880-decision normal replay;
- canonical closure rebuilt before numeric loading;
- Admission inaccessible;
- seed/checkpoint pairing identity;
- simple-return/net-wealth reconciliation;
- no gross metric claim;
- no real Development artifact path or fixture accidentally committed;
- no debug/temporary files.

Fix any discovered issue and rerun nearest tests before broader checks.

- [ ] **Step 5: Perform independent/falsification review**

Reconstruct verification from:
1. Task 7C-1 spec;
2. replay seed amendment;
3. U1 frozen runtime contract;
4. actual final diff;
5. actual tests/assertions.

Actively search for:
- a malformed closure that still causes numeric I/O;
- off-by-one paths that pass happy tests;
- baseline/candidate runtime asymmetry;
- seed/checkpoint mismatch not bound into evidence;
- environment object reuse;
- economic termination misclassified as time-limit completion;
- mock-only coverage hiding real U1 incompatibility.

- [ ] **Step 6: Run full exact-head repository Quality Gate**

On the same final commit require, when triggered:
- full tests and combined coverage;
- critical branch coverage ratchets;
- package/uv identity;
- Ubuntu compatibility;
- Windows compatibility;
- training image/non-root runtime probe;
- full training capability audit;
- PostgreSQL/Nautilus workflows where triggered.

Do not report Task 7C-1 complete until the exact final head has the required evidence.

- [ ] **Step 7: Record remaining limitations**

Explicitly retain:
- real Development numeric evaluation not performed;
- real PPO checkpoint evaluation not performed;
- gross accounting still undefined pending a separate amendment;
- Selection/bootstrap not implemented here;
- Admission sealed;
- Production `NO-GO`.

- [ ] **Step 8: Final Task 7C-1 commit/report**

Report what is guaranteed by the synthetic replay verification and what is not guaranteed. Do not claim profitability, Development pass, Admission eligibility, or Production readiness.
