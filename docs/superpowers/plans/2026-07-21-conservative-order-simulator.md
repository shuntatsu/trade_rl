# Conservative OHLC Order Simulator Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace implicit target filling with a deterministic, auditable OHLCV order lifecycle that uses processing-bar liquidity, carries partial fills, and requires conservative execution evidence for promotion.

**Architecture:** Add focused order-domain, bar-path, liquidity, and reconciliation modules. `MarketExecutor` orchestrates these modules while `BookState` remains the only accounting authority. The legacy `execute_interval()` API becomes an adapter over a new stateful `execute_orders()` API, and the RL environment carries `OrderBookState` between decisions.

**Tech Stack:** Python 3.12, NumPy, dataclasses, pytest, Stable-Baselines3 environment integration, canonical JSON digests, PostgreSQL-backed experiment evidence, Docker and GitHub Actions.

## Global Constraints

- Use only information available at order submission to size requested quantity.
- Use `dataset.volume[processing_index]` for fills processed at that index.
- Preserve `BookState` accounting equations and the independent P0 accounting oracle.
- Keep all fills deterministic and replayable; do not introduce stochastic rejection in this change.
- Final promotion requires conservative path mode, processing-bar capacity, partial-fill carry, complete order evidence, and a matching execution-policy digest.
- Preserve `ExecutionCostConfig.order_type`, `limit_offset_rate`, and `execute_interval()` through a compatibility adapter.
- Do not add exchange connectivity or claim that OHLCV reproduces an order book.

---

### Task 1: Order domain and canonical evidence

**Files:**
- Create: `trade_rl/simulation/orders.py`
- Create: `tests/simulation/test_orders.py`

**Interfaces:**
- Produces: `OrderType`, `TimeInForce`, `OrderStatus`, `OrderIntent`, `PendingOrder`, `OrderBookState`, `OrderEvent`, `OrderDomainError`.
- Produces: `OrderBookState.active_for_symbol(symbol_index: int) -> tuple[PendingOrder, ...]`.
- Produces: `OrderEvent.canonical_payload() -> dict[str, object]` and `execution_policy_digest(payload: Mapping[str, object]) -> str`.

- [ ] **Step 1: Write failing construction and invariant tests**

```python
def test_pending_order_preserves_quantity_identity() -> None:
    intent = OrderIntent.create(
        dataset_id="d" * 64,
        target_identity="target-1",
        execution_policy_digest="e" * 64,
        symbol_index=0,
        requested_quantity=10.0,
        order_type=OrderType.LIMIT,
        time_in_force=TimeInForce.GTC,
        limit_price=99.0,
        stop_price=None,
        submit_index=4,
        eligible_index=5,
        expiry_index=None,
        submission_reference_price=100.0,
        decision_equity=1_000.0,
    )
    pending = PendingOrder.from_intent(intent).apply_fill(
        quantity=4.0,
        notional=396.0,
        processing_index=5,
    )
    assert pending.remaining_quantity == pytest.approx(6.0)
    assert pending.cumulative_filled_quantity == pytest.approx(4.0)
    assert pending.status is OrderStatus.PARTIALLY_FILLED
```

Also test deterministic IDs, type-specific price validation, impossible transitions, duplicate active IDs, canonical event ordering, and terminal-state immutability.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/simulation/test_orders.py -q`

Expected: import failure for `trade_rl.simulation.orders`.

- [ ] **Step 3: Implement immutable domain types and explicit transition methods**

```python
class OrderStatus(StrEnum):
    SUBMITTED = "submitted"
    LATENCY_WAIT = "latency_wait"
    ELIGIBLE = "eligible"
    TRIGGERED = "triggered"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CANCELLED = "cancelled"

@dataclass(frozen=True, slots=True)
class PendingOrder:
    intent: OrderIntent
    remaining_quantity: float
    cumulative_filled_quantity: float = 0.0
    cumulative_filled_notional: float = 0.0
    status: OrderStatus = OrderStatus.SUBMITTED
    trigger_index: int | None = None
    last_processed_index: int | None = None
    terminal_reason: str | None = None
    evidence_version: int = 0

    def apply_fill(self, *, quantity: float, notional: float, processing_index: int) -> PendingOrder:
        ...
```

Use SHA-256 over canonical JSON fields for deterministic order IDs. Validate the signed identity `requested = filled + remaining` within `1e-12`.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/simulation/test_orders.py -q && uv run ruff check trade_rl/simulation/orders.py tests/simulation/test_orders.py && uv run mypy trade_rl/simulation/orders.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/orders.py tests/simulation/test_orders.py
git commit -m "feat: add persistent order domain"
```

---

### Task 2: Deterministic OHLC path and trigger engine

**Files:**
- Create: `trade_rl/simulation/bar_path.py`
- Create: `tests/simulation/test_bar_path.py`

**Interfaces:**
- Consumes: `OrderType`, `PendingOrder`.
- Produces: `PathMode`, `TriggerSegment`, `BarPath`, `TriggerDecision`.
- Produces: `select_bar_path(*, open_price: float, high: float, low: float, close: float, mode: PathMode, active_directions: frozenset[int]) -> BarPath`.
- Produces: `evaluate_trigger(order: PendingOrder, path: BarPath) -> TriggerDecision`.

- [ ] **Step 1: Write failing path and gap tests**

```python
def test_neutral_path_uses_closest_extreme_and_low_first_on_tie() -> None:
    closer_low = select_bar_path(
        open_price=100.0, high=110.0, low=98.0, close=105.0,
        mode=PathMode.NEUTRAL, active_directions=frozenset({1}),
    )
    assert closer_low.points == (100.0, 98.0, 110.0, 105.0)


def test_buy_limit_gap_executes_at_open_below_limit() -> None:
    decision = evaluate_trigger(buy_limit(limit=99.0), path(98.0, 102.0, 97.0, 101.0))
    assert decision.executable
    assert decision.execution_price == pytest.approx(98.0)
    assert decision.segment is TriggerSegment.OPEN
```

Cover optimistic/conservative paths, mixed-direction neutral fallback, sell limits, buy/sell stop gaps, triggered-stop persistence, and close-only no-capacity decisions.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/simulation/test_bar_path.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement one path per symbol/bar**

```python
@dataclass(frozen=True, slots=True)
class BarPath:
    mode: PathMode
    points: tuple[float, float, float, float]
    mixed_direction_fallback: bool = False

@dataclass(frozen=True, slots=True)
class TriggerDecision:
    executable: bool
    triggered: bool
    execution_price: float | None
    segment: TriggerSegment | None
    available_volume_fraction: float
    reason: str | None = None
```

Conservative limits delay favorable touch behind adverse movement. Conservative stops apply the worst reachable price after triggering without exceeding the modeled bar range. Do not choose separate paths per order.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/simulation/test_bar_path.py -q && uv run ruff check trade_rl/simulation/bar_path.py tests/simulation/test_bar_path.py && uv run mypy trade_rl/simulation/bar_path.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/bar_path.py tests/simulation/test_bar_path.py
git commit -m "feat: add deterministic OHLC path engine"
```

---

### Task 3: Shared processing-bar liquidity allocator

**Files:**
- Create: `trade_rl/simulation/liquidity.py`
- Create: `tests/simulation/test_liquidity.py`

**Interfaces:**
- Produces: `LiquidityRequest`, `LiquidityAllocation`, `SymbolCapacityEvidence`, `LiquidityAllocationError`.
- Produces: `allocate_symbol_capacity(requests: Sequence[LiquidityRequest], *, processing_volume: float, price: float, contract_multiplier: float, participation_limit: float, lot_size: float, minimum_notional: float) -> tuple[tuple[LiquidityAllocation, ...], SymbolCapacityEvidence]`.

- [ ] **Step 1: Write failing current-bar and no-overallocation tests**

```python
def test_allocator_uses_processing_bar_volume_and_shared_pool() -> None:
    allocations, evidence = allocate_symbol_capacity(
        requests=(market_request("a", 8.0), limit_request("b", 8.0, fraction=0.5)),
        processing_volume=10.0,
        price=100.0,
        contract_multiplier=1.0,
        participation_limit=0.5,
        lot_size=1.0,
        minimum_notional=0.0,
    )
    assert evidence.initial_capacity_notional == pytest.approx(500.0)
    assert sum(item.filled_notional for item in allocations) <= 500.0
```

Cover fractions 1.0/0.5/0.25/0.0, deterministic priority, lot rounding, minimum notional, zero capacity, and a property-style loop proving no capacity over-allocation.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/simulation/test_liquidity.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement allocation with explicit remaining capacity**

```python
remaining_capacity = processing_volume * price * contract_multiplier * participation_limit
for request in sorted(requests, key=LiquidityRequest.priority_key):
    order_capacity = remaining_capacity * request.available_volume_fraction
    fill_notional = min(abs(request.remaining_quantity) * price * contract_multiplier, order_capacity)
    ...
    remaining_capacity -= exact_rounded_notional
```

The trigger fraction limits what an individual order may access; all orders still draw from one symbol-level pool. Recompute exact notional after quantity rounding.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/simulation/test_liquidity.py -q && uv run ruff check trade_rl/simulation/liquidity.py tests/simulation/test_liquidity.py && uv run mypy trade_rl/simulation/liquidity.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/liquidity.py tests/simulation/test_liquidity.py
git commit -m "feat: allocate shared processing-bar liquidity"
```

---

### Task 4: Target reconciliation, admission, and lifecycle policy

**Files:**
- Create: `trade_rl/simulation/order_reconciliation.py`
- Create: `trade_rl/simulation/order_admission.py`
- Create: `tests/simulation/test_order_reconciliation.py`
- Create: `tests/simulation/test_order_admission.py`

**Interfaces:**
- Produces: `reconcile_target(*, dataset_id: str, target_identity: str, execution_policy_digest: str, target_weights: np.ndarray, book: BookState, order_book: OrderBookState, reference_prices: np.ndarray, decision_equity: float, submit_index: int, latency_bars: int, order_type: OrderType, time_in_force: TimeInForce, expiry_index: int | None, limit_offset_rate: float) -> ReconciliationResult`.
- Produces: `OrderAdmissionPolicy.evaluate(...) -> AdmissionDecision`.

- [ ] **Step 1: Write failing cancel-and-replace and causal sizing tests**

```python
def test_reconciliation_does_not_double_submit_partial_residual() -> None:
    result = reconcile_target(
        target_weights=np.array([0.5]),
        book=book_with_quantity(2.0),
        order_book=book_with_active_residual(3.0),
        reference_prices=np.array([100.0]),
        decision_equity=1_000.0,
        submit_index=4,
        ...,
    )
    assert result.new_intents[0].requested_quantity == pytest.approx(0.0, abs=1e-12)


def test_quantity_is_fixed_from_submission_price_not_eligible_open() -> None:
    result = reconcile_target(reference_prices=np.array([100.0]), decision_equity=1_000.0, target_weights=np.array([0.5]), ...)
    assert result.new_intents[0].requested_quantity == pytest.approx(5.0)
```

Cover reversal, target reduction, replacement linkage, zero residual, borrow denial, inactive/tradability/direction denial, invalid rules, margin denial, and identity mismatch.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/simulation/test_order_reconciliation.py tests/simulation/test_order_admission.py -q`

Expected: module import failures.

- [ ] **Step 3: Implement reconciliation against holdings plus active residuals**

```python
effective_quantity = book.quantities + order_book.active_remaining_quantities(n_symbols)
desired_quantity = dataset.notional_to_quantity(submit_index, target_weights * decision_equity, reference_prices)
residual = desired_quantity - effective_quantity
```

Cancel orders moving away from the new target, then create only the still-required residual. Admission failures return evidence decisions rather than raising for economic non-fill conditions.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/simulation/test_order_reconciliation.py tests/simulation/test_order_admission.py -q && uv run ruff check trade_rl/simulation/order_reconciliation.py trade_rl/simulation/order_admission.py tests/simulation/test_order_reconciliation.py tests/simulation/test_order_admission.py && uv run mypy trade_rl/simulation/order_reconciliation.py trade_rl/simulation/order_admission.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/order_reconciliation.py trade_rl/simulation/order_admission.py tests/simulation/test_order_reconciliation.py tests/simulation/test_order_admission.py
git commit -m "feat: add target reconciliation and order admission"
```

---

### Task 5: Stateful execution engine and compatibility adapter

**Files:**
- Modify: `trade_rl/simulation/execution.py`
- Create: `tests/simulation/test_stateful_execution.py`
- Modify: `tests/simulation/test_execution_slippage_coverage.py`
- Modify: `tests/simulation/test_independent_accounting_oracle.py`

**Interfaces:**
- Adds: `ExecutionCostConfig.path_mode`, `default_time_in_force`, `processing_bar_volume_capacity`, `partial_fill_carry`, and `trigger_volume_fractions`.
- Adds: `MarketExecutor.execute_orders(book: BookState, order_book: OrderBookState, intents: Sequence[OrderIntent], *, start_index: int, bars: int) -> StatefulExecutionResult`.
- Preserves: `MarketExecutor.execute_interval(book, target, *, start_index, bars) -> ExecutionResult`.

- [ ] **Step 1: Write failing lifecycle integration tests**

```python
def test_partial_limit_fill_carries_to_next_processing_bar() -> None:
    first = executor.execute_orders(book, OrderBookState.empty(), [intent], start_index=0, bars=1)
    assert first.order_book.active_orders[0].status is OrderStatus.PARTIALLY_FILLED
    second = executor.execute_orders(first.book, first.order_book, [], start_index=1, bars=1)
    assert second.order_book.active_orders == ()
    assert second.completed_fill_count == 1
```

Cover latency wait, IOC remainder expiry, day expiry, GTC persistence, stop trigger persistence, interval-end cancellation, current-bar volume, deterministic replay, shared capacity, split/delisting/margin interaction, and exact accounting-oracle agreement.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/simulation/test_stateful_execution.py tests/simulation/test_independent_accounting_oracle.py -q`

Expected: missing `execute_orders` and new result types.

- [ ] **Step 3: Implement orchestration while delegating accounting to `BookState`**

For each processing index:

```python
processing_index = start_index + offset + 1
result_book.apply_split(dataset.resolved_array("split_factor")[processing_index])
result_book.revalue(dataset.open[processing_index])
path = select_bar_path(...)
triggered = tuple(evaluate_trigger(order, path) for order in eligible_orders)
allocations, capacity = allocate_symbol_capacity(..., processing_volume=dataset.volume[processing_index], ...)
result_book.execute(fill_prices=prices, target_quantities=next_quantities, cost_amount=cost, turnover=turnover)
```

Generate ordered events for every transition and no-fill result. Never use `dataset.volume[processing_index - 1]` for a processing-bar fill.

- [ ] **Step 4: Implement the legacy adapter**

`execute_interval()` reconciles one target, creates IOC market children or interval-expiring limits, invokes `execute_orders()`, cancels active residuals with `interval_end`, and maps the stateful summary into the existing `ExecutionResult` fields.

- [ ] **Step 5: Run simulation tests and static checks**

Run: `uv run pytest tests/simulation -q && uv run ruff check trade_rl/simulation tests/simulation && uv run mypy trade_rl/simulation`

Expected: PASS with legacy tests retained.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/simulation/execution.py tests/simulation
git commit -m "feat: add stateful conservative execution engine"
```

---

### Task 6: RL environment state and observation parity

**Files:**
- Modify: `trade_rl/rl/environment.py`
- Modify: `trade_rl/rl/observations.py`
- Modify: `trade_rl/serving/runtime.py`
- Modify: `tests/serving/test_observation_parity.py`
- Create: `tests/rl/test_pending_order_observation.py`

**Interfaces:**
- Adds `OrderBookState` to environment reset/step state.
- Adds fixed-shape pending-order observation arrays: signed residual quantity ratio, order-type code, status code, age bars, eligible delay, trigger flag, and expiry distance per symbol.
- Extends `PolicyObservationSnapshot` with order-state arrays and `execution_policy_digest`.

- [ ] **Step 1: Write failing causal-state and parity tests**

```python
def test_pending_order_observation_is_causal_and_prevents_duplicate_residual() -> None:
    env.reset(seed=3)
    env.step(np.array([1.0], dtype=np.float32))
    snapshot = env.observation_snapshot()
    assert snapshot.pending_order_remaining[0] > 0.0
    assert snapshot.pending_order_age_bars[0] >= 0
    assert snapshot.execution_policy_digest == env.execution_policy_digest
```

Extend serving parity to compare every pending-order component, raw/normalized observations, member actions, and ensemble action.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/rl/test_pending_order_observation.py tests/serving/test_observation_parity.py -q`

Expected: missing snapshot fields.

- [ ] **Step 3: Carry order state through reset and step**

Initialize `OrderBookState.empty()` at reset. At every action, call reconciliation and `execute_orders()`; do not recreate residual orders already represented in state.

- [ ] **Step 4: Extend canonical observation reconstruction**

Append order arrays in one fixed symbol-major order and include them in the snapshot digest. Serving must reject missing, mismatched, non-finite, or wrong-digest order state.

- [ ] **Step 5: Run RL/Serving focused tests**

Run: `uv run pytest tests/rl tests/serving/test_observation_parity.py tests/serving/test_observation_snapshot_fail_closed.py -q && uv run ruff check trade_rl/rl trade_rl/serving/runtime.py tests/rl tests/serving && uv run mypy trade_rl/rl trade_rl/serving/runtime.py`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add trade_rl/rl/environment.py trade_rl/rl/observations.py trade_rl/serving/runtime.py tests/rl tests/serving/test_observation_parity.py
git commit -m "feat: carry pending orders through policy observations"
```

---

### Task 7: Execution evidence, manifests, and promotion gate

**Files:**
- Create: `trade_rl/evaluation/execution_promotion.py`
- Modify: `trade_rl/workflows/training_run.py`
- Modify: `trade_rl/serving/package.py`
- Modify: `trade_rl/workflows/execution_sensitivity.py`
- Create: `tests/evaluation/test_execution_promotion.py`
- Modify: `tests/serving/test_package.py`
- Modify: `tests/evaluation/test_execution_sensitivity_matrix.py`

**Interfaces:**
- Produces: `ExecutionEvidence`, `ExecutionPromotionDecision`, `validate_execution_promotion(evidence: ExecutionEvidence, *, expected_policy_digest: str) -> ExecutionPromotionDecision`.

- [ ] **Step 1: Write failing promotion tests**

```python
@pytest.mark.parametrize("mode", ["neutral", "optimistic"])
def test_non_conservative_primary_evidence_cannot_promote(mode: str) -> None:
    with pytest.raises(ExecutionPromotionError, match="conservative"):
        validate_execution_promotion(valid_evidence(path_mode=mode), expected_policy_digest="e" * 64)
```

Also reject preceding-bar capacity, disabled partial carry, incomplete events, mismatched policy digest, and optimistic-only sensitivity evidence.

- [ ] **Step 2: Run tests and confirm RED**

Run: `uv run pytest tests/evaluation/test_execution_promotion.py tests/serving/test_package.py -q`

Expected: module import failure.

- [ ] **Step 3: Implement versioned evidence and fail-closed promotion**

```python
@dataclass(frozen=True, slots=True)
class ExecutionEvidence:
    schema_version: str
    dataset_id: str
    execution_policy_digest: str
    path_mode: str
    processing_bar_volume_capacity: bool
    partial_fill_carry: bool
    order_event_count: int
    complete_order_evidence: bool
```

Write `execution-evidence.json` into selected-final artifacts and require it during Serving packaging. Keep neutral/optimistic outputs as sensitivity-only artifacts.

- [ ] **Step 4: Run focused tests and static checks**

Run: `uv run pytest tests/evaluation/test_execution_promotion.py tests/evaluation/test_execution_sensitivity_matrix.py tests/serving/test_package.py -q && uv run ruff check trade_rl/evaluation/execution_promotion.py trade_rl/workflows/training_run.py trade_rl/serving/package.py tests/evaluation/test_execution_promotion.py && uv run mypy trade_rl/evaluation/execution_promotion.py trade_rl/workflows/training_run.py trade_rl/serving/package.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/evaluation/execution_promotion.py trade_rl/workflows/training_run.py trade_rl/serving/package.py trade_rl/workflows/execution_sensitivity.py tests/evaluation tests/serving/test_package.py
git commit -m "feat: require conservative execution evidence for promotion"
```

---

### Task 8: End-to-end migration and deterministic smoke

**Files:**
- Modify: `tests/e2e/test_research_to_serving_v2.py`
- Create: `tests/e2e/test_stateful_order_replay.py`
- Modify: `examples/full_training.py` or the maintained full-training example resolved during implementation
- Modify: `docs/verification/2026-07-21-conservative-order-simulator.md`

**Interfaces:**
- Uses all prior task interfaces.
- Produces one exact-head verification record with workflow IDs, artifact digests, execution-policy digest, and deterministic replay digest.

- [ ] **Step 1: Write failing E2E replay test**

```python
def test_same_dataset_seed_and_actions_replay_identically() -> None:
    first = run_stateful_episode(seed=7, actions=ACTIONS)
    second = run_stateful_episode(seed=7, actions=ACTIONS)
    assert first.order_event_digest == second.order_event_digest
    assert first.equity_curve == pytest.approx(second.equity_curve, abs=1e-12)
    assert first.policy_actions == pytest.approx(second.policy_actions, abs=1e-8)
```

Also prove research-to-serving packaging contains promotable conservative evidence and matching pending-order observation schema.

- [ ] **Step 2: Run E2E tests and confirm RED**

Run: `uv run pytest tests/e2e/test_stateful_order_replay.py tests/e2e/test_research_to_serving_v2.py -q`

Expected: missing E2E helpers/evidence.

- [ ] **Step 3: Update maintained training configuration**

Select conservative path mode, processing-bar capacity, partial-fill carry, GTC stateful orders for the environment, and explicit trigger fractions `(1.0, 0.5, 0.25, 0.0)`. Persist the execution-policy digest in run manifests.

- [ ] **Step 4: Run focused E2E and a small multi-seed smoke**

Run a deterministic 3-seed smoke with a fully unused outer range. Record per-seed results, order-event digests, aggregate fill/reject/expire counts, and whether candidate promotion remains NO-GO or becomes eligible. Do not claim profitability from the smoke.

- [ ] **Step 5: Commit**

```bash
git add tests/e2e examples docs/verification/2026-07-21-conservative-order-simulator.md
git commit -m "test: verify stateful execution end to end"
```

---

### Task 9: Full verification and publication

**Files:**
- Modify: `docs/verification/2026-07-21-conservative-order-simulator.md`
- Modify only if required by failures: `.github/workflows/ci.yml`, `.github/workflows/postgres.yml`

- [ ] **Step 1: Run repository-wide local verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run pytest -q
```

Expected: all pass and total branch coverage remains at or above the repository threshold.

- [ ] **Step 2: Run integration verification**

Run PostgreSQL catalog tests, research-to-serving E2E, Docker training-image build, non-root runtime probe, Studio tests/typecheck/build, Ubuntu compatibility, and Windows compatibility on one exact PR head.

- [ ] **Step 3: Inspect evidence for trust-boundary completeness**

Confirm:

```text
processing-bar volume used: true
partial-fill carry enabled: true
path mode: conservative
order evidence complete: true
execution-policy digest matches experiment plan: true
pending-order Training-Serving parity: true
```

- [ ] **Step 4: Update the verification record**

Record exact head SHA, workflow/run/job IDs, test count, coverage, source-tree digest, lockfile digest, Docker image ID, artifact digests, smoke result, and all known limitations.

- [ ] **Step 5: Commit and prepare a draft PR**

```bash
git add docs/verification/2026-07-21-conservative-order-simulator.md
git commit -m "docs: record conservative execution verification"
git push -u origin agent/conservative-order-simulator-20260721
gh pr create --draft --base main --head agent/conservative-order-simulator-20260721 --title "feat: add conservative stateful order simulation" --body-file <prepared-body>
```

Do not merge until the exact-head CI, PostgreSQL, Serving E2E, Docker, and deterministic smoke evidence are reviewed.
