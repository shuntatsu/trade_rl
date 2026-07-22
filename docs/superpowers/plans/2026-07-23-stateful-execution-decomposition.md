# Stateful Execution Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the 614-line `execute_stateful_orders()` policy monolith with an explicit invocation-local runtime and focused bar-lifecycle, order-transition, and symbol-fill services without changing any public execution behavior or evidence.

**Architecture:** `execute_stateful_orders()` remains the public orchestration function. It creates a `StatefulExecutionRuntime`, submits intents, and calls three focused services in the existing per-bar order; the runtime centralizes invocation-local mutation, event sequencing, metrics, and final result payload construction.

**Tech Stack:** Python 3.12, NumPy, dataclasses, Pytest, Ruff, Mypy, coverage.py, GitHub Actions.

## Global Constraints

- Preserve `MarketExecutor.execute_orders()` and `StatefulExecutionResult` signatures and fields.
- Preserve every `OrderEvent`, `SymbolCapacityEvidence`, `BookState`, and `OrderBookState` value and ordering for identical inputs.
- Preserve conservative OHLC path selection, trigger semantics, admission, rounding, capacity sharing, fee/cost, carry, corporate-action, and insolvency behavior.
- Preserve exception types and messages for invalid bars, indices, shapes, equity, and contract multipliers.
- Keep primary execution evidence and schema identifiers unchanged.
- Do not add direct exchange routing or change production `NO-GO` status.
- Use TDD: the architecture contract must fail before production service modules exist.
- Add a measured non-regressing branch-coverage group; do not lower any existing threshold.

---

## File map

- Create `trade_rl/simulation/stateful_runtime.py`: invocation-local books, order state, ordered events/capacity evidence, accumulators, cancellation helper, intent submission, and final result payload.
- Create `trade_rl/simulation/stateful_bar_lifecycle.py`: split/inactive handling, open revaluation, gap return, carry, mark-to-market, margin, and insolvency flattening.
- Create `trade_rl/simulation/stateful_order_transitions.py`: pre-fill expiry, latency, admission, eligibility, projected-book reservation, and post-attempt remainder expiry.
- Create `trade_rl/simulation/stateful_symbol_fills.py`: path/trigger evaluation, symbol-capacity allocation, fill application, costs, and fill evidence.
- Modify `trade_rl/simulation/stateful_execution.py`: retain public result type and thin orchestration; move phase policy into the new services.
- Create `tests/architecture/test_stateful_execution_decomposition.py`: module, delegation, mutation-ownership, and source-span contracts.
- Create `tests/simulation/test_stateful_runtime.py`: event sequence, intent submission, aggregate, and result-payload tests.
- Create `tests/simulation/test_stateful_bar_lifecycle.py`: split, inactive asset, carry, mark-to-market, and insolvency ordering tests.
- Create `tests/simulation/test_stateful_order_transitions.py`: expiry, latency, admission, eligibility, and remainder-expiry tests.
- Create `tests/simulation/test_stateful_symbol_fills.py`: path, trigger, capacity, no-fill, partial-fill, full-fill, and accounting tests.
- Modify `tests/simulation/test_stateful_execution.py`: add one exact mixed-order multi-bar characterization fixture covering every result field.
- Modify `pyproject.toml`: add a measured critical branch-coverage group for the new services.
- Create `docs/verification/2026-07-23-stateful-execution-decomposition.md`: RED, focused GREEN, full exact-head CI, PostgreSQL, artifact, and safety evidence.

---

### Task 1: Add RED architecture contracts and characterization evidence

**Files:**
- Create: `tests/architecture/test_stateful_execution_decomposition.py`
- Modify: `tests/simulation/test_stateful_execution.py`

**Interfaces:**
- Consumes: current `trade_rl/simulation/stateful_execution.py` and public `MarketExecutor.execute_orders()`.
- Produces: a failing architectural boundary plus a passing exact behavior characterization which all later tasks must preserve.

- [ ] **Step 1: Write the failing architecture contract**

Create a source-level test with these exact requirements:

```python
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "trade_rl" / "simulation" / "stateful_execution.py"
REQUIRED = {
    "stateful_runtime.py": "StatefulExecutionRuntime",
    "stateful_bar_lifecycle.py": "StatefulBarLifecycle",
    "stateful_order_transitions.py": "StatefulOrderTransitionProcessor",
    "stateful_symbol_fills.py": "StatefulSymbolFillProcessor",
}


def _execute_node() -> ast.FunctionDef:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    return next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "execute_stateful_orders"
    )


def test_stateful_execution_phase_modules_exist() -> None:
    directory = SOURCE.parent
    for filename, class_name in REQUIRED.items():
        path = directory / filename
        assert path.is_file(), filename
        assert class_name in path.read_text(encoding="utf-8")


def test_execute_stateful_orders_is_bounded_orchestration() -> None:
    node = _execute_node()
    assert node.end_lineno is not None
    assert node.end_lineno - node.lineno + 1 <= 180
    source = ast.unparse(node)
    for name in REQUIRED.values():
        assert name in source
    for low_level in (
        "OrderAdmissionPolicy",
        "select_bar_path",
        "evaluate_trigger",
        "allocate_symbol_capacity",
        "apply_dividend",
        "apply_cash_interest",
    ):
        assert low_level not in source
```

- [ ] **Step 2: Add a mixed-order characterization test before refactoring**

Extend `tests/simulation/test_stateful_execution.py` with a two-symbol, three-bar fixture containing:

- one immediately eligible market order;
- one delayed limit order;
- one stop-market order;
- constrained shared volume;
- non-zero spread/fee/carry/dividend inputs;
- one partial fill carried to a later bar.

Assert all of the following from the current implementation before changing production code:

```python
assert result.next_index == 3
assert result.bars_advanced == 3
assert tuple(event.sequence for event in result.order_events) == tuple(
    range(len(result.order_events))
)
assert result.book.canonical_payload() == EXPECTED_BOOK_PAYLOAD
assert result.order_book.canonical_payload() == EXPECTED_ORDER_BOOK_PAYLOAD
assert [event.canonical_payload() for event in result.order_events] == EXPECTED_EVENTS
assert [item.canonical_payload() for item in result.capacity_evidence] == EXPECTED_CAPACITY
np.testing.assert_allclose(result.requested_notional_by_symbol, EXPECTED_REQUESTED)
np.testing.assert_allclose(result.filled_notional_by_symbol, EXPECTED_FILLED)
np.testing.assert_allclose(result.participation_by_symbol, EXPECTED_PARTICIPATION)
np.testing.assert_allclose(result.cost_by_symbol, EXPECTED_COST)
assert result.interval_cost == pytest.approx(EXPECTED_INTERVAL_COST)
assert result.interval_funding == pytest.approx(EXPECTED_FUNDING)
assert result.interval_borrow_cost == pytest.approx(EXPECTED_BORROW)
assert result.interval_dividend == pytest.approx(EXPECTED_DIVIDEND)
assert result.interval_cash_interest == pytest.approx(EXPECTED_CASH_INTEREST)
assert result.interval_gross_return == pytest.approx(EXPECTED_GROSS_RETURN)
assert result.interval_net_return == pytest.approx(EXPECTED_NET_RETURN)
assert result.interval_log_return == pytest.approx(EXPECTED_LOG_RETURN)
```

Generate and review the concrete `EXPECTED_*` literals from the pre-refactor implementation once, then commit them as immutable regression evidence. Do not regenerate them after production code changes.

- [ ] **Step 3: Run characterization and RED architecture tests**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_execution.py \
  tests/architecture/test_stateful_execution_decomposition.py
```

Expected: characterization tests PASS; architecture tests FAIL because the four modules do not exist and the function exceeds 180 lines.

- [ ] **Step 4: Commit RED evidence**

```bash
git add tests/architecture/test_stateful_execution_decomposition.py tests/simulation/test_stateful_execution.py
git commit -m "test: require stateful execution phase services"
```

### Task 2: Implement invocation-local runtime and event sequencing

**Files:**
- Create: `trade_rl/simulation/stateful_runtime.py`
- Create: `tests/simulation/test_stateful_runtime.py`

**Interfaces:**
- Consumes: `MarketExecutor`, `BookState`, `OrderBookState`, `OrderIntent`, `PendingOrder`, `OrderEvent`, and `SymbolCapacityEvidence`.
- Produces:

```python
@dataclass(slots=True)
class StatefulExecutionRuntime:
    book: BookState
    order_book: OrderBookState
    events: list[OrderEvent]
    capacities: list[SymbolCapacityEvidence]
    starting_value: float
    starting_rebalance_events: int
    requested_notional: float
    filled_notional: float
    total_cost: float
    total_funding: float
    total_borrow: float
    total_dividend: float
    total_cash_interest: float
    completed_fills: int
    rejected_count: int
    expired_count: int
    fill_count: int
    max_participation: float
    gross_factor: float
    requested_by_symbol: np.ndarray
    filled_by_symbol: np.ndarray
    participation_by_symbol: np.ndarray
    cost_by_symbol: np.ndarray

    @classmethod
    def create(
        cls,
        executor: MarketExecutor,
        book: BookState,
        order_book: OrderBookState,
    ) -> StatefulExecutionRuntime: ...

    def submit_intents(
        self,
        executor: MarketExecutor,
        intents: Sequence[OrderIntent],
    ) -> None: ...

    def cancel_active_orders(
        self,
        executor: MarketExecutor,
        *,
        processing_index: int,
        reason: str,
        symbol_mask: np.ndarray | None = None,
    ) -> None: ...

    def append_event(
        self,
        executor: MarketExecutor,
        *,
        previous: PendingOrder,
        updated: PendingOrder,
        event_type: str,
        processing_index: int,
        **evidence: object,
    ) -> None: ...

    def result_payload(
        self,
        *,
        start_index: int,
        bars: int,
    ) -> dict[str, object]: ...
```

- [ ] **Step 1: Write failing runtime tests**

Cover:

- cloned book ownership and unchanged caller book;
- positive starting-equity and contract-multiplier validation with exact messages;
- caller-order intent submission;
- event sequences `0..n-1` across submitted and cancelled events;
- symbol-mask cancellation;
- requested-notional/per-symbol initialization;
- zero-request fill ratio of `1.0`;
- result payload scalar and array fields;
- terminal reason normalization through `EconomicTerminationReason`.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/simulation/test_stateful_runtime.py
```

Expected: import failure for `trade_rl.simulation.stateful_runtime`.

- [ ] **Step 3: Implement the minimal runtime**

Move `_timestamp_ns`, `_event`, `_cancel_active_orders`, initial metric allocation,
intent submission, and final aggregate calculations from `stateful_execution.py`.
Keep `StatefulExecutionResult` in `stateful_execution.py`; construct it there with
`StatefulExecutionResult(**runtime.result_payload(...))` so its public module and
class identity remain unchanged.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_runtime.py \
  tests/simulation/test_stateful_execution.py
```

Expected: PASS for runtime tests which do not yet require orchestration delegation; existing execution characterization remains PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/stateful_runtime.py tests/simulation/test_stateful_runtime.py
git commit -m "refactor: add stateful execution runtime"
```

### Task 3: Extract processing-bar lifecycle

**Files:**
- Create: `trade_rl/simulation/stateful_bar_lifecycle.py`
- Create: `tests/simulation/test_stateful_bar_lifecycle.py`

**Interfaces:**
- Consumes: `MarketExecutor` and `StatefulExecutionRuntime`.
- Produces:

```python
@dataclass(frozen=True, slots=True)
class StatefulBarContext:
    previous_index: int
    processing_index: int
    period_start_value: float
    open_prices: np.ndarray
    tick_size: np.ndarray
    lot_size: np.ndarray
    minimum_notional: np.ndarray
    gap_return: float

class StatefulBarLifecycle:
    def __init__(self, executor: MarketExecutor) -> None: ...

    def begin_bar(
        self,
        runtime: StatefulExecutionRuntime,
        *,
        previous_index: int,
        processing_index: int,
    ) -> StatefulBarContext: ...

    def finish_bar(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
    ) -> None: ...
```

- [ ] **Step 1: Write failing lifecycle tests**

Cover exact ordering and values for:

- split cancellation before `BookState.apply_split()`;
- inactive-asset cancellation before delisting settlement;
- open revaluation and gap return;
- effective tick/lot/minimum rule arrays for the processing index;
- pre-carry insolvency cancellation and flattening;
- intrabar gross-factor update;
- dividend before cash interest before funding/borrow before mark-to-market;
- post-carry margin update and insolvency flattening;
- accumulated dividend, cash interest, funding, and borrow values.

Use a recording `MarketExecutor` subclass only to record `_charge_carry`,
`_update_margin`, and `_flatten_after_termination` call order; do not replace the
accounting implementation.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/simulation/test_stateful_bar_lifecycle.py
```

Expected: import failure for `trade_rl.simulation.stateful_bar_lifecycle`.

- [ ] **Step 3: Implement lifecycle service**

Move only the corporate-action/open and end-of-bar carry/mark blocks. Use
`runtime.cancel_active_orders()` so event sequencing remains centralized. Return
copied rule arrays only where the current executor already returns independent
arrays; otherwise retain the existing read-only dataset views.

- [ ] **Step 4: Run focused accounting tests**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_bar_lifecycle.py \
  tests/simulation/test_accounting_oracle.py \
  tests/simulation/test_stateful_execution.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/stateful_bar_lifecycle.py tests/simulation/test_stateful_bar_lifecycle.py
git commit -m "refactor: extract stateful bar lifecycle"
```

### Task 4: Extract order admission and non-fill transitions

**Files:**
- Create: `trade_rl/simulation/stateful_order_transitions.py`
- Create: `tests/simulation/test_stateful_order_transitions.py`

**Interfaces:**
- Consumes: `MarketExecutor`, `StatefulExecutionRuntime`, and `StatefulBarContext`.
- Produces:

```python
class StatefulOrderTransitionProcessor:
    def __init__(self, executor: MarketExecutor) -> None: ...

    def admit(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
    ) -> tuple[PendingOrder, ...]: ...

    def expire_attempted_remainders(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
        attempted_order_ids: set[str],
    ) -> None: ...
```

- [ ] **Step 1: Write failing transition tests**

Cover:

- stable active-order sorting by `(eligible_index, order_id)`;
- expiry before latency/admission;
- latency-wait status and event;
- exact admission rejection reason;
- submitted/latency-wait to eligible transition;
- projected-book reservation across multiple accepted orders;
- missing projected contract multipliers exact error;
- IOC remainder expiry;
- partial-fill carry disabled expiry;
- no expiry for unattempted or GTC carry-enabled orders.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/simulation/test_stateful_order_transitions.py
```

Expected: import failure for `trade_rl.simulation.stateful_order_transitions`.

- [ ] **Step 3: Implement transition processor**

Move `OrderAdmissionPolicy` construction and the two transition blocks without
changing dataset array lookups, projected-book cash math, event types, reasons, or
counter increments.

- [ ] **Step 4: Run focused tests**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_order_transitions.py \
  tests/simulation/test_order_admission.py \
  tests/simulation/test_stateful_execution.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/stateful_order_transitions.py tests/simulation/test_stateful_order_transitions.py
git commit -m "refactor: extract stateful order transitions"
```

### Task 5: Extract symbol trigger, capacity, and fill processing

**Files:**
- Create: `trade_rl/simulation/stateful_symbol_fills.py`
- Create: `tests/simulation/test_stateful_symbol_fills.py`

**Interfaces:**
- Consumes: `MarketExecutor`, `StatefulExecutionRuntime`, `StatefulBarContext`, and admitted `PendingOrder` values.
- Produces:

```python
class StatefulSymbolFillProcessor:
    def __init__(self, executor: MarketExecutor) -> None: ...

    def execute(
        self,
        runtime: StatefulExecutionRuntime,
        context: StatefulBarContext,
        accepted_orders: Sequence[PendingOrder],
    ) -> set[str]: ...
```

The returned set contains every order ID for which trigger/fill evaluation was
attempted, including no-fill outcomes, and is passed unchanged to remainder expiry.

- [ ] **Step 1: Write failing fill tests**

Cover:

- conservative path selection using active buy/sell directions;
- newly triggered versus previously triggered stop priority;
- configured trigger-segment volume fraction;
- trigger event before no-fill/fill event;
- untriggered no-fill reason and path evidence;
- execution-price rounding by direction;
- one shared capacity pool per symbol;
- base-volume and quote-notional capacity behavior;
- zero allocation no-fill evidence;
- partial/full fill transitions;
- execution cost, book execution, margin update, and all aggregate counters;
- stable symbol iteration and allocation order;
- attempted-order ID return value.

- [ ] **Step 2: Confirm RED**

```bash
uv run pytest -q tests/simulation/test_stateful_symbol_fills.py
```

Expected: import failure for `trade_rl.simulation.stateful_symbol_fills`.

- [ ] **Step 3: Implement fill processor**

Move `_priority`, `_configured_fraction`, `_execution_cost`, per-symbol path and
request construction, capacity allocation, and fill application. Retrieve the
latest order from `runtime.order_book.active_orders` before applying each
allocation exactly as the current code does.

- [ ] **Step 4: Run focused simulation tests**

```bash
uv run pytest -q \
  tests/simulation/test_stateful_symbol_fills.py \
  tests/simulation/test_liquidity_allocation.py \
  tests/simulation/test_bar_path.py \
  tests/simulation/test_stateful_execution.py
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add trade_rl/simulation/stateful_symbol_fills.py tests/simulation/test_stateful_symbol_fills.py
git commit -m "refactor: extract stateful symbol fills"
```

### Task 6: Replace the monolith with bounded orchestration

**Files:**
- Modify: `trade_rl/simulation/stateful_execution.py`
- Modify: `tests/architecture/test_stateful_execution_decomposition.py`

**Interfaces:**
- Consumes: the runtime and three phase services from Tasks 2–5.
- Produces: unchanged `execute_stateful_orders(...) -> StatefulExecutionResult`.

- [ ] **Step 1: Rewrite only the orchestration function**

The resulting function must follow this shape:

```python
def execute_stateful_orders(
    executor: MarketExecutor,
    book: BookState,
    order_book: OrderBookState,
    intents: Sequence[OrderIntent],
    *,
    start_index: int,
    bars: int,
) -> StatefulExecutionResult:
    dataset = executor.dataset
    if bars <= 0:
        raise ValueError("bars must be positive")
    if start_index < 0 or start_index + bars >= dataset.n_bars:
        raise ValueError("execution interval is outside the dataset")
    if book.quantities.shape != (dataset.n_symbols,):
        raise ValueError("book quantities do not match market symbols")

    runtime = StatefulExecutionRuntime.create(executor, book, order_book)
    runtime.submit_intents(executor, intents)
    lifecycle = StatefulBarLifecycle(executor)
    transitions = StatefulOrderTransitionProcessor(executor)
    fills = StatefulSymbolFillProcessor(executor)

    for offset in range(bars):
        previous_index = start_index + offset
        context = lifecycle.begin_bar(
            runtime,
            previous_index=previous_index,
            processing_index=previous_index + 1,
        )
        accepted = transitions.admit(runtime, context)
        attempted = fills.execute(runtime, context, accepted)
        transitions.expire_attempted_remainders(runtime, context, attempted)
        lifecycle.finish_bar(runtime, context)

    return StatefulExecutionResult(
        **runtime.result_payload(start_index=start_index, bars=bars)
    )
```

Names may change only if the architecture test, design, plan, and all callers are
updated consistently in the same commit. Do not add conditional compatibility
paths around the new services.

- [ ] **Step 2: Remove moved low-level helpers from `stateful_execution.py`**

Delete `_timestamp_ns`, `_event`, `_priority`, `_configured_fraction`,
`_cancel_active_orders`, and `_execution_cost` after all consumers have moved.
Retain only imports required by the public result dataclass and orchestration.

- [ ] **Step 3: Run architecture and full simulation regression tests**

```bash
uv run pytest -q \
  tests/architecture/test_stateful_execution_decomposition.py \
  tests/simulation \
  tests/rl/test_stateful_replay.py \
  tests/serving/test_training_serving_parity.py
```

Expected: PASS, including the exact mixed-order characterization fixture.

- [ ] **Step 4: Run static checks**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
```

Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add \
  trade_rl/simulation/stateful_execution.py \
  tests/architecture/test_stateful_execution_decomposition.py
git commit -m "refactor: orchestrate stateful execution phases"
```

### Task 7: Add coverage ratchet and verification evidence

**Files:**
- Modify: `pyproject.toml`
- Create: `docs/verification/2026-07-23-stateful-execution-decomposition.md`

**Interfaces:**
- Consumes: final coverage JSON, focused RED run, focused GREEN runs, exact-head CI, PostgreSQL workflow, and uploaded artifact digests.
- Produces: a non-regressing service coverage group and auditable merge evidence.

- [ ] **Step 1: Measure service branch coverage**

Run:

```bash
uv run pytest -q tests --cov=trade_rl --cov-branch --cov-report=json
python scripts/check_critical_coverage.py coverage.json
```

Read covered and total branches for:

- `trade_rl/simulation/stateful_runtime.py`;
- `trade_rl/simulation/stateful_bar_lifecycle.py`;
- `trade_rl/simulation/stateful_order_transitions.py`;
- `trade_rl/simulation/stateful_symbol_fills.py`.

- [ ] **Step 2: Add the measured ratchet**

Add:

```toml
[tool.trade_rl.critical_coverage.groups.stateful_execution_services]
minimum = <measured percentage rounded down to one decimal place>
paths = [
    "trade_rl/simulation/stateful_runtime.py",
    "trade_rl/simulation/stateful_bar_lifecycle.py",
    "trade_rl/simulation/stateful_order_transitions.py",
    "trade_rl/simulation/stateful_symbol_fills.py",
]
```

The minimum must equal the observed aggregate rounded down; it must not be guessed,
inflated, or lower than the evidence supports.

- [ ] **Step 3: Run complete local verification**

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy .
uv run lint-imports
uv run pytest -q tests --cov=trade_rl --cov-branch --cov-report=json
python scripts/check_critical_coverage.py coverage.json
uv run trade-rl --help
```

Expected: all PASS and repository coverage remains at or above `80.0%`.

- [ ] **Step 4: Create verification document**

Record:

- original function span and confirmed `AUD-SIM-001` finding;
- architecture RED commit/run/artifact and expected failures;
- focused service GREEN commits and test counts;
- exact behavior-characterization result;
- final function span and module ownership;
- exact final head;
- CI and PostgreSQL run IDs and conclusions;
- full test count, total and branch coverage, service ratchet;
- artifact IDs and SHA-256 digests;
- compatibility review and production `NO-GO` boundary.

- [ ] **Step 5: Run exact-head GitHub verification**

Require successful jobs for:

- Rebuilt Core;
- Ubuntu compatibility;
- Windows compatibility;
- training-image build and packaged non-root probe;
- PostgreSQL Catalog.

If the verification document creates a new commit, rerun all exact-head workflows
on that documentation-inclusive head before merge.

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml docs/verification/2026-07-23-stateful-execution-decomposition.md
git commit -m "docs: verify stateful execution decomposition"
```

## Self-review checklist

- [ ] Every selected-design responsibility is assigned to exactly one task.
- [ ] `StatefulExecutionResult` remains defined in `stateful_execution.py`.
- [ ] Event sequencing has one owner: `StatefulExecutionRuntime.append_event()`.
- [ ] Corporate actions and carry are not mixed into symbol fill processing.
- [ ] Admission and expiry are not mixed into result finalization.
- [ ] No service persists state beyond one invocation.
- [ ] Characterization literals were captured before the production rewrite.
- [ ] No existing branch-coverage threshold is reduced.
- [ ] No temporary workflow, generated patch, or duplicate legacy implementation remains.
