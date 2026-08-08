# Nautilus Historical Engine Replay Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox syntax for tracking.

**Goal:** Execute factual single-instrument Stage A historical intervals through the pinned NautilusTrader `BacktestEngine`, then bind the resulting position/funding evidence into a deterministic replay result suitable for differential dual-shadow work.

**Architecture:** Keep Stage A workflow identity separate from the Nautilus runtime. A new integration runner consumes a small framework-neutral interval contract containing target exposure, factual decision equity, and canonical source bars. It reuses `TargetExposureController` and `submit_target_exposure_plan`, observes actual Nautilus fills, and snapshots actual signed position at requested funding boundaries. The Stage A workflow adapter converts replay-v4 intervals into that integration contract and performs candidate funding settlement from the actual candidate position while retaining factual equity endpoints as replay evidence.

**Tech Stack:** Python 3.12, `nautilus_trader==1.230.0`, pytest, Ruff, MyPy, Import Linter, GitHub Actions `Nautilus Capability`.

## Global Constraints

- Maintained instrument: `BTCUSDT-PERP.BINANCE` only.
- Nautilus runtime: exactly `1.230.0`.
- OMS: `NETTING`; account type: `MARGIN`.
- Production remains `NO-GO`.
- Stage A actions are single-instrument target exposures in `[-1, 1]`.
- Target activation is causal: the first interval open quote must be observed before reconciliation.
- Sign reversals must reduce to flat first and may open the opposite side only after terminal fill evidence.
- No new target-to-quantity formula: reuse `TargetExposureController`.
- Funding remains integration-boundary settlement; do not claim native Python `BacktestEngine` funding.
- Do not synthesize fill-level equity. Factual Stage A decision/funding equity anchors remain authoritative replay evidence.

---

### Task 1: Generic historical BacktestEngine target runner

**Files:**
- Create: `trade_rl/integrations/nautilus/historical_execution.py`
- Create: `tests/integrations/test_nautilus_historical_execution.py`
- Modify: `.github/workflows/nautilus-capability.yml`

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class NautilusHistoricalTargetInterval:
    sequence: int
    target_exposure: float
    allocated_equity: float
    source_bars: tuple[SourceBar, ...]

@dataclass(frozen=True, slots=True)
class NautilusHistoricalPositionSnapshot:
    timestamp_ns: int
    signed_quantity: Decimal

@dataclass(frozen=True, slots=True)
class NautilusHistoricalExecutionResult:
    runtime_version: str
    fills: tuple[CanonicalFillSignature, ...]
    fee_minor: int
    final_balance_minor: int
    terminal_position_lots: int
    terminal_open_orders: int
    position_snapshots: tuple[NautilusHistoricalPositionSnapshot, ...]


def run_historical_target_intervals(
    intervals: tuple[NautilusHistoricalTargetInterval, ...],
    *,
    snapshot_timestamps_ns: tuple[int, ...] = (),
    starting_balance: Decimal = Decimal("100000"),
    no_trade_band: float = 0.05,
) -> NautilusHistoricalExecutionResult:
    ...
```

- [ ] RED: two flat-price intervals with targets `0.1 -> 0.0` produce one opening and one reduce-only closing fill and terminate flat.
- [ ] RED: a sign flip `0.1 -> -0.1 -> 0.0` produces a reduce-to-flat fill before the opposite-side opening fill.
- [ ] RED: requested snapshot timestamps return the actual candidate signed quantity after all events at that physical boundary.
- [ ] GREEN: feed only projected OHLC quote phases to `BacktestEngine`; activate each target from `on_quote_tick` after the interval's first open quote has entered the cache.
- [ ] GREEN: re-plan a pending sign-flip target only from terminal fill evidence; never cross through flat with one non-reduce order.
- [ ] GREEN: canonicalize actual `OrderFilled` events with the existing trace adapter and derive terminal lots from those fills.
- [ ] Verify in `Nautilus Capability` as its own isolated-process step.

### Task 2: Stage A historical execution bridge and funding settlement

**Files:**
- Modify: `trade_rl/workflows/stage_a_nautilus_historical_replay.py`
- Create: `tests/workflows/test_stage_a_nautilus_historical_execution.py`
- Modify: `.github/workflows/nautilus-capability.yml`

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalExecutionResult:
    execution: NautilusHistoricalExecutionResult
    funding_records: tuple[CanonicalExecutionRecord, ...]


def execute_stage_a_nautilus_historical_replay(
    artifact: StageAExecutionReplayArtifact,
    market: MarketDataset,
    *,
    funding_evidence: Sequence[FundingBoundaryEvidence] = (),
    no_trade_band: float = 0.05,
) -> StageANautilusHistoricalExecutionResult:
    ...
```

- [ ] RED: reject multi-value Stage A actions because this migration slice is formally single-instrument.
- [ ] RED: use `equity_before` as the factual decision-equity anchor for each replayed target; do not infer intermediate equity.
- [ ] RED: candidate funding uses the actual candidate position snapshot and the factual mark/rate/multiplier boundary inputs.
- [ ] RED: fail closed when candidate position lots disagree with the replay's factual funding signed quantity on a due boundary.
- [ ] GREEN: compose the existing interval builder with Task 1; collect boundary snapshots, settle through `CanonicalFundingLedger`, and canonicalize funding records.
- [ ] GREEN: keep funding records separate from fill signatures until a full trace schema explicitly defines fill-level equity.
- [ ] Verify workflow tests plus the exact-wheel isolated Nautilus step.

### Task 3: Differential replay readiness evidence

**Files:**
- Modify: `docs/NAUTILUS_MIGRATION.md`
- Modify: `docs/implementation/nautilus-historical-engine-replay.md`

- [ ] Record the exact capabilities now covered: real historical `BacktestEngine` consumption, safe target reconciliation, candidate position snapshots, and full-replay funding settlement evidence.
- [ ] Leave representative-window differential parity, RL subprocess runtime, 3-step PPO/Lagrangian smoke, performance benchmark, and promotion wiring in `Remaining work` unless separately implemented and verified on the same head.
- [ ] Run final same-head verification: Ruff, format, MyPy, Import Linter, full pytest/coverage, `Nautilus Capability`, `PostgreSQL Catalog`, compatibility jobs, package identity.
