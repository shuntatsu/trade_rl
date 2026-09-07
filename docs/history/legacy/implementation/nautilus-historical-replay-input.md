# Nautilus Historical Replay Input Bridge Implementation Plan

**Goal:** Bind validated Stage A historical step intervals to the exact canonical source bars consumed by a Nautilus historical replay without changing execution authority.

**Architecture:** Keep `StageAHistoricalIntervalEvidence` framework-neutral. Add a Stage A workflow adapter that composes the existing interval-evidence builder with `project_historical_interval_source_bars`, producing one immutable replay-input object per environment step. The workflow validates dataset identity before projection and preserves action, equity endpoints, and funding-boundary evidence through composition rather than copying those fields into a second model.

**Tech Stack:** Python 3.12, pytest, Ruff, MyPy, Import Linter, GitHub Actions CI and Nautilus Capability.

## Global Constraints

- Maintained instrument remains `BTCUSDT-PERP.BINANCE`.
- Pinned Nautilus runtime remains `nautilus_trader==1.230.0`.
- Production remains `NO-GO`.
- Existing replay-v4 transition boundaries remain the only factual step boundaries.
- Funding evidence must remain assigned exactly once to one interval.
- The new bridge must not instantiate NautilusTrader or promote execution authority.

## Task 1: Define the replay-input contract with TDD

**Files:**
- Create: `tests/workflows/test_stage_a_nautilus_historical_replay.py`
- Create: `trade_rl/workflows/stage_a_nautilus_historical_replay.py`

**Produces:**

```python
@dataclass(frozen=True, slots=True)
class StageANautilusHistoricalReplayInterval:
    evidence: StageAHistoricalIntervalEvidence
    source_bars: tuple[SourceBar, ...]


def build_stage_a_nautilus_historical_replay_intervals(
    artifact: StageAExecutionReplayArtifact,
    market: MarketDataset,
    *,
    funding_evidence: Sequence[FundingBoundaryEvidence] = (),
) -> tuple[StageANautilusHistoricalReplayInterval, ...]:
    ...
```

- [ ] RED: assert the workflow module/builder exists and binds each replay-v4 interval to the exact `(start_index, end_index]` source bars while preserving interval evidence.
- [ ] RED: assert a dataset whose `dataset_id` differs from the replay cell is rejected before projection.
- [ ] GREEN: implement the minimum composition layer using existing interval and historical source-bar projectors.
- [ ] REFACTOR: keep validation and naming explicit; add no new replay engine abstraction yet.

## Verification

Run on the final head:

```bash
uv run pytest -q tests/workflows/test_stage_a_nautilus_historical_replay.py
uv run pytest -q tests/workflows/test_stage_a_historical_interval_evidence.py tests/integrations/test_nautilus_historical_projection.py tests/workflows/test_stage_a_nautilus_historical_replay.py
uv run ruff check trade_rl/workflows/stage_a_nautilus_historical_replay.py tests/workflows/test_stage_a_nautilus_historical_replay.py
uv run mypy trade_rl/workflows/stage_a_nautilus_historical_replay.py
uv run lint-imports
```

Then require the repository `CI` and `Nautilus Capability` workflows to succeed on the same head before treating this slice as complete.
