# Lean Package Boundaries Design Amendment 1

Date: 2026-09-08 (JST)
Applies to: `docs/specs/2026-09-08-lean-package-boundaries-design.md`
Status: normative clarification discovered during implementation inventory

## Conclusion

The complete pre-refactor file inventory exposed one omitted maintained primitive and confirms two ownership details that must be unambiguous before destructive moves begin.

This amendment does not change the design direction or behavioral contract.

## 1. `evaluation/series.py` is KEEP at evaluation root

`trade_rl/evaluation/series.py` defines the return-series primitives used by evaluation metrics/comparison. It was omitted from the target tree by documentation error.

Normative final location:

```text
trade_rl/evaluation/series.py
```

It is classified:

```text
KEEP trade_rl/evaluation/series.py
```

It must not be deleted or moved merely because the original target-tree diagram omitted it.

## 2. `MarketDatasetView` belongs at `data/view.py`

The source `trade_rl/data/artifacts.py` currently mixes loader compatibility with `MarketDatasetView`.

Normative split:

```text
MarketDatasetView -> trade_rl/data/view.py
artifact loading/publication -> trade_rl/data/artifacts/
old trade_rl/data/artifacts.py -> DELETE after migration
```

No `data/artifacts/model.py` is required by the design.

## 3. `execution_stress.py` belongs under simulation diagnostics

Normative move:

```text
trade_rl/simulation/execution_stress.py
  -> trade_rl/simulation/diagnostics/execution_stress.py
```

`trade_rl.simulation` must continue to export `ExecutionEnvironmentStress` at package level.

## 4. Implementation-plan precedence

`docs/plans/2026-09-08-lean-package-boundaries.md` is the whole-cleanup roadmap.

Focused phase plans are the executable source of truth for their phase because they contain the TDD ordering and exact commands. For Phase 1 the normative plan is:

`docs/plans/2026-09-08-lean-package-boundaries-foundation.md`

Any contradiction between the broad roadmap and a focused phase plan is resolved in favor of this design + amendments + focused phase plan.

## Invariants

All original invariants remain unchanged: no strategy/economic change, no fit/evaluation scope drift, no dataset/candidate identity drift, no compatibility shims for retired private paths, and same-head CI remains required.
