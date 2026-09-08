# Package boundaries

## Principle

The filesystem should reveal ownership. Package-level maintained APIs may be stable, while private module paths are free to move when ownership is clarified. Do not keep forwarding-only compatibility modules for retired private paths.

The current lean source tree is organized around these owners:

```text
trade_rl/
├── _validation.py
├── artifacts/
├── data/
│   ├── artifacts/
│   ├── build/
│   └── features/
├── integrations/
│   └── binance/
├── risk/
├── strategies/
│   ├── rules/
│   ├── forecasts/
│   └── rl/
├── simulation/
│   ├── orders/
│   ├── stateful/
│   ├── targets/
│   └── diagnostics/
└── evaluation/
    ├── gates/
    ├── comparison/
    ├── robustness/
    └── runs/
```

`evaluation/experiments/` is deliberately **not part of the current cleanup/runtime path**. A future experiment-loop project may introduce it only under a separately reviewed contract.

## Responsibilities

### Foundation

- `_validation.py`: dependency-light validation helpers.
- `artifacts/`: generic deterministic serialization, hashing, verified-file, atomic-write/pointer, and artifact-store primitives.

### Data

- `data/`: market/data contracts, identity, source definitions, and dataset views.
- `data/artifacts/`: dataset codec/publication ownership.
- `data/build/`: build configuration and dataset construction.
- `data/features/`: maintained feature computation families.

Data may use foundation utilities but must not depend upward on strategies, simulation orchestration, or evaluation.

### Integrations

- `integrations/binance/`: Binance-specific types, Vision parsing/planning, cache/evidence, metadata, bounded transport, and dataset-source assembly.

Integration code may depend on data/artifact contracts. It must not depend on strategies or evaluation. Cache/metadata boundaries use protocols where concrete transport ownership would otherwise create unnecessary coupling.

### Strategies

- `strategies/rules/`: trend and mean-reversion rule candidates.
- `strategies/forecasts/`: forecast controller, supervised fit data, Ridge, and LightGBM.
- `strategies/rl/`: teacher-free PPO strategy/training boundary.

Strategies may consume data contracts. They must not depend on evaluation.

### Risk

`risk/` owns hard pre-trade safety, feasibility, portfolio inputs/limits, and emergency handling. It does not own alpha-generation decisions and must not depend on evaluation.

### Simulation

Core economic primitives stay visible at the simulation root: accounting, execution, bar-path, and liquidity.

- `simulation/orders/`: order model, admission, and reconciliation.
- `simulation/stateful/`: stateful runtime, bar lifecycle, order transitions, and symbol fills.
- `simulation/targets/`: target execution and exposure-controller ownership.
- `simulation/diagnostics/`: execution stress, funding evidence, and runtime-performance evidence/IO.

Simulation may depend on lower data/risk contracts but must not depend upward on strategies or evaluation. `MarketExecutor` and `BookState` remain the single economic/accounting authority.

### Evaluation

Root evaluation primitives retain replay, metrics, evidence, and return-series ownership.

- `evaluation/gates/`: gate models and resolution.
- `evaluation/comparison/`: paired comparison, strategy comparison, bootstrap, and seed robustness.
- `evaluation/robustness/`: capacity, closed-trade/fold diagnostics, perfect-information bounds, and walk-forward contracts.
- `evaluation/runs/`: immutable candidate configuration/run orchestration and the maintained candidate CLI.

Evaluation is the top research readout/orchestration layer and may consume lower packages. Lower packages must not depend back on it.

## Dependency direction

The intended direction is approximately:

```text
_validation + artifacts
          |
          v
         data <------ integrations
        /   \
       v     v
strategies  risk
       \     |
        \    v
        simulation
            ^
            |
        evaluation
```

More precisely, evaluation consumes lower layers; the diagram arrow into simulation from evaluation represents orchestration/use, not an allowed simulation-to-evaluation dependency.

## Public and private contracts

A maintained package-level export is a compatibility surface and requires an explicit change when removed. Internal module paths are not preserved merely because a historical consumer once imported them.

When restructuring:

1. classify every current production file as KEEP, MOVE, or DELETE;
2. prove DELETE has no maintained responsibility or that a named replacement owns it;
3. move implementation once rather than leaving a compatibility twin;
4. update package-level facades and maintained callers;
5. add architecture tests for ownership and forbidden dependency directions;
6. verify observable semantics, not just import success.

Active design documents under `docs/specs/` and active execution plans under `docs/plans/` may guide a change, but they are never the runtime authority. After a completed change is reflected here or in another current authority, Git history retains its chronology and the completed spec/plan should leave the working tree.
