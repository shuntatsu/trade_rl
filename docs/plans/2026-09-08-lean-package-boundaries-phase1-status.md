# Lean Package Boundaries Phase 1 Status

Current stage: TDD RED contract ready.

No production migration has been applied yet.

The architecture tests intentionally require the Phase 1 final ownership:

- no `trade_rl/domain` package;
- standard-library-only `trade_rl/_validation.py`;
- artifact-owned canonical JSON at `trade_rl/artifacts/canonical.py`;
- gate models/resolution under `trade_rl/evaluation/gates/`;
- no forwarding-only `trade_rl/artifacts/codec.py`;
- no production import of `trade_rl.domain`.

Later-phase strategy/Binance/data/simulation/evaluation folder moves are not part of this RED gate yet. They will receive separate test-first contracts in their own focused phases.
