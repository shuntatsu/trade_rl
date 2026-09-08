# Lean Package Boundaries Phase 1 RED Evidence

Expected RED head before production migration: `fc9cf58fd88662c69b2a22f417c507eadccd63af` plus this evidence-only commit.

The architecture tests were written before production changes. They are expected to fail on the old tree because:

- `trade_rl/domain/` still exists;
- `trade_rl/_validation.py` does not exist;
- `trade_rl/artifacts/canonical.py` does not exist;
- `trade_rl/evaluation/gates/` is still a module rather than a package;
- `trade_rl/artifacts/codec.py` still exists;
- production imports still reference `trade_rl.domain`;
- later-phase layout assertions for strategy families and Binance package are intentionally still RED.

The Phase 1 implementation must not claim GREEN for the whole final-layout test module until later phases. During Phase 1, the gate is split: domain/artifact/gate assertions must become GREEN while strategy/Binance assertions remain expected future-phase failures. Before final cleanup, the architecture test suite will be reorganized so each phase has an independently green contract.
