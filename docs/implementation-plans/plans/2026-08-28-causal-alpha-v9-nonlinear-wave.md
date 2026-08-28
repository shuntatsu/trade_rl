# Causal Alpha V9 Implementation Plan

1. Define immutable V9 config, fit, forecast, target, replay, and gate evidence.
2. Test deterministic causal fitting, no symbol identity, head agreement, entry
   confirmation, neutral continuation, opposite exit, and no direct flips.
3. Add DB-backed restart-safe Signal and Selection execution.
4. Run focused tests, Ruff, Mypy, and Docker provenance validation.
5. Run all 216 real-data Selection replays and inspect each episode boundary.
6. Open Admission only on a passed Selection; otherwise report rejection and
   redesign without touching holdout or BC/RL.
