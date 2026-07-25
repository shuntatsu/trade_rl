# Constrained Growth PPO PR A — TDD RED Evidence

The first production change is intentionally absent.

The branch currently contains tests importing `trade_rl.rl.environment_constraints`. CI is expected to fail during test collection with `ModuleNotFoundError` until the minimal action-path and cost contracts are implemented.

This commit exists to make the RED phase auditable before production code is added.
