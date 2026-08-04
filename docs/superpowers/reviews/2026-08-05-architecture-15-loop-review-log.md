# Architecture 15-Loop Review Log

## Baseline

- Fifteen regression contracts were added before production changes.
- Focused RED run: 15 failed, 0 passed.
- The generic private-import scan found three real violations.

## Loop 1: Public training-environment contract

**Check:** Environment identity and validation lived as private helpers inside `trade_rl.rl.training`.

**Fix:** Added `trade_rl.rl.training_environment_contract` with framework-neutral public functions. Kept private aliases in `rl.training` for compatibility.

**Review:** Numerical and serialized behavior is unchanged; only ownership and API visibility moved. Focused result: 1 passed, 14 failed.

## Loop 2: SB3 adapter uses the public contract

**Check:** `integrations.sb3_training` imported two private names from `rl.training`.

**Fix:** Replaced both imports and call sites with `training_environment_identity()` and `validate_training_environment()` from the public contract.

**Review:** Framework adapter behavior is unchanged; the dependency is now explicit and versionable. Focused verification follows on this commit.
