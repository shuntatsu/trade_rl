# Architecture 15-Loop Review Log

## Baseline

- Fifteen regression contracts were added before production changes.
- Focused RED run: 15 failed, 0 passed.
- The generic private-import scan found three real violations.

## Loop 1: Public training-environment contract

**Check:** Environment identity and validation lived as private helpers inside `trade_rl.rl.training`.

**Fix:** Added `trade_rl.rl.training_environment_contract` with framework-neutral public functions. Kept private aliases in `rl.training` for compatibility.

**Review:** Numerical and serialized behavior is unchanged; only ownership and API visibility moved. Focused verification follows on this commit.
