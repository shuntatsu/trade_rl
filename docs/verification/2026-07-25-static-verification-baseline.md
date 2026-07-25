# Static Verification Baseline Restoration

Date: 2026-07-25

## Scope

This change restores the repository static-verification baseline without changing
training or execution behavior.

- Normalize `np.finfo(np.float64).eps` to a Python `float` before using it in
  behavior-cloning quality arithmetic. Current NumPy typing otherwise leaves the
  `max()` result as a comparison-protocol union and repository-wide Mypy rejects
  the subsequent division.
- Apply the repository-pinned Ruff formatter to the four files reported by the
  exact `ruff format --check .` gate.

The four formatting-only files are:

- `tests/rl/test_rollout_memory.py`
- `tests/simulation/test_orders.py`
- `trade_rl/rl/policies.py`
- `trade_rl/simulation/orders.py`

## Root-cause evidence

A clean checkout of the previous `main` reproduced two Mypy operator errors at
`trade_rl/integrations/sb3_training.py:129`. The same checkout reproduced exactly
four Ruff formatting differences, listed above. No additional source file was
reported by the formatter.

## Focused verification

Before publication, an isolated GitHub Actions checkout ran successfully:

- `ruff check .`
- `ruff format --check .`
- repository-wide `mypy`
- `tests/integrations/test_sb3_training.py`
- `tests/rl/test_rollout_memory.py`
- `tests/simulation/test_orders.py`

The standard exact-head CI, compatibility jobs, training-image build, PostgreSQL
catalog, full test suite, and critical branch-coverage ratchets remain the merge
gates.
