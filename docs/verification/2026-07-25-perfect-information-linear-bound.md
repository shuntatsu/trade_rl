# Perfect-Information Linear Bound Verification

Date: 2026-07-25

## Scope

This change adds a research-only perfect-information benchmark for measuring an
optimistic linearized trading bound. It is evaluation evidence, not a deployable
policy and not a behavior-cloning teacher.

The implementation:

- solves the primary linearized economic objective with SciPy HiGHS;
- solves a secondary lexicographic objective to select the minimum-turnover
  witness while preserving the primary optimum within tolerance;
- independently replays target weights, turnover, costs, wealth factors, and
  constraint violations;
- rejects malformed, infeasible, inconsistent, non-finite, or non-optimal solver
  evidence;
- exposes immutable result arrays and a canonical digest;
- leaves the maintained `residual-ppo-15m` training path unchanged.

## Focused verification

The synchronized checkout passed:

- `ruff check .`
- `ruff format --check .`
- repository-wide `mypy`
- all perfect-information bound tests

The focused suite covers deterministic market cases, randomized/property-style
problems, a brute-force comparison, missing SciPy behavior, malformed solver
outputs, signed-zero identity, immutable result contracts, numerical overflow,
and fail-closed replay validation. Both new production modules reached 100%
statement and branch coverage in focused measurement.

## Safety boundary

- Future information is used only to construct an explicitly labeled evaluation
  upper bound.
- The result must not be exported as a causal policy, BC teacher, or production
  authorization.
- No direct exchange routing, profitability claim, or production-readiness status
  changes.
- Production remains `NO-GO`.

## Final merge gates

The exact head must pass the standard CI, Windows and Ubuntu compatibility,
training-image build and packaged runtime probe, complete Pytest suite, critical
branch-coverage ratchets, CLI smoke, import architecture, dead-code checks, and
PostgreSQL catalog workflow before merge.
