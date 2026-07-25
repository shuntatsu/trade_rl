# Perfect-Information Linear Bound Design

## Purpose

Add a research-only perfect-information benchmark that measures an optimistic upper bound on achievable compounded log growth over a fixed train range. The benchmark must not be used as a behavior-cloning teacher and must not change the maintained `residual-ppo-15m` training candidate.

This first implementation slice deliberately excludes the causal scenario planner, student-state aggregation, IQL/AWR, and PPO integration. Those are separate subsystems and require independent design and validation cycles.

## Mathematical contract

For decision periods `t = 0, ..., T-1` and assets `i = 0, ..., N-1`, the solver chooses signed risky-asset target weights `w[t, i]` after observing the complete future return matrix. The primary linear program maximizes

`sum_t (returns[t] dot w[t] - transaction_cost dot |w[t] - w[t-1]|) - liquidation_cost dot |w[T-1]|`.

The initial previous weight is a declared input and the terminal position is liquidated to cash. A second lexicographic solve minimizes total turnover while constraining the primary economic objective to remain within `lexicographic_objective_tolerance` of the optimum. The reported upper bound is always the primary optimum; the second-stage path is only the deterministic low-turnover witness.

For every period, the feasible set enforces:

- `abs(w[t, i]) <= max_abs_weight[i]`;
- `sum_i abs(w[t, i]) <= max_gross`;
- `abs(sum_i w[t, i]) <= max_net_exposure` when configured;
- the linearized net simple return is at least `minimum_period_net_return > -1`;
- terminal liquidation cost leaves a strictly positive wealth factor.

The result is called a linear bound, not an exact executable optimum. For any fixed feasible path with positive wealth factors, `log(1 + x) <= x`; therefore the optimized linear objective upper-bounds the exact replayed log return under the same simplified weight-and-cost contract. Discrete lot sizes, minimum notional, partial-fill state, margin liquidation, nonlinear impact, funding, and exchange restrictions are explicitly outside this first bound.

## Solver and dependency boundary

Use `scipy.optimize.linprog(method="highs")` behind an optional `oracle` dependency. SciPy is not part of the base runtime dependency set. Importing `trade_rl.evaluation` must continue to work without SciPy; a focused runtime error is raised only when the solver is invoked without the optional dependency.

The solver reports both primary and secondary HiGHS status, messages, iteration counts, the primary optimum, the selected-path objective, and independently reconstructed primal constraint violation. Non-optimal, non-finite, economically inconsistent, or constraint-violating results fail closed.

## Components

`trade_rl/evaluation/perfect_information_bound.py` owns:

- `PERFECT_INFORMATION_BOUND_SCHEMA`;
- `PerfectInformationBoundConfig`;
- `PerfectInformationBoundResult`;
- input validation, canonical identities, independent replay, and public solve orchestration.

`trade_rl/evaluation/_perfect_information_lp.py` owns:

- private LP variable layout and sparse constraint construction;
- lazy SciPy loading;
- primary economic optimization;
- secondary lexicographic turnover minimization;
- raw solver evidence.

This split keeps public economic validation independent of the optimizer implementation.

## Public result

The result contains immutable arrays for target weights, absolute weights, turnover, period gross returns, period transaction costs, period net returns, terminal liquidation cost, the primary linearized upper bound, selected-path linear objective, exact replay log return, replay total return when it is representable as a finite float, primary and secondary solver evidence, maximum primal violation, problem digest, configuration digest, and result digest. If exponentiating the log return would overflow, the optional simple-return field is `None` while the finite log return remains authoritative.

Scalar or per-asset configuration values are normalized to immutable tuples. The default net exposure limit is `None` because `max_gross` already bounds absolute net exposure; an explicit tighter net limit remains supported.

## Failure behavior

Reject malformed dimensions, empty horizons, non-finite values, asset returns at or below `-1`, negative costs, non-positive weight or gross limits, invalid net limits, invalid initial weights, `minimum_period_net_return <= -1`, negative lexicographic tolerance, non-positive feasibility tolerance, and unsupported solver methods.

Reject solver results unless both HiGHS stages report optimality, all returned arrays are finite, independent economic replay agrees with the LP objective, the second-stage path remains within the declared primary tolerance, all replay wealth factors are positive, the replay log return does not exceed the primary bound, and all constraints hold within tolerance.

## Verification

Tests cover:

1. monotonic positive returns choose the maximum long allocation;
2. monotonic negative returns choose the maximum short allocation;
3. flat returns with costs remain in cash;
4. zero-cost flat returns use the lexicographic minimum-turnover path;
5. gross, net, and per-asset limits are respected;
6. large costs suppress otherwise profitable switching;
7. the primary objective upper-bounds exact replay log growth;
8. repeated runs produce identical arrays and digests;
9. malformed and infeasible inputs fail closed;
10. deterministic randomized small problems satisfy all constraints and the upper-bound inequality;
11. a tiny brute-force grid never exceeds the LP primary optimum;
12. missing optional SciPy is reported only at invocation;
13. malformed primary/secondary solver vectors and inconsistent replay evidence fail closed;
14. signed zero is canonicalized for cross-platform digest stability;
15. unrepresentable simple returns remain optional while finite log returns are preserved;
16. non-finite solver objectives and malformed target-weight matrices fail closed;
17. both production modules achieve 100% statement and branch coverage in the focused suite.

Required merge gates are focused Pytest with branch coverage, Ruff check, Ruff format check, MyPy, the complete evaluation suite, and the full project test suite. This environment cannot materialize the complete GitHub checkout, so this session can provide fresh focused local evidence only; full-repository verification remains mandatory before merge.
