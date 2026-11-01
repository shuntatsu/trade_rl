# Synthetic Ridge outer-refit conformance harness

`trade_rl.evaluation.runs.synthetic_ridge_outer_refit` is a closed software
conformance harness for the scheduled Ridge update and continuous native
portfolio path. It manufactures its own two-symbol fixture and accepts no
Dataset, file path, or external-data loader. Its only fixture variants are the
baseline and a deterministic future-tail mutation. It does not change the
four-family manifest's zero-fit cap or authorize a real-data fit, replay, or
study.

The runner binds a fixed hourly clock, shared USD 1,000 initial capital,
explicit fee/spread execution costs, a 20% drawdown stop, terminal settlement,
and a predeclared schedule. It performs the initial canonical `fit_ridge_forecast`
before replay starts. Later fits run at their declared synthetic readiness
index, are staged, and become active only at a strictly later decision index.
The schedule, timestamps, symbol roster, capital, cost, risk, and objective
clock must match the internally generated fixture contract.

All decisions pass through one persistent policy and one
`run_shared_portfolio_replay` invocation. A vintage update changes only the
forecast strategy used by that policy. It does not recreate the session,
account, risk controls, executor, or pending orders. The policy keeps one
monotone decision index and becomes poisoned after a fit or decision failure;
the runner does not fall back to an older vintage or publish a successful
partial result.

Each vintage separates causal identity from full-run identity. The causal
vintage ID binds the eligible training-prefix digest, fit cutoff, feature and
horizon configuration, alpha, and fit-symbol roster; it does not use the
future fixture tail. The full artifact digest additionally binds the fixture
identity, coordinator source digest, runtime digest, fitted model, declared
synthetic readiness and activation indexes, and a deterministic unit-signal
prediction. Artifacts can be serialized and strictly reconstructed in memory;
this module does not publish durable files. The coordinator source SHA covers
this module only; it is not a dependency-closure attestation. Exact-head
package/source closure remains a separate verification gate. Fit timing is
reported as `fit_elapsed_seconds`, a measured wall elapsed duration from
`perf_counter`; it is not CPU time and does not establish model readiness at
any real wall-clock time.

The regressions check multiple actual canonical fits, causal future-tail
invariance, malformed schedules and artifact identities, fail-closed repeated
use after a fit error, boundary continuity of native cash/quantities/equity/
drawdown and pending orders, an independent literal fee/spread/cash oracle,
and a final settled flat account with no active orders. These are synthetic
software-path checks. They do not estimate profitability, validate an
economic hypothesis, complete the original Issue #810 research plan, or
replace its independent review and real-data gates.
