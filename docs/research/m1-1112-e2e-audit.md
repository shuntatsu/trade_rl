# M1 #1112: result-blind software audit and remaining gaps

This is a **bounded source/software audit**, not a zero-defect certificate, performance study,
research gate, real-venue model, or permission to place live orders. The change is one
consolidated implementation and evidence PR for [Issue #1112](https://github.com/shuntatsu/trade_rl/issues/1112).

## Newly reproduced RED → GREEN defects

| Boundary | Reproduced failure before fix | Required fail-closed/retry behavior |
| --- | --- | --- |
| Hard risk, numeric evidence | 3 finite-input combinations produced infinite L1 turnover; 10 invalid direct risk-result metadata values passed validation | Reject nonfinite/negative turnover and projection metadata or out-of-range scale before exposing target |
| Emergency order safety | 3 integer/float/string masks were implicitly converted to emergency booleans | Accept boolean masks only, matching reduce-only input contract |
| Stateful execution retry | Fault after a random slippage sample consumed executor RNG state even when the invocation aborted | Restore executor RNG on failed invocation; same seed/inputs retry identical fills and fees |

`tests/risk/test_m1_e2e_audit_boundaries.py` provides 17 tests: 16
independent failing boundaries before production changes and one valid
negative control. `tests/simulation/test_m1_end_to_end_audit.py` provides
an independent two-symbol `Decimal` cash/equity/fee oracle with known fill
capacity and order-event conservation, plus two injected failures (first or
second fill, after stochastic fee sampling). The latter verifies unchanged
input BookState/OrderBookState, RNG restoration and byte-compatible event
payload on deterministic retry. This is not a model of actual exchange fills.

## Readback of existing integration protections

| M1 stage | Locally exercised evidence | Audit status |
| --- | --- | --- |
| Dataset timestamps/availability and artifacts | `tests/data` and `tests/artifacts`; existing delayed-source tests | Software tests pass; unobserved venue timing not covered |
| Strategy, risk, pre-trade | `tests/risk` and `test_shared_cash_replay.py`; 17 new risk cases | Risk input boundary now explicitly guarded; no full strategy-family metamorphic proof |
| Order lifecycle, capacity, signed accounting | `tests/simulation`, including MARKET/LIMIT/STOP, carry, margin and exact-quantity tests; independent two-symbol oracle | Synthetic invariants pass; no full independent oracle for every corporate action and insolvency path |
| Publication and source boundaries | `tests/artifacts`, `tests/architecture`, `tests/integrations/market_store` | Real PostgreSQL exercised by separate hosted CI gate, not local URL-disabled cases |
| Crash/retry and stochastic reproducibility | Two new injected failure offsets, plus existing atomic-accounting tests | Failed call RNG restored; no external-callback rollback or shared-executor concurrency guarantee |

## Concrete local verification (pre-publication worktree)

- `pytest -q tests/risk tests/simulation`: **975 passed**.
- `pytest -q tests/data tests/artifacts tests/evaluation/test_shared_cash_replay.py tests/architecture`: **921 passed**.
- `pytest -q -rs tests/integrations/market_store`: **300 passed, 26 skipped**, with all 26 skips reporting missing `TRADE_RL_TEST_POSTGRES_URL` in the local environment. Host PostgreSQL job must independently run with **zero skipped**, not infer DB coverage from this result.
- Ruff check/format (`trade_rl tests tools`), mypy production + listed tooling, Guide content reference validation, docs routing and Markdown links: **passed**.

The 26 local integration skips are **not** a classification of the approximately
189 skips reported by prior full Lean Core jobs. A current-head full-suite
JUnit/`-rs` skip inventory, separate real PostgreSQL results, branch-coverage
comparison, genuine `gemini-4-hydrogen-high` exact-head reviewer receipt,
Full PR CI and postmerge CI must be read back before marking the corresponding
Issue acceptance items complete. AGY provider invocation is not attested merely
by a GitHub model label; direct Fedora CLI authentication was unavailable
when this audit was performed.

## Deliberately not established

No unused/sealed/final research inputs were opened or relabeled; no market-data
results, profit estimates or trading authorization arise from the software
checks. This audit does not establish broad venue microstructure fidelity,
corruption resilience of every artifact format, universal fault atomicity,
thread-safe executor reuse or universal absence of M1 defects. Existing
public software semantics and research evidence must not be rewritten to
manufacture such claims.
