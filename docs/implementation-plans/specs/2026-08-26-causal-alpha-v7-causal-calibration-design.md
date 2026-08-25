# Causal Alpha V7 Causal Calibration Design

## Understanding summary

- Build a V7 research path that turns the V6 gross-loss diagnosis into a causal, train-only calibration layer and a fixed paired candidate comparison.
- Optimize independent per-symbol after-cost wealth, not cross-sectional ranking or signal activity.
- Preserve the V4 shared model, V6 execution/accounting simulator, pure net-log reward, and sealed Admission boundary.
- Compare `v6_control`, `symmetric_contrarian`, and `causal_calibrated` on identical symbol/episode scopes and execution inputs.
- Persist bounded attribution and checkpoint evidence so intermediate failures can be diagnosed and resumed under Docker's 8 GB limit.
- Do not exclude losing symbols, encode symbol identity, relax gates, tune on Admission, or introduce 1-minute data while aggregate gross edge is negative.

## Assumptions

- V6 Selection is development and candidate-selection data for V7; Admission remains unopened and may be evaluated only once after V7 Selection passes.
- The current PostgreSQL/Binance multi-timeframe dataset, V4 context manifest, episode contracts, cost model, and signal delays remain authoritative.
- The user's prior blanket approval applies to this in-scope redesign and inline execution.
- A candidate that fails any fixed universal wealth, cost, turnover, execution, or risk gate is not admissible even if its aggregate mean is positive.

## Non-functional requirements

- Peak working memory must remain below the existing 8 GB Docker limit; pooled prediction and calibration work is block bounded to at most 4,096 rows.
- All artifacts are content addressed and bound to source, lockfile, runtime manifest, V4 context, config, and episode identities.
- Checkpoints are append-only and resume only when every bound digest matches.
- Attribution stores fixed aggregate bins and support counts, not raw market rows or Admission values.
- The learning/domain layer remains NumPy-only and independent of workflow, Docker, PostgreSQL, SB3, and symbol-specific lookup tables.

## Architecture

### Causal calibration boundary

For each Selection knowledge cutoff, reserve a fixed calibration tail that ends before the Selection episode and is separated by the maximum 72-hour label horizon. Fit a base V4 model only through the start of that tail, predict the tail, and join predictions to labels whose end indices remain strictly before the Selection cutoff. Fit one shared calibration model across all training symbols without symbol ID.

The calibrator consumes 4-hour expected return, 4-hour direction score, uncertainty, realized volatility, liquidity, basis/positioning stress, and 24-hour/72-hour directional agreement. It produces a calibrated 4-hour expected gross return and a direction reliability score. Every fit records row ranges, purging, feature schema, support by direction, block size, and content digests.

### Fixed candidates

- `v6_control`: the current V6 fast-only path, unchanged.
- `symmetric_contrarian`: negate the control forecast and direction score before the same target compiler; uncertainty, costs, caps, cadence, and thresholds remain identical.
- `causal_calibrated`: feed calibrated expected return and reliability into the same target compiler. It may abstain only through the existing cost/uncertainty objective and consensus rule.

No candidate has candidate-specific cost, risk, liquidity, or reward behavior. The contrarian branch is a falsification control, not an automatic fallback.

### Attribution

Each replay emits fixed aggregates for long, short, and flat exposure; entry, add, reduce, exit, and hold transitions; calibrated confidence quartile; volatility quartile; liquidity quartile; slow-context agreement; gross and net log-return contribution; execution cost; exposure-hours; and support. Quartile boundaries are learned from the causal calibration tail and frozen for the Selection scope. Attribution totals must reconcile to simulator gross/net return within fixed numerical tolerance.

### Gates and stage order

Signal requires nonzero long and short calibration support, finite calibrated outputs, stable feature availability, and causal range proof. Selection retains the V6 universal gates: symbol-balanced gross and net wealth above one, every symbol net wealth at least one, median symbol net wealth at least one, positive-net scope fraction at least one half, bounded turnover, meaningful execution, and zero hard-risk or unexplained execution failures. Reward remains exactly `100 * net_log_return`.

Only a passing Selection opens Admission. Only a passing Admission may package a teacher and start BC/RL. A rejection terminates normally with evidence and no holdout leakage.

## Failure handling and observability

- Append one checkpoint record after each cutoff/candidate replay group with RSS, elapsed time, metric digests, attribution digest, and stage status.
- Reject missing, duplicate, reordered, or digest-mismatched checkpoint records.
- Bound calibrator fit and prediction arrays to 4,096 rows; report cutoff and RSS before and after fit.
- Preserve rejected Selection artifacts and produce a Japanese GPT handoff report with per-symbol wealth and attribution.

## Test strategy

- Unit-test purge boundaries, symbol-ID exclusion, deterministic calibration, long/short support, and bounded block calls.
- Unit-test symmetric candidate construction and identical execution/cost inputs.
- Property-test attribution reconciliation, reward equality, and paired candidate identities.
- Integration-test Signal to Selection fail-closed ordering, checkpoint resume, artifact provenance, and no Admission/BC/RL on rejection.
- Validate with focused pytest, Ruff, Linux Mypy, Import Linter, Docker provenance probes, a single-cutoff memory diagnostic, and then the complete real-data run.

## Decision log

1. Keep V4 as the base forecaster instead of replacing it. This isolates forecast calibration from representation changes and reuses the proven causal fit boundary.
2. Use a purged train-only calibration tail instead of in-sample calibration. In-sample predictions would overstate reliability.
3. Include a symmetric contrarian control instead of silently flipping the strategy. It tests the V6 gross-loss hypothesis under identical economics.
4. Keep fixed universal gates and all symbols. Aggregate profit cannot hide a losing instrument.
5. Keep 1-minute data out of V7. Execution resolution is not the current bottleneck while aggregate gross wealth is below one.
6. Preserve pure net-log equity reward. Diagnostic penalties must not redefine the optimization objective.

## Explicit non-claims

- Calibration is not assumed to be profitable until Selection and Admission prove it.
- A profitable contrarian result does not prove a permanent inverse relationship.
- Signal support, calibration accuracy, or low turnover does not substitute for after-cost wealth growth.
