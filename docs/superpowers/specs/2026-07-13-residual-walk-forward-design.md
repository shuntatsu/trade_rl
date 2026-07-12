# Residual Walk-Forward Design

Date: 2026-07-13
Status: approved concept, written specification pending user review
Branch: `feature/residual-walk-forward`

## 1. Problem

The existing `phase_wf()` validates the legacy direct-weight PPO path. It calls the generic `run_walk_forward()` and never selects `action_mode="baseline-residual"`. Therefore files named `walk_forward_cost1x.json` and `walk_forward_cost2x.json` do not evaluate the baseline-anchored residual architecture.

The residual workflow currently performs one train/validation/test split. It can select configurations A/B/C/D and evaluate normal and doubled costs, but it cannot establish that the result is stable across multiple chronological folds.

## 2. Considered approaches

### A. Reuse the generic direct-weight `run_walk_forward()`

This would require injecting residual training into the existing `TrainFn` interface. It is rejected because that interface returns one agent and has no place for the inner A/B/C/D development matrix, frozen alpha artifact, configuration selection, seed fallbacks, or shadow-relative diagnostics.

### B. Select one residual configuration globally, then test it across folds

This is cheaper, but the configuration would be selected using data that overlaps later fold tests or would be frozen from one arbitrary period. It does not answer whether A/B/D selection itself is stable through time.

### C. Dedicated nested residual Walk-Forward

This is the selected approach. Every outer fold contains an inner train/validation split. A/B/C/D selection occurs only on the inner validation period. The selected configuration is then frozen and evaluated on the outer OOS test period under both 1x and 2x costs.

## 3. Scope

The implementation is research-only. It will:

- add a residual-specific nested expanding Walk-Forward runner;
- expose it through `--action-mode baseline-residual --phase wf` and a dedicated script;
- train and select A/B/C/D independently inside every fold;
- report hybrid and shadow activity, relative performance, alpha state, fallback state, and action diagnostics;
- leave the legacy direct Walk-Forward behavior unchanged;
- keep Registry registration fail-closed.

It will not:

- activate or register a model;
- claim production readiness or profitability;
- replace the existing single-split residual research run;
- implement rolling-window training in the first version;
- refit the selected policy on the complete outer training range after selection.

The last point is deliberate. The policy evaluated on the outer test is the exact policy whose checkpoint and configuration were chosen on the inner validation period. This avoids changing the trained object after selection.

## 4. Fold layout

For total bars `N`, use expanding outer folds with the same broad chronology as the existing Walk-Forward runner:

- outer fold edges start at 40% of `N` and end at 100% of `N`;
- outer training range: `[0, outer_train_end)`;
- outer purge: `max(configured_purge, horizon, 24)`;
- outer test range: `[outer_train_end + purge, outer_test_end)`.

Inside each outer training range:

- inner training range: the first 80% of outer training bars;
- inner purge: the same effective purge;
- inner validation range: `[inner_train_end + purge, outer_train_end)`.

A fold is skipped with an explicit reason when any of these are true:

- inner training has fewer than 200 bars;
- inner validation has fewer than 100 scored bars;
- outer test has fewer than 50 scored bars;
- history context cannot be constructed.

Trend history context is prepended to inner validation and outer test with `with_history_context()`. Scoring begins only at the declared evaluation start marker. Context bars are never counted in returns.

## 5. Per-fold data flow

For each valid fold:

1. Create the inner training FeatureSet.
2. Run the leak self-test on the inner training data.
3. Run predictor-aligned residual alpha Walk-Forward IC on inner training only.
4. Fit one `FrozenResidualAlpha` artifact on inner training only.
5. Build the inner validation context window.
6. Evaluate A, the identity base-trend configuration, on inner validation at 1x and 2x costs.
7. Train B, the trend-mix residual ensemble, using inner training and inner validation.
8. Evaluate B at 1x and 2x costs.
9. If the frozen alpha artifact is enabled, evaluate fixed-alpha diagnostic C and train/evaluate combined configuration D at 1x and 2x costs.
10. Select A, B, or D using `select_residual_configuration()`.
11. Freeze the selected configuration before reading the outer test results.
12. Evaluate that exact selected agent on the outer test at 1x and 2x costs.
13. Evaluate diagnostic baselines on the same scored outer test range and execution-cost assumptions.
14. Save fold artifacts under `residual_wf/fold_<k>/`.

The 1x and 2x OOS evaluations must reuse the same trained policy and frozen alpha artifact. Cost multiplier changes must not retrain or reselect the model.

## 6. Configuration semantics

- A: identity residual action `[0, 0]`; hybrid equals `base_trend_v2` before stateful execution differences.
- B: PPO trend mixing with alpha disabled.
- C: fixed positive alpha diagnostic; never release-selected.
- D: PPO trend mixing with alpha enabled.

D is eligible only when C independently beats A under the existing selection contract. If B and D fail, A is selected. A selecting does not mean a zero-weight portfolio; it means baseline-only trend execution.

## 7. New components

### `mars_lite/eval/residual_walk_forward.py`

Owns:

- fold construction and validation;
- one-fold residual train/select/evaluate orchestration;
- report dataclasses or typed payload builders;
- cross-fold summary calculation;
- deterministic JSON serialization.

This module must not register candidates or mutate global CLI state.

### Residual pipeline helpers

Refactor only the minimum reusable functions from `residual_pipeline.py` so that the single-split run and each WF fold share the same implementation for:

- alpha fitting and gate evaluation;
- A/B/C/D development matrix construction;
- residual ensemble training;
- configuration selection;
- selected-agent resolution.

The shared helper must receive explicit FeatureSets and output paths. It must not build the full dataset internally.

### CLI dispatch

- `scripts/run_pipeline.py`: when `action_mode == "baseline-residual"` and `phase == "wf"`, call the residual WF runner; other residual phases keep the existing behavior.
- `scripts/run_baseline_residual.py`: dispatch `phase == "wf"` to the same runner.
- Residual invocation remains research-only and requires `--no-register` in the control-plane script.

## 8. Report contract

Authoritative output:

`residual_walk_forward.json`

Top-level fields:

- `mode`: `baseline_residual_walk_forward_v1`;
- `action_schema`: `baseline_residual_v1`;
- `config`: folds, purge, horizon, decision interval, seeds, ensemble size, run tier, bar count, and split rules;
- `summary`;
- `folds`;
- `skipped_folds`;
- `release_eligible`: always `false` in this implementation;
- `release_blocker`: sealed residual release workflow remains incomplete.

Each fold records:

- exact absolute indices and bar counts for inner train, inner validation, purge zones, history context, and outer test;
- leak self-test result;
- alpha gate and alpha dataset identity;
- 1x and 2x development matrices;
- selected configuration and selection reasons;
- selected policy mode and whether alpha is enabled;
- per-member identity-fallback flags;
- outer OOS `relative_1x` and `relative_2x` results;
- hybrid and shadow return, Sharpe, drawdown, turnover, costs, and trade counts;
- paired excess log return and bootstrap statistics;
- action and alpha-budget distributions;
- proposal, HTF-constrained, and executed gross diagnostics;
- diagnostic baselines for 1x and 2x costs.

Cross-fold summary records:

- number of requested, completed, and skipped folds;
- selection counts for A/B/D;
- alpha-enabled fold count;
- selected ensemble-member fallback count and rate;
- hybrid zero-trade fold count;
- shadow zero-trade fold count;
- median and mean hybrid return at 1x and 2x;
- median and mean shadow return at 1x and 2x;
- median and mean paired excess log return at 1x and 2x;
- fraction of folds where hybrid beats shadow at 1x and 2x;
- fraction of folds surviving doubled costs without negative excess;
- total scored OOS bars.

For diagnosis, a shadow zero-trade fold is treated as a structural warning because A/base trend should normally trade. A hybrid zero-trade fold is not automatically a structural failure when the selected configuration is not A, but the report must make it explicit.

## 9. Artifact layout

```
<output>/
  residual_walk_forward.json
  residual_wf/
    fold_0/
      residual_alpha.json
      B_trend_mix_model.zip | B_trend_mix_ensemble/
      D_combined_model.zip | D_combined_ensemble/   # only when alpha enabled
      fold_report.json
    fold_1/
      ...
```

No fold artifact is copied into the Registry.

## 10. Failure behavior

The runner fails closed when:

- the dataset leak self-test fails for any attempted fold;
- a trained policy produces non-finite actions or metrics;
- 1x and 2x evaluations do not refer to the same selected model identity;
- the selected configuration is absent from the development matrix;
- context or split boundaries overlap illegally;
- JSON contains NaN or infinity.

Individual folds may be skipped only for declared minimum-size constraints. Unexpected exceptions stop the run rather than being converted into skipped folds.

## 11. Tests

Add tests covering:

1. Expanding outer folds are chronological, disjoint in OOS scoring ranges, and separated by purge.
2. Inner validation and outer test never overlap inner training.
3. History context is present but excluded from scored bars.
4. Alpha fitting receives only inner training data.
5. A/B/C/D selection is completed before outer OOS evaluation.
6. 1x and 2x OOS evaluation reuse the same selected agent and artifact.
7. A reports nonzero shadow activity on a deterministic trending fixture.
8. A shadow-zero fold is surfaced in the summary warning count.
9. B/D fallback-member counts are preserved.
10. Report JSON is deterministic and rejects non-finite values.
11. `--action-mode baseline-residual --phase wf` dispatches to the residual runner.
12. Legacy direct `phase_wf()` remains unchanged.
13. Registry registration remains blocked.

Training-heavy tests use stubs for PPO and alpha fitting. One small integration test uses the real residual environment without long PPO training.

## 12. Acceptance criteria

The implementation is accepted when:

- the command below creates `residual_walk_forward.json` rather than the legacy `walk_forward_cost*.json` files as its authoritative result;
- every completed fold contains A/B and, when alpha passes, C/D development diagnostics;
- every completed fold has both 1x and 2x OOS metrics from one frozen selected policy;
- report summaries distinguish zero residual action from zero portfolio trading by reporting both hybrid and shadow trades;
- no outer test metric influences fold configuration selection;
- direct-mode tests and behavior remain compatible;
- Ruff, Ruff format, mypy, the focused residual contracts, and the complete pytest suite pass.

Example research command:

```bash
uv run python scripts/run_pipeline.py \
  --action-mode baseline-residual \
  --phase wf \
  --no-register \
  --source postgres \
  --folds 3 \
  --ensemble 3 \
  --n-seeds 3 \
  --run-tier research \
  --output output/realdata_residual_wf
```

`n-seeds` remains recorded for compatibility, but the residual fold uses the configured ensemble members as its policy seed set rather than training a second outer seed dimension.
