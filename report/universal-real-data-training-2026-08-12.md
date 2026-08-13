# Universal real-data training report — 2026-08-12

## Scope and branch

- Branch: `codex/universal-real-data-training`
- Repository: `https://github.com/shuntatsu/trade_rl`
- Goal: complete the canonical real-data Universal U6 training while fixing failures as they are observed, preserving intermediate economic evidence.
- Status at this checkpoint: training admission investigation is still active. Full 9-member training has **not** been admitted yet.

## Immutable data and reward contract

- Runtime manifest: `artifacts/universal/runtime-manifest.json`
- Manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
- Source: PostgreSQL-backed Binance USD-M tables recorded in the manifest.
- Cache: `binance-usds-m-native-indicators-15x-20241113-20260705-v1`
- Rows: 57,504, from 2024-11-13 through 2026-07-05.
- Timeframes: 15m, 1h, 4h, 1d.
- Train symbols: APT, ARB, BCH, BNB, BTC, LINK, LTC, SOL, XRP (USDT pairs).
- Validation symbols: AVAX, DOGE, ETH (USDT pairs).
- Test symbols: ADA, OP, SUI (USDT pairs).
- Reward remains pure net log growth: `100 * log(net_equity_after / net_equity_before)`.
- Execution costs already enter net equity. No duplicate scalar cost penalty, baseline penalty, or drawdown penalty was added to reward.

## Main observations

### Original 384-decision CUDA economic smoke

The original stochastic policy produced approximately 96,034 final equity versus a 104,607 baseline. PPO and Lagrangian were effectively identical because a 384-decision smoke does not complete the canonical 2,880-decision episode, and Lagrangian remained inside its warm-up/update eligibility conditions.

- PPO/Lagrangian execution cost: about 2,173.
- Gross PnL: about -1,802.
- Net PnL: about -3,966.
- Target-weight delta sum: about 43.98.
- Sign flips: 68.

This showed both poor position selection and excessive stochastic turnover.

### Policy-stage audit

On one fixed BTC segment, deterministic BC was nearly inactive and did not churn:

- BC deterministic net return: about -0.25%.
- Turnover/day: about 0.025x.
- Cost: about 7.1.

The same stage evaluated stochastically churned heavily:

- BC stochastic net return: about -2.99%.
- Turnover/day: about 8.13x.
- Cost: about 2,254.
- Sign flips: 85.

The learned policy log standard deviation stayed near its initial value (`-2.3`, standard deviation about 0.10). Reducing `log_std_init` to `-4.0` reduced the 384-decision PPO smoke cost from about 2,173 to 18.8, but final equity was still only about 98,871 versus the 104,607 baseline. This removed most exploration churn but did not repair position selection.

### Lagrangian mechanics smoke

Generation: `universal-u6-20260812-lagrangian-mechanics-r2`

- Exit code: 0; OOM: false.
- Short mechanics-only episode: 8h; 1,024 timesteps; 8 rollouts.
- All seven dual constraints updated at least once.
- Daily-turnover multiplier increased from 0 to `0.0135416`.
- Controller update counts: `[4, 1, 4, 1, 4, 4, 4]`.

This verifies that Lagrangian mechanics work when complete episodes and warm-up eligibility are deliberately exercised. It does not replace the canonical 720h economic evaluation.

## Failure-driven corrections

| Commit | Observed failure/result | Correction |
| --- | --- | --- |
| `1a6b731a`, `a50fb20d` | SB3 info filtering discarded filled-turnover and fill-count scalars. | Preserve execution scalars, including liquidation telemetry. |
| `dff60d3a`, `8a43898b` | Random/BC/critic/rollout economics were not separately observable. | Add deterministic and stochastic policy-stage evaluation. |
| `eef0bdb9` | Stochastic exploration caused roughly 8x capital/day turnover in the fixed-segment audit. | Change canonical Universal U6 `log_std_init` from `-2.3` to `-4.0`; reward unchanged. |
| `ea4faa34` through `e82f2503` | A 384-step canonical smoke could not activate Lagrangian dual updates. | Add an isolated short-episode mechanics smoke and verifier; disable causal BC holdout only in this mechanics-only job. |
| `379e4a2c` | Universal pretraining accepted reconstruction loss without causal after-cost admission evidence. | Evaluate complete held-out episodes per train symbol, aggregate them, persist holdout/gate artifacts, and fail closed before PPO. |
| `cbc48103` | With one Oracle episode per symbol, the causal validation episode set was empty. | Require multiple complete Oracle episodes whenever Universal causal BC admission is enabled. |
| `520e743d` | Ten episodes per symbol (90 total) exceeded the Docker memory limit while combining observations. | Bound the default to three episodes per symbol. With seed 17, two temporally separated train episodes and one complete holdout remain per symbol. |
| `61bc7fbc` | The explicit complete-episode split (one of three episodes) disagreed with the scalar 10% validation fraction. | Pass the realized explicit episode split fraction into BC without fragmenting episodes or disabling the gate. |
| `c3e6b17a` | Universal BC persisted only a result digest, so best epoch, validation MSE, and early-stopping progression could not be audited after a failed admission. | Persist epoch-level `behavior-cloning-progress.json` and final `behavior-cloning-result.json` before the causal gate. |
| `f00bbecf` | Aggregate loss and terminal equity could not show whether BC reproduced the Oracle's direction changes or collapsed to a narrow action mode. | Persist per-holdout teacher/policy quantiles, histograms, direction rates, sign flips, target deltas, signed error, and correlation. |

## Current BC admission run

Generation: `universal-u6-20260812-cuda-low-std-bc45-r7`

- Source commit: `61bc7fbc747a4ff1eba61cb53acf9c5bd169e237`.
- Configuration: 45 BC epochs, `log_std_init=-4.0`, 384 PPO decisions if admission passes.
- Complete Oracle episodes: 3 per train symbol; 2 train + 1 causal holdout.
- BC train samples: 360 per symbol, 3,240 total, equal-symbol mini-batch sampling.
- Random and BC snapshots were written successfully.
- PPO had not started at this checkpoint; all values below are BC-only.

Completed causal holdouts at this checkpoint showed:

| Symbol | BC gross return | BC net return | Execution cost | Turnover total | Action MAE | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -1.41% | -7.44% | 6,007.7 | 85.77 | 0.816 | 0% |
| ARBUSDT | -0.84% | -7.38% | 6,657.2 | 94.84 | 0.848 | 0% |
| BCHUSDT | -0.63% | -9.10% | 8,501.1 | 123.60 | 0.799 | 0% |
| BNBUSDT | +0.85% | -1.16% | 2,045.7 | 28.57 | 0.814 | 0% |

APTUSDT Oracle net return on the exact held-out episode was +38.96%, while the BC policy returned -7.44%; normalized Oracle regret was about 1.19. The emerging cross-symbol pattern indicates that BC/action reproduction is currently the dominant problem. It is not attributable to PPO because PPO has not begun, and it is not solved by lowering exploration noise.

## Validation evidence

- Lagrangian mechanics container: exit 0, OOM false, all seven constraints updated.
- Universal holdout/split fixes: related test group 7 passed.
- Ruff on changed files: passed.
- Targeted mypy on the changed Universal module reached only the repository's known Windows-only `os.O_DIRECTORY` unreachable diagnostic; no new changed-module typing error was reported.
- Worktree was clean at every container launch, and the image recorded the exact Git commit, source-tree digest, lockfile digest, and runtime-manifest digest.

## Admission decision and next actions

- System/runtime health: PASS.
- Lagrangian mechanism: PASS in its isolated mechanics smoke.
- BC causal economic admission: likely FAIL based on the completed symbols; final aggregate gate pending at this checkpoint.
- PPO algorithm comparison: not yet admissible.
- Full canonical 9-member training: NO-GO until BC causal admission is repaired.

Next actions:

1. Finish all nine causal holdouts and persist the aggregate BC gate.
2. Diagnose why 45-epoch BC has near-zero Oracle action agreement despite balanced training data.
3. Correct the action-head/teacher reproduction path with tests, then rerun the same causal admission.
4. Only after BC passes, run low-exploration PPO/Lagrangian/discounted economic smoke and inspect gross/net PnL, costs, turnover, sign flips, baseline excess, and reward trends.
5. Admit and complete the canonical 3 algorithms × 3 seeds × 524,288 timesteps only after the economic GO conditions pass.

This report is a checkpoint, not a completion claim. It will be updated as the active training goal progresses.

## Checkpoint update: r7 final causal admission

The r7 run completed all nine 720h causal holdouts and then stopped before critic warm-start or PPO, as required by the fail-closed admission contract.

- Container OOM: false.
- Teacher reconstruction relative improvement: 5.06%, gate passed.
- Causal net-return 95% lower confidence bound: -6.93% versus required floor -5.00%, gate failed.
- Cash-baseline after-cost regret: 9.10% versus allowed 20.00%, gate passed.
- Aggregate executed changes: 14,219.
- Aggregate submitted changes: 25,920, meaning every held-out decision proposed a change.
- Aggregate action agreement within tolerance: 0%.
- Aggregate action MAE: 0.956.
- PPO updates performed: zero.

All symbol holdouts:

| Symbol | BC gross | BC net | Oracle net diagnostic | Cost | Turnover | Action MAE | Agreement |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -1.41% | -7.44% | +38.96% | 6,007.7 | 85.77 | 0.816 | 0% |
| ARBUSDT | -0.84% | -7.38% | +35.17% | 6,657.2 | 94.84 | 0.848 | 0% |
| BCHUSDT | -0.63% | -9.10% | +33.17% | 8,501.1 | 123.60 | 0.799 | 0% |
| BNBUSDT | +0.85% | -1.16% | +83.12% | 2,045.7 | 28.57 | 0.814 | 0% |
| BTCUSDT | -1.80% | -2.92% | +1,121.37% | 1,134.6 | 16.21 | 0.956 | 0% |
| LINKUSDT | -0.36% | -8.83% | +64.80% | 8,632.8 | 124.12 | 0.812 | 0% |
| LTCUSDT | +0.87% | -7.00% | +16.24% | 7,892.3 | 112.74 | 0.758 | 0% |
| SOLUSDT | -0.97% | -2.18% | +750.27% | 1,164.9 | 16.43 | 0.885 | 0% |
| XRPUSDT | -0.66% | -1.62% | +216.20% | 963.8 | 13.48 | 0.843 | 0% |

Oracle results are hindsight diagnostics, not deployable baselines. The decisive causal evidence is that the BC policy is negative after costs on every symbol, has zero action agreement, and proposes changes at every decision. The next run will retain exact epoch and early-stopping evidence via commit `c3e6b17a` before any optimization hyperparameter is changed.

## Checkpoint update: BC early stopping and patience isolation

Commit `3f90b9bc` extended the BC progress artifact to retain the full epoch history. Run `universal-u6-20260812-cuda-low-std-bc45-r8` reproduced the original patience-3 condition and established:

- Requested epochs: 45.
- Actual stopping epoch: 5.
- Restored best epoch: 2.
- Initial MSE: 0.84300.
- Final MSE: 0.80670.
- Validation MSE: 0.77635.

The run was intentionally stopped after this evidence was persisted, before repeating all nine unchanged economic holdouts. It was not an OOM event; Docker reports exit 137 because the running evaluation was explicitly stopped after the diagnostic objective was met.

Run `universal-u6-20260812-cuda-low-std-bc45-patience45-r9` changed only BC patience from 3 to 45. The three algorithm configurations retained one common fixed-condition digest: `c4e7ed995ddfe464a9304ee973070a34a1ce7f3e0f49f9356b873173a66bb364`.

The full learning curve showed that patience 3 was too short:

- Epoch 2 validation loss: 0.79758.
- Epoch 18 validation loss: 0.77535.
- Best epoch 29 validation loss: 0.76686.
- Epoch 45 validation loss: 0.91281; the implementation correctly restored epoch 29.
- Final MSE at restored epoch 29: 0.78941, a 6.36% improvement from initialization.

However, the causal economics did not improve enough:

| Metric | r7 patience 3 | r9 patience 45 | Change |
| --- | ---: | ---: | ---: |
| Net-return 95% lower bound | -6.93% | -7.07% | -0.15 pp |
| Worst-symbol net return | -9.10% | -9.34% | -0.24 pp |
| Executed changes | 14,219 | 13,854 | -365 |
| Aggregate action MAE | 0.9558 | 0.9538 | -0.0020 |

The r9 per-symbol net returns were APT -7.30%, ARB -6.88%, BCH -9.33%, BNB -4.04%, BTC -2.57%, LINK -9.34%, LTC -6.06%, SOL -4.05%, and XRP -1.02%. PPO again performed zero updates because the causal gate stopped the run first.

Conclusion: patience 3 did miss a later validation optimum and should not be treated as a true 45-epoch run. Increasing patience improves supervised loss modestly but does not solve the economic admission failure. The remaining pattern—submitted changes at every decision and near-zero held-out Oracle agreement—requires an action-head/HOLD representation comparison rather than further reward shaping or gate relaxation.

## Checkpoint update: r10 direct-head action-distribution diagnosis

Generation: `universal-u6-20260812-cuda-low-std-bc45-patience45-r10`

The r10 run used commit `f00bbecf5411d7874182b290f88a31836893bdd1`, the same PostgreSQL-backed Binance manifest, the same two-train/one-holdout episode split, seed 17, BC patience 45, and `u_medium_direct`. Only diagnostic persistence changed. It completed all nine 720h causal holdouts, stopped at the BC gate with exit 1, reported OOM false, and performed zero PPO updates.

- Teacher reconstruction relative improvement: 4.12%, passed.
- Causal net-return 95% lower bound: -5.033% versus floor -5.000%, failed.
- Worst-symbol net return: BCH -6.486%.
- Mean symbol net return: -3.546%; mean gross return: -0.215%.
- Aggregate executed changes: 12,132; submitted changes: 25,919 of 25,920 decisions.
- Aggregate worst action MAE: 0.9551; agreement within 0.05: 0%.

The new action diagnostics identify a direct-head mode collapse before critic warm-start or PPO:

| Symbol | Gross | Net | Cost | Turnover | Policy mean | Policy std | Policy short rate | Teacher short/flat/long | Correlation | Teacher/Policy sign flips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -0.248% | -5.586% | 5,275.9 | 74.64 | -0.1681 | 0.0265 | 100% | 48.1% / 19.1% / 32.8% | -0.0127 | 872 / 0 |
| ARBUSDT | -0.782% | -6.357% | 5,674.9 | 80.40 | -0.1690 | 0.0291 | 100% | 51.8% / 14.0% / 34.2% | +0.0154 | 997 / 0 |
| BCHUSDT | -0.556% | -6.486% | 5,943.5 | 85.24 | -0.1707 | 0.0222 | 100% | 47.2% / 20.9% / 31.9% | +0.0282 | 749 / 0 |
| BNBUSDT | +0.850% | -0.033% | 896.3 | 12.49 | -0.1907 | 0.0170 | 100% | 44.3% / 20.6% / 35.1% | +0.0282 | 655 / 0 |
| BTCUSDT | -1.101% | -1.280% | 170.6 | 2.43 | -0.1887 | 0.0188 | 100% | 46.5% / 5.7% / 47.7% | +0.0081 | 939 / 0 |
| LINKUSDT | -0.799% | -6.108% | 5,416.2 | 77.06 | -0.1822 | 0.0212 | 100% | 49.9% / 18.4% / 31.7% | +0.0550 | 784 / 0 |
| LTCUSDT | +0.516% | -5.725% | 6,272.2 | 89.12 | -0.1646 | 0.0222 | 100% | 45.6% / 25.8% / 28.6% | +0.0536 | 619 / 0 |
| SOLUSDT | -0.113% | -0.382% | 202.3 | 2.85 | -0.1981 | 0.0139 | 100% | 48.0% / 12.2% / 39.9% | -0.0142 | 909 / 0 |
| XRPUSDT | +0.302% | +0.040% | 238.4 | 3.35 | -0.1939 | 0.0156 | 100% | 49.9% / 15.6% / 34.6% | +0.0103 | 800 / 0 |

The teacher is predominantly ternary (`-1`, `0`, `+1`) and changes direction hundreds of times. The direct BC policy instead emits a narrow negative band on every symbol, with an average symbol mean of -0.181, no sign flips, and teacher correlation ranging only from -0.014 to +0.055. Its small numerical target variations still submit a change on nearly every decision; execution-layer suppression determines whether those proposals become high turnover (APT/ARB/BCH/LINK/LTC) or almost no trading (BTC/SOL/XRP). This is evidence that the direct regression head averages a multimodal teacher target into an almost constant short position. It is not a PPO failure and does not justify adding a duplicate cost penalty to reward.

r10 also exposed GPU nondeterminism in the performance runtime. Although r9 and r10 used the same seed, data, and fixed-condition configuration, r9 restored epoch 29 at validation MSE 0.76686, while r10 restored epoch 2 at 0.77969 and deteriorated to 1.0395 by epoch 45. The architecture artifact records `cudnn_benchmark=true`, `cudnn_deterministic=false`, and `deterministic_algorithms=false`. Any candidate that passes economics must therefore be repeated in deterministic mode before canonical admission.

An existing-head ablation, generation `universal-u6-20260812-cuda-low-std-bc45-patience45-gate-r11`, was launched from the same commit and fixed condition with only `u_medium_direct` replaced by `u_medium_gate`. The hierarchical head explicitly learns change/HOLD and target components. This is a diagnostic comparison, not yet a canonical default change; reward, teacher, data, split, seed, and economic gates remain fixed.

## Checkpoint update: r11 hierarchical-head collapse and r12 calibration

r11 completed all 45 BC epochs and restored epoch 39. Initial MSE was 0.87294 and final MSE was 0.83570, a 4.27% reconstruction improvement. The hierarchical diagnostics nevertheless reported the same invalid decision at every epoch:

- Teacher change/activity rate: 82.28% on the sampled BC set.
- Gate precision: 82.28%; recall: 100%.
- Policy activity ratio: 1.2153.
- `all_trade_collapse=true`; `all_hold_collapse=false`.

The precision exactly equalled teacher prevalence because the gate predicted change for every active sample. The fixed config explained the collapse: `behavior_cloning_max_positive_class_weight=1.4` bounded the majority-positive correction at `1/1.4 = 0.714`, even though the observed negative/positive ratio required about `0.215`. The operational threshold of 0.49 then placed the constant optimum on the trade side.

Two causal holdouts confirmed that this was not only a supervised metric artifact:

| Symbol | Gross | Net | Cost | Turnover | Submitted / decisions | Executed | Policy short rate | Teacher correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -0.87% | -6.46% | 5,526.9 | 78.69 | 2,880 / 2,880 | 2,779 | 100% | -0.0086 |
| ARBUSDT | -1.07% | -7.05% | 6,094.8 | 86.57 | 2,880 / 2,880 | 2,776 | 100% | -0.0359 |

Because the predeclared collapse diagnostic and two independent symbol economics agreed, r11 was intentionally stopped instead of spending another seven full 720h holdouts on the same failed candidate. Docker exit 137 records that explicit stop; OOM was false, and the partial artifacts were preserved under `artifacts/universal/smoke/cuda-low-std-bc45-patience45-gate-r11`.

Generation `universal-u6-20260812-cuda-low-std-bc45-patience45-gate-balanced-r12` was then launched with the same source image, real data, teacher, seed, epochs, patience, reward, and gates. Its ignored diagnostic configs change only the two coupled calibration values:

- `behavior_cloning_max_positive_class_weight`: 1.4 to 20.0, matching the established gate-head profile/default and allowing the observed 0.215 class ratio.
- `behavior_cloning_gate_prediction_threshold`: 0.49 to 0.51, putting an uninformative balanced probability on the HOLD side rather than admitting an all-trade null predictor.

Canonical example configs remain unchanged until the calibrated candidate produces non-collapsed held-out actions and passes after-cost economics.

### r12 user-directed stop and r13 resume

r12 completed the calibrated 45-epoch BC stage before the user-directed pause:

- Restored best epoch: 12.
- Initial/final MSE: 0.87294 / 0.82718, a 5.24% improvement.
- Final gate precision/recall: 0.9042 / 0.6512.
- Final activity ratio: 0.7202.
- Both `all_trade_collapse` and `all_hold_collapse`: false.
- Completed causal holdouts before the pause: APTUSDT only.

The process was explicitly stopped on request. Docker recorded exit 137 and `OOMKilled=false`. Eleven files (22,889,603 bytes), including the restored BC policy and APT holdout, were copied to `artifacts/universal/smoke/cuda-low-std-bc45-patience45-gate-balanced-r12` before restart.

The current Universal workflow supports PPO checkpoint resume but not mid-BC-holdout continuation. After the user requested resume, generation `universal-u6-20260812-cuda-low-std-bc45-patience45-gate-balanced-r13` was therefore launched from the same immutable image and exact r12 calibration. The r12 evidence remains preserved; r13 recomputes the shared teacher/BC stage before completing the causal holdouts.

### r13 independent reproduction and causal state-distribution failure

r13 independently reproduced the balanced supervised gate despite CUDA performance-mode variation:

- Restored best epoch: 11.
- Initial/final MSE: 0.87294 / 0.82474, a 5.52% improvement.
- Epoch-45 gate precision/recall: 0.8883 / 0.8590.
- Epoch-45 activity ratio: 0.9670.
- Both `all_trade_collapse` and `all_hold_collapse`: false.

Autonomous causal evaluation nevertheless failed immediately:

| Symbol | Gross | Net | Cost | Turnover | Submitted | Policy short rate | Teacher correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -0.49% | -7.34% | 6,806.1 | 97.04 | 2,877 / 2,880 | 99.9% | -0.0050 |
| ARBUSDT | +0.75% | -6.43% | 7,186.6 | 102.55 | 2,849 / 2,880 | 99.2% | +0.0119 |
| BCHUSDT | -1.49% | -14.05% | 12,587.6 | 188.56 | 2,764 / 2,880 | 97.8% | -0.0340 |
| BNBUSDT | -0.53% | -12.62% | 12,277.2 | 181.27 | 2,086 / 2,880 | 97.8% | -0.0006 |

Four independent symbols were sufficient to reject the candidate, so r13 was explicitly stopped with OOM false and preserved under `artifacts/universal/smoke/cuda-low-std-bc45-patience45-gate-balanced-r13` (17 files, 22,904,747 bytes).

The split between good teacher-forced validation and failed autonomous rollout identifies state-distribution shift. BC validation observations are collected while the Oracle target path controls `current_weights`; causal evaluation feeds back the policy's own positions. Once the policy deviates, it enters states absent from the teacher trajectory and converges to an almost permanent short target. Class balancing repaired the supervised change/HOLD null solution but did not make the hindsight Oracle causally predictable.

Commit `4e0f1185` upgrades future `behavior-cloning-result.json` artifacts to schema v2 and persists initial/final/validation hierarchical losses and metrics. This closes the audit gap that previously discarded validation-only precision, recall, target RMSE, activity ratio, event recalls, and collapse flags after a run.

## Checkpoint update: causal trend teacher implementation and r14 integration failure

The hindsight Oracle remained economically strong but was not causally predictable from the policy observation. Commit `033c3e06` therefore added an explicit `trend_baseline` Universal teacher using the existing point-in-time `TrendStrategy` target on the same PostgreSQL-backed Binance datasets, episode contracts, multi-timeframe observations, execution model, reward, and economic gates. No reward term or admission threshold was changed.

Generation `universal-u6-20260812-cuda-trend-gate-r14` used image `trade-rl-universal:033c3e060c6f-6726b3737df9`. It assembled the causal teacher successfully, then exited 1 before BC because `trade_rl/integrations/sb3_universal_pretraining.py` retained an Oracle-only guard. Docker reported `OOMKilled=false`. Commit `1994108b` changed that integration guard to accept only the two supported teachers, `oracle` and `trend_baseline`. The targeted integration suite passed 19 tests, and Ruff plus mypy passed before the corrected image was built.

Corrected immutable image:

- Image: `trade-rl-universal:1994108bc8e0-6726b3737df9`.
- Git commit: `1994108bc8e0b37aae9564b2ae3e52667fc56d63`.
- Source-tree digest: `e381aa518eb865fd16cc1630abf675d8be23129d1ac3a9c91acf6fbf3752c865`.
- Lockfile digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`.
- Runtime-manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`.

## Checkpoint update: r15 causal teacher admission

Generation `universal-u6-20260812-cuda-trend-gate-r15` ran the corrected image with three complete episodes per train symbol, two train episodes and one untouched causal holdout, BC seed 17, 45 epochs, and the calibrated hierarchical gate head. It evaluated all nine 720h holdouts and then exited 1 at the pre-PPO economic gate, as intended. Docker reported `OOMKilled=false`; critic warm-start and PPO performed zero updates. Twenty-eight files (22,969,058 bytes) were preserved under `artifacts/universal/smoke/cuda-low-std-bc45-patience45-gate-balanced-trend-r15`.

BC itself learned the causal teacher accurately:

- Restored best epoch: 23.
- Initial/final MSE: `0.218522 / 0.002895`.
- Validation MSE: `0.003765`.
- Validation composed RMSE: `0.06136`.
- Validation active-target RMSE: `0.07835`.
- Validation gate precision/recall/F1: `0.98369 / 0.92967 / 0.95592`.
- Teacher/policy validation activity: `0.64074 / 0.60556`.
- All-hold, all-trade, and constant-action collapse: false.

The autonomous holdout action correlations were also high (`0.8811` to `0.9871`), unlike the Oracle state-distribution failure. The teacher itself was economically invalid, however:

| Symbol | Policy gross | Policy net | Teacher gross | Teacher net | Policy cost | Policy turnover | Correlation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| APTUSDT | -5.024% | -11.113% | -5.482% | -11.631% | 6,067.0 | 91.01 | 0.9824 |
| ARBUSDT | -0.404% | -7.395% | -0.605% | -7.708% | 6,952.4 | 99.32 | 0.9871 |
| BCHUSDT | -3.960% | -10.460% | -5.230% | -11.913% | 6,599.1 | 98.39 | 0.9566 |
| BNBUSDT | -0.475% | -6.647% | -0.690% | -7.627% | 6,128.7 | 89.88 | 0.8831 |
| BTCUSDT | -3.202% | -7.192% | -4.059% | -5.943% | 4,011.2 | 58.63 | 0.9393 |
| LINKUSDT | -7.734% | -16.456% | -6.303% | -15.631% | 8,948.7 | 137.96 | 0.9705 |
| LTCUSDT | -0.744% | -6.883% | -1.197% | -7.634% | 6,160.8 | 89.23 | 0.9649 |
| SOLUSDT | -10.015% | -15.211% | -11.988% | -15.469% | 5,285.5 | 83.43 | 0.9316 |
| XRPUSDT | -4.675% | -10.734% | -4.527% | -9.103% | 6,038.4 | 92.08 | 0.8811 |

Aggregate causal net-return 95% lower bound was `-12.149%` versus the required `-5.000%` floor. The reconstruction gate passed, while only this after-cost economic metric failed. This localizes the failure upstream of RL: the policy successfully reproduced a causal teacher that had negative gross alpha on every held-out symbol and high turnover. Adding another cost penalty to reward would neither repair the teacher's negative gross return nor be economically correct because costs are already present in net-equity reward.

## Trend-teacher root-cause diagnostics

An initial same-holdout counterfactual compared the raw 168h trend target, its sign reversal, and 1h/4h target holding. Sign reversal improved gross return on most symbols and 4h holding reduced some turnover, but this diagnostic had observed the holdout and was not eligible for model selection. It was retained only as a root-cause clue.

A second diagnostic enforced a strict split. It selected direction, holding interval, and exposure from the two BC-train episodes only across 18 candidates (`trend/contrarian` x `4h/1d/1w hold` x `25%/50%/100% exposure`), then evaluated the selected condition once on the untouched third episode. Train-only selection chose ordinary trend, 1-day holding, and 100% exposure:

- Train mean net return: `+1.620%`.
- Train 10th-percentile net return: `-15.940%`.
- Train worst net return: `-18.140%`.

The untouched holdout rejected it on every symbol:

| Symbol | Gross | Net | Cost | Turnover |
| --- | ---: | ---: | ---: | ---: |
| APTUSDT | -3.324% | -9.198% | 5,878.5 | 86.34 |
| ARBUSDT | +0.213% | -7.064% | 7,232.9 | 104.09 |
| BCHUSDT | -4.217% | -10.863% | 6,736.1 | 99.60 |
| BNBUSDT | -0.109% | -6.287% | 6,166.2 | 88.82 |
| BTCUSDT | -3.401% | -3.967% | 586.8 | 8.71 |
| LINKUSDT | -7.489% | -16.030% | 8,724.4 | 134.57 |
| LTCUSDT | -0.668% | -6.893% | 6,223.0 | 89.60 |
| SOLUSDT | -12.576% | -14.795% | 2,175.3 | 34.29 |
| XRPUSDT | -3.318% | -6.581% | 3,182.4 | 46.79 |

Holdout mean gross/net were `-3.877% / -9.075%`; worst net was `-16.030%`; positive-net symbols were `0/9`. The train-only candidate was already unstable, and it did not generalize. Therefore the causal trend teacher is rejected rather than tuned against the holdout.

Current decision:

- Runtime and data integrity: PASS.
- Hierarchical BC mechanics: PASS.
- Causal teacher reproduction: PASS for trend teacher.
- Causal teacher economic validity: FAIL.
- PPO/Lagrangian/discounted comparison: still not admissible.
- Canonical 3 algorithms x 3 seeds x 524,288 timesteps: NO-GO.

Three teacher/head paths have now failed for distinct reasons: direct Oracle BC averaged a multimodal target into a constant short action; hierarchical Oracle BC remained vulnerable to autonomous state-distribution shift; causal trend BC reproduced its teacher but the teacher had negative held-out gross alpha and excessive turnover. This is now a teacher-architecture problem, not a reward-shaping or optimizer-tuning problem. The next candidate must be a genuinely causal, train-only-fitted policy with explicit turnover-aware target construction and untouched validation/test promotion, not another hand-tuned transform of the failed trend holdout.

## Checkpoint update: pooled causal-alpha teacher selection (2026-08-13)

The next teacher route is a pooled, expanding-window causal ridge model. It predicts 24h and 72h forward returns from the existing point-in-time Universal features, then applies a bounded controller grid. The 24h and 72h values are prediction horizons, not minimum holding periods. The reward remains pure net log growth; execution costs are still represented exactly once through net equity.

### Branch and commits

- Branch: `codex/universal-real-data-training`.
- `56acc0e4` (`perf: unblock causal teacher training`): enabled `causal_alpha_ridge` in the SB3 integration, cached expanding fits and predictions, reused selection environments, added progress/telemetry fields, bounded monitor reads, and recorded submitted-command delta/sign-flip metrics.
- `bae7b020` (`fix: persist causal teacher rejection evidence`): made all-candidate rejection preserve complete evidence, added an fsync-backed per-replay checkpoint, enabled restart from completed replay identities, and persisted intermediate gross/net return, turnover, cost, trades, and risk flags.
- Both commits were pushed to `origin/codex/universal-real-data-training` before this checkpoint.

### r2 complete selection and fail-closed result

Container `trade-rl-causal-teacher-smoke-r2` completed all `1,728 / 1,728` production replays across 12 candidates, 9 train symbols, and 16 earlier complete episodes per symbol. The fit/prediction cache reduced the computation to 32 unique pooled fits and 288 unique predictions, with 1,696 fit-cache hits and 1,440 prediction-cache hits. Runtime was approximately 6h04m; Docker reported `OOMKilled=false`.

The run then failed closed with `RuntimeError: no admissible causal alpha candidate`. No teacher admission, BC, critic warm-start, or PPO update ran. Because the pre-r2 implementation wrote selection evidence only after a winner existed, the exact candidate rejection table was lost. Commit `bae7b020` corrects that durability defect without relaxing the gate or changing reward.

### r3 OOM and evidence-backed resume

Container `trade-rl-causal-teacher-smoke-r3` uses immutable image `trade-rl:causal-main-smoke-r3` with:

- image ID: `sha256:f5504be21c9be8af050a856b83147888ea03836f80d54510516faed3f0cc60a9`;
- source revision: `56acc0e474497009fc1d2df2b24c78f0eac35f81`;
- source-tree digest: `387ef7cd8d9cdd8b0d20c5d011042fa8572843028b0dc72ab958e2b7ec6c972b`;
- lockfile digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`;
- runtime-manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`.

The first r3 attempt was globally OOM-killed during its first pooled fit at 13:25:40Z. The Linux OOM record identified the training Python process at approximately 4.28 GiB anonymous RSS. Docker had 7.63 GiB total, while an unrelated CVAT stack consumed roughly another 2.5 GiB. No replay had completed, so no checkpoint row yet existed. After the user stopped CVAT, Docker available memory recovered to approximately 6.8 GiB plus 1.9 GiB free swap. The exact same r3 container was restarted at 13:30:09Z; no duplicate run or container was created.

At this report checkpoint, r3 is running with `OOMKilled=false`, approximately 2.63 GiB RAM, and `26 / 1,728` replay metrics durably recorded. It is processing APTUSDT and has entered the second candidate. The aggregate early evidence is:

| Scope | Replays | Mean gross | Mean net | Lower-tail net | Turnover/day | Execution cost | Trades | Risk violations |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Candidate 1 | 16 | -0.883% | -11.267% | -19.092% | 5.148x | 165,471.4 | 452 | 16/16 |
| Candidate 2 (partial) | 10 | -2.287% | -14.636% | -19.308% | 6.239x | 123,387.8 | 248 | 10/10 |
| Aggregate | 26 | -1.423% | -12.563% | -19.164% | 5.568x | 288,859.2 | 700 | 26/26 |

This sample covers only one train symbol and is not a final selection result. It is nevertheless already unfavorable: every observed episode is negative after costs and has a risk violation. The workflow will continue fail-closed. Only a selected teacher that subsequently passes admission may advance to the real-data CUDA random -> BC -> critic -> RL smoke. PPO reward trajectories do not exist yet for this generation because the teacher gate has not been reached.

## Checkpoint update: cost-aware causal-alpha v2 (2026-08-13)

The historical r3 run confirmed that the v1 controller was not merely paying too much execution cost. At `264 / 1,728` durable replay records, spanning APTUSDT and the beginning of ARBUSDT, its aggregate mean gross return was `-0.707%`, mean net return was `-10.950%`, worst net return was `-19.672%`, and mean turnover was `5.039x/day`. It accumulated `2,695,876.95` of execution cost and `7,481` trades; all `264 / 264` records carried a risk violation. Thus position selection was already negative before costs, while churn amplified the loss. This evidence does not justify changing the scalar reward.

The approved A correction was implemented without adding a minimum holding period and without changing pure net-log-growth reward:

- `51abf8c6` adds the cost-aware causal target controller;
- `29c5c5fb` classifies execution rejections and risk projections instead of treating every projection as unexplained teacher failure;
- `2435cbd9` adds prediction-versus-realized-return signal diagnostics;
- `4e01d4bc` completes the v2 candidate grid, exact execution timing/cost inputs, fail-closed economic selection, resumable v2 checkpoints, cost-aware teacher-batch propagation, and bounded live checkpoint aggregation.

The v2 selection gate rejects hard-risk or unexplained-rejection evidence, zero-trade candidates, negative mean net return, net lower-tail below `-5%`, mean turnover above `1.0x/day`, and candidates with a majority of negative-gross episodes. Admissible candidates rank by lower-tail net return, mean net return, lower turnover, then lower execution cost. Signal Pearson/rank correlation and directional accuracy are diagnostic fields rather than reward terms.

Verification at commit `4e01d4bc`:

- targeted causal teacher, selection, fitting, BC integration, learning evaluation, and monitor suite: `82 passed`;
- monitor-only suite: `6 passed`;
- Ruff: pass for all changed source and tests;
- mypy: pass for all six changed source modules;
- `git diff --check`: pass.

Immutable corrected image `trade-rl:causal-cost-aware-v2-r1` was built from the clean pushed commit:

- image ID: `sha256:a25c89d3df2c02f91b4ee09e09c6a7cbde8b951e8128db8fbbacd2fad42aa748`;
- source revision: `4e01d4bca5e69ae8fb323a26f6da92fc5fd0cd31`;
- source-tree digest: `d93357ed9553017c8a245e8b4e0a90ad8c4b444ceb6de8615d9d4e33bb6e8646`;
- lockfile digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`;
- runtime-manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`.

The corrected v2 generation has intentionally not been started concurrently with r3. At the latest check r3 was `OOMKilled=false`, used approximately `2.88 GiB / 7.63 GiB`, and saturated one CPU. Launching a second pooled-fit process would reintroduce the same global-memory pressure that killed the first r3 attempt. The next safe transition is one v2 container with a distinct generation/checkpoint after r3 releases resources; it must not reuse the v1 checkpoint. CUDA random -> BC -> critic -> RL remains conditional on v2 teacher selection and admission.

### Safe r3 termination and v2 launch

The r3 checkpoint later reached 271 durable records. Code inspection confirmed that v1 rejects a candidate if any episode has `risk_violation=true`; checkpoint inspection confirmed all 12 candidate digests had already accumulated between 16 and 32 such violations. The final no-admissible-candidate result was therefore irreversible regardless of the remaining replays. Continuing r3 could not change the gate outcome and would only delay the corrected route.

The terminal r3 progress and checkpoint were copied to `artifacts/universal/diagnostics/causal-teacher-r3-terminal` before the container was stopped. Docker reported `OOMKilled=false`; the process did not exit within the 30-second stop grace period and was terminated with exit code 137. The named container and its complete shared-volume evidence remain available and restartable.

One corrected generation, `trade-rl-causal-teacher-smoke-v2-r1`, was launched at `2026-08-13T14:37:33Z` from image `trade-rl:causal-cost-aware-v2-r1`. It writes to the distinct generation root `causal-teacher-main-20260813-v2-r1` and the distinct v2 checkpoint `causal-teacher-selection-checkpoint-v2.jsonl`; no v1 checkpoint is reused. Initial Docker inspection showed `state=running`, `OOMKilled=false`, and approximately `1.36 GiB / 7.63 GiB` memory while the first pooled fit was in progress. No second selection container is running.

### v2 initial-state boundary fix and r2 restart

The first v2 generation wrote three complete APTUSDT records and then exited 1, `OOMKilled=false`, before episode 3. The failure was `ValueError: initial_weight must be finite and within max_abs_target`. Runtime probing reproduced the boundary: baseline reset weight was `0.799449` for episode 3 while the candidate's `max_abs_target` was `0.5`. The latter constrains submitted target exposure, not the immutable real portfolio state at reset, so rejecting the episode was a controller-boundary defect.

Commit `3e9e1268` preserves the real initial weight in evidence and forces the first target to deleverage to the configured cap, including the transition in proposed turnover and estimated cost. The fix does not bypass execution accounting, change reward, add a holding lock, or discard baseline reset coverage. A regression test observed the pre-fix exception before implementation. Post-fix verification passed 18 controller tests, 22 directly related workflow tests, the full 83-test causal/BC/monitor surface, Ruff, mypy, and `git diff --check`. The commit and the earlier launch report were pushed to `origin/codex/universal-real-data-training`.

The failed r1 progress, v2 checkpoint, and container log were preserved under `artifacts/universal/diagnostics/causal-teacher-v2-r1-failure`. Its three metrics are not resumed because doing so would mix target-controller behavior across source revisions. Image `trade-rl:causal-cost-aware-v2-r2` was built from clean commit `3e9e1268` with:

- image ID: `sha256:c95e4f3f6e83275b0bbb0d2bcdc9b544a25cfaece3df100ae60de5444111ddb9`;
- source-tree digest: `0edfe838ddf23994107a6ea185e81829157f223c65e569b496ccaacba69324c8`;
- lockfile digest: `95dddd1ed146c4738004a0f3c97458737184cb5c03c730167af46f345e9c213b`;
- runtime-manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`.

Exactly one corrected container, `trade-rl-causal-teacher-smoke-v2-r2`, now runs against the distinct root `causal-teacher-main-20260813-v2-r2`. Docker launch inspection reported `state=running` and `OOMKilled=false`. CUDA stage evaluation remains fail-closed behind complete v2 selection and teacher admission.

The r2 runtime crossed the previously failing baseline-reset episode 3 and wrote four durable v2 records, proving the initial-state correction against the real environment rather than only unit fixtures. Docker remained `OOMKilled=false`. The first candidate's four-episode APTUSDT aggregate was mean gross `+1.781%`, mean net `-9.896%`, worst net `-15.229%`, mean turnover `5.627x/day`, execution cost `45,591.87`, and 58 trades, with zero hard-risk failures. Mean 24h/72h Pearson correlations were `0.054 / 0.052`. Although `2,075` proposed command changes were cost-suppressed and only `432` were submitted, the environment reported thousands of `portfolio:liquidity_cap` projections. This points to executable-target projection and repeated rebalancing as the dominant remaining churn path; candidate-level exposure and scale variants must be compared before changing the controller again.

The first cost-aware baseline candidate subsequently completed all 16 APTUSDT selection episodes. Its mean gross/net returns were `+0.339% / -9.465%`, worst net was `-17.304%`, mean turnover was `4.782x/day`, total cost was `155,886.75`, and it executed 268 trades. It had no hard-risk violation but accumulated 38,332 liquidity-cap projections. The second, 24h-only candidate was also negative-net in its first episodes and initially increased turnover.

Read-only root-cause tracing found that the portfolio risk model applies `max_position_to_market_notional=0.02` to the current 15-minute quote-notional volume before execution. Across the 16 APTUSDT selection episodes, the corresponding cap weight, using the configured initial capital, had 1%/10%/25%/50% quantiles of approximately `0.019 / 0.041 / 0.067 / 0.119`; it was below `0.25` for `79.47%` of bars and below `0.10` for `42.31%`. Therefore both the baseline `0.5` exposure cap and the one-factor `0.25` variant are frequently projected downstream. Because the controller's cost hurdle and no-trade band run before this time-varying projection, the risk layer can create executable-target changes that the controller never prices. This is a confirmed architectural mismatch, but no further target change is applied until the lower-exposure/scale candidates provide direct comparative evidence. Reward remains pure net log growth.

The next three complete APTUSDT comparisons did not remove the mismatch. The 24h-only candidate produced mean gross/net `+0.471% / -8.764%`, worst net `-17.572%`, turnover `4.487x/day`, and cost `146,546.86`. The 72h-only candidate produced `+0.516% / -9.445%`, worst net `-18.138%`, turnover `4.834x/day`, and cost `157,962.57`. Raising the execution-cost multiplier from `1.5` to `2.0` produced `+0.255% / -9.471%`, worst net `-17.529%`, turnover `4.734x/day`, and cost `154,447.53`. The higher hurdle suppressed 20,512 controller proposals and submitted only 1,803 changes, yet still accumulated 38,273 liquidity-cap projections. This direct one-factor test shows that increasing the controller-local cost penalty does not price or suppress downstream executable-cap changes.

### v2-r2 irreversible rejection on the first symbol

All 12 candidates completed all 16 APTUSDT selection episodes. None produced a positive-net episode. The best mean-net candidate was the low-exposure variant (`max_abs_target=0.25`): mean gross/net `+0.371% / -6.370%`, worst net `-12.905%`, turnover `3.230x/day`, total cost `107,407.74`, and 32,975 liquidity-cap projections. This was a material improvement over the approximately `-9.5%` mean net and `4.7-4.8x/day` turnover of most variants, but it still failed every economic promotion level that matters. The low-scale variant reduced turnover to `4.229x/day` and cost to `137,744.45`, but gross became negative (`-0.455%`) and net remained `-9.074%`. Wider no-trade and smaller max-delta variants also remained near `-9.3%` net.

At 193 durable records (192 complete APTUSDT records plus the first ARBUSDT record), code and evidence were checked together. The v2 selection lower tail is the minimum net return over every symbol episode, and the configured floor is `-5%`. Every candidate had already recorded an APTUSDT minimum between `-12.905%` and `-18.138%`. Additional symbols cannot increase a minimum, so all 12 candidates were irreversibly rejected before the remaining 1,535 replays. The run was stopped to avoid spending compute on an unchangeable gate outcome. Progress and checkpoint evidence were copied to `artifacts/universal/diagnostics/causal-teacher-v2-r2-terminal`; Docker reported `OOMKilled=false` and exit 137 after the process exceeded the 30-second graceful-stop window.

The result is a teacher-controller FAIL, not an RL reward result. Teacher admission, BC, critic warm-start, PPO, Lagrangian, and discounted PPO did not run. The scalar reward remains pure net log growth. The next correction must preserve the hard `max_position_to_market_notional=0.02` rule while exposing a causal executable liquidity cap to target construction, so downstream risk projection cannot silently introduce unpriced target changes.

### Approved liquidity-aware correction

Option 1 was approved after the v2-r2 stop. The hard
`max_position_to_market_notional=0.02` contract remains unchanged. Target
construction now uses only the preceding 96 decisions of quote-notional volume,
takes the 10th percentile, applies an 80% safety multiplier, and converts that
notional capacity to a weight using the artifact-bound reference equity. This
cap is applied before the controller's turnover, edge, cost-hurdle,
confirmation, max-delta, and no-trade decisions. A falling cap is handled as an
explicit teacher deleveraging rather than an implicit downstream risk
projection. Float32 action rounding is constrained inward by one ULP.

The cap parameters are bound into every candidate digest and checked against
the environment's hard portfolio-risk ratio before selection. Repeated
controller candidates reuse a per-contract cap cache. Durable v2 checkpoint
metrics now include liquidity-deleveraging count and cap min/median/max, and the
monitor aggregates these alongside gross/net return, turnover, execution cost,
trades, signal quality, and risk projection reasons. The reward remains pure
net log growth with no added cost, baseline, or drawdown penalty.
