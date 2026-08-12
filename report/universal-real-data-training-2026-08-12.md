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
