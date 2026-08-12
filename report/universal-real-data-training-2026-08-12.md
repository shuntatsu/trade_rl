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
