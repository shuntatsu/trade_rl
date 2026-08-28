# Causal Alpha V7 Selection Rejection — 2026-08-26

## Terminal result

Causal Alpha V7 passed Signal and was rejected at Selection. Admission and
BC/RL were not opened.

- Branch: `codex/causal-alpha-v5-research`
- Corrected source commit: `351dd29a6eb4415e10c94e2510630fee672b22b1`
- Docker image: `trade-rl-causal-alpha-v7:351dd29a6eb4-6726b3737df9`
- Docker image ID: `sha256:7af2aa3e981f3ce5809b66ee856c7bfcf3d5fb14bb72562d116e438e1cd825a3`
- Source-tree digest: `34af31a8ea2305e8cfe94a5abc1f2b6b9194e75141a0c7875262eea400329912`
- Runtime-manifest digest: `6726b3737df9fbacf6787f3d02894e846c512a840bec4dd037538a02af1480b0`
- Run: `causal-alpha-v7-prod-20260826-r4`
- Docker volume root: `/workspace/var/runs/causal-alpha-v7-prod-20260826-r4`
- Signal evidence digest: `a204e1cdde80c718affe3d8c2093e948e401344067286bb745f9a02b013bf0a5`
- Selection evidence digest: `a97a4e1d533be3970a435a83a4cdf6d89ef1e3b98c0025d8a5040580584d2017`
- Terminal result digest: `565053cc367a7b11c7fb0e85810f394cd28704a2e26c3019667c50476f1c938a`
- Exit code: `3` (`selection_rejected`)

The corrected run also proved that centered relative-volume can legitimately be
negative. The implementation defect that rejected it as invalid was fixed by
commit `351dd29a`; 49 V7 tests, Ruff, and Mypy passed before this run.

## Candidate economics

| Candidate | Balanced gross wealth | Balanced net wealth | Minimum symbol net wealth | Median symbol net wealth | Positive net scopes | CVaR 10% | Net/gross retention |
|---|---:|---:|---:|---:|---:|---:|---:|
| `v6_control` | 0.941122 | 0.935641 | 0.833795 | 0.963561 | 19.44% | -6.60% | 99.42% |
| `symmetric_contrarian` | 0.968042 | 0.961676 | 0.886882 | 0.982323 | 25.00% | -5.80% | 99.34% |
| `causal_calibrated` | 0.956625 | 0.952768 | 0.867895 | 0.981505 | 16.67% | -5.78% | 99.60% |

Every candidate failed the same fixed gates:

- `symbol_balanced_gross_wealth`
- `symbol_balanced_net_wealth`
- `minimum_symbol_net_wealth`
- `median_symbol_net_wealth`
- `positive_net_scope_fraction`

No threshold was relaxed, no symbol was excluded, and the holdout remained
sealed.

## Per-symbol net wealth

| Symbol | V6 control | Symmetric contrarian | Causal calibrated |
|---|---:|---:|---:|
| APTUSDT | 0.977434 | 0.989477 | 0.981505 |
| ARBUSDT | 0.991723 | 0.970030 | 0.979576 |
| BCHUSDT | 0.983534 | 0.983534 | 0.983534 |
| BNBUSDT | 0.955233 | 1.032384 | 1.004488 |
| BTCUSDT | 0.887610 | 0.927261 | 0.907943 |
| LINKUSDT | 0.963561 | 1.007207 | 0.989526 |
| LTCUSDT | 0.998360 | 0.982323 | 0.996268 |
| SOLUSDT | 0.833795 | 0.886882 | 0.867895 |
| XRPUSDT | 0.847534 | 0.887148 | 0.876781 |

Only BNB and LINK finished profitable under the best candidate. The objective
of robust, symbol-independent trading was not achieved.

## Episode progression

The values below are per-episode net wealth aggregated across the nine
independent symbol simulations.

| Cutoff | V6 control | Symmetric contrarian | Causal calibrated | Behavior |
|---:|---:|---:|---:|---|
| 31575 | 0.849371 | 1.087363 | 1.000000 | candidate directions differ |
| 34456 | 1.082432 | 1.082432 | 1.082432 | shared short exposure |
| 37337 | 1.000000 | 1.000000 | 1.000000 | all flat |
| 40218 | 0.703162 | 0.703162 | 0.703162 | shared long exposure, all symbols lose |
| 43099 | 1.000000 | 1.000000 | 1.000000 | all flat |
| 45980 | 0.986598 | 0.986598 | 0.986598 | mostly shared long exposure |
| 48861 | 1.000000 | 1.000000 | 1.000000 | all flat |
| 51742 | 0.861566 | 0.861566 | 0.861566 | mostly shared long exposure |

V7 did avoid pointless activity in three low-confidence episodes. However, the
large losses in cutoffs 40218 and 51742 dominate all earlier gains.

## Root-cause evidence

The failure is gross directional/retention error, not execution resolution:

- At cutoff 40218 all candidates remained long for all 25,920 decisions.
  Gross log return was -0.333969 and net log return was -0.352168.
- `cadence_hold` alone contributed -0.289747 net log return.
- The nominally `supportive` slow state contributed -0.281230.
- The highest confidence quartile contributed -0.327746 for the control path.
- At cutoff 51742, 23,040 of 25,920 decisions were long and `cadence_hold`
  contributed -0.136194 net log return.
- Across the full best candidate, net/gross retention was 99.34%; changing
  fees or adding 1-minute execution data cannot recover a gross-loss strategy.
- The three candidate forecast transformations often collapsed to the same
  target path because the shared V6 target compiler retained the inherited
  position when no new entry-quality edge was present.

The architectural defect is that entry suppression and position exit share the
same `hold previous` behavior. Avoiding a low-confidence new trade is correct;
retaining an unsupported inherited position for the same reason is not.

## Required next iteration

V8 should separate entry, continuation, and exit decisions while preserving the
pure net-log reward and all fixed gates:

1. Keep strict cost-aware confirmation for new entries.
2. Evaluate the current position's signed, risk-adjusted continuation value at
   every fast decision cadence.
3. Exit toward flat when continuation value is not positive for a causal
   confirmation window; do not jump directly from long to short or vice versa.
4. Preserve a position for hours or days while continuation evidence remains
   positive, so the exit rule does not create churn.
5. Add full replay-level durable Selection checkpoints so a corrected run can
   resume without repeating completed symbol/candidate simulations.
6. Do not add 1-minute data unless a profitable gross path later fails because
   of execution/slippage evidence.

