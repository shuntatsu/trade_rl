# RL execution overlay for the agy anchor (`anchor_rl_execution_dev_v1`, development study)

Status: Active (draft; it will be frozen before any training run).

## Idea

The ported agy anchor has a real gross edge in the agy lab. It loses most of it to turnover costs. A fixed rebalance stride is a crude answer to "when is moving toward the target worth its cost?". A learned policy can condition that decision on information that a fixed rule ignores:

- the size of the gap to the target;
- the conviction of the target;
- volatility and cross-sectional dispersion;
- liquidity, through quote volume;
- funding;
- the current drawdown.

This uses RL for execution and sizing, not for alpha. It is a low-dimensional control problem with a dense and honest reward, which keeps overfitting risk lower than learning raw alpha.

## Design

| Element | Definition |
|---|---|
| Base target | The ported anchor's synthesized spot/perp target, at daily decision points (every 6 bars). It uses the cash-account scale and the risk config of `agy_anchor_realistic_v1`: drawdown 0.125→0.25, max gross 2.5, participation 0.01. |
| Action | One discrete choice per decision: trade fraction `f ∈ {0, 1/3, 2/3, 1}` of the gap between the current weights and the base target. It applies to all legs proportionally. |
| Observation | Gap L1; anchor conviction (mean abs alpha); trailing 60-day portfolio vol; cross-sectional dispersion of 3-day returns; mean log quote volume relative to 60 days; mean funding; current drawdown; days since the last trade. All observations are causal. |
| Reward | Equity change net of every cost, from the trade_rl shared-portfolio engine (gamma = 1, as the repository trainer enforces). |
| Trainer | `agents.train_agent` using the lazily loaded `stable-baselines3` / `sb3-contrib` agent wrapper (PPO first). Seeds 0–4. The per-symbol-action `fit_shared_portfolio_ppo` trainer is not this overlay trainer. |
| Train window | Tier D 2022-01-01 .. 2023-12-31 (known-used). |
| Validation window | Tier D 2024-01-01 .. 2025-06-30 (known-used; development selection, reported in two halves). |
| Diagnostic test window | Tier D 2025-07-01 .. 2026-07-31 (known-used; separate internal test split, not unseen FORWARD/unused/final data). |
| Baselines | The same base target with a fixed full trade at strides of 6, 18 and 42 bars, plus CASH. |

## Development decision

- **`DEV_PROMISING`** requires all of the following:
  - The median-seed validation net Sharpe beats the best fixed-stride baseline.
  - At least 4 of the 5 seeds beat it.
  - It wins in both validation halves.
  - No validation drawdown reaches 0.25.
- **Next step if promising:** freeze one configuration, without seed picking (the median-seed policy, chosen by a rule fixed here). It then enters the pre-registered FORWARD evaluation together with the other candidates, under a family-wise multiplicity correction.
- **Otherwise the line stops.**

## Caveats

- The anchor data are known-used, and agy explored them heavily.
- The fixed-stride baselines were also informed by agy diagnostics.
- The `test` split above is already known-used Tier D data; it is not the unseen FORWARD evaluation, must not be treated as a fresh holdout, and is not an extra condition for the `DEV_PROMISING` decision. No real-data `test` evaluation is authorized by this draft or by the software-only harness.
- This is development evidence only.
