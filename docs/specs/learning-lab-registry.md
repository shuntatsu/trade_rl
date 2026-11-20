# Learning lab: method registry (draft)

Status: Active (draft; software harness in `trade_rl/evaluation/learning_lab/`, synthetic tests only, no economic claim).

Every method is evaluated in one common harness. The harness provides:

- the trade_rl shared-portfolio engine, with realistic costs, a 1% participation cap, a cash account, and a 0.125→0.25 drawdown ramp;
- a walk-forward split on known-used Tier D data: train on 2022-01..2023-12, validate on 2024-01..2025-06 and test on 2025-07..2026-07, with the decayed regime held out last;
- 5 seeds;
- a trial registry in which every run counts toward a deflated Sharpe.

A method is shelved with the reason it stopped. Nothing is deleted.

| ID | Method | Role | Why it might help | Status |
|---|---|---|---|---|
| L1-PPO | PPO, discrete trade fraction | execution and rebalance timing on the anchor target | the anchor's gross edge is eaten by turnover | drafted |
| L1-SAC | SAC, continuous trade fraction | same task | smoother control, off-policy sample efficiency | planned |
| L1-QRDQN | QR-DQN (distributional) | same task | uses the return distribution, so it can avoid tail-risk actions | planned |
| L1-ENS | ensemble of the L1 agents (seed and algorithm bagging) | same task | lowers the variance of single-agent policies | planned |
| L2-RARL | adversarial robust RL: an adversary perturbs costs, slippage, latency and returns within bounds | training regime for L1 | robustness to regime and cost shift, such as the 2025–26 decay | planned |
| L2-DR | domain randomization of costs, delay and vol | simpler L2 baseline | the same aim at lower cost | planned |
| L3-META | meta-labeling: a gradient-boosted classifier of whether each sleeve position will pay over its holding horizon, used to size positions | sizing overlay | filters weak signals | planned |
| L3-AVAL | adversarial validation: a classifier that separates train from test periods | feature and regime-drift detection | finds features that are stable across regimes, which addresses decay | planned |
| L3-RANK | learning-to-rank (LambdaMART) of cross-sectional 3-day returns | alpha (cross-section) | non-linear combination of micro, funding and OI features | planned |
| L4-ALLOC | online sleeve allocation (Bayesian bandit or exponentiated gradient) | portfolio | cuts decaying sleeves automatically | planned |
| L4-CVAR | risk-sensitive RL with a CVaR or drawdown-aware objective | portfolio and sizing | matches the drawdown budget directly | planned |
| L5-OFFRL | offline RL (CQL/IQL) from logged baseline trajectories | execution | conservative and less prone to overfitting | backlog |
| L5-GEN | generative market simulator (diffusion or TimeGAN) for training augmentation | data | more diverse training paths; the realism risk is high | backlog |
| L5-SEQ | sequence model (Transformer or TCN) on 15m/1h order flow | alpha | uses micro-structure beyond hand-built factors | backlog |

**Order:** L1-PPO first, so the harness is proven on one method. Then L1-SAC, L1-QRDQN and L2-RARL in parallel. Then L3 and L4. Finalists are frozen and go to FORWARD together, under a family-wise correction.

**Hardware:** RTX 4070 Ti SUPER (16 GB).
