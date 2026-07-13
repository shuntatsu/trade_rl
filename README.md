# Trade RL

Trade RL is a research-grade, baseline-anchored residual reinforcement-learning core for portfolio allocation. The maintained policy adjusts a deterministic causal baseline through a bounded, explicitly identified action specification while an independent shadow book supplies the comparison path.

> Production status: **NO-GO**. The repository provides research, evaluation, artifact and serving contracts. It does not claim profitability or production authorization.

## Action and environment v3

The maintained environment now provides:

- a dynamic residual action contract with independent fast, slow and risk controls, optional alpha scale and optional causal factor residuals;
- no dead alpha action when alpha is disabled, with exact action names and an ActionSpec digest bound into training and serving artifacts;
- time-series, cross-sectional, long-only, market-neutral and cash-or-trend deterministic baselines, including a nonzero one-symbol mode;
- continuous 24/7 and irregular session-calendar datasets with feature-level availability, staleness and execution metadata;
- next-open execution with fees, spread, impact, paired stochastic slippage, participation limits, minimum notional, lot/tick rounding, borrow, funding schedules, market/limit orders, latency, margin, corporate actions and delisting settlement;
- hard risk limits that override soft turnover throttles during concentration, leverage or emergency drawdown violations;
- economic terminal states for insolvency, minimum equity, execution-cost exhaustion, margin call and liquidation instead of training-process crashes;
- cash, baseline, random, stressed, partial-fill and restored causal reset states, duration curricula and regime/stress episode sampling;
- observation schema v3 with per-feature masks and staleness, factor loadings, requested and realized execution state, cash/net/gross/margin state, previous action and optional finite-horizon time;
- fold-fitted, frozen, content-addressed observation normalization that preserves categorical masks exactly;
- reward schema v3 prioritizing absolute log-wealth growth, then excess growth, with incremental drawdown and rolling baseline-underperformance progressive hinges;
- PPO exploration/network controls plus sealed-comparison support for SAC, TD3 and TQC;
- AUM capacity curves, action saturation/projection diagnostics and permutation-aware shared per-asset encoding.

## Identity and serving

Environment identity includes dataset, calendar, action specification, alpha/factor artifacts, normalizer, episode curriculum, trend, reward, risk, execution and AUM. Serving bundle v3 binds the exact action size, action names, ActionSpec digest, observation schema and size, environment digest and normalizer digest. Runtime inference rejects non-finite, incorrectly shaped or out-of-range actions rather than clipping them silently.

## Commands

```bash
uv sync --extra dev
uv run trade-rl train config \
  --timesteps 102400 --decision-hours 4 --discount-half-life-hours 168 \
  --n-steps 2048 --batch-size 64 --log-std-init -0.5 --target-kl 0.02 \
  --policy-net-arch 128 --policy-net-arch 128 --seed 0 --seed 1 --seed 2
uv run trade-rl environment config \
  --initial-capital 100000 --calendar-kind continuous_24_7 \
  --episode-hour-choice 168 --episode-hour-choice 720 \
  --initial-state-mode cash --initial-state-mode stress
```

## Verification

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch
```

See [Architecture](docs/ARCHITECTURE.md) and [Research Status](docs/RESEARCH_STATUS.md).
