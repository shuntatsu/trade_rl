# Trade RL

Trade RL is a research-grade baseline-anchored residual reinforcement-learning core for low-frequency perpetual-futures portfolio allocation. The maintained policy does not choose arbitrary portfolio weights directly. It makes bounded residual decisions around a deterministic trend baseline, while an independently accounted shadow book provides the reward and evaluation reference.

> Production status: **NO-GO**. The repository now models substantially more realistic market mechanics, but the historical research result predates those mechanics and must be rerun. No profitable or production-authorized strategy is claimed.

## Maintained design

The repository has one authoritative `trade_rl` package. The former `mars_lite` stack, direct-action PPO path, duplicate metric implementations and legacy tests remain removed.

The current core includes:

- an exact regular-time OHLCV/mark/index market contract with tradability, warm-up, feature availability and feature-age masks;
- signed-quantity, cash-based self-financing accounting with natural weight drift;
- decisions made after a completed bar and executed from the next bar open;
- volume-participation limits, partial fills, quantity steps, minimum notionals, dynamic fee/spread inputs and nonlinear impact;
- funding, maintenance margin, forced liquidation and liquidation fees;
- seeded execution-domain randomization shared by hybrid and shadow books;
- one operational guardrail and pre-trade risk path shared by training, evaluation and serving;
- observations containing both hybrid and shadow state, masks, risk scales and relative NAV;
- real-hour strategy, episode, decision and discount-half-life configuration;
- random or explicitly carried initial book state;
- independent-fold and continuous-account OOS identities that cannot be confused;
- validated serving bundles, atomic registry activation and fail-closed hot swaps.

## Install

```bash
uv sync --all-extras --dev
```

## Commands

```bash
uv run trade-rl --version

# Preferred: keep discounting stable in physical time.
uv run trade-rl train config \
  --timesteps 1024 \
  --decision-hours 4 \
  --discount-half-life-hours 24 \
  --seed 0 --seed 1

uv run trade-rl walk-forward plan \
  --bars 220 --train-bars 80 --checkpoint-bars 10 \
  --selection-bars 10 --test-bars 20 --purge-bars 2 --max-folds 2
```

An explicit `--gamma` remains available for controlled comparisons, but `gamma=0.5` is no longer presented as the recommended default for a multi-day trend strategy.

## Verification

```bash
uv run ruff check .
uv run ruff format --check --diff .
uv run mypy trade_rl
uv run lint-imports
uv run pytest --cov=trade_rl --cov-branch --cov-report=term-missing
uv run trade-rl --version
```

See [Architecture](docs/ARCHITECTURE.md), [Research Status](docs/RESEARCH_STATUS.md), and the [environment hardening design](docs/superpowers/specs/2026-07-13-realistic-environment-hardening-design.md).
