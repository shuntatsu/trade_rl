# Complete 96-Feature Multi-Timeframe Training Design

## Objective

Replace the lightweight ten-feature Binance research preset with an exact 96-feature causal contract and make the maintained example a one-hour, three-asset, three-seed, two-fold complete research pipeline.

## Feature architecture

The dataset has four native clocks: 15 minutes, 1 hour, 4 hours, and 1 day. Every clock calculates the same 24 features from its own OHLCV and funding series. The generic multi-timeframe builder aligns only completed and available native events to the one-hour decision clock.

Indicator calculation is centralized in `trade_rl/data/features.py` so base-clock and auxiliary-clock features share identical mathematics and validity rules. Each event returns its earliest source index, allowing availability to be the maximum `available_at` timestamp across the complete source window.

The 24-feature contract includes returns, realized volatility, volume, funding, RSI, MACD, Bollinger Bands, ATR, ADX, Stochastic, CCI, Williams %R, OBV, and four causal Ichimoku representations. Ichimoku chart shifts are not applied because the training feature contract cannot use future information.

## Training architecture

The base dataset and policy decision interval are both one hour. The action space contains fast tilt, slow tilt, and three net-zero relative allocation factors. Risk tilt is disabled to remove the learned all-cash local optimum while preserving backward-compatible internal action objects with `risk_tilt=0`.

The complete run uses 262,144 PPO timesteps for each of three seeds. Nested walk-forward uses two folds and 65,536 timesteps for each seed in each fold. Dataset identity, feature names, repeated build digest, factor artifact, policies, checkpoints, and sealed test evidence are mandatory outputs.

## Verification

Tests lock exact feature names and cardinality, numerical indicator behavior, prefix causality, delayed availability, one-hour configuration, five-dimensional action layout, and full example settings. Standard lint, formatting, typing, architecture, and full repository tests remain required before integration.
