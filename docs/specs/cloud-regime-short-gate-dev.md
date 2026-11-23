# C2 conditional short-risk overlay R1/R2

Status: Active — opt-in prototype on **previously used historical data only**. This is post-hoc exploration, not independent economic evidence or live/paper trading authorization.

## G0 / competing hypotheses

Prior BTC/ETH 2022–2024 C2/C3 analyses suggest short-side losses offset gains in uptrending years. Requiring historical macro-direction confirmation could reduce wrong-side shorts; equally, it may eliminate profitable bear-market shorts or worsen costs.

## Frozen comparison (before R1/R2 economic outcomes)

R0 = original C2, six-hour confirmed daily visible Ichimoku cloud. R1 = identical long logic, but short entry/continuation requires strictly negative trailing 30×24-hour close return. R2 = R1 plus a strictly falling visible daily cloud midpoint versus seven completed daily candles ago. When a held short fails the gate, request zero via common account. Diagnostic counters track rejected short *proposals per hourly decision* and requests to flatten ineligible shorts, not unique rejected or filled orders. Long side, C2 confirmation, 25% target, risk and costs remain unchanged. No parameter search or retrospective winner promotion.

## Implementation and causal boundary

data/features/cloud_regime_context.py computes as-of 30-day return, 7-day visible-cloud slope, price distance to cloud and 30-day annualized realized vol. Volatility and distance are diagnostic-only (not additional gates). Strict input regularity and observation availability mask incomplete/late data. No use of future chart displacement or unfinished 4h/daily candles. strategies/cloud_regime_short_gate.py wraps C2, delegates exact quantity HOLD/risk/execution/accounting, and does not change canonical feature schemas, package-root APIs or frozen evidence.

## Research use

BTCUSDT and ETHUSDT Binance USDT-M perpetual history, 2022/2023/2024 annual resets of 10,000 USDT, warmup 2021-10. Existing replay costs: 0.0005 fee, 0.0002 spread, 0.0001 impact, 0.05 maximum volume participation, zero *additional* order latency. Full R1/R2 roster under normal and double costs even for failed cases. Unused/future/FS1 data not opened. G0–G3 review and independent validation remain separate.
