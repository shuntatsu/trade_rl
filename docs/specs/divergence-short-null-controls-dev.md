# R2 negative-control study: long-only and causal weekly short nulls

Status: Active — software-only research extension built on PR #1344. Known-used/post-hoc Binance data only; NOT independent or prospective economic evidence, G4 authorization, or live trading.

## Purpose and falsifier

Previous 2022–24 BTC/ETH historical R2 outcomes were profitable in six consumed yearly cohorts. That may reflect simply avoiding short exposure, rather than predictive information in the momentum/cloud filter. Compare R2 with long-only C2 and direction-blind controls. If simple controls perform as well or better, R2's incremental informational value remains unproven.

## Fixed controls

N0: original C2 except no short exposure at any time. Original long decisions unchanged.
N1/N2/N3: original C2 with weekly deterministic randomized short permissions using seeds 17, 37, 97 fixed before these control returns. For each UTC Monday-to-Monday week a SHA-256 uniform value is drawn only at the first **available** hourly context observation. It is accepted when less than the trailing 90-day as-of fraction of ready hourly context where both R2 filters would be true. If fewer than 48 ready hours, all shorts fail closed. Decision is frozen for that calendar week, with missing context still preventing use.

The random controls have roughly matched historical eligibility frequency, **not** exactly matched realized short position hours, fills, holding periods or turnover. Report those separately for each control; do not claim that return gaps prove information value.

Source ownership: data/features/short_null_permissions.py calculates causal boolean permissions, strategies/short_null_control.py wraps C2 proposals; common native risk/execution/accounting alone owns P&L. No changes to canonical features, frozen Dataset identities, package-root exports or production routes.

## G0-G3 and research use

Focused tests: registered seeds, deterministic prefix, future mutation independence, unavailable or delayed context refusal, regular 1h clock, exact HOLD, disallowed short flat requests, and native single-book fills. Research runner only reads previously consumed Binance USD-M 1h/mark/funding BTCUSDT and ETHUSDT 2022/2023/2024 with 2021-10 warmup, annual 10k USDT and target 25%. Fees .0005, spread .0002, impact .0001, 5% participation, no extra latency; double costs for all controls. 4 controls x 6 cohorts x 2 cost regimes = **48** recorded cells, including negative results. Three random seeds were not selected by outcome.

External scratch protocol short_null_control_protocol_20261123.md is fixed before the N0-N3 result observations but after R2 results. Thus **all** cohorts are known-used and post-hoc; claims of independent significance are forbidden. Sealed/unused/FS1 data and paper/live orders remain out of scope. The source, results and hashes must be retained in research receipts and PR discussion.

## Conditional daily short-opportunity diagnostic (offline-only)

The read-only evaluation/short_opportunity_audit.py module tests whether
R2's visible-cloud slope carries directional information within a common
C2 six-hour bearish-cloud opportunity set. It does not execute trades or
compute a second P&L. At completed UTC 00:00 bars, when the native daily
visible cloud is bearish for all prior six completed hourly decisions and
macro context is ready, classify the 30-day momentum and visible-cloud slope:
G0 momentum >= 0, G1 momentum < 0 and slope >= 0, G2 both < 0.
Use the **future 24h traded-close** price change, negative for a short,
as an offline response label *only after* the full 24h has occurred inside
the already-used yearly scope. Report counts, means, medians, favorable
fractions and a past-only realized-vol normalized response for all three
categories. Distinct daily samples avoid overlapping 24h labels in one
symbol, but they are not independent across assets, year regimes or other
experiments.

The conditional analysis does not match actual executed short durations,
fees, funding, order book fills or trade P&L. Thus it is **not** an
exposure-matched strategy evaluation; that is a separate, still-open
research question. Source labels are not usable by portfolio policies
or ML features at their observation time. Protocol and all six research
receipts are retained in the Fedora scratch workspace with hashes;
result interpretation belongs to the PR conversation, not a frozen
future-data or G4 study. No data-driven filter tuning is authorized.
## Native paired short-hour disagreement diagnostic (used-data only)

The new read-only evaluation module trade_rl/evaluation/paired_short_exposure.py
compares two existing, executed single-symbol portfolio replays **hour by
hour**, not by selecting equal annual trade durations retrospectively.
At native hourly interval [t,t+1], use each canonical ledger's exact
quantity **before** the interval to classify four disjoint states: both
short, only R2 short, only control short, neither short. Record count,
mean and median next-hour short-signed traded-price change, and favorable
fraction for every bucket, including empty ones. The price at t+1 is an
**offline response label**, NEVER an input to a decision, a new economic
PnL estimate, or a replacement fill/cost/funding calculation.

The previously published native directional equity buckets also count
entry intervals (flat before, short after); those are not equivalent
to pre-interval short-held hours. The scratch runner reconciles the two
definitions using the actual native flat-to-short entry intervals and
refuses inconsistent counts, fees, returns, or funding compared with
the existing saved R2/N0-N3 receipts. Any direct long/short reversal,
gapped native ledger, nonfinite price, unexpected timestamp or noncontinuous
portfolio equity is rejected. The null comparison cannot claim equal risk
because pre-interval signed positions also omit exact intrabar execution,
position sizing, and conditional market-state differences.

This analysis only uses the already consumed BTCUSDT/ETHUSDT 2022-2024
Binance study years, with no future/opened unused data or live orders.
All 24 comparisons (4 controls for each of 6 asset-year cohorts) and
the research protocol/runner hashes must be retained even when R2 loses.
No economically validated strategy follows from these diagnostics.
