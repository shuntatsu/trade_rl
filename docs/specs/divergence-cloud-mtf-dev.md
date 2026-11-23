# Divergence + visible Ichimoku + multi-timeframe research strategy

Status: Active (opt-in software prototype). No registered economic study, real-data
result, selected winner, unused/final evaluation or live-trading authority.

## Research question and counterhypothesis (G0)

Can causally confirmed divergences on 1h/4h improve *net* decisions when
filtered by the visible daily Ichimoku cloud, compared under identical
accounting/cost/risk with cash, cloud-only, and spot hold? Conversely, the
signals may be coincidental mean-reversion patterns whose edge vanishes with
latency, trading costs and cross-regime validation.

All reported synthetic replay returns are mechanism checks, not evidence of
market profitability. Market research requires a separate, result-blind G0-G3
protocol, fixed dataset/scope/controls and authorized G4 stage.

## Code ownership and opt-in registration

- \`trade_rl/data/features/indicators/divergence_bank.py\`: named indicator
  registry. Twelve names reuse existing \`FeatureKind\` and canonical
  \`calculate_feature_events\`; eight optional additions live separately in
  \`indicators/extra.py\`. Warm-up and unavailable values are NaN.
- \`trade_rl/data/features/divergence.py\`: strict-tie pivot confirmation;
  four types (regular/hidden, bullish/bearish), individual pivot and detection
  timestamps, per-indicator count, optional line filter and deduplication.
- \`trade_rl/data/features/indicators/ichimoku_cloud.py\`: **visible** cloud.
  A span computed at native day s is plotted 26 days ahead. Therefore at
  decision day t the visible span comes from s=t-26. A 52-day span needs 78
  complete days before it becomes visible. The legacy \`FeatureKind\` cloud
  position uses unshifted values; it is *not* silently overwritten.
- \`trade_rl/data/features/divergence_mtf.py\`: gap-free hourly source,
  completed UTC 4h/1d aggregation, as-of observation availability, one-shot
  1h/4h divergence events and latest available daily visible cloud.
- \`trade_rl/strategies/divergence_cloud_mtf.py\`: one-symbol
  \`PortfolioPolicy\`; the existing shared Risk/Execution/Accounting runtime
  alone owns fill, fees, pending orders, margin and settlement.

All data entry points are imported explicitly from their owning
\`trade_rl.data.features.divergence\` / \`divergence_mtf\` /
\`indicators.*\` modules. The policy is imported explicitly from
\`trade_rl.strategies.divergence_cloud_mtf\`. Neither data nor strategies
package-root exports change, preserving existing provenance-bound source. **No existing MarketDataset
schema, default FeatureSpec roster, frozen protocol, or historic Dataset
identity is modified.**

## Explicit prototype strategy (G1)

1. Daily cloud: price above the visible top permits long, price below bottom
   permits short, inside/unavailable permits cash only.
2. The most recent eight 1h decision rows must include an aligned 4h
   same-side divergence event; the last four hourly rows must include a 1h
   same-side event. At least one event must be newly published now.
3. Entry target is +0.25 long or -0.25 short. Do not reverse on a lone event.
   An unchanged position requests **exact-quantity HOLD**, not free daily
   target-weight rebalancing.
4. A cloud regime flip or a confirmed opposing pair of signals closes the
   position; pending orders block new entries. Account risk constraints
   remain authoritative.
5. Default detector: pivot period 5, strict ties, up to 10 previous pivots
   and 100 bars, two distinct indicators, penetration-line validation.
   A looser detector is used only to exercise the synthetic E2E path; it
   has not been selected from economic evidence.

## Causality, oracles, limitations (G2-G3)

- Pivot p cannot be used before p+period right candles close. The signal is
  first observed at decision close t, evaluating endpoint t-1: never
  retroactively enter on p.
- The 1h price candle, every underlying native observation and its dependent
  features must be available at the decision timestamp. A late stale signal
  is refused; gap-filled or incomplete native 1h clocks are not accepted.
- Unit tests prove price/indicator prefix invariance, delayed-source
  rejection, no unconfirmed 4h/daily bar, cloud displacement, missing
  indicator continuity and the native shared-account exact HOLD and fees.
- Synthetic E2E test verifies calculation -> policy -> shared native replay;
  it does **not** constitute economic research. No broker, training,
  optimizer, actual market data, multi-symbol strategy or orders are provided.
- Extra indicators are research formulas; bit-for-bit TradingView parity is
  not claimed. Future evaluation should include fee/latency stress,
  long/short attribution, unused-data controls and trial-count correction.
