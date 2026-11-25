# Cloud trend/range switch (synthetic mechanism study)

Status: Active — software-only. No economic execution authorized.

## G0 / question, counter-hypothesis, and stop rule

Can separating trend-following and range-reversion decisions reduce wrong-regime
exposure, and does a newly observed divergence add information to a range entry?
ADX is a lagging measure of strength, not an oracle for regime or direction.
A cloud interior need not be a stable range; contrarian trades can lose heavily
on breakouts. Extra filters can reduce activity without improving information.
The cheapest falsifier is a literal decision truth table and a native shared-book
synthetic fill/partial-exit oracle. Stop this mechanism on any failed oracle.

## G1 / fixed synthetic modes (not economic parameters)

Use the same one-symbol hourly MarketDataset for every feature and native replay.
The read-only adapter uses existing visible-cloud MTF (default strict RSI,
MACD_HIST, MFI), completed 4h ADX(14)/100 and ATR(14)/close, canonical hourly
RSI(14) centered on [-1,1] and Bollinger position(20) = (close-mean)/(2*std).
Existing with_price_channels supplies prior-only 20/10-bar channel distances.
The derived channel Dataset is temporary and is never published or substituted
for the replay's source. All indicators must be ready at the decision; source
rows unavailable at their original close are conservatively excluded. No data
are backfilled. Every mode has the same common mask.

- Trend entry: ADX >= 25, six consecutive ready hourly cloud decisions above
  or below the cloud, and a matching close beyond the prior 20-hour channel.
  Held trend: keep exact quantity until ADX <= 20, cloud direction fails,
  or close breaks the opposite prior 10-hour exit channel.
- Range entry: ADX <= 20, six ready hourly cloud-inside decisions, and
  centered RSI <= -0.4 (RSI <= 30) / >= 0.4 (RSI >= 70) together with
  Bollinger position <= -1 / >= 1 for long / short respectively.
  Held range: exit at the Bollinger center or beyond, cloud exterior,
  ADX >= 25, or 24 hours from the first observed fill, whichever comes first.
- New entries never occur in 20 < ADX < 25. Held positions may persist in
  that band if their own exit condition is not met. Missing context exits.
- S0 trend_only: no range trades.
- S1 hybrid: both branches.
- S2 hybrid_divergence: S1 plus a new same-side, non-conflicting confirmed
  hourly divergence for range entry. No change to trend branch.
- Every arm uses the same entry weight cap min(0.25, 0.0025 / completed
  4h ATR fraction, native gross and symbol limits), with exact quantity
  hold until exit. This is a sizing proxy, not a stop or guaranteed risk.
- One owning regime per submitted entry. Unknown inherited exposure is
  flattened. No switching a held position's regime or direct reversal.
  Exit intent persists through partial fills, even if the signal recovers.
  Once flat is observed, wait four hours before any new entry. Flat pending
  orders are cancelled via exact-quantity HOLD, not repeatedly increased.
  Consequently a delayed entry may be cancelled before eligibility; the
  policy does not claim order-latency invariance. The future runner must bind
  the exact context source Dataset to the replay; snapshots alone carry no
  Dataset identity.
  Termination and native hard risk retain authority.

## G2/G3 / acceptance and boundaries

Tests use only generated candles and literal snapshots, independent
channel/extrema and RSI/BB unit checks, delayed and missing data, prefix
invariance, mirrored range and trend thresholds, no conflicting signals,
latched exit, native partial fill, cooldown, exact quantity, risk cap,
and a native common-engine integration. A deliberately adverse synthetic
breakout must be reported as a limitation, not rescued by tuning.

No economic result, market data, sealed/unused scope, live or paper order,
or old-study rerun is authorized by this implementation. Previous T0/T1/T2,
C2/C3 and closed trend-stop research are unchanged. A future economic
protocol must fix scope, data identity, controls (including cash, trend-only,
simple hold and a no-divergence hybrid), costs, risk, metrics, and independent
result-blind review and owner authorization. No independent profitability claim.


## Predeclared development comparison v1 (awaiting review and G4)

The frozen machine-readable design is
[cloud-regime-switch-dev-protocol-v1.json](../../report/cloud-regime-switch-dev-protocol-v1.json).
The 11 arms, six independent asset-year books and three cost/latency
settings define 198 fixed comparison cells. This is a bounded development
falsification of S2; S0 and S1 identify regime-routing and divergence effects.
All arms run in the same native book with the same cost, capital and risk
assumptions. The 25% futures buy-and-hold control retains exact quantities
subject to native risk; it is not unlevered 100% spot performance.

Only the previously consumed 2022-24 BTC/ETH archive scope is proposed.
Reconstructed canonical input artifacts are now content-bound before the
new economic results; the JSON freezes both Dataset IDs, manifest/NPZ
hashes and the reconstruction receipt. Historical available-at times and
settlement mark/quantity remain approximations. Do not call this a fresh
sample or assert verified historical publication timing.

The author packet must bind the exact committed protocol and implementation
head. Separate authenticated result-blind review and owner G4 authorization
are prerequisites to the first economic replay. No self-approval, no result
rescue, no unused/FS1 access, and no new threshold search. The complete
roster is required for any decision. Failed or incomplete studies do not
create permission to change the rules.

The authorized runner tools/cloud_regime_switch_dev.py publishes each cell
exclusively, with native ledger and admission identity hashes. New software
uses the opt-in native portfolio ledger v2 exposure observer, without changing
the fixed reporting formula or execution. Its --summarize mode independently
reads back all recorded cells, verifies the ledger digest, window, dataset,
symbol, execution-policy identity and reported gross exposure, and applies
the fixed S2-only comparison rule. The expected execution identity comes from
the frozen scenario and independently SHA-256-bound Dataset manifest metadata
(including funding-price-rate presence), read without decoding market arrays.
The metadata reader refuses nonregular/symlinked inputs and manifests above
8 MiB. It does not attest historical market data or the publisher of the
ledger. A v1 ledger cannot be silently promoted to v2 evidence; no historical
result may be rewritten or rerun under the successor. Missing/invalid primary cells are INCONCLUSIVE; technical
failures do not authorize a retry. Latency results are descriptive and cannot
by themselves establish latency robustness.
