# Multi-symbol funding-carry development screen

Status: Active

Issue #810 Track B. The BTC/ETH spot-long / USD-M-perpetual-short carry passed all
twelve of its development screens, but the net return was small at gross 0.5 (see
[the carry capability record](../research/current-status.md)). This pre-registration
asks the one question that decides whether that line is worth continuing: does the
same fixed carry survive on more USD-M symbols, in more regimes, after the same
costs? It is registered, implemented and tested **before any delta-neutral carry
outcome for these symbols exists**, and this change reads no market data. It is
**not free of prior exposure**: earlier studies in this repository already read price
series (and, for some symbols, funding series) of most candidates. That exposure is
named under "Prior exposure and the limits of result-blindness" below, and it is why
the claim here is carry-outcome-blind, not data-blind. No real-data run is part of the
change that adds it.

## Claim boundary (G3)

Every bar before 2026-08-01 is known-used Tier D data. A result on it is exploratory
development evidence: it is never an edge claim, never unused-data validation and
never authorizes an order. The only confirmation path is a separately pre-registered
forward paper protocol (see [the prospective paper spec](funding-carry-paper.md)).
The FS1 warm-up span, the sealed window and any unused/final data are off limits;
the planner, the archive parser and the window validator all refuse them.

## Prior exposure and the limits of result-blindness

The first draft of this document said no funding, price or return of any candidate was
read. That absolute claim was wrong and is withdrawn. What is true:

- **Prior exposure.** `tests/fixtures/legacy/realdata_20260713_signal_metadata.json`
  records a 2026-07-13 real-data signal run over 15 symbols (fit cutoff 2025-12-06,
  features including `funding_bps`, gate not passed). Eight of the original nine
  candidates (SOL, XRP, BNB, ADA, DOGE, AVAX, LINK, LTC) and BCH are in it, so their
  price series, and funding as a feature, were read. The Issue #810 P2 development
  study (`dev_realdata_study`, BTC, ETH, BNB, XRP and ADA perpetuals, 2021-01-01 to
  2026-08-01) read prices and funding events of BNB, XRP and ADA, charged funding to
  directional arms, and ran a funding-carry tilt trial (a directional rule on trailing
  funding, not a delta-neutral pair) whose `NO_WINNER` outcome was seen. Other
  directional/PPO studies and tests name some candidate symbols and were not audited
  one by one. DOT is the only candidate absent from the legacy list.
- **What was not computed.** A search of the repository and its full history (the
  commits that introduced `assemble_carry_dataset` and `replay_carry`, the commit
  messages and remote branches matching `carry`, and the repository files naming a
  candidate symbol) found carry replays only for
  BTC and ETH (the twelve development screens and the forward paper attempts) and
  nothing that replays or reports a spot-long / perpetual-short carry, or its funding
  P&L, for any other symbol. A scratch run outside the repository would not appear in
  that search, so this is "none found", not a proof.
- **Consequence.** The protocol is carry-outcome-blind, not price-blind or
  funding-blind. The window, thresholds, cost model and decision rule were fixed from
  mechanism and arithmetic, not from these series, but the candidate choice could have
  been informed by general knowledge of them. The screen's result is exploratory for
  that reason as well as because every bar is known-used.
- **Candidate provenance.** The nine original candidates were named in the planner's
  brief as large-cap USDT pairs with both a spot and a USD-M perpetual market. That
  is not an independent derivation: eight of them overlap the legacy universe, and this
  change cannot show the overlap was unplanned.
- **BCHUSDT.** It has both markets and 2021 Q4 history, and no principled non-data
  reason to leave it out exists, so it is **included** (ten candidates, K for ten is 7).
  SUI, APT, ARB and OP, the other symbols of the legacy list, are not candidates only
  because they were listed after the 2021-10-01 lookback begins, which the listing-age
  rule already excludes without reading any value.
- **Attestation about this change.** The implementer (Claude Sonnet 5.5) read no market
  data in this task and computed no volume, funding or price statistic for any symbol;
  no real-data process was run and nothing was fetched. The liquidity floor moved from
  250,000 to 1,000,000 USDT per hour by arithmetic only: the first draft sized the
  largest leg at nine pairs (about 280 USDT), but the worst case is the minimum of six
  admitted pairs, 10,000 x 0.5 / (2 x 6) = 416.67 USDT, which the 0.1% participation
  stress fills in one bar only from 416,667 USDT of hourly volume; 1,000,000 keeps a
  2.4-fold margin. A test recomputes this. The attestation cannot cover what the planner
  or the owner had seen before the brief.

## Registered protocol `carry-multisymbol-dev-v1`

The constants live in `trade_rl/evaluation/carry_multisymbol/protocol.py`; the
canonical digest is
`49d3597e3c83dd3c13d71098111075a392c3afe595e5f48edc87aad36475d961`
(`python -m tools.carry_multisymbol_screen plan`). A changed constant is a new
protocol version and a new attempt; nothing is edited after a cell exists.

**Universe and inclusion.** Candidates, fixed in this order: SOL, BNB, XRP, ADA,
DOGE, LINK, DOT, AVAX, LTC, BCH against USDT, each needing both a spot and a USD-M
perpetual market (provenance, including why BCH is in, is stated below). A candidate is admitted only if all of these hold, decided from
timestamps and quote volume alone, never from a price change or a return:

- *listing age and minimum history*: both legs have a complete hourly series over
  the 92-day lookback 2021-10-01 to 2021-12-31;
- *liquidity proxy*: the median hourly quote turnover of each leg over that lookback
  is at least 1,000,000 USDT (about 24 million USDT a day). The largest equal-weight
  leg is about 420 USDT (six admitted pairs); the 0.1% participation stress fills it
  in one bar only from 420,000 USDT of hourly volume, and a leg part-filled on one
  side only would trip the matched-hedge stop for a liquidity reason, not a carry
  reason, so the threshold keeps a margin of about 2.4x at the median hour;
- *window coverage*: both legs are complete over the window except the single
  externally evidenced spot halt (2023-03-24 11:27 to 14:00 UTC, the Binance
  incident already used by the BTC/ETH screens), and funding coverage passes the
  assembler's existing no-gap rule.

An excluded symbol is recorded with its reason code and is **never replaced or
dropped later**. Fewer than six admitted symbols is `INCONCLUSIVE_UNIVERSE`, which is
terminal for this protocol: the universe, window and exclusion rule are never changed
and the screen is never rerun on a different universe. The decision records
`terminal_for_protocol: true` and `rerun_with_a_different_universe_allowed: false`, and
the runner refuses to reopen an output directory that holds it. The only way forward is
a new protocol version (for example with an externally evidenced halt calendar), decided
without reading returns. BTCUSDT and ETHUSDT are loaded as descriptive reference arms on the
identical window; they never enter the decision.

**Window and sub-periods.** Development window `[2022-01-01, 2026-08-01)`: the last
admissible bar opens 2026-07-31T23:00Z. It adds the 2022 bear/negative-funding
regime to the 2023-2024 regime of the BTC/ETH screens, because the counter-hypothesis
is that the premium was a 2023-2024 regime effect. Sub-periods are the calendar
years 2022, 2023, 2024, 2025 and 2026 (seven months). The window is part of the
protocol: the runner accepts exactly this `--start 2022-01-01 --end 2026-08-01`. The end
is the exclusive instant 2026-08-01T00:00Z (the last admissible bar opens
2026-07-31T23:00Z); any end after that instant is refused with a distinct embargo
message, and any other start or end is refused as not the registered window. The end
is exclusive for funding as for klines: a funding event at exactly 2026-08-01T00:00Z is
excluded by the archive source and again by the pair loader (the PostgreSQL store also
returns a row at the end instant), so no event of the warm-up span can enter.

**Strategy, unchanged.** `FundingCarryBot`: monthly equal-notional matched-quantity
spot-long / perpetual-short targets from the first usable bar, a real exit on the
final bar, no funding forecast and no funding-sign filter. Negative-funding stretches
are paid, which is the honest test of regime dependence.

**Cost model, unchanged.** Spot taker 10bp, perpetual taker 5bp, 5bp adverse
execution per leg, participation 1% of the preceding hour's quote volume, lot 0.001,
minimum notional 10 (`assemble_multisymbol_carry_dataset` reuses the BTC/ETH
assembler). Stress treatments on every arm: `cost2x` (fees and adverse execution
doubled), `delay1h` (initial entry delayed one hour), `capacity0p1` (participation
10x lower). A test pins the protocol constants to the assembled arrays.

**Hard risk, unchanged.** The irreversible 10% drawdown stop, matched-hedge check,
open/intrabar futures-collateral checks and canonical-termination invalidation of
`replay_carry`. A cell fails on any stop or margin breach, an incomplete replay, a
non-flat terminal state or a ledger drawdown of 10% or more.

**Gross and capital split.** Gross stays 0.5 of equity, split equally across the
admitted pairs (the portfolio arm runs 10,000 USDT shared; each symbol arm runs
10,000 / admitted USDT, i.e. the same per-pair size as inside the portfolio; the
reference pairs keep the original 5,000 / 10,000). The gross is deliberately not
tuned: net return scales about linearly with gross, so the screen decides sign and
robustness, and a profitable-but-small result is scaled only inside a later forward
protocol under its own pre-registration. Realized gross, cost and funding are
reported for every cell.

**Cells and pass rule.** Decision arms are the portfolio plus one independent arm per
admitted symbol; each runs the four treatments (`base`, `cost2x`, `delay1h`,
`capacity0p1`), up to 44 decision cells, plus 12 descriptive reference cells. Cell
pass: complete, flat, valid, no stop, ledger drawdown below 10%, full-period net
return above zero; the `base` cell additionally needs net return above zero in
every sub-period. The every-sub-period condition applies to the base cell only: the
stress cells are not required to be positive in every sub-period, only over the full
period. An arm passes only if all four of its cells pass.

**Family decision and multiple testing.** `CARRY_GENERALIZES_DEVELOPMENT` requires the
portfolio arm to pass **and** at least K admitted symbol arms to pass, with
K = ceil(20 x admitted / 32) (admitted 6, 7, 8, 9, 10 give K = 4, 5, 5, 6, 7). The family
is fixed: n symbol hypotheses and one portfolio gate, and no failing or excluded symbol
is removed after the cells exist. K comes from a Markov step: if one symbol clears the
five-sub-period base rule with probability at most p0 = 2^-5 (each sub-period sign a
fair coin), then P(passes >= K) <= n x p0 / K <= 1/20 for any dependence between
symbols. **That figure is not a valid error rate for this data.** The value
p0 = 2^-5 needs independent sub-period signs inside one symbol; a persistent funding
regime can make a symbol's five sub-periods positive together with probability far
above 2^-5 (up to one half under a common factor), so the 1/20 bound can be invalid,
not merely weakened. K is therefore a heuristic threshold and the family decision is
**not a controlled error rate**; the decision output carries `error_rate_controlled:
false` and reports the bound only as `markov_bound_if_subperiods_independent`. No
dependence-robust rule is adopted in this version; adding one is a new protocol
version. An independent-binomial tail is reported only as a diagnostic. Otherwise the
outcome is `CARRY_DOES_NOT_GENERALIZE`.

**Stop rule.** No cell, symbol, window, cost, gross or rule is changed after a cell
exists. A failed family stops this line; `INCONCLUSIVE_UNIVERSE` is likewise terminal for
this protocol and is never rerun on a different universe; a pass authorizes only a
separately pre-registered forward paper protocol, never live use. A software or data fault that
invalidates a run is preserved and answered by a new protocol version, not repaired
in place.

## Research assurance (G0-G2)

The author packet
[`report/issue810-carry-multisymbol-research-assurance-20261113-after1208.json`](../../report/issue810-carry-multisymbol-research-assurance-20261113-after1208.json)
states the counter-hypothesis (a 2023-2024 regime effect that fees and basis risk
erase on alts), the cheapest falsifier (the 2x-cost stress and every-sub-period rule
on the 2022 regime), the mechanism chain (leveraged long demand pays a funding
premium that the delta-neutral pair collects) and the independent oracle. Its digest is
`1a37bb8646e9cb6cbe22d02012411d74556655bb8fb8447fc0062bb3f3f8a213` and its status
is **UNREVIEWED** and `economic_execution_authorized=false`: the repository records no
review of this exact digest and these heads. This source-only reissue binds #1208's write-once Vision archive cache publication and the already merged #1226 fail-closed two-leg lookback-liquidity admission; the preceding after1226, after1203, 2026-11-13 and 2026-11-10 packets remain unmodified and UNREVIEWED. It does not claim descriptor-atomic protection from hostile concurrent cache-directory swaps. The oracle is a hand-computed three-pair ledger in exact rational
arithmetic (`tests/evaluation/test_carry_multisymbol_replay.py`) that does not call the
production sizing, cost or funding code. The packet's heads bind the computed static import
closure of the screen entry points (the screen, the carry replay and the command line,
with every module they import, lazy imports and package facades included), resolved
from the packet head by `tests/evaluation/test_carry_multisymbol_packet.py`: a change
to any file in that closure after the head fails the test until the packet is re-pinned
and re-reviewed.

## Data source and runbook

**Chosen source: Binance Vision archives with official `.CHECKSUM` verification, with no
database and no Docker.** `VisionArchiveSource` fetches the planned archives (1,392
kline and 660 funding archives, each with its checksum file) through the market-data
store's bounded, URL-gated fetcher, verifies every archive against its checksum,
parses it with the store's exact parser (which refuses the FS1 warm-up span and
sealed window by itself) and writes the verified bytes write-once into
`--archive-dir`. A cached archive is re-verified on every read. The checksum-verified
manifest is written to `data-manifest.json`. The Docker/PostgreSQL store is also
supported (`--source store`, the same adapter) but is not used for this run.

```bash
python -m tools.carry_multisymbol_screen plan --code-commit <40-hex>
python -m tools.carry_multisymbol_screen screen --start 2022-01-01 --end 2026-08-01 \
  --g4-authorization-url <issue comment URL> --output-dir <new dir> \
  --archive-dir <dir> --allow-network --coverage-only      # stage 1: coverage only
python -m tools.carry_multisymbol_screen screen ... (same, without --coverage-only)
```

The run refuses without `--g4-authorization-url`, refuses any window but the registered
one, needs a clean checkout and records the exact commit, the authorization URL (an
attestation, not verified here), every archive digest and every cell. Stage 1 loads
and checks coverage and liquidity and writes `inclusion.json` without replaying
anything; a second invocation on the same output directory resumes and never
recomputes a finished cell. Cell outputs are `cells/<arm>__<treatment>.json.gz` plus a
`.summary.json`, written for passing and failing cells alike, then `decision.json`.

**Inclusion versus failure.** A finite, nonnegative median hourly quote turnover
below the frozen 1,000,000 USDT threshold excludes a symbol with
`liquidity_below_threshold`; a documented coverage absence (HTTP 404 or an unexplained
hourly/funding gap) may also exclude it. Every exclusion is recorded in
`inclusion.json` with its reason and evidence. Missing or extra spot/USD-M liquidity
legs, nonfinite/negative values and ill-typed medians instead halt with
`invalid_lookback_liquidity` before the pair is loaded. Other acquisition or
integrity failures halt and never exclude: HTTP 5xx, 429, transport errors and
timeouts are retried
three times with backoff and then halt as an operational, resumable failure; a response
larger than the fetcher's byte bound (256 MiB) halts the same way at once, without a
retry, as `response_too_large` rather than as an unclassified error; an offline
cache miss halts at once, so an empty `--archive-dir` without `--allow-network` halts
instead of producing `INCONCLUSIVE_UNIVERSE`; a checksum mismatch, a missing checksum
file, a malformed archive and duplicated, unordered or malformed rows halt as integrity
failures. A halt writes `halt.json` (kind, resumable, symbol, evidence), writes no
decision and exits with status 3; a resumed run archives the earlier marker under
`halts/`.

**G4 owner authorization.** The run needs one Issue #810 comment by the owner with
this text, naming the code commit that will be run (`plan` prints it with the digest):

> G4 authorization (carry-multisymbol-dev-v1): I authorize exactly one development
> screen run of protocol digest `<digest>` at code commit `<commit>` on known-used
> Tier D Binance Vision data in [2022-01-01, 2026-08-01). The result is exploratory
> and carries no edge claim. The coverage-only stage and any resume on the same output
> directory are part of that one run. No orders, no keys, no FS1 warm-up, sealed or other
> unused data, and no live use are authorized.

One authorization is one run. The coverage-only stage and any resume on the same output
directory are part of that one run. A run on a different output directory is a
different run and is not covered.

The network fetch is the only external access. Synthetic replays took about 2 ms per
bar for nine pairs and 0.4 ms for one, so replays are roughly 25 to 45 minutes in
total for about 40,200 bars per cell; the archive download of about 4,100 requests
dominates and is likely to take one to two hours. Peak memory is one symbol's parsed
rows (a few hundred MB).

## Limitations

Close-price perpetual marks stand in for mark prices; the fee, spread and 1%
participation figures are assumptions, not recorded depth; a single 5bp adverse cost
may understate alt spreads in 2022 (the `cost2x` stress is the check); the exchange
lot step is approximated by the common 0.001 lot; wallet transfers are instant; the
halt calendar contains one incident, so any other spot gap excludes a symbol rather
than being explained away. Nothing here covers larger capital, venue account rules,
funding-interval changes beyond the assembler's coverage rule, or live trading.
