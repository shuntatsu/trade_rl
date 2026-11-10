# Multi-symbol funding-carry development screen

Status: Active

Issue #810 Track B. The BTC/ETH spot-long / USD-M-perpetual-short carry passed all
twelve of its development screens, but the net return was small at gross 0.5 (see
[the carry capability record](../research/current-status.md)). This pre-registration
asks the one question that decides whether that line is worth continuing: does the
same fixed carry survive on more USD-M symbols, in more regimes, after the same
costs? It is registered, implemented and tested **before any funding, price or
return of any candidate symbol was read**. No real-data run is part of the change
that adds it.

## Claim boundary (G3)

Every bar before 2026-08-01 is known-used Tier D data. A result on it is exploratory
development evidence: it is never an edge claim, never unused-data validation and
never authorizes an order. The only confirmation path is a separately pre-registered
forward paper protocol (see [the prospective paper spec](funding-carry-paper.md)).
The FS1 warm-up span, the sealed window and any unused/final data are off limits;
the planner, the archive parser and the window validator all refuse them.

## Registered protocol `carry-multisymbol-dev-v1`

The constants live in `trade_rl/evaluation/carry_multisymbol/protocol.py`; the
canonical digest is
`7526710f22d15031edf1943ed1749d61515c8917986c827a51d7bc7d97cb2150`
(`python -m tools.carry_multisymbol_screen plan`). A changed constant is a new
protocol version and a new attempt; nothing is edited after a cell exists.

**Universe and inclusion.** Candidates, fixed in this order: SOL, BNB, XRP, ADA,
DOGE, LINK, DOT, AVAX, LTC against USDT, each needing both a spot and a USD-M
perpetual market. A candidate is admitted only if all of these hold, decided from
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
dropped later**. Fewer than six admitted symbols is `INCONCLUSIVE_UNIVERSE`; the fix is
a new protocol version with an externally evidenced halt calendar, decided without
reading returns. BTCUSDT and ETHUSDT are loaded as descriptive reference arms on the
identical window; they never enter the decision.

**Window and sub-periods.** Development window `[2022-01-01, 2026-08-01)`: the last
admissible bar opens 2026-07-31T23:00Z. It adds the 2022 bear/negative-funding
regime to the 2023-2024 regime of the BTC/ETH screens, because the counter-hypothesis
is that the premium was a 2023-2024 regime effect. Sub-periods are the calendar
years 2022, 2023, 2024, 2025 and 2026 (seven months). The window is part of the
protocol: the runner accepts exactly this `--start/--end` and refuses a window ending
after 2026-07-31 with a distinct embargo message.

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
`capacity0p1`), up to 40 decision cells, plus 12 descriptive reference cells. Cell
pass: complete, flat, valid, no stop, ledger drawdown below 10%, full-period net
return above zero; the `base` cell additionally needs net return above zero in
every sub-period. An arm passes only if all four of its cells pass.

**Family decision and multiple testing.** `CARRY_GENERALIZES_DEVELOPMENT` requires the
portfolio arm to pass **and** at least K admitted symbol arms to pass, with
K = ceil(20 x admitted / 32) (admitted 6, 7, 8, 9 give K = 4, 5, 5, 6). The family is
fixed: n symbol hypotheses and one portfolio gate, and no failing or excluded symbol
is removed after the cells exist. K is derived, not tuned: under the sign-null that
each sub-period net return is positive with probability at most one half, a symbol
clears the five-sub-period base rule with probability at most p0 = 2^-5, and
Markov's inequality gives P(passes >= K) <= n x p0 / K <= 1/20 for any dependence
between symbols. The bound assumes independent sub-period signs inside one symbol;
serial correlation of funding regimes weakens that, so it is a stated limitation, not
a corrected claim. An independent-binomial tail is reported only as a diagnostic.
Otherwise the outcome is `CARRY_DOES_NOT_GENERALIZE`.

**Stop rule.** No cell, symbol, window, cost, gross or rule is changed after a cell
exists. A failed family stops this line; a pass authorizes only a separately
pre-registered forward paper protocol, never live use. A software or data fault that
invalidates a run is preserved and answered by a new protocol version, not repaired
in place.

## Research assurance (G0-G2)

The author packet
[`report/issue810-carry-multisymbol-research-assurance-20261110.json`](../../report/issue810-carry-multisymbol-research-assurance-20261110.json)
states the counter-hypothesis (a 2023-2024 regime effect that fees and basis risk
erase on alts), the cheapest falsifier (the 2x-cost stress and every-sub-period rule
on the 2022 regime), the mechanism chain (leveraged long demand pays a funding
premium that the delta-neutral pair collects) and the independent oracle. Its digest is
`6868a5f482cf335f4126a3af8484e3e565173bc8be1d0443af6a446204d51212` and its status
is **UNREVIEWED** and `economic_execution_authorized=false`: no independent review has
been posted. The oracle is a hand-computed three-pair ledger in exact rational
arithmetic (`tests/evaluation/test_carry_multisymbol_replay.py`) that does not call the
production sizing, cost or funding code.

## Data source and runbook

**Chosen source: Binance Vision archives with official `.CHECKSUM` verification, with no
database and no Docker.** `VisionArchiveSource` fetches the planned archives (1,276
kline and 605 funding archives, each with its checksum file) through the market-data
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

**G4 owner authorization.** The run needs one Issue #810 comment by the owner with
this text, naming the code commit that will be run (`plan` prints it with the digest):

> G4 authorization (carry-multisymbol-dev-v1): I authorize exactly one development
> screen run of protocol digest `<digest>` at code commit `<commit>` on known-used
> Tier D Binance Vision data in [2022-01-01, 2026-08-01). The result is exploratory
> and carries no edge claim. No orders, no keys, no FS1 warm-up, sealed or other
> unused data, and no live use are authorized.

The network fetch is the only external access. Synthetic replays took about 2 ms per
bar for nine pairs and 0.4 ms for one, so replays are roughly 25 to 40 minutes in
total for about 40,200 bars per cell; the archive download of about 3,800 requests
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
