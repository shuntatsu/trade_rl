"""Pre-registered multi-symbol funding-carry development protocol (Issue #810 Track B).

Every outcome-relevant constant is fixed here, before any funding, price or return
of any candidate symbol was read. The protocol extends the BTC/ETH development
carry screen without changing its entry/exit rule, cost model or risk stop: the
same monthly matched-quantity spot-long / USD-M-perpetual-short bot, the same
canonical ledger and the same fee, spread, participation and latency assumptions.
Only the symbol set, the window, the capital split and the family decision rule
are new. All data before 2026-08-01 is known-used Tier D data; nothing here is an
edge claim and nothing here authorizes an order.

The digest binds the constants below; a changed constant is a new protocol version
and a new attempt, never an in-place edit.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

from trade_rl.artifacts import content_digest

PROTOCOL_ID: Final = "carry-multisymbol-dev-v1"
PROTOCOL_SCHEMA: Final = "carry_multisymbol_protocol_v1"

#: Candidate USDT symbols; each needs both a spot and a USD-M perpetual market.
#: BCHUSDT has no principled non-data reason to be left out, so it is included.
CANDIDATES: Final = (
    "SOLUSDT",
    "BNBUSDT",
    "XRPUSDT",
    "ADAUSDT",
    "DOGEUSDT",
    "LINKUSDT",
    "DOTUSDT",
    "AVAXUSDT",
    "LTCUSDT",
    "BCHUSDT",
)
#: Descriptive control arms on the identical window; never part of the decision.
REFERENCE: Final = ("BTCUSDT", "ETHUSDT")

WINDOW_START: Final = datetime(2022, 1, 1, tzinfo=UTC)
#: Exclusive. The last admissible bar opens at 2026-07-31T23:00Z.
WINDOW_END: Final = datetime(2026, 8, 1, tzinfo=UTC)
#: Three whole calendar months before the window: the liquidity/listing lookback.
LOOKBACK_START: Final = datetime(2021, 10, 1, tzinfo=UTC)
SUB_PERIODS: Final = ("2022", "2023", "2024", "2025", "2026")

#: The only externally evidenced spot halt (Binance incident report, already used by
#: the BTC/ETH screens). Any other missing hour is an unexplained gap.
SPOT_HALTS: Final = (
    (
        datetime(2023, 3, 24, 11, 27, tzinfo=UTC),
        datetime(2023, 3, 24, 14, 0, tzinfo=UTC),
    ),
)

#: Median hourly quote turnover over the lookback, each leg (about 24 million USDT a
#: day). The largest equal-weight leg is about 420 USDT (six admitted pairs), which the
#: most severe stress (0.1% participation) can fill in one bar only from 420,000 USDT
#: of hourly volume; a part-filled leg on one side only would trip the matched-hedge
#: stop for a liquidity reason, not a carry reason. The threshold keeps a margin of
#: about 2.4x at the median hour.
MIN_MEDIAN_HOURLY_QUOTE_VOLUME: Final = 1_000_000.0
LOOKBACK_HOURS: Final = 92 * 24
MIN_ADMITTED: Final = 6

INITIAL_CAPITAL: Final = 10_000.0
REFERENCE_PAIR_CAPITAL: Final = 5_000.0
GROSS_BUDGET: Final = 0.5
COMMON_LOT: Final = 0.001
MAXIMUM_DRAWDOWN: Final = 0.10
SPOT_TAKER_FEE: Final = 0.001
PERP_TAKER_FEE: Final = 0.0005
ADVERSE_EXECUTION: Final = 0.0005
MAX_PARTICIPATION: Final = 0.01
MINIMUM_NOTIONAL: Final = 10.0

#: Sign-null per-sub-period probability that a net return is positive.
NULL_SUB_PERIOD_POSITIVE_PROBABILITY: Final = 0.5
FAMILY_ALPHA_DENOMINATOR: Final = 20  # alpha = 1/20

G4_AUTHORIZATION_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/issues/[0-9]+#issuecomment-[0-9]+$"
)


@dataclass(frozen=True, slots=True)
class Treatment:
    name: str
    cost_multiplier: float
    capacity_multiplier: float
    entry_delay_bars: int


TREATMENTS: Final = (
    Treatment("base", 1.0, 1.0, 0),
    Treatment("cost2x", 2.0, 1.0, 0),
    Treatment("delay1h", 1.0, 1.0, 1),
    Treatment("capacity0p1", 1.0, 0.1, 0),
)

PORTFOLIO_ARM: Final = "portfolio"
REFERENCE_SHARED_ARM: Final = "ref_shared"


def reference_arm(symbol: str) -> str:
    return f"ref_{symbol}"


def sub_period_count() -> int:
    return len(SUB_PERIODS)


def null_pass_probability() -> float:
    """P(one symbol clears the base every-sub-period rule) under the sign-null."""
    return NULL_SUB_PERIOD_POSITIVE_PROBABILITY ** sub_period_count()


def required_symbol_passes(admitted: int) -> int:
    """Smallest K with ``admitted * p0 / K <= alpha`` (Markov, any dependence).

    ``p0 = 2**-5`` and ``alpha = 1/20``, so ``K = ceil(20 * admitted / 32)`` in exact
    integer arithmetic. For 10 admitted symbols K is 7. The Markov step needs no
    independence between symbols, but ``p0 = 2**-5`` assumes independent sub-period
    signs inside one symbol. A persistent funding regime can make a symbol's five
    sub-periods positive together with probability far above ``2**-5`` (up to one
    half under a common factor), so the 1/20 figure can be invalid: the family rule
    is a heuristic threshold, not a controlled error rate.
    """
    if isinstance(admitted, bool) or not isinstance(admitted, int) or admitted < 1:
        raise ValueError("admitted symbol count must be a positive integer")
    denominator = 2 ** sub_period_count()
    return -(-FAMILY_ALPHA_DENOMINATOR * admitted // denominator)


def protocol_document() -> dict[str, object]:
    """The canonical, hash-bound protocol (constants only; no data, no result)."""
    return {
        "schema": PROTOCOL_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "extends": "funding_carry_development_v1 (BTC/ETH, halt-aware, twelve screens)",
        "data_scope": "known-used Tier D data before 2026-08-01; exploratory only",
        "candidates": list(CANDIDATES),
        "candidate_provenance": (
            "named in the planner's brief as large-cap USDT pairs with both a spot "
            "and a USD-M perpetual market; BCHUSDT added because no principled "
            "non-data reason to omit it exists; SUI, APT, ARB and OP of the 2026-07-13 "
            "legacy universe are omitted only because they were listed after the "
            "lookback starts (listing age)"
        ),
        "prior_exposure": (
            "prices of eight candidates (and BTC/ETH) were read by the 2026-07-13 "
            "legacy signal run, which also used funding_bps as a feature; prices, "
            "funding events and funding P&L of BNB, XRP and ADA (with BTC/ETH) were "
            "read in the Issue 810 P2 development study (2021-01-01 to 2026-08-01); "
            "no delta-neutral spot/perpetual carry replay of any candidate was ever "
            "computed"
        ),
        "reference_symbols": list(REFERENCE),
        "window": {
            "start": WINDOW_START.isoformat(),
            "end_exclusive": WINDOW_END.isoformat(),
            "lookback_start": LOOKBACK_START.isoformat(),
            "sub_periods": list(SUB_PERIODS),
        },
        "inclusion": {
            "rule": (
                "both legs complete hourly over the 92-day lookback (listing age "
                "and minimum history), median hourly quote turnover of each leg at "
                "least the threshold over that lookback, complete window coverage "
                "and funding coverage; decided from timestamps and quote volume "
                "only, never from a price change or return"
            ),
            "min_median_hourly_quote_volume": MIN_MEDIAN_HOURLY_QUOTE_VOLUME,
            "lookback_hours": LOOKBACK_HOURS,
            "min_admitted": MIN_ADMITTED,
            "spot_halts": [
                [start.isoformat(), end.isoformat()] for start, end in SPOT_HALTS
            ],
        },
        "account": {
            "portfolio_capital": INITIAL_CAPITAL,
            "symbol_arm_capital": "portfolio_capital / admitted_symbols",
            "reference_shared_capital": INITIAL_CAPITAL,
            "reference_pair_capital": REFERENCE_PAIR_CAPITAL,
            "gross_budget": GROSS_BUDGET,
            "gross_budget_note": (
                "unchanged from BTC/ETH; capital split equally across admitted pairs; "
                "not tuned, because net return scales about linearly with gross"
            ),
            "common_lot": COMMON_LOT,
            "minimum_notional": MINIMUM_NOTIONAL,
        },
        "strategy": {
            "entry_exit": (
                "FundingCarryBot unchanged: monthly equal-notional matched-quantity "
                "spot-long/perpetual-short targets from the first usable bar, actual "
                "exit on the final bar, no funding forecast"
            ),
        },
        "costs": {
            "spot_taker_fee": SPOT_TAKER_FEE,
            "perp_taker_fee": PERP_TAKER_FEE,
            "adverse_execution_per_leg": ADVERSE_EXECUTION,
            "max_participation_of_preceding_hour_quote_volume": MAX_PARTICIPATION,
        },
        "treatments": [
            {
                "name": item.name,
                "cost_multiplier": item.cost_multiplier,
                "capacity_multiplier": item.capacity_multiplier,
                "entry_delay_bars": item.entry_delay_bars,
            }
            for item in TREATMENTS
        ],
        "hard_risk": {
            "maximum_drawdown_strictly_below": MAXIMUM_DRAWDOWN,
            "stops": (
                "any stop, unmatched hedge, margin breach, canonical termination, "
                "incomplete replay or non-flat terminal state fails the cell"
            ),
        },
        "cell_pass": {
            "base": (
                "complete, flat, valid, no stop, drawdown below cap, full-period net "
                "return > 0 and net return > 0 in every sub-period"
            ),
            "stress": (
                "the same without the sub-period condition: full-period net "
                "return > 0 for cost2x, delay1h and capacity0p1"
            ),
            "arm": "base and all three stress cells pass",
        },
        "family_decision": {
            "decision_arms": "portfolio plus one independent arm per admitted symbol",
            "generalizes": (
                "portfolio arm passes AND at least K admitted symbol arms pass AND "
                "admitted >= min_admitted"
            ),
            "k_rule": "K = ceil(20 * admitted / 32)",
            "k_derivation": (
                "Markov step: P(passes >= K) <= admitted * p0 / K <= 1/20 holds for "
                "any cross-symbol dependence IF p0 = 2**-5 per symbol, which assumes "
                "independent sub-period signs inside a symbol"
            ),
            "error_rate_controlled": False,
            "k_status": (
                "heuristic threshold: serial correlation of funding regimes can make "
                "p0 far larger than 2**-5, so the 1/20 bound can be invalid and the "
                "family decision is not a controlled error rate"
            ),
            "outcomes": [
                "CARRY_GENERALIZES_DEVELOPMENT",
                "CARRY_DOES_NOT_GENERALIZE",
                "INCONCLUSIVE_UNIVERSE",
            ],
            "no_post_hoc_subset": (
                "failing or excluded symbols are never removed after the cells exist"
            ),
            "reference_arms": "BTC/ETH on the same window are descriptive only",
        },
        "stop_rule": (
            "no cell, symbol, window, cost, gross or rule is changed after a cell "
            "exists; a failed family stops this line; a pass authorizes only a "
            "separately pre-registered forward paper protocol"
        ),
    }


def protocol_digest() -> str:
    return content_digest(protocol_document())


def validate_g4_authorization_url(url: object) -> str:
    if not isinstance(url, str) or G4_AUTHORIZATION_PATTERN.fullmatch(url) is None:
        raise ValueError(
            "a G4 authorization is an issue comment URL "
            "(https://github.com/shuntatsu/trade_rl/issues/<n>#issuecomment-<id>)"
        )
    return url


def validate_window(start: datetime, end: datetime) -> None:
    """Only the registered window is admissible; later data is refused outright."""
    for value in (start, end):
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("window bounds must be timezone-aware")
    if end.astimezone(UTC) > WINDOW_END:
        raise ValueError(
            f"window end {end.isoformat()} is after the exclusive end "
            f"{WINDOW_END.isoformat()} (the last admissible bar opens "
            "2026-07-31T23:00Z); data from 2026-08-01 is FS1 warm-up or "
            "sealed data and is refused unconditionally"
        )
    if start.astimezone(UTC) != WINDOW_START or end.astimezone(UTC) != WINDOW_END:
        raise ValueError(
            "the window is part of the registered protocol: "
            f"[{WINDOW_START.isoformat()}, {WINDOW_END.isoformat()})"
        )


def lease_text(*, code_commit: str) -> str:
    """The exact comment the owner posts to authorize one development run."""
    return (
        f"G4 authorization ({PROTOCOL_ID}): I authorize exactly one development "
        f"screen run of protocol digest {protocol_digest()} at code commit "
        f"{code_commit} on known-used Tier D Binance Vision data in "
        f"[{WINDOW_START.isoformat()}, {WINDOW_END.isoformat()}). The result is "
        "exploratory and carries no edge claim. No orders, no keys, no FS1 warm-up, "
        "sealed or other unused data, and no live use are authorized."
    )


__all__ = [
    "CANDIDATES",
    "REFERENCE",
    "SPOT_HALTS",
    "SUB_PERIODS",
    "TREATMENTS",
    "WINDOW_END",
    "WINDOW_START",
    "Treatment",
    "lease_text",
    "null_pass_probability",
    "protocol_digest",
    "protocol_document",
    "required_symbol_passes",
    "validate_g4_authorization_url",
    "validate_window",
]
