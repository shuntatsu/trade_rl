"""Deterministic prospective carry transitions using the canonical account."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from fractions import Fraction
from itertools import groupby
from typing import Any

import numpy as np

from trade_rl._validation import require_aware_datetime
from trade_rl.evaluation.carry import futures_collateral
from trade_rl.simulation import BookState
from trade_rl.simulation.depth import DepthOrderRules, execute_depth_order
from trade_rl.simulation.quantities import exact_quantity
from trade_rl.strategies.carry import CarryConfig, FundingCarryBot

SYMBOLS = ("BTCUSDT", "ETHUSDT")
VENUES = ("spot", "perpetual")


def timestamp(value: str) -> datetime:
    result = datetime.fromisoformat(value)
    require_aware_datetime(result, field="paper timestamp")
    return result.astimezone(UTC)


@dataclass(frozen=True)
class PaperSettings:
    start_at: datetime
    close_at: datetime
    initial_capital: float = 10_000.0
    maximum_gap_seconds: float = 180.0
    spot_fee_bps: float = 10.0
    perpetual_fee_bps: float = 5.0
    adverse_bps: float = 5.0
    depth_fraction: float = 0.1

    def __post_init__(self) -> None:
        for name in ("start_at", "close_at"):
            require_aware_datetime(getattr(self, name), field=name)
        if self.close_at <= self.start_at:
            raise ValueError("paper close must follow start")
        for name in (
            "initial_capital",
            "maximum_gap_seconds",
            "spot_fee_bps",
            "perpetual_fee_bps",
            "adverse_bps",
            "depth_fraction",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
            ):
                raise ValueError(f"{name} must be finite numeric")
        if self.initial_capital <= 0 or not 0 < self.maximum_gap_seconds <= 180:
            raise ValueError("invalid paper capital or maximum gap")
        if (
            any(
                not 0 <= fee < 10_000
                for fee in (self.spot_fee_bps, self.perpetual_fee_bps, self.adverse_bps)
            )
            or not 0 < self.depth_fraction <= 1
        ):
            raise ValueError("invalid paper cost or depth assumptions")

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result.update(
            start_at=self.start_at.isoformat(), close_at=self.close_at.isoformat()
        )
        return result


class PaperAccount:
    """In-memory replay state; callers must commit before adopting a new copy."""

    def __init__(self, settings: PaperSettings) -> None:
        self.settings = settings
        self.book = BookState.zero(4, settings.initial_capital)
        self.bot = FundingCarryBot(CarryConfig())
        self.last_at = settings.start_at
        self.observed_at: datetime | None = None
        self.pending: dict[str, Any] | None = None
        self.quality_failures: list[str] = []
        self.settlements: dict[tuple[str, int], tuple[str, str]] = {}
        self.expected_funding: set[tuple[str, int]] = set()
        self.holdings: list[tuple[datetime, tuple[Fraction, ...]]] = []
        self.observations = 0
        self.funding_watermark: datetime | None = None

    def status(self) -> dict[str, Any]:
        book = self.book
        return dict(
            account=dict(
                cash=book.cash,
                equity=book.portfolio_value,
                quantities=book.quantities.tolist(),
                exact_quantities=[str(q) for q in book.exact_quantities],
                mark_prices=book.mark_prices.tolist(),
                peak_value=book.peak_value,
                maximum_drawdown=book.max_drawdown,
                total_cost=book.total_cost,
                funding_pnl=book.funding_pnl,
                fill_count=book.fill_count,
                turnover_total=book.turnover_total,
                insolvent=book.insolvent,
                collateral=futures_collateral(
                    book.cash, book.quantities, book.mark_prices
                ),
            ),
            stop_reason=self.bot.stop_reason,
            quality_failures=list(self.quality_failures),
            pending=self.pending is not None,
            terminal_flat=not any(book.exact_quantities) and self.pending is None,
            observed_at=None
            if self.observed_at is None
            else self.observed_at.isoformat(),
            last_command_at=self.last_at.isoformat(),
            observations=self.observations,
            funding_watermark=None
            if self.funding_watermark is None
            else self.funding_watermark.isoformat(),
            production_eligible=False,
        )

    def _fail(self, reason: str, detail: str | None = None) -> None:
        self.bot.stop(reason)
        message = detail or reason
        if message not in self.quality_failures:
            self.quality_failures.append(message)

    def _risk(self) -> None:
        book = self.book
        if book.insolvent or book.portfolio_value <= 0:
            self._fail("insolvent")
        if book.max_drawdown >= self.bot.config.maximum_drawdown:
            self._fail("maximum_drawdown")
        collateral = futures_collateral(book.cash, book.quantities, book.mark_prices)
        requirement = float(np.abs(book.position_values[1::2]).sum())
        if collateral < requirement:
            self._fail("collateral_guard")
        if collateral < requirement * 0.5:
            self._fail("maintenance_breach")

    def _clock(self, at: datetime) -> None:
        require_aware_datetime(at, field="paper command")
        if at <= self.last_at:
            raise ValueError("paper command must follow the previous command")

    def _funding(self, snapshot: dict[str, Any], at: datetime) -> list[dict[str, Any]]:
        payments: list[dict[str, Any]] = []
        now_ms = int(at.timestamp() * 1000)
        for pair, symbol in enumerate(SYMBOLS):
            market = snapshot["market"][symbol]
            for row in market["settled_funding"]:
                milliseconds = row["fundingTime"]
                key = (symbol, milliseconds)
                values = (
                    str(Fraction(str(row["fundingRate"]))),
                    str(Fraction(str(row["markPrice"]))),
                )
                settled_at = datetime.fromtimestamp(milliseconds / 1000, UTC)
                if settled_at > at:
                    raise ValueError("cannot pay future funding")
                self.expected_funding.discard(key)
                previous = self.settlements.get(key)
                if previous is not None:
                    if previous != values:
                        self._fail(
                            "funding_revision",
                            f"funding_revision:{symbol}:{milliseconds}",
                        )
                    continue
                self.settlements[key] = values
                quantity = Fraction(0)
                for filled_at, quantities in reversed(self.holdings):
                    if filled_at < settled_at:
                        quantity = quantities[2 * pair + 1]
                        break
                amount = float(-quantity * Fraction(values[0]) * Fraction(values[1]))
                if (
                    amount
                    and self.funding_watermark is not None
                    and settled_at <= self.funding_watermark
                ):
                    self._fail(
                        "funding_out_of_order",
                        f"funding_out_of_order:{symbol}:{milliseconds}",
                    )
                if (
                    settled_at >= self.settings.start_at
                    and (at - settled_at).total_seconds()
                    > self.settings.maximum_gap_seconds
                ):
                    self._fail("late_funding", f"late_funding:{symbol}:{milliseconds}")
                payments.append(
                    dict(
                        symbol=symbol,
                        funding_at=settled_at.isoformat(),
                        known_at=at.isoformat(),
                        quantity=str(quantity),
                        rate=values[0],
                        mark=values[1],
                        amount=amount,
                    )
                )
            next_time = market["mark_quote"]["nextFundingTime"]
            # A schedule may change before it is due; retain already-due promises.
            self.expected_funding = {
                key
                for key in self.expected_funding
                if key[0] != symbol or key[1] <= now_ms
            }
            self.expected_funding.add((symbol, next_time))
        for symbol, milliseconds in sorted(self.expected_funding):
            if now_ms - milliseconds > self.settings.maximum_gap_seconds * 1000:
                self._fail(
                    "missing_funding", f"missing_funding:{symbol}:{milliseconds}"
                )
        payments.sort(key=lambda row: (row["funding_at"], row["symbol"]))
        for _, simultaneous in groupby(payments, key=lambda row: row["funding_at"]):
            amount = math.fsum(payment["amount"] for payment in simultaneous)
            if amount:
                self.book.mark_to_market(
                    mark_prices=self.book.mark_prices,
                    funding_amount=amount,
                    period_start_value=self.book.portfolio_value
                    if self.book.portfolio_value > 0
                    else self.settings.initial_capital,
                )
            self._risk()
        nonzero_times = [
            timestamp(row["funding_at"]) for row in payments if row["amount"]
        ]
        if nonzero_times:
            self.funding_watermark = max(
                nonzero_times
                + ([] if self.funding_watermark is None else [self.funding_watermark])
            )
        return payments

    def _observe(self, snapshot: dict[str, Any], at: datetime) -> list[dict[str, Any]]:
        self._clock(at)
        if timestamp(snapshot["started_at"]) <= self.last_at:
            raise ValueError("new market capture must start after the previous command")
        if (at - self.last_at).total_seconds() > self.settings.maximum_gap_seconds:
            self._fail("data_gap", "observation_gap")
        marks = []
        for symbol in SYMBOLS:
            market = snapshot["market"][symbol]
            depth = market["spot_depth"]
            marks.extend(
                [
                    (float(depth["bids"][0][0]) + float(depth["asks"][0][0])) / 2,
                    float(market["mark_quote"]["markPrice"]),
                ]
            )
        self.book.revalue(np.array(marks))
        self.book.refresh_drawdown()
        self._risk()  # Never let a newly published payment rescue an earlier breach.
        if at >= self.settings.close_at:
            self.bot.stop("terminal_close")
        payments = self._funding(snapshot, at)
        self.last_at = at
        self.observed_at = at
        self.observations += 1
        return payments

    def decide(
        self,
        snapshot: dict[str, Any],
        rules: dict[str, Any],
        rule_ref: dict[str, Any],
        at: datetime,
    ) -> dict[str, Any]:
        if self.pending is not None:
            raise ValueError("pending decision must execute or be cancelled by a gap")
        payments = self._observe(snapshot, at)
        if self.book.insolvent:
            target = np.zeros(4)
        else:
            target = self.bot.decide(
                timestamp=at,
                prices=self.book.mark_prices,
                quantities=self.book.quantities,
                equity=self.book.portfolio_value,
                drawdown=self.book.max_drawdown,
            )
        orders = []
        for i, quantity in enumerate(self.book.exact_quantities):
            rule = rules["rules"][VENUES[i % 2]][SYMBOLS[i // 2]]
            step = exact_quantity(rule["lot_size"])
            if exact_quantity(self.bot.config.common_lot) % step:
                raise ValueError("venue lot is incompatible with the fixed carry lot")
            lots = (exact_quantity(float(target[i])) - quantity) / step
            if lots.denominator != 1:
                raise ValueError("residual cannot be expressed as exact venue lots")
            if lots:
                orders.append(dict(symbol_index=i, lot_count=int(lots)))
        if orders and not self.book.insolvent:
            self.pending = dict(at=at.isoformat(), orders=orders, rules=rule_ref)
        return dict(**self.status(), orders=orders, funding=payments, fills=[])

    def execute(
        self, snapshot: dict[str, Any], rules: dict[str, Any], at: datetime
    ) -> dict[str, Any]:
        if self.pending is None:
            raise ValueError("no saved pending decision")
        pending = self.pending
        decision_at = timestamp(pending["at"])
        if (at - decision_at).total_seconds() > 10:
            raise ValueError("paper execution decision is stale")
        payments = self._observe(snapshot, at)
        increasing = any(
            abs(
                self.book.exact_quantities[row["symbol_index"]]
                + row["lot_count"]
                * exact_quantity(
                    rules["rules"][VENUES[row["symbol_index"] % 2]][
                        SYMBOLS[row["symbol_index"] // 2]
                    ]["lot_size"]
                )
            )
            > abs(self.book.exact_quantities[row["symbol_index"]])
            for row in pending["orders"]
        )
        desired = self.book.quantities.copy()
        for order in pending["orders"]:
            i = order["symbol_index"]
            desired[i] += (
                order["lot_count"]
                * rules["rules"][VENUES[i % 2]][SYMBOLS[i // 2]]["lot_size"]
            )
        if increasing and self.book.portfolio_value < float(
            np.abs(desired * self.book.mark_prices).sum()
        ):
            self._fail("projected_collateral_guard")
        fills = []
        cancelled = bool(self.book.insolvent or (self.bot.stop_reason and increasing))
        if not cancelled:
            receipts = {
                row["label"]: timestamp(row["received_at"])
                for row in snapshot["responses"]
            }
            for order in pending["orders"]:
                i = order["symbol_index"]
                symbol, venue = SYMBOLS[i // 2], VENUES[i % 2]
                depth = snapshot["market"][symbol][f"{venue}_depth"]
                rule = rules["rules"][venue][symbol]
                outcome = execute_depth_order(
                    book=self.book,
                    symbol_index=i,
                    lot_count=order["lot_count"],
                    rules=DepthOrderRules(
                        **{
                            key: rule[key]
                            for key in DepthOrderRules.__dataclass_fields__
                        }
                    ),
                    bids=tuple((float(p), float(q)) for p, q in depth["bids"]),
                    asks=tuple((float(p), float(q)) for p, q in depth["asks"]),
                    decision_at=decision_at,
                    quote_at=receipts[
                        f"{symbol}_{'spot' if venue == 'spot' else 'perp'}_depth"
                    ],
                    execution_at=at,
                    fee_bps=self.settings.spot_fee_bps
                    if i % 2 == 0
                    else self.settings.perpetual_fee_bps,
                    adverse_bps=self.settings.adverse_bps,
                    depth_fraction=self.settings.depth_fraction,
                )
                fills.append(dict(**order, **asdict(outcome)))
                self._risk()
                if self.book.insolvent or (increasing and self.bot.stop_reason):
                    break
            self.holdings.append((at, self.book.exact_quantities))
            if any(
                self.book.exact_quantities[i] + self.book.exact_quantities[i + 1]
                for i in (0, 2)
            ):
                self._fail("unmatched_hedge")
        self.pending = None
        return dict(**self.status(), funding=payments, fills=fills, cancelled=cancelled)

    def gap(self, *, at: datetime, reason: str) -> dict[str, Any]:
        self._clock(at)
        if not isinstance(reason, str) or not reason.strip() or len(reason) > 1000:
            raise ValueError("paper gap requires a bounded nonempty reason")
        self._fail("data_gap", reason)
        self.pending = None
        self.last_at = at
        return dict(**self.status(), funding=[], fills=[])
