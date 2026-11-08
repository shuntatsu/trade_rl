"""Pre-registered collection-failure, retry and skipped-slot policy (carry attempt 3).

The numbers are protocol parameters fixed by the attempt-3 research-assurance packets
(Issue #810 comment 6715501420) and must not change after a seal:

* Class D (HTTP 401/403/418/429/451 or any denial) and Class I (integrity, unknown)
  failures are terminal; Class T (timeout, connection or DNS error, HTTP
  500/502/503/504, a freshness rejection of one capture) may be retried inside the
  same slot, before a capture is consumed, never after a decision is committed.
* A slot that cannot capture, or whose held rules are older than one hour, is skipped.
  A skip appends no command and counts toward three budgets; any breach ends the
  attempt.

This module is pure policy (no I/O, no clock). ``supervisor`` applies it and
``control`` stores the registry it reads.
"""

from __future__ import annotations

import socket
import urllib.error
from collections.abc import Iterable
from typing import Literal

from trade_rl.integrations.binance.forward import ForwardFreshnessError

FailureClass = Literal["D", "I", "T"]

RETRIES_PER_SLOT = 2
RETRY_DELAY_SECONDS = 2
RETRY_START_WINDOW_SECONDS = 30
MAX_CONSECUTIVE_SKIPPED_SLOTS = 1
MAX_SKIPPED_SLOTS_PER_HOUR = 2
MAX_SKIPPED_SLOTS_TOTAL = 129
# The frozen sixty-second cadence; the per-hour budget looks at sixty such slots.
CADENCE_SECONDS = 60
SKIP_WINDOW_SLOTS = 3600 // CADENCE_SECONDS
# Exchange rules are refreshed from this age and never used beyond the hold; the hold
# equals the forward-rules reader's own one-hour bound.
RULES_REFRESH_SECONDS = 3300
RULES_HOLD_SECONDS = 3600

DENIAL_STATUS_CODES = frozenset({401, 403, 418, 429, 451})
TRANSIENT_STATUS_CODES = frozenset({500, 502, 503, 504})
_TRANSIENT_TYPES = (
    TimeoutError,
    ConnectionError,
    socket.gaierror,
    urllib.error.URLError,
    ForwardFreshnessError,
)


class SkipBudgetExceeded(RuntimeError):
    """A skipped-slot budget was breached: the attempt is INVALID."""

    def __init__(self, budget: str, slot: int) -> None:
        super().__init__(f"skipped-slot budget '{budget}' breached at slot {slot}")
        self.budget = budget
        self.slot = slot


def _chain(error: BaseException) -> list[BaseException]:
    items: list[BaseException] = []
    seen: set[int] = set()
    current: BaseException | None = error
    while current is not None and id(current) not in seen and len(items) < 8:
        seen.add(id(current))
        items.append(current)
        current = current.__cause__
    return items


def classify_failure(error: BaseException) -> FailureClass:
    """Classify one acquisition failure; anything not named as transient is terminal.

    An HTTP status anywhere in the cause chain decides first (a denial is never
    softened by a transient type further out). Unknown exceptions, malformed or
    oversized responses, crossed books, clock reversal, journal and lock errors, and
    every non-``Exception`` such as a termination signal are Class I.
    """
    chain = _chain(error)
    for item in chain:
        if isinstance(item, urllib.error.HTTPError):
            if item.code in DENIAL_STATUS_CODES:
                return "D"
            if item.code in TRANSIENT_STATUS_CODES:
                return "T"
            return "I"
    if any(
        isinstance(item, Exception) and isinstance(item, _TRANSIENT_TYPES)
        for item in chain
    ):
        return "T"
    return "I"


def first_budget_breach(skipped_slots: Iterable[int]) -> tuple[int, str] | None:
    """The first skip, in slot order, that breaches a budget, as ``(slot, budget)``.

    Budgets: more than ``MAX_CONSECUTIVE_SKIPPED_SLOTS`` adjacent slots; more than
    ``MAX_SKIPPED_SLOTS_PER_HOUR`` inside any ``SKIP_WINDOW_SLOTS`` consecutive slots;
    more than ``MAX_SKIPPED_SLOTS_TOTAL`` overall.
    """
    ordered = sorted(set(skipped_slots))
    present = set(ordered)
    for count, slot in enumerate(ordered, start=1):
        run = 0
        while slot - run in present:
            run += 1
        if run > MAX_CONSECUTIVE_SKIPPED_SLOTS:
            return slot, "consecutive"
        window = sum(1 for other in ordered if slot - SKIP_WINDOW_SLOTS < other <= slot)
        if window > MAX_SKIPPED_SLOTS_PER_HOUR:
            return slot, "per_hour"
        if count > MAX_SKIPPED_SLOTS_TOTAL:
            return slot, "total"
    return None


__all__ = [
    "CADENCE_SECONDS",
    "DENIAL_STATUS_CODES",
    "MAX_CONSECUTIVE_SKIPPED_SLOTS",
    "MAX_SKIPPED_SLOTS_PER_HOUR",
    "MAX_SKIPPED_SLOTS_TOTAL",
    "RETRIES_PER_SLOT",
    "RETRY_DELAY_SECONDS",
    "RETRY_START_WINDOW_SECONDS",
    "RULES_HOLD_SECONDS",
    "RULES_REFRESH_SECONDS",
    "SKIP_WINDOW_SLOTS",
    "TRANSIENT_STATUS_CODES",
    "FailureClass",
    "SkipBudgetExceeded",
    "classify_failure",
    "first_budget_breach",
]
