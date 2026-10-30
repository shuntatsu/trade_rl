"""Sealed-window embargo for the market-data store (Issue #810).

The FS1 forward study (``trade_rl.evaluation.forward_study``) owns an unused window
that starts at ``2026-11-01T00:00Z`` and is scored exactly once after 2027-05-02. No
one may see data from that window before then, so the store refuses to ingest it.

``SEALED_WINDOW_START`` is pinned here, not imported: integrations must not depend on
evaluation. ``tests/integrations/market_store/test_embargo.py`` asserts equality with
``Fs1Registry.scoring_start`` and with the ``market_embargo`` row of the SQL
migration, so the three can never drift apart silently.

The embargo is enforced at three independent layers:

1. planning: no archive whose period reaches the boundary is ever planned or fetched;
2. row validation: a parsed row at or after the boundary raises before any write;
3. the database: ``market_embargo`` triggers reject such rows and sources even if the
   Python layers are bypassed.

The only override is :meth:`EmbargoPolicy.released`, which requires an explicit
authorization comment URL on Issue #810. The user must authorize it; FS1 scoring
itself still uses its own sealed acquisition tool, never this store.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

SEALED_WINDOW_START: Final = datetime(2026, 11, 1, tzinfo=UTC)
SEALED_WINDOW_START_MS: Final = int(SEALED_WINDOW_START.timestamp() * 1_000)

RELEASE_AUTHORIZATION_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-[0-9]+$"
)


class SealedWindowEmbargoError(ValueError):
    """Data at or after the FS1 sealed window start was requested or encountered."""


@dataclass(frozen=True, slots=True)
class EmbargoPolicy:
    """The embargo boundary and whether an authorized release is in force."""

    start: datetime = SEALED_WINDOW_START
    released: bool = False
    authorization_url: str | None = None

    def __post_init__(self) -> None:
        if self.start.tzinfo is None or self.start.utcoffset() is None:
            raise ValueError("embargo start must be timezone-aware")
        object.__setattr__(self, "start", self.start.astimezone(UTC))
        if self.released != (self.authorization_url is not None):
            raise ValueError(
                "a released embargo requires an authorization URL and vice versa"
            )
        if self.authorization_url is not None and (
            RELEASE_AUTHORIZATION_URL_PATTERN.fullmatch(self.authorization_url) is None
        ):
            raise ValueError(
                "release authorization must be an Issue #810 comment URL "
                "(https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-<id>)"
            )

    @classmethod
    def enforced(cls, start: datetime = SEALED_WINDOW_START) -> EmbargoPolicy:
        return cls(start=start)

    @classmethod
    def release(
        cls, authorization_url: str, start: datetime = SEALED_WINDOW_START
    ) -> EmbargoPolicy:
        return cls(start=start, released=True, authorization_url=authorization_url)

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1_000)

    def check_ms(self, timestamp_ms: int, *, context: str) -> None:
        """Refuse a row timestamp (epoch milliseconds) inside the sealed window."""

        if not self.released and timestamp_ms >= self.start_ms:
            moment = datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).isoformat()
            raise SealedWindowEmbargoError(
                f"{context}: timestamp {moment} is inside the FS1 sealed window "
                f"(>= {self.start.isoformat()}); refusing"
            )

    def check_archive_end(self, period_end: datetime, *, url: str) -> None:
        """Refuse an archive whose exclusive period end passes the boundary."""

        if not self.released and period_end > self.start:
            raise SealedWindowEmbargoError(
                f"archive {url} covers a period ending {period_end.isoformat()}, "
                f"past the FS1 sealed window start {self.start.isoformat()}; "
                "it must not be downloaded"
            )

    def effective_end(self, requested_end: datetime) -> datetime:
        """The range end an enforced embargo allows (the request unchanged if released)."""

        if self.released:
            return requested_end
        return min(requested_end, self.start)


__all__ = [
    "RELEASE_AUTHORIZATION_URL_PATTERN",
    "SEALED_WINDOW_START",
    "SEALED_WINDOW_START_MS",
    "EmbargoPolicy",
    "SealedWindowEmbargoError",
]
