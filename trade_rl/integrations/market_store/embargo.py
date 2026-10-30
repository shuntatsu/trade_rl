"""FS1 data-protection boundaries for the market-data store (Issue #810).

The FS1 forward study (``trade_rl.evaluation.forward_study``) controls two spans:

* the **warm-up span** ``[2026-08-01T00:00Z, 2026-11-01T00:00Z)``: per
  ``docs/architecture/research-assurance.md`` (FS1 acquisition procedure) it may be
  fetched only with a separate Issue #810 data-fetch authorization. The store refuses
  it by default and admits it only for a run that carries ``--allow-fs1-warmup-span``
  together with an Issue #810 authorization comment URL (recorded on the run);
* the **sealed window** ``[2026-11-01T00:00Z, ...)``: scored once after 2027-05-02 and
  never visible before. The store refuses it **unconditionally**. There is no flag,
  parameter or SQL path that lifts this; lifting it after FS1 is scored needs a
  future reviewed migration and pull request.

Both constants are pinned here (integrations must not depend on evaluation) and
``tests/integrations/market_store/test_embargo.py`` asserts equality with
``Fs1Registry.warmup_only_start`` / ``scoring_start`` and with the SQL functions of the
migration. Even when the warm-up span is admitted, FS1 scoring still re-acquires and
re-verifies its own data through its sealed acquisition tool
(the FS1 acquisition tool under ``trade_rl.evaluation``); store rows never feed FS1 scoring.

The boundaries are enforced at three independent layers: planning (no archive whose
period reaches a closed span is planned or fetched), row validation (before any
write) and PostgreSQL triggers (even if the Python layers are bypassed).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Final

FS1_WARMUP_START: Final = datetime(2026, 8, 1, tzinfo=UTC)
SEALED_WINDOW_START: Final = datetime(2026, 11, 1, tzinfo=UTC)
FS1_WARMUP_START_MS: Final = int(FS1_WARMUP_START.timestamp() * 1_000)
SEALED_WINDOW_START_MS: Final = int(SEALED_WINDOW_START.timestamp() * 1_000)

WARMUP_AUTHORIZATION_URL_PATTERN: Final = re.compile(
    r"^https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-[0-9]+$"
)


class SealedWindowEmbargoError(ValueError):
    """Data at or after the FS1 sealed window start was requested or encountered."""


class Fs1WarmupSpanError(SealedWindowEmbargoError):
    """FS1 warm-up data was requested without the explicit authorization."""


@dataclass(frozen=True, slots=True)
class EmbargoPolicy:
    """Which FS1 spans a run may touch: everything before :attr:`start`.

    The two boundaries are NOT parameters: they are the module constants
    ``FS1_WARMUP_START`` and ``SEALED_WINDOW_START`` (pinned to the FS1 registry), so no
    public construction can move them. The only input is the optional warm-up
    authorization URL. Tests that need other boundaries monkeypatch the module
    constants; no production code path does.
    """

    warmup_authorization_url: str | None = None

    def __post_init__(self) -> None:
        if self.warmup_authorization_url is not None and (
            WARMUP_AUTHORIZATION_URL_PATTERN.fullmatch(self.warmup_authorization_url)
            is None
        ):
            raise ValueError(
                "FS1 warm-up authorization must be an Issue #810 comment URL "
                "(https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-<id>)"
            )

    @classmethod
    def enforced(cls) -> EmbargoPolicy:
        """Default policy: the warm-up span and the sealed window are both refused."""

        return cls()

    @classmethod
    def with_warmup(cls, authorization_url: str) -> EmbargoPolicy:
        """Admit the warm-up span for one run; the sealed window stays closed."""

        return cls(warmup_authorization_url=authorization_url)

    @property
    def warmup_start(self) -> datetime:
        return FS1_WARMUP_START

    @property
    def sealed_start(self) -> datetime:
        return SEALED_WINDOW_START

    @property
    def warmup_allowed(self) -> bool:
        return self.warmup_authorization_url is not None

    @property
    def start(self) -> datetime:
        """First instant this run must not touch."""

        return self.sealed_start if self.warmup_allowed else self.warmup_start

    @property
    def start_ms(self) -> int:
        return int(self.start.timestamp() * 1_000)

    def _refusal(self, moment: str, *, sealed: bool, context: str) -> ValueError:
        if sealed:
            return SealedWindowEmbargoError(
                f"{context}: {moment} is inside the FS1 sealed window "
                f"(>= {self.sealed_start.isoformat()}); refusing unconditionally"
            )
        return Fs1WarmupSpanError(
            f"{context}: {moment} is inside the FS1 warm-up span "
            f"[{self.warmup_start.isoformat()}, {self.sealed_start.isoformat()}); "
            "it needs --allow-fs1-warmup-span with an Issue #810 authorization URL"
        )

    def check_ms(self, timestamp_ms: int, *, context: str) -> None:
        """Refuse a row timestamp (epoch milliseconds) in a closed span."""

        if timestamp_ms >= int(self.sealed_start.timestamp() * 1_000):
            moment = datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).isoformat()
            raise self._refusal(moment, sealed=True, context=context)
        if timestamp_ms >= self.start_ms:
            moment = datetime.fromtimestamp(timestamp_ms / 1_000, tz=UTC).isoformat()
            raise self._refusal(moment, sealed=False, context=context)

    def check_archive_end(self, period_end: datetime, *, url: str) -> None:
        """Refuse an archive whose exclusive period end passes a closed span."""

        context = f"archive {url} (period ends {period_end.isoformat()})"
        if period_end > self.sealed_start:
            raise self._refusal("its period", sealed=True, context=context)
        if period_end > self.start:
            raise self._refusal("its period", sealed=False, context=context)

    def check_range_end(self, requested_end: datetime, *, what: str) -> None:
        """Refuse a requested exclusive range end past a closed span."""

        context = f"{what} end {requested_end.isoformat()}"
        if requested_end > self.sealed_start:
            raise self._refusal("the range", sealed=True, context=context)
        if requested_end > self.start:
            raise self._refusal("the range", sealed=False, context=context)


__all__ = [
    "FS1_WARMUP_START",
    "FS1_WARMUP_START_MS",
    "SEALED_WINDOW_START",
    "SEALED_WINDOW_START_MS",
    "WARMUP_AUTHORIZATION_URL_PATTERN",
    "EmbargoPolicy",
    "Fs1WarmupSpanError",
    "SealedWindowEmbargoError",
]
