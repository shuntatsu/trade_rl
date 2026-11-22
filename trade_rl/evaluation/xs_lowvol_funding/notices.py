"""Causal delisting-notice evidence of ``xs_lowvol_funding_v1`` (Amendment 5).

A perpetual's archive end (or the start of a relisting gap) is an *event*. The policy may close
a name early only from public knowledge: the latest published notice of that symbol released
at or before the event and at most ``NOTICE_LOOKBACK_DAYS`` before it. The target is 0 from the
first 4h decision bar whose **open** time is at or after the notice's release (strictly after
publication) through the event. An event without such a notice is refused before any replay.

The notice file is a recorded document pinned by ``protocol.DELISTING_NOTICES_SHA256`` and its
bytes are verified before they are parsed. Nothing here reads a network, a store or a result.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import numpy as np

from trade_rl.evaluation.trend_oos.data import StageDataError

from . import protocol

SCHEMA_VERSION: Final = 1
KINDS: Final[tuple[str, ...]] = ("delist", "migration", "swap", "rebrand")
_DAY_MS: Final = 86_400_000
_BAR_MS: Final = protocol.BAR_HOURS * 3_600_000
_FIELDS: Final = frozenset({"symbol", "release_utc", "article_code", "title", "kind"})
_OVERRIDE_FIELDS: Final = _FIELDS | {"evidence"}
_TIME_FORMAT: Final = "%Y-%m-%dT%H:%M:%S.%fZ"
_SYMBOL: Final = re.compile(r"[A-Z0-9]+USDT")


class NoticeError(StageDataError):
    """The notice file is malformed or does not match its pin."""


class MissingNoticeError(StageDataError):
    """A perpetual event of the stage has no causal public notice."""


def release_ms(text: str) -> int:
    """Integer milliseconds of an ISO ``release_utc`` string (``...SS.mmmZ``)."""
    try:
        moment = datetime.strptime(text, _TIME_FORMAT).replace(tzinfo=UTC)
    except (TypeError, ValueError) as error:
        raise NoticeError(f"malformed release_utc {text!r}") from error
    return int(moment.timestamp()) * 1000 + moment.microsecond // 1000


def release_utc(milliseconds: int) -> str:
    """The canonical ``release_utc`` string of integer milliseconds."""
    moment = datetime.fromtimestamp(milliseconds // 1000, tz=UTC)
    return moment.strftime("%Y-%m-%dT%H:%M:%S") + f".{milliseconds % 1000:03d}Z"


@dataclass(frozen=True, slots=True)
class Notice:
    """One public notice naming one USDⓈ-M perpetual."""

    symbol: str
    release_ms: int
    article_code: str
    title: str
    kind: str
    override: bool = False

    def to_dict(self) -> dict[str, object]:
        return {
            "symbol": self.symbol,
            "release_utc": release_utc(self.release_ms),
            "article_code": self.article_code,
            "title": self.title,
            "kind": self.kind,
        }


@dataclass(frozen=True, slots=True)
class NoticeBook:
    """The verified notices (entries and overrides) with the digest of the file they came from."""

    notices: tuple[Notice, ...] = ()
    sha256: str | None = None

    def causal(self, symbol: str, event_ms: int) -> Notice | None:
        """The latest notice of ``symbol`` with ``event - 90 days <= release <= event``."""
        floor = event_ms - protocol.NOTICE_LOOKBACK_DAYS * _DAY_MS
        found = [
            n
            for n in self.notices
            if n.symbol == symbol and floor <= n.release_ms <= event_ms
        ]
        return max(found, key=lambda n: (n.release_ms, n.article_code), default=None)


def _entry(raw: object, *, allowed: frozenset[str], override: bool) -> Notice:
    if not isinstance(raw, dict) or set(raw) != allowed:
        raise NoticeError(
            f"a notice entry must have exactly the keys {sorted(allowed)}"
        )
    for key in allowed:
        if not isinstance(raw[key], str) or not raw[key]:
            raise NoticeError(f"notice field {key!r} must be a non-empty string")
    symbol = raw["symbol"]
    if not _SYMBOL.fullmatch(symbol):
        raise NoticeError(f"notice symbol {symbol!r} is not an ASCII USDT perpetual")
    if raw["kind"] not in KINDS:
        raise NoticeError(f"unknown notice kind {raw['kind']!r}")
    return Notice(
        symbol=symbol,
        release_ms=release_ms(raw["release_utc"]),
        article_code=raw["article_code"],
        title=raw["title"],
        kind=raw["kind"],
        override=override,
    )


def parse_notices(raw: bytes) -> NoticeBook:
    """Parse the notice file bytes (no pin check; see :func:`load_notices`)."""
    try:
        document = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise NoticeError("the notice file is not valid JSON") from error
    if (
        not isinstance(document, dict)
        or document.get("schema_version") != SCHEMA_VERSION
        or not isinstance(document.get("entries"), list)
        or not isinstance(document.get("overrides"), list)
    ):
        raise NoticeError("the notice file has the wrong schema")
    entries = [
        _entry(item, allowed=_FIELDS, override=False) for item in document["entries"]
    ]
    overrides = [
        _entry(item, allowed=_OVERRIDE_FIELDS, override=True)
        for item in document["overrides"]
    ]
    keys = [(n.symbol, n.release_ms, n.article_code) for n in entries]
    if keys != sorted(keys) or len(set(keys)) != len(keys):
        raise NoticeError("notice entries must be unique and sorted")
    merged = tuple(sorted(entries + overrides, key=lambda n: (n.symbol, n.release_ms)))
    return NoticeBook(notices=merged, sha256=hashlib.sha256(raw).hexdigest())


def load_notices(
    path: Path | None = None, *, expected: str | None = None
) -> NoticeBook:
    """The pinned notice book; the SHA-256 is verified before the bytes are parsed."""
    pinned = protocol.DELISTING_NOTICES_SHA256 if expected is None else expected
    if pinned is None:
        raise NoticeError(
            "DELISTING_NOTICES_SHA256 is not pinned; real runs are refused"
        )
    target = protocol.delisting_notices_path() if path is None else path
    raw = Path(target).read_bytes()
    actual = hashlib.sha256(raw).hexdigest()
    if actual != pinned:
        raise NoticeError(
            f"delisting notice file sha256 {actual} differs from the pinned {pinned}"
        )
    return parse_notices(raw)


def last_bar_rows(active: np.ndarray) -> np.ndarray:
    """Rows ``L`` where a leg is available at ``L`` and not at ``L + 1`` (an event).

    A leg merely cut off by the end of the dataset is never flagged.
    """
    act = np.asarray(active, dtype=np.bool_)
    last = np.zeros_like(act)
    if act.shape[0] >= 2:
        last[:-1] = act[:-1] & ~act[1:]
    return last


def _iso_ms(milliseconds: int) -> str:
    return datetime.fromtimestamp(milliseconds / 1000, tz=UTC).isoformat()


def forced_schedule(
    active: np.ndarray,
    member: np.ndarray,
    close_ms: np.ndarray,
    symbols: Sequence[str],
    book: NoticeBook,
) -> tuple[np.ndarray, list[dict[str, object]], list[tuple[str, str]]]:
    """``(forced rows, evidence, events without a causal notice)``.

    An event is the last available 4h bar ``L`` of a leg (archive end or gap start); its time
    is the close of row ``L``. Only a leg that was a universe member at some row up to ``L``
    can hold a position at the event, so only those events need a notice. The leg's target is
    0 on the rows ``s..L`` where ``s`` is the first row whose bar *open* is at or after the
    notice release. A notice released after the last closable bar flags no row, and a held
    position then refuses at the last available bar.
    """
    act = np.asarray(active, dtype=np.bool_)
    mask = np.asarray(member, dtype=np.bool_)
    closes = np.asarray(close_ms, dtype=np.int64)
    forced = np.zeros_like(act)
    evidence: list[dict[str, object]] = []
    missing: list[tuple[str, str]] = []
    opens = closes - _BAR_MS
    last = last_bar_rows(act)
    for row, leg in zip(*np.nonzero(last), strict=True):
        row, leg = int(row), int(leg)
        if not mask[: row + 1, leg].any():
            continue
        event_ms = int(closes[row])
        symbol = str(symbols[leg])
        notice = book.causal(symbol, event_ms)
        if notice is None:
            missing.append((symbol, _iso_ms(event_ms)))
            continue
        start = int(np.searchsorted(opens, notice.release_ms, side="left"))
        if start <= row:
            forced[start : row + 1, leg] = True
        evidence.append(
            {
                "symbol": symbol,
                "event_close": _iso_ms(event_ms),
                "event_row": row,
                "notice_code": notice.article_code,
                "notice_release": release_utc(notice.release_ms),
                "first_forced_row": start,
                "override": notice.override,
            }
        )
    return forced, evidence, missing


def require_notices(missing: Sequence[tuple[str, str]]) -> None:
    """Refuse (naming the symbols) when any stage event has no causal notice."""
    if missing:
        names = ", ".join(f"{symbol}@{when}" for symbol, when in missing)
        raise MissingNoticeError(
            "no causal delisting notice (release within 90 days before the event) for: "
            f"{names}"
        )


__all__ = [
    "KINDS",
    "MissingNoticeError",
    "Notice",
    "NoticeBook",
    "NoticeError",
    "SCHEMA_VERSION",
    "forced_schedule",
    "last_bar_rows",
    "load_notices",
    "parse_notices",
    "release_ms",
    "release_utc",
    "require_notices",
]
