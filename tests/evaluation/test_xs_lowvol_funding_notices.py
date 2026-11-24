"""Synthetic tests of the causal delisting-notice evidence (Amendment 5).

No network and no market data: titles, article bodies and the CMS API are fakes, and every
stage is a deterministic fixture. The real notice file is only checked against its pin.
"""

from __future__ import annotations

import hashlib
import io
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation import xs_fixtures as fx
from tools import xs_lowvol_funding as driver
from trade_rl.evaluation.xs_lowvol_funding import data, notices, protocol

BAR_MS = protocol.BAR_HOURS * 3_600_000
DAY_MS = 86_400_000


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000)


def _entry(symbol: str, release: int, code: str = "c1") -> notices.Notice:
    return notices.Notice(
        symbol=symbol,
        release_ms=release,
        article_code=code,
        title="t",
        kind="delist",
    )


# ================================================================== matching window


def test_a_later_postponement_notice_never_delays_the_forced_window() -> None:
    """Regression: the earliest notice in the window starts it (AKRO, ANT, OMG cases)."""
    event = 1_800_000_000_000
    book = notices.NoticeBook(
        notices=(
            _entry("AUSDT", event - 91 * DAY_MS, "too-old"),
            _entry("AUSDT", event - 60 * DAY_MS, "first"),
            _entry("AUSDT", event - 10 * DAY_MS, "postponement"),
            _entry("AUSDT", event + 1, "after"),
            _entry("BUSDT", event - 5 * DAY_MS, "other"),
        )
    )
    found = book.causal("AUSDT", event)
    assert found is not None and found.article_code == "first"
    assert book.causal("CUSDT", event) is None
    # through the schedule: forced from the bar after the first notice, not the postponement
    active = np.ones((100, 1), dtype=np.bool_)
    active[61:, 0] = False
    close_ms = (np.arange(100, dtype=np.int64) + 1) * BAR_MS
    early, late = 20 * BAR_MS, 50 * BAR_MS
    both = notices.NoticeBook(
        notices=(_entry("AUSDT", early, "first"), _entry("AUSDT", late, "postponement"))
    )
    forced, evidence, _ = notices.forced_schedule(
        active, np.ones_like(active), close_ms, ("AUSDT",), both
    )
    assert list(np.flatnonzero(forced[:, 0])) == list(range(20, 61))
    assert evidence[0]["notice_code"] == "first"


def test_the_window_is_inclusive_at_90_days_and_at_the_event() -> None:
    event = 1_800_000_000_000
    inside = notices.NoticeBook(notices=(_entry("AUSDT", event - 90 * DAY_MS),))
    on_event = notices.NoticeBook(notices=(_entry("AUSDT", event),))
    outside = notices.NoticeBook(notices=(_entry("AUSDT", event - 90 * DAY_MS - 1),))
    assert inside.causal("AUSDT", event) is not None
    assert on_event.causal("AUSDT", event) is not None
    assert outside.causal("AUSDT", event) is None


# ====================================================== decision bar strictly after notice


def _schedule(release_ms: int, *, last_row: int = 60) -> tuple[Any, ...]:
    active = np.ones((100, 1), dtype=np.bool_)
    active[last_row + 1 :, 0] = False
    member = np.ones_like(active)
    close_ms = (np.arange(100, dtype=np.int64) + 1) * BAR_MS
    book = notices.NoticeBook(notices=(_entry("AUSDT", release_ms),))
    return notices.forced_schedule(active, member, close_ms, ("AUSDT",), book)


def test_the_first_forced_bar_opens_at_or_after_the_release() -> None:
    open_30 = 30 * BAR_MS  # the close of row 29 is the open of row 30
    forced, evidence, missing = _schedule(open_30)
    assert not missing
    assert list(np.flatnonzero(forced[:, 0])) == list(range(30, 61))
    assert evidence[0]["first_forced_row"] == 30
    # one millisecond after the open of row 30 the notice is not yet public at that open
    forced, _, _ = _schedule(open_30 + 1)
    assert list(np.flatnonzero(forced[:, 0])) == list(range(31, 61))
    # one millisecond before: the bar that opens exactly at the previous bar's close
    forced, _, _ = _schedule(open_30 - 1)
    assert list(np.flatnonzero(forced[:, 0])) == list(range(30, 61))


def test_a_notice_too_late_to_close_flags_no_row_and_a_cut_off_leg_is_no_event() -> (
    None
):
    # released inside the last available bar: nothing can be closed before the event
    last_open = 60 * BAR_MS
    forced, evidence, missing = _schedule(last_open + 1)
    assert not forced.any() and not missing
    assert evidence[0]["first_forced_row"] == 61
    active = np.ones((100, 1), dtype=np.bool_)  # never ends: no event, no notice needed
    out, evidence, missing = notices.forced_schedule(
        active,
        np.ones_like(active),
        (np.arange(100, dtype=np.int64) + 1) * BAR_MS,
        ("AUSDT",),
        notices.NoticeBook(),
    )
    assert not out.any() and not evidence and not missing


def test_an_event_without_a_causal_notice_is_missing_unless_the_leg_never_was_a_member() -> (
    None
):
    active = np.ones((100, 2), dtype=np.bool_)
    active[61:, 0] = False
    active[71:, 1] = False
    member = np.zeros_like(active)
    member[:, 0] = True  # leg 1 never belongs to a universe: its event is irrelevant
    close_ms = (np.arange(100, dtype=np.int64) + 1) * BAR_MS
    forced, _, missing = notices.forced_schedule(
        active, member, close_ms, ("AUSDT", "BUSDT"), notices.NoticeBook()
    )
    assert not forced.any()
    assert [name for name, _ in missing] == ["AUSDT"]
    with pytest.raises(notices.MissingNoticeError, match="AUSDT"):
        notices.require_notices(missing)
    # a notice published after the event is not causal
    late = notices.NoticeBook(notices=(_entry("AUSDT", 80 * BAR_MS),))
    _, _, missing = notices.forced_schedule(
        active, member, close_ms, ("AUSDT", "BUSDT"), late
    )
    assert [name for name, _ in missing] == ["AUSDT"]


# ======================================================================= stage assembly


def _delisting_stage(book: notices.NoticeBook | None) -> data.StageData:
    return fx.stage(
        overrides={
            "AAAUSDT": {"last_hour": fx.last_hour_of_row(fx.WINDOW, 470)},
            "BBBUSDT": {"last_hour": fx.last_hour_of_row(fx.WINDOW, 450)},
        },
        notice_book=book,
    )


def test_the_assembly_refuses_before_any_replay_naming_every_unmatched_symbol() -> None:
    with pytest.raises(notices.MissingNoticeError) as caught:
        _delisting_stage(None)
    assert "AAAUSDT" in str(caught.value) and "BBBUSDT" in str(caught.value)
    partial = fx.book(("AAAUSDT", fx.close_of(427)))
    with pytest.raises(notices.MissingNoticeError) as caught:
        _delisting_stage(partial)
    assert "BBBUSDT" in str(caught.value) and "AAAUSDT" not in str(caught.value)


def test_the_assembly_schedules_the_forced_rows_and_records_the_evidence() -> None:
    book = fx.book(("AAAUSDT", fx.close_of(427)), ("BBBUSDT", fx.close_of(430)))
    stage = _delisting_stage(book)
    legs = stage.dataset.symbols
    a, b = legs.index("AAAUSDT"), legs.index("BBBUSDT")
    assert list(np.flatnonzero(stage.forced[:, a])) == list(range(428, 471))
    assert list(np.flatnonzero(stage.forced[:, b])) == list(range(431, 451))
    assert {item["symbol"] for item in stage.notice_evidence} == {"AAAUSDT", "BBBUSDT"}
    assert not stage.forced[:, [i for i in range(len(legs)) if i not in (a, b)]].any()
    # the notice digest is part of the dataset identity
    assert "delisting_notices_sha256" in (stage.dataset.identity_payload_json or "")


# ================================================================== file and digest pins


def _document(entries: list[dict[str, str]]) -> bytes:
    return driver.notice_document_bytes(
        {
            "schema_version": notices.SCHEMA_VERSION,
            "entries": entries,
            "overrides": [],
        }
    )


_ENTRY = {
    "symbol": "AAAUSDT",
    "release_utc": "2024-05-06T05:00:00.000Z",
    "article_code": "abc",
    "title": "Binance Futures Will Delist AAAUSDT Perpetual Contract",
    "kind": "delist",
}


def test_the_notice_file_is_verified_against_its_pin_before_it_is_parsed(
    tmp_path: Path,
) -> None:
    good = tmp_path / "good.json"
    good.write_bytes(_document([_ENTRY]))
    digest = hashlib.sha256(good.read_bytes()).hexdigest()
    book = notices.load_notices(good, expected=digest)
    assert book.sha256 == digest and book.notices[0].symbol == "AAAUSDT"
    # a one-byte change is refused by the digest, even though it is not even valid JSON
    bad = tmp_path / "bad.json"
    bad.write_bytes(good.read_bytes() + b"x")
    with pytest.raises(notices.NoticeError, match="differs from the pinned"):
        notices.load_notices(bad, expected=digest)


def test_the_committed_notice_file_matches_the_protocol_pin() -> None:
    assert protocol.DELISTING_NOTICES_SHA256 is not None
    book = notices.load_notices()
    assert book.sha256 == protocol.DELISTING_NOTICES_SHA256
    assert book.notices


def test_the_notice_pin_is_part_of_the_protocol_digest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = protocol.protocol_payload()
    assert (
        payload["forced_close"]["notices_sha256"] == protocol.DELISTING_NOTICES_SHA256
    )
    before = protocol.protocol_digest()
    monkeypatch.setattr(protocol, "DELISTING_NOTICES_SHA256", "f" * 64)
    assert protocol.protocol_digest() != before


@pytest.mark.parametrize(
    "mutation",
    [
        lambda e: {**e, "symbol": "aaausdt"},
        lambda e: {**e, "kind": "other"},
        lambda e: {**e, "release_utc": "2024-05-06"},
        lambda e: {k: v for k, v in e.items() if k != "title"},
        lambda e: {**e, "extra": "x"},
    ],
)
def test_a_malformed_notice_entry_is_refused(mutation: Any) -> None:
    with pytest.raises(notices.NoticeError):
        notices.parse_notices(_document([mutation(_ENTRY)]))


def test_unsorted_or_duplicate_entries_are_refused() -> None:
    other = {**_ENTRY, "symbol": "AAAUSDT", "release_utc": "2024-05-05T05:00:00.000Z"}
    with pytest.raises(notices.NoticeError, match="sorted"):
        notices.parse_notices(_document([_ENTRY, other]))
    with pytest.raises(notices.NoticeError, match="sorted"):
        notices.parse_notices(_document([_ENTRY, _ENTRY]))


# ========================================================================= the real run gate


def _fail(name: str) -> Any:
    def fail(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError(f"{name} was called before the notice gate")

    return fail


def _argv(tmp_path: Path, *flags: str) -> list[str]:
    return [
        "run",
        "--stage",
        "DEV",
        *flags,
        "--archive-dir",
        str(tmp_path / "archive"),
        "--exchange-info-perp",
        str(tmp_path / "rules.json"),
        "--output-dir",
        str(tmp_path / "out"),
        "--authorization-url",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
    ]


def test_a_real_run_is_refused_before_any_io_when_the_notice_hash_mismatches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(protocol, "DELISTING_NOTICES_SHA256", "a" * 64)
    wrong = tmp_path / "notices.json"
    wrong.write_bytes(_document([_ENTRY]))  # a valid file, but not the pinned bytes
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        _argv(tmp_path, "--delisting-notices", str(wrong)),
        stdout=out,
        stderr=err,
        now=_fail("clock"),
        state=_fail("git"),
        gh=_fail("G4"),
    )
    assert code == driver.EXIT_REFUSED
    assert "differs from the pinned" in err.getvalue() and not out.getvalue()
    assert not (tmp_path / "out").exists()


def test_a_real_run_passes_the_notice_gate_only_with_the_pinned_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good = tmp_path / "notices.json"
    good.write_bytes(_document([_ENTRY]))
    monkeypatch.setattr(
        protocol,
        "DELISTING_NOTICES_SHA256",
        hashlib.sha256(good.read_bytes()).hexdigest(),
    )
    reached: list[str] = []

    def clock() -> datetime:
        reached.append("clock")
        raise RuntimeError("stop after the gate")

    with pytest.raises(RuntimeError, match="stop after the gate"):
        driver.main(
            _argv(tmp_path, "--delisting-notices", str(good)),
            stdout=io.StringIO(),
            stderr=io.StringIO(),
            now=clock,
            state=_fail("git"),
            gh=_fail("G4"),
        )
    assert reached == ["clock"]


def test_a_real_run_is_refused_while_the_pin_is_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(protocol, "DELISTING_NOTICES_SHA256", None)
    err = io.StringIO()
    code = driver.main(
        _argv(tmp_path),
        stdout=io.StringIO(),
        stderr=err,
        now=_fail("clock"),
        state=_fail("git"),
        gh=_fail("G4"),
    )
    assert code == driver.EXIT_REFUSED and "not pinned" in err.getvalue()


# ======================================================================== parsing and CMS


@pytest.mark.parametrize(
    ("title", "kind", "symbols"),
    [
        (
            "Binance Futures Will Delist USDⓈ-M XEMUSDT, ORBSUSDT and LOOMUSDT "
            "Perpetual Contracts (2024-12-09)",
            "delist",
            ["LOOMUSDT", "ORBSUSDT", "XEMUSDT"],
        ),
        (
            "Binance Futures Will Delist AKRO and DODO USDT-Margined Contracts",
            "delist",
            ["AKROUSDT", "DODOUSDT"],
        ),
        (
            "Binance Futures Will Delist USDT-Margined SC Perpetual Contract",
            "delist",
            ["SCUSDT"],
        ),
        (
            "Binance Futures Will Delist LUNA USDT-Margined Contract",
            "delist",
            ["LUNAUSDT"],
        ),
        (
            "Binance Futures Will Delist COIN-M GALAUSD, ZILUSD and VETUSD Perpetual "
            "Contracts",
            "delist",
            [],
        ),
        (
            "Binance Futures Will Delist FTTBUSD USDT-Margined Contract",
            "delist",
            [],
        ),
        (
            "Binance Futures Will Delist USDⓈ-M SLERFUSDT and COIN-M CHZUSD "
            "Perpetual Contracts (2025-10-20 & 2025-10-21)",
            "delist",
            ["SLERFUSDT"],
        ),
        (
            "Binance Futures Will Delist Multiple USDⓈ-M Perpetual Contracts "
            "(2026-10-05)",
            "delist",
            [],
        ),
        ("Binance Will Delist ABC, DEF, GHI on 2024-01-01", "delist", []),
        ("Notice of Removal of Spot Trading Pairs - 2026-11-20", None, None),
        ("Binance Margin Will Delist XYZ on 2023-01-01", None, None),
        ("Binance Will Delist UTK and Support Rebranding on Binance Alpha", None, None),
        (
            "Binance Futures Will Support the MATIC to POL Token Swap and Migration",
            "swap",
            [],
        ),
        (
            "Binance Will Support the Polygon (MATIC) Token Swap to Polygon (POL)",
            "swap",
            [],
        ),
        (
            "Binance Will Support the Frontier (FRONT) Token Swap and Rebranding to Self "
            "Chain (SLF)",
            "swap",
            [],
        ),
        (
            "Binance Has Completed the Render (RNDR) Token Swap and Rebranding",
            None,
            None,
        ),
        ("Binance Will Support the Foo (FOO) Contract Swap", None, None),
        ("Binance Funding Account to be Renamed Stocks Account", None, None),
        (
            "USDⓈ-Margined PUMPUSDT Perpetual Contract Will Be Renamed to "
            "USDⓈ-Margined PUMPBTCUSDT Perpetual Contract (2025-06-13)",
            "rebrand",
            ["PUMPUSDT"],
        ),
    ],
)
def test_title_parsing(title: str, kind: str | None, symbols: list[str] | None) -> None:
    assert driver.notice_kind(title) == kind
    if kind is not None:
        assert driver.symbols_from_title(title, kind=kind) == symbols


def _body(*lines: str) -> str:
    def paragraph(text: str) -> dict[str, Any]:
        return {
            "node": "element",
            "tag": "p",
            "child": [
                {
                    "node": "element",
                    "tag": "span",
                    "child": [{"node": "text", "text": text}],
                }
            ],
        }

    return json.dumps({"node": "root", "child": [paragraph(line) for line in lines]})


def test_body_parsing_reads_futures_statements_of_spot_notices() -> None:
    delist = _body(
        "Binance Futures will close all positions and conduct an automatic settlement on "
        "the contracts of the aforementioned token(s) at 2026-03-24 09:00 (UTC).",
        "Please Note: For futures perpetual contracts, please refer to the Futures notices.",
    )
    title = "Binance Will Delist A2Z, FORTH, HOOK on 2026-04-01"
    assert driver.symbols_from_body(delist, title=title) == [
        "A2ZUSDT",
        "FORTHUSDT",
        "HOOKUSDT",
    ]
    # without the statement about the tokens, a spot delisting names no perpetual
    assert driver.symbols_from_body(_body("Spot trading stops."), title=title) == []
    swap = _body(
        "At 2023-02-11 04:00 (UTC), Binance Futures will close all positions on the "
        "USDT-Margined BNX Perpetual Contract, and delist this contract.",
        "At 2021-12-19 02:00 (UTC), Binance Futures will close all BZRX futures positions "
        "and delist the BZRX/USDT futures contracts.",
        "At 2024-09-04 09:00 (UTC), Binance Futures will close all positions on the "
        "MATICUSDT and MATICUSDC USDT-Margined contracts.",
        "Binance Futures will adjust the leverage of the CCCUSDT Perpetual Contract.",
    )
    assert driver.symbols_from_body(swap) == [
        "BNXUSDT",
        "BZRXUSDT",
        "CCCUSDT",
        "MATICUSDT",
    ]


def test_body_parsing_takes_only_perpetual_contract_lines() -> None:
    body = _body(
        "Fellow Binancians, BTCUSDT is mentioned in a risk note",
        "2026-10-05 09:00 (UTC)",
        ": USDⓈ-M PROMPTUSDT, PUMPBTCUSDT and 1000000BOBUSDT Perpetual Contracts",
        ": COIN-M MANAUSD Perpetual Contracts",
    )
    assert driver.symbols_from_body(body) == [
        "1000000BOBUSDT",
        "PROMPTUSDT",
        "PUMPBTCUSDT",
    ]
    assert driver.symbols_from_body("not json") == []


class FakeCms:
    """A fake Binance CMS API served from dictionaries (no network)."""

    def __init__(
        self,
        catalogs: dict[int, list[dict[str, Any]]],
        details: dict[str, dict[str, Any]],
    ) -> None:
        self.catalogs, self.details, self.calls = catalogs, details, []

    def __call__(self, url: str) -> bytes:
        self.calls.append(url)
        if "list/query" in url:
            catalog = int(url.split("catalogId=")[1].split("&")[0])
            page = int(url.split("pageNo=")[1].split("&")[0])
            rows = self.catalogs.get(catalog, [])
            chunk = rows[(page - 1) * 20 : page * 20]
            body = {
                "catalogs": [{"articles": chunk, "total": len(rows)}],
            }
        else:
            code = url.split("articleCode=")[1]
            body = self.details[code]
        return json.dumps({"code": "000000", "success": True, "data": body}).encode()


def _cms() -> FakeCms:
    release = _ms(datetime(2024, 5, 6, 5, tzinfo=UTC))
    multi = _ms(datetime(2025, 11, 18, 4, 1, tzinfo=UTC))
    swap = _ms(datetime(2024, 9, 1, 4, tzinfo=UTC))
    articles = [
        {
            "code": "a1",
            "title": "Binance Futures Will Delist USDⓈ-M AAAUSDT and BBBUSDT "
            "Perpetual Contracts",
            "releaseDate": release,
        },
        {
            "code": "a2",
            "title": "Binance Futures Will Delist Multiple USDⓈ-M Perpetual Contracts",
            "releaseDate": multi,
        },
        {
            "code": "a3",
            "title": "Binance Will Delist CCC on 2025-01-01",
            "releaseDate": release,
        },
        {
            "code": "a4",
            "title": "Binance Futures Will Support the OLD to NEW Token Swap",
            "releaseDate": swap,
        },
    ]
    details = {
        "a2": {
            "code": "a2",
            "title": articles[1]["title"],
            "publishDate": multi,
            "body": _body(": USDⓈ-M DDDUSDT and EEEUSDT Perpetual Contracts"),
        },
        "a4": {
            "code": "a4",
            "title": articles[3]["title"],
            "publishDate": swap,
            "body": _body(": USDⓈ-M OLDUSDT Perpetual Contract is swapped to NEWUSDT"),
        },
        "a3": {
            "code": "a3",
            "title": articles[2]["title"],
            "publishDate": release,
            "body": _body("Spot trading of CCC stops."),
        },
        "m1": {
            "code": "m1",
            "title": "Binance Will Migrate OLD to NEW (rebrand)",
            "publishDate": swap,
            "body": _body("The OLD token and OLDUSDT contract move."),
        },
    }
    return FakeCms({161: articles, 49: [], 48: []}, details)


def test_the_document_holds_matched_symbols_sorted_with_codes_and_release_times() -> (
    None
):
    cms = _cms()
    document = driver.build_notice_document(
        cms, since=datetime(2020, 10, 1, tzinfo=UTC)
    )
    rows = [(e["symbol"], e["article_code"], e["kind"]) for e in document["entries"]]
    assert rows == [
        ("AAAUSDT", "a1", "delist"),
        ("BBBUSDT", "a1", "delist"),
        ("DDDUSDT", "a2", "delist"),
        ("EEEUSDT", "a2", "delist"),
        ("NEWUSDT", "a4", "swap"),
        ("OLDUSDT", "a4", "swap"),
    ]
    assert document["entries"][0]["release_utc"] == "2024-05-06T05:00:00.000Z"
    # the title-less ("Multiple"), swap and spot-delisting articles needed a detail fetch
    assert sum("detail/query" in url for url in cms.calls) == 3
    raw = driver.notice_document_bytes(document)
    assert raw == driver.notice_document_bytes(json.loads(raw))
    assert [n.symbol for n in notices.parse_notices(raw).notices] == [
        "AAAUSDT",
        "BBBUSDT",
        "DDDUSDT",
        "EEEUSDT",
        "NEWUSDT",
        "OLDUSDT",
    ]


def test_an_override_takes_code_title_and_time_from_the_real_article() -> None:
    cms = _cms()
    document = driver.build_notice_document(
        cms,
        overrides=[
            {
                "symbol": "OLDUSDT",
                "article_code": "m1",
                "kind": "migration",
                "evidence": "the OLD to NEW migration moved OLDUSDT",
            }
        ],
    )
    (override,) = document["overrides"]
    assert override["release_utc"] == "2024-09-01T04:00:00.000Z"
    assert override["title"] == "Binance Will Migrate OLD to NEW (rebrand)"
    book = notices.parse_notices(driver.notice_document_bytes(document))
    assert sorted(n.override for n in book.notices if n.symbol == "OLDUSDT") == [
        False,
        True,
    ]
    # an article that does not name the symbol cannot support an override
    with pytest.raises(ValueError, match="does not name"):
        driver.build_notice_document(
            cms,
            overrides=[
                {
                    "symbol": "ZZZUSDT",
                    "article_code": "m1",
                    "kind": "migration",
                    "evidence": "invented",
                }
            ],
        )
    with pytest.raises(ValueError, match="exactly"):
        driver.build_notice_document(cms, overrides=[{"symbol": "OLDUSDT"}])


def test_the_notices_command_needs_the_network_flag_and_is_write_once(
    tmp_path: Path,
) -> None:
    target = tmp_path / "notices.json"
    err = io.StringIO()
    code = driver.main(
        ["notices", "--output", str(target)], stdout=io.StringIO(), stderr=err
    )
    assert code == driver.EXIT_REFUSED and "--allow-network" in err.getvalue()
    assert not target.exists()
    out = io.StringIO()
    code = driver.main(
        ["notices", "--output", str(target)],
        stdout=out,
        stderr=io.StringIO(),
        notice_fetch=_cms(),
    )
    assert code == driver.EXIT_OK
    report = json.loads(out.getvalue())
    assert report["sha256"] == hashlib.sha256(target.read_bytes()).hexdigest()
    assert report["entries"] == 6
    code = driver.main(
        ["notices", "--output", str(target)],
        stdout=io.StringIO(),
        stderr=io.StringIO(),
        notice_fetch=_cms(),
    )
    assert code == driver.EXIT_REFUSED


def test_the_release_time_round_trips_milliseconds() -> None:
    moment = datetime(2026, 10, 30, 7, 40, 9, 176000, tzinfo=UTC)
    text = notices.release_utc(_ms(moment))
    assert text == "2026-10-30T07:40:09.176Z"
    assert notices.release_ms(text) == _ms(moment)
    with pytest.raises(notices.NoticeError):
        notices.release_ms("2026-10-30")
