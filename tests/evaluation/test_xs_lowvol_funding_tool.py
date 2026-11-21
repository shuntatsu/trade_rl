"""Synthetic tests of ``tools.xs_lowvol_funding`` (plan, coverage, halts, run order, guards).

A deterministic fake row source serves USD-M perpetual klines and funding; nothing touches the
network, the market store or an economic result.
"""

from __future__ import annotations

import hashlib
import io
import json
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.evaluation import test_core7_mtf_tierd as tierd_tests
from tools import xs_lowvol_funding as driver
from trade_rl.evaluation.xs_lowvol_funding import data, policy, protocol, runner
from trade_rl.integrations.binance.carry import CarryCoverageError
from trade_rl.integrations.market_store.contracts import (
    FundingRow,
    KlineRow,
    StoreMarket,
)

HOUR_MS = 3_600_000
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"
DATA_AUTH = protocol.DATA_AUTHORIZATION_URL
NOW = datetime(2026, 11, 21, 6, 30, tzinfo=UTC)
FINAL_NOW = datetime(2027, 5, 9, 0, 0, 1, tzinfo=UTC)
COMMIT = "c" * 40
SYMBOLS = tuple(f"S{i:02d}USDT" for i in range(14))
EPOCH = datetime(2019, 1, 1, tzinfo=UTC)

SMALL = {
    "DEV": {
        "load_start": "2022-01-03T00:00:00Z",
        "first_decision": "2022-03-07T00:00:00Z",
        "last_decision_bar": "2022-03-21T00:00:00Z",
        "settles": "2022-03-25T00:00:00Z",
    },
    "DISJOINT": {
        "load_start": "2021-01-01T00:00:00Z",
        "first_decision": "2021-04-05T00:00:00Z",
        "last_decision_bar": "2021-04-19T00:00:00Z",
        "settles": "2021-04-23T00:00:00Z",
    },
}


_MP: list[pytest.MonkeyPatch] = []


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> Any:
    _MP[:] = [monkeypatch]
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)
    yield
    _MP.clear()


def small_window(monkeypatch: pytest.MonkeyPatch, stage: str) -> data.StageWindow:
    monkeypatch.setitem(protocol.STAGE_WINDOWS, stage, SMALL[stage])
    return data.frozen_window(stage)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000)


class Rows:
    """Deterministic fake Vision source: each symbol is listed in ``[first, last)``."""

    def __init__(self, listed: dict[str, tuple[datetime, datetime | None]]) -> None:
        self.listed = listed
        self.skip: dict[str, set[int]] = {}
        self.requests: list[tuple[str, int, int]] = []
        self.daily: set[tuple[str, int]] = set()
        self.funding_skip: dict[str, set[int]] = {}
        self.funding_missing: dict[str, set[tuple[int, int]]] = {}
        self.zombie_from: dict[str, int] = {}
        self.zero_hours: dict[str, set[int]] = {}
        self._cache: dict[str, np.ndarray] = {}
        self._epoch = _ms(EPOCH)

    def _path(self, symbol: str) -> np.ndarray:
        if symbol not in self._cache:
            rng = np.random.default_rng(sum(map(ord, symbol)))
            self._cache[symbol] = 100.0 * np.exp(
                np.cumsum(rng.normal(0.0004, 0.01, 100_000))
            )
        return self._cache[symbol]

    def _span(self, symbol: str, start_ms: int, end_ms: int) -> tuple[int, int]:
        first, last = self.listed[symbol]
        lo = max(start_ms, _ms(first))
        hi = end_ms if last is None else min(end_ms, _ms(last))
        return lo, hi

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        assert interval == "1h" and market is StoreMarket.USDS_M
        self.requests.append((symbol, start_ms, end_ms))
        lo, hi = self._span(symbol, start_ms, end_ms)
        if lo >= hi:
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        path = self._path(symbol)
        skip = self.skip.get(symbol, set())
        rows: list[KlineRow] = []
        for open_ms in range(lo, hi, HOUR_MS):
            if open_ms in skip:
                continue
            i = (open_ms - self._epoch) // HOUR_MS
            o, c = float(path[i - 1]), float(path[i])
            dead = open_ms >= self.zombie_from.get(symbol, 1 << 62) or open_ms in (
                self.zero_hours.get(symbol, set())
            )
            quote = (
                0.0
                if dead
                else (
                    150_000_000.0
                    + (i % 7) * 1_000_000.0
                    + 1_000.0 * (sum(map(ord, symbol)) % 50)
                )
            )
            rows.append(
                KlineRow(
                    open_time_ms=open_ms,
                    open=Decimal(f"{o:.8f}"),
                    high=Decimal(f"{max(o, c) * 1.001:.8f}"),
                    low=Decimal(f"{min(o, c) * 0.999:.8f}"),
                    close=Decimal(f"{c:.8f}"),
                    volume=Decimal("0" if dead else "1000000.0"),
                    close_time_ms=open_ms + HOUR_MS - 1,
                    quote_volume=Decimal(f"{quote:.2f}"),
                    trades=100,
                    taker_buy_base_volume=Decimal("400000.0"),
                    taker_buy_quote_volume=Decimal(f"{quote * 0.4:.2f}"),
                )
            )
        return rows

    def read_funding(
        self, market: StoreMarket, symbol: str, start_ms: int, end_ms: int
    ) -> list[FundingRow]:
        assert market is StoreMarket.USDS_M
        lo, hi = self._span(symbol, start_ms, end_ms)
        if lo >= hi:
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        moment = datetime.fromtimestamp(start_ms / 1_000, UTC)
        if (moment.year, moment.month) in self.funding_missing.get(symbol, set()):
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        skip = self.funding_skip.get(symbol, set())
        return [
            FundingRow(funding_time_ms=t, rate=Decimal("0.0001"))
            for t in range(lo, min(end_ms, hi) + 1, HOUR_MS)
            if (t // HOUR_MS) % 8 == 0 and t not in skip
        ]

    def probe_published_day(
        self, market: StoreMarket, symbol: str, day_ms: int
    ) -> bool:
        return (symbol, day_ms) in self.daily


def listings(
    window: data.StageWindow,
    symbols: Sequence[str] = SYMBOLS,
    *,
    last: dict[str, datetime] | None = None,
    first: dict[str, datetime] | None = None,
) -> dict[str, tuple[datetime, datetime | None]]:
    return {
        s: ((first or {}).get(s, window.load_start), (last or {}).get(s))
        for s in symbols
    }


def _body(
    digest: str | None = None, *, pid: bool = True, commit: str | None = COMMIT
) -> str:
    lines = ["G4 authorization: xs lowvol funding"]
    if pid:
        lines.append(f"protocol_id: {protocol.PROTOCOL_ID}")
    lines.append(f"protocol_digest: {digest or protocol.protocol_digest()}")
    if commit:
        lines.append(f"code_commit: {commit}")
    return "\n".join(lines) + "\n"


class Gh:
    """Serves the G4 comment and the data-authorization comment by comment id."""

    def __init__(self, *, body: str | None = None, **g4_kwargs: Any) -> None:
        self.g4 = tierd_tests.FakeGh(url=AUTH, body=body or _body(), **g4_kwargs)
        self.data = tierd_tests.FakeGh(url=DATA_AUTH, body="owner relay comment\n")
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> str:
        self.calls.append(list(args))
        target = self.data if str(args[1]).endswith("/7040247247") else self.g4
        return target(args)


def perp_payload(symbols: Sequence[str], *, absent: Sequence[str] = ()) -> bytes:
    entries = [
        {
            "symbol": s,
            "status": "TRADING",
            "contractType": "PERPETUAL",
            "filters": [
                {"filterType": "LOT_SIZE", "stepSize": "0.001"},
                {"filterType": "MIN_NOTIONAL", "notional": "5"},
            ],
        }
        for s in symbols
        if s not in absent
    ]
    return json.dumps({"symbols": entries}).encode("utf-8")


def tool(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    gh: Any,
    now: datetime,
    rows: Any,
    clean: bool = True,
    symbols: Sequence[str] = SYMBOLS,
    absent: Sequence[str] = (),
    with_rules: bool = True,
    rest_fetch: Any = None,
) -> tuple[int, str, str]:
    perp = tmp_path / "perp.json"
    perp.write_bytes(perp_payload([*symbols, *protocol.DEV_ROSTER], absent=absent))
    extra = ["--exchange-info-perp", str(perp)] if with_rules else []
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        [*argv, "--archive-dir", str(tmp_path / "archives"), *extra],
        stdout=out,
        stderr=err,
        state=lambda: (COMMIT, clean),
        rows_factory=lambda plan: rows,
        now=lambda: now,
        gh=gh,
        rest_fetch=rest_fetch,
    )
    return code, out.getvalue(), err.getvalue()


def run_argv(
    tmp_path: Path, stage: str, *flags: str, name: str = "out", url: str = AUTH
) -> list[str]:
    confirm = (
        ["--confirm-single-use"]
        if stage == "DISJOINT" and "--no-confirm" not in flags
        else []
    )
    flags = tuple(f for f in flags if f != "--no-confirm")
    return [
        "run",
        "--stage",
        stage,
        *confirm,
        *flags,
        "--output-dir",
        str(tmp_path / name),
        "--authorization-url",
        url,
    ]


def _write(tmp_path: Path, name: str, document: dict[str, Any]) -> Path:
    stem = hashlib.sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()[
        :10
    ]
    path = tmp_path / f"{stem}-{name}"
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


def dev_result(tmp_path: Path, **over: Any) -> Path:
    return _write(
        tmp_path,
        "dev-result.json",
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_digest": protocol.protocol_digest(),
            "stage": "DEV",
            "mode": "decision",
            "status": "DEV_DIAGNOSTIC_NOT_A_DECISION",
            "invalid_cells": [],
            **over,
        },
    )


def disjoint_decision(tmp_path: Path, label: str = "XS_SUPPORTED", **over: Any) -> Path:
    return _write(
        tmp_path,
        "disjoint-decision.json",
        {
            "protocol_id": protocol.PROTOCOL_ID,
            "protocol_digest": protocol.protocol_digest(),
            "stage": "DISJOINT",
            "mode": "decision",
            "label": label,
            **over,
        },
    )


def symbol_file(tmp_path: Path, symbols: Sequence[str] = SYMBOLS) -> Path:
    stem = hashlib.sha256(json.dumps(list(symbols)).encode()).hexdigest()[:10]
    path = tmp_path / f"{stem}-forward-symbols.json"
    path.write_text(json.dumps(list(symbols)), encoding="utf-8")
    if _MP:
        _MP[0].setattr(
            protocol,
            "FORWARD_SYMBOLS_SHA256",
            hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    return path


class NoRun:
    """A fake ``run_stage`` that records its call and writes nothing."""

    def __init__(self, label: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.label = label

    def __call__(self, stage: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {
            "label": self.label,
            "status": None,
            "failed_conditions": [],
            "invalid_cells": [],
        }


@pytest.fixture
def no_run(monkeypatch: pytest.MonkeyPatch) -> NoRun:
    fake = NoRun()
    monkeypatch.setattr(driver.runner, "run_stage", fake)
    return fake


# ======================================================================== plan


def test_plan_prints_the_digest_the_bindings_and_every_window() -> None:
    for stage in protocol.STAGES:
        out, err = io.StringIO(), io.StringIO()
        code = driver.main(
            ["plan", "--stage", stage], stdout=out, stderr=err, now=lambda: NOW
        )
        assert code == driver.EXIT_OK, err.getvalue()
        document = json.loads(out.getvalue())
        assert document["protocol_digest"] == protocol.protocol_digest()
        assert document["protocol_id"] == "xs_lowvol_funding_v1"
        assert document["candidates_sha256"] == protocol.CANDIDATES_SHA256
        assert document["initial_margin_rate"] == 0.4
        assert set(document["windows"]) == set(protocol.STAGES)
        assert document["window"]["stage"] == stage
        assert document["final_not_before"] == "2027-05-08T00:00:00Z"
        assert document["harness_source_digest"] == protocol.harness_source_digest()
        assert "daily_fallback" in json.dumps(document["protocol"])


def test_plan_refuses_when_the_candidate_binding_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(protocol, "CANDIDATES_SHA256", "9" * 64)
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        ["plan", "--stage", "DISJOINT"], stdout=out, stderr=err, now=lambda: NOW
    )
    assert code == driver.EXIT_REFUSED and out.getvalue() == ""
    assert "refused" in err.getvalue()


def test_the_digest_binds_the_harness_and_the_constants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    digest = protocol.protocol_digest()
    monkeypatch.setattr(protocol, "VOL_TARGET", 0.25)
    assert protocol.protocol_digest() != digest
    monkeypatch.undo()
    monkeypatch.setattr(protocol, "harness_source_digest", lambda **_: "f" * 64)
    assert protocol.protocol_digest() != digest


# =================================================================== run order


def test_the_run_order_dev_then_disjoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    dev_window = small_window(monkeypatch, protocol.STAGE_DEV)
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    rows = Rows({**listings(window), **listings(dev_window, protocol.DEV_ROSTER)})

    def go(argv: list[str], now: datetime = NOW) -> tuple[int, str]:
        code, _, err = tool(tmp_path, argv, gh=Gh(), now=now, rows=rows)
        return code, err

    code, err = go(run_argv(tmp_path, "DISJOINT"))
    assert code == driver.EXIT_REFUSED and "needs --dev-result" in err
    for bad in (
        dev_result(tmp_path, protocol_digest="0" * 64),
        dev_result(tmp_path, stage="DISJOINT"),
        dev_result(tmp_path, mode="interim"),
        dev_result(tmp_path, protocol_id="agy_anchor_realistic_v1"),
    ):
        code, err = go(run_argv(tmp_path, "DISJOINT", "--dev-result", str(bad)))
        assert code == driver.EXIT_REFUSED and "DEV result of this exact" in err, bad
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    code, err = go(run_argv(tmp_path, "DISJOINT", "--dev-result", str(junk)))
    assert code == driver.EXIT_REFUSED and "not JSON" in err
    # a DEV result with an invalid cell stops the line: DISJOINT is refused
    stopped = dev_result(tmp_path, invalid_cells=["CANDIDATE__BASE"])
    code, err = go(run_argv(tmp_path, "DISJOINT", "--dev-result", str(stopped)))
    assert code == driver.EXIT_REFUSED and "DISJOINT is refused" in err
    no_list = dev_result(tmp_path, invalid_cells=None)
    code, err = go(run_argv(tmp_path, "DISJOINT", "--dev-result", str(no_list)))
    assert code == driver.EXIT_REFUSED and "no invalid_cells list" in err
    assert not no_run.calls and not (tmp_path / "out").exists()
    # DEV takes no prior result and no mode flag
    good = dev_result(tmp_path)
    code, err = go(run_argv(tmp_path, "DEV", "--dev-result", str(good)))
    assert code == driver.EXIT_REFUSED and "belongs to DISJOINT" in err
    code, err = go(run_argv(tmp_path, "DEV", "--interim"))
    assert code == driver.EXIT_REFUSED and "takes no" in err
    code, err = go(run_argv(tmp_path, "DISJOINT", "--final", "--dev-result", str(good)))
    assert code == driver.EXIT_REFUSED and "takes no" in err
    code, err = go(
        run_argv(
            tmp_path,
            "DISJOINT",
            "--dev-result",
            str(good),
            "--disjoint-decision",
            str(good),
        )
    )
    assert code == driver.EXIT_REFUSED and "belongs to FORWARD" in err
    # a valid DEV result lets DISJOINT run, with the DEV result recorded in the receipt
    code, err = go(run_argv(tmp_path, "DISJOINT", "--dev-result", str(good)))
    assert code == driver.EXIT_OK, err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["dev_result"]["status"] == "DEV_DIAGNOSTIC_NOT_A_DECISION"
    assert len(receipt["dev_result"]["sha256"]) == 64
    assert no_run.calls[-1]["disjoint_label"] is None
    code, err = go(run_argv(tmp_path, "DEV", name="devout"))
    assert code == driver.EXIT_OK, err


def test_forward_needs_a_supported_disjoint_decision_of_the_same_digest(
    tmp_path: Path, no_run: NoRun
) -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    forward = symbol_file(tmp_path)

    def go(
        *flags: str, now: datetime = FINAL_NOW, name: str = "fwd"
    ) -> tuple[int, str]:
        code, _, err = tool(
            tmp_path,
            run_argv(tmp_path, "FORWARD", *flags, name=name),
            gh=Gh(),
            now=now,
            rows=rows,
        )
        return code, err

    code, err = go("--forward-symbols", str(forward))
    assert code == driver.EXIT_REFUSED and "exactly one of --final and --interim" in err
    code, err = go("--final", "--forward-symbols", str(forward))
    assert code == driver.EXIT_REFUSED and "needs --disjoint-decision" in err
    good = disjoint_decision(tmp_path)
    flags = ("--forward-symbols", str(forward), "--disjoint-decision", str(good))
    code, err = go("--final", *flags, "--through", "2026-09-10T00:00:00Z")
    assert code == driver.EXIT_REFUSED and "for --interim only" in err
    code, err = go("--final", *flags, "--dev-result", str(good))
    assert code == driver.EXIT_REFUSED and "belongs to DISJOINT" in err
    for bad in (
        disjoint_decision(tmp_path, protocol_digest="0" * 64),
        disjoint_decision(tmp_path, stage="DEV"),
        disjoint_decision(tmp_path, mode="interim"),
        disjoint_decision(tmp_path, label="XS_NOT_SUPPORTED"),
        disjoint_decision(tmp_path, label="INVALID"),
        disjoint_decision(tmp_path, label=None),
    ):
        code, err = go(
            "--final",
            "--forward-symbols",
            str(forward),
            "--disjoint-decision",
            str(bad),
        )
        assert code == driver.EXIT_REFUSED, bad
        assert "DISJOINT" in err and not no_run.calls
    # a NOT_SUPPORTED label ends the line: no FORWARD run, interim included
    stopped = disjoint_decision(tmp_path, label="XS_NOT_SUPPORTED")
    code, err = go(
        "--interim",
        "--forward-symbols",
        str(forward),
        "--disjoint-decision",
        str(stopped),
    )
    assert code == driver.EXIT_REFUSED and "the line stops" in err
    # the forward listing must obey the spec's exclusions
    for bad_symbols in (
        ["币安人生USDT", "BTCUSDT"],
        ["USDCUSDT"],
        ["ABCUSDT", "ABCUSDT"],
    ):
        bad_file = symbol_file(tmp_path, bad_symbols)  # re-pins, so the digest moves
        code, err = go(
            "--final",
            "--forward-symbols",
            str(bad_file),
            "--disjoint-decision",
            str(disjoint_decision(tmp_path)),
        )
        assert code == driver.EXIT_REFUSED and "FORWARD" in err
    symbol_file(tmp_path)  # re-pin the good listing
    code, err = go("--final", *flags)
    assert code == driver.EXIT_OK, err
    receipt = json.loads(
        (tmp_path / "fwd" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["disjoint_decision"]["label"] == "XS_SUPPORTED"
    assert receipt["forward_symbol_file_sha256"] and receipt["data_authorization"]
    assert no_run.calls[-1]["disjoint_label"] == "XS_SUPPORTED"


def test_final_is_refused_before_2027_05_08_even_when_everything_else_is_valid(
    tmp_path: Path, no_run: NoRun
) -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    flags = (
        "--final",
        "--forward-symbols",
        str(symbol_file(tmp_path)),
        "--disjoint-decision",
        str(disjoint_decision(tmp_path)),
    )
    for now in (
        datetime(2027, 5, 7, 23, 59, 59, tzinfo=UTC),
        datetime(2027, 5, 2, tzinfo=UTC),
    ):
        code, _, err = tool(
            tmp_path, run_argv(tmp_path, "FORWARD", *flags), gh=Gh(), now=now, rows=rows
        )
        assert code == driver.EXIT_REFUSED and "--final is refused before" in err
    assert not no_run.calls and rows.requests == []
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "FORWARD", *flags),
        gh=Gh(),
        now=datetime(2027, 5, 8, tzinfo=UTC),
        rows=rows,
    )
    assert code == driver.EXIT_OK, err


def test_a_forward_interim_writes_a_dated_report_and_reads_only_closed_days(
    tmp_path: Path,
) -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    code, out, err = tool(
        tmp_path,
        run_argv(
            tmp_path,
            "FORWARD",
            "--interim",
            "--forward-symbols",
            str(symbol_file(tmp_path)),
            "--disjoint-decision",
            str(disjoint_decision(tmp_path)),
            "--through",
            "2026-09-30T00:00:00Z",
        ),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_OK, err
    report = tmp_path / "out" / "INTERIM_NOT_A_DECISION-20261121.json"
    document = json.loads(report.read_text("utf-8"))
    assert document["label"] is None and document["status"] == "INTERIM_NOT_A_DECISION"
    assert document["mode"] == "interim" and document["stage"] == "FORWARD"
    assert document["through_close"].startswith("2026-09-30")
    assert "conditions" not in document  # no verdict in an interim report
    assert not (tmp_path / "out" / "decision.json").exists()
    last_requested = max(end for _, _, end in rows.requests)
    assert last_requested <= _ms(datetime(2026, 9, 30, tzinfo=UTC))
    # months of the stage after the read limit have no universe yet
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert set(receipt["universe_months"]) == {
        "2026-08-01T00:00:00+00:00",
        "2026-09-01T00:00:00+00:00",
    }
    assert json.loads(out)["status"] == "INTERIM_NOT_A_DECISION"


# ==================================================================== G4 gate


def test_a_wrong_or_edited_g4_comment_authenticates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    cases: list[Any] = [
        Gh(body=_body(digest="0" * 64)),  # quotes another digest
        Gh(body=_body(pid=False)),  # does not quote the protocol id
        Gh(body=_body(commit=None)),  # does not quote the code commit
        Gh(body=_body(commit="d" * 40)),  # quotes another commit
        Gh(login="someone-else"),  # not the owner
        Gh(
            variants={2: {"updated_at": "2026-11-21T00:00:00Z"}}
        ),  # edited between fetches
    ]
    for index, gh in enumerate(cases):
        code, out, err = tool(
            tmp_path,
            run_argv(tmp_path, "DEV", name=f"o{index}"),
            gh=gh,
            now=NOW,
            rows=rows,
        )
        assert code == driver.EXIT_UNAUTHENTICATED, (index, err)
        assert out == "" and "authorization failed" in err
        assert not (tmp_path / f"o{index}").exists()
    assert not no_run.calls


def test_the_g4_check_precedes_any_row_and_a_dirty_tree_or_existing_output_refuses(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DEV"),
        gh=Gh(body=_body(digest="0" * 64)),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_UNAUTHENTICATED and rows.requests == []
    code, _, err = tool(
        tmp_path, run_argv(tmp_path, "DEV"), gh=Gh(), now=NOW, rows=rows, clean=False
    )
    assert code == driver.EXIT_REFUSED and "working tree must be clean" in err
    assert rows.requests == [] and not (tmp_path / "out").exists()
    (tmp_path / "taken").mkdir()
    code, _, err = tool(
        tmp_path, run_argv(tmp_path, "DEV", name="taken"), gh=Gh(), now=NOW, rows=rows
    )
    assert code == driver.EXIT_REFUSED and "write-once" in err and rows.requests == []
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DEV", url="https://example.com/not-a-comment"),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and rows.requests == []


def test_a_stage_that_is_not_over_is_refused_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    gh = Gh()
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DEV"),
        gh=gh,
        now=datetime(2022, 3, 10, 5, tzinfo=UTC),
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "not over yet" in err
    assert gh.calls == [] and rows.requests == []


def test_the_data_authorization_comment_must_not_change_between_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    gh = Gh()
    gh.data.variants = {2: {"updated_at": "2026-11-21T00:00:00Z"}}
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", "--dev-result", str(dev_result(tmp_path))),
        gh=gh,
        now=NOW,
        rows=Rows(listings(window)),
    )
    assert code == driver.EXIT_UNAUTHENTICATED and "data-authorization" in err
    assert not (tmp_path / "out").exists()


# ================================================================== end to end


def test_dev_run_end_to_end_writes_receipts_cells_and_a_dev_report_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    gh = Gh()
    code, out, err = tool(
        tmp_path, run_argv(tmp_path, "DEV"), gh=gh, now=NOW, rows=rows
    )
    assert code == driver.EXIT_OK, err
    out_dir = tmp_path / "out"
    assert {p.name for p in out_dir.iterdir()} == {
        driver.DATA_RECEIPT_NAME,
        driver.RUN_RECEIPT_NAME,
        runner.MANIFEST_NAME,
        runner.DEV_NAME,
        runner.CELL_DIR,
    }
    receipt = json.loads((out_dir / driver.DATA_RECEIPT_NAME).read_text("utf-8"))
    assert receipt["protocol_digest"] == protocol.protocol_digest()
    assert receipt["candidates_sha256"] == protocol.CANDIDATES_SHA256
    assert receipt["code_commit"] == COMMIT and receipt["working_tree_clean"] is True
    assert receipt["data_authorization"] is None  # DEV reads known-used data
    assert receipt["legs"] == sorted(protocol.DEV_ROSTER)
    assert receipt["symbols_on_fallback_rules"] == []
    assert not any(str(call[1]).endswith("/7040247247") for call in gh.calls)
    dev = json.loads((out_dir / runner.DEV_NAME).read_text("utf-8"))
    assert dev["label"] is None and dev["gates"] is False
    assert dev["window"]["stage"] == "DEV"
    run_receipt = json.loads((out_dir / driver.RUN_RECEIPT_NAME).read_text("utf-8"))
    assert run_receipt["code_commit"] == COMMIT and run_receipt["label"] is None
    assert json.loads(out)["status"] == "DEV_DIAGNOSTIC_NOT_A_DECISION"
    # outputs are write-once
    code, _, err = tool(
        tmp_path, run_argv(tmp_path, "DEV"), gh=Gh(), now=NOW, rows=rows
    )
    assert code == driver.EXIT_REFUSED and "write-once" in err


def test_disjoint_run_builds_the_monthly_universe_and_records_the_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    rows = Rows(listings(window))
    gh = Gh()
    code, out, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", "--dev-result", str(dev_result(tmp_path))),
        gh=gh,
        now=NOW,
        rows=rows,
        absent=[SYMBOLS[0]],
    )
    assert code == driver.EXIT_OK, err
    out_dir = tmp_path / "out"
    receipt = json.loads((out_dir / driver.DATA_RECEIPT_NAME).read_text("utf-8"))
    assert receipt["symbols_with_data"] == 14 and receipt["legs"] == sorted(SYMBOLS)
    (month,) = receipt["universe_months"].values()
    assert month["book"] is True and len(month["members"]) == 14
    assert receipt["symbols_on_fallback_rules"] == [SYMBOLS[0]]
    assert receipt["data_authorization"]["url"] == DATA_AUTH
    assert any(str(call[1]).endswith("/7040247247") for call in gh.calls)
    decision_file = json.loads((out_dir / runner.DECISION_NAME).read_text("utf-8"))
    assert decision_file["label"] in protocol.LABELS
    assert decision_file["stage"] == "DISJOINT" and set(
        decision_file["conditions"]
    ) == {
        "C1_no_invalid_cell",
        "C2_base_sharpe_at_least_min",
        "C3_high_cost_and_delay_sharpe",
        "C4_base_max_drawdown_below_limit",
        "C5_base_sharpe_positive_in_both_halves",
        "C6_bootstrap_lower_bound_positive",
    }
    assert json.loads(out)["label"] == decision_file["label"]
    # every row read stayed strictly before the 2026-08-01 embargo and inside the window
    assert max(end for _, _, end in rows.requests) <= _ms(window.last_row_close)
    assert min(start for _, start, _ in rows.requests) >= _ms(window.load_start)


def test_an_invalid_result_exits_with_the_invalid_code_and_still_writes_the_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    fake = NoRun(label=protocol.LABEL_INVALID)
    monkeypatch.setattr(driver.runner, "run_stage", fake)
    code, out, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", "--dev-result", str(dev_result(tmp_path))),
        gh=Gh(),
        now=NOW,
        rows=Rows(listings(window)),
    )
    assert code == driver.EXIT_INVALID and "INVALID" in err
    assert json.loads(out)["label"] == "INVALID"
    assert (tmp_path / "out" / driver.RUN_RECEIPT_NAME).exists()


# ==================================================================== loading


def _load(
    rows: Rows, window: data.StageWindow, symbol: str, **kwargs: Any
) -> tuple[data.PerpHourly | None, dict[str, object]]:
    limit = window.last_row_close
    guarded = data.StageRowSource(
        rows, window=window, limit_close=limit, symbols=[symbol]
    )
    return driver.load_perp(
        guarded, guarded, symbol, window=window, limit_close=limit, halt_intervals=()
    )


def test_a_delisted_perp_loads_to_its_last_row_and_an_unlisted_one_is_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    end = window.load_start + timedelta(days=50, hours=5)  # ends mid-bar
    rows = Rows(
        {
            "AAAUSDT": (window.load_start, end),
            "BBBUSDT": (window.last_row_close + timedelta(days=1), None),
        }
    )
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None and coverage["rows"] == int(
        (end - window.load_start) / timedelta(hours=1)
    )
    assert leg.present[: int((end - window.load_start) / timedelta(hours=1))].all()
    assert not leg.present[int((end - window.load_start) / timedelta(hours=1)) :].any()
    assert coverage["last_published_open"] == (end - timedelta(hours=1)).isoformat()
    none, coverage = _load(rows, window, "BBBUSDT")
    assert none is None and coverage["first_listed_open"] is None


def test_a_perp_gap_is_absent_and_a_declared_halt_marks_the_hours_halted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    gap = _ms(window.load_start + timedelta(days=10, hours=3))
    rows.skip["AAAUSDT"] = {gap, gap + HOUR_MS}
    leg, _ = _load(rows, window, "AAAUSDT")
    assert leg is not None
    slot = 10 * 24 + 3
    assert not leg.present[slot] and not leg.present[slot + 1] and not leg.halted.any()
    limit = window.last_row_close
    guarded = data.StageRowSource(
        rows, window=window, limit_close=limit, symbols=["AAAUSDT"]
    )
    halt = (
        window.load_start + timedelta(days=10, hours=3),
        window.load_start + timedelta(days=10, hours=5),
    )
    leg, coverage = driver.load_perp(
        guarded,
        guarded,
        "AAAUSDT",
        window=window,
        limit_close=limit,
        halt_intervals=[halt],
    )
    assert leg is not None and leg.halted[slot] and leg.halted[slot + 1]
    assert coverage["halted_hours"] == 2


def test_a_listing_hidden_by_an_unpublished_month_with_daily_archives_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    listed = window.load_start + timedelta(
        days=45
    )  # not listed in the first two months
    rows = Rows({"AAAUSDT": (listed, None)})
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None and coverage["unpublished_leading_months"] == 1
    month_end = int(datetime(2021, 2, 1, tzinfo=UTC).timestamp() * 1_000)
    rows.daily.add(("AAAUSDT", month_end - 24 * HOUR_MS))
    with pytest.raises(data.StageDataError, match="listing would silently move"):
        _load(rows, window, "AAAUSDT")


def test_duplicate_or_unordered_rows_are_an_integrity_fault(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)

    class Dup(Rows):
        def read_klines(self, *args: Any, **kwargs: Any) -> list[KlineRow]:
            rows = super().read_klines(*args, **kwargs)
            return [*rows, rows[0]] if rows else rows

    with pytest.raises(driver.CarryScreenHalt):
        _load(Dup({"AAAUSDT": (window.load_start, None)}), window, "AAAUSDT")


def test_exchange_rules_fall_back_for_symbols_absent_from_the_snapshot() -> None:
    rules, fallback = driver.load_perp_rules(
        perp_payload(["AAAUSDT", "BBBUSDT"], absent=["BBBUSDT"]),
        ["AAAUSDT", "BBBUSDT", "CCCUSDT"],
    )
    assert fallback == ["BBBUSDT", "CCCUSDT"]
    assert (rules["AAAUSDT"].lot_size, rules["AAAUSDT"].min_notional) == (0.001, 5.0)
    assert rules["BBBUSDT"].lot_size == protocol.FALLBACK_LOT_SIZE
    assert rules["BBBUSDT"].min_notional == protocol.FALLBACK_MIN_NOTIONAL
    only_fallback, missing = driver.load_perp_rules(perp_payload([]), ["AAAUSDT"])
    assert missing == ["AAAUSDT"] and only_fallback["AAAUSDT"].lot_size > 0.0


def test_the_stage_symbols_are_the_roster_the_candidates_and_the_forward_listing(
    tmp_path: Path,
) -> None:
    assert driver.stage_symbols("DEV", None) == tuple(sorted(protocol.DEV_ROSTER))
    candidates = driver.stage_symbols("DISJOINT", None)
    assert len(candidates) == 441 and candidates == tuple(sorted(candidates))
    assert (
        len(driver.stage_symbols("FORWARD", None)) == 684
    )  # the packaged, pinned file
    listing = symbol_file(tmp_path, ["ZZZUSDT", "AAAUSDT"])
    assert driver.stage_symbols("FORWARD", listing) == ("AAAUSDT", "ZZZUSDT")
    text = tmp_path / "listing.txt"
    text.write_text("BBBUSDT\nAAAUSDT\n", encoding="utf-8")
    assert driver.read_symbol_file(text)[0] == ("AAAUSDT", "BBBUSDT")
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"x": 1}), encoding="utf-8")
    with pytest.raises(ValueError):
        driver.read_symbol_file(bad)


# ================================================================== coverage


def test_coverage_is_result_blind_and_reports_gaps_funding_and_exchange_wide_hours(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS[:4])
    rows = Rows(
        listings(
            window,
            SYMBOLS[:4],
            last={SYMBOLS[3]: window.load_start + timedelta(days=30)},
        )
    )
    shared = _ms(window.load_start + timedelta(days=20, hours=6))
    for symbol in SYMBOLS[:4]:
        rows.skip.setdefault(symbol, set()).update({shared, shared + HOUR_MS})
    rows.skip[SYMBOLS[0]].add(
        _ms(window.load_start + timedelta(days=40))
    )  # only one symbol
    code, out, err = tool(
        tmp_path,
        ["coverage", "--stage", "DISJOINT", "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:4],
        with_rules=False,
    )
    assert code == driver.EXIT_OK, err
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    assert document["schema_version"] == driver.COVERAGE_SCHEMA
    assert (
        document["stage"] == "DISJOINT"
        and "no universe, signal, replay" in document["not_computed"]
    )
    assert document["data_authorization"]["url"] == DATA_AUTH
    wide = document["exchange_wide_perp_gaps"]
    assert wide["hours"] == 2 and wide["ranges"][0]["hours"] == 2
    entry = document["symbols"][SYMBOLS[0]]["usds_m"]
    assert entry["missing_hours"] == 3 and entry["funding"]["events"] > 0
    # a delisted symbol's hours after its last row are the archive's end, not a gap
    ended = document["symbols"][SYMBOLS[3]]["usds_m"]
    assert ended["missing_hours"] == 2 and ended["trailing_hours_after_last_row"] > 0
    summary = json.loads(out)
    assert summary["exchange_wide_perp_gap_hours"] == 2
    assert summary["legs_with_read_errors"] == []
    assert not (tmp_path / "out").exists()
    # write-once
    code, _, err = tool(
        tmp_path,
        ["coverage", "--stage", "DISJOINT", "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:4],
        with_rules=False,
    )
    assert code == driver.EXIT_REFUSED and "write-once" in err


def test_coverage_refuses_forward_only_flags_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    code, _, err = tool(
        tmp_path,
        [
            "coverage",
            "--stage",
            "DEV",
            "--through",
            "2022-03-20T00:00:00Z",
            "--output-dir",
            str(tmp_path / "cov"),
        ],
        gh=Gh(),
        now=NOW,
        rows=rows,
        with_rules=False,
    )
    assert code == driver.EXIT_REFUSED and "FORWARD only" in err


def _coverage_document(**over: Any) -> dict[str, Any]:
    def leg(missing: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "errors": [],
            "unread_chunks": [],
            "leading_unpublished_months_with_daily_archives": [],
            "missing_ranges": missing,
        }

    shared = {"first_missing_open": "2021-02-01T10:00:00+00:00", "hours": 2}
    own = {"first_missing_open": "2021-03-01T00:00:00+00:00", "hours": 1}
    document: dict[str, Any] = {
        "schema_version": driver.COVERAGE_SCHEMA,
        "stage": "DISJOINT",
        "protocol_digest": "d" * 64,
        "code_commit": COMMIT,
        "symbols": {
            "AUSDT": {"usds_m": leg([shared, own])},
            "BUSDT": {"usds_m": leg([shared])},
        },
        "exchange_wide_perp_gaps": {"ranges": [shared]},
    }
    document.update(over)
    return document


def test_halts_from_coverage_declares_only_exchange_wide_ranges_and_lists_what_remains(
    tmp_path: Path,
) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_coverage_document()), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err
    )
    assert code == driver.EXIT_OK, err.getvalue()
    result = json.loads(out.getvalue())
    assert result["declared_halts"] == [
        ["2021-02-01T10:00:00Z", "2021-02-01T12:00:00Z"]
    ]
    assert result["declared_hours"] == 2
    assert len(result["coverage_json_sha256"]) == 64
    assert list(result["gaps_that_remain_and_make_bars_unavailable"]) == [
        "AUSDT:usds_m"
    ]


@pytest.mark.parametrize(
    "document",
    [
        {"schema_version": "other"},
        {"stage": "DEV"},
        {
            "symbols": {
                "AUSDT": {
                    "usds_m": {
                        "errors": [{"phase": "listing"}],
                        "unread_chunks": [],
                        "leading_unpublished_months_with_daily_archives": [],
                    }
                }
            }
        },
        {
            "symbols": {
                "AUSDT": {
                    "usds_m": {
                        "errors": [],
                        "unread_chunks": [],
                        "leading_unpublished_months_with_daily_archives": ["2021-01"],
                    }
                }
            }
        },
    ],
)
def test_halts_from_coverage_refuses_the_wrong_stage_errors_and_hidden_listings(
    tmp_path: Path, document: dict[str, Any]
) -> None:
    path = tmp_path / "coverage.json"
    path.write_text(json.dumps(_coverage_document(**document)), encoding="utf-8")
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err
    )
    assert code == driver.EXIT_REFUSED and out.getvalue() == ""


def test_the_forward_embargo_policy_is_bound_to_the_owner_relay_and_dev_is_enforced() -> (
    None
):
    policy = driver.embargo_for(datetime(2026, 8, 1, tzinfo=UTC), stage="DISJOINT")
    assert type(policy).__name__ == "EmbargoPolicy"
    with pytest.raises(driver.StageEmbargoError, match="only FORWARD"):
        driver.embargo_for(datetime(2026, 8, 2, tzinfo=UTC), stage="DISJOINT")
    forward = driver.embargo_for(datetime(2026, 9, 1, tzinfo=UTC), stage="FORWARD")
    assert isinstance(forward, driver.ForwardEmbargoPolicy)
    assert forward.warmup_authorization_url == DATA_AUTH


# ===================================================== B1: funding months, B2, B3


def test_an_unpublished_funding_month_inside_the_listing_refuses_naming_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window, protocol.DEV_ROSTER))
    rows.funding_missing["BTCUSDT"] = {(2022, 2)}
    code, _, err = tool(
        tmp_path, run_argv(tmp_path, "DEV"), gh=Gh(), now=NOW, rows=rows
    )
    assert code == driver.EXIT_REFUSED
    assert "BTCUSDT" in err and "2022-02" in err and "unpublished" in err
    assert not (tmp_path / "out" / driver.RUN_RECEIPT_NAME).exists()


def test_funding_months_without_any_kline_bar_are_admissible_and_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    ended = window.load_start + timedelta(days=45)  # delisted mid February
    rows = Rows(listings(window, last={SYMBOLS[2]: ended}))
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", "--dev-result", str(dev_result(tmp_path))),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    records = receipt["unpublished_funding_months"]
    assert records and all(
        r["admissible"] and r["symbol"] == SYMBOLS[2] for r in records
    )
    assert {r["month"] for r in records} == {"2021-03", "2021-04"}


def test_coverage_records_an_inadmissible_funding_month_without_refusing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS[:3])
    rows = Rows(listings(window, SYMBOLS[:3]))
    rows.funding_missing[SYMBOLS[1]] = {(2021, 3)}
    code, _, err = tool(
        tmp_path,
        ["coverage", "--stage", "DISJOINT", "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:3],
        with_rules=False,
    )
    assert code == driver.EXIT_OK, err
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    funding = document["symbols"][SYMBOLS[1]]["usds_m"]["funding"]
    assert funding["unpublished_funding_months"] == [
        {"symbol": SYMBOLS[1], "month": "2021-03", "admissible": False}
    ]


def test_forward_is_refused_while_the_symbol_file_is_unpinned_or_differs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = data.frozen_window(protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    listing = symbol_file(tmp_path)

    def argv_now() -> list[str]:
        # the pin is part of the protocol digest, so the DISJOINT decision is re-written
        decision_file = disjoint_decision(tmp_path)
        return run_argv(
            tmp_path,
            "FORWARD",
            "--interim",
            "--forward-symbols",
            str(listing),
            "--disjoint-decision",
            str(decision_file),
        )

    monkeypatch.setattr(protocol, "FORWARD_SYMBOLS_SHA256", None)
    argv = argv_now()
    code, _, err = tool(tmp_path, argv, gh=Gh(), now=NOW, rows=rows)
    assert code == driver.EXIT_REFUSED and "not pinned" in err and rows.requests == []
    cov = ["coverage", "--stage", "FORWARD", "--forward-symbols", str(listing)]
    code, _, err = tool(
        tmp_path,
        [*cov, "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        with_rules=False,
    )
    assert code == driver.EXIT_REFUSED and "not pinned" in err
    monkeypatch.setattr(protocol, "FORWARD_SYMBOLS_SHA256", "a" * 64)
    argv = argv_now()
    code, _, err = tool(tmp_path, argv, gh=Gh(), now=NOW, rows=rows)
    assert code == driver.EXIT_REFUSED and "differs from the pinned" in err
    code, _, err = tool(
        tmp_path,
        [*cov, "--output-dir", str(tmp_path / "cov2")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        with_rules=False,
    )
    assert code == driver.EXIT_REFUSED and "differs from the pinned" in err
    assert rows.requests == [] and not no_run.calls
    assert "forward_symbols_sha256" in json.dumps(protocol.protocol_payload())


def test_disjoint_needs_the_single_use_confirmation_and_records_the_run_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    rows = Rows(listings(window))
    flags = ("--dev-result", str(dev_result(tmp_path)))
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", "--no-confirm", *flags),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "--confirm-single-use" in err
    assert rows.requests == []
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DEV", "--confirm-single-use"),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "belongs to DISJOINT" in err
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DISJOINT", *flags, name="first-disjoint-run"),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
    result = json.loads(
        (tmp_path / "first-disjoint-run" / runner.DECISION_NAME).read_text("utf-8")
    )
    assert result["output_dir_name"] == "first-disjoint-run"
    assert result["single_use_stage"] is True
    assert result["protocol_digest"] == protocol.protocol_digest()


def test_forward_defaults_to_the_packaged_pinned_symbol_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    packaged = protocol.forward_symbols_path()
    assert packaged.is_file()
    assert hashlib.sha256(packaged.read_bytes()).hexdigest() == (
        "48656a5dfb73491dacd2234490ba7c34c79b5692fcc91812a15095434900ba66"
    )
    real_pin = protocol.FORWARD_SYMBOLS_SHA256
    assert real_pin == hashlib.sha256(packaged.read_bytes()).hexdigest()
    symbols = driver.stage_symbols("FORWARD", None)
    assert len(symbols) == 684 and symbols == tuple(sorted(symbols))
    assert all(protocol.forward_symbol_allowed(sym) for sym in symbols)
    # an explicit path must hash-match the pin
    other = tmp_path / "other.json"
    other.write_text(json.dumps(["AAAUSDT", "BBBUSDT"]), encoding="utf-8")
    monkeypatch.setattr(protocol, "FORWARD_SYMBOLS_SHA256", real_pin)
    with pytest.raises(ValueError, match="differs from the pinned"):
        driver.stage_symbols("FORWARD", other)
    assert driver.stage_symbols("FORWARD", packaged) == symbols


# ============================================== Amendment 2: zombie tail, REST funding


class RestStub:
    """Injected REST fetcher (no HTTP): 8-hourly events, per-symbol overrides."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int, int]] = []
        self.empty: set[str] = set()
        self.fail: set[str] = set()
        self.times: dict[str, list[int]] = {}

    def __call__(self, symbol: str, start_ms: int, end_ms: int) -> bytes:
        self.calls.append((symbol, start_ms, end_ms))
        if symbol in self.fail:
            raise driver.RestFundingError(f"{symbol}: the REST funding request failed")
        if symbol in self.empty:
            stamps: list[int] = []
        elif symbol in self.times:
            stamps = [t for t in self.times[symbol] if start_ms <= t <= end_ms]
        else:
            first = -(-start_ms // (8 * HOUR_MS)) * 8 * HOUR_MS
            stamps = list(range(first, end_ms + 1, 8 * HOUR_MS))
        page = stamps[: driver.REST_PAGE_LIMIT]
        return json.dumps(
            [
                {
                    "symbol": symbol,
                    "fundingTime": t,
                    "fundingRate": "0.00010000",
                    "markPrice": "1.0",
                }
                for t in page
            ]
        ).encode("utf-8")


def _rest(tmp_path: Path, fetch: Any = None) -> driver.RestFundingSource:
    return driver.RestFundingSource(tmp_path / "rest" / "fundingRate", fetch)


def _load_with(
    rows: Rows,
    window: data.StageWindow,
    symbol: str,
    rest: driver.RestFundingSource | None,
) -> tuple[data.PerpHourly | None, dict[str, object]]:
    limit = window.last_row_close
    guarded = data.StageRowSource(
        rows, window=window, limit_close=limit, symbols=[symbol]
    )
    return driver.load_perp(
        guarded,
        guarded,
        symbol,
        window=window,
        limit_close=limit,
        halt_intervals=(),
        rest=rest,
    )


def test_split_zombie_tail_drops_only_the_trailing_zero_volume_run() -> None:
    def bar(volume: str, quote: str = "0") -> Any:
        return type(
            "R", (), {"volume": Decimal(volume), "quote_volume": Decimal(quote)}
        )()

    rows = [bar("1"), bar("0"), bar("2"), bar("0"), bar("0")]
    kept, dropped = driver.split_zombie_tail(rows)
    assert kept == rows[:3] and dropped == 2
    assert driver.split_zombie_tail([bar("0"), bar("0")]) == ([], 2)
    assert driver.split_zombie_tail([bar("0", "5")])[1] == 0  # quote volume counts
    assert driver.split_zombie_tail([]) == ([], 0)


def test_a_zombie_tail_is_dropped_and_the_forced_close_keys_on_the_effective_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    dead = window.load_start + timedelta(days=50)
    rows.zombie_from["AAAUSDT"] = _ms(dead)
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None
    slot = 50 * 24
    assert leg.present[:slot].all() and not leg.present[slot:].any()
    assert coverage["effective_end"] == dead.isoformat()
    assert coverage["zombie_tail_bars"] == leg.n_hours - slot
    assert coverage["rows"] == slot
    # the forced-close schedule is derived from the leg's last active 4h bar
    n_rows = leg.n_hours // protocol.BAR_HOURS
    clock = (
        _ms(window.load_start) + (np.arange(n_rows, dtype=np.int64) + 1) * data.BAR_MS
    )
    active = data._aggregate_4h(leg, clock_ms=clock, grid_ms=data.BAR_MS).active
    last = int(np.flatnonzero(active)[-1])
    assert last == slot // protocol.BAR_HOURS - 1
    forced = policy.XsInputs.forced_rows(active[:, None])[:, 0]
    assert int(np.flatnonzero(forced)[-1]) == last
    assert int(forced.sum()) == protocol.FORCED_CLOSE_LEAD_BARS + 1


def test_interior_zero_volume_bars_are_kept(monkeypatch: pytest.MonkeyPatch) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    hole = _ms(window.load_start + timedelta(days=10))
    rows.zero_hours["AAAUSDT"] = {hole, hole + HOUR_MS}
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None and coverage["zombie_tail_bars"] == 0
    assert leg.present[10 * 24] and leg.present[10 * 24 + 1]
    assert leg.quote_volume[10 * 24] == 0.0
    assert coverage["effective_end"] == window.last_row_close.isoformat()


def test_a_leg_of_only_zero_volume_bars_is_not_listed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.zombie_from["AAAUSDT"] = 0
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is None and coverage["effective_end"] is None
    assert coverage["zombie_tail_bars"] > 0


def test_an_unpublished_funding_month_with_only_zombie_bars_no_longer_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.zombie_from["AAAUSDT"] = _ms(window.load_start + timedelta(days=45))
    rows.funding_missing["AAAUSDT"] = {(2021, 3), (2021, 4)}
    leg, coverage = _load(rows, window, "AAAUSDT")  # no REST source, no network
    assert leg is not None
    records = coverage["unpublished_funding_months"]
    assert [r["month"] for r in records] == ["2021-03", "2021-04"]  # type: ignore[union-attr]
    assert all(r["admissible"] for r in records)  # type: ignore[union-attr]
    # the same unpublished month with real trading still refuses without the fallback
    live = Rows({"AAAUSDT": (window.load_start, None)})
    live.funding_missing["AAAUSDT"] = {(2021, 3)}
    with pytest.raises(data.StageDataError, match="2021-03"):
        _load(live, window, "AAAUSDT")


def _zero_run(
    rows: Rows, symbol: str, start: datetime, hours: int, *, keep_funding: bool
) -> None:
    first = _ms(start)
    stamps = {first + i * HOUR_MS for i in range(hours)}
    rows.zero_hours.setdefault(symbol, set()).update(stamps)
    if not keep_funding:
        rows.funding_skip.setdefault(symbol, set()).update(stamps)


def test_an_interior_zero_volume_run_without_funding_becomes_missing_hours(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    start = window.load_start + timedelta(days=10)
    _zero_run(rows, "AAAUSDT", start, 48, keep_funding=False)
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None
    slot = 10 * 24
    assert not leg.present[slot : slot + 48].any() and not leg.halted.any()
    assert leg.present[slot - 1] and leg.present[slot + 48]
    assert coverage["relisting_gap_runs"] == [
        [start.isoformat(), (start + timedelta(hours=48)).isoformat(), 48]
    ]
    assert coverage["zombie_tail_bars"] == 0


def test_a_zero_volume_month_of_a_relisted_perp_no_longer_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    march = datetime(2021, 3, 1, tzinfo=UTC)
    _zero_run(rows, "AAAUSDT", march, 31 * 24, keep_funding=False)
    rows.funding_missing["AAAUSDT"] = {(2021, 3)}
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None
    records = coverage["unpublished_funding_months"]
    assert [r["month"] for r in records] == ["2021-03"]  # type: ignore[union-attr]
    assert records[0]["admissible"]  # type: ignore[index]
    # a month with a positive-volume bar still needs the fallback
    live = Rows({"AAAUSDT": (window.load_start, None)})
    live.funding_missing["AAAUSDT"] = {(2021, 3)}
    with pytest.raises(data.StageDataError, match="2021-03"):
        _load(live, window, "AAAUSDT")


def test_a_held_name_is_force_closed_at_the_start_of_a_relisting_gap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    start = window.load_start + timedelta(days=60)
    _zero_run(rows, "AAAUSDT", start, 48, keep_funding=False)
    leg, _ = _load(rows, window, "AAAUSDT")
    assert leg is not None
    n_rows = leg.n_hours // protocol.BAR_HOURS
    clock = (
        _ms(window.load_start) + (np.arange(n_rows, dtype=np.int64) + 1) * data.BAR_MS
    )
    active = data._aggregate_4h(leg, clock_ms=clock, grid_ms=data.BAR_MS).active
    first_gap = 60 * 24 // protocol.BAR_HOURS
    assert active[first_gap - 1] and not active[first_gap]
    forced = policy.XsInputs.forced_rows(active[:, None])[:, 0]
    assert forced[first_gap - 1] and not forced[first_gap]
    assert forced[first_gap - 1 - protocol.FORCED_CLOSE_LEAD_BARS]


def test_short_zero_volume_runs_are_kept_and_long_runs_with_funding_are_converted(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    short = window.load_start + timedelta(days=5)
    funded = window.load_start + timedelta(days=20)
    _zero_run(rows, "AAAUSDT", short, 10, keep_funding=True)
    _zero_run(rows, "AAAUSDT", funded, 48, keep_funding=True)
    leg, coverage = _load(rows, window, "AAAUSDT")
    assert leg is not None
    assert coverage["relisting_gap_runs"] == [
        [funded.isoformat(), (funded + timedelta(hours=48)).isoformat(), 48]
    ]
    assert leg.present[5 * 24 : 5 * 24 + 10].all()  # 10 h run kept
    slot = 20 * 24
    assert not leg.present[slot : slot + 48].any()
    # the fake archive publishes an event every 8 h; those strictly inside are dropped
    start_ms, end_ms = _ms(funded), _ms(funded + timedelta(hours=48))
    assert not np.any((leg.funding_ms > start_ms) & (leg.funding_ms < end_ms))
    dropped = coverage["dropped_funding_events_in_gaps"]
    assert (
        dropped == 5
    )  # events at +8, +16, +24, +32, +40 h; +0 and +48 are at the edges
    assert coverage["funding_events"] == len(leg.funding_ms)


def test_relisting_runs_need_24_hours_and_ignore_funding() -> None:
    hour = HOUR_MS
    bars = [(i * hour, i not in range(10, 40)) for i in range(60)]
    assert driver.relisting_runs(bars) == [(10 * hour, 40 * hour)]
    short = [(i * hour, i not in range(10, 33)) for i in range(60)]
    assert driver.relisting_runs(short) == []
    broken = [(i * hour, i not in range(10, 40) or i == 20) for i in range(60)]
    assert driver.relisting_runs([b for b in broken if b[0] != 25 * hour]) == []
    events = [(t * hour, 0.0001) for t in (10, 11, 25, 39, 40)]
    kept, dropped = driver.drop_funding_in_runs(events, [(10 * hour, 40 * hour)])
    assert dropped == 3 and [t for t, _ in kept] == [10 * hour, 40 * hour]


def test_an_unpublished_traded_month_is_filled_from_rest_and_cached_raw(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.funding_missing["AAAUSDT"] = {(2021, 2)}
    stub = RestStub()
    rest = _rest(tmp_path, stub)
    leg, coverage = _load_with(rows, window, "AAAUSDT", rest)
    assert leg is not None
    feb = datetime(2021, 2, 1, tzinfo=UTC)
    mar = datetime(2021, 3, 1, tzinfo=UTC)
    in_feb = [t for t in leg.funding_ms if _ms(feb) <= t < _ms(mar)]
    assert len(in_feb) == 28 * 3 and np.all(np.diff(leg.funding_ms) > 0)
    assert stub.calls == [("AAAUSDT", _ms(feb), _ms(mar) - 1)]
    record = coverage["unpublished_funding_months"][0]  # type: ignore[index]
    assert record["admissible"] and record["rest_events"] == 28 * 3
    cache = (
        tmp_path
        / "rest"
        / "fundingRate"
        / "AAAUSDT"
        / "AAAUSDT-fundingRate-2021-02.json"
    )
    (evidence,) = rest.evidence()
    assert evidence["sha256"] == hashlib.sha256(cache.read_bytes()).hexdigest()
    assert evidence["events"] == 28 * 3 and evidence["source"] == "network"


def test_rest_paginates_by_start_time(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(driver, "REST_PAGE_LIMIT", 10)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.funding_missing["AAAUSDT"] = {(2021, 2)}
    stub = RestStub()
    leg, _ = _load_with(rows, window, "AAAUSDT", _rest(tmp_path, stub))
    assert leg is not None
    assert len(stub.calls) == 9  # 84 events, 10 per page
    assert stub.calls[1][1] > stub.calls[0][1]
    feb = datetime(2021, 2, 1, tzinfo=UTC)
    mar = datetime(2021, 3, 1, tzinfo=UTC)
    assert len([t for t in leg.funding_ms if _ms(feb) <= t < _ms(mar)]) == 28 * 3


def test_an_empty_rest_month_is_admissible_only_when_the_gap_check_passes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    stub = RestStub()
    stub.empty.add("AAAUSDT")
    # listed at the end of January: the first funding event is on February 1st
    late = Rows({"AAAUSDT": (datetime(2021, 1, 31, 12, tzinfo=UTC), None)})
    late.funding_missing["AAAUSDT"] = {(2021, 1)}
    leg, coverage = _load_with(late, window, "AAAUSDT", _rest(tmp_path, stub))
    assert leg is not None
    record = coverage["unpublished_funding_months"][0]  # type: ignore[index]
    assert record["admissible"] and record["rest_events"] == 0
    # listed early in the month: 27 days without an event is not continuous
    early = Rows({"AAAUSDT": (datetime(2021, 1, 5, tzinfo=UTC), None)})
    early.funding_missing["AAAUSDT"] = {(2021, 1)}
    with pytest.raises(data.StageDataError, match="gap of"):
        _load_with(early, window, "AAAUSDT", _rest(tmp_path / "b", stub))


def test_without_network_and_cache_the_traded_unpublished_month_refuses(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.funding_missing["AAAUSDT"] = {(2021, 2)}
    with pytest.raises(data.StageDataError, match="not cached"):
        _load_with(rows, window, "AAAUSDT", _rest(tmp_path, None))


def test_a_cached_rest_month_is_reused_offline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.funding_missing["AAAUSDT"] = {(2021, 2)}
    online = _rest(tmp_path, RestStub())
    first, _ = _load_with(rows, window, "AAAUSDT", online)
    offline = _rest(tmp_path, None)
    second, _ = _load_with(rows, window, "AAAUSDT", offline)
    assert first is not None and second is not None
    assert np.array_equal(first.funding_ms, second.funding_ms)
    (before,), (after,) = online.evidence(), offline.evidence()
    assert after["source"] == "cache" and after["sha256"] == before["sha256"]


def test_malformed_cached_rest_funding_is_refused_without_mutation(
    tmp_path: Path,
) -> None:
    month_start = datetime(2021, 2, 1, tzinfo=UTC)
    next_month = datetime(2021, 3, 1, tzinfo=UTC)
    stamp = _ms(month_start) + 8 * HOUR_MS
    cache = (
        tmp_path
        / "rest"
        / "fundingRate"
        / "AAAUSDT"
        / "AAAUSDT-fundingRate-2021-02.json"
    )
    cache.parent.mkdir(parents=True)
    cache.write_bytes(
        json.dumps(
            [
                {
                    "symbol": "AAAUSDT",
                    "fundingTime": stamp + 0.75,
                    "fundingRate": "0.00010000",
                }
            ]
        ).encode()
    )
    before = cache.read_bytes()
    source = _rest(tmp_path, None)
    with pytest.raises(driver.RestFundingError, match="malformed"):
        source.month("AAAUSDT", _ms(month_start), _ms(next_month), _ms(next_month))
    assert cache.read_bytes() == before
    assert source.evidence() == []


@pytest.mark.parametrize(
    "mutate",
    [
        lambda e: [{**e[0], "symbol": "OTHERUSDT"}, *e[1:]],
        lambda e: [e[1], e[0], *e[2:]],
        lambda e: [{**e[0], "fundingTime": 0}, *e[1:]],
        lambda e: [*e[:-1], {**e[-1], "fundingTime": 1_617_235_200_000}],
        lambda e: [{**e[0], "fundingRate": "nan"}, *e[1:]],
        # Binance REST requires integer millisecond stamps and numeric string rates.
        lambda e: [{**e[0], "fundingTime": float(e[0]["fundingTime"])}, *e[1:]],
        lambda e: [{**e[0], "fundingTime": True}, *e[1:]],
        lambda e: [{**e[0], "fundingTime": str(e[0]["fundingTime"])}, *e[1:]],
        lambda e: [{**e[0], "fundingTime": float("inf")}, *e[1:]],
        lambda e: [{**e[0], "fundingRate": True}, *e[1:]],
        lambda e: [{**e[0], "fundingRate": 10**1000}, *e[1:]],
        lambda e: {"code": -1121},
    ],
)
def test_invalid_rest_responses_are_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutate: Any,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    rows = Rows({"AAAUSDT": (window.load_start, None)})
    rows.funding_missing["AAAUSDT"] = {(2021, 2)}
    good = RestStub()

    def bad(symbol: str, start_ms: int, end_ms: int) -> bytes:
        return json.dumps(mutate(json.loads(good(symbol, start_ms, end_ms)))).encode()

    with pytest.raises(data.StageDataError, match="2021-02"):
        _load_with(rows, window, "AAAUSDT", _rest(tmp_path, bad))
    assert not list((tmp_path / "rest").rglob("*.json"))  # nothing invalid is cached


def test_coverage_summary_names_inadmissible_rest_and_zombie_months(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS[:3])
    rows = Rows(listings(window, SYMBOLS[:3]))
    rows.zombie_from[SYMBOLS[0]] = _ms(window.load_start + timedelta(days=45))
    rows.funding_missing[SYMBOLS[0]] = {(2021, 3), (2021, 4)}  # zombie months
    rows.funding_missing[SYMBOLS[1]] = {(2021, 2)}  # traded, REST fills it
    rows.funding_missing[SYMBOLS[2]] = {(2021, 3)}  # traded, REST fails
    stub = RestStub()
    stub.fail.add(SYMBOLS[2])
    code, out, err = tool(
        tmp_path,
        [
            "coverage",
            "--stage",
            "DISJOINT",
            "--allow-network",
            "--output-dir",
            str(tmp_path / "cov"),
        ],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:3],
        with_rules=False,
        rest_fetch=stub,
    )
    assert code == driver.EXIT_OK, err
    summary = json.loads(out)
    assert summary["zombie_tail_legs"] == 1
    assert summary["relisting_gap_legs"] == 0
    assert summary["rest_funding_months"] == 1
    assert summary["inadmissible_unpublished_funding_months"] == {
        "count": 1,
        "months": [f"{SYMBOLS[2]}:2021-03"],
    }
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    zombie = document["symbols"][SYMBOLS[0]]["usds_m"]
    assert zombie["zombie_tail_bars"] > 0 and zombie["effective_end"]
    assert zombie["trailing_hours_after_last_row"] == zombie["zombie_tail_bars"]
    assert document["rest_funding_evidence"][0]["symbol"] == SYMBOLS[1]
    # without the network the traded month is inadmissible instead of REST-filled
    code, out, err = tool(
        tmp_path,
        [
            "coverage",
            "--stage",
            "DISJOINT",
            "--output-dir",
            str(tmp_path / "cov2"),
        ],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:3],
        with_rules=False,
    )
    assert code == driver.EXIT_OK, err
    summary = json.loads(out)
    assert summary["inadmissible_unpublished_funding_months"]["months"] == [
        f"{SYMBOLS[2]}:2021-03"
    ]
    assert summary["rest_funding_months"] == 1  # SYMBOLS[1] is served from the cache


def test_a_disjoint_run_records_zombie_legs_and_rest_months_in_the_data_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS)
    rows = Rows(listings(window))
    rows.zombie_from[SYMBOLS[2]] = _ms(window.load_start + timedelta(days=45))
    rows.funding_missing[SYMBOLS[2]] = {(2021, 3), (2021, 4)}
    rows.funding_missing[SYMBOLS[3]] = {(2021, 2)}
    code, _, err = tool(
        tmp_path,
        run_argv(
            tmp_path,
            "DISJOINT",
            "--dev-result",
            str(dev_result(tmp_path)),
            "--allow-network",
        ),
        gh=Gh(),
        now=NOW,
        rows=rows,
        rest_fetch=RestStub(),
    )
    assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["zombie_tail"][SYMBOLS[2]]["zombie_tail_bars"] > 0
    assert receipt["zombie_tail"][SYMBOLS[2]]["effective_end"]
    (month,) = receipt["rest_funding"]["months"]
    assert month["symbol"] == SYMBOLS[3] and month["month"] == "2021-02"
    assert receipt["rest_funding"]["digest"] and len(month["sha256"]) == 64


def test_coverage_counts_relisting_gap_legs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_DISJOINT)
    monkeypatch.setattr(protocol, "load_candidates", lambda *a, **k: SYMBOLS[:2])
    rows = Rows(listings(window, SYMBOLS[:2]))
    _zero_run(
        rows, SYMBOLS[0], datetime(2021, 3, 1, tzinfo=UTC), 31 * 24, keep_funding=False
    )
    rows.funding_missing[SYMBOLS[0]] = {(2021, 3)}
    code, out, err = tool(
        tmp_path,
        ["coverage", "--stage", "DISJOINT", "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        symbols=SYMBOLS[:2],
        with_rules=False,
    )
    assert code == driver.EXIT_OK, err
    summary = json.loads(out)
    assert summary["relisting_gap_legs"] == 1
    assert summary["inadmissible_unpublished_funding_months"]["count"] == 0
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    entry = document["symbols"][SYMBOLS[0]]["usds_m"]
    assert entry["relisting_gap_runs"][0][2] == 31 * 24
    assert entry["missing_hours"] == 31 * 24
