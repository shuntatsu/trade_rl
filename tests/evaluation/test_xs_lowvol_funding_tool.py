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
from trade_rl.evaluation.xs_lowvol_funding import data, protocol, runner
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


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


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
            quote = (
                150_000_000.0
                + (i % 7) * 1_000_000.0
                + 1_000.0 * (sum(map(ord, symbol)) % 50)
            )
            rows.append(
                KlineRow(
                    open_time_ms=open_ms,
                    open=Decimal(f"{o:.8f}"),
                    high=Decimal(f"{max(o, c) * 1.001:.8f}"),
                    low=Decimal(f"{min(o, c) * 0.999:.8f}"),
                    close=Decimal(f"{c:.8f}"),
                    volume=Decimal("1000000.0"),
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
    )
    return code, out.getvalue(), err.getvalue()


def run_argv(
    tmp_path: Path, stage: str, *flags: str, name: str = "out", url: str = AUTH
) -> list[str]:
    return [
        "run",
        "--stage",
        stage,
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
    code, err = go("--final", "--disjoint-decision", str(disjoint_decision(tmp_path)))
    assert code == driver.EXIT_REFUSED and "needs --forward-symbols" in err
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
        code, err = go(
            "--final",
            "--forward-symbols",
            str(symbol_file(tmp_path, bad_symbols)),
            "--disjoint-decision",
            str(good),
        )
        assert code == driver.EXIT_REFUSED and "FORWARD" in err
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
    with pytest.raises(ValueError, match="--forward-symbols"):
        driver.stage_symbols("FORWARD", None)
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
