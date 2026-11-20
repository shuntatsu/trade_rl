"""Synthetic tests of ``tools.agy_anchor_oos`` (plan, coverage, halts, run order and guards).

A deterministic fake row source serves spot and perpetual klines (with taker-buy volume) and
perpetual funding; nothing touches the network, the market store or an economic result.
"""

from __future__ import annotations

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
from tests.evaluation.test_agy_anchor_oos import LISTINGS, small_window
from tools import agy_anchor_oos as driver
from trade_rl.evaluation.agy_anchor_oos import data, decision, protocol, runner
from trade_rl.integrations.binance.carry import CarryCoverageError
from trade_rl.integrations.market_store.contracts import (
    FundingRow,
    KlineRow,
    StoreMarket,
)

HOUR_MS = 3_600_000
BAR = timedelta(hours=4)
AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1"
DATA_AUTH = protocol.DATA_AUTHORIZATION_URL
NOW = datetime(2026, 11, 20, 6, 30, tzinfo=UTC)
FINAL_NOW = datetime(2027, 5, 2, 0, 0, 1, tzinfo=UTC)
EPOCH = datetime(2019, 1, 1, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


def _ms(moment: datetime) -> int:
    return int(moment.timestamp() * 1_000)


class Rows:
    """Deterministic fake Vision source: listings per (market, symbol), optional gaps."""

    def __init__(self, listed: dict[tuple[StoreMarket, str], datetime | None]) -> None:
        self.listed = listed
        self.skip: dict[tuple[StoreMarket, str], set[int]] = {}
        self.requests: list[tuple[str, str, int, int]] = []
        self.daily: set[tuple[StoreMarket, str, int]] = set()
        self.probes: list[tuple[StoreMarket, str, int]] = []
        self.funding_skip: dict[str, set[int]] = {}
        self._cache: dict[tuple[StoreMarket, str], np.ndarray] = {}
        self._epoch = _ms(EPOCH)

    def _path(self, market: StoreMarket, symbol: str) -> np.ndarray:
        key = (market, symbol)
        if key not in self._cache:
            rng = np.random.default_rng(
                sum(map(ord, symbol)) + (7 if market is StoreMarket.USDS_M else 0)
            )
            self._cache[key] = 100.0 * np.exp(
                np.cumsum(rng.normal(0.0004, 0.01, 100_000))
            )
        return self._cache[key]

    def _first(self, market: StoreMarket, symbol: str) -> int | None:
        listed = self.listed[(market, symbol)]
        return None if listed is None else _ms(listed)

    def read_klines(
        self,
        market: StoreMarket,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: int,
    ) -> list[KlineRow]:
        assert interval == "1h"
        self.requests.append((market.value, symbol, start_ms, end_ms))
        first = self._first(market, symbol)
        if first is None or end_ms <= first:
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        path = self._path(market, symbol)
        skip = self.skip.get((market, symbol), set())
        rows: list[KlineRow] = []
        for open_ms in range(max(start_ms, first), end_ms, HOUR_MS):
            if open_ms in skip:
                continue
            i = (open_ms - self._epoch) // HOUR_MS
            o, c = float(path[i - 1]), float(path[i])
            quote = 150_000_000.0 + (i % 7) * 1_000_000.0
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
        first = self._first(market, symbol)
        if first is None:
            raise CarryCoverageError(
                f"archive is not published: {symbol}",
                {"kind": "archive_not_published", "status": 404},
            )
        skip = self.funding_skip.get(symbol, set())
        return [
            FundingRow(funding_time_ms=t, rate=Decimal("0.0001"))
            for t in range(start_ms, end_ms + 1, HOUR_MS)
            if (t // HOUR_MS) % 8 == 0 and t >= first and t not in skip
        ]

    def probe_published_day(
        self, market: StoreMarket, symbol: str, day_ms: int
    ) -> bool:
        self.probes.append((market, symbol, day_ms))
        return (market, symbol, day_ms) in self.daily


def listings(window: data.StageWindow, *, mid_bar: bool = False) -> dict[Any, Any]:
    """Row listings of the synthetic tests as datetimes (optionally one mid-bar listing)."""
    out: dict[tuple[StoreMarket, str], datetime | None] = {}
    for index, symbol in enumerate(protocol.ROSTER):
        for market, row in zip(
            (StoreMarket.SPOT, StoreMarket.USDS_M), LISTINGS[index], strict=True
        ):
            out[(market, symbol)] = (
                None if row is None else window.load_start + row * BAR
            )
    if mid_bar:
        out[(StoreMarket.USDS_M, protocol.ROSTER[5])] += timedelta(hours=1)
    return out


def _body(digest: str | None = None, *, pid: bool = True) -> str:
    lines = ["G4 authorization: agy anchor realistic OOS"]
    if pid:
        lines.append(f"protocol_id: {protocol.PROTOCOL_ID}")
    lines.append(f"protocol_digest: {digest or protocol.protocol_digest()}")
    return "\n".join(lines) + "\n"


class Gh:
    """Serves the G4 comment and the data-authorization comment by comment id."""

    def __init__(self, *, body: str | None = None, **g4_kwargs: Any) -> None:
        self.g4 = tierd_tests.FakeGh(url=AUTH, body=body or _body(), **g4_kwargs)
        self.data = tierd_tests.FakeGh(url=DATA_AUTH, body="owner relay comment\n")
        self.calls: list[list[str]] = []

    def __call__(self, args: Any) -> str:
        self.calls.append(list(args))
        target = self.data if str(args[1]).endswith("/7028416072") else self.g4
        return target(args)


def tool(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    gh: Any,
    now: datetime,
    rows: Any,
    clean: bool = True,
    with_rules: bool = True,
) -> tuple[int, str, str]:
    spot, perp = tmp_path / "spot.json", tmp_path / "perp.json"
    spot.write_bytes(tierd_tests._payload("spot"))
    perp.write_bytes(tierd_tests._payload("usds_m"))
    extra = (
        ["--exchange-info-spot", str(spot), "--exchange-info-perp", str(perp)]
        if with_rules
        else []
    )
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        [*argv, "--archive-dir", str(tmp_path / "archives"), *extra],
        stdout=out,
        stderr=err,
        state=lambda: ("c" * 40, clean),
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
    path = tmp_path / name
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
            **over,
        },
    )


def backcast_decision(
    tmp_path: Path, verdict: str | None = "STAGE_SUPPORTED", **over: Any
) -> Path:
    document: dict[str, Any] = {
        "protocol_id": protocol.PROTOCOL_ID,
        "protocol_digest": protocol.protocol_digest(),
        "stage": "BACKCAST_2020",
        "mode": "decision",
        "label": None,
        **over,
    }
    if verdict is not None:
        document["stage_verdict"] = verdict
    return _write(tmp_path, "backcast-decision.json", document)


class NoRun:
    """A fake ``run_stage`` that records its call and writes a minimal decision."""

    def __init__(self, label: str | None = None, verdict: str | None = None) -> None:
        self.calls: list[dict[str, Any]] = []
        self.label = label
        self.verdict = verdict

    def __call__(self, stage: Any, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        return {
            "label": self.label,
            "stage_verdict": self.verdict,
            "status": None,
            "failed_conditions": [],
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
        assert document["protocol_id"] == "agy_anchor_realistic_v1"
        assert document["port_source_digest"] == protocol.PORT_SOURCE_DIGEST
        assert document["parity_fixture_sha256"] == protocol.FIXTURE_SHA256
        assert document["initial_margin_rate"] == 0.4
        assert set(document["windows"]) == set(protocol.STAGES)
        assert document["window"]["stage"] == stage
        assert document["final_not_before"] == "2027-05-02T00:00:00Z"
        assert "daily_fallback" in json.dumps(document["protocol"])


def test_plan_refuses_when_the_fixture_or_the_port_binding_is_broken(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("FIXTURE_SHA256", "PORT_SOURCE_DIGEST"):
        with monkeypatch.context() as patch:
            patch.setattr(protocol, name, "9" * 64)
            out, err = io.StringIO(), io.StringIO()
            code = driver.main(
                ["plan", "--stage", "DEV"], stdout=out, stderr=err, now=lambda: NOW
            )
            assert code == driver.EXIT_REFUSED and out.getvalue() == ""
            assert "refused" in err.getvalue()


# =================================================================== run order


def test_the_run_order_dev_then_backcast_then_forward_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    small_window(monkeypatch, protocol.STAGE_DEV)
    small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(data.frozen_window(protocol.STAGE_BACKCAST)))

    def go(argv: list[str], now: datetime = NOW) -> tuple[int, str]:
        code, _, err = tool(tmp_path, argv, gh=Gh(), now=now, rows=rows)
        return code, err

    # BACKCAST_2020 needs the DEV result of this exact digest
    code, err = go(run_argv(tmp_path, "BACKCAST_2020"))
    assert code == driver.EXIT_REFUSED and "needs --dev-result" in err
    for bad in (
        dev_result(tmp_path, protocol_digest="0" * 64),
        dev_result(tmp_path, stage="BACKCAST_2020"),
        dev_result(tmp_path, mode="interim"),
        dev_result(tmp_path, protocol_id="trend_long_spot_oos_v1"),
    ):
        code, err = go(run_argv(tmp_path, "BACKCAST_2020", "--dev-result", str(bad)))
        assert code == driver.EXIT_REFUSED and "DEV result of this exact" in err, bad
    junk = tmp_path / "junk.json"
    junk.write_text("not json", encoding="utf-8")
    code, err = go(run_argv(tmp_path, "BACKCAST_2020", "--dev-result", str(junk)))
    assert code == driver.EXIT_REFUSED and "not JSON" in err
    assert not no_run.calls and not (tmp_path / "out").exists()
    # DEV takes no prior result and no mode flag
    code, err = go(run_argv(tmp_path, "DEV", "--dev-result", str(dev_result(tmp_path))))
    assert code == driver.EXIT_REFUSED and "belongs to BACKCAST_2020" in err
    code, err = go(run_argv(tmp_path, "DEV", "--interim"))
    assert code == driver.EXIT_REFUSED and "takes no" in err
    code, err = go(
        run_argv(
            tmp_path,
            "BACKCAST_2020",
            "--final",
            "--dev-result",
            str(dev_result(tmp_path)),
        )
    )
    assert code == driver.EXIT_REFUSED and "takes no" in err
    # a valid DEV result lets BACKCAST_2020 run, with the DEV result recorded in the receipt
    code, err = go(
        run_argv(tmp_path, "BACKCAST_2020", "--dev-result", str(dev_result(tmp_path)))
    )
    assert code == driver.EXIT_OK, err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["dev_result"]["status"] == "DEV_DIAGNOSTIC_NOT_A_DECISION"
    assert len(receipt["dev_result"]["sha256"]) == 64
    assert no_run.calls[-1]["backcast_verdict"] is None
    code, err = go(run_argv(tmp_path, "DEV", name="devout"))
    assert code == driver.EXIT_OK, err


def test_forward_final_needs_a_backcast_decision_of_the_same_digest_whatever_its_label(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    rows = Rows(listings(window))

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

    code, err = go()
    assert code == driver.EXIT_REFUSED and "exactly one of --final and --interim" in err
    code, err = go("--final")
    assert code == driver.EXIT_REFUSED and "needs --backcast-decision" in err
    good = backcast_decision(tmp_path)
    code, err = go("--interim", "--backcast-decision", str(good))
    assert code == driver.EXIT_REFUSED and "for --final only" in err
    code, err = go(
        "--final", "--backcast-decision", str(good), "--through", "2026-08-20T00:00:00Z"
    )
    assert code == driver.EXIT_REFUSED and "for --interim only" in err
    code, err = go(
        "--final", "--backcast-decision", str(good), "--dev-result", str(good)
    )
    assert code == driver.EXIT_REFUSED and "belongs to BACKCAST_2020" in err
    for bad in (
        backcast_decision(tmp_path, protocol_digest="0" * 64),
        backcast_decision(tmp_path, stage="DEV"),
        backcast_decision(tmp_path, mode="interim"),
        backcast_decision(tmp_path, verdict=None),
        backcast_decision(tmp_path, verdict="OOS_SUPPORTED"),
    ):
        code, err = go("--final", "--backcast-decision", str(bad))
        assert code == driver.EXIT_REFUSED, err
        assert "BACKCAST_2020 decision" in err
    assert not no_run.calls and not (tmp_path / "fwd").exists()
    # the label of the BACKCAST_2020 decision is irrelevant: any verdict lets FORWARD run
    for index, verdict in enumerate(decision.VERDICTS):
        path = backcast_decision(tmp_path, verdict=verdict)
        code, err = go("--final", "--backcast-decision", str(path), name=f"fwd{index}")
        assert code == driver.EXIT_OK, err
        assert no_run.calls[-1]["backcast_verdict"] == verdict
        assert no_run.calls[-1]["mode"] == runner.MODE_DECISION


def test_final_is_refused_before_the_scoring_instant_even_when_everything_else_is_valid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    good = backcast_decision(tmp_path)
    for when in (
        NOW,
        datetime(2027, 5, 1, 23, 59, 59, tzinfo=UTC),
        datetime(2027, 5, 2, tzinfo=UTC) - timedelta(microseconds=1),
    ):
        code, _, err = tool(
            tmp_path,
            run_argv(tmp_path, "FORWARD", "--final", "--backcast-decision", str(good)),
            gh=Gh(),
            now=when,
            rows=rows,
        )
        assert code == driver.EXIT_REFUSED and "--final is refused before" in err, when
    assert not no_run.calls and not (tmp_path / "out").exists()
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "FORWARD", "--final", "--backcast-decision", str(good)),
        gh=Gh(),
        now=datetime(2027, 5, 2, tzinfo=UTC),
        rows=rows,
    )
    assert code == driver.EXIT_OK, err


def test_a_forward_interim_needs_no_prior_stage_and_reads_only_closed_days(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    now = datetime(2026, 8, 20, 6, 30, tzinfo=UTC)
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "FORWARD", "--interim"),
        gh=Gh(),
        now=now,
        rows=rows,
    )
    assert code == driver.EXIT_OK, err
    call = no_run.calls[-1]
    assert call["mode"] == runner.MODE_INTERIM and call["backcast_verdict"] is None
    # the last row read is before the start of the current UTC day (a closed day archive)
    assert max(end for *_, end in rows.requests) <= int(
        datetime(2026, 8, 20, tzinfo=UTC).timestamp() * 1_000
    )
    assert call["run_date"] == now.date()


# ====================================================================== G4 and guards


def _backcast(tmp_path: Path, name: str = "out") -> list[str]:
    return run_argv(
        tmp_path, "BACKCAST_2020", "--dev-result", str(dev_result(tmp_path)), name=name
    )


def test_a_wrong_or_edited_g4_comment_authenticates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window))
    for gh in (
        Gh(body=_body(digest="0" * 64)),
        Gh(body=_body(pid=False)),
        Gh(login="someone-else"),
        Gh(
            variants={2: {"updated_at": "2026-11-21T00:00:00Z"}}
        ),  # edited after first fetch
        Gh(fail=True),
    ):
        code, _, err = tool(tmp_path, _backcast(tmp_path), gh=gh, now=NOW, rows=rows)
        assert code == driver.EXIT_UNAUTHENTICATED, err
        assert not no_run.calls
    assert not (tmp_path / "out").exists()


def test_a_dirty_tree_an_existing_output_and_a_bad_url_refuse_before_any_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window))
    code, _, err = tool(
        tmp_path, _backcast(tmp_path), gh=Gh(), now=NOW, rows=rows, clean=False
    )
    assert code == driver.EXIT_REFUSED and "working tree must be clean" in err
    assert not rows.requests and not (tmp_path / "out").exists()
    (tmp_path / "taken").mkdir()
    code, _, err = tool(
        tmp_path, _backcast(tmp_path, "taken"), gh=Gh(), now=NOW, rows=rows
    )
    assert code == driver.EXIT_REFUSED and "write-once" in err
    code, _, err = tool(
        tmp_path,
        run_argv(
            tmp_path,
            "BACKCAST_2020",
            "--dev-result",
            str(dev_result(tmp_path)),
            url="https://example.com/x",
        ),
        gh=Gh(),
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "authorization_url" in err
    assert not rows.requests and not no_run.calls


def test_a_stage_that_is_not_over_is_refused_before_authentication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    rows = Rows(listings(window))
    gh = Gh()
    code, _, err = tool(
        tmp_path,
        _backcast(tmp_path),
        gh=gh,
        now=window.first_decision + 30 * BAR,
        rows=rows,
    )
    assert code == driver.EXIT_REFUSED and "not over yet" in err
    assert not gh.calls and not rows.requests


def test_the_data_authorization_comment_must_not_change_between_fetches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, no_run: NoRun
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    gh = Gh()
    gh.data.variants = {2: {"updated_at": "2026-11-21T00:00:00Z"}}
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "FORWARD", "--interim"),
        gh=gh,
        now=datetime(2026, 8, 20, 6, 30, tzinfo=UTC),
        rows=rows,
    )
    assert code == driver.EXIT_UNAUTHENTICATED
    assert "data-authorization comment was edited" in err
    assert not (tmp_path / "out").exists() and not no_run.calls


def test_an_invalid_result_exits_with_the_invalid_code_and_still_writes_the_receipts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    monkeypatch.setattr(
        driver.runner, "run_stage", NoRun(label="INVALID", verdict="INVALID")
    )
    rows = Rows(listings(window))
    code, out, err = tool(tmp_path, _backcast(tmp_path), gh=Gh(), now=NOW, rows=rows)
    assert code == driver.EXIT_INVALID and "INVALID" in err
    assert json.loads(out)["stage_verdict"] == "INVALID"
    receipt = json.loads(
        (tmp_path / "out" / driver.RUN_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["label"] == "INVALID" and receipt["code_commit"] == "c" * 40


# ======================================================================= data loading


def _load(
    rows: Rows, window: data.StageWindow, **kwargs: Any
) -> tuple[data.StageData, dict[str, Any]]:
    return driver.load_stage_data(
        rows,
        rows,
        window=window,
        limit_close=window.last_row_close,
        rules=_rules(),
        **kwargs,
    )


def _rules() -> dict[str, Any]:
    return driver.tierd_driver.load_rules(
        tierd_tests._payload("spot"), tierd_tests._payload("usds_m")
    )


def test_loading_assembles_the_stage_with_taker_buy_from_the_same_klines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window, mid_bar=True))
    stage, coverage = _load(rows, window)
    assert stage.activation_rows[:7] == (
        0,
        0,
        0,
        0,
        0,
        151,
        600,
    )  # perp 5 lists mid-bar: its first full bar
    assert (
        coverage["BTCUSDT"]["spot"]["first_bar_open"] == window.load_start.isoformat()
    )
    late = coverage[protocol.ROSTER[5]]["usds_m"]
    assert late["dropped_leading_partial_hours"] == 3
    assert coverage[protocol.ROSTER[5]]["spot"]["dropped_leading_partial_hours"] == 0
    assert coverage[protocol.ROSTER[9]]["usds_m"]["first_listed_open"] is None
    assert coverage["BTCUSDT"]["usds_m"]["funding_events"] > 100
    # taker = 0.4 x quote volume on every listed bar
    listed = np.asarray(stage.dataset.volume) > 0
    np.testing.assert_allclose(
        stage.taker_buy_quote_volume[listed],
        0.4 * np.asarray(stage.dataset.volume)[listed],
        rtol=1e-6,
    )
    # perp funding reached the engine's dataset on the perp legs only
    counts = np.asarray(stage.dataset.funding_event_count)
    assert counts[:, 1].sum() > 0 and not counts[:, 0].any()
    # the guard saw spot and perp requests, never anything past the stage
    kinds = {kind for kind, *_ in rows.requests}
    assert kinds == {"spot", "usds_m"}
    assert max(end for *_, end in rows.requests) <= _ms(window.last_row_close)
    data.validate_stage_dataset(stage, window)


def test_a_perpetual_gap_is_refused_and_is_never_explained_by_a_spot_halt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    gap_hour = _ms(window.load_start + 200 * BAR)
    rows = Rows(listings(window))
    rows.skip[(StoreMarket.USDS_M, "BTCUSDT")] = {gap_hour}
    halt = (
        datetime.fromtimestamp(gap_hour / 1_000, UTC),
        datetime.fromtimestamp(gap_hour / 1_000, UTC) + timedelta(hours=1),
    )
    with pytest.raises(CarryCoverageError, match="usds_m BTCUSDT.*unexplained"):
        _load(rows, window, halt_intervals=[halt])


def test_a_spot_halt_explains_a_spot_gap_with_zero_taker_and_unexplained_gaps_refuse(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    start = window.load_start + 100 * BAR
    hours = [_ms(start) + h * HOUR_MS for h in range(4)]
    rows = Rows(listings(window))
    for symbol in protocol.ROSTER[:5]:
        rows.skip[(StoreMarket.SPOT, symbol)] = set(hours)
    with pytest.raises(CarryCoverageError, match="unexplained"):
        _load(rows, window)
    lo = datetime.fromtimestamp(hours[0] / 1_000, UTC)
    stage, _ = _load(rows, window, halt_intervals=[(lo, lo + BAR)])
    assert not np.asarray(stage.dataset.tradable)[100, 0]
    assert (
        stage.taker_buy_quote_volume[100, 0] == 0.0
        and stage.taker_buy_quote_volume[100, 1] > 0.0
    )
    assert np.asarray(stage.dataset.tradable)[100, 1]  # the perp keeps trading


def test_funding_gaps_and_duplicate_funding_events_refuse_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    stamps = {
        _ms(window.load_start) + h * HOUR_MS for h in range(0, 24 * 40) if h % 8 == 0
    }
    rows.funding_skip["BTCUSDT"] = {t for t in stamps if t // HOUR_MS % 24 < 24}
    with pytest.raises(CarryCoverageError, match="funding"):
        _load(rows, window)


def test_two_funding_events_in_one_hourly_bar_refuse_the_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    original = rows.read_funding

    def doubled(market: Any, symbol: str, start: int, end: int) -> list[FundingRow]:
        base = original(market, symbol, start, end)
        if symbol == "BTCUSDT" and len(base) > 10:
            t = base[10].funding_time_ms
            extra = [
                FundingRow(funding_time_ms=t + 30 * 60_000, rate=Decimal("0.0001")),
                FundingRow(funding_time_ms=t + 31 * 60_000, rate=Decimal("0.0001")),
            ]  # both inside one hourly bar, whichever way events are binned
            return sorted([*base, *extra], key=lambda r: r.funding_time_ms)
        return base

    rows.read_funding = doubled  # type: ignore[method-assign]
    with pytest.raises(driver.CarryIntegrityHalt, match="two funding events"):
        _load(rows, window)


def test_a_listing_hidden_by_an_unpublished_month_with_daily_archives_refuses(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    # the perp of pair 6 lists at row 600 (April): January-March are leading 404s
    day = _ms(datetime(2020, 3, 31, tzinfo=UTC))
    rows.daily.add((StoreMarket.USDS_M, protocol.ROSTER[6], day))
    with pytest.raises(data.StageDataError, match="listing would silently move"):
        _load(rows, window)
    assert (StoreMarket.USDS_M, protocol.ROSTER[6], day) in rows.probes
    # without a daily archive the leading months are simply "not listed yet"
    quiet = Rows(listings(window))
    stage, coverage = _load(quiet, window)
    assert coverage[protocol.ROSTER[6]]["usds_m"]["unpublished_leading_months"] == 3
    assert stage.activation_rows[6] == 600


def test_a_declared_halt_must_be_exchange_wide(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    halt = (("2020-02-09T02:00:00Z", "2020-02-09T03:00:00Z"),)
    monkeypatch.setattr(protocol, "DECLARED_HALTS", halt)
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", "a" * 64)
    with pytest.raises(data.StageDataError, match="not exchange-wide"):
        _load(rows, window)


# ======================================================================== coverage


def cov_argv(
    tmp_path: Path, stage: str = "BACKCAST_2020", name: str = "cov"
) -> list[str]:
    return ["coverage", "--stage", stage, "--output-dir", str(tmp_path / name)]


def cov_tool(
    tmp_path: Path,
    argv: Sequence[str],
    *,
    rows: Any,
    gh: Any = None,
    now: datetime = NOW,
) -> tuple[int, str, str]:
    return tool(tmp_path, argv, gh=gh or Gh(), now=now, rows=rows, with_rules=False)


def test_coverage_is_result_blind_and_reports_both_markets_and_funding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)

    def boom(*a: Any, **k: Any) -> Any:
        raise AssertionError("coverage must compute no signal, replay or return")

    monkeypatch.setattr(driver.runner, "run_stage", boom)
    monkeypatch.setattr(driver.runner, "anchor_signals_of", boom)
    monkeypatch.setattr(driver.data, "assemble_stage_dataset", boom)
    rows = Rows(listings(window))
    shared = {_ms(window.load_start) + (300 * 4 + h) * HOUR_MS for h in (2, 3)}
    for symbol in protocol.ROSTER:
        rows.skip.setdefault((StoreMarket.SPOT, symbol), set()).update(shared)
    rows.skip[(StoreMarket.USDS_M, "BTCUSDT")] = {_ms(window.load_start) + 50 * HOUR_MS}
    code, out, err = cov_tool(tmp_path, cov_argv(tmp_path), rows=rows)
    assert code == driver.EXIT_OK, err
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    assert document["schema_version"] == driver.COVERAGE_SCHEMA
    assert document["protocol_digest"] == protocol.protocol_digest()
    assert "no signal, replay, return or label" in document["not_computed"]
    btc = document["symbols"]["BTCUSDT"]
    assert btc["spot"]["missing_hours"] == 2 and btc["usds_m"]["missing_hours"] == 1
    assert (
        btc["usds_m"]["funding"]["events"] > 100
        and btc["usds_m"]["funding"]["max_gap_hours"] == 8.0
    )
    assert (
        btc["spot"]["checksums"]["archives_verified"] == 0
    )  # the fake has no archives
    assert document["symbols"][protocol.ROSTER[9]]["spot"]["first_listed_open"] is None
    gaps = document["exchange_wide_spot_gaps"]
    assert gaps["hours"] == 2 and len(gaps["ranges"]) == 1
    summary = json.loads(out)
    assert summary["exchange_wide_spot_gap_hours"] == 2
    assert "BTCUSDT:usds_m" in summary["legs_with_missing_hours"]
    # outputs are write-once and a missing 404 is a coverage fact, never a refusal
    code, _, err = cov_tool(tmp_path, cov_argv(tmp_path), rows=rows)
    assert code == driver.EXIT_REFUSED and "write-once" in err


def test_coverage_with_a_read_error_exits_halted_and_through_is_forward_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    original = rows.read_klines

    def flaky(market: Any, symbol: str, interval: str, s: int, e: int) -> Any:
        if (
            symbol == "ETHUSDT"
            and market is StoreMarket.USDS_M
            and s > _ms(window.load_start)
        ):
            raise driver.CarryIntegrityHalt(
                "bad archive", {"kind": "checksum_mismatch"}
            )
        return original(market, symbol, interval, s, e)

    rows.read_klines = flaky  # type: ignore[method-assign]
    code, out, _ = cov_tool(tmp_path, cov_argv(tmp_path), rows=rows)
    assert code == driver.EXIT_HALTED
    assert json.loads(out)["legs_with_read_errors"] == ["ETHUSDT:usds_m"]
    code, _, err = cov_tool(
        tmp_path,
        [*cov_argv(tmp_path, name="c2"), "--through", "2020-04-01T00:00:00Z"],
        rows=Rows(listings(window)),
    )
    assert code == driver.EXIT_REFUSED and "FORWARD only" in err


def test_forward_coverage_authenticates_the_data_authorization_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_FORWARD)
    rows = Rows(listings(window))
    now = datetime(2026, 8, 20, 6, 30, tzinfo=UTC)
    code, _, err = cov_tool(tmp_path, cov_argv(tmp_path, "FORWARD"), rows=rows, now=now)
    assert code == driver.EXIT_OK, err
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    auth = document["forward_data_authorization"]
    assert auth["url"] == DATA_AUTH and auth["embargo_policy"] == "ForwardEmbargoPolicy"
    assert max(end for *_, end in rows.requests) <= _ms(
        datetime(2026, 8, 20, tzinfo=UTC)
    )
    bad = Gh()
    bad.data.fail = True
    code, _, err = cov_tool(
        tmp_path,
        cov_argv(tmp_path, "FORWARD", "c3"),
        rows=Rows(listings(window)),
        gh=bad,
        now=now,
    )
    assert code == driver.EXIT_UNAUTHENTICATED


def test_the_dev_stage_embargo_policy_is_the_enforced_default() -> None:
    dev = data.frozen_window(protocol.STAGE_DEV)
    limit = data.read_limit(dev, now=NOW)
    policy = driver.embargo_for(limit, stage=protocol.STAGE_DEV)
    assert type(policy).__name__ == "EmbargoPolicy"
    with pytest.raises(data.StageEmbargoError):
        driver.embargo_for(
            datetime(2026, 8, 5, tzinfo=UTC), stage=protocol.STAGE_BACKCAST
        )
    plan = driver.build_plan(dev, limit, NOW, policy)
    assert {m.value for m in plan.universe.markets} == {"spot", "usds_m"}
    assert plan.end == datetime(2026, 8, 1, tzinfo=UTC)


# ============================================================== halts-from-coverage


def _coverage_document(**over: Any) -> dict[str, Any]:
    def leg(**extra: Any) -> dict[str, Any]:
        return {"errors": [], "unread_chunks": [], "missing_ranges": [], **extra}

    document: dict[str, Any] = {
        "schema_version": driver.COVERAGE_SCHEMA,
        "stage": "BACKCAST_2020",
        "protocol_digest": "d" * 64,
        "code_commit": "e" * 40,
        "symbols": {
            "BTCUSDT": {
                "spot": leg(
                    missing_ranges=[
                        {"first_missing_open": "2020-02-09T02:00:00+00:00", "hours": 3}
                    ]
                ),
                "usds_m": leg(),
            },
            "ETHUSDT": {
                "spot": leg(
                    missing_ranges=[
                        {"first_missing_open": "2020-02-09T02:00:00+00:00", "hours": 3}
                    ]
                ),
                "usds_m": leg(
                    missing_ranges=[
                        {"first_missing_open": "2020-02-09T02:00:00+00:00", "hours": 1}
                    ]
                ),
            },
        },
        "exchange_wide_spot_gaps": {
            "hours": 3,
            "ranges": [{"first_missing_open": "2020-02-09T02:00:00+00:00", "hours": 3}],
        },
    }
    document.update(over)
    return document


def test_halts_from_coverage_declares_only_spot_exchange_wide_ranges_and_lists_what_remains(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "coverage.json", _coverage_document())
    result = driver.halts_from_coverage(path)
    assert result["declared_halts"] == [
        ["2020-02-09T02:00:00Z", "2020-02-09T05:00:00Z"]
    ]
    assert result["declared_hours"] == 3
    assert len(result["coverage_json_sha256"]) == 64
    # a perpetual gap is never a halt: it remains and would refuse the run
    remaining = result["gaps_that_would_remain_and_refuse_the_run"]
    assert list(remaining) == ["ETHUSDT:usds_m"]
    out, err = io.StringIO(), io.StringIO()
    code = driver.main(
        ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err
    )
    assert code == driver.EXIT_OK and json.loads(out.getvalue())["declared_hours"] == 3


def test_halts_from_coverage_refuses_the_wrong_stage_errors_and_hidden_listings(
    tmp_path: Path,
) -> None:
    errored = _coverage_document()
    errored["symbols"]["BTCUSDT"]["usds_m"]["errors"] = [{"phase": "listing"}]
    hidden = _coverage_document()
    hidden["symbols"]["ETHUSDT"]["usds_m"][
        "leading_unpublished_months_with_daily_archives"
    ] = ["2020-01-01T00:00:00+00:00"]
    for document, text in (
        (_coverage_document(stage="DEV"), "BACKCAST_2020 coverage"),
        (_coverage_document(schema_version="x"), "not an agy_anchor_oos coverage"),
        (errored, "read errors"),
        (hidden, "listing would move"),
    ):
        path = _write(tmp_path, "c.json", document)
        out, err = io.StringIO(), io.StringIO()
        code = driver.main(
            ["halts-from-coverage", "--coverage", str(path)], stdout=out, stderr=err
        )
        assert code == driver.EXIT_REFUSED and text in err.getvalue(), text


# ============================================================ the whole sequence, e2e


def test_dev_then_backcast_then_forward_final_run_end_to_end_through_the_tool(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    small_window(monkeypatch, protocol.STAGE_DEV)
    back = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    fwd = small_window(monkeypatch, protocol.STAGE_FORWARD)

    def go(
        stage: str, *flags: str, name: str, now: datetime, window: data.StageWindow
    ) -> dict[str, Any]:
        rows = Rows(listings(window))
        code, out, err = tool(
            tmp_path,
            run_argv(tmp_path, stage, *flags, name=name),
            gh=Gh(),
            now=now,
            rows=rows,
        )
        assert code in (driver.EXIT_OK, driver.EXIT_INVALID), err
        return json.loads(out)

    dev_window = data.frozen_window(protocol.STAGE_DEV)
    dev_out = go("DEV", name="dev", now=NOW, window=dev_window)
    assert dev_out["label"] is None and dev_out["status"] == decision.DEV_STATUS
    dev_files = {p.name for p in (tmp_path / "dev").iterdir()}
    assert {
        "dev-result.json",
        driver.DATA_RECEIPT_NAME,
        driver.RUN_RECEIPT_NAME,
    } <= dev_files
    back_out = go(
        "BACKCAST_2020",
        "--dev-result",
        str(tmp_path / "dev" / "dev-result.json"),
        name="back",
        now=NOW,
        window=back,
    )
    assert back_out["stage_verdict"] in decision.VERDICTS
    decision_path = tmp_path / "back" / "decision.json"
    assert decision_path.exists()
    fwd_out = go(
        "FORWARD",
        "--final",
        "--backcast-decision",
        str(decision_path),
        name="fwd",
        now=FINAL_NOW,
        window=fwd,
    )
    assert fwd_out["label"] == decision.combine_label(
        back_out["stage_verdict"], fwd_out["stage_verdict"]
    )
    receipt = json.loads(
        (tmp_path / "fwd" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    assert receipt["backcast_decision"]["stage_verdict"] == back_out["stage_verdict"]
    assert receipt["forward_data_authorization"]["url"] == DATA_AUTH
    assert receipt["taker_buy"].startswith("sum of the four hourly")
    assert receipt["cash_account"]["initial_margin_rate"] == 0.4
    assert receipt["activation_rows"][5] is not None
    cells = list((tmp_path / "fwd" / "cells").glob("*.json"))
    assert len(cells) == 9
