"""Synthetic-data tests for tools/btc_trend_ensemble.py (no network, no market data)."""

from __future__ import annotations

import json
import math
import os
import re
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime
from pathlib import Path
from threading import Barrier
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from tools import btc_trend_ensemble as m

DIGEST_PIN = "837a61d29502880a27d58794b7445c4d418c0ec29871b85e308b04d748e8cbe7"
COMMIT = "a" * 40
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040999999"


# ----------------------------------------------------------------------------- digest


def test_digest_pin() -> None:
    assert m.protocol_digest() == DIGEST_PIN
    assert re.fullmatch(r"[0-9a-f]{64}", m.protocol_digest())


def test_constants_frozen() -> None:
    assert m.PROTOCOL_ID == "btc_trend_ensemble_v1"
    assert m.SOURCE_CLOSE_POLICY.value == "strict_source_close_v2"
    assert m.LOOKBACKS == (30, 50, 75, 100, 150, 200)
    assert (m.VOL_WINDOW, m.TARGET_VOL, m.DEADBAND) == (30, 0.20, 0.05)
    assert (m.EARLY_STOP_DD, m.MAX_DD, m.MIN_SHARPE) == (0.25, 0.25, 0.5)
    assert m.DATA_SCOPE_COMMENT_ID == 7040247247
    assert m.OWNER_LOGIN == "shuntatsu"
    assert str(m.LEDGER_DIR) == r"C:\trade_rl_runs\btc_trend_ensemble_ledger"
    b, f, fin = (
        m.STAGES["BACKCAST"],
        m.STAGES["FORWARD_INTERIM"],
        m.STAGES["FORWARD_FINAL"],
    )
    assert (b.load_from, b.decision_start, b.decision_end) == (
        date(2014, 6, 1),
        date(2015, 1, 1),
        date(2018, 6, 30),
    )
    assert (f.load_from, f.decision_start, f.decision_end) == (
        date(2026, 5, 1),
        date(2026, 12, 1),
        date(2027, 11, 30),
    )
    assert (fin.load_from, fin.decision_start, fin.decision_end) == (
        date(2026, 5, 1),
        date(2026, 12, 1),
        date(2027, 11, 30),
    )
    assert f.not_before == datetime(2026, 12, 2, tzinfo=UTC)
    assert f.not_after == datetime(2027, 12, 2, tzinfo=UTC)
    assert fin.not_before == datetime(2027, 12, 2, tzinfo=UTC)
    assert m.DEFECTIVE_BACKCAST_DIGEST.startswith("7e276c4f")
    assert m.DEFECTIVE_BACKCAST_LABEL == "TREND_BACKCAST_ONLY"
    assert (m.LABEL_SUPPORTED, m.LABEL_NOT_SUPPORTED) == (
        "TREND_FORWARD_SUPPORTED",
        "TREND_FORWARD_NOT_SUPPORTED",
    )
    assert (m.PROBE_START, m.PROBE_DAYS) == (date(2026, 10, 1), 7)
    assert m.PROBE_START < date(2026, 11, 1)
    assert m.CELLS["TREND__BASE"] == ("trend", 0.0010, 1, True)
    assert m.CELLS["TREND__HIGH_COST"] == ("trend", 0.0020, 1, True)
    assert m.CELLS["TREND__DELAY"] == ("trend", 0.0010, 2, True)


# ----------------------------------------------------------------------------- rule functions


def _naive_score(c: np.ndarray, t: int) -> float:
    if t < 199:
        return 0.0
    hits = sum(1 for n in m.LOOKBACKS if c[t] > c[t - n + 1 : t + 1].mean())
    return hits / 6


def test_trend_score_monotone_and_warmup() -> None:
    up = np.arange(1.0, 251.0)
    s = m.trend_score(up)
    assert np.all(s[:199] == 0.0)
    assert np.all(s[199:] == 1.0)
    assert np.all(m.trend_score(up[::-1].copy()) == 0.0)


def test_trend_score_partial_and_strict() -> None:
    c = np.concatenate([np.full(100, 100.0), np.full(100, 200.0)])
    # at t=199 close 200 is not above SMA_30..SMA_100 (all 200), above SMA_150 and SMA_200
    assert m.trend_score(c)[199] == pytest.approx(2 / 6)
    flat = np.full(200, 100.0)
    assert m.trend_score(flat)[199] == 0.0  # equal is not above
    c2 = c.copy()
    c2[199] = 201.0
    assert m.trend_score(c2)[199] == 1.0


def test_trend_score_matches_naive_oracle() -> None:
    rng = np.random.default_rng(0)
    c = 100 * np.exp(np.cumsum(rng.normal(0.0, 0.03, 400)))
    s = m.trend_score(c)
    assert np.allclose(s, [_naive_score(c, t) for t in range(len(c))])


def test_realized_vol_uses_log_returns_ddof0_sqrt365() -> None:
    r = np.tile([0.01, -0.01], 15)  # 30 log returns, mean 0, population std 0.01
    c = 100 * np.exp(np.concatenate([[0.0], np.cumsum(r)]))
    v = m.realized_vol(c)
    assert np.all(np.isnan(v[:30]))
    assert v[30] == pytest.approx(0.01 * math.sqrt(365))
    assert v[30] != pytest.approx(0.01 * math.sqrt(365) * math.sqrt(30 / 29))


def test_target_exposure_clipping() -> None:
    e = m.target_exposure(
        [1.0, 1.0, 0.5, 1.0, 1.0, 0.0], [0.1, 0.4, 0.4, np.nan, 0.0, 0.1]
    )
    assert e == pytest.approx([1.0, 0.5, 0.25, 0.0, 1.0, 0.0])
    assert np.all((e >= 0) & (e <= 1))


def test_deadband_strict_and_sticky() -> None:
    out = m.apply_deadband([0.05, 0.0500001, 0.09, 0.0100001, 0.0400001])
    # 0.05 is not > 0.05 from 0; 0.0500001 moves; 0.09 and 0.0100001 within the band of 0.0500001
    assert out == pytest.approx([0.0, 0.0500001, 0.0500001, 0.0500001, 0.0500001])
    assert m.apply_deadband([0.2, 0.3], held0=0.2)[1] == 0.3
    assert m.apply_deadband([0.0], held0=0.2)[0] == 0.0


# ----------------------------------------------------------------------------- replay


def _rising(n: int = 300, g: float = 0.001) -> np.ndarray:
    return 100.0 * (1.0 + g) ** np.arange(n)


def test_replay_accounting_identity_and_stats() -> None:
    rng = np.random.default_rng(1)
    c = 100 * np.exp(np.cumsum(rng.normal(0.001, 0.03, 600)))
    cell = m.replay_cell(c, 0.001, 1)
    assert cell["valid"] and cell["max_identity_err"] <= 1e-9
    assert cell["equity"][-1] == pytest.approx(np.prod(1 + cell["daily_returns"]))
    assert cell["total_return"] == pytest.approx(cell["equity"][-1] - 1)
    r = cell["daily_returns"]
    assert cell["sharpe"] == pytest.approx(r.mean() / r.std(ddof=0) * math.sqrt(365))
    peak = np.maximum.accumulate(cell["equity"])
    assert cell["max_dd"] == pytest.approx(np.max((peak - cell["equity"]) / peak))
    assert cell["trades"] > 0 and cell["turnover"] > 0
    assert len(cell["equity"]) == len(c)


def test_replay_delay_semantics_known_jump() -> None:
    c = _rising()
    c[200:] *= 1.5  # jump on day index 200: first day held under delay 1
    b = m.replay_cell(c, 0.0, 1)
    d = m.replay_cell(c, 0.0, 2)
    # day d has return index d-1 (start 0); exposure decided at 199 is 1.0
    assert b["daily_returns"][199] == pytest.approx(c[200] / c[199] - 1)
    assert d["daily_returns"][199] == 0.0
    assert d["daily_returns"][200] != 0.0  # delay-2 is invested from day 201
    assert np.all(b["daily_returns"][:199] == 0.0)


def test_replay_charges_cost_on_exposure_change() -> None:
    c = _rising()
    c[200:] *= 1.5
    cell = m.replay_cell(c, 0.01, 1)
    # Full post-fee BTC entry: E_after + rate * E_after = E_before.
    assert cell["daily_returns"][199] == pytest.approx(c[200] / c[199] / (1 + 0.01) - 1)
    free = m.replay_cell(c, 0.0, 1)
    assert cell["total_return"] < free["total_return"]
    assert cell["trades"] == free["trades"]


def test_replay_start_and_argument_checks() -> None:
    c = _rising()
    assert len(m.replay_cell(c, 0.0, 1, start=250)["daily_returns"]) == len(c) - 251
    with pytest.raises(ValueError):
        m.replay_cell(c, 0.0, 0)
    with pytest.raises(ValueError):
        m.replay_cell(c, 0.0, 1, start=len(c) - 1)
    with pytest.raises(ValueError):
        m.replay_cell(np.array([1.0, -1.0, 2.0]), 0.0, 1)


def test_hold_cell_cost_at_entry_only() -> None:
    c = _rising(50)
    cell = m.hold_cell(c, 0.001, start=10)
    assert cell["trades"] == 1 and cell["turnover"] == pytest.approx(1 / 1.001)
    assert cell["equity"][-1] == pytest.approx(c[-1] / c[10] / 1.001)
    assert cell["valid"]


def test_cash_cell_is_flat() -> None:
    cell = m.cash_cell(30)
    assert (
        cell["total_return"] == 0.0 and cell["max_dd"] == 0.0 and cell["sharpe"] == 0.0
    )
    assert cell["valid"]


def test_invalid_cell_on_bad_exposure_and_nonpositive_equity() -> None:
    c = _rising(10)
    bad = m._account(c, np.full(10, 1.5), 0.0, 0)
    assert not bad["valid"] and "outside [0, 1]" in bad["invalid_reasons"][0]
    wipe = m._account(np.array([1.0, 1e-30, 1.0]), np.array([0.0, 1.0, 1.0]), 0.0, 0)
    assert wipe["max_dd"] > 0.99
    gap = m._account(c, np.full(10, 1.0), 2.0, 0)  # cost > 100% drives equity negative
    assert not gap["valid"]


# ----------------------------------------------------------------------------- decide


def _cell(
    ret: float = 0.1, sharpe: float = 1.0, dd: float = 0.1, valid: bool = True
) -> dict[str, Any]:
    return {"total_return": ret, "sharpe": sharpe, "max_dd": dd, "valid": valid}


def _cells(**over: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cells = {k: _cell() for k in m.CELLS}
    cells["CASH"] = _cell(0.0, 0.0, 0.0)
    cells.update(over)
    return cells


def test_decide_forward_final_branches_and_no_backcast_labels() -> None:
    assert m.decide("FORWARD_FINAL", _cells())["label"] == "TREND_FORWARD_SUPPORTED"
    assert (
        m.decide("FORWARD_FINAL", _cells(TREND__BASE=_cell(ret=-0.01)))["label"]
        == "TREND_FORWARD_NOT_SUPPORTED"
    )
    inv = m.decide("FORWARD_FINAL", _cells(HOLD__BASE=_cell(valid=False)))
    assert inv["label"] == "INVALID" and inv["invalid_cells"] == ["HOLD__BASE"]
    # BACKCAST is consumed: it has no label and never decides
    with pytest.raises(ValueError, match="consumed"):
        m.decide("BACKCAST", _cells())
    for cells in (_cells(), _cells(TREND__BASE=_cell(ret=-1.0))):
        label = m.decide("FORWARD_FINAL", cells)["label"]
        assert label not in {
            "TREND_SUPPORTED",
            "TREND_BACKCAST_ONLY",
            "TREND_NOT_SUPPORTED",
        }


def test_decide_threshold_strictness() -> None:
    ok = lambda **kw: m.decide("FORWARD_FINAL", _cells(**kw))["supported"]  # noqa: E731
    assert not ok(TREND__BASE=_cell(ret=0.0))  # return strictly above 0
    assert ok(TREND__BASE=_cell(sharpe=0.5))  # sharpe at least 0.5
    assert not ok(TREND__BASE=_cell(sharpe=0.4999))
    assert not ok(TREND__HIGH_COST=_cell(ret=0.0))
    assert not ok(TREND__DELAY=_cell(ret=0.0))
    assert ok(TREND__BASE=_cell(dd=0.2499))  # max DD strictly below 0.25
    assert not ok(TREND__BASE=_cell(dd=0.25))
    assert not ok(TREND__HIGH_COST=_cell(dd=0.25))
    assert not ok(TREND__DELAY=_cell(dd=0.25))
    assert ok(HOLD__BASE=_cell(ret=-0.5, dd=0.9))  # reported, not gating
    assert ok(TREND__HIGH_COST=_cell(sharpe=-3.0))  # only BASE has a sharpe gate


def test_decide_final_ignores_backcast_entirely() -> None:
    # No second argument exists: the verdict is a function of the FORWARD cells alone.
    with pytest.raises(TypeError):
        m.decide("FORWARD_FINAL", _cells(), True)  # type: ignore[call-arg]
    bad = _cells(TREND__DELAY=_cell(ret=-0.1))
    assert m.decide("FORWARD_FINAL", bad)["label"] == "TREND_FORWARD_NOT_SUPPORTED"


def test_decide_interim_branches() -> None:
    assert m.decide("FORWARD_INTERIM", _cells())["label"] == "INTERIM_NOT_A_DECISION"
    # an interim never claims a decision, even when unsupported
    weak = _cells(TREND__BASE=_cell(ret=-0.1))
    assert m.decide("FORWARD_INTERIM", weak)["label"] == "INTERIM_NOT_A_DECISION"
    assert (
        m.decide("FORWARD_INTERIM", _cells(TREND__BASE=_cell(dd=0.2499)))["label"]
        == "INTERIM_NOT_A_DECISION"
    )
    assert (
        m.decide("FORWARD_INTERIM", _cells(TREND__BASE=_cell(dd=0.25)))["label"]
        == "STOPPED_EARLY"
    )
    # only TREND__BASE drawdown stops the line
    assert (
        m.decide("FORWARD_INTERIM", _cells(TREND__DELAY=_cell(dd=0.5)))["label"]
        == "INTERIM_NOT_A_DECISION"
    )
    inv = _cells(TREND__BASE=_cell(dd=0.9, valid=False))
    assert m.decide("FORWARD_INTERIM", inv)["label"] == "INVALID"


# ----------------------------------------------------------------------------- parsers


def _ts(d: date) -> int:
    return (d - m.EPOCH).days * 86400


def _bitstamp(days: list[date], close: str = "100.5") -> dict[str, Any]:
    return {
        "data": {
            "pair": "BTC/USD",
            "ohlc": [
                {
                    "timestamp": str(_ts(d)),
                    "open": "1",
                    "high": "1",
                    "low": "1",
                    "close": close,
                    "volume": "1",
                }
                for d in days
            ],
        }
    }


def _binance(days: list[date], close: str = "100.5") -> list[list[Any]]:
    return [
        [
            _ts(d) * 1000,
            "1",
            "1",
            "1",
            close,
            "1",
            _ts(d) * 1000 + 86399999,
            "1",
            1,
            "1",
            "1",
            "0",
        ]
        for d in days
    ]


D = [date(2020, 1, k) for k in range(1, 6)]


def test_parse_bitstamp_and_binance_roundtrip() -> None:
    b = m.parse_bitstamp(_bitstamp(D))
    k = m.parse_binance(_binance(D))
    assert b == k == [(_ts(d), 100.5) for d in D]
    assert list(m.build_daily(b, D[0], D[-1])) == [100.5] * 5


@pytest.mark.parametrize(
    "bad",
    [
        {},
        {"data": {}},
        {"data": {"ohlc": "x"}},
        {"data": {"ohlc": [{"close": "1"}]}},
        [],
    ],
)
def test_parse_bitstamp_malformed(bad: Any) -> None:
    if bad == []:
        bad = None
    with pytest.raises(SystemExit, match="malformed"):
        m.parse_bitstamp(bad)


def test_parse_binance_malformed() -> None:
    with pytest.raises(SystemExit, match="malformed"):
        m.parse_binance({"x": 1})
    with pytest.raises(SystemExit, match="malformed"):
        m.parse_binance([[1, 2]])
    with pytest.raises(SystemExit, match="whole seconds"):
        m.parse_binance([[1500, "1", "1", "1", "1"]])


@pytest.mark.parametrize("close", ["0", "-5", "nan", "inf"])
def test_build_daily_refuses_bad_close(close: str) -> None:
    rows = m.parse_bitstamp(_bitstamp(D[:2], close))
    with pytest.raises(SystemExit, match="non-positive"):
        m.build_daily(rows, D[0], D[1])


def test_build_daily_refuses_missing_duplicate_misaligned_and_short() -> None:
    rows = m.parse_bitstamp(_bitstamp([D[0], D[1], D[3], D[4]]))
    with pytest.raises(SystemExit, match="missing UTC day 2020-01-03"):
        m.build_daily(rows, D[0], D[4])
    dup = m.parse_bitstamp(_bitstamp([D[0], D[1], D[1]]))
    with pytest.raises(SystemExit, match="duplicate"):
        m.build_daily(dup, D[0], D[1])
    with pytest.raises(SystemExit, match="UTC day boundary"):
        m.build_daily([(_ts(D[0]) + 3600, 1.0)], D[0], D[0])
    short = m.parse_bitstamp(_bitstamp(D[:3]))
    with pytest.raises(SystemExit, match="ends before"):
        m.build_daily(short, D[0], D[4])
    with pytest.raises(SystemExit, match="ends before"):
        m.build_daily([], D[0], D[4])
    with pytest.raises(SystemExit, match="missing UTC day 2020-01-01"):
        m.build_daily(m.parse_bitstamp(_bitstamp(D[1:])), D[0], D[4])


def test_build_daily_drops_bars_after_last() -> None:
    rows = m.parse_binance(_binance(D))
    assert len(m.build_daily(rows, D[0], D[2])) == 3


def _price(d: date) -> float:
    i = (d - m.EPOCH).days
    return 100.0 * math.exp(0.002 * (i - 16000) + 0.05 * math.sin(i / 7.0))


def fake_market(url: str) -> bytes:
    u = urlparse(url)
    q = {
        k: int(v[0]) for k, v in parse_qs(u.query).items() if v[0].lstrip("-").isdigit()
    }
    if "bitstamp" in u.netloc:
        lo, hi, lim = q["start"], q["end"], q["limit"]
        # Real Bitstamp semantics (observed 2026-11-22): ``start`` is ignored and the API
        # returns the LAST ``limit`` bars at or before ``end`` (Amendment 1).
        del lo
        top = hi // 86400 * 86400
        ts = list(range(top - (lim - 1) * 86400, top + 1, 86400))
        days = [m.EPOCH.fromordinal(m.EPOCH.toordinal() + t // 86400) for t in ts]
        items = [
            {
                "timestamp": str(t),
                "open": "1",
                "high": "1",
                "low": "1",
                "close": f"{_price(d):.6f}",
                "volume": "1",
            }
            for t, d in zip(ts, days, strict=True)
        ]
        return json.dumps({"data": {"pair": "BTC/USD", "ohlc": items}}).encode()
    lo, hi, lim = q["startTime"] // 1000, q["endTime"] // 1000, q["limit"]
    ts = list(range((lo + 86399) // 86400 * 86400, hi + 1, 86400))[:lim]
    rows = [
        [
            t * 1000,
            "1",
            "1",
            "1",
            f"{_price(m.EPOCH.fromordinal(m.EPOCH.toordinal() + t // 86400)):.6f}",
            "1",
            t * 1000 + 86399999,
        ]
        for t in ts
    ]
    return json.dumps(rows).encode()


def test_fetch_daily_pages_caches_and_hashes(tmp_path: Path) -> None:
    first, last = date(2014, 6, 1), date(2018, 6, 30)  # 1491 days: two pages
    calls: list[str] = []

    def f(url: str) -> bytes:
        calls.append(url)
        return fake_market(url)

    closes, receipt = m.fetch_daily("bitstamp", first, last, tmp_path / "raw", f)
    assert len(closes) == (last - first).days + 1
    assert len(receipt["pages"]) == 2 and len(calls) == 2
    assert "step=86400" in calls[0] and "limit=1000" in calls[0]
    for p in receipt["pages"]:
        assert m.sha256_file(tmp_path / "raw" / p["file"]) == p["sha256"]
    closes2, _ = m.fetch_daily("binance", first, last, tmp_path / "raw2", fake_market)
    assert np.allclose(closes, closes2, rtol=1e-6)


def test_fetch_daily_refuses_short_and_nonprogress(tmp_path: Path) -> None:
    pages = iter([json.dumps(_bitstamp(D[:3])).encode(), b'{"data": {"ohlc": []}}'])
    with pytest.raises(SystemExit, match="ends before"):
        m.fetch_daily("bitstamp", D[0], D[4], tmp_path / "a", lambda u: next(pages))
    # A server that keeps returning the same stale bars: out-of-window rows are dropped, the
    # next page is empty, and the run refuses loudly on the short series (Amendment 1).
    with pytest.raises(SystemExit, match="ends before|missing"):
        m.fetch_daily(
            "bitstamp",
            D[0],
            D[4],
            tmp_path / "a2",
            lambda u: json.dumps(_bitstamp(D[:3])).encode(),
        )
    with pytest.raises(SystemExit, match="not JSON"):
        m.fetch_daily("binance", D[0], D[4], tmp_path / "b", lambda u: b"<html>")
    with pytest.raises(SystemExit, match="ends before"):
        m.fetch_daily("binance", D[0], D[4], tmp_path / "c", lambda u: b"[]")


# ----------------------------------------------------------------------------- authorization


def _comment(
    body: str,
    login: str = "shuntatsu",
    issue_url: str = m.ISSUE_URL,
    updated: str = "t1",
) -> dict[str, Any]:
    return {
        "user": {"login": login},
        "issue_url": issue_url,
        "body": body,
        "updated_at": updated,
    }


def _gh(relay_body: str | None = None, **over: Any) -> Any:
    digest = "d" * 64
    relay_body = (
        relay_body
        if relay_body is not None
        else f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\nG4 claim: FORWARD_FINAL"
    )

    def fetch(cid: int) -> dict[str, Any]:
        if cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", **over.get("scope", {}))
        return _comment(relay_body, **over.get("relay", {}))

    return fetch


def test_authorization_ok() -> None:
    rec = m.authorization_receipt(URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh())
    assert rec["g4_relay"]["comment_id"] == 7040999999
    assert rec["data_scope"]["comment_id"] == m.DATA_SCOPE_COMMENT_ID
    assert len(rec["g4_relay"]["body_sha256"]) == 64


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/pull/810#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        URL + "x",
        " " + URL,
        "https://github.com/shuntatsu/trade_rl/issues/810",
    ],
)
def test_authorization_wrong_url(url: str) -> None:
    with pytest.raises(SystemExit, match="Issue #810 comment"):
        m.authorization_receipt(url, "d" * 64, COMMIT, "FORWARD_FINAL", _gh())


def test_authorization_owner_and_issue_binding() -> None:
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay={"login": "mallory"})
        )
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(scope={"login": "mallory"})
        )
    other = "https://api.github.com/repos/shuntatsu/trade_rl/issues/8100"
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay={"issue_url": other})
        )
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(
            URL,
            "d" * 64,
            COMMIT,
            "FORWARD_FINAL",
            _gh(scope={"issue_url": "x/issues/810"}),
        )


def test_authorization_edit_between_fetches() -> None:
    calls: dict[int, int] = {}

    def fetch(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        body = f"G4 identity: {m.PROTOCOL_ID} {'d' * 64} {COMMIT}"
        if cid != m.DATA_SCOPE_COMMENT_ID and calls[cid] == 2:
            body += " edited"
        return _comment(body)

    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, "FORWARD_FINAL", fetch)

    def fetch_ts(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        return _comment("scope", updated=f"t{calls[cid]}")

    calls.clear()
    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, "FORWARD_FINAL", fetch_ts)


@pytest.mark.parametrize(
    "body",
    [
        f"G4 identity: {m.PROTOCOL_ID} {COMMIT}",
        f"G4 identity: {m.PROTOCOL_ID} {'d' * 64}",
        f"{'d' * 64} {COMMIT}",
        f"G4 identity: {m.PROTOCOL_ID} {'d' * 64} {COMMIT[:12]}",
    ],
)
def test_authorization_relay_must_quote_id_digest_commit(body: str) -> None:
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


def test_authorization_requires_full_commit() -> None:
    with pytest.raises(SystemExit, match="40-hex"):
        m.authorization_receipt(URL, "d" * 64, "abc1234", "FORWARD_FINAL", _gh())


def test_harness_commit_requires_clean_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        m, "_git", lambda *a: " M tools/x.py\n" if a[0] == "status" else COMMIT
    )
    with pytest.raises(SystemExit, match="tracked modifications"):
        m.harness_commit()
    monkeypatch.setattr(m, "_git", lambda *a: "" if a[0] == "status" else COMMIT + "\n")
    assert m.harness_commit() == COMMIT
    monkeypatch.setattr(m, "_git", lambda *a: "" if a[0] == "status" else "abc\n")
    with pytest.raises(SystemExit, match="unexpected HEAD"):
        m.harness_commit()


# ----------------------------------------------------------------------------- ledger & clocks


def test_ledger_claim_is_exclusive(tmp_path: Path) -> None:
    now = datetime(2026, 12, 5, tzinfo=UTC)
    p = m.claim_stage("BACKCAST", "d", COMMIT, 1, now, tmp_path)
    assert json.loads(p.read_text())["protocol_digest"] == "d"
    with pytest.raises(SystemExit, match="already claimed"):
        m.claim_stage("BACKCAST", "d", COMMIT, 1, now, tmp_path)
    m.write_ledger_result("BACKCAST", "d", "TREND_BACKCAST_ONLY", True, "s", tmp_path)
    with pytest.raises(SystemExit, match="already exists"):
        m.write_ledger_result("BACKCAST", "d", "x", False, "s", tmp_path)


def test_interim_claim_is_once_per_month(tmp_path: Path) -> None:
    st = m.STAGES["FORWARD_INTERIM"]
    dec = datetime(2026, 12, 5, tzinfo=UTC)
    jan = datetime(2027, 1, 5, tzinfo=UTC)
    assert m.claim_name(st, dec) == "FORWARD_INTERIM_2026-12"
    assert m.claim_name(m.STAGES["BACKCAST"], dec) == "BACKCAST"
    m.claim_stage(m.claim_name(st, dec), "d", COMMIT, 1, dec, tmp_path)
    m.claim_stage(m.claim_name(st, jan), "d", COMMIT, 1, jan, tmp_path)
    with pytest.raises(SystemExit, match="already claimed"):
        m.claim_stage(
            m.claim_name(st, dec.replace(day=20)), "d", COMMIT, 1, dec, tmp_path
        )


def test_not_before_refusals() -> None:
    inter, fin = m.STAGES["FORWARD_INTERIM"], m.STAGES["FORWARD_FINAL"]
    with pytest.raises(SystemExit, match="may not run before"):
        m.check_clock(inter, datetime(2026, 12, 1, 23, 59, tzinfo=UTC))
    m.check_clock(inter, datetime(2026, 12, 2, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before 2027-12-02"):
        m.check_clock(fin, datetime(2027, 12, 1, 23, 59, tzinfo=UTC))
    m.check_clock(fin, datetime(2027, 12, 2, tzinfo=UTC))
    m.check_clock(inter, datetime(2027, 12, 1, 23, 59, tzinfo=UTC))
    with pytest.raises(SystemExit, match="at or after"):
        m.check_clock(inter, datetime(2027, 12, 2, tzinfo=UTC))


def test_stage_window_interim_ends_at_last_complete_day() -> None:
    inter = m.STAGES["FORWARD_INTERIM"]
    assert m.stage_window(inter, datetime(2026, 12, 5, 13, tzinfo=UTC)) == (
        date(2026, 5, 1),
        date(2026, 12, 4),
    )
    assert m.stage_window(m.STAGES["FORWARD_FINAL"], datetime(2028, 1, 1, tzinfo=UTC))[
        1
    ] == date(2027, 11, 30)


def test_interim_without_an_evaluated_day_refuses_before_any_claim() -> None:
    inter = m.STAGES["FORWARD_INTERIM"]
    with pytest.raises(SystemExit, match="no evaluated day complete yet"):
        m.stage_window(inter, datetime(2026, 12, 2, 9, tzinfo=UTC))
    assert m.stage_window(inter, datetime(2026, 12, 3, 0, 1, tzinfo=UTC))[1] == date(
        2026, 12, 2
    )


# ----------------------------------------------------------------------------- stage runs (synthetic)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(m, "harness_commit", lambda: COMMIT)
    digest = m.protocol_digest()
    gh = _gh(
        relay_body=f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\nG4 claim: FORWARD_INTERIM_2026-12"
    )
    return {"ledger": tmp_path / "ledger", "tmp": tmp_path, "gh": gh, "digest": digest}


def _seed_defective_backcast(ledger: Path, label: str = "TREND_BACKCAST_ONLY") -> Path:
    """The consumed defective BACKCAST record exactly as it sits in the real ledger."""
    ns = ledger / m.DEFECTIVE_BACKCAST_DIGEST
    m.claim_stage("BACKCAST", m.DEFECTIVE_BACKCAST_DIGEST, COMMIT, 1, NOW, ns)
    m.write_ledger_result(
        "BACKCAST",
        m.DEFECTIVE_BACKCAST_DIGEST,
        label,
        label == "TREND_BACKCAST_ONLY",
        "e63486ce",
        ns,
    )
    return ns


def _run(
    env: dict[str, Any], stage: str, now: datetime, tag: str = "x"
) -> dict[str, Any]:
    name = m.claim_name(m.STAGES[stage], now)
    # Each distinct synthetic stage receives a separate relay; repeating a
    # stage deliberately retains its ID so the stage lock is still tested.
    cid = (
        7040999999
        if stage == "FORWARD_FINAL"
        else 7041000000 + now.year * 12 + now.month
    )
    url = URL.rsplit("-", 1)[0] + f"-{cid}"
    gh = env.get("gh_override") or _gh(
        relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: {name}"
    )
    return m.run_stage(
        stage,
        url,
        env["tmp"] / f"work_{stage}_{tag}",
        env["tmp"] / f"out_{stage}_{tag}",
        now=now,
        fetch=fake_market,
        gh_fetch=gh,
        ledger_dir=env["ledger"],
    )


NOW = datetime(2026, 12, 5, 9, tzinfo=UTC)


def test_clock_refusal_precedes_claim(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 1, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_INTERIM", datetime(2026, 12, 1, tzinfo=UTC))
    assert not env["ledger"].exists()


def test_authorization_refusal_precedes_claim(env: dict[str, Any]) -> None:
    env2 = {**env, "gh_override": _gh(relay_body="no quote")}
    _seed_defective_backcast(env["ledger"])
    with pytest.raises(SystemExit, match="does not quote"):
        _run(env2, "FORWARD_INTERIM", NOW)
    assert not (env["ledger"] / "FORWARD_INTERIM_2026-12.claim").exists()
    assert not (env["ledger"] / env["digest"]).exists()


def test_cli_digest(capsys: pytest.CaptureFixture[str]) -> None:
    m.main(["digest"])
    assert capsys.readouterr().out.strip() == m.protocol_digest()
    with pytest.raises(SystemExit):
        m.main(
            [
                "run",
                "--stage",
                "NOPE",
                "--authorization-url",
                URL,
                "--work",
                "w",
                "--output-dir",
                "o",
            ]
        )


def test_unchanged_target_never_rebalances_btc_quantity_for_free() -> None:
    """One purchase at 100 and no more trades must return to parity at 100."""
    c = np.array([100.0, 200.0, 100.0])
    w = np.array([0.0, 0.5, 0.5])
    result = m._account(c, w, 0.0, 0)
    assert result["valid"]
    assert result["trades"] == 1
    assert result["turnover"] == pytest.approx(0.5)
    np.testing.assert_allclose(result["equity"], [1.0, 1.5, 1.0], rtol=0, atol=1e-12)


def test_actual_fill_notional_is_charged_without_hidden_second_sale() -> None:
    """Independent exact rational accounting of a half-weight entry at 10 bp."""
    from decimal import Decimal

    c = np.array([100.0, 200.0, 200.0])
    w = np.array([0.0, 0.5, 0.5])
    result = m._account(c, w, 0.001, 0)
    # When buying 50% of post-cost equity at 100, both BTC notional and fee
    # depend on after-fee wealth, so the independent cash equation is:
    # after = 1 - 0.001 * (0.5 * after).
    after = Decimal(1) / (Decimal(1) + Decimal("0.001") * Decimal("0.5"))
    expected = float(after * Decimal("1.5"))
    assert result["valid"]
    assert result["trades"] == 1
    assert result["turnover"] == pytest.approx(float(after * Decimal("0.5")), abs=1e-12)
    assert result["equity"][-1] == pytest.approx(expected, abs=1e-12, rel=0)


def test_buy_and_sell_costs_match_independent_decimal_bisection_ledger() -> None:
    """A separate Decimal cash/quantity solver reconstructs each actual fill."""
    from decimal import Decimal, localcontext

    closes = [100.0, 200.0, 100.0, 130.0, 70.0]
    targets = [0.0, 0.4, 0.4, 0.8, 0.2]
    fee_rate = 0.002
    actual = m._account(np.array(closes), np.array(targets), fee_rate, 0)
    with localcontext() as ctx:
        ctx.prec = 70
        cash, units, previous = Decimal(1), Decimal(0), Decimal(0)
        rate = Decimal(str(fee_rate))
        reference = [1.0]
        turnover = Decimal(0)
        fills = 0
        for d in range(1, len(closes)):
            p0, p1 = Decimal(str(closes[d - 1])), Decimal(str(closes[d]))
            target = Decimal(str(targets[d]))
            if target != previous:
                wealth = cash + units * p0
                low, high = Decimal(0), wealth
                for _ in range(240):
                    mid = (low + high) / 2
                    if mid + rate * abs(target * mid - units * p0) > wealth:
                        high = mid
                    else:
                        low = mid
                after_fee = (low + high) / 2
                new_units = target * after_fee / p0
                notional = abs(new_units - units) * p0
                cash -= (new_units - units) * p0 + rate * notional
                units = new_units
                turnover += notional / wealth
                fills += int(notional > 0)
                previous = target
            reference.append(float(cash + units * p1))
    np.testing.assert_allclose(actual["equity"], reference, rtol=1e-12, atol=1e-12)
    assert actual["turnover"] == pytest.approx(float(turnover), rel=1e-12)
    assert actual["trades"] == fills
    assert actual["max_identity_err"] < m.IDENTITY_TOL


def test_every_representable_executed_btc_fill_is_counted() -> None:
    """A sub-epsilon notional is still a filled trade, not a hidden rebalance."""
    prices = np.array([100.0, 100.0000000006, 100.0000000006])
    targets = np.array([0.0, 0.5, 0.50000000000175])
    result = m._account(prices, targets, 0.0, 0)
    assert result["valid"]
    assert result["trades"] == 2
    assert result["turnover"] > 0.5


# Binance spot forward source-clock regression only; Bitstamp BACKCAST is unaffected.
@pytest.mark.parametrize(
    ("native_unit", "close_shift"),
    [
        (1, -m.DAY_S * 1000),
        (1, -1),
        (1, 1),
        (1000, -1),
        (1000, 1),
    ],
)
def test_binance_daily_refuses_premature_or_offbyone_source_close(
    native_unit: int,
    close_shift: int,
) -> None:
    source_open_ms = 1735689600000  # 2025-01-01 UTC
    row = _binance([date(2025, 1, 1)])[0]
    row[0] = source_open_ms * native_unit
    row[6] = (source_open_ms + m.DAY_S * 1000) * native_unit - 1 + close_shift
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_binance([row])


@pytest.mark.parametrize(("open_unit", "close_unit"), [(1, 1000), (1000, 1)])
def test_binance_daily_refuses_mixed_native_timestamp_units(
    open_unit: int,
    close_unit: int,
) -> None:
    source_open_ms = 1735689600000
    row = _binance([date(2025, 1, 1)])[0]
    row[0] = source_open_ms * open_unit
    row[6] = (source_open_ms + m.DAY_S * 1000) * close_unit - 1
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_binance([row])


@pytest.mark.parametrize("native_unit", [1, 1000])
def test_binance_daily_accepts_exact_source_end_and_normalizes_to_seconds(
    native_unit: int,
) -> None:
    source_open_ms = 1735689600000
    row = _binance([date(2025, 1, 1)])[0]
    row[0] = source_open_ms * native_unit
    row[6] = (source_open_ms + m.DAY_S * 1000) * native_unit - 1
    assert m.parse_binance([row]) == [(source_open_ms // 1000, 100.5)]


@pytest.mark.parametrize("bad", [None, 2.5, "1735775999999.0", -1])
def test_binance_daily_refuses_non_native_source_close(bad: object) -> None:
    row = _binance([date(2025, 1, 1)])[0]
    row[6] = bad
    with pytest.raises(SystemExit, match="close_time"):
        m.parse_binance([row])


def test_binance_daily_rejected_source_close_is_not_silently_filled(
    tmp_path: Path,
) -> None:
    row = _binance([date(2025, 1, 1)])[0]
    row[6] = row[0] - 1
    body = json.dumps([row]).encode()
    with pytest.raises(SystemExit, match="close_time"):
        m.fetch_daily(
            "binance",
            date(2025, 1, 1),
            date(2025, 1, 1),
            tmp_path / "raw",
            lambda _url: body,
        )
    assert (tmp_path / "raw" / "binance_000.json").read_bytes() == body


@pytest.mark.parametrize(
    "bad_open", [True, "1735689600000.0", 1735693200000, 1735689600000001]
)
def test_binance_daily_rejects_invalid_native_open(bad_open: object) -> None:
    row = _binance([date(2025, 1, 1)])[0]
    row[0] = bad_open
    with pytest.raises(SystemExit, match="malformed|UTC day boundary"):
        m.parse_binance([row])


@pytest.mark.parametrize(
    "closes",
    [
        [0.0, 100.0],
        [-1.0, 100.0],
        [float("nan"), 100.0],
        [float("inf"), 100.0],
        [100.0, 0.0],
        [100.0, float("-inf")],
    ],
)
def test_hold_cell_refuses_nonpositive_or_nonfinite_prices_without_division(
    closes: list[float],
) -> None:
    with pytest.raises(ValueError, match="closes must be positive and finite"):
        m.hold_cell(np.asarray(closes), cost=0.001)


def test_hold_cell_cost_at_valid_boundary_keeps_exact_hold_semantics() -> None:
    closes = np.asarray([0.01, 100.0, 1000.0])
    actual = m.hold_cell(closes, cost=0.001)
    assert actual["valid"] is True
    assert actual["trades"] == 1
    assert actual["equity"][-1] == pytest.approx(1000.0 / 0.01 / 1.001)


# ----------------------------------------------------------------------------- Amendment 3 stage runs

DEC = datetime(2026, 12, 5, 9, tzinfo=UTC)
FINAL_AT = datetime(2027, 12, 2, 1, tzinfo=UTC)


def _root_claims(ledger: Path) -> list[str]:
    return sorted(p.name for p in ledger.glob("*.claim"))


def test_forward_runs_with_only_the_old_defective_backcast_record(
    env: dict[str, Any],
) -> None:
    _seed_defective_backcast(env["ledger"])
    r = _run(env, "FORWARD_INTERIM", DEC)
    assert r["label"] in {"INTERIM_NOT_A_DECISION", "STOPPED_EARLY", "INVALID"}
    assert r["window"]["decision_start"] == "2026-12-01"
    assert r["window"]["decision_end"] == "2026-12-04"
    assert r["disclosed_prior"]["role"] == "disclosed_prior_defective_not_evidence"
    assert r["disclosed_prior"]["digest"] == m.DEFECTIVE_BACKCAST_DIGEST
    assert "backcast_ledger" not in r
    # the stage ran without any same-digest BACKCAST claim or result
    ns = env["ledger"] / env["digest"]
    assert not (ns / "BACKCAST.claim").exists()
    assert not (ns / "BACKCAST.result.json").exists()
    assert (ns / "FORWARD_INTERIM_2026-12.claim").exists()
    assert "FORWARD_INTERIM_2026-12.claim" in _root_claims(env["ledger"])


def test_final_uses_forward_labels_only_and_window(env: dict[str, Any]) -> None:
    _seed_defective_backcast(env["ledger"])
    fin = _run(env, "FORWARD_FINAL", FINAL_AT)
    assert fin["label"] in {
        "TREND_FORWARD_SUPPORTED",
        "TREND_FORWARD_NOT_SUPPORTED",
        "INVALID",
    }
    assert fin["window"]["decision_start"] == "2026-12-01"
    assert fin["window"]["decision_end"] == "2027-11-30"
    assert fin["disclosed_prior"]["label"] == "TREND_BACKCAST_ONLY"
    assert fin["label"] != fin["disclosed_prior"]["label"]


def test_forward_refuses_without_the_disclosed_prior(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="disclosed prior missing"):
        _run(env, "FORWARD_INTERIM", DEC)
    assert not env["ledger"].exists()
    ns = env["ledger"] / m.DEFECTIVE_BACKCAST_DIGEST
    m.claim_stage("BACKCAST", m.DEFECTIVE_BACKCAST_DIGEST, COMMIT, 1, NOW, ns)
    with pytest.raises(SystemExit, match="disclosed prior missing"):
        _run(env, "FORWARD_INTERIM", DEC)  # claim without result
    assert _root_claims(env["ledger"]) == []


def test_forward_refuses_a_mismatched_prior_record(env: dict[str, Any]) -> None:
    ns = _seed_defective_backcast(env["ledger"], label="TREND_NOT_SUPPORTED")
    with pytest.raises(SystemExit, match="does not match the recorded defective"):
        m.check_disclosed_prior(env["ledger"])
    (ns / "BACKCAST.result.json").unlink()
    m.write_ledger_result("BACKCAST", "other", "TREND_BACKCAST_ONLY", True, "s", ns)
    with pytest.raises(SystemExit, match="does not match the recorded defective"):
        m.check_disclosed_prior(env["ledger"])


@pytest.mark.parametrize("stage", ["BACKCAST", "FORWARD_INTERIM"])
def test_any_backcast_attempt_is_refused_with_the_old_record_present(
    env: dict[str, Any], stage: str
) -> None:
    _seed_defective_backcast(env["ledger"])
    if stage == "BACKCAST":
        with pytest.raises(SystemExit, match="BACKCAST result already exists"):
            _run(env, "BACKCAST", NOW)
    else:
        _run(env, "FORWARD_INTERIM", DEC)  # forward never creates a BACKCAST artefact
    assert not (env["ledger"] / env["digest"] / "BACKCAST.claim").exists()
    assert not (env["ledger"] / env["digest"] / "BACKCAST.result.json").exists()
    assert "BACKCAST.claim" not in _root_claims(env["ledger"])


def test_backcast_is_refused_even_with_an_empty_ledger(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="BACKCAST is consumed"):
        _run(env, "BACKCAST", NOW)
    assert not env["ledger"].exists()


def test_backcast_is_refused_under_a_new_digest_namespace(env: dict[str, Any]) -> None:
    (env["ledger"] / "other-digest").mkdir(parents=True)
    m.write_ledger_result(
        "BACKCAST",
        "other-digest",
        "TREND_NOT_SUPPORTED",
        False,
        "s",
        env["ledger"] / "other-digest",
    )
    with pytest.raises(SystemExit, match="BACKCAST result already exists"):
        _run(env, "BACKCAST", NOW)
    assert not (env["ledger"] / env["digest"]).exists()


def test_full_sequence_interims_then_final_are_single_use(
    env: dict[str, Any],
) -> None:
    _seed_defective_backcast(env["ledger"])
    _run(env, "FORWARD_INTERIM", DEC)
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_INTERIM", DEC.replace(day=20), tag="dup")
    r2 = _run(env, "FORWARD_INTERIM", datetime(2027, 1, 3, tzinfo=UTC), tag="jan")
    assert r2["claim_name"] == "FORWARD_INTERIM_2027-01"
    fin = _run(env, "FORWARD_FINAL", FINAL_AT)
    assert fin["window"]["decision_end"] == "2027-11-30"
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_FINAL", datetime(2027, 12, 3, tzinfo=UTC), tag="dup")


# ----------------------------------------------------------------------------- single use across digests


def test_single_use_holds_across_digests_at_the_root(env: dict[str, Any]) -> None:
    _seed_defective_backcast(env["ledger"])
    # a stage claimed at the root by an earlier digest, with no per-digest trace at all
    m.claim_stage("FORWARD_FINAL", "old", COMMIT, 1, FINAL_AT, env["ledger"])
    with pytest.raises(SystemExit, match="already claimed at the ledger root"):
        _run(env, "FORWARD_FINAL", FINAL_AT)
    assert not (env["ledger"] / env["digest"]).exists()


@pytest.mark.parametrize("kind", ["claim", "result", "ledger_row"])
def test_single_use_holds_across_digest_namespaces(
    env: dict[str, Any], kind: str
) -> None:
    _seed_defective_backcast(env["ledger"])
    old = env["ledger"] / "old-digest"
    if kind == "claim":
        m.claim_stage("FORWARD_INTERIM_2026-12", "old", COMMIT, 1, DEC, old)
    elif kind == "result":
        (old).mkdir(parents=True)
        (old / "FORWARD_INTERIM_2026-12.result.json").write_text("{}", encoding="utf-8")
    else:
        old.mkdir(parents=True)
        (old / "x.result.json").write_text(
            json.dumps({"stage": "FORWARD_INTERIM_2026-12", "label": "x"}),
            encoding="utf-8",
        )
    with pytest.raises(SystemExit, match="already claimed under digest old-digest"):
        _run(env, "FORWARD_INTERIM", DEC)
    assert not (env["ledger"] / env["digest"]).exists()
    # another month and the final remain open
    _run(env, "FORWARD_INTERIM", datetime(2027, 1, 3, tzinfo=UTC), tag="jan")


def test_stopped_early_under_any_digest_blocks_later_runs(env: dict[str, Any]) -> None:
    _seed_defective_backcast(env["ledger"])
    old = env["ledger"] / "old-digest"
    m.claim_stage("FORWARD_INTERIM_2026-12", "old", COMMIT, 1, DEC, old)
    m.write_ledger_result(
        "FORWARD_INTERIM_2026-12", "old", "STOPPED_EARLY", False, "s", old
    )
    for stage, now in (
        ("FORWARD_INTERIM", datetime(2027, 1, 3, tzinfo=UTC)),
        ("FORWARD_FINAL", FINAL_AT),
    ):
        with pytest.raises(SystemExit, match="line stopped by FORWARD_INTERIM_2026-12"):
            _run(env, stage, now, tag=stage)
    assert not (env["ledger"] / env["digest"]).exists()


def test_old_backcast_only_label_does_not_stop_the_line(env: dict[str, Any]) -> None:
    _seed_defective_backcast(env["ledger"])
    m.check_cross_digest(env["ledger"], "FORWARD_FINAL")


def test_cross_digest_check_on_missing_root_passes(tmp_path: Path) -> None:
    m.check_cross_digest(tmp_path / "nope", "FORWARD_FINAL")


# ----------------------------------------------------------------------------- pre-claim checks


def test_connectivity_probe_failure_refuses_before_any_claim(
    env: dict[str, Any],
) -> None:
    _seed_defective_backcast(env["ledger"])

    def down(url: str) -> bytes:
        raise OSError("unreachable")

    with pytest.raises(SystemExit, match="connectivity probe failed before any claim"):
        m.run_stage(
            "FORWARD_INTERIM",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=DEC,
            fetch=down,
            gh_fetch=env["gh"],
            ledger_dir=env["ledger"],
        )
    assert _root_claims(env["ledger"]) == []
    assert not (env["ledger"] / env["digest"]).exists()


@pytest.mark.parametrize("body", [b"[]", b"not json", b'{"x": 1}'])
def test_connectivity_probe_rejects_empty_or_malformed(body: bytes) -> None:
    with pytest.raises(SystemExit, match="before any claim"):
        m.connectivity_probe(lambda url: body)


def test_connectivity_probe_runs_source_close_admission() -> None:
    bad = _binance([date(2026, 10, 1)])
    bad[0][6] = bad[0][6] - 1  # off-by-one native close
    with pytest.raises(SystemExit, match="before any claim"):
        m.connectivity_probe(lambda url: json.dumps(bad).encode())


def test_connectivity_probe_reads_only_data_before_2026_11_01() -> None:
    urls: list[str] = []

    def f(url: str) -> bytes:
        urls.append(url)
        return fake_market(url)

    m.connectivity_probe(f)
    assert len(urls) == 1
    q = {
        k: int(v[0])
        for k, v in parse_qs(urlparse(urls[0]).query).items()
        if v[0].isdigit()
    }
    limit_ms = int(datetime(2026, 11, 1, tzinfo=UTC).timestamp()) * 1000
    assert q["startTime"] < q["endTime"] < limit_ms


def test_authorization_failure_precedes_probe_and_claim(env: dict[str, Any]) -> None:
    _seed_defective_backcast(env["ledger"])
    calls: list[str] = []

    def spy(url: str) -> bytes:
        calls.append(url)
        return fake_market(url)

    env2 = {**env, "gh": _gh(relay_body="no quote")}
    with pytest.raises(SystemExit, match="does not quote"):
        m.run_stage(
            "FORWARD_INTERIM",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=DEC,
            fetch=spy,
            gh_fetch=env2["gh"],
            ledger_dir=env["ledger"],
        )
    assert calls == [] and _root_claims(env["ledger"]) == []


def test_root_claim_is_exclusive_and_written_before_the_per_digest_claim(
    env: dict[str, Any],
) -> None:
    _seed_defective_backcast(env["ledger"])
    _run(env, "FORWARD_INTERIM", DEC)
    root = json.loads((env["ledger"] / "FORWARD_INTERIM_2026-12.claim").read_text())
    per = json.loads(
        (env["ledger"] / env["digest"] / "FORWARD_INTERIM_2026-12.claim").read_text()
    )
    assert root == per and root["protocol_digest"] == env["digest"]
    marker = env["ledger"] / f"g4-{root['authorization_comment_id']}.claim"
    assert json.loads(marker.read_text(encoding="utf-8")) == root
    with pytest.raises(SystemExit, match="already claimed"):
        m.claim_stage("FORWARD_INTERIM_2026-12", "x", COMMIT, 1, DEC, env["ledger"])


def test_interim_before_an_evaluated_day_refuses_without_claiming(
    env: dict[str, Any],
) -> None:
    _seed_defective_backcast(env["ledger"])
    with pytest.raises(SystemExit, match="no evaluated day complete yet"):
        _run(env, "FORWARD_INTERIM", datetime(2026, 12, 2, 9, tzinfo=UTC))
    assert _root_claims(env["ledger"]) == []
    # the month is not burned: the same month can still run later
    _run(env, "FORWARD_INTERIM", DEC, tag="later")


# Regression: do not allow a POSIX-relative Windows literal to fork the single-use ledger.
def test_stage_refuses_relative_ledger_before_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    when = datetime(2026, 12, 5, 9, tzinfo=UTC)
    for dirname in ("cwd1", "cwd2"):
        root = tmp_path / dirname
        root.mkdir()
        monkeypatch.chdir(root)
        # The argument must be rejected before any API call, claim or data read.
        with pytest.raises(SystemExit, match="absolute.*ledger|ledger.*absolute"):
            m.run_stage(
                "FORWARD_INTERIM",
                URL,
                root / "work",
                root / "out",
                now=when,
                ledger_dir=Path("relative_ledger"),
                fetch=lambda _url: (_ for _ in ()).throw(AssertionError("data read")),
            )
        assert not any(root.iterdir())


def test_claim_identity_is_utc_and_naive_datetimes_refuse() -> None:
    from datetime import timedelta, timezone

    jst = timezone(timedelta(hours=9))
    stage = m.STAGES["FORWARD_INTERIM"]
    assert (
        m.claim_name(stage, datetime(2026, 12, 1, 0, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-11"
    )
    assert (
        m.claim_name(stage, datetime(2026, 12, 1, 9, 30, tzinfo=jst))
        == "FORWARD_INTERIM_2026-12"
    )
    with pytest.raises(SystemExit, match="timezone-aware"):
        m.claim_name(stage, datetime(2026, 12, 1))
    with pytest.raises(SystemExit, match="timezone-aware"):
        m.check_clock(stage, datetime(2026, 12, 1))


def test_posix_windows_legacy_default_must_not_be_accepted() -> None:
    import os

    if os.name != "nt":
        assert not m.LEDGER_DIR.is_absolute()


@pytest.mark.parametrize(
    "relay_claim",
    [
        "",
        "G4 claim: FORWARD_INTERIM_2027-01",
        "G4 claim: FORWARD_FINAL\nG4 claim: FORWARD_INTERIM_2027-01",
        "G4 claim: FORWARD_FINAL\nG4 claim: final",
    ],
)
def test_btc_g4_exact_stage_required_before_claim(
    env: dict[str, Any], relay_claim: str
) -> None:
    _seed_defective_backcast(env["ledger"])

    def unexpected(_url: str) -> bytes:
        raise AssertionError("no probe or data request without exact G4 stage")

    with pytest.raises(SystemExit, match="G4 claim"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=FINAL_AT,
            fetch=unexpected,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\n{relay_claim}"
            ),
            ledger_dir=env["ledger"],
        )
    assert not (env["ledger"] / "FORWARD_FINAL.claim").exists()


def test_btc_g4_comment_cannot_be_reused_for_another_stage(env: dict[str, Any]) -> None:
    root = env["ledger"]
    _seed_defective_backcast(root)
    prior = m.claim_stage(
        "FORWARD_INTERIM_2027-01",
        "9" * 64,
        COMMIT,
        7040999999,
        DEC,
        root / ("9" * 64),
    )
    before = prior.read_bytes()

    def unexpected(_url: str) -> bytes:
        raise AssertionError("no probe or data request after known G4 consumption")

    with pytest.raises(SystemExit, match="already consumed"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=FINAL_AT,
            fetch=unexpected,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: FORWARD_FINAL"
            ),
            ledger_dir=root,
        )
    assert prior.read_bytes() == before
    assert not (root / "FORWARD_FINAL.claim").exists()


@pytest.mark.parametrize("kind", ["stage", "marker"])
def test_claim_is_complete_before_exclusive_publication(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    real_link = os.link
    published: list[Path] = []

    def observed_link(src: Any, dst: Any) -> None:
        source, target = Path(src), Path(dst)
        assert source.parent == target.parent == root
        assert source.suffix != ".claim" and not target.exists()
        assert (
            json.loads(source.read_text(encoding="utf-8"))["authorization_comment_id"]
            == 42
        )
        real_link(src, dst)
        published.append(target)

    monkeypatch.setattr(os, "link", observed_link)
    if kind == "stage":
        target = m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
    else:
        target = m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now)
    assert published == [target]
    assert list(root.iterdir()) == [target]
    assert (
        json.loads(target.read_text(encoding="utf-8"))["authorization_comment_id"] == 42
    )


@pytest.mark.parametrize("kind", ["stage", "marker"])
def test_failed_claim_flush_never_publishes_partial_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)

    def fail(_fd: int) -> None:
        raise OSError("synthetic fsync failure")

    monkeypatch.setattr(os, "fsync", fail)
    with pytest.raises(OSError, match="synthetic fsync failure"):
        if kind == "stage":
            m.claim_stage("FORWARD_FINAL", "d" * 64, COMMIT, 42, now, root)
        else:
            m.reserve_g4_comment(root, 42, "FORWARD_FINAL", "d" * 64, COMMIT, now)
    assert list(root.iterdir()) == []


def test_concurrent_g4_publication_has_one_winner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "ledger"
    now = datetime(2027, 1, 20, tzinfo=UTC)
    real_link = os.link
    gate = Barrier(2, timeout=10)

    def simultaneous_link(src: Any, dst: Any) -> None:
        gate.wait()
        real_link(src, dst)

    monkeypatch.setattr(os, "link", simultaneous_link)

    def reserve(stage: str) -> str:
        try:
            m.reserve_g4_comment(root, 42, stage, "d" * 64, COMMIT, now)
        except SystemExit as exc:
            assert "already consumed" in str(exc)
            return "refused"
        return stage

    stages = ["FORWARD_INTERIM_2027-01", "FORWARD_FINAL"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(reserve, stages))
    assert results.count("refused") == 1
    winner = next(x for x in results if x != "refused")
    marker = root / "g4-42.claim"
    assert list(root.iterdir()) == [marker]
    assert json.loads(marker.read_text(encoding="utf-8"))["stage"] == winner


@pytest.mark.parametrize(
    "history", ["linked", "broken-link", "zero-id", "negative-id", "duplicate-id"]
)
def test_g4_invalid_history_refuses_before_claim_or_fetch(
    env: dict[str, Any], history: str
) -> None:
    root = env["ledger"]
    _seed_defective_backcast(root)
    ns = root / ("9" * 64)
    if history in {"linked", "broken-link"}:
        outside = env["tmp"] / "external-history"
        if history == "linked":
            outside.mkdir()
            (outside / "old.claim").write_text(
                '{"authorization_comment_id":7040999999}', encoding="utf-8"
            )
        try:
            ns.symlink_to(outside, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlinks unavailable: {exc}")
    else:
        ns.mkdir()
        record = {
            "zero-id": '{"authorization_comment_id":0}',
            "negative-id": '{"authorization_comment_id":-1}',
            "duplicate-id": '{"authorization_comment_id":7040999999,"authorization_comment_id":1}',
        }[history]
        (ns / "old.claim").write_text(record, encoding="utf-8")

    def unexpected(_url: str) -> bytes:
        raise AssertionError("no market request after invalid ledger history")

    with pytest.raises(SystemExit, match="historical G4|G4 ledger"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=unexpected,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: FORWARD_FINAL"
            ),
            ledger_dir=env["ledger"],
        )
    assert not (root / "FORWARD_FINAL.claim").exists()
    assert not (root / "g4-7040999999.claim").exists()
    assert not (env["tmp"] / "w").exists()


def test_btc_g4_accepts_single_crlf_stage_claim() -> None:
    digest = "d" * 64
    body = (
        f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}\r\nG4 claim: FORWARD_FINAL\r\n"
    )
    receipt = m.authorization_receipt(
        URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
    )
    assert receipt["g4_relay"]["comment_id"] == 7040999999


def test_unreadable_history_refuses_before_external_access(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = env["ledger"]
    ns = root / ("9" * 64)
    ns.mkdir(parents=True)
    real_iterdir = Path.iterdir

    def inaccessible(path: Path) -> Any:
        if path == ns:
            raise PermissionError("synthetic historical directory error")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", inaccessible)

    def unexpected(_arg: Any) -> Any:
        raise AssertionError("no external request after unsafe ledger")

    with pytest.raises(SystemExit, match="unreadable historical G4 ledger"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=datetime(2027, 12, 2, 1, tzinfo=UTC),
            fetch=unexpected,
            gh_fetch=unexpected,
            ledger_dir=env["ledger"],
        )
    assert not (root / "FORWARD_FINAL.claim").exists()
    assert not (env["tmp"] / "w").exists()


def test_public_run_rechecks_g4_after_root_claim(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    root = env["ledger"]
    _seed_defective_backcast(root)
    now = datetime(2027, 12, 2, 1, tzinfo=UTC)
    real_claim = m.claim_stage

    def competing_claim(*args: Any, **kwargs: Any) -> Path:
        path = real_claim(*args, **kwargs)
        if path.parent == root:
            # A competing reservation arrives after read-only preflight, before the marker.
            m._publish_claim(
                root / "g4-7040999999.claim",
                {
                    "stage": "FORWARD_INTERIM_2027-01",
                    "protocol_digest": "9" * 64,
                    "commit": COMMIT,
                    "authorization_comment_id": 7040999999,
                    "claimed_at": now.isoformat(),
                },
            )
        return path

    monkeypatch.setattr(m, "claim_stage", competing_claim)
    # The trend probe is deliberately simulated; stage-window access is forbidden.
    if hasattr(m, "connectivity_probe"):
        monkeypatch.setattr(m, "connectivity_probe", lambda _fetch: None)

    def unexpected(_url: str) -> bytes:
        raise AssertionError("stage data must not be read after a G4 conflict")

    with pytest.raises(SystemExit, match="already consumed"):
        m.run_stage(
            "FORWARD_FINAL",
            URL,
            env["tmp"] / "w",
            env["tmp"] / "o",
            now=now,
            fetch=unexpected,
            gh_fetch=_gh(
                relay_body=f"G4 identity: {m.PROTOCOL_ID} {env['digest']} {COMMIT}\nG4 claim: FORWARD_FINAL"
            ),
            ledger_dir=env["ledger"],
        )
    assert (root / "FORWARD_FINAL.claim").exists()
    assert (root / "g4-7040999999.claim").exists()
    assert not (root / env["digest"] / "FORWARD_FINAL.claim").exists()
    assert not (env["tmp"] / "w").exists()


@pytest.mark.parametrize(
    "mutation",
    (
        "long_id",
        "long_digest",
        "long_commit",
        "duplicate_identity",
        "conflicting_identity",
        "missing_identity",
    ),
)
def test_g4_identity_must_be_one_exact_tuple(mutation: str) -> None:
    digest = "d" * 64
    identity = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}"
    if mutation == "long_id":
        identity = identity.replace(m.PROTOCOL_ID, m.PROTOCOL_ID + "_other")
    elif mutation == "long_digest":
        identity = identity.replace(digest, digest + "e")
    elif mutation == "long_commit":
        identity = identity.replace(COMMIT, "b" + COMMIT)
    elif mutation == "duplicate_identity":
        identity += "\n" + identity
    elif mutation == "conflicting_identity":
        identity += "\n" + identity.replace(digest, "e" * 64)
    else:
        identity = f"{m.PROTOCOL_ID} {digest} {COMMIT}"
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(
            URL,
            digest,
            COMMIT,
            "FORWARD_FINAL",
            _gh(relay_body=identity + "\nG4 claim: FORWARD_FINAL"),
        )


@pytest.mark.parametrize("digest", ["d" * 63, "d" * 65, "D" * 64, "z" * 64])
def test_g4_invalid_digest_refuses_before_comment_fetch(digest: str) -> None:
    def unexpected_fetch(_cid: int) -> dict[str, Any]:
        pytest.fail("invalid digest must not reach comment fetch")

    with pytest.raises(SystemExit, match="full 64-hex"):
        m.authorization_receipt(URL, digest, COMMIT, "FORWARD_FINAL", unexpected_fetch)


@pytest.mark.parametrize(
    "separator", ["\r", "\v", "\f", "\x1c", "\x1d", "\x1e", "\x85", "\u2028", "\u2029"]
)
def test_g4_refuses_noncanonical_record_separators(separator: str) -> None:
    digest = "d" * 64
    body = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}{separator}G4 claim: FORWARD_FINAL"
    with pytest.raises(SystemExit, match="unsupported line separator"):
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )


@pytest.mark.parametrize("field", ["identity", "claim"])
def test_g4_refuses_malformed_second_record(field: str) -> None:
    digest = "d" * 64
    identity = f"G4 identity: {m.PROTOCOL_ID} {digest} {COMMIT}"
    claim = "G4 claim: FORWARD_FINAL"
    extra = (
        f"G4 identity {m.PROTOCOL_ID} {'e' * 64} {COMMIT}"
        if field == "identity"
        else "G4 claim FORWARD_INTERIM_2026-12"
    )
    body = "\n".join([identity, claim, extra])
    with pytest.raises(SystemExit, match="does not quote|must name exactly"):
        m.authorization_receipt(
            URL, digest, COMMIT, "FORWARD_FINAL", _gh(relay_body=body)
        )
