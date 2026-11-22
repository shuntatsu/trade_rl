"""Synthetic-data tests for tools/btc_trend_ensemble.py (no network, no market data)."""

from __future__ import annotations

import json
import math
import re
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
import pytest

from tools import btc_trend_ensemble as m

DIGEST_PIN = "9691bd342684bed3d079c03ea98341c7d3af47da1526180da2eef703638e323c"
COMMIT = "a" * 40
URL = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040999999"


# ----------------------------------------------------------------------------- digest


def test_digest_pin() -> None:
    assert m.protocol_digest() == DIGEST_PIN
    assert re.fullmatch(r"[0-9a-f]{64}", m.protocol_digest())


def test_constants_frozen() -> None:
    assert m.PROTOCOL_ID == "btc_trend_ensemble_v1"
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
        date(2026, 11, 23),
        date(2027, 11, 22),
    )
    assert fin.not_before == datetime(2027, 11, 24, tzinfo=UTC)
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
    # entry trade of |1.0 - 0| at close 199, charged before day 200's return
    assert cell["daily_returns"][199] == pytest.approx(
        (1 - 0.01) * (c[200] / c[199]) - 1
    )
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
    assert cell["trades"] == 1 and cell["turnover"] == 1.0
    assert cell["equity"][-1] == pytest.approx((1 - 0.001) * c[-1] / c[10])
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


def test_decide_backcast_branches() -> None:
    assert m.decide("BACKCAST", _cells())["label"] == "TREND_BACKCAST_ONLY"
    assert (
        m.decide("BACKCAST", _cells(TREND__BASE=_cell(ret=-0.01)))["label"]
        == "TREND_NOT_SUPPORTED"
    )
    inv = m.decide("BACKCAST", _cells(HOLD__BASE=_cell(valid=False)))
    assert inv["label"] == "INVALID" and inv["invalid_cells"] == ["HOLD__BASE"]


def test_decide_threshold_strictness() -> None:
    ok = lambda **kw: m.decide("BACKCAST", _cells(**kw))["supported"]  # noqa: E731
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


def test_decide_final_branches() -> None:
    assert m.decide("FORWARD_FINAL", _cells(), True)["label"] == "TREND_SUPPORTED"
    assert m.decide("FORWARD_FINAL", _cells(), False)["label"] == "TREND_NOT_SUPPORTED"
    bad = _cells(TREND__DELAY=_cell(ret=-0.1))
    assert m.decide("FORWARD_FINAL", bad, True)["label"] == "TREND_NOT_SUPPORTED"
    inv = _cells(TREND__BASE=_cell(valid=False))
    assert m.decide("FORWARD_FINAL", inv, True)["label"] == "INVALID"
    with pytest.raises(ValueError):
        m.decide("FORWARD_FINAL", _cells())


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
        # Real Bitstamp semantics (observed 2026-11-22): with start and end both given, the
        # API returns the LAST ``limit`` bars at or before ``end`` (Amendment 1).
        ts = [t for t in range((lo + 86399) // 86400 * 86400, hi + 1, 86400)][-lim:]
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
    with pytest.raises(SystemExit, match="no progress"):
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
        relay_body if relay_body is not None else f"{m.PROTOCOL_ID} {digest} {COMMIT}"
    )

    def fetch(cid: int) -> dict[str, Any]:
        if cid == m.DATA_SCOPE_COMMENT_ID:
            return _comment("scope", **over.get("scope", {}))
        return _comment(relay_body, **over.get("relay", {}))

    return fetch


def test_authorization_ok() -> None:
    rec = m.authorization_receipt(URL, "d" * 64, COMMIT, _gh())
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
        m.authorization_receipt(url, "d" * 64, COMMIT, _gh())


def test_authorization_owner_and_issue_binding() -> None:
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay={"login": "mallory"}))
    with pytest.raises(SystemExit, match="not owner-authored"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(scope={"login": "mallory"}))
    other = "https://api.github.com/repos/shuntatsu/trade_rl/issues/8100"
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay={"issue_url": other}))
    with pytest.raises(SystemExit, match="not on Issue #810"):
        m.authorization_receipt(
            URL, "d" * 64, COMMIT, _gh(scope={"issue_url": "x/issues/810"})
        )


def test_authorization_edit_between_fetches() -> None:
    calls: dict[int, int] = {}

    def fetch(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        body = f"{m.PROTOCOL_ID} {'d' * 64} {COMMIT}"
        if cid != m.DATA_SCOPE_COMMENT_ID and calls[cid] == 2:
            body += " edited"
        return _comment(body)

    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, fetch)

    def fetch_ts(cid: int) -> dict[str, Any]:
        calls[cid] = calls.get(cid, 0) + 1
        return _comment("scope", updated=f"t{calls[cid]}")

    calls.clear()
    with pytest.raises(SystemExit, match="changed between fetches"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, fetch_ts)


@pytest.mark.parametrize(
    "body",
    [
        f"{m.PROTOCOL_ID} {COMMIT}",
        f"{m.PROTOCOL_ID} {'d' * 64}",
        f"{'d' * 64} {COMMIT}",
        f"{m.PROTOCOL_ID} {'d' * 64} {COMMIT[:12]}",
    ],
)
def test_authorization_relay_must_quote_id_digest_commit(body: str) -> None:
    with pytest.raises(SystemExit, match="does not quote"):
        m.authorization_receipt(URL, "d" * 64, COMMIT, _gh(relay_body=body))


def test_authorization_requires_full_commit() -> None:
    with pytest.raises(SystemExit, match="40-hex"):
        m.authorization_receipt(URL, "d" * 64, "abc1234", _gh())


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
        m.check_clock(inter, datetime(2026, 11, 30, 23, 59, tzinfo=UTC))
    m.check_clock(inter, datetime(2026, 12, 1, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before 2027-11-24"):
        m.check_clock(fin, datetime(2027, 11, 23, 23, 59, tzinfo=UTC))
    m.check_clock(fin, datetime(2027, 11, 24, tzinfo=UTC))
    with pytest.raises(SystemExit, match="at or after"):
        m.check_clock(inter, datetime(2027, 11, 24, tzinfo=UTC))
    m.check_clock(m.STAGES["BACKCAST"], datetime(2026, 11, 22, tzinfo=UTC))


def test_stage_window_interim_ends_at_last_complete_day() -> None:
    inter = m.STAGES["FORWARD_INTERIM"]
    assert m.stage_window(inter, datetime(2026, 12, 5, 13, tzinfo=UTC)) == (
        date(2026, 5, 1),
        date(2026, 12, 4),
    )
    assert m.stage_window(m.STAGES["FORWARD_FINAL"], datetime(2028, 1, 1, tzinfo=UTC))[
        1
    ] == date(2027, 11, 22)


# ----------------------------------------------------------------------------- stage runs (synthetic)


@pytest.fixture
def env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(m, "harness_commit", lambda: COMMIT)
    digest = m.protocol_digest()
    gh = _gh(relay_body=f"{m.PROTOCOL_ID} {digest} {COMMIT}")
    return {"ledger": tmp_path / "ledger", "tmp": tmp_path, "gh": gh, "digest": digest}


def _run(
    env: dict[str, Any], stage: str, now: datetime, tag: str = "x"
) -> dict[str, Any]:
    return m.run_stage(
        stage,
        URL,
        env["tmp"] / f"work_{stage}_{tag}",
        env["tmp"] / f"out_{stage}_{tag}",
        now=now,
        fetch=fake_market,
        gh_fetch=env["gh"],
        ledger_dir=env["ledger"],
    )


NOW = datetime(2026, 12, 5, 9, tzinfo=UTC)


def test_backcast_end_to_end_and_write_once(env: dict[str, Any]) -> None:
    res = _run(env, "BACKCAST", NOW)
    assert res["label"] in {"TREND_BACKCAST_ONLY", "TREND_NOT_SUPPORTED"}
    assert set(res["cells"]) == set(m.CELLS)
    assert res["data"]["days"] == 1491 and len(res["data"]["pages"]) == 2
    out = env["tmp"] / "out_BACKCAST_x" / "result.json"
    led = json.loads(
        (env["ledger"] / env["digest"] / "BACKCAST.result.json").read_text()
    )
    assert led["result_sha256"] == m.sha256_file(out) and led["label"] == res["label"]
    assert json.loads(out.read_text())["protocol_digest"] == env["digest"]
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "BACKCAST", NOW, tag="again")
    with pytest.raises(SystemExit, match="output directory exists"):
        m.run_stage(
            "BACKCAST",
            URL,
            env["tmp"] / "w2",
            out,
            now=NOW,
            fetch=fake_market,
            gh_fetch=env["gh"],
            ledger_dir=env["ledger"],
        )


def test_forward_stages_require_backcast_ledger(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="no BACKCAST ledger claim"):
        _run(env, "FORWARD_INTERIM", NOW)
    assert not (
        env["ledger"] / env["digest"] / "FORWARD_INTERIM_2026-12.claim"
    ).exists()  # refused before claiming
    m.claim_stage(
        "BACKCAST", env["digest"], COMMIT, 1, NOW, env["ledger"] / env["digest"]
    )
    with pytest.raises(SystemExit, match="no BACKCAST ledger result"):
        _run(env, "FORWARD_INTERIM", NOW)


def test_forward_requires_same_digest_and_supported_backcast(
    env: dict[str, Any],
) -> None:
    led = env["ledger"] / env["digest"]
    m.claim_stage("BACKCAST", "other", COMMIT, 1, NOW, led)
    with pytest.raises(SystemExit, match="another protocol digest"):
        m.check_backcast_ledger(env["digest"], led)
    (led / "BACKCAST.claim").unlink()
    m.claim_stage("BACKCAST", env["digest"], COMMIT, 1, NOW, led)
    m.write_ledger_result("BACKCAST", "other", "TREND_BACKCAST_ONLY", True, "s", led)
    with pytest.raises(SystemExit, match="another digest or stage"):
        m.check_backcast_ledger(env["digest"], led)
    (led / "BACKCAST.result.json").unlink()
    m.write_ledger_result(
        "BACKCAST", env["digest"], "TREND_NOT_SUPPORTED", False, "s", led
    )
    with pytest.raises(SystemExit, match="stops this version"):
        m.check_backcast_ledger(env["digest"], led)
    with pytest.raises(SystemExit, match="stops this version"):
        _run(env, "FORWARD_FINAL", datetime(2027, 11, 25, tzinfo=UTC))
    (led / "BACKCAST.result.json").unlink()
    m.write_ledger_result(
        "BACKCAST", env["digest"], "TREND_BACKCAST_ONLY", True, "s", led
    )
    assert m.check_backcast_ledger(env["digest"], led)["label"] == "TREND_BACKCAST_ONLY"


def test_clock_refusal_precedes_claim(env: dict[str, Any]) -> None:
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_FINAL", datetime(2027, 11, 23, tzinfo=UTC))
    with pytest.raises(SystemExit, match="may not run before"):
        _run(env, "FORWARD_INTERIM", datetime(2026, 11, 29, tzinfo=UTC))
    assert not env["ledger"].exists()


def test_authorization_refusal_precedes_claim(env: dict[str, Any]) -> None:
    env2 = {**env, "gh": _gh(relay_body="no quote")}
    with pytest.raises(SystemExit, match="does not quote"):
        _run(env2, "BACKCAST", NOW)
    assert not env["ledger"].exists()


def test_full_sequence_backcast_interim_final(
    env: dict[str, Any], monkeypatch: pytest.MonkeyPatch
) -> None:
    # force SUPPORTED(BACKCAST) so the forward stages can be exercised on synthetic data
    real = m.decide

    def forced(
        stage: Any, cells: Any, backcast_supported: Any = None
    ) -> dict[str, Any]:
        out = real(stage, cells, backcast_supported)
        if m.STAGES["BACKCAST"] is stage or stage == "BACKCAST":
            out.update(label=m.BACKCAST_LABEL_OK, supported=True)
        return out

    monkeypatch.setattr(m, "decide", forced)
    assert _run(env, "BACKCAST", NOW)["label"] == m.BACKCAST_LABEL_OK
    monkeypatch.setattr(m, "decide", real)
    dec = datetime(2026, 12, 5, 9, tzinfo=UTC)
    r = _run(env, "FORWARD_INTERIM", dec)
    assert r["label"] in {"INTERIM_NOT_A_DECISION", "STOPPED_EARLY", "INVALID"}
    assert (
        r["window"]["decision_end"] == "2026-12-04" and r["data"]["source"] == "binance"
    )
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_INTERIM", dec.replace(day=20), tag="dup")
    r2 = _run(env, "FORWARD_INTERIM", datetime(2027, 1, 3, tzinfo=UTC), tag="jan")
    assert r2["claim_name"] == "FORWARD_INTERIM_2027-01"
    fin = _run(env, "FORWARD_FINAL", datetime(2027, 11, 24, 1, tzinfo=UTC))
    assert fin["label"] in {"TREND_SUPPORTED", "TREND_NOT_SUPPORTED", "INVALID"}
    assert fin["window"]["decision_end"] == "2027-11-22"
    assert fin["backcast_ledger"]["label"] == m.BACKCAST_LABEL_OK
    with pytest.raises(SystemExit, match="already claimed"):
        _run(env, "FORWARD_FINAL", datetime(2027, 11, 25, tzinfo=UTC), tag="dup")


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
