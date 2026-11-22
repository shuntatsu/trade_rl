"""Vol-targeted BTC trend ensemble, protocol ``btc_trend_ensemble_v1``.

Protocol: docs/specs/btc-trend-ensemble.md (the spec is the authority for every constant).

Stages (``STAGES``) read public daily bars (Bitstamp BTC/USD for BACKCAST, Binance BTCUSDT
for FORWARD), replay five frozen cells and write a write-once result. A local ledger claim
is created exclusively before any data is fetched. Importing this module needs only the
standard library and numpy; the network code runs only in real ``run`` invocations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import subprocess
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "btc_trend_ensemble_v1"

# ----------------------------------------------------------------------------- rule constants

LOOKBACKS = (30, 50, 75, 100, 150, 200)
VOL_WINDOW = 30
TARGET_VOL = 0.20
DEADBAND = 0.05
DAYS_PER_YEAR = 365.0
WARMUP_CLOSES = max(LOOKBACKS)
IDENTITY_TOL = 1e-9
EARLY_STOP_DD = 0.25
MAX_DD = 0.25
MIN_SHARPE = 0.5
TRADE_EPS = 1e-12

# name -> (kind, one-way cost, execution delay in days, gates the decision)
CELLS: dict[str, tuple[str, float, int, bool]] = {
    "TREND__BASE": ("trend", 0.0010, 1, True),
    "TREND__HIGH_COST": ("trend", 0.0020, 1, True),
    "TREND__DELAY": ("trend", 0.0010, 2, True),
    "HOLD__BASE": ("hold", 0.0010, 1, False),
    "CASH": ("cash", 0.0, 0, False),
}


@dataclass(frozen=True)
class Stage:
    name: str
    source: str  # "bitstamp" | "binance"
    load_from: date  # first daily bar to load (warm-up)
    decision_start: date  # first decision day
    decision_end: date  # last decision day (cap for interims)
    interim: bool
    final: bool
    not_before: datetime | None
    not_after: datetime | None


STAGES: dict[str, Stage] = {
    "BACKCAST": Stage(
        "BACKCAST",
        "bitstamp",
        date(2014, 6, 1),
        date(2015, 1, 1),
        date(2018, 6, 30),
        interim=False,
        final=False,
        not_before=None,
        not_after=None,
    ),
    "FORWARD_INTERIM": Stage(
        "FORWARD_INTERIM",
        "binance",
        date(2026, 5, 1),
        date(2026, 11, 23),
        date(2027, 11, 22),
        interim=True,
        final=False,
        not_before=datetime(2026, 12, 1, tzinfo=UTC),
        not_after=datetime(2027, 11, 24, tzinfo=UTC),
    ),
    "FORWARD_FINAL": Stage(
        "FORWARD_FINAL",
        "binance",
        date(2026, 5, 1),
        date(2026, 11, 23),
        date(2027, 11, 22),
        interim=False,
        final=True,
        not_before=datetime(2027, 11, 24, tzinfo=UTC),
        not_after=None,
    ),
}

BACKCAST_LABEL_OK = (
    "TREND_BACKCAST_ONLY"  # SUPPORTED(BACKCAST) and FORWARD not yet final
)

DATA_SCOPE_COMMENT_ID = 7040247247  # owner scope relay on Issue #810
OWNER_LOGIN = "shuntatsu"
REPO = "shuntatsu/trade_rl"
ISSUE_URL = f"https://api.github.com/repos/{REPO}/issues/810"
LEDGER_DIR = Path(r"C:\trade_rl_runs\btc_trend_ensemble_ledger")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_RELPATH = "docs/specs/btc-trend-ensemble.md"

USER_AGENT = "trade_rl-research/1.0"
HTTP_TIMEOUT_S = 30
PAGE_LIMIT = 1000
MAX_PAGES = 100
BITSTAMP_URL = "https://www.bitstamp.net/api/v2/ohlc/btcusd/"
BINANCE_URL = "https://api.binance.com/api/v3/klines"
DAY_S = 86400
EPOCH = date(1970, 1, 1)


# ----------------------------------------------------------------------------- digests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _constants() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "lookbacks": list(LOOKBACKS),
        "vol_window": VOL_WINDOW,
        "target_vol": TARGET_VOL,
        "deadband": DEADBAND,
        "days_per_year": DAYS_PER_YEAR,
        "identity_tol": IDENTITY_TOL,
        "early_stop_dd": EARLY_STOP_DD,
        "max_dd": MAX_DD,
        "min_sharpe": MIN_SHARPE,
        "cells": {k: list(v) for k, v in CELLS.items()},
        "stages": {
            k: [
                v.source,
                v.load_from.isoformat(),
                v.decision_start.isoformat(),
                v.decision_end.isoformat(),
                v.interim,
                v.final,
                v.not_before.isoformat() if v.not_before else None,
                v.not_after.isoformat() if v.not_after else None,
            ]
            for k, v in STAGES.items()
        },
        "backcast_label_ok": BACKCAST_LABEL_OK,
        "data_scope_comment": DATA_SCOPE_COMMENT_ID,
        "owner_login": OWNER_LOGIN,
        "issue_url": ISSUE_URL,
        "ledger_dir": str(LEDGER_DIR),
        "bitstamp_url": BITSTAMP_URL,
        "binance_url": BINANCE_URL,
        "page_limit": PAGE_LIMIT,
    }


def spec_sha256() -> str:
    """sha256 of the spec with CRLF normalized and the ``Protocol digest:`` line removed."""
    text = (REPO_ROOT / SPEC_RELPATH).read_bytes().replace(b"\r\n", b"\n")
    kept = [ln for ln in text.split(b"\n") if not ln.startswith(b"Protocol digest:")]
    return hashlib.sha256(b"\n".join(kept)).hexdigest()


def protocol_digest() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    payload = json.dumps(
        {
            "constants": _constants(),
            "harness_sha256": hashlib.sha256(source).hexdigest(),
            "spec_path": SPEC_RELPATH,
            "spec_sha256": spec_sha256(),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


# ----------------------------------------------------------------------------- rule functions


def trend_score(closes: Any) -> np.ndarray:
    """s_t = mean over LOOKBACKS of 1[close_t > SMA_N(close)_t]; 0 until 200 closes exist."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    s = np.zeros(n)
    if n < WARMUP_CLOSES:
        return s
    cs = np.concatenate(([0.0], np.cumsum(c)))
    for lb in LOOKBACKS:
        sma = (cs[lb:] - cs[:-lb]) / lb  # sma[k] is the SMA ending at index k + lb - 1
        above = np.zeros(n)
        above[lb - 1 :] = c[lb - 1 :] > sma
        s += above
    s /= len(LOOKBACKS)
    s[: WARMUP_CLOSES - 1] = 0.0
    return s


def realized_vol(closes: Any) -> np.ndarray:
    """Annualized std (ddof 0, sqrt 365) of the last VOL_WINDOW log returns; NaN before."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    out = np.full(n, np.nan)
    if n <= VOL_WINDOW:
        return out
    r = np.diff(np.log(c))  # r[k] is the return ending at index k + 1
    for t in range(VOL_WINDOW, n):
        out[t] = float(np.std(r[t - VOL_WINDOW : t], ddof=0)) * math.sqrt(DAYS_PER_YEAR)
    return out


def target_exposure(score: Any, vol: Any) -> np.ndarray:
    """e_t = s_t * min(1, TARGET_VOL / sigma_t); 0 where sigma is undefined."""
    s = np.asarray(score, dtype=float)
    v = np.asarray(vol, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        scale = np.where(v > 0.0, np.minimum(1.0, TARGET_VOL / v), 1.0)
    e = s * scale
    return np.where(np.isfinite(v), e, 0.0)


def apply_deadband(targets: Any, held0: float = 0.0) -> np.ndarray:
    """Held exposure moves to the target only when |target - held| > DEADBAND."""
    t = np.asarray(targets, dtype=float)
    out = np.empty(len(t))
    held = float(held0)
    for i, e in enumerate(t):
        if abs(float(e) - held) > DEADBAND:
            held = float(e)
        out[i] = held
    return out


def _account(
    closes: np.ndarray, pos: np.ndarray, cost: float, start: int
) -> dict[str, Any]:
    """Replay a position path. ``pos[d]`` is the exposure held over day d (d > start); the
    trade into it executes at the close of d-1 and costs ``cost * |change| * equity``.

    Equity is tracked twice: as cash + BTC units, and as a running product. The accounting
    identity is the maximum absolute gap between the two."""
    n = len(closes)
    cash, units, prev = 1.0, 0.0, 0.0
    mult = 1.0
    equity = [1.0]
    turnover, trades, err = 0.0, 0, 0.0
    reasons: list[str] = []
    for d in range(start + 1, n):
        p0, p1, w = float(closes[d - 1]), float(closes[d]), float(pos[d])
        if not 0.0 <= w <= 1.0:
            reasons.append(f"exposure {w!r} outside [0, 1] on day index {d}")
        delta = abs(w - prev)
        if delta > TRADE_EPS:
            trades += 1
        turnover += delta
        e0 = cash + units * p0
        e1 = e0 - cost * delta * e0
        units = w * e1 / p0
        cash = e1 - units * p0
        eq = cash + units * p1
        mult = mult * (1.0 - cost * delta) * (1.0 + w * (p1 / p0 - 1.0))
        err = max(err, abs(eq - mult))
        equity.append(eq)
        prev = w
        if not eq > 0.0 or not math.isfinite(eq):
            reasons.append(f"equity {eq!r} not positive on day index {d}")
            break
    if err > IDENTITY_TOL:
        reasons.append(f"accounting identity gap {err!r} > {IDENTITY_TOL!r}")
    eq_arr = np.asarray(equity)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = eq_arr[1:] / eq_arr[:-1] - 1.0
        peak = np.maximum.accumulate(eq_arr)
        max_dd = float(np.max((peak - eq_arr) / peak)) if len(eq_arr) else 0.0
    sd = float(np.std(rets, ddof=0)) if len(rets) else 0.0
    sharpe = float(np.mean(rets)) / sd * math.sqrt(DAYS_PER_YEAR) if sd > 0.0 else 0.0
    return {
        "daily_returns": rets,
        "equity": eq_arr,
        "max_dd": max_dd,
        "sharpe": sharpe,
        "total_return": float(eq_arr[-1] - 1.0),
        "turnover": turnover,
        "trades": trades,
        "max_identity_err": err,
        "invalid_reasons": reasons,
        "valid": not reasons,
    }


def replay_cell(closes: Any, cost: float, delay: int, start: int = 0) -> dict[str, Any]:
    """Trend cell. The exposure decided at the close of day t (t >= start; held starts flat)
    is held over day t + delay. Evaluated days are start+1 .. len(closes)-1."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if delay < 1:
        raise ValueError("delay must be >= 1")
    if not 0 <= start < n - 1:
        raise ValueError("start must leave at least one evaluated day")
    if not np.all(np.isfinite(c)) or np.any(c <= 0.0):
        raise ValueError("closes must be positive and finite")
    decided = np.zeros(n)
    e = target_exposure(trend_score(c), realized_vol(c))
    decided[start:] = apply_deadband(e[start:])
    pos = np.zeros(n)
    for d in range(start + 1, n):
        k = d - delay
        pos[d] = decided[k] if k >= start else 0.0
    return _account(c, pos, cost, start)


def hold_cell(closes: Any, cost: float, start: int = 0) -> dict[str, Any]:
    """Buy at the close of ``start`` (cost at entry only) and hold fully invested."""
    c = np.asarray(closes, dtype=float)
    n = len(c)
    if not 0 <= start < n - 1:
        raise ValueError("start must leave at least one evaluated day")
    pos = np.zeros(n)
    pos[start + 1 :] = 1.0
    return _account(c, pos, cost, start)


def cash_cell(n_days: int) -> dict[str, Any]:
    return _account(np.ones(n_days + 1), np.zeros(n_days + 1), 0.0, 0)


def summarize_cell(cell: dict[str, Any]) -> dict[str, Any]:
    """JSON-safe cell record: stats plus digests of the daily series."""
    out = {k: v for k, v in cell.items() if k not in ("daily_returns", "equity")}
    out["days"] = int(len(cell["daily_returns"]))
    out["equity_sha256"] = hashlib.sha256(
        np.asarray(cell["equity"], dtype="<f8").tobytes()
    ).hexdigest()
    return out


# ----------------------------------------------------------------------------- decision


def supported(cells: dict[str, dict[str, Any]]) -> tuple[bool, list[str]]:
    """SUPPORTED(S): all cells valid and conditions 2-5 of the spec. Returns failures."""
    failed: list[str] = []
    if any(not c["valid"] for c in cells.values()):
        failed.append("all cells valid")
    if not cells["TREND__BASE"]["total_return"] > 0.0:
        failed.append("TREND__BASE total return > 0")
    if not cells["TREND__BASE"]["sharpe"] >= MIN_SHARPE:
        failed.append(f"TREND__BASE sharpe >= {MIN_SHARPE}")
    for name in ("TREND__HIGH_COST", "TREND__DELAY"):
        if not cells[name]["total_return"] > 0.0:
            failed.append(f"{name} total return > 0")
    for name, spec in CELLS.items():
        if spec[3] and not cells[name]["max_dd"] < MAX_DD:
            failed.append(f"{name} max DD < {MAX_DD}")
    return not failed, failed


def decide(
    stage: Stage | str,
    cells: dict[str, dict[str, Any]],
    backcast_supported: bool | None = None,
) -> dict[str, Any]:
    """Label for a stage. Priority: INVALID, then (interim) STOPPED_EARLY, then the stage rule.

    FORWARD_FINAL needs ``backcast_supported`` (the ledgered SUPPORTED(BACKCAST) fact)."""
    st = STAGES[stage] if isinstance(stage, str) else stage
    invalid = sorted(k for k, c in cells.items() if not c["valid"])
    ok, failed = supported(cells)
    out: dict[str, Any] = {
        "invalid_cells": invalid,
        "failed_conditions": failed,
        "supported": ok,
    }
    if invalid:
        out["label"] = "INVALID"
    elif st.interim:
        stopped = cells["TREND__BASE"]["max_dd"] >= EARLY_STOP_DD
        out["label"] = "STOPPED_EARLY" if stopped else "INTERIM_NOT_A_DECISION"
    elif st.final:
        if backcast_supported is None:
            raise ValueError("FORWARD_FINAL needs backcast_supported")
        out["label"] = (
            "TREND_SUPPORTED" if ok and backcast_supported else "TREND_NOT_SUPPORTED"
        )
    else:
        out["label"] = BACKCAST_LABEL_OK if ok else "TREND_NOT_SUPPORTED"
    return out


# ----------------------------------------------------------------------------- data parsing


def _day_index(d: date) -> int:
    return (d - EPOCH).days


def parse_bitstamp(payload: Any) -> list[tuple[int, float]]:
    """Bitstamp ``ohlc`` page -> [(unix seconds, close)]."""
    try:
        items = payload["data"]["ohlc"]
        if not isinstance(items, list):
            raise TypeError
        return [(int(it["timestamp"]), float(it["close"])) for it in items]
    except (KeyError, TypeError, ValueError):
        raise SystemExit("malformed Bitstamp OHLC payload") from None


def parse_binance(payload: Any) -> list[tuple[int, float]]:
    """Binance klines page -> [(open time in unix seconds, close)]."""
    if not isinstance(payload, list):
        raise SystemExit("malformed Binance klines payload")
    rows: list[tuple[int, float]] = []
    for it in payload:
        try:
            open_ms, close = int(it[0]), float(it[4])
        except (IndexError, KeyError, TypeError, ValueError):
            raise SystemExit("malformed Binance klines row") from None
        if open_ms % 1000:
            raise SystemExit(f"Binance open time {open_ms} is not whole seconds")
        rows.append((open_ms // 1000, close))
    return rows


def build_daily(rows: list[tuple[int, float]], first: date, last: date) -> np.ndarray:
    """Strict UTC daily closes for first..last inclusive. Refuses on a duplicate, a bar not
    opening at UTC midnight, a non-positive close, a missing day, or data ending early.
    Bars after ``last`` (e.g. today's partial bar) are dropped; none are filled."""
    seen: dict[int, float] = {}
    for ts, close in rows:
        if ts % DAY_S:
            raise SystemExit(f"bar at {ts} does not open at a UTC day boundary")
        day = ts // DAY_S
        if day in seen:
            raise SystemExit(f"duplicate bar for UTC day {EPOCH + timedelta(days=day)}")
        if not math.isfinite(close) or close <= 0.0:
            raise SystemExit(
                f"non-positive close on {EPOCH + timedelta(days=day)}: {close!r}"
            )
        seen[day] = close
    lo, hi = _day_index(first), _day_index(last)
    if not seen or max(seen) < hi:
        raise SystemExit(f"data ends before the required last bar {last.isoformat()}")
    missing = [d for d in range(lo, hi + 1) if d not in seen]
    if missing:
        raise SystemExit(
            f"missing UTC day {EPOCH + timedelta(days=missing[0])} ({len(missing)} missing)"
        )
    return np.asarray([seen[d] for d in range(lo, hi + 1)], dtype=float)


# ----------------------------------------------------------------------------- data fetching


def _http_get(url: str) -> bytes:  # pragma: no cover - real network
    if not url.startswith("https://"):
        raise SystemExit(f"refusing non-https URL {url}")
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=HTTP_TIMEOUT_S) as resp:  # noqa: S310
        data: bytes = resp.read()
    return data


def _page_url(source: str, start_s: int, end_s: int) -> str:
    if source == "bitstamp":
        return f"{BITSTAMP_URL}?step={DAY_S}&limit={PAGE_LIMIT}&start={start_s}&end={end_s}"
    return (
        f"{BINANCE_URL}?symbol=BTCUSDT&interval=1d&startTime={start_s * 1000}"
        f"&endTime={end_s * 1000 + 999}&limit={PAGE_LIMIT}"
    )


def fetch_daily(
    source: str,
    first: date,
    last: date,
    raw_dir: Path,
    fetch: Callable[[str], bytes] = _http_get,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Paged fetch with raw-response cache (write-once) and sha256 receipt."""
    parse = parse_bitstamp if source == "bitstamp" else parse_binance
    raw_dir.mkdir(parents=True, exist_ok=True)
    start_s = _day_index(first) * DAY_S
    end_s = _day_index(last) * DAY_S + DAY_S - 1
    rows: list[tuple[int, float]] = []
    pages: list[dict[str, Any]] = []
    cursor = start_s
    for page_no in range(MAX_PAGES):
        # Each request spans at most PAGE_LIMIT days, so the page is complete whichever end the
        # API anchors to (Bitstamp returns the last ``limit`` bars before ``end``; Amendment 1).
        page_end = min(end_s, cursor + PAGE_LIMIT * DAY_S - 1)
        url = _page_url(source, cursor, page_end)
        body = fetch(url)
        path = raw_dir / f"{source}_{page_no:03d}.json"
        with path.open("xb") as f:
            f.write(body)
        try:
            payload = json.loads(body)
        except ValueError:
            raise SystemExit(f"{source} page {page_no} is not JSON") from None
        page = parse(payload)
        pages.append(
            {
                "file": path.name,
                "url": url,
                "sha256": hashlib.sha256(body).hexdigest(),
                "rows": len(page),
            }
        )
        if not page:
            break
        rows.extend(page)
        newest = max(ts for ts, _ in page)
        if newest >= _day_index(last) * DAY_S:
            break
        nxt = newest + DAY_S
        if nxt <= cursor:
            raise SystemExit(f"{source} paging made no progress at {cursor}")
        cursor = nxt
    else:
        raise SystemExit(f"{source} paging exceeded {MAX_PAGES} pages")
    closes = build_daily(rows, first, last)
    receipt = {
        "source": source,
        "pages": pages,
        "first_day": first.isoformat(),
        "last_day": last.isoformat(),
        "days": len(closes),
        "closes_sha256": hashlib.sha256(closes.astype("<f8").tobytes()).hexdigest(),
    }
    return closes, receipt


# ----------------------------------------------------------------------------- authorization


def _gh_comment(comment_id: int) -> dict[str, Any]:  # pragma: no cover - real network
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/comments/{comment_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    parsed: dict[str, Any] = json.loads(out.stdout)
    return parsed


_COMMENT_URL = re.compile(
    r"https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-(\d+)"
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def comment_id_from_url(url: str) -> int:
    m = _COMMENT_URL.fullmatch(url)
    if m is None:
        raise SystemExit("authorization URL must be an Issue #810 comment")
    return int(m.group(1))


def _git(*args: str) -> str:  # pragma: no cover - real git
    out = subprocess.run(
        ["git", "-C", str(REPO_ROOT), *args],
        capture_output=True,
        text=True,
        check=True,
    )
    return out.stdout


def harness_commit() -> str:
    """Exact HEAD of the repository holding this file; refuses on tracked modifications."""
    if _git("status", "--porcelain", "--untracked-files=no").strip():
        raise SystemExit("harness repository has tracked modifications; commit first")
    commit = _git("rev-parse", "HEAD").strip()
    if not _COMMIT_RE.fullmatch(commit):
        raise SystemExit(f"unexpected HEAD from git: {commit!r}")
    return commit


def authorization_receipt(
    url: str,
    digest: str,
    commit: str,
    fetch: Callable[[int], dict[str, Any]] = _gh_comment,
) -> dict[str, Any]:
    if not _COMMIT_RE.fullmatch(commit):
        raise SystemExit("commit must be a full 40-hex SHA")
    out: dict[str, Any] = {}
    for role, cid in (
        ("g4_relay", comment_id_from_url(url)),
        ("data_scope", DATA_SCOPE_COMMENT_ID),
    ):
        first, second = fetch(cid), fetch(cid)
        if first["user"]["login"] != OWNER_LOGIN:
            raise SystemExit(f"{role} comment {cid} is not owner-authored")
        if first.get("issue_url") != ISSUE_URL:
            raise SystemExit(f"{role} comment {cid} is not on Issue #810")
        if (
            first["body"] != second["body"]
            or first["updated_at"] != second["updated_at"]
        ):
            raise SystemExit(f"{role} comment {cid} changed between fetches")
        if role == "g4_relay" and not all(
            token in first["body"] for token in (PROTOCOL_ID, digest, commit)
        ):
            raise SystemExit(
                "G4 relay does not quote this protocol id, digest and exact commit"
            )
        out[role] = {
            "comment_id": cid,
            "author": first["user"]["login"],
            "updated_at": first["updated_at"],
            "body_sha256": hashlib.sha256(first["body"].encode("utf-8")).hexdigest(),
        }
    return out


# ----------------------------------------------------------------------------- ledger & clocks


def claim_name(stage: Stage, now: datetime) -> str:
    """Ledger name: interims are once per UTC month."""
    return f"{stage.name}_{now:%Y-%m}" if stage.interim else stage.name


def check_clock(stage: Stage, now: datetime) -> None:
    if stage.not_before is not None and now < stage.not_before:
        raise SystemExit(
            f"{stage.name} may not run before {stage.not_before.isoformat()}"
        )
    if stage.not_after is not None and now >= stage.not_after:
        raise SystemExit(
            f"{stage.name} may not run at or after {stage.not_after.isoformat()}"
        )


def claim_stage(
    name: str,
    digest: str,
    commit: str,
    auth_comment_id: int,
    now: datetime,
    ledger_dir: Path = LEDGER_DIR,
) -> Path:
    """Local single-use enforcement: exclusive-create ``<ledger>/<name>.claim``."""
    ledger_dir.mkdir(parents=True, exist_ok=True)
    path = ledger_dir / f"{name}.claim"
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"stage {name} was already claimed: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": name,
                    "protocol_digest": digest,
                    "commit": commit,
                    "authorization_comment_id": auth_comment_id,
                    "claimed_at": now.isoformat(),
                },
                indent=1,
                sort_keys=True,
            )
        )
    return path


def write_ledger_result(
    name: str,
    digest: str,
    label: str,
    supported_flag: bool,
    result_sha256: str,
    ledger_dir: Path = LEDGER_DIR,
) -> Path:
    path = ledger_dir / f"{name}.result.json"
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"ledger result already exists: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": name,
                    "protocol_digest": digest,
                    "label": label,
                    "supported": supported_flag,
                    "result_sha256": result_sha256,
                },
                indent=1,
                sort_keys=True,
            )
        )
    return path


def check_backcast_ledger(digest: str, ledger_dir: Path = LEDGER_DIR) -> dict[str, Any]:
    """FORWARD stages need a BACKCAST ledger claim and result of this digest. A BACKCAST that
    is not ``TREND_BACKCAST_ONLY`` (SUPPORTED) stops this version, so it also refuses."""
    claim = ledger_dir / "BACKCAST.claim"
    result = ledger_dir / "BACKCAST.result.json"
    if not claim.is_file():
        raise SystemExit(f"no BACKCAST ledger claim: {claim}")
    if json.loads(claim.read_text(encoding="utf-8")).get("protocol_digest") != digest:
        raise SystemExit("BACKCAST ledger claim has another protocol digest")
    if not result.is_file():
        raise SystemExit(f"no BACKCAST ledger result: {result}")
    led = json.loads(result.read_text(encoding="utf-8"))
    if led.get("protocol_digest") != digest or led.get("stage") != "BACKCAST":
        raise SystemExit("BACKCAST ledger result has another digest or stage")
    if led.get("label") != BACKCAST_LABEL_OK or led.get("supported") is not True:
        raise SystemExit(
            f"BACKCAST label {led.get('label')!r} stops this version; "
            "FORWARD stages need SUPPORTED(BACKCAST)"
        )
    return {
        "label": led["label"],
        "result_sha256": led["result_sha256"],
        "ledger_result": str(result),
    }


# ----------------------------------------------------------------------------- stage run


def _write_once(path: Path, obj: dict[str, Any]) -> None:
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"refusing to overwrite {path}") from None
    with f:
        f.write(json.dumps(obj, indent=1, sort_keys=True))


def stage_window(stage: Stage, now: datetime) -> tuple[date, date]:
    """(first loaded day, last decision day). Interims end at the last complete UTC day."""
    last = stage.decision_end
    if stage.interim:
        last = min(last, now.date() - timedelta(days=1))
    return stage.load_from, last


def evaluate(
    closes: np.ndarray, load_from: date, decision_start: date
) -> dict[str, dict[str, Any]]:
    start = _day_index(decision_start) - _day_index(load_from)
    cells: dict[str, dict[str, Any]] = {}
    for name, (kind, cost, delay, _gate) in CELLS.items():
        if kind == "trend":
            cells[name] = replay_cell(closes, cost, delay, start)
        elif kind == "hold":
            cells[name] = hold_cell(closes, cost, start)
        else:
            cells[name] = cash_cell(len(closes) - 1 - start)
    return cells


def run_stage(
    stage_name: str,
    authorization_url: str,
    work: Path,
    output_dir: Path,
    now: datetime | None = None,
    fetch: Callable[[str], bytes] = _http_get,
    gh_fetch: Callable[[int], dict[str, Any]] = _gh_comment,
    ledger_dir: Path = LEDGER_DIR,
) -> dict[str, Any]:
    stage = STAGES[stage_name]
    now = now or datetime.now(UTC)
    check_clock(stage, now)
    if output_dir.exists():
        raise SystemExit(f"output directory exists: {output_dir}")
    if work.exists() and any(work.iterdir()):
        raise SystemExit(f"work directory is not empty: {work}")
    digest = protocol_digest()
    # One ledger namespace per protocol digest (Amendment 1): each digest gets exactly one claim
    # per stage, and a new digest exists only after a reviewed, relayed amendment.
    ledger_dir = ledger_dir / digest
    commit = harness_commit()
    auth = authorization_receipt(authorization_url, digest, commit, gh_fetch)
    backcast = (
        None if stage.name == "BACKCAST" else check_backcast_ledger(digest, ledger_dir)
    )
    name = claim_name(stage, now)
    claim = claim_stage(
        name, digest, commit, auth["g4_relay"]["comment_id"], now, ledger_dir
    )

    first, last = stage_window(stage, now)
    closes, data_receipt = fetch_daily(stage.source, first, last, work / "raw", fetch)
    cells = evaluate(closes, first, stage.decision_start)
    decision = decide(stage, cells, True if stage.final else None)
    result = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": digest,
        "commit": commit,
        "stage": stage.name,
        "claim_name": name,
        "claim": str(claim),
        "authorization": auth,
        "backcast_ledger": backcast,
        "data": data_receipt,
        "window": {
            "load_from": first.isoformat(),
            "decision_start": stage.decision_start.isoformat(),
            "decision_end": last.isoformat(),
        },
        "cells": {k: summarize_cell(v) for k, v in cells.items()},
        **decision,
        "run_at": now.isoformat(),
    }
    output_dir.mkdir(parents=True)
    _write_once(output_dir / "result.json", result)
    write_ledger_result(
        name,
        digest,
        result["label"],
        result["supported"],
        sha256_file(output_dir / "result.json"),
        ledger_dir,
    )
    return result


# ----------------------------------------------------------------------------- CLI


def cmd_digest(_args: argparse.Namespace) -> None:
    print(protocol_digest())


def cmd_run(args: argparse.Namespace) -> None:  # pragma: no cover - real network
    result = run_stage(
        args.stage, args.authorization_url, Path(args.work), Path(args.output_dir)
    )
    print(
        json.dumps(
            {
                k: result[k]
                for k in ("stage", "label", "invalid_cells", "failed_conditions")
            },
            indent=1,
        )
    )
    for name, st in result["cells"].items():
        print(
            f"{name:18s} ret {st['total_return'] * 100:+8.2f}% dd {st['max_dd'] * 100:5.2f}% "
            f"sr {st['sharpe']:+.2f} valid {st['valid']}"
        )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="btc_trend_ensemble")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("digest").set_defaults(fn=cmd_digest)
    p = sub.add_parser("run")
    p.add_argument("--stage", required=True, choices=sorted(STAGES))
    p.add_argument("--authorization-url", required=True)
    p.add_argument("--work", required=True)
    p.add_argument("--output-dir", required=True)
    p.set_defaults(fn=cmd_run)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
