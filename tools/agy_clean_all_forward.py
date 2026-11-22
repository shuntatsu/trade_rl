"""Forward test of agy's Round 17 non-RL arm ``R17_NONRL_CLEAN_ALL_175`` at gross <= 1.0.

Protocol: ``agy_clean_all_forward_v1`` (docs/specs/agy-clean-all-forward.md).

The signal code is agy's lab, frozen as a hash-pinned snapshot outside the repository.
This harness never edits the snapshot. For a stage it

1. verifies the snapshot manifests against the pins below,
2. copies the lab into a fresh work directory and moves the lab's embargo constants from
   2026-08-01 to the stage's data end with a fixed list of literal replacements,
3. rebuilds the lab caches from the local market store and public archives,
4. checks that every rebuilt pack10/pack17 row before 2026-08-01 equals agy's frozen
   pre-2026-08 packs (parity oracle; a mismatch refuses before any return is computed),
5. replays the frozen cells from 2026-08-01 to the stage end and writes the result.

Only ``--stage`` values listed in ``STAGES`` exist. The lab needs torch, so the
``prepare``/``build``/``run`` subcommands run under an interpreter with the lab's
dependencies. Importing this module needs only the standard library and numpy.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

PROTOCOL_ID = "agy_clean_all_forward_v1"

# Snapshot of agy session 7e52c53c scratch/rl_lab, taken 2026-11-22 (47 top-level .py files).
LAB_MANIFEST_SHA256 = "c4457c8693ba64980f7735c54cc71b4a1058735f27ab35d46be2c4d9a09b1683"
# agy's own pre-2026-08 round10/round17 pack caches: the parity oracle.
FROZEN_CACHE_MANIFEST_SHA256 = (
    "d07130ae24bd744b95f21a68d92f07950c5b68af7dd4df3cf81622d306749ef7"
)
FROZEN_PACK10 = "round10_mtf6_pack_pre20260801.npz"
FROZEN_PACK17 = "round17_clean_mtf_pack_pre20260801.npz"
# agy's original Binance metrics cache (Binance Vision republished some pre-embargo days).
FROZEN_METRICS = "binance_metrics_4h_pre20260801.npz"

OOS_START = datetime(2026, 8, 1, tzinfo=UTC)
LAB_EMBARGO = datetime(2026, 8, 1, tzinfo=UTC)


@dataclass(frozen=True)
class Stage:
    name: str
    data_end: datetime  # exclusive open time; last bar closes here
    final: bool
    not_before: datetime  # earliest wall-clock time the stage may run


STAGES: dict[str, Stage] = {
    "INTERIM_A": Stage(
        "INTERIM_A",
        datetime(2026, 9, 30, tzinfo=UTC),
        False,
        datetime(2026, 11, 22, tzinfo=UTC),
    ),
    "FINAL": Stage(
        "FINAL",
        datetime(2027, 5, 1, tzinfo=UTC),
        True,
        datetime(2027, 5, 2, tzinfo=UTC),
    ),
}

ARM_CANDIDATE = "R17_NONRL_CLEAN_ALL_175"
ARM_CONTROL = "ANC_BIDIR_75_25"
GROSS_CAP = 1.0
DECISION_STRIDE = 6  # 4h bars; the lab's 4H decision cadence
ECON_MODE = "ECON_BRAKE_CAP"

# name -> (arm, one-way cost, signal lag in bars, mask lag in bars, gates)
CELLS: dict[str, tuple[str, float, int, int, bool]] = {
    "CANDIDATE__BASE": (ARM_CANDIDATE, 0.0006, 1, 1, True),
    "CANDIDATE__HIGH_COST": (ARM_CANDIDATE, 0.0012, 1, 1, True),
    "CANDIDATE__STRESS": (ARM_CANDIDATE, 0.0020, 1, 1, False),
    "CANDIDATE__NO_DELAY": (ARM_CANDIDATE, 0.0006, 0, 1, False),
    "CANDIDATE__LAB_MASK": (ARM_CANDIDATE, 0.0006, 1, 0, False),
    "CONTROL_ANC__BASE": (ARM_CONTROL, 0.0006, 1, 1, False),
}

# Local single-use ledger: one exclusive-create claim file per stage (see ``claim_stage``).
LEDGER_DIR = Path(r"C:\trade_rl_runs\agy_clean_all_forward_ledger")
REPO_ROOT = Path(__file__).resolve().parent.parent
SPEC_RELPATH = "docs/specs/agy-clean-all-forward.md"

ORACLE_START = datetime(2023, 1, 1, tzinfo=UTC)
ORACLE_RTOL = 1e-9
REQUIRED_PACK10 = (
    "spot_valid",
    "perp_valid",
    "perp_mask",
    "carry_mask",
    "spot_next_ret",
    "perp_next_ret",
    "funding_realized",
    "vol_cushion_tail",
    "c_weekly_iv",
    "w_aw_bidir_75_25",
)
REQUIRED_PACK17 = ("w_clean_anc", "w_clean_all", "perp_mask")
REQUIRED_ECON = ("eff_deadband", "eff_freeze_thr", "a_mult", "c_mult")

BARS_PER_YEAR = 2190.0
FINAL_MIN_SHARPE = 0.5
MAX_DD_LIMIT = 0.20
EARLY_STOP_RETURN = -0.08
PARITY_ATOL = 1e-12
CONSERVATION_TOL = 1e-9
# Initial-margin rate of the engine (1 / max_leverage), as in agy_anchor_realistic_v1 Amendment 1.
CASH_IM_RATE = 0.4

OWNER_LOGIN = "shuntatsu"
REPO = "shuntatsu/trade_rl"
DATA_SCOPE_COMMENT_ID = (
    7040247247  # owner scope relay for pre-registered OOS/forward reads
)


# ----------------------------------------------------------------------------- digests


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def manifest(directory: Path, pattern: str) -> dict[str, str]:
    return {
        p.name: sha256_file(p) for p in sorted(directory.glob(pattern)) if p.is_file()
    }


def manifest_digest(entries: dict[str, str]) -> str:
    blob = json.dumps(
        dict(sorted(entries.items())), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _constants() -> dict[str, Any]:
    return {
        "protocol_id": PROTOCOL_ID,
        "lab_manifest": LAB_MANIFEST_SHA256,
        "frozen_cache_manifest": FROZEN_CACHE_MANIFEST_SHA256,
        "oos_start": OOS_START.isoformat(),
        "stages": {
            k: [v.data_end.isoformat(), v.final, v.not_before.isoformat()]
            for k, v in STAGES.items()
        },
        "cells": {k: list(v) for k, v in CELLS.items()},
        "gross_cap": GROSS_CAP,
        "stride": DECISION_STRIDE,
        "econ_mode": ECON_MODE,
        "final_min_sharpe": FINAL_MIN_SHARPE,
        "max_dd_limit": MAX_DD_LIMIT,
        "early_stop_return": EARLY_STOP_RETURN,
        "parity_atol": PARITY_ATOL,
        "data_scope_comment": DATA_SCOPE_COMMENT_ID,
        "oracle_start": ORACLE_START.isoformat(),
        "oracle_rtol": ORACLE_RTOL,
        "cash_im_rate": CASH_IM_RATE,
        "required_pack10": list(REQUIRED_PACK10),
        "required_pack17": list(REQUIRED_PACK17),
        "required_econ": list(REQUIRED_ECON),
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


# ----------------------------------------------------------------------------- snapshot & patch


def verify_snapshot(snapshot: Path) -> dict[str, str]:
    lab = manifest(snapshot / "lab", "*.py")
    cache = manifest(snapshot / "frozen_cache", "*.npz")
    got_lab, got_cache = manifest_digest(lab), manifest_digest(cache)
    if got_lab != LAB_MANIFEST_SHA256:
        raise SystemExit(f"lab snapshot manifest mismatch: {got_lab}")
    if got_cache != FROZEN_CACHE_MANIFEST_SHA256:
        raise SystemExit(f"frozen cache manifest mismatch: {got_cache}")
    return {"lab": got_lab, "frozen_cache": got_cache}


def embargo_patches(data_end: datetime) -> list[tuple[str, str]]:
    """Literal replacements that move the lab's 2026-08-01 embargo to ``data_end``."""
    if data_end.hour or data_end.minute or data_end.second or data_end.tzinfo is None:
        raise ValueError("data_end must be a UTC midnight")
    last_day = datetime.fromtimestamp(data_end.timestamp() - 86400, tz=UTC)
    tag = data_end.strftime("%Y%m%d")
    return [
        ("2026-08-01 00:00:00+00", f"{data_end:%Y-%m-%d} 00:00:00+00"),
        (
            "datetime(2026, 8, 1, 0, 0, tzinfo=UTC)",
            f"datetime({data_end.year}, {data_end.month}, {data_end.day}, 0, 0, tzinfo=UTC)",
        ),
        (
            "date(2026, 7, 31)",
            f"date({last_day.year}, {last_day.month}, {last_day.day})",
        ),
        ("1785542400", str(int(data_end.timestamp()))),
        ("pre20260801", f"fwd{tag}"),
    ]


RESIDUE_FORBIDDEN = (
    "2026-08-01 00:00:00+00",
    "datetime(2026, 8, 1, 0, 0",
    "date(2026, 7, 31)",
    "1785542400",
    "pre20260801",
)


def apply_patches(
    src_dir: Path, dst_dir: Path, patches: list[tuple[str, str]]
) -> dict[str, dict[str, int]]:
    """Copy ``*.py`` from ``src_dir`` to ``dst_dir`` applying ``patches``; refuse on residue."""
    if dst_dir.exists() and any(dst_dir.iterdir()):
        raise SystemExit(f"work lab directory is not empty: {dst_dir}")
    dst_dir.mkdir(parents=True, exist_ok=True)
    counts: dict[str, dict[str, int]] = {}
    totals = dict.fromkeys((old for old, _ in patches), 0)
    for src in sorted(src_dir.glob("*.py")):
        text = src.read_text(encoding="utf-8")
        per: dict[str, int] = {}
        for old, new in patches:
            n = text.count(old)
            if n:
                text = text.replace(old, new)
                per[old] = n
                totals[old] += n
        for bad in RESIDUE_FORBIDDEN:
            if bad in text:
                raise SystemExit(f"embargo literal {bad!r} survives in {src.name}")
        (dst_dir / src.name).write_text(text, encoding="utf-8", newline="\n")
        if per:
            counts[src.name] = per
    missing = [old for old, n in totals.items() if n == 0]
    if missing:
        raise SystemExit(f"patch literal never matched: {missing}")
    return counts


# ----------------------------------------------------------------------------- authorization


def _gh_comment(comment_id: int) -> dict[str, Any]:
    out = subprocess.run(
        ["gh", "api", f"repos/{REPO}/issues/comments/{comment_id}"],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


_COMMENT_URL = re.compile(
    r"https://github\.com/shuntatsu/trade_rl/issues/810#issuecomment-(\d+)"
)
_COMMIT_RE = re.compile(r"[0-9a-f]{40}")


def comment_id_from_url(url: str) -> int:
    m = _COMMENT_URL.fullmatch(url)
    if m is None:
        raise SystemExit("authorization URL must be an Issue #810 comment")
    return int(m.group(1))


def _git(*args: str) -> str:
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
    url: str, digest: str, commit: str, fetch: Any = _gh_comment
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
        if not str(first.get("issue_url", "")).endswith("/issues/810"):
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


def claim_stage(
    stage: str, digest: str, commit: str, auth_comment_id: int, now: datetime
) -> Path:
    """Local single-use enforcement: exclusive-create ``<LEDGER_DIR>/<stage>.claim``."""
    LEDGER_DIR.mkdir(parents=True, exist_ok=True)
    path = LEDGER_DIR / f"{stage}.claim"
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"stage {stage} was already claimed: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": stage,
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
    stage: str, digest: str, label: str, result_sha256: str
) -> Path:
    """Bind a stage result to the ledger by exclusive create of ``<stage>.result.json``."""
    path = LEDGER_DIR / f"{stage}.result.json"
    try:
        f = path.open("x", encoding="utf-8")
    except FileExistsError:
        raise SystemExit(f"ledger result already exists: {path}") from None
    with f:
        f.write(
            json.dumps(
                {
                    "stage": stage,
                    "protocol_digest": digest,
                    "label": label,
                    "result_sha256": result_sha256,
                },
                indent=1,
                sort_keys=True,
            )
        )
    return path


def check_interim_result(path: Path, digest: str) -> dict[str, Any]:
    """FINAL needs an INTERIM_A result of this protocol whose label is INTERIM_NOT_A_DECISION
    and a local INTERIM_A ledger claim recording the same digest."""
    claim_path = LEDGER_DIR / "INTERIM_A.claim"
    if not claim_path.is_file():
        raise SystemExit(f"no INTERIM_A ledger claim: {claim_path}")
    if (
        json.loads(claim_path.read_text(encoding="utf-8")).get("protocol_digest")
        != digest
    ):
        raise SystemExit("INTERIM_A ledger claim has another protocol digest")
    ledger_result = LEDGER_DIR / "INTERIM_A.result.json"
    if not ledger_result.is_file():
        raise SystemExit(f"no INTERIM_A ledger result: {ledger_result}")
    led = json.loads(ledger_result.read_text(encoding="utf-8"))
    if led.get("protocol_digest") != digest or led.get("stage") != "INTERIM_A":
        raise SystemExit("INTERIM_A ledger result has another digest or stage")
    if led.get("label") != "INTERIM_NOT_A_DECISION":
        raise SystemExit(
            f"interim label {led.get('label')!r} does not allow FINAL "
            "(STOPPED_EARLY ends the line; INVALID needs an amendment)"
        )
    if led.get("result_sha256") != sha256_file(path):
        raise SystemExit("--interim-result does not match the INTERIM_A ledger result")
    obj = json.loads(path.read_text(encoding="utf-8"))
    if obj.get("protocol_digest") != digest:
        raise SystemExit("interim result has another protocol digest")
    if obj.get("stage") != "INTERIM_A":
        raise SystemExit("interim result is not stage INTERIM_A")
    if obj.get("label") != "INTERIM_NOT_A_DECISION":
        raise SystemExit(
            f"interim label {obj.get('label')!r} does not allow FINAL "
            "(STOPPED_EARLY ends the line; INVALID needs an amendment)"
        )
    return {"path": str(path), "sha256": sha256_file(path), "label": obj["label"]}


def verify_work_lab(work: Path, prep: dict[str, Any]) -> None:
    if manifest(work / "lab", "*.py") != prep["lab_files"]:
        raise SystemExit("work/lab differs from the prepare receipt")


# ----------------------------------------------------------------------------- decision rule


def cell_stats(log_returns: list[float]) -> dict[str, float]:
    eq, peak, max_dd = 1.0, 1.0, 0.0
    for lr in log_returns:
        eq *= math.exp(lr)
        peak = max(peak, eq)
        max_dd = max(max_dd, 1.0 - eq / peak)
    n = len(log_returns)
    mean = sum(log_returns) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in log_returns) / n if n else 0.0
    sharpe = mean / math.sqrt(var) * math.sqrt(BARS_PER_YEAR) if var > 0 else 0.0
    return {
        "total_return": eq - 1.0,
        "max_drawdown": max_dd,
        "annual_sharpe": sharpe,
        "bars": float(n),
    }


def decide(stage: Stage, cells: dict[str, dict[str, Any]]) -> dict[str, Any]:
    invalid = sorted(k for k, v in cells.items() if not v.get("valid", False))
    if set(cells) != set(CELLS):
        raise ValueError("cell set differs from the frozen CELLS")
    if invalid:
        return {"label": "INVALID", "invalid_cells": invalid, "failed_conditions": None}
    base, high = cells["CANDIDATE__BASE"], cells["CANDIDATE__HIGH_COST"]
    gate_dd = max(cells[k]["max_drawdown"] for k, v in CELLS.items() if v[4])
    if not stage.final:
        stop = []
        if base["total_return"] <= EARLY_STOP_RETURN:
            stop.append("base_return_at_or_below_early_stop")
        if gate_dd >= MAX_DD_LIMIT:
            stop.append("gating_drawdown_at_or_above_limit")
        label = "STOPPED_EARLY" if stop else "INTERIM_NOT_A_DECISION"
        return {"label": label, "invalid_cells": [], "failed_conditions": stop}
    failed = []
    if not base["total_return"] > 0.0:
        failed.append("C1_base_return_positive")
    if not base["annual_sharpe"] >= FINAL_MIN_SHARPE:
        failed.append("C2_base_sharpe")
    if not high["total_return"] > 0.0:
        failed.append("C3_high_cost_return_positive")
    if not gate_dd < MAX_DD_LIMIT:
        failed.append("C4_gating_drawdown")
    label = (
        "CLEAN_ALL_FORWARD_SUPPORTED"
        if not failed
        else "CLEAN_ALL_FORWARD_NOT_SUPPORTED"
    )
    return {"label": label, "invalid_cells": [], "failed_conditions": failed}


def cap_gross(ws: Any, wp: Any, cap: float = GROSS_CAP) -> tuple[Any, Any, bool]:
    gross = float(abs(ws).sum() + abs(wp).sum())
    if gross > cap:
        return ws * (cap / gross), wp * (cap / gross), True
    return ws, wp, False


def with_lagged_masks(p10: Any, lag: int = 1) -> Any:
    """Copy of ``p10`` whose ``perp_mask``/``carry_mask`` are shifted down ``lag`` rows.

    The masks are bar-completeness flags known only at the bar close, so the decision at bar
    k may use only ``mask[k - lag]``. The first ``lag`` rows become False; nothing else changes.
    """
    if lag <= 0:
        return p10

    def shift(arr: Any) -> Any:
        a = np.asarray(arr, dtype=bool)
        out = np.zeros_like(a)
        out[lag:] = a[:-lag]
        return out

    return replace(
        p10, perp_mask=shift(p10.perp_mask), carry_mask=shift(p10.carry_mask)
    )


def cash_need(ws: Any, wp: Any) -> float:
    """Cash-account need as a fraction of equity: spot long notional + IM rate x perp gross."""
    return float(np.maximum(ws, 0.0).sum() + CASH_IM_RATE * np.abs(wp).sum())


def replay_cell(
    *,
    he: Any,
    r3c: Any,
    r15g: Any,
    r17: Any,
    p10: Any,
    p17: Any,
    econ: Any,
    arm: str,
    cost: float,
    start: int,
    stop: int,
    cap: float | None,
    econ_mode: str = ECON_MODE,
    p10_select: Any = None,
) -> dict[str, Any]:
    """Replay one cell over bars ``[start, stop)``; pure given the lab objects.

    Decisions are taken every ``DECISION_STRIDE`` bars from ``start``. A decision target is
    scaled to ``cap`` (``None`` = uncapped, used only by the loop oracle). Held bars pass the
    current weights through as the target with deadband 0. Executed gross is reported only:
    held weights drift and halted legs keep their weight, so it may exceed the cap legitimately.
    """
    # ``p10_select`` (masks lagged) is used only to choose symbols for the decision target;
    # execution keeps the current bar's validity, since a halted bar cannot trade.
    sel = p10 if p10_select is None else p10_select
    n = len(p10.symbols)
    ws, wp, eq = np.zeros(n), np.zeros(n), 1.0
    lrs: list[float] = []
    valid, reasons = True, []
    max_err = turn = cost_sum = max_exec_gross = max_cash = max_net = 0.0
    capped = decisions = incomplete_legs = incomplete_cells = 0
    for i in range(stop - start):
        k = start + i
        if i % DECISION_STRIDE == 0:
            db, _f, a_m, c_m = r15g._econ_scalars(econ, k, econ_mode)
            alpha, scale = r17.nonrl_alpha_and_scale_np(p10, p17, arm, k)
            ws_t, wp_t = r17.synthesize_r17_np(sel, k, alpha, scale, a_m, c_m)
            if cap is not None:
                ws_t, wp_t, was = cap_gross(ws_t, wp_t, cap)
                capped += int(was)
                gross_t = float(np.abs(ws_t).sum() + np.abs(wp_t).sum())
                if gross_t > cap + 1e-9:
                    valid = False
                    reasons.append(f"target_gross_above_cap@{k}")
            decisions += 1
            # Reported only: decision-bar legs whose bar-completeness flag is False.
            bad_s = ~np.asarray(p10.spot_valid[k], dtype=bool)
            bad_p = ~np.asarray(p10.perp_valid[k], dtype=bool)
            incomplete_cells += int(bad_s.sum() + bad_p.sum())
            incomplete_legs += int(
                np.sum(bad_s & (np.asarray(ws_t) != ws))
                + np.sum(bad_p & (np.asarray(wp_t) != wp))
            )
            if not (np.all(np.isfinite(ws_t)) and np.all(np.isfinite(wp_t))):
                valid = False
                reasons.append(f"non_finite_target@{k}")
        else:
            ws_t, wp_t, db = ws, wp, 0.0
        ws_e, wp_e, eq_n, ws_n, wp_n, t, c, px, fr = he.execute_hybrid_step(
            w_spot_cur=ws,
            w_perp_cur=wp,
            w_spot_target=ws_t,
            w_perp_target=wp_t,
            equity_before=eq,
            spot_valid=p10.spot_valid[k],
            perp_valid=p10.perp_valid[k],
            spot_next_ret=p10.spot_next_ret[k],
            perp_next_ret=p10.perp_next_ret[k],
            funding_realized=p10.funding_realized[k],
            one_way_cost=cost,
            impact_quad=r3c.IMPACT_QUAD,
            deadband=db,
        )
        err = abs((eq_n - eq) - (px + fr - c))
        max_err = max(max_err, err)
        if err > CONSERVATION_TOL or not math.isfinite(eq_n) or eq_n <= 0.0:
            valid = False
            reasons.append(f"accounting@{k}")
        max_exec_gross = max(
            max_exec_gross, float(np.abs(ws_e).sum() + np.abs(wp_e).sum())
        )
        for sv, pv in ((ws_e, wp_e), (ws_n, wp_n)):
            need = cash_need(sv, pv)
            max_cash = max(max_cash, need)
            max_net = max(max_net, abs(float(np.sum(sv) + np.sum(pv))))
            if need > 1.0 + 1e-9 and not any(
                r.startswith("cash_account_need_above_equity") for r in reasons
            ):
                valid = False
                reasons.append(f"cash_account_need_above_equity@{k}")
        lrs.append(math.log(max(eq_n, 1e-12) / eq))
        turn += float(t)
        cost_sum += float(c)
        eq, ws, wp = float(eq_n), ws_n, wp_n
    st: dict[str, Any] = dict(cell_stats(lrs))
    st.update(
        {
            "valid": valid,
            "invalid_reasons": reasons[:20],
            "max_conservation_error": max_err,
            "turnover": turn,
            "trading_cost": cost_sum,
            "cap_bound_decisions": capped,
            "decisions": decisions,
            "max_executed_gross": max_exec_gross,
            "max_cash_need": max_cash,
            "incomplete_decision_legs": incomplete_legs,
            "incomplete_decision_bar_cells": incomplete_cells,
            "max_abs_net": max_net,
            "final_equity": eq,
            "arm": arm,
            "one_way_cost": cost,
        }
    )
    return st


def array_diff(a: Any, b: Any) -> float:
    """Max abs difference (floats, NaN-aware) or count of unequal entries (other dtypes)."""
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape:
        raise SystemExit(f"parity shape mismatch: {a.shape} vs {b.shape}")
    if b.dtype.kind in "fc":
        both_nan = np.isnan(a) & np.isnan(b)
        return float(np.max(np.where(both_nan, 0.0, np.abs(a - b)), initial=0.0))
    return float(np.sum(a != b))


def pack_parity(
    tag: str, new: Any, old: Any, required: tuple[str, ...], n_pre: int, n_cmp: int
) -> dict[str, float]:
    """Compare ``new`` with the frozen ``old`` over rows ``[:n_cmp]``; refuse on a missed field."""
    parity: dict[str, float] = {}
    for field in old.__dataclass_fields__:
        b = getattr(old, field)
        if not isinstance(b, np.ndarray) or b.ndim == 0 or b.shape[0] != n_pre:
            continue
        parity[f"{tag}.{field}"] = array_diff(
            np.asarray(getattr(new, field))[:n_cmp], b[:n_cmp]
        )
    missed = [f for f in required if f"{tag}.{f}" not in parity]
    if missed:
        raise SystemExit(f"parity did not compare required fields of {tag}: {missed}")
    return parity


def econ_parity(new: Any, old: Any, n_cmp: int) -> dict[str, float]:
    if set(new) != set(old) or not set(REQUIRED_ECON) <= set(new):
        raise SystemExit(
            f"econ key mismatch: {sorted(new)} vs {sorted(old)} "
            f"(required {sorted(REQUIRED_ECON)})"
        )
    return {
        f"econ.{key}": array_diff(
            np.asarray(new[key])[:n_cmp], np.asarray(old[key])[:n_cmp]
        )
        for key in sorted(new)
    }


def _ns(dt: datetime) -> int:
    return int(dt.timestamp() * 1e9)


def splice_frozen_metrics(
    fresh_path: Path, frozen_path: Path, base_path: Path, embargo_ns: int
) -> dict[str, Any]:
    """Overwrite the pre-embargo rows of the fresh metrics cache with agy's frozen rows.

    Binance Vision republished some archived daily files after agy's fetch, so the fresh
    pre-embargo rows differ slightly from the frozen ones. Rows at or after the embargo stay
    as fetched. Row alignment is checked against the base 4h cache's ``open_ns``.
    """
    sha_before = sha256_file(fresh_path)
    with np.load(base_path, allow_pickle=False) as b:
        open_ns = np.asarray(b["open_ns"])
    n_pre = int(np.searchsorted(open_ns, embargo_ns, side="left"))
    with np.load(fresh_path, allow_pickle=False) as f:
        fresh = {k: np.asarray(f[k]) for k in f.files}
    with np.load(frozen_path, allow_pickle=False) as z:
        frozen = {k: np.asarray(z[k]) for k in z.files}
    if set(fresh) != set(frozen) or "symbols" not in fresh:
        raise SystemExit("metrics cache field sets differ from the frozen cache")
    if not np.array_equal(fresh["symbols"], frozen["symbols"]):
        raise SystemExit("metrics cache symbols differ from the frozen cache")
    if len(fresh["symbols"]) == 0 or any(
        len(frozen[k]) != n_pre for k in frozen if k != "symbols"
    ):
        raise SystemExit("frozen metrics row count differs from pre-embargo bars")
    for k in fresh:
        if k == "symbols":
            continue
        if len(fresh[k]) < n_pre or fresh[k].shape[1:] != frozen[k].shape[1:]:
            raise SystemExit(f"fresh metrics array {k} does not cover the frozen rows")
        if fresh[k].dtype != frozen[k].dtype:
            raise SystemExit(f"metrics array {k} dtype differs from the frozen cache")
        fresh[k] = np.concatenate([frozen[k], fresh[k][n_pre:]])
    np.savez_compressed(fresh_path, **fresh)
    return {
        "spliced_rows": n_pre,
        "sha256_before": sha_before,
        "sha256_after": sha256_file(fresh_path),
    }


def _only(directory: Path, pattern: str) -> Path:
    found = sorted(directory.glob(pattern))
    if len(found) != 1:
        raise SystemExit(f"expected exactly one {pattern} in {directory}: {found}")
    return found[0]


# ----------------------------------------------------------------------------- lab-backed steps


def _lab_env(work: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["PYTHONPATH"] = str(work / "lab")
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


BUILD_SCRIPTS = (
    "extract_real_data.py",
    "fetch_external_data.py",
    "extract_1h_micro.py",
    "extract_mtf_data.py",
)


def cmd_prepare(args: argparse.Namespace) -> None:
    stage = STAGES[args.stage]
    snapshot, work = Path(args.snapshot), Path(args.work)
    pins = verify_snapshot(snapshot)
    counts = apply_patches(
        snapshot / "lab", work / "lab", embargo_patches(stage.data_end)
    )
    (work / "lab" / "cache").mkdir(exist_ok=True)
    receipt = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": protocol_digest(),
        "stage": stage.name,
        "snapshot": pins,
        "patch_counts": counts,
        "lab_files": manifest(work / "lab", "*.py"),
        "data_end": stage.data_end.isoformat(),
    }
    _write_once(work / "prepare-receipt.json", receipt)
    print(
        json.dumps(
            {
                "protocol_digest": receipt["protocol_digest"],
                "patched_files": len(counts),
            }
        )
    )


def _preflight(args: argparse.Namespace) -> dict[str, Any]:
    """Shared gate of ``build`` and ``run``: runs before any data is fetched or read."""
    stage = STAGES[args.stage]
    now = datetime.now(tz=UTC)
    if now < stage.not_before:
        raise SystemExit(
            f"stage {stage.name} may not run before {stage.not_before.isoformat()}"
        )
    work, snapshot = Path(args.work), Path(args.snapshot)
    pins = verify_snapshot(snapshot)
    digest = protocol_digest()
    prep = json.loads((work / "prepare-receipt.json").read_text(encoding="utf-8"))
    if prep["protocol_digest"] != digest or prep["stage"] != stage.name:
        raise SystemExit("work directory was prepared under another digest or stage")
    verify_work_lab(work, prep)
    commit = harness_commit()
    auth = authorization_receipt(args.authorization_url, digest, commit)
    return {
        "stage": stage,
        "now": now,
        "pins": pins,
        "digest": digest,
        "commit": commit,
        "auth": auth,
    }


def cmd_build(
    args: argparse.Namespace,
) -> None:  # pragma: no cover - needs the lab and network
    work, snapshot = Path(args.work), Path(args.snapshot)
    if (work / "build-receipt.json").exists():
        raise SystemExit("build is once-only: build-receipt.json already exists")
    pre = _preflight(args)
    for script in BUILD_SCRIPTS:
        print(f"[build] {script}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(work / "lab" / script)],
            cwd=work / "lab",
            env=_lab_env(work),
            check=True,
        )
    cache = work / "lab" / "cache"
    splice = splice_frozen_metrics(
        _only(cache, "binance_metrics_4h_fwd*.npz"),
        snapshot / "frozen_cache" / FROZEN_METRICS,
        _only(cache, "real_dev_4h_fwd*.npz"),
        _ns(LAB_EMBARGO),
    )
    caches = manifest(cache, "*.npz")
    _write_once(
        work / "build-receipt.json",
        {
            "raw_caches": caches,
            "metrics_splice": splice,
            "commit": pre["commit"],
            "authorization": pre["auth"],
        },
    )
    print(json.dumps(caches, indent=1))


def _write_once(path: Path, obj: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def _importable_pretrade() -> tuple[str | None, str | None]:
    """Path and sha256 of ``trade_rl.risk.pretrade`` as importable under this interpreter."""
    try:
        spec = importlib.util.find_spec("trade_rl.risk.pretrade")
    except (ImportError, ValueError):
        return None, None
    if spec is None or not spec.origin or not Path(spec.origin).is_file():
        return None, None
    return spec.origin, sha256_file(Path(spec.origin))


def cmd_run(
    args: argparse.Namespace,
) -> None:  # pragma: no cover - needs the lab and its data
    stage = STAGES[args.stage]
    out_dir = Path(args.output_dir)
    if out_dir.exists():
        raise SystemExit(f"output directory exists: {out_dir}")
    if stage.final == (args.interim_result is None):
        raise SystemExit(
            "FINAL requires --interim-result; INTERIM stages must not pass it"
        )
    pre = _preflight(args)
    digest, commit, auth, now = pre["digest"], pre["commit"], pre["auth"], pre["now"]
    interim = (
        check_interim_result(Path(args.interim_result), digest) if stage.final else None
    )
    work, snapshot = Path(args.work), Path(args.snapshot)
    build_receipt_path = work / "build-receipt.json"
    if not build_receipt_path.is_file():
        raise SystemExit("build-receipt.json is missing; run build first")
    build_receipt = json.loads(build_receipt_path.read_text(encoding="utf-8"))
    cache_files = manifest(work / "lab" / "cache", "*.npz")
    if cache_files != build_receipt["raw_caches"]:
        raise SystemExit("work/lab/cache differs from the build receipt")
    claim = claim_stage(stage.name, digest, commit, auth["g4_relay"]["comment_id"], now)

    sys.dont_write_bytecode = True
    sys.path.insert(0, str(work / "lab"))
    import run_round17_breakthrough_gpu as r17  # type: ignore[import-not-found]

    pretrade = _importable_pretrade()

    he, r3c, r10f, r17f, r15g, r12g = (
        r17.he,
        r17.r3c,
        r17.r10f,
        r17.r17f,
        r17.r15g,
        r17.r12g,
    )
    data = he.load_hybrid_data()
    m1h, m15m = r17.r4f.load_micro_data(), r17.r5f.load_15m_micro_data()
    end_ns = _ns(stage.data_end)
    if int(data.close_ns[-1]) != end_ns:
        raise SystemExit(f"dataset does not end at {stage.data_end.isoformat()}")
    p4 = r17.r4f.build_round4_pack(data, m1h)
    p5 = r17.r5f.build_round5_mtf_pack(data, m1h, m15m, p4)
    p6 = r17.r6f.build_round6_aw_pack(data, m1h, m15m, p4, p5)
    p7 = r17.r7f.build_round7_frontier_pack(data, m1h, m15m, p4, p5, p6)
    p8 = r17.r8f.build_round8_nonlinear_pack(data, m1h, m15m, p4, p5, p6, p7)
    pack10 = r10f.build_round10_mtf6_pack(data, m1h, m15m, p4, p5, p6, p7, p8)
    pack17 = r17f.build_round17_clean_pack(data, m1h, m15m, p4, p5, p6, p7, p8, pack10)

    # ---- parity oracle: every pre-2026-08 row equals agy's frozen packs and econ controls. The
    # last frozen row is excluded: its next-bar return and funding belong to the first OOS bar.
    n_pre = int(np.searchsorted(data.close_ns, _ns(LAB_EMBARGO), side="right"))
    n_cmp = n_pre - 1
    frozen10 = r10f.load_r10_pack(snapshot / "frozen_cache" / FROZEN_PACK10)
    frozen17 = r17f.load_r17_pack(snapshot / "frozen_cache" / FROZEN_PACK17)

    def _econ(pack: Any) -> Any:
        return r15g.compute_r15_econ_controls(r12g.compute_r12_opp_feats(pack))

    econ0 = _econ(pack10)
    parity = {
        **pack_parity("pack10", pack10, frozen10, REQUIRED_PACK10, n_pre, n_cmp),
        **pack_parity("pack17", pack17, frozen17, REQUIRED_PACK17, n_pre, n_cmp),
        **econ_parity(econ0, _econ(frozen10), n_cmp),
    }
    worst = max(parity.values(), default=math.inf)
    if worst > PARITY_ATOL:
        bad = {k: v for k, v in parity.items() if v > PARITY_ATOL}
        _write_once(work / "parity-refusal.json", {"parity": parity, "n_pre": n_pre})
        raise SystemExit(
            f"parity oracle failed before any return was computed: {sorted(bad)[:10]}"
        )

    packs = {
        0: (pack10, pack17, econ0),
        1: (
            r10f.lag_r10_pack(pack10, 1),
            r17f.lag_r17_pack(pack17, 1),
            r15g.lag_r15_econ(econ0, 1),
        ),
    }
    lab_kw = {"he": he, "r3c": r3c, "r15g": r15g, "r17": r17}

    # ---- loop oracle: the uncapped replay on the known-used window equals the lab's evaluator.
    o_start = int(np.searchsorted(data.open_ns, _ns(ORACLE_START), side="left"))
    o_stop = n_cmp
    zeros = np.zeros(len(data.open_ns))
    loop_oracle: dict[str, dict[str, float]] = {}
    for arm in (ARM_CANDIDATE, ARM_CONTROL):
        p10, p17, econ = packs[1]
        mine = replay_cell(
            **lab_kw,
            p10=p10,
            p17=p17,
            econ=econ,
            arm=arm,
            cost=0.0006,
            start=o_start,
            stop=o_stop,
            cap=None,
        )["final_equity"]
        lab = float(
            r17.eval_r17_4h(
                arm=arm,
                pack10=p10,
                pack14=None,
                pack16=None,
                pack17=p17,
                econ=econ,
                econ_mode=ECON_MODE,
                comps=[],
                mods=None,
                om44=None,
                os44=None,
                cm10=None,
                cs10=None,
                dm14=None,
                ds14=None,
                dm16=None,
                ds16=None,
                start_idx=o_start,
                stop_idx=o_stop,
                one_way_cost=0.0006,
                years_all=zeros,
                mkt_all=zeros,
            )["final_equity"]
        )
        rel = abs(mine - lab) / max(abs(lab), 1e-12)
        loop_oracle[arm] = {"harness": mine, "lab": lab, "rel_diff": rel}
        if not rel <= ORACLE_RTOL:
            _write_once(work / "oracle-refusal.json", {"loop_oracle": loop_oracle})
            raise SystemExit(
                f"loop oracle failed for {arm}: harness {mine!r} vs lab {lab!r}"
            )

    start = int(np.searchsorted(data.open_ns, _ns(OOS_START), side="left"))
    stop = len(data.open_ns)
    if int(data.open_ns[start]) != _ns(OOS_START):
        raise SystemExit("no bar opens exactly at the OOS start")

    cells: dict[str, dict[str, Any]] = {}
    for name, (arm, cost, lag, mask_lag, _gate) in CELLS.items():
        p10, p17, econ = packs[lag]
        st = replay_cell(
            **lab_kw,
            p10=p10,
            p10_select=with_lagged_masks(p10, mask_lag),
            p17=p17,
            econ=econ,
            arm=arm,
            cost=cost,
            start=start,
            stop=stop,
            cap=GROSS_CAP,
        )
        st["lag"] = lag
        st["mask_lag"] = mask_lag
        cells[name] = st

    decision = decide(stage, cells)
    result = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": digest,
        "commit": commit,
        "stage": stage.name,
        "snapshot": pre["pins"],
        "authorization": auth,
        "interim_result": interim,
        "claim": str(claim),
        "pretrade_py_path": pretrade[0],
        "pretrade_py_sha256": pretrade[1],
        "build_receipt_sha256": sha256_file(build_receipt_path),
        "cache_files": cache_files,
        "cache_manifest_digest": manifest_digest(cache_files),
        "window": [OOS_START.isoformat(), stage.data_end.isoformat()],
        "bars": stop - start,
        "parity_worst": worst,
        "parity_fields": len(parity),
        "loop_oracle": loop_oracle,
        "cells": cells,
        "cash_return": 0.0,
        **decision,
        "run_at": now.isoformat(),
    }
    out_dir.mkdir(parents=True)
    _write_once(out_dir / "result.json", result)
    write_ledger_result(
        stage.name, digest, result["label"], sha256_file(out_dir / "result.json")
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
    for name, st in cells.items():
        print(
            f"{name:22s} ret {st['total_return'] * 100:+7.2f}% dd {st['max_drawdown'] * 100:5.2f}% "
            f"sr {st['annual_sharpe']:+.2f} valid {st['valid']}"
        )


def cmd_digest(_args: argparse.Namespace) -> None:
    print(protocol_digest())


def cmd_snapshot_digest(args: argparse.Namespace) -> None:
    snap = Path(args.snapshot)
    print(
        json.dumps(
            {
                "lab": manifest_digest(manifest(snap / "lab", "*.py")),
                "frozen_cache": manifest_digest(
                    manifest(snap / "frozen_cache", "*.npz")
                ),
            }
        )
    )


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="agy_clean_all_forward")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("digest").set_defaults(fn=cmd_digest)
    s = sub.add_parser("snapshot-digest")
    s.add_argument("--snapshot", required=True)
    s.set_defaults(fn=cmd_snapshot_digest)
    for name, fn in (("prepare", cmd_prepare), ("build", cmd_build), ("run", cmd_run)):
        p = sub.add_parser(name)
        p.add_argument("--work", required=True)
        p.add_argument("--stage", required=True, choices=sorted(STAGES))
        p.add_argument("--snapshot", required=True)
        if name != "prepare":
            p.add_argument("--authorization-url", required=True)
        if name == "run":
            p.add_argument("--output-dir", required=True)
            p.add_argument("--interim-result", default=None)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
