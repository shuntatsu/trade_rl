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
import json
import math
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROTOCOL_ID = "agy_clean_all_forward_v1"

# Snapshot of agy session 7e52c53c scratch/rl_lab, taken 2026-11-22 (47 top-level .py files).
LAB_MANIFEST_SHA256 = "c4457c8693ba64980f7735c54cc71b4a1058735f27ab35d46be2c4d9a09b1683"
# agy's own pre-2026-08 round10/round17 pack caches: the parity oracle.
FROZEN_CACHE_MANIFEST_SHA256 = (
    "60aaea5e3243dbea7917e08d6fd53504802afb54c3edff5563d26454fb53305a"
)
FROZEN_PACK10 = "round10_mtf6_pack_pre20260801.npz"
FROZEN_PACK17 = "round17_clean_mtf_pack_pre20260801.npz"

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

# name -> (arm, one-way cost, signal lag in bars, gates)
CELLS: dict[str, tuple[str, float, int, bool]] = {
    "CANDIDATE__BASE": (ARM_CANDIDATE, 0.0006, 1, True),
    "CANDIDATE__HIGH_COST": (ARM_CANDIDATE, 0.0012, 1, True),
    "CANDIDATE__STRESS": (ARM_CANDIDATE, 0.0020, 1, False),
    "CANDIDATE__NO_DELAY": (ARM_CANDIDATE, 0.0006, 0, False),
    "CONTROL_ANC__BASE": (ARM_CONTROL, 0.0006, 1, False),
}

BARS_PER_YEAR = 2190.0
FINAL_MIN_SHARPE = 0.5
MAX_DD_LIMIT = 0.20
EARLY_STOP_RETURN = -0.08
PARITY_ATOL = 1e-12
CONSERVATION_TOL = 1e-9

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
    }


def protocol_digest() -> str:
    source = Path(__file__).read_bytes().replace(b"\r\n", b"\n")
    payload = json.dumps(
        {
            "constants": _constants(),
            "harness_sha256": hashlib.sha256(source).hexdigest(),
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


def comment_id_from_url(url: str) -> int:
    marker = "#issuecomment-"
    if not url.startswith(f"https://github.com/{REPO}/issues/810") or marker not in url:
        raise SystemExit("authorization URL must be an Issue #810 comment")
    return int(url.split(marker, 1)[1])


def authorization_receipt(
    url: str, digest: str, fetch: Any = _gh_comment
) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for role, cid in (
        ("g4_relay", comment_id_from_url(url)),
        ("data_scope", DATA_SCOPE_COMMENT_ID),
    ):
        first, second = fetch(cid), fetch(cid)
        if first["user"]["login"] != OWNER_LOGIN:
            raise SystemExit(f"{role} comment {cid} is not owner-authored")
        if (
            first["body"] != second["body"]
            or first["updated_at"] != second["updated_at"]
        ):
            raise SystemExit(f"{role} comment {cid} changed between fetches")
        if role == "g4_relay" and (
            digest not in first["body"] or PROTOCOL_ID not in first["body"]
        ):
            raise SystemExit("G4 relay does not quote this protocol id and digest")
        out[role] = {
            "comment_id": cid,
            "author": first["user"]["login"],
            "updated_at": first["updated_at"],
            "body_sha256": hashlib.sha256(first["body"].encode("utf-8")).hexdigest(),
        }
    return out


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
    gate_dd = max(cells[k]["max_drawdown"] for k, v in CELLS.items() if v[3])
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


def cmd_build(args: argparse.Namespace) -> None:
    work = Path(args.work)
    for script in BUILD_SCRIPTS:
        print(f"[build] {script}", flush=True)
        subprocess.run(
            [sys.executable, "-u", str(work / "lab" / script)],
            cwd=work / "lab",
            env=_lab_env(work),
            check=True,
        )
    caches = manifest(work / "lab" / "cache", "*.npz")
    _write_once(work / "build-receipt.json", {"raw_caches": caches})
    print(json.dumps(caches, indent=1))


def _write_once(path: Path, obj: dict[str, Any]) -> None:
    if path.exists():
        raise SystemExit(f"refusing to overwrite {path}")
    path.write_text(json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")


def cmd_run(
    args: argparse.Namespace,
) -> None:  # pragma: no cover - needs the lab and its data
    stage = STAGES[args.stage]
    now = datetime.now(tz=UTC)
    if now < stage.not_before:
        raise SystemExit(
            f"stage {stage.name} may not run before {stage.not_before.isoformat()}"
        )
    work, snapshot, out_dir = (
        Path(args.work),
        Path(args.snapshot),
        Path(args.output_dir),
    )
    if out_dir.exists():
        raise SystemExit(f"output directory exists: {out_dir}")
    pins = verify_snapshot(snapshot)
    digest = protocol_digest()
    prep = json.loads((work / "prepare-receipt.json").read_text(encoding="utf-8"))
    if prep["protocol_digest"] != digest or prep["stage"] != stage.name:
        raise SystemExit("work directory was prepared under another digest or stage")
    auth = authorization_receipt(args.authorization_url, digest)
    sys.path.insert(0, str(work / "lab"))
    import numpy as np
    import run_round17_breakthrough_gpu as r17  # type: ignore[import-not-found]

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
    end_ns = int(stage.data_end.timestamp() * 1e9)
    if int(data.close_ns[-1]) != end_ns:
        raise SystemExit(f"dataset does not end at {stage.data_end.isoformat()}")
    p4 = r17.r4f.build_round4_pack(data, m1h)
    p5 = r17.r5f.build_round5_mtf_pack(data, m1h, m15m, p4)
    p6 = r17.r6f.build_round6_aw_pack(data, m1h, m15m, p4, p5)
    p7 = r17.r7f.build_round7_frontier_pack(data, m1h, m15m, p4, p5, p6)
    p8 = r17.r8f.build_round8_nonlinear_pack(data, m1h, m15m, p4, p5, p6, p7)
    pack10 = r10f.build_round10_mtf6_pack(data, m1h, m15m, p4, p5, p6, p7, p8)
    pack17 = r17f.build_round17_clean_pack(data, m1h, m15m, p4, p5, p6, p7, p8, pack10)

    # ---- parity oracle: every pre-2026-08 row equals agy's frozen packs. The last frozen row is
    # excluded: its next-bar return and funding belong to the first out-of-sample bar.
    n_pre = int(
        np.searchsorted(data.close_ns, int(LAB_EMBARGO.timestamp() * 1e9), side="right")
    )
    n_cmp = n_pre - 1
    frozen10 = r10f.load_r10_pack(snapshot / "frozen_cache" / FROZEN_PACK10)
    frozen17 = r17f.load_r17_pack(snapshot / "frozen_cache" / FROZEN_PACK17)
    parity: dict[str, float] = {}
    for tag, new, old in (("pack10", pack10, frozen10), ("pack17", pack17, frozen17)):
        for field in old.__dataclass_fields__:
            a, b = getattr(new, field), getattr(old, field)
            if not isinstance(b, np.ndarray) or b.ndim == 0 or b.shape[0] != n_pre:
                continue
            a, b = np.asarray(a)[:n_cmp], b[:n_cmp]
            if a.shape != b.shape:
                raise SystemExit(
                    f"parity shape mismatch {tag}.{field}: {a.shape} vs {b.shape}"
                )
            if b.dtype.kind in "fc":
                both_nan = np.isnan(a) & np.isnan(b)
                diff = float(
                    np.max(np.where(both_nan, 0.0, np.abs(a - b)), initial=0.0)
                )
            else:
                diff = float(np.sum(a != b))
            parity[f"{tag}.{field}"] = diff
    worst = max(parity.values(), default=math.inf)
    if not parity or worst > PARITY_ATOL:
        bad = {k: v for k, v in parity.items() if v > PARITY_ATOL}
        _write_once(work / "parity-refusal.json", {"parity": parity, "n_pre": n_pre})
        raise SystemExit(
            f"parity oracle failed before any return was computed: {sorted(bad)[:10]}"
        )

    econ0 = r15g.compute_r15_econ_controls(r12g.compute_r12_opp_feats(pack10))
    packs = {
        0: (pack10, pack17, econ0),
        1: (
            r10f.lag_r10_pack(pack10, 1),
            r17f.lag_r17_pack(pack17, 1),
            r15g.lag_r15_econ(econ0, 1),
        ),
    }
    start = int(
        np.searchsorted(data.open_ns, int(OOS_START.timestamp() * 1e9), side="left")
    )
    stop = len(data.open_ns)
    if int(data.open_ns[start]) != int(OOS_START.timestamp() * 1e9):
        raise SystemExit("no bar opens exactly at the OOS start")

    cells: dict[str, dict[str, Any]] = {}
    for name, (arm, cost, lag, _gate) in CELLS.items():
        p10, p17, econ = packs[lag]
        n = len(p10.symbols)
        ws, wp, eq = np.zeros(n), np.zeros(n), 1.0
        lrs: list[float] = []
        valid, reasons, max_err, turn, cost_sum, capped, decisions = (
            True,
            [],
            0.0,
            0.0,
            0.0,
            0,
            0,
        )
        for i in range(stop - start):
            k = start + i
            if i % DECISION_STRIDE == 0:
                db, _f, a_m, c_m = r15g._econ_scalars(econ, k, ECON_MODE)
                alpha, scale = r17.nonrl_alpha_and_scale_np(p10, p17, arm, k)
                ws_t, wp_t = r17.synthesize_r17_np(p10, k, alpha, scale, a_m, c_m)
                ws_t, wp_t, was = cap_gross(ws_t, wp_t)
                capped += int(was)
                decisions += 1
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
            gross_e = float(np.abs(ws_e).sum() + np.abs(wp_e).sum())
            if err > CONSERVATION_TOL or not math.isfinite(eq_n) or eq_n <= 0.0:
                valid = False
                reasons.append(f"accounting@{k}")
            if gross_e > GROSS_CAP + 1e-9:
                valid = False
                reasons.append(f"gross_above_cap@{k}")
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
                "arm": arm,
                "one_way_cost": cost,
                "lag": lag,
            }
        )
        cells[name] = st

    decision = decide(stage, cells)
    result = {
        "protocol_id": PROTOCOL_ID,
        "protocol_digest": digest,
        "stage": stage.name,
        "snapshot": pins,
        "authorization": auth,
        "window": [OOS_START.isoformat(), stage.data_end.isoformat()],
        "bars": stop - start,
        "parity_worst": worst,
        "parity_fields": len(parity),
        "cells": cells,
        "cash_return": 0.0,
        **decision,
        "run_at": now.isoformat(),
    }
    out_dir.mkdir(parents=True)
    _write_once(out_dir / "result.json", result)
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
        if name != "build":
            p.add_argument("--stage", required=True, choices=sorted(STAGES))
            p.add_argument("--snapshot", required=True)
        if name == "run":
            p.add_argument("--authorization-url", required=True)
            p.add_argument("--output-dir", required=True)
        p.set_defaults(fn=fn)
    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
