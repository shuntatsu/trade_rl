"""Tests for the agy CLEAN_ALL forward harness (no lab, no network, no data)."""

from __future__ import annotations

import dataclasses
import json
import types
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import agy_clean_all_forward as fwd

PINNED_DIGEST = "dacd5039d7cd7271a98467e0f27e5fd62d0303b7eee43aa51687b48a75851204"


def test_protocol_digest_is_pinned() -> None:
    assert fwd.protocol_digest() == PINNED_DIGEST


def test_digest_tracks_constants(monkeypatch: pytest.MonkeyPatch) -> None:
    base = fwd.protocol_digest()
    monkeypatch.setattr(fwd, "GROSS_CAP", 1.5)
    assert fwd.protocol_digest() != base


def test_manifest_digest_is_order_independent(tmp_path: Path) -> None:
    (tmp_path / "b.py").write_text("b")
    (tmp_path / "a.py").write_text("a")
    m = fwd.manifest(tmp_path, "*.py")
    assert list(m) == ["a.py", "b.py"]
    assert fwd.manifest_digest(m) == fwd.manifest_digest(
        dict(reversed(list(m.items())))
    )


def test_verify_snapshot_refuses_modified_lab(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "lab").mkdir()
    (tmp_path / "frozen_cache").mkdir()
    (tmp_path / "lab" / "x.py").write_text("x = 1\n")
    (tmp_path / "frozen_cache" / "p.npz").write_bytes(b"z")
    lab = fwd.manifest_digest(fwd.manifest(tmp_path / "lab", "*.py"))
    cache = fwd.manifest_digest(fwd.manifest(tmp_path / "frozen_cache", "*.npz"))
    monkeypatch.setattr(fwd, "LAB_MANIFEST_SHA256", lab)
    monkeypatch.setattr(fwd, "FROZEN_CACHE_MANIFEST_SHA256", cache)
    assert fwd.verify_snapshot(tmp_path) == {"lab": lab, "frozen_cache": cache}
    (tmp_path / "lab" / "x.py").write_text("x = 2\n")
    with pytest.raises(SystemExit, match="lab snapshot manifest mismatch"):
        fwd.verify_snapshot(tmp_path)


LAB_SAMPLE = """
FS1_WARMUP_START_UTC = datetime(2026, 8, 1, 0, 0, tzinfo=UTC)
END = date(2026, 7, 31)
SQL = "AND k.open_time < '2026-08-01 00:00:00+00'"
assert close <= 1785542400.0
PATH = "real_dev_4h_pre20260801.npz"
"""


def test_embargo_patch_moves_every_literal(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    (src / "m.py").write_text(LAB_SAMPLE)
    end = datetime(2026, 9, 30, tzinfo=UTC)
    counts = fwd.apply_patches(src, dst, fwd.embargo_patches(end))
    out = (dst / "m.py").read_text()
    assert "datetime(2026, 9, 30, 0, 0, tzinfo=UTC)" in out
    assert "date(2026, 9, 29)" in out
    assert "'2026-09-30 00:00:00+00'" in out
    assert str(int(end.timestamp())) in out
    assert "real_dev_4h_fwd20260930.npz" in out
    assert sum(counts["m.py"].values()) == 5
    for bad in fwd.RESIDUE_FORBIDDEN:
        assert bad not in out


def test_embargo_patch_refuses_unmatched_literal(tmp_path: Path) -> None:
    src = tmp_path / "src"
    src.mkdir()
    (src / "m.py").write_text("x = 1\n")
    with pytest.raises(SystemExit, match="never matched"):
        fwd.apply_patches(
            src,
            tmp_path / "dst",
            fwd.embargo_patches(datetime(2026, 9, 30, tzinfo=UTC)),
        )


def test_embargo_patch_refuses_non_empty_destination(tmp_path: Path) -> None:
    src, dst = tmp_path / "src", tmp_path / "dst"
    src.mkdir()
    dst.mkdir()
    (dst / "old.py").write_text("")
    (src / "m.py").write_text(LAB_SAMPLE)
    with pytest.raises(SystemExit, match="not empty"):
        fwd.apply_patches(
            src, dst, fwd.embargo_patches(datetime(2026, 9, 30, tzinfo=UTC))
        )


def test_embargo_patch_requires_midnight() -> None:
    with pytest.raises(ValueError):
        fwd.embargo_patches(datetime(2026, 9, 30, 16, tzinfo=UTC))


def test_stages_are_frozen() -> None:
    assert fwd.STAGES["INTERIM_A"].data_end == datetime(2026, 9, 30, tzinfo=UTC)
    assert not fwd.STAGES["INTERIM_A"].final
    assert fwd.STAGES["FINAL"].data_end == datetime(2027, 5, 1, tzinfo=UTC)
    assert fwd.STAGES["FINAL"].final
    assert fwd.OOS_START == datetime(2026, 8, 1, tzinfo=UTC)


def test_cap_gross_scales_proportionally() -> None:
    ws, wp = np.array([0.6, 0.0]), np.array([0.5, -0.9])
    cs, cp, capped = fwd.cap_gross(ws, wp)
    assert capped
    assert np.isclose(np.abs(cs).sum() + np.abs(cp).sum(), 1.0)
    assert np.isclose(cs[0], 0.3)
    assert np.isclose(cp[1] / cp[0], wp[1] / wp[0])
    small_s, small_p, capped2 = fwd.cap_gross(ws * 0.1, wp * 0.1)
    assert (
        not capped2
        and np.allclose(small_s, ws * 0.1)
        and np.allclose(small_p, wp * 0.1)
    )


def test_cell_stats() -> None:
    st = fwd.cell_stats([0.01, -0.02, 0.005])
    assert np.isclose(st["total_return"], np.exp(-0.005) - 1)
    assert np.isclose(st["max_drawdown"], 1 - np.exp(-0.02))
    assert st["bars"] == 3


def _cells(**over: dict[str, Any]) -> dict[str, dict[str, Any]]:
    good = {
        "valid": True,
        "total_return": 0.05,
        "max_drawdown": 0.05,
        "annual_sharpe": 1.0,
    }
    cells = {name: dict(good) for name in fwd.CELLS}
    for name, patch in over.items():
        cells[name].update(patch)
    return cells


def test_final_supported_and_each_condition() -> None:
    final = fwd.STAGES["FINAL"]
    assert fwd.decide(final, _cells())["label"] == "CLEAN_ALL_FORWARD_SUPPORTED"
    cases = {
        "C1_base_return_positive": {"CANDIDATE__BASE": {"total_return": 0.0}},
        "C2_base_sharpe": {"CANDIDATE__BASE": {"annual_sharpe": 0.49}},
        "C3_high_cost_return_positive": {
            "CANDIDATE__HIGH_COST": {"total_return": -0.01}
        },
        "C4_gating_drawdown": {"CANDIDATE__HIGH_COST": {"max_drawdown": 0.20}},
    }
    for cond, over in cases.items():
        out = fwd.decide(final, _cells(**over))
        assert out["label"] == "CLEAN_ALL_FORWARD_NOT_SUPPORTED"
        assert out["failed_conditions"] == [cond]


def test_reported_cells_never_gate() -> None:
    final = fwd.STAGES["FINAL"]
    bad = {"total_return": -0.5, "max_drawdown": 0.6, "annual_sharpe": -3.0}
    out = fwd.decide(
        final,
        _cells(CANDIDATE__STRESS=bad, CANDIDATE__NO_DELAY=bad, CONTROL_ANC__BASE=bad),
    )
    assert out["label"] == "CLEAN_ALL_FORWARD_SUPPORTED"


def test_invalid_cell_wins() -> None:
    out = fwd.decide(fwd.STAGES["FINAL"], _cells(CONTROL_ANC__BASE={"valid": False}))
    assert out["label"] == "INVALID" and out["invalid_cells"] == ["CONTROL_ANC__BASE"]


def test_interim_is_not_a_decision_and_early_stop() -> None:
    interim = fwd.STAGES["INTERIM_A"]
    assert (
        fwd.decide(interim, _cells(CANDIDATE__BASE={"total_return": -0.07}))["label"]
        == "INTERIM_NOT_A_DECISION"
    )
    out = fwd.decide(interim, _cells(CANDIDATE__BASE={"total_return": -0.08}))
    assert out["label"] == "STOPPED_EARLY"
    out = fwd.decide(interim, _cells(CANDIDATE__HIGH_COST={"max_drawdown": 0.2}))
    assert out["failed_conditions"] == ["gating_drawdown_at_or_above_limit"]


def test_decide_refuses_wrong_cell_set() -> None:
    cells = _cells()
    cells.pop("CANDIDATE__STRESS")
    with pytest.raises(ValueError):
        fwd.decide(fwd.STAGES["FINAL"], cells)


COMMIT = "a" * 40
ISSUE = "https://api.github.com/repos/shuntatsu/trade_rl/issues/810"


def _comment(
    body: str,
    login: str = "shuntatsu",
    updated: str = "2026-11-22T00:00:00Z",
    issue_url: str = ISSUE,
) -> dict[str, Any]:
    return {
        "user": {"login": login},
        "body": body,
        "updated_at": updated,
        "issue_url": issue_url,
    }


def _relay(digest: str) -> str:
    return f"relay {fwd.PROTOCOL_ID} {digest} {COMMIT}"


def test_authorization_receipt_binds_owner_digest_commit_and_scope() -> None:
    digest = fwd.protocol_digest()
    url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-123"
    comments = {
        123: _comment(_relay(digest)),
        fwd.DATA_SCOPE_COMMENT_ID: _comment("scope"),
    }

    def fetch(cid: int) -> dict[str, Any]:
        return comments[cid]

    rec = fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    assert rec["g4_relay"]["comment_id"] == 123
    assert rec["data_scope"]["comment_id"] == fwd.DATA_SCOPE_COMMENT_ID
    comments[123] = _comment(f"relay {fwd.PROTOCOL_ID} other-digest {COMMIT}")
    with pytest.raises(SystemExit, match="does not quote"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    comments[123] = _comment(f"relay {fwd.PROTOCOL_ID} {digest} {'b' * 40}")
    with pytest.raises(SystemExit, match="does not quote"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    comments[123] = _comment(f"relay {fwd.PROTOCOL_ID} {digest} {COMMIT[:12]}")
    with pytest.raises(SystemExit, match="does not quote"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    comments[123] = _comment(_relay(digest), login="someone")
    with pytest.raises(SystemExit, match="not owner"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    comments[123] = _comment(_relay(digest), issue_url=ISSUE + "1")
    with pytest.raises(SystemExit, match="not on Issue #810"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=fetch)
    comments[123] = _comment(_relay(digest))
    with pytest.raises(SystemExit, match="40-hex"):
        fwd.authorization_receipt(url, digest, "abc", fetch=fetch)


def test_authorization_receipt_refuses_edit_between_fetches() -> None:
    digest = fwd.protocol_digest()
    seq = iter([_comment(_relay(digest)), _comment(_relay(digest) + " edited")])
    url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-5"
    with pytest.raises(SystemExit, match="changed between fetches"):
        fwd.authorization_receipt(url, digest, COMMIT, fetch=lambda cid: next(seq))


@pytest.mark.parametrize(
    "url",
    [
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/8101#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1x",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1\n",
        "http://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
    ],
)
def test_authorization_url_must_be_exactly_issue_810(url: str) -> None:
    with pytest.raises(SystemExit):
        fwd.comment_id_from_url(url)


def test_authorization_url_parses_comment_id() -> None:
    url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-42"
    assert fwd.comment_id_from_url(url) == 42


def test_harness_commit_refuses_dirty_and_bad_head(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = {"status": " M x.py\n", "head": COMMIT + "\n"}

    def fake_git(*args: str) -> str:
        return state["status"] if args[0] == "status" else state["head"]

    monkeypatch.setattr(fwd, "_git", fake_git)
    with pytest.raises(SystemExit, match="tracked modifications"):
        fwd.harness_commit()
    state["status"] = ""
    assert fwd.harness_commit() == COMMIT
    state["head"] = "HEAD\n"
    with pytest.raises(SystemExit, match="unexpected HEAD"):
        fwd.harness_commit()


def test_claim_stage_is_single_use(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(fwd, "LEDGER_DIR", tmp_path / "ledger")
    now = datetime(2026, 11, 22, tzinfo=UTC)
    path = fwd.claim_stage("INTERIM_A", "d", COMMIT, 7, now)
    body = json.loads(path.read_text())
    assert body["commit"] == COMMIT and body["authorization_comment_id"] == 7
    assert path == tmp_path / "ledger" / "INTERIM_A.claim"
    with pytest.raises(SystemExit, match="already claimed"):
        fwd.claim_stage("INTERIM_A", "d", COMMIT, 8, now)
    fwd.claim_stage("FINAL", "d", COMMIT, 9, now)


def _interim(tmp_path: Path, **over: Any) -> Path:
    obj = {
        "protocol_digest": "dig",
        "stage": "INTERIM_A",
        "label": "INTERIM_NOT_A_DECISION",
    }
    obj.update(over)
    path = tmp_path / "result.json"
    path.write_text(json.dumps(obj))
    return path


def test_check_interim_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fwd, "LEDGER_DIR", tmp_path / "ledger")
    with pytest.raises(SystemExit, match="no INTERIM_A ledger claim"):
        fwd.check_interim_result(_interim(tmp_path), "dig")
    fwd.claim_stage("INTERIM_A", "other", COMMIT, 1, datetime(2026, 11, 22, tzinfo=UTC))
    with pytest.raises(SystemExit, match="ledger claim has another"):
        fwd.check_interim_result(_interim(tmp_path), "dig")
    (tmp_path / "ledger" / "INTERIM_A.claim").write_text('{"protocol_digest": "dig"}')
    rec = fwd.check_interim_result(_interim(tmp_path), "dig")
    assert rec["label"] == "INTERIM_NOT_A_DECISION" and len(rec["sha256"]) == 64
    for over, match in (
        ({"protocol_digest": "x"}, "digest"),
        ({"stage": "FINAL"}, "INTERIM_A"),
        ({"label": "STOPPED_EARLY"}, "does not allow FINAL"),
        ({"label": "INVALID"}, "does not allow FINAL"),
    ):
        with pytest.raises(SystemExit, match=match):
            fwd.check_interim_result(_interim(tmp_path, **over), "dig")


def test_verify_work_lab(tmp_path: Path) -> None:
    (tmp_path / "lab").mkdir()
    (tmp_path / "lab" / "a.py").write_text("a = 1\n")
    prep = {"lab_files": fwd.manifest(tmp_path / "lab", "*.py")}
    fwd.verify_work_lab(tmp_path, prep)
    (tmp_path / "lab" / "a.py").write_text("a = 2\n")
    with pytest.raises(SystemExit, match="differs from the prepare receipt"):
        fwd.verify_work_lab(tmp_path, prep)


def test_digest_binds_spec_text_but_not_digest_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    spec = tmp_path / fwd.SPEC_RELPATH
    spec.parent.mkdir(parents=True)
    monkeypatch.setattr(fwd, "REPO_ROOT", tmp_path)
    spec.write_bytes(b"# t\nProtocol digest: `aaa`\nbody\n")
    base = fwd.protocol_digest()
    spec.write_bytes(b"# t\r\nProtocol digest: `bbb`\r\nbody\r\n")
    assert fwd.protocol_digest() == base
    spec.write_bytes(b"# t\nProtocol digest: `aaa`\nbody changed\n")
    assert fwd.protocol_digest() != base


# ----------------------------------------------------------------------------- replay_cell


class _Pack:
    def __init__(self, t: int, n: int, ret: float) -> None:
        self.symbols = tuple(f"S{i}" for i in range(n))
        self.spot_valid = np.ones((t, n), dtype=bool)
        self.perp_valid = np.ones((t, n), dtype=bool)
        self.spot_next_ret = np.full((t, n), ret)
        self.perp_next_ret = np.zeros((t, n))
        self.funding_realized = np.zeros((t, n))


def _fake_lab(
    target: float, db: float = 0.05, perp: float = 0.0
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Fake lab: target spot weight ``target`` on symbol 0, no costs, drifting books."""
    log: dict[str, Any] = {"synth_k": [], "calls": []}

    def execute_hybrid_step(**kw: Any) -> tuple[Any, ...]:
        log["calls"].append(kw)
        ws_e, wp_e = kw["w_spot_target"], kw["w_perp_target"]
        eq = kw["equity_before"]
        px = eq * float(ws_e @ kw["spot_next_ret"])
        ws_n = ws_e * (1.0 + kw["spot_next_ret"])
        return ws_e, wp_e, eq + px, ws_n, wp_e, 0.0, 0.0, px, 0.0

    def synthesize(p10: Any, k: int, alpha: Any, scale: Any, a_m: Any, c_m: Any) -> Any:
        log["synth_k"].append(k)
        ws = np.zeros(len(p10.symbols))
        ws[0] = target
        wp = np.zeros(len(p10.symbols))
        wp[0] = perp
        return ws, wp

    he = types.SimpleNamespace(execute_hybrid_step=execute_hybrid_step)
    r3c = types.SimpleNamespace(IMPACT_QUAD=0.002)
    r15g = types.SimpleNamespace(
        _econ_scalars=lambda econ, k, mode: (db, 0.01, 1.0, 1.0)
    )
    r17 = types.SimpleNamespace(
        nonrl_alpha_and_scale_np=lambda p10, p17, arm, k: (np.zeros(2), 1.0),
        synthesize_r17_np=synthesize,
    )
    return {"he": he, "r3c": r3c, "r15g": r15g, "r17": r17}, log


def _replay(
    lab: dict[str, Any], cap: float | None, ret: float = 0.01
) -> dict[str, Any]:
    pack = _Pack(30, 2, ret)
    return fwd.replay_cell(
        **lab,
        p10=pack,
        p17=None,
        econ=None,
        arm="X",
        cost=0.0006,
        start=3,
        stop=3 + 14,
        cap=cap,
    )


def test_replay_cell_stride_hold_and_cap() -> None:
    lab, log = _fake_lab(target=3.0)
    st = _replay(lab, cap=1.0, ret=0.0)
    assert log["synth_k"] == [3, 9, 15]
    assert st["decisions"] == 3 and st["cap_bound_decisions"] == 3
    calls = log["calls"]
    assert len(calls) == 14
    assert np.isclose(calls[0]["w_spot_target"][0], 1.0)  # capped from 3.0
    assert calls[0]["deadband"] == 0.05
    for i in (1, 2, 3, 4, 5, 7):  # held bars: current weights pass through, deadband 0
        assert calls[i]["deadband"] == 0.0
        assert calls[i]["w_spot_target"] is calls[i]["w_spot_cur"]
        assert calls[i]["w_perp_target"] is calls[i]["w_perp_cur"]
    assert calls[6]["deadband"] == 0.05
    assert st["valid"] and st["invalid_reasons"] == []


def test_replay_cell_uncapped_keeps_target() -> None:
    lab, log = _fake_lab(target=3.0)
    st = _replay(lab, cap=None)
    assert np.isclose(log["calls"][0]["w_spot_target"][0], 3.0)
    assert st["cap_bound_decisions"] == 0
    assert not any(r.startswith("target_gross") for r in st["invalid_reasons"])


def test_replay_cell_reports_executed_gross_drift_without_invalidating() -> None:
    lab, _ = _fake_lab(target=0.5, perp=-0.5)  # gross 1.0, cash need 0.7
    st = _replay(lab, cap=1.0, ret=0.02)
    assert st["max_executed_gross"] > 1.0 + 1e-6
    assert st["valid"] and st["invalid_reasons"] == []


def test_replay_cell_target_gross_above_cap_is_invalid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lab, _ = _fake_lab(target=3.0)
    monkeypatch.setattr(fwd, "cap_gross", lambda ws, wp, cap=1.0: (ws, wp, False))
    st = _replay(lab, cap=1.0)
    assert not st["valid"]
    assert st["invalid_reasons"][0].startswith("target_gross_above_cap@")


# ----------------------------------------------------------------------------- parity & splice


@dataclasses.dataclass
class _DC:
    a: np.ndarray
    b: np.ndarray
    s: tuple[str, ...] = ("x",)


def test_pack_parity_compares_rows_and_requires_fields() -> None:
    old = _DC(np.arange(5.0), np.array([True, False, True, True, False]))
    new = _DC(np.arange(7.0), np.array([True, False, True, True, False, True, True]))
    par = fwd.pack_parity("p", new, old, ("a", "b"), 5, 4)
    assert par == {"p.a": 0.0, "p.b": 0.0}
    new.a[2] = 9.0
    assert fwd.pack_parity("p", new, old, ("a",), 5, 4)["p.a"] == 7.0
    with pytest.raises(SystemExit, match="required fields"):
        fwd.pack_parity("p", new, old, ("a", "missing"), 5, 4)
    with pytest.raises(SystemExit, match="required fields"):
        fwd.pack_parity("p", new, old, ("s",), 5, 4)  # not an array


def test_econ_parity() -> None:
    keys = {k: np.arange(6.0) for k in fwd.REQUIRED_ECON}
    other = {k: v.copy() for k, v in keys.items()}
    par = fwd.econ_parity(keys, other, 5)
    assert set(par) == {f"econ.{k}" for k in keys} and max(par.values()) == 0.0
    other["a_mult"][1] = 2.0
    assert fwd.econ_parity(keys, other, 5)["econ.a_mult"] == 1.0
    other["a_mult"][5] = 99.0  # beyond n_cmp is ignored
    other["a_mult"][1] = 1.0
    assert fwd.econ_parity(keys, other, 5)["econ.a_mult"] == 0.0
    with pytest.raises(SystemExit, match="econ key mismatch"):
        fwd.econ_parity(keys, {"a_mult": np.arange(6.0)}, 5)


def _metrics(
    path: Path, rows: int, fill: float, symbols: tuple[str, ...] = ("A", "B")
) -> None:
    n = len(symbols)
    np.savez_compressed(
        path,
        symbols=np.array(symbols),
        oi_value=np.full((rows, n), fill),
        metrics_valid=np.ones((rows, n), dtype=bool),
    )


def _splice_files(
    tmp_path: Path, frozen_rows: int = 4, fresh_rows: int = 6
) -> tuple[Path, Path, Path, int]:
    base = tmp_path / "base.npz"
    np.savez_compressed(base, open_ns=np.arange(fresh_rows, dtype=np.int64) * 10)
    fresh, frozen = tmp_path / "fresh.npz", tmp_path / "frozen.npz"
    _metrics(fresh, fresh_rows, 2.0)
    _metrics(frozen, frozen_rows, 1.0)
    return fresh, frozen, base, 40  # embargo at open_ns 40 -> 4 pre rows


def test_splice_replaces_pre_embargo_rows_only(tmp_path: Path) -> None:
    fresh, frozen, base, embargo = _splice_files(tmp_path)
    rec = fwd.splice_frozen_metrics(fresh, frozen, base, embargo)
    assert rec["spliced_rows"] == 4 and rec["sha256_before"] != rec["sha256_after"]
    assert rec["sha256_after"] == fwd.sha256_file(fresh)
    with np.load(fresh) as z:
        assert z["oi_value"][:4].tolist() == [[1.0, 1.0]] * 4
        assert z["oi_value"][4:].tolist() == [[2.0, 2.0]] * 2
        assert z["metrics_valid"].all() and z["symbols"].tolist() == ["A", "B"]


def test_splice_refuses_mismatches(tmp_path: Path) -> None:
    fresh, frozen, base, embargo = _splice_files(tmp_path, frozen_rows=3)
    with pytest.raises(SystemExit, match="row count"):
        fwd.splice_frozen_metrics(fresh, frozen, base, embargo)
    fresh, frozen, base, embargo = _splice_files(tmp_path)
    _metrics(frozen, 4, 1.0, symbols=("A", "C"))
    with pytest.raises(SystemExit, match="symbols"):
        fwd.splice_frozen_metrics(fresh, frozen, base, embargo)
    fresh, frozen, base, embargo = _splice_files(tmp_path, fresh_rows=3)
    np.savez_compressed(base, open_ns=np.arange(6, dtype=np.int64) * 10)
    with pytest.raises(SystemExit, match="does not cover"):
        fwd.splice_frozen_metrics(fresh, frozen, base, embargo)


def test_only_requires_exactly_one_match(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="exactly one"):
        fwd._only(tmp_path, "m_fwd*.npz")
    (tmp_path / "m_fwd1.npz").write_bytes(b"")
    assert fwd._only(tmp_path, "m_fwd*.npz").name == "m_fwd1.npz"
    (tmp_path / "m_fwd2.npz").write_bytes(b"")
    with pytest.raises(SystemExit, match="exactly one"):
        fwd._only(tmp_path, "m_fwd*.npz")


def test_write_once(tmp_path: Path) -> None:
    fwd._write_once(tmp_path / "r.json", {"a": 1})
    assert json.loads((tmp_path / "r.json").read_text()) == {"a": 1}
    with pytest.raises(SystemExit):
        fwd._write_once(tmp_path / "r.json", {"a": 2})


def test_run_refuses_before_not_before(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Clock(datetime):
        @classmethod
        def now(cls, tz: Any = None) -> datetime:  # type: ignore[override]
            return datetime(2027, 5, 1, tzinfo=UTC)

    monkeypatch.setattr(fwd, "datetime", _Clock)
    with pytest.raises(SystemExit, match="may not run before"):
        fwd.main(
            [
                "run",
                "--stage",
                "FINAL",
                "--work",
                str(tmp_path),
                "--snapshot",
                str(tmp_path),
                "--authorization-url",
                "u",
                "--output-dir",
                str(tmp_path / "o"),
                "--interim-result",
                str(tmp_path / "i.json"),
            ]
        )


def test_final_requires_interim_result_and_interim_refuses_it(tmp_path: Path) -> None:
    base = ["--work", str(tmp_path), "--snapshot", str(tmp_path)]
    base += ["--authorization-url", "u", "--output-dir", str(tmp_path / "o")]
    with pytest.raises(SystemExit, match="requires --interim-result"):
        fwd.main(["run", "--stage", "FINAL", *base])
    with pytest.raises(SystemExit, match="must not pass it"):
        fwd.main(["run", "--stage", "INTERIM_A", *base, "--interim-result", "x"])


def test_build_requires_gate_arguments() -> None:
    with pytest.raises(SystemExit):
        fwd.main(["build", "--work", "w"])


def test_cash_need_formula() -> None:
    ws, wp = np.array([0.3, -0.2]), np.array([-0.5, 0.25])
    assert np.isclose(fwd.cash_need(ws, wp), 0.3 + 0.4 * 0.75)
    assert fwd.CASH_IM_RATE == 0.4


def test_replay_cell_reports_cash_need_and_net() -> None:
    lab, _ = _fake_lab(target=0.5)
    st = _replay(lab, cap=1.0)
    assert 0.5 <= st["max_cash_need"] <= 1.0
    assert st["max_abs_net"] >= 0.5
    assert st["valid"]


def test_replay_cell_cash_need_above_equity_is_invalid_on_executed_book() -> None:
    lab, _ = _fake_lab(target=1.2)  # spot long 1.2 > equity, uncapped fake target
    st = _replay(lab, cap=None)
    assert not st["valid"]
    assert any(
        r.startswith("cash_account_need_above_equity@") for r in st["invalid_reasons"]
    )
    assert st["max_cash_need"] > 1.0


def test_replay_cell_cash_need_checks_post_return_book() -> None:
    # executed book 0.99 passes; the post-return book 0.99 * 1.05 > 1 must invalidate.
    lab, _ = _fake_lab(target=0.99)
    st = _replay(lab, cap=1.0, ret=0.05)
    assert not st["valid"]
    assert st["max_cash_need"] > 1.0 + 1e-9


def test_replay_cell_perp_gross_is_charged_at_im_rate() -> None:
    lab, log = _fake_lab(target=0.0)
    orig = lab["r17"].synthesize_r17_np

    def synth(p10: Any, k: int, *a: Any) -> Any:
        ws, wp = orig(p10, k, *a)
        wp = wp.copy()
        wp[0], wp[1] = 1.2, -1.2  # perp gross 2.4 -> need 0.96 <= 1
        return ws, wp

    lab["r17"].synthesize_r17_np = synth
    st = _replay(lab, cap=None, ret=0.0)
    assert np.isclose(st["max_cash_need"], 0.96)
    assert st["valid"]


def test_build_is_once_only(tmp_path: Path) -> None:
    (tmp_path / "build-receipt.json").write_text("{}")
    with pytest.raises(SystemExit, match="once-only"):
        fwd.main(
            ["build", "--work", str(tmp_path), "--stage", "INTERIM_A"]
            + ["--snapshot", str(tmp_path), "--authorization-url", "u"]
        )


def test_importable_pretrade_provenance(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    f = tmp_path / "pretrade.py"
    f.write_text("x = 1\n")
    monkeypatch.setattr(
        fwd.importlib.util,
        "find_spec",
        lambda name: types.SimpleNamespace(origin=str(f)),
    )
    assert fwd._importable_pretrade() == (str(f), fwd.sha256_file(f))
    monkeypatch.setattr(fwd.importlib.util, "find_spec", lambda name: None)
    assert fwd._importable_pretrade() == (None, None)

    def boom(name: str) -> Any:
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(fwd.importlib.util, "find_spec", boom)
    assert fwd._importable_pretrade() == (None, None)
