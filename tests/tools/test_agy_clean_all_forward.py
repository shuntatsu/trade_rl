"""Tests for the agy CLEAN_ALL forward harness (no lab, no network, no data)."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tools import agy_clean_all_forward as fwd

PINNED_DIGEST = "9d55aade760c93084b6420ca9a183885e2aa892ab60ce30a49590bee532c67d5"


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


def _comment(
    body: str, login: str = "shuntatsu", updated: str = "2026-11-22T00:00:00Z"
) -> dict[str, Any]:
    return {"user": {"login": login}, "body": body, "updated_at": updated}


def test_authorization_receipt_binds_owner_digest_and_scope() -> None:
    digest = fwd.protocol_digest()
    url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-123"
    comments = {
        123: _comment(f"relay {fwd.PROTOCOL_ID} {digest}"),
        fwd.DATA_SCOPE_COMMENT_ID: _comment("scope"),
    }
    rec = fwd.authorization_receipt(url, digest, fetch=lambda cid: comments[cid])
    assert rec["g4_relay"]["comment_id"] == 123
    assert rec["data_scope"]["comment_id"] == fwd.DATA_SCOPE_COMMENT_ID
    comments[123] = _comment(f"relay {fwd.PROTOCOL_ID} other-digest")
    with pytest.raises(SystemExit, match="does not quote"):
        fwd.authorization_receipt(url, digest, fetch=lambda cid: comments[cid])
    comments[123] = _comment(f"relay {fwd.PROTOCOL_ID} {digest}", login="someone")
    with pytest.raises(SystemExit, match="not owner"):
        fwd.authorization_receipt(url, digest, fetch=lambda cid: comments[cid])


def test_authorization_receipt_refuses_edit_between_fetches() -> None:
    digest = fwd.protocol_digest()
    seq = iter(
        [
            _comment(f"{fwd.PROTOCOL_ID} {digest}"),
            _comment(f"{fwd.PROTOCOL_ID} {digest} edited"),
        ]
    )
    url = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-5"
    with pytest.raises(SystemExit, match="changed between fetches"):
        fwd.authorization_receipt(url, digest, fetch=lambda cid: next(seq))


def test_authorization_url_must_be_issue_810() -> None:
    with pytest.raises(SystemExit):
        fwd.comment_id_from_url(
            "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1"
        )


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
            ]
        )
