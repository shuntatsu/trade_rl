"""Follow-ups of the report-only pre-review: held legs, harness digest, authorization scope.

Synthetic data only; nothing reads market data, the network or an economic result.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from tests.evaluation.test_agy_anchor_oos import (
    assemble,
    small_window,
)
from tests.evaluation.test_agy_anchor_oos_core import _Fixed, _w
from tests.evaluation.test_agy_anchor_oos_tool import (
    DATA_AUTH,
    NOW,
    Gh,
    NoRun,
    Rows,
    dev_result,
    listings,
    run_argv,
    tool,
)
from tests.evaluation.test_lowturn_screen import _snapshot
from tools import agy_anchor_oos as driver
from tools import core7_g4_comment as g4
from trade_rl.evaluation.agy_anchor_oos import cash, data, protocol, runner

N_LEGS = 22


@pytest.fixture(autouse=True)
def _no_real_halts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(protocol, "DECLARED_HALTS", ())
    monkeypatch.setattr(protocol, "COVERAGE_EVIDENCE_SHA256", None)


# ---------------------------------------------------------- B1: held legs and the net bound


def test_held_legs_count_at_their_current_weight_in_the_net_bound() -> None:
    """Legs held while invalid or halted keep their current weight; the net bound is on the
    effective vector, so they push the net up and the whole vector is scaled (holds dropped)."""
    current = _w(s0=0.55, s1=0.55)  # held: net already 1.1 > 0.97 on its own
    proposal = _Fixed(_w(s0=0.55, s1=0.55, p2=0.3), holds=(0, 2))
    snap = _snapshot(0, weights=current)
    policy = cash.AnchorCashAccountPolicy(proposal, start_index=0)
    capped = np.asarray(policy.decide_portfolio(snap).target_weights)
    effective = np.where(np.isin(np.arange(N_LEGS), (0, 2)), current, proposal.weights)
    assert cash.net_exposure(effective) == pytest.approx(1.4)
    assert cash.net_exposure(capped) == pytest.approx(protocol.NET_EXPOSURE_LIMIT)
    assert policy.stats.bound_by_net_exposure == 1
    # holds are dropped: every leg is re-targeted
    result = policy.decide_portfolio(_snapshot(18, weights=current))
    assert result.hold_current_quantity_symbols == ()


def test_a_held_leg_alone_above_the_bound_is_rescaled_with_the_rest() -> None:
    current = _w(s0=0.55, s1=0.55, s2=0.2)  # net 1.3 held entirely (a halted book)
    hold_all = _Fixed(current, holds=tuple(range(N_LEGS)))
    policy = cash.AnchorCashAccountPolicy(hold_all, start_index=0)
    untouched = policy.decide_portfolio(_snapshot(0, weights=current))
    # a pure hold never trades, so it is never capped (a trade-free proposal cannot change cash)
    assert untouched.hold_current_quantity_symbols == tuple(range(N_LEGS))
    assert policy.stats.bound_decisions == 0
    # as soon as the proposal trades one leg, the held net counts and the book is scaled
    mixed = _Fixed(_w(s0=0.55, s1=0.55, s2=0.2, p3=0.1), holds=(0, 2, 4))
    policy = cash.AnchorCashAccountPolicy(mixed, start_index=0)
    out = np.asarray(
        policy.decide_portfolio(_snapshot(0, weights=current)).target_weights
    )
    assert cash.net_exposure(out) == pytest.approx(protocol.NET_EXPOSURE_LIMIT)
    assert policy.stats.bound_by_net_exposure == 1


# ------------------------------------------------------- controls share the cash account, N2


def test_every_arm_uses_the_same_cash_account_and_the_anchor_strides_every_18_bars(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    anchor = runner.anchor_signals_of(stage)
    control = runner.build_oos_signals(stage.dataset)
    start = window.first_decision_index
    for arm in protocol.ARMS:
        policy = runner.make_policy(
            arm,
            anchor=anchor,
            control=control,
            start_index=start,
            lag=0 if arm != "AGY_ANCHOR" else 1,
        )
        assert isinstance(policy, cash.AnchorCashAccountPolicy), arm
    candidate = runner.make_policy(
        protocol.CANDIDATE, anchor=anchor, control=control, start_index=start, lag=1
    )
    assert candidate._inner._every == protocol.REBALANCE_EVERY_BARS == 18  # type: ignore[attr-defined]
    assert candidate._every == 18  # type: ignore[attr-defined]
    assert candidate._inner._deadband == protocol.DEADBAND  # type: ignore[attr-defined]
    ew = runner.make_policy(
        protocol.CONTROL_EW_SPOT,
        anchor=anchor,
        control=control,
        start_index=start,
        lag=0,
    )
    assert ew._every == runner.CONTROL_REBALANCE_EVERY_BARS  # type: ignore[attr-defined]


# ------------------------------------------------------------- B2: the harness digest


def _copy_harness(tmp_path: Path) -> tuple[Path, Path]:
    package = Path(protocol.__file__).resolve().parent
    root = tmp_path / "pkg"
    root.mkdir()
    for name in protocol.HARNESS_PACKAGE_FILES:
        (root / name).write_bytes((package / name).read_bytes())
    tool_copy = tmp_path / "agy_anchor_oos.py"
    tool_copy.write_bytes(
        (package.parents[2] / protocol.HARNESS_TOOL_RELATIVE_PATH).read_bytes()
    )
    return root, tool_copy


def test_the_harness_digest_covers_every_harness_file_and_ignores_line_endings(
    tmp_path: Path,
) -> None:
    assert protocol.HARNESS_PACKAGE_FILES == (
        "protocol.py",
        "cash.py",
        "decision.py",
        "data.py",
        "runner.py",
    )
    real = protocol.harness_source_digest()
    root, tool_copy = _copy_harness(tmp_path)
    assert protocol.harness_source_digest(package_dir=root, tool_path=tool_copy) == real
    for path in (*(root / n for n in protocol.HARNESS_PACKAGE_FILES), tool_copy):
        original = path.read_bytes()
        crlf = original.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
        path.write_bytes(crlf)
        assert (
            protocol.harness_source_digest(package_dir=root, tool_path=tool_copy)
            == real
        )
        path.write_bytes(original + b"# changed\n")
        changed = protocol.harness_source_digest(package_dir=root, tool_path=tool_copy)
        assert changed != real, path.name
        path.write_bytes(original)
    (root / "cash.py").unlink()
    with pytest.raises(protocol.ProtocolBindingError, match="missing"):
        protocol.harness_source_digest(package_dir=root, tool_path=tool_copy)
    tool_copy.unlink()
    (root / "cash.py").write_bytes(
        (Path(protocol.__file__).resolve().parent / "cash.py").read_bytes()
    )
    with pytest.raises(protocol.ProtocolBindingError, match="missing"):
        protocol.harness_source_digest(package_dir=root, tool_path=tool_copy)


def test_the_protocol_digest_changes_with_the_harness_and_verifies_the_fixture(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    digest = protocol.protocol_digest()
    payload = protocol.protocol_payload()
    harness = payload["harness"]
    assert isinstance(harness, dict)
    assert harness["harness_source_digest"] == protocol.harness_source_digest()
    assert harness["files"][-1] == "tools/agy_anchor_oos.py"
    monkeypatch.setattr(protocol, "harness_source_digest", lambda **_: "f" * 64)
    assert protocol.protocol_digest() != digest  # any harness edit is a new digest
    monkeypatch.undo()
    # the fixture binding is verified inside protocol_digest (N7)
    monkeypatch.setattr(protocol, "fixture_path", lambda: tmp_path / "absent.npz")
    with pytest.raises(protocol.ProtocolBindingError, match="fixture"):
        protocol.protocol_digest()
    other = tmp_path / "fixture.npz"
    other.write_bytes(b"not the fixture")
    monkeypatch.setattr(protocol, "fixture_path", lambda: other)
    with pytest.raises(protocol.ProtocolBindingError, match="sha256"):
        protocol.protocol_digest()


def test_the_receipts_and_results_record_the_harness_digest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    stage = assemble(window)
    result = runner.run_stage(
        stage,
        window=window,
        output_dir=tmp_path / "o",
        expected_protocol_digest=protocol.protocol_digest(),
        run_date=datetime(2026, 11, 20, tzinfo=UTC).date(),
    )
    assert result["harness_source_digest"] == protocol.harness_source_digest()


# ------------------------------------------------------------------ B3: authorization scope


def test_the_owner_scope_relay_is_the_frozen_data_authorization() -> None:
    assert protocol.DATA_AUTHORIZATION_URL == (
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7040247247"
    )
    assert protocol.AUTHORIZED_STAGES == ("BACKCAST_2020", "FORWARD")
    assert driver._auth_url("DEV") is None
    assert driver._auth_url("BACKCAST_2020") == protocol.DATA_AUTHORIZATION_URL
    assert driver._auth_url("FORWARD") == protocol.DATA_AUTHORIZATION_URL
    payload = protocol.protocol_payload()
    assert payload["data_authorization"]["stages"] == ["BACKCAST_2020", "FORWARD"]  # type: ignore[index]
    spec = (
        Path(__file__).resolve().parents[2] / "docs/specs/agy-anchor-realistic.md"
    ).read_text("utf-8")
    assert "7040247247" in spec and "Amendment 5" in spec


def test_backcast_and_forward_reads_need_the_relay_and_dev_does_not(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dev = small_window(monkeypatch, protocol.STAGE_DEV)
    assert data.read_limit(dev, now=NOW) == dev.last_row_close  # known-used data
    back = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    with pytest.raises(
        data.StageEmbargoError, match="BACKCAST_2020 reads need the owner"
    ):
        data.read_limit(back, now=NOW)
    other = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7028416072"
    with pytest.raises(data.StageEmbargoError, match="owner authorization"):
        data.read_limit(
            back, now=NOW, authorization_url=other
        )  # the older relay is not it
    assert (
        data.read_limit(
            back, now=NOW, authorization_url=protocol.DATA_AUTHORIZATION_URL
        )
        == back.last_row_close
    )
    with pytest.raises(
        data.StageEmbargoError, match="BACKCAST_2020 reads need the owner"
    ):
        assemble(back, authorization_url=None)
    stage = assemble(back)
    assert protocol.DATA_AUTHORIZATION_URL in (
        stage.dataset.identity_payload_json or ""
    )
    assert data.validate_stage_dataset(stage, back) == back.last_row_index


def test_the_forward_embargo_policy_is_bound_to_this_relay_only() -> None:
    limit = datetime(2026, 11, 1, tzinfo=UTC)
    policy = driver.embargo_for(limit, stage="FORWARD")
    assert type(policy).__name__ == "AnchorForwardEmbargoPolicy"
    assert policy.warmup_authorization_url == protocol.DATA_AUTHORIZATION_URL
    with pytest.raises(ValueError, match="owner authorization"):
        driver.AnchorForwardEmbargoPolicy(
            warmup_authorization_url=(
                "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-7028416072"
            ),
            admit_before=limit,
        )
    with pytest.raises(data.StageEmbargoError):
        driver.embargo_for(limit, stage="BACKCAST_2020")
    assert type(
        driver.embargo_for(datetime(2021, 1, 1, 4, tzinfo=UTC), stage="BACKCAST_2020")
    ).__name__ == ("EmbargoPolicy")


def test_a_backcast_run_binds_the_relay_owner_authored_unedited_and_records_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    monkeypatch.setattr(driver.runner, "run_stage", NoRun())
    rows = Rows(listings(window))
    dev = str(dev_result(tmp_path))
    argv = run_argv(tmp_path, "BACKCAST_2020", "--dev-result", dev)
    gh = Gh()
    code, _, err = tool(tmp_path, argv, gh=gh, now=NOW, rows=rows)
    assert code == driver.EXIT_OK, err
    receipt = json.loads(
        (tmp_path / "out" / driver.DATA_RECEIPT_NAME).read_text("utf-8")
    )
    auth = receipt["data_authorization"]
    assert auth["url"] == DATA_AUTH and auth["author_login"] == g4.OWNER_LOGIN
    assert auth["embargo_policy"] == "EmbargoPolicy"
    assert receipt["harness_source_digest"] == protocol.harness_source_digest()
    fetched = [call for call in gh.calls if str(call[1]).endswith("/7040247247")]
    assert len(fetched) == 2  # first fetch and the unedited re-check
    # an edit of the relay between the two fetches authorizes nothing
    edited = Gh()
    edited.data.variants = {2: {"updated_at": "2026-11-21T00:00:00Z"}}
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "BACKCAST_2020", "--dev-result", dev, name="o2"),
        gh=edited,
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_UNAUTHENTICATED and "edited" in err
    assert not (tmp_path / "o2").exists()
    # a relay that is not owner-authored, or cannot be fetched, authorizes nothing
    broken = Gh()
    broken.data.fail = True
    code, _, _ = tool(
        tmp_path,
        run_argv(tmp_path, "BACKCAST_2020", "--dev-result", dev, name="o3"),
        gh=broken,
        now=NOW,
        rows=rows,
    )
    assert code == driver.EXIT_UNAUTHENTICATED
    assert not (tmp_path / "o3").exists()
    # DEV reads known-used data: the relay is not fetched at all
    gh_dev = Gh()
    code, _, err = tool(
        tmp_path,
        run_argv(tmp_path, "DEV", name="d"),
        gh=gh_dev,
        now=NOW,
        rows=Rows(listings(data.frozen_window(protocol.STAGE_DEV))),
    )
    assert code == driver.EXIT_OK, err
    assert not [c for c in gh_dev.calls if str(c[1]).endswith("/7040247247")]
    assert gh_dev.calls  # only the per-run G4 comment was fetched


def test_backcast_coverage_records_the_relay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    rows = Rows(listings(window))
    code, _, err = tool(
        tmp_path,
        ["coverage", "--stage", "BACKCAST_2020", "--output-dir", str(tmp_path / "cov")],
        gh=Gh(),
        now=NOW,
        rows=rows,
        with_rules=False,
    )
    assert code == driver.EXIT_OK, err
    document = json.loads((tmp_path / "cov" / driver.COVERAGE_NAME).read_text("utf-8"))
    assert document["data_authorization"]["url"] == DATA_AUTH
    bad = Gh()
    bad.data.fail = True
    code, _, _ = tool(
        tmp_path,
        [
            "coverage",
            "--stage",
            "BACKCAST_2020",
            "--output-dir",
            str(tmp_path / "cov2"),
        ],
        gh=bad,
        now=NOW,
        rows=Rows(listings(window)),
        with_rules=False,
    )
    assert code == driver.EXIT_UNAUTHENTICATED and not (tmp_path / "cov2").exists()


# ------------------------------------------------------------------ N4: DEV may stop the line


def test_backcast_is_refused_when_the_dev_result_has_an_invalid_cell(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = small_window(monkeypatch, protocol.STAGE_BACKCAST)
    small_window(monkeypatch, protocol.STAGE_DEV)
    fake = NoRun()
    monkeypatch.setattr(driver.runner, "run_stage", fake)
    rows = Rows(listings(window))
    for over, text in (
        ({"invalid_cells": ["AGY_ANCHOR__BASE"]}, "invalid cells"),
        ({"invalid_cells": None}, "no invalid_cells list"),
    ):
        code, _, err = tool(
            tmp_path,
            run_argv(
                tmp_path,
                "BACKCAST_2020",
                "--dev-result",
                str(dev_result(tmp_path, **over)),
            ),
            gh=Gh(),
            now=NOW,
            rows=rows,
        )
        assert code == driver.EXIT_REFUSED and text in err, err
        assert "stop the line" in err or "invalid_cells" in err
    assert not fake.calls and not (tmp_path / "out").exists()


def test_the_spec_records_amendment_5_with_every_review_item() -> None:
    spec = (
        Path(__file__).resolve().parents[2] / "docs/specs/agy-anchor-realistic.md"
    ).read_text("utf-8")
    for phrase in (
        "Amendment 5",
        "harness_source_digest",
        "Controls share the cash account",
        "DEV may stop the line",
        "Activation transient",
        "Cold EMA seed residual",
        "FORWARD warm-up is known-used data",
        "0.5^(180/126) = 37%",
    ):
        assert phrase in spec, phrase
    assert hashlib.sha256(spec.encode()).hexdigest()  # the spec is read as text
