"""Attempt-2 draft, seal and launch gates. Offline: no network, no collector run."""

from __future__ import annotations

import hashlib
import json
import math
import socket
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from trade_rl.evaluation.forward_study.seal import CodeIdentity
from trade_rl.evaluation.paper import attempt2, operations
from trade_rl.evaluation.paper.attempt2 import (
    ATTEMPT_ONE_FINAL_TIP_SHA256,
    ATTEMPT_ONE_LAST_OBSERVED_AT,
    ATTEMPT_ONE_PROTOCOL_SHA256,
    Attempt2Refusal,
    attempt2_lineage,
    capacity_plan,
    launch_attempt2,
    plan_attempt2,
    seal_attempt2,
    write_draft,
)
from trade_rl.evaluation.paper.watchdog import write_heartbeat
from trade_rl.evaluation.runs import build_candidate_run_provenance

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"
NOW = datetime(2026, 10, 31, 9, 0, tzinfo=UTC)
START = NOW + timedelta(hours=2)
CLEAN = CodeIdentity("a" * 40, True)
Usage = namedtuple("Usage", "total used free")
REPO = Path(__file__).resolve().parents[2]


def clock():
    return NOW


def roomy(_path):
    return Usage(10**13, 0, 10**13)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("attempt-2 preparation must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def plan(tmp_path, **overrides):
    options = dict(
        start_at=START,
        authorization=AUTH,
        bytes_per_observation=2_000_000,
        clock=clock,
        code_identity=lambda: CLEAN,
        disk_usage=roomy,
    )
    options.update(overrides)
    return plan_attempt2(tmp_path / "attempt2", **options)


def seal(tmp_path, **overrides):
    options = dict(
        start_at=START,
        authorization=AUTH,
        bytes_per_observation=2_000_000,
        clock=clock,
        code_identity=lambda: CLEAN,
        disk_usage=roomy,
    )
    options.update(overrides)
    return seal_attempt2(tmp_path / "attempt2", **options)


def refusal_of(call, **kwargs):
    with pytest.raises(Attempt2Refusal) as error:
        call(**kwargs)
    return error.value.reasons


# ------------------------------------------------------------------ binding


def test_lineage_constants_match_the_recorded_attempt_one_facts():
    lineage = attempt2_lineage()
    assert lineage["attempt_number"] == 2
    assert lineage["predecessor"] == {
        "attempt_number": 1,
        "protocol_sha256": "843467870d85d0b085e65cf904e9d458287c14fc1e31615b9aaf77029f0669b7",
        "final_tip_sha256": "1a84746aa875ee5a21b72944a5101c2ffb5b00860c9d085a1ccb1fe780d5836b",
        "disposition": "invalidated",
        "reason_codes": ["observation_gap"],
        "last_observed_at": "2026-09-17T23:11:01.134389+00:00",
    }
    status = (REPO / "docs/research/current-status.md").read_text(encoding="utf-8")
    for fact in (
        ATTEMPT_ONE_PROTOCOL_SHA256,
        ATTEMPT_ONE_FINAL_TIP_SHA256,
        ATTEMPT_ONE_LAST_OBSERVED_AT,
    ):
        assert fact in status
    operations.validate_attempt_lineage(lineage, sealed_at=NOW)


# --------------------------------------------------------------------- plan


def test_plan_is_an_unsealed_draft_that_creates_nothing(tmp_path):
    draft = plan(tmp_path)
    assert draft["schema"] == attempt2.DRAFT_SCHEMA
    assert draft["status"] == "DRAFT_NOT_SEALED"
    assert draft["sealed"] is False and draft["launched"] is False
    assert draft["production_eligible"] is False
    assert not (tmp_path / "attempt2").exists()
    assert draft["attempt_lineage"] == attempt2_lineage()
    assert draft["research_plan"] == operations.screen_plan(
        attempt_lineage=attempt2_lineage()
    )
    assert draft["start_at"] == START.isoformat()
    assert draft["close_at"] == (START + timedelta(days=90)).isoformat()
    assert (
        draft["terminal_deadline"]
        == (START + timedelta(days=90, seconds=180)).isoformat()
    )
    provenance = build_candidate_run_provenance()
    identity = draft["source_runtime_identity"]
    assert identity["implementation_digest"] == provenance["implementation_digest"]
    assert (
        identity["runtime_environment_digest"]
        == provenance["runtime_environment_digest"]
    )
    assert identity["git_head"] == "a" * 40 and identity["tracked_tree_clean"]
    assert draft["authorization"] == AUTH
    assert draft["seal_blockers"] == []
    assert draft["capacity"]["status"] == "ok"
    steps = {step["step"]: step for step in draft["steps"]}
    assert steps["launch"]["network"] is True
    assert steps["seal"]["network"] is False
    assert "--allow-network" in steps["launch"]["command"]
    json.dumps(draft, allow_nan=False)


def test_plan_reports_blockers_that_seal_would_refuse(tmp_path):
    draft = plan(
        tmp_path,
        bytes_per_observation=None,
        code_identity=lambda: CodeIdentity("a" * 40, False),
    )
    assert draft["seal_blockers"] == ["checkout_dirty", "capacity_not_assessed"]
    (tmp_path / "attempt2").mkdir()
    assert "root_exists" in plan(tmp_path)["seal_blockers"]


def test_plan_refuses_missing_or_malformed_authorization(tmp_path):
    assert refusal_of(plan, tmp_path=tmp_path, authorization=None) == [
        "authorization_missing"
    ]
    assert refusal_of(plan, tmp_path=tmp_path, authorization="") == [
        "authorization_missing"
    ]
    for bad in (
        "https://github.com/shuntatsu/trade_rl/pull/1050#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/811#issuecomment-1",
        "https://github.com/other/trade_rl/issues/810#issuecomment-1",
        "http://github.com/shuntatsu/trade_rl/issues/810#issuecomment-1",
        "https://github.com/shuntatsu/trade_rl/issues/810",
        "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-0",
        AUTH + "/extra",
    ):
        assert refusal_of(plan, tmp_path=tmp_path, authorization=bad) == [
            "authorization_malformed"
        ]


@pytest.mark.parametrize(
    ("start", "reason"),
    [
        (NOW - timedelta(days=1), "start_not_in_future"),
        (NOW, "start_not_in_future"),
        (NOW + timedelta(minutes=4, seconds=59), "start_notice_too_short"),
        (datetime(2026, 11, 1, 0, 0), "start_not_timezone_aware"),
    ],
)
def test_plan_and_seal_refuse_a_start_that_is_not_safely_in_the_future(
    tmp_path, start, reason
):
    assert refusal_of(plan, tmp_path=tmp_path, start_at=start) == [reason]
    assert refusal_of(seal, tmp_path=tmp_path, start_at=start) == [reason]
    assert not (tmp_path / "attempt2").exists()


def test_exactly_five_minutes_notice_is_accepted(tmp_path):
    assert plan(tmp_path, start_at=NOW + timedelta(minutes=5))["sealed"] is False


def test_plan_collects_every_refusal_reason(tmp_path):
    reasons = refusal_of(
        plan, tmp_path=tmp_path, authorization=None, start_at=NOW - timedelta(hours=1)
    )
    assert reasons == ["authorization_missing", "start_not_in_future"]


def test_lineage_gap_rule_is_reused_when_attempt_one_is_too_recent(tmp_path):
    just_after = datetime(2026, 9, 17, 23, 11, 2, tzinfo=UTC)
    reasons = refusal_of(
        plan,
        tmp_path=tmp_path,
        clock=lambda: just_after,
        start_at=just_after + timedelta(hours=1),
    )
    assert len(reasons) == 1 and reasons[0].startswith("lineage_invalid:")
    assert "180 seconds" in reasons[0]


def test_plan_output_is_write_once(tmp_path):
    draft = plan(tmp_path)
    output = tmp_path / "draft.json"
    write_draft(draft, output)
    stored = json.loads(output.read_bytes())
    assert stored["status"] == "DRAFT_NOT_SEALED"
    with pytest.raises(FileExistsError):
        write_draft(draft, output)


# ----------------------------------------------------------------- capacity


@pytest.mark.parametrize(
    ("rate", "free", "status"),
    [
        (None, 10**12, "not_assessed"),
        (float("nan"), 10**12, "invalid_rate"),
        (-1.0, 10**12, "invalid_rate"),
        (0.0, 10**12, "invalid_rate"),
        (True, 10**12, "invalid_rate"),
        (1_000_000.0, 10**9, "insufficient"),
        (1_000_000.0, 10**12, "ok"),
    ],
)
def test_capacity_requires_ninety_days_of_measured_storage(
    tmp_path, rate, free, status
):
    result = capacity_plan(
        tmp_path / "missing" / "root",
        bytes_per_observation=rate,
        disk_usage=lambda _p: Usage(0, 0, free),
    )
    assert result["status"] == status
    assert result["minute_slots"] == 90 * 24 * 60 + 4
    assert result["location"] == str(tmp_path)  # nearest existing parent
    if status in {"ok", "insufficient"}:
        assert result["required_bytes"] == math.ceil(129_604 * 1_000_000 * 1.25)


@pytest.mark.parametrize(
    ("rate", "free", "reason"),
    [
        (None, 10**12, "capacity_not_assessed"),
        (2_000_000.0, 10**9, "capacity_insufficient"),
        (float("inf"), 10**12, "capacity_invalid_rate"),
    ],
)
def test_seal_refuses_without_adequate_storage_evidence(tmp_path, rate, free, reason):
    assert refusal_of(
        seal,
        tmp_path=tmp_path,
        bytes_per_observation=rate,
        disk_usage=lambda _p: Usage(0, 0, free),
    ) == [reason]
    assert not (tmp_path / "attempt2").exists()


# --------------------------------------------------------------------- seal


def test_seal_refusals_never_create_a_root_or_call_the_seal(tmp_path):
    def forbidden(*_a, **_k):
        raise AssertionError("seal must not be reached")

    base = dict(tmp_path=tmp_path, seal=forbidden)
    assert refusal_of(seal, authorization=None, **base) == ["authorization_missing"]
    assert refusal_of(
        seal, code_identity=lambda: CodeIdentity("a" * 40, False), **base
    ) == ["checkout_dirty"]
    assert refusal_of(
        seal, code_identity=lambda: CodeIdentity(None, False), **base
    ) == ["checkout_identity_unavailable"]
    (tmp_path / "attempt2").mkdir()
    assert refusal_of(seal, **base) == ["root_exists"]


def test_seal_writes_a_new_root_bound_to_attempt_one_and_current_identity(tmp_path):
    result = seal(tmp_path)
    root = tmp_path / "attempt2"
    raw = (root / "protocol.json").read_bytes()
    assert result["protocol_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["sealed"] is True and result["launched"] is False
    protocol = json.loads(raw)["protocol"]
    assert protocol["settings"]["start_at"] == START.isoformat()
    plan_ = protocol["study"]["research_plan"]
    assert plan_["attempt_lineage"] == attempt2_lineage()
    assert plan_["schema"] == "carry_paper_screen_v2"
    assert protocol["study"]["provenance"] == build_candidate_run_provenance()
    assert protocol["study"]["created_at"] == NOW.isoformat()
    assert result["production_eligible"] is False
    # Nothing was collected: no sources, control record or failure marker.
    assert not (root / "sources").exists()
    assert not (root / "collection-failure.json").exists()
    assert not (root / "collector-heartbeat.json").exists()
    # A second seal into the same root is refused, so attempt 2 is never rewritten.
    assert refusal_of(seal, tmp_path=tmp_path) == ["root_exists"]


def test_seal_start_check_uses_the_clock_read_at_seal_time(tmp_path):
    ticks = iter([NOW, NOW + timedelta(hours=3)])
    # The first read (pre-check) accepts; the sealer's own later read is stricter.
    with pytest.raises(ValueError, match="five minutes"):
        seal(tmp_path, clock=lambda: next(ticks))
    assert not (tmp_path / "attempt2" / "protocol.json").exists()


# ------------------------------------------------------------------- launch


class Recorder:
    def __init__(self):
        self.calls = []

    def launcher(self, root, digest, log):
        self.calls.append(("launcher", Path(root), digest, Path(log)))
        return lambda: pytest.fail("the child must not be started by tests")

    def supervise(self, root, **kwargs):
        self.calls.append(("supervise", Path(root), kwargs["protocol_sha256"]))
        return dict(outcome="finished", exit_code=0)


def launch(tmp_path, digest, recorder=None, **overrides):
    recorder = recorder or Recorder()
    options = dict(
        protocol_sha256=digest,
        start_at=START,
        authorization=AUTH,
        allow_network=True,
        log_file=tmp_path / "logs" / "child.log",
        clock=clock,
        code_identity=lambda: CLEAN,
        launcher=recorder.launcher,
        supervise=recorder.supervise,
    )
    options.update(overrides)
    return launch_attempt2(tmp_path / "attempt2", **options), recorder


def test_launch_hands_the_sealed_root_to_the_supervisor_once(tmp_path):
    digest = seal(tmp_path)["protocol_sha256"]
    outcome, recorder = launch(tmp_path, digest)
    assert outcome == dict(outcome="finished", exit_code=0)
    kinds = [call[0] for call in recorder.calls]
    assert kinds == ["launcher", "supervise"]
    assert recorder.calls[0][2] == digest == recorder.calls[1][2]


def test_supervisor_relaunch_after_start_needs_a_verified_root_that_already_ran(
    tmp_path,
):
    digest = seal(tmp_path)["protocol_sha256"]
    root = tmp_path / "attempt2"
    after_start = START + timedelta(days=3)
    recorder = Recorder()
    assert refusal_of(
        lambda **kw: launch(tmp_path, digest, recorder, **kw),
        clock=lambda: after_start,
    ) == ["start_not_in_future"]
    assert recorder.calls == []
    write_heartbeat(
        root,
        protocol_sha256=digest,
        state="cycle_completed",
        pid=1,
        at=after_start,
        last_command_at=None,
        cycles=1,
    )
    outcome, recorder = launch(tmp_path, digest, clock=lambda: after_start)
    assert outcome["exit_code"] == 0
    assert [call[0] for call in recorder.calls] == ["launcher", "supervise"]
    # A heartbeat belonging to another protocol never unlocks a post-start launch.
    (root / "collector-heartbeat.json").write_bytes(b"torn")
    assert refusal_of(
        lambda **kw: launch(tmp_path, digest, Recorder(), **kw),
        clock=lambda: after_start,
    ) == ["start_not_in_future"]


def test_launch_requires_the_explicit_network_flag(tmp_path):
    digest = seal(tmp_path)["protocol_sha256"]
    recorder = Recorder()
    assert refusal_of(
        launch_attempt2,
        root=tmp_path / "attempt2",
        protocol_sha256=digest,
        start_at=START,
        authorization=AUTH,
        allow_network=False,
        log_file=tmp_path / "child.log",
        clock=clock,
        code_identity=lambda: CLEAN,
        launcher=recorder.launcher,
        supervise=recorder.supervise,
    ) == ["network_not_allowed"]
    assert recorder.calls == []


def test_launch_refusals_do_not_reach_the_supervisor(tmp_path, monkeypatch):
    digest = seal(tmp_path)["protocol_sha256"]
    cases = {
        "authorization_missing": dict(authorization=None),
        "authorization_malformed": dict(authorization="https://example.com/x"),
        "start_not_in_future": dict(start_at=NOW),
        "checkout_dirty": dict(code_identity=lambda: CodeIdentity("a" * 40, False)),
        "protocol_digest_mismatch": dict(protocol_sha256="e" * 64),
        "protocol_start_differs": dict(start_at=START + timedelta(hours=1)),
    }
    for reason, overrides in cases.items():
        recorder = Recorder()
        reasons = refusal_of(
            lambda **kw: launch(tmp_path, digest, recorder, **kw), **overrides
        )
        assert reason in reasons, (reason, reasons)
        assert recorder.calls == []
    monkeypatch.setattr(
        attempt2, "build_candidate_run_provenance", lambda: {"source": "changed"}
    )
    recorder = Recorder()
    assert refusal_of(lambda **kw: launch(tmp_path, digest, recorder, **kw)) == [
        "protocol_provenance_changed"
    ]
    assert recorder.calls == []


def test_launch_refuses_a_root_that_is_not_the_attempt_two_lineage(tmp_path):
    root = tmp_path / "attempt2"
    first_attempt = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    digest = operations.seal_paper_study(
        root, start_at=START, attempt_lineage=first_attempt, clock=clock
    )
    recorder = Recorder()
    assert refusal_of(lambda **kw: launch(tmp_path, digest, recorder, **kw)) == [
        "protocol_not_attempt2_lineage"
    ]
    assert recorder.calls == []


def test_launch_with_a_missing_root_is_refused_not_created(tmp_path):
    recorder = Recorder()
    assert refusal_of(lambda **kw: launch(tmp_path, "f" * 64, recorder, **kw)) == [
        "protocol_not_a_regular_file"
    ]
    assert not (tmp_path / "attempt2").exists()


# ---------------------------------------------------------------------- CLI


def test_cli_refuses_without_authorization_before_any_side_effect(tmp_path, capsys):
    root = tmp_path / "attempt2"
    start = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    for command in ("plan", "seal"):
        code = attempt2.main([command, "--root", str(root), "--start-at", start])
        assert code == 2
        assert "authorization_missing" in json.loads(capsys.readouterr().err)["refused"]
    code = attempt2.main(
        [
            "launch",
            "--root",
            str(root),
            "--start-at",
            start,
            "--protocol-sha256",
            "a" * 64,
            "--log-file",
            str(tmp_path / "x.log"),
        ]
    )
    refused = json.loads(capsys.readouterr().err)["refused"]
    assert code == 2
    assert {"authorization_missing", "network_not_allowed"} <= set(refused)
    assert not root.exists()


def test_cli_plan_prints_and_writes_only_the_requested_draft(tmp_path, capsys):
    start = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    output = tmp_path / "draft.json"
    args = [
        "plan",
        "--root",
        str(tmp_path / "attempt2"),
        "--start-at",
        start,
        "--authorization",
        AUTH,
        "--output",
        str(output),
    ]
    assert attempt2.main(args) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["status"] == "DRAFT_NOT_SEALED" and output.is_file()
    assert not (tmp_path / "attempt2").exists()
    assert attempt2.main(args) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"
