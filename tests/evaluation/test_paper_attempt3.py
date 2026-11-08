"""Attempt-3 draft, seal and launch gates. Offline: no network, no collector run.

The predecessor (attempt 2) is represented by a fake v1 reader and an empty tmp
directory holding a stand-in failure marker, so no real root, price or timing figure is
read. The oracle is the spec: lineage v4 from a verified, invalid predecessor root,
screen v3 sealed through the private attempt-3 token, a new root outside the
predecessor, and the attempt-2 recipe's other gates unchanged.
"""

from __future__ import annotations

import hashlib
import json
import socket
from collections import namedtuple
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from trade_rl.evaluation.forward_study.seal import CodeIdentity
from trade_rl.evaluation.paper import attempt3, operations
from trade_rl.evaluation.paper import predecessor as pred
from trade_rl.evaluation.paper.attempt3 import (
    Attempt3Refusal,
    launch_attempt3,
    plan_attempt3,
    resolve_lineage,
    seal_attempt3,
)
from trade_rl.evaluation.paper.operations import screen_plan_v3
from trade_rl.evaluation.paper.watchdog import write_heartbeat
from trade_rl.evaluation.runs import build_candidate_run_provenance

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"
NOW = datetime(2026, 11, 20, 9, 0, tzinfo=UTC)
START = NOW + timedelta(hours=2)
CLEAN = CodeIdentity("a" * 40, True)
Usage = namedtuple("Usage", "total used free")
DIGEST_A2 = hashlib.sha256(b"synthetic attempt-2 protocol").hexdigest()
TIP_A2 = hashlib.sha256(b"synthetic attempt-2 tip").hexdigest()
OBSERVED_A2 = "2026-11-07T00:59:00.123456+00:00"


def clock() -> datetime:
    return NOW


def roomy(_path: Any) -> Usage:
    return Usage(10**13, 0, 10**13)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("attempt-3 preparation must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


def fake_reader(
    events: int = 121, tip: str = TIP_A2, observed: str | None = OBSERVED_A2
):
    calls: list[tuple[Path, str, bool]] = []

    def read(path: Any, *, expected_protocol_sha256: str, read_only: bool) -> Any:
        calls.append((Path(path), expected_protocol_sha256, read_only))
        return {"events": events, "tip": tip, "status": {"observed_at": observed}}

    read.calls = calls  # type: ignore[attr-defined]
    return read


@pytest.fixture
def predecessor(tmp_path):
    root = tmp_path / "attempt2-root"
    root.mkdir()
    (root / "collection-failure.json").write_text("{}")
    return root


def options(tmp_path: Path, predecessor: Path, **overrides: Any) -> dict[str, Any]:
    values: dict[str, Any] = dict(
        start_at=START,
        authorization=AUTH,
        predecessor_root=predecessor,
        predecessor_protocol_sha256=DIGEST_A2,
        clock=clock,
        code_identity=lambda: CLEAN,
        disk_usage=roomy,
        read_predecessor=fake_reader(),
    )
    values.update(overrides)
    return values


def plan(tmp_path, predecessor, **overrides):
    return plan_attempt3(
        tmp_path / "attempt3",
        bytes_per_observation=2_000_000,
        **options(tmp_path, predecessor, **overrides),
    )


def seal(tmp_path, predecessor, **overrides):
    values = options(tmp_path, predecessor, **overrides)
    return seal_attempt3(
        tmp_path / "attempt3", bytes_per_observation=2_000_000, **values
    )


def refusal_of(call, **kwargs):
    with pytest.raises(Attempt3Refusal) as error:
        call(**kwargs)
    return error.value.reasons


# ----------------------------------------------------------------- lineage


def test_resolve_lineage_reads_only_the_identity_fields_from_the_preserved_root(
    predecessor,
):
    read = fake_reader()

    lineage, reasons = resolve_lineage(predecessor, DIGEST_A2, read=read)

    assert reasons == []
    assert lineage == pred.canonical_lineage_v4(
        protocol_sha256=DIGEST_A2,
        final_tip_sha256=TIP_A2,
        last_observed_at=OBSERVED_A2,
        event_count=121,
        root_path=str(predecessor.absolute()),
    )
    # Read twice (build, then the collector-entry gate), always read-only with the
    # digest the operator named.
    assert read.calls == [(predecessor.absolute(),) + (DIGEST_A2, True)] * 2


@pytest.mark.parametrize(
    "kwargs,reason",
    [
        ({"predecessor_root": None}, "predecessor_root_required"),
        ({"predecessor_protocol_sha256": None}, "predecessor_protocol_sha256_required"),
        (
            {"predecessor_protocol_sha256": "A" * 64},
            "predecessor_protocol_sha256_required",
        ),
        (
            {"predecessor_protocol_sha256": "a" * 63},
            "predecessor_protocol_sha256_required",
        ),
        ({"read": fake_reader(observed=None)}, "predecessor_root_has_no_observation"),
        (
            {"read": fake_reader(observed="garbage")},
            "predecessor_root_last_observation_invalid",
        ),
        ({"read": fake_reader(events=0)}, "predecessor_root_report_invalid"),
    ],
)
def test_resolve_lineage_names_why_it_cannot_claim_a_predecessor(
    predecessor, kwargs, reason
):
    arguments: dict[str, Any] = dict(
        predecessor_root=predecessor,
        predecessor_protocol_sha256=DIGEST_A2,
        read=fake_reader(),
    )
    arguments.update(kwargs)

    lineage, reasons = resolve_lineage(
        arguments["predecessor_root"],
        arguments["predecessor_protocol_sha256"],
        read=arguments["read"],
    )

    assert lineage is None and reason in reasons


def test_an_unreadable_root_or_a_missing_directory_or_marker_is_refused(
    tmp_path, predecessor
):
    def unreadable(*_a: Any, **_k: Any) -> Any:
        raise OSError("hot journal")

    assert resolve_lineage(predecessor, DIGEST_A2, read=unreadable)[1] == [
        "predecessor_root_unreadable:OSError"
    ]
    assert resolve_lineage(tmp_path / "missing", DIGEST_A2, read=fake_reader())[1] == [
        "predecessor_root_not_a_directory"
    ]
    (predecessor / "collection-failure.json").unlink()
    reasons = resolve_lineage(predecessor, DIGEST_A2, read=fake_reader())[1]
    assert any("predecessor_root_failure_marker_missing" in item for item in reasons)


def test_the_attempt_two_predecessor_digest_cannot_be_the_attempt_one_digest(
    predecessor,
):
    lineage, reasons = resolve_lineage(
        predecessor, pred.ATTEMPT_ONE_PROTOCOL_SHA256, read=fake_reader()
    )
    assert lineage is None
    assert any("predecessor_digest_is_attempt_one" in item for item in reasons)


# -------------------------------------------------------------------- plan


def test_plan_is_an_unsealed_draft_that_creates_nothing(tmp_path, predecessor):
    draft = plan(tmp_path, predecessor)

    assert draft["schema"] == attempt3.DRAFT_SCHEMA
    assert draft["status"] == "DRAFT_NOT_SEALED"
    assert draft["sealed"] is False and draft["launched"] is False
    assert draft["production_eligible"] is False
    assert not (tmp_path / "attempt3").exists()
    assert draft["research_plan"] == screen_plan_v3(
        attempt_lineage=draft["attempt_lineage"]
    )
    assert draft["research_plan"]["schema"] == "carry_paper_screen_v3"
    assert draft["attempt_lineage"]["predecessor"]["attempt_number"] == 2
    assert draft["capacity"]["status"] == "ok"
    assert draft["seal_blockers"] == []
    steps = {step["step"]: step for step in draft["steps"]}
    assert steps["launch"]["network"] is True and steps["seal"]["network"] is False
    assert "paper.attempt3" in steps["seal"]["command"]
    assert "paper.attempt2" not in json.dumps(draft)


def test_plan_reports_blockers_that_seal_would_refuse(tmp_path, predecessor):
    draft = plan_attempt3(
        tmp_path / "attempt3",
        bytes_per_observation=None,
        **options(
            tmp_path,
            predecessor,
            code_identity=lambda: CodeIdentity("a" * 40, False),
        ),
    )
    assert set(draft["seal_blockers"]) == {"checkout_dirty", "capacity_not_assessed"}


def test_plan_and_seal_refuse_the_authorization_start_and_overlap_gates(
    tmp_path, predecessor
):
    for call in (plan, seal):
        assert refusal_of(
            call, tmp_path=tmp_path, predecessor=predecessor, authorization=None
        ) == ["authorization_missing"]
        assert "start_notice_too_short" in refusal_of(
            call,
            tmp_path=tmp_path,
            predecessor=predecessor,
            start_at=NOW + timedelta(minutes=4),
        )
        assert "start_not_in_future" in refusal_of(
            call, tmp_path=tmp_path, predecessor=predecessor, start_at=NOW
        )
    inside = predecessor / "run"
    for target in (inside, predecessor, predecessor.parent):
        reasons = refusal_of(
            plan_attempt3,
            root=target,
            bytes_per_observation=2_000_000,
            **options(tmp_path, predecessor),
        )
        assert reasons == ["root_overlaps_predecessor_root"], target
    assert not inside.exists()


def test_plan_collects_every_reason_at_once(tmp_path, predecessor):
    reasons = refusal_of(
        plan,
        tmp_path=tmp_path,
        predecessor=predecessor,
        authorization=None,
        start_at=NOW,
        predecessor_protocol_sha256=None,
    )
    assert {
        "authorization_missing",
        "start_not_in_future",
        "predecessor_protocol_sha256_required",
    } <= set(reasons)


def test_cli_plan_refuses_an_unreadable_predecessor_and_writes_nothing(
    tmp_path, predecessor, capsys
):
    start = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    output = tmp_path / "draft.json"
    args = [
        "plan",
        "--root",
        str(tmp_path / "attempt3"),
        "--start-at",
        start,
        "--authorization",
        AUTH,
        "--predecessor-root",
        str(predecessor),
        "--predecessor-protocol-sha256",
        DIGEST_A2,
        "--output",
        str(output),
    ]
    # The CLI reads the real v1 reader, which has no attempt-2 root here: refused, and
    # nothing is written.
    assert attempt3.main(args) == 2
    assert not output.exists()
    assert "predecessor_root_unreadable" in capsys.readouterr().err


# -------------------------------------------------------------------- seal


def test_seal_refusals_never_create_a_root_or_call_the_seal(tmp_path, predecessor):
    def forbidden(*_a, **_k):
        raise AssertionError("seal must not be reached")

    base = dict(tmp_path=tmp_path, predecessor=predecessor, seal=forbidden)
    assert refusal_of(seal, authorization=None, **base) == ["authorization_missing"]
    assert refusal_of(
        seal, code_identity=lambda: CodeIdentity("a" * 40, False), **base
    ) == ["checkout_dirty"]
    assert refusal_of(
        seal, code_identity=lambda: CodeIdentity(None, False), **base
    ) == ["checkout_identity_unavailable"]
    assert "capacity_insufficient" in refusal_of(
        seal, disk_usage=lambda _p: Usage(1, 1, 1), **base
    )
    (tmp_path / "attempt3").mkdir()
    assert refusal_of(seal, **base) == ["root_exists"]


def test_seal_writes_a_new_screen_v3_root_bound_to_the_attempt_two_lineage(
    tmp_path, predecessor
):
    result = seal(tmp_path, predecessor)
    root = tmp_path / "attempt3"
    raw = (root / "protocol.json").read_bytes()

    assert result["protocol_sha256"] == hashlib.sha256(raw).hexdigest()
    assert result["sealed"] is True and result["launched"] is False
    assert result["schema"] == attempt3.SEAL_SCHEMA
    protocol = json.loads(raw)["protocol"]
    plan_ = protocol["study"]["research_plan"]
    assert plan_["schema"] == "carry_paper_screen_v3"
    assert plan_["attempt_lineage"]["schema"] == "carry_paper_attempt_lineage_v4"
    assert plan_["attempt_lineage"]["predecessor"]["protocol_sha256"] == DIGEST_A2
    assert plan_["attempt_lineage"]["predecessor_seal"]["reused"] is False
    assert result["protocol_sha256"] not in {
        DIGEST_A2,
        pred.ATTEMPT_ONE_PROTOCOL_SHA256,
    }
    assert protocol["study"]["provenance"] == build_candidate_run_provenance()
    assert protocol["study"]["created_at"] == NOW.isoformat()
    assert protocol["settings"]["start_at"] == START.isoformat()
    assert not (root / "sources").exists()
    assert not (root / "collection-failure.json").exists()
    assert refusal_of(seal, tmp_path=tmp_path, predecessor=predecessor) == [
        "root_exists"
    ]


def test_seal_passes_the_private_attempt_three_token_to_the_sealer(
    tmp_path, predecessor
):
    seen: dict[str, Any] = {}

    def recording(root, **kwargs):
        seen.update(kwargs)
        return "f" * 64

    seal(tmp_path, predecessor, seal=recording)

    assert seen["lineage_seal_token"] is pred._ATTEMPT3_SEAL_TOKEN
    assert seen["attempt_lineage"]["schema"] == pred.LINEAGE_V4_SCHEMA


def test_the_generic_sealing_entry_points_refuse_lineage_v4(tmp_path, predecessor):
    lineage = resolve_lineage(predecessor, DIGEST_A2, read=fake_reader())[0]
    with pytest.raises(ValueError, match="attempt3"):
        operations.seal_paper_study(
            tmp_path / "generic", start_at=START, attempt_lineage=lineage, clock=clock
        )
    lineage_file = tmp_path / "lineage.json"
    lineage_file.write_text(json.dumps(lineage))
    from trade_rl.evaluation.paper import cli

    code = cli.main(
        [
            "seal",
            "--root",
            str(tmp_path / "generic-cli"),
            "--start-at",
            (datetime.now(UTC) + timedelta(days=1)).isoformat(),
            "--attempt-lineage-json",
            str(lineage_file),
        ]
    )
    assert code == 2
    assert not (tmp_path / "generic").exists()
    assert not (tmp_path / "generic-cli").exists()


# ------------------------------------------------------------------ launch


class Recorder:
    def __init__(self):
        self.calls = []

    def launcher(self, root, digest, log):
        self.calls.append(("launcher", Path(root), digest, Path(log)))
        return lambda: pytest.fail("the child must not be started by tests")

    def supervise(self, root, **kwargs):
        self.calls.append(("supervise", Path(root), kwargs["protocol_sha256"]))
        return dict(outcome="finished", exit_code=0)


def launch(tmp_path, predecessor, digest, recorder=None, **overrides):
    recorder = recorder or Recorder()
    values: dict[str, Any] = dict(
        protocol_sha256=digest,
        start_at=START,
        authorization=AUTH,
        allow_network=True,
        log_file=tmp_path / "logs" / "child.log",
        clock=clock,
        code_identity=lambda: CLEAN,
        launcher=recorder.launcher,
        supervise=recorder.supervise,
        read_predecessor=fake_reader(),
    )
    values.update(overrides)
    return launch_attempt3(tmp_path / "attempt3", **values), recorder


def test_launch_hands_the_sealed_root_to_the_supervisor_once(tmp_path, predecessor):
    digest = seal(tmp_path, predecessor)["protocol_sha256"]

    outcome, recorder = launch(tmp_path, predecessor, digest)

    assert outcome == dict(outcome="finished", exit_code=0)
    assert [call[0] for call in recorder.calls] == ["launcher", "supervise"]


def test_every_launch_rereads_the_predecessor_root_and_refuses_a_changed_one(
    tmp_path, predecessor
):
    digest = seal(tmp_path, predecessor)["protocol_sha256"]
    for broken in (
        fake_reader(events=120),
        fake_reader(tip="f" * 64),
        fake_reader(observed="2026-11-07T01:00:00+00:00"),
    ):
        recorder = Recorder()
        reasons = refusal_of(
            lambda **kw: launch(tmp_path, predecessor, digest, recorder, **kw),
            read_predecessor=broken,
        )
        assert any(
            item.startswith("predecessor_root_not_reverifiable") for item in reasons
        )
        assert recorder.calls == []
    (predecessor / "collection-failure.json").unlink()
    recorder = Recorder()
    assert any(
        "failure_marker_missing" in item
        for item in refusal_of(
            lambda **kw: launch(tmp_path, predecessor, digest, recorder, **kw)
        )
    )


def test_launch_refuses_a_root_that_is_not_the_attempt_three_lineage(tmp_path):
    root = tmp_path / "attempt3"
    first_attempt = {
        "schema": "carry_paper_attempt_lineage_v2",
        "attempt_number": 1,
        "predecessor": None,
    }
    digest = operations.seal_paper_study(
        root, start_at=START, attempt_lineage=first_attempt, clock=clock
    )
    recorder = Recorder()
    assert refusal_of(lambda **kw: launch(tmp_path, None, digest, recorder, **kw)) == [
        "protocol_not_attempt3_lineage"
    ]
    assert recorder.calls == []


def test_launch_refusals_do_not_reach_the_supervisor(
    tmp_path, predecessor, monkeypatch
):
    digest = seal(tmp_path, predecessor)["protocol_sha256"]
    cases = {
        "authorization_missing": dict(authorization=None),
        "authorization_malformed": dict(authorization="https://example.com/x"),
        "start_not_in_future": dict(start_at=NOW),
        "checkout_dirty": dict(code_identity=lambda: CodeIdentity("a" * 40, False)),
        "protocol_digest_mismatch": dict(protocol_sha256="e" * 64),
        "protocol_start_differs": dict(start_at=START + timedelta(hours=1)),
        "network_not_allowed": dict(allow_network=False),
    }
    for reason, overrides in cases.items():
        recorder = Recorder()
        reasons = refusal_of(
            lambda **kw: launch(tmp_path, predecessor, digest, recorder, **kw),
            **overrides,
        )
        assert reason in reasons, (reason, reasons)
        assert recorder.calls == []
    monkeypatch.setattr(
        attempt3, "build_candidate_run_provenance", lambda: {"source": "changed"}
    )
    recorder = Recorder()
    assert refusal_of(
        lambda **kw: launch(tmp_path, predecessor, digest, recorder, **kw)
    ) == ["protocol_provenance_changed"]
    assert recorder.calls == []


def test_supervisor_relaunch_after_start_needs_a_verified_root_that_already_ran(
    tmp_path, predecessor
):
    digest = seal(tmp_path, predecessor)["protocol_sha256"]
    root = tmp_path / "attempt3"
    after_start = START + timedelta(days=3)
    assert refusal_of(
        lambda **kw: launch(tmp_path, predecessor, digest, Recorder(), **kw),
        clock=lambda: after_start,
    ) == ["start_not_in_future"]
    write_heartbeat(
        root,
        protocol_sha256=digest,
        state="cycle_completed",
        pid=1,
        at=after_start,
        last_command_at=None,
        cycles=1,
    )
    outcome, recorder = launch(tmp_path, predecessor, digest, clock=lambda: after_start)
    assert outcome["exit_code"] == 0
    assert [call[0] for call in recorder.calls] == ["launcher", "supervise"]


def test_launch_refuses_a_root_owned_by_a_live_collector_and_writes_nothing(
    tmp_path, predecessor, monkeypatch
):
    from trade_rl.evaluation.paper import supervisor
    from trade_rl.evaluation.paper.supervisor import PaperCollector

    digest = seal(tmp_path, predecessor)["protocol_sha256"]
    root = tmp_path / "attempt3"
    monkeypatch.setattr(supervisor, "collection_lineage_problems", lambda *a, **k: [])
    recorder = Recorder()
    with PaperCollector(root, expected_protocol_sha256=digest):
        before = sorted(path.name for path in root.iterdir())
        assert refusal_of(
            lambda **kw: launch(tmp_path, predecessor, digest, recorder, **kw)
        ) == ["collector_already_running"]
        assert sorted(path.name for path in root.iterdir()) == before
    assert recorder.calls == []


def test_launch_cli_maps_a_live_collector_to_exit_four(tmp_path, capsys, monkeypatch):
    def contended(*_a, **_k):
        raise Attempt3Refusal(["collector_already_running"])

    monkeypatch.setattr(attempt3, "launch_attempt3", contended)
    code = attempt3.main(
        [
            "launch",
            "--root",
            str(tmp_path / "x"),
            "--start-at",
            START.isoformat(),
            "--protocol-sha256",
            "a" * 64,
            "--log-file",
            str(tmp_path / "x.log"),
        ]
    )
    assert code == 4
    assert json.loads(capsys.readouterr().err)["refused"] == [
        "collector_already_running"
    ]


def test_cli_refuses_without_authorization_before_any_side_effect(tmp_path, capsys):
    root = tmp_path / "attempt3"
    start = (datetime.now(UTC) + timedelta(days=1)).isoformat()
    for command in ("plan", "seal"):
        code = attempt3.main([command, "--root", str(root), "--start-at", start])
        assert code == 2
        assert "authorization_missing" in json.loads(capsys.readouterr().err)["refused"]
    code = attempt3.main(
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
