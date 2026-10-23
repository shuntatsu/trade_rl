"""Pure progress transport and clock contracts; no Dataset/native execution."""

import json
from pathlib import Path

import pytest

from tools.tier_d_nonrl.progress import ArmProgressSpec, validate_progress_policy
from tools.tier_d_nonrl.progress_io import (
    ProgressJournal,
    ProgressReader,
    publish_roster,
)
from tools.tier_d_nonrl.tier_d_protocol import json_sha256

POLICY = {
    "schema": "issue810_native_progress_v1",
    "no_progress_seconds": 10,
    "sample_seconds": 1,
}


def spec(account="A", **changes):
    return ArmProgressSpec(
        **(
            {
                "configuration": "C",
                "account": account,
                "scenario": "BASEL0",
                "role": "nonrl",
                "plan_digest": "a" * 64,
                "start_index": 10,
                "actor_stop": 13,
                "settlement_stop": 15,
            }
            | changes
        )
    )


def setup_attempt(tmp_path, *, accounts=("A",)):
    root = tmp_path / "attempt"
    root.mkdir()
    protocol = {
        "progress_policy": dict(POLICY),
        "accounts": list(accounts),
        "roles": ["nonrl"],
        "scenarios": ["BASEL0"],
        "configurations": [{"id": "C"}],
        "actor": {"start": 10, "stop": 13},
        "terminal": {"stop_inclusive": 15},
    }
    (root / "protocol.json").write_text(json.dumps(protocol), encoding="utf-8")
    specs = tuple(spec(account) for account in accounts)
    publish_roster(root, protocol, specs)
    return root, protocol, specs


def row(protocol, arm, event="native_started", count=0):
    processing = arm.start_index + count
    advance = event == "native_advanced"
    return {
        "schema": "issue810_native_progress_record_v1",
        "protocol_digest": json_sha256(protocol),
        "arm_id": arm.arm_id,
        "plan_digest": arm.plan_digest,
        "event": event,
        "seq": count + (event in {"native_returned", "native_failed"}),
        "count": count,
        "total": arm.target,
        "phase": ("actor" if processing <= arm.actor_stop else "reserve")
        if advance
        else None,
        "start_index": processing - 1 if advance else processing,
        "processing_index": processing,
    }


def append(root, arm, value, *, newline=True):
    directory = root / "native-progress"
    directory.mkdir(exist_ok=True)
    path = directory / f"{arm.arm_id}.jsonl"
    with path.open("ab") as output:
        output.write(json.dumps(value).encode() + (b"\n" if newline else b""))
    return path


def test_spec_identity_and_payload_are_detached_and_path_safe():
    arm = spec(account="../arbitrary account")
    assert arm.arm_id == spec(account="../arbitrary account").arm_id
    assert len(arm.arm_id) == 64 and set(arm.arm_id) <= set("0123456789abcdef")
    assert arm.target == 5
    payload = arm.payload
    payload["account"] = "changed"
    assert arm.payload["account"] == "../arbitrary account"
    assert spec(plan_digest="b" * 64).arm_id != spec().arm_id


@pytest.mark.parametrize(
    "changes",
    [
        {"start_index": True},
        {"actor_stop": 10},
        {"role": "other"},
        {"plan_digest": "x"},
    ],
)
def test_spec_rejects_wrong_primitives_and_ranges(changes):
    with pytest.raises(ValueError):
        spec(**changes)


@pytest.mark.parametrize(
    "value",
    [
        None,
        {},
        POLICY | {"sample_seconds": True},
        POLICY | {"no_progress_seconds": 0},
        POLICY | {"extra": 1},
    ],
)
def test_policy_has_no_implicit_actual_protocol_defaults(value):
    with pytest.raises(ValueError):
        validate_progress_policy(value)


def test_roster_is_exclusive_complete_and_bound_to_existing_protocol(tmp_path):
    root, protocol, arms = setup_attempt(tmp_path)
    before = (root / "native-progress-roster.json").read_bytes()
    with pytest.raises(FileExistsError):
        publish_roster(root, protocol, arms)
    assert (root / "native-progress-roster.json").read_bytes() == before
    with pytest.raises(ValueError):
        publish_roster(tmp_path, protocol, ())


def test_journal_durability_notifications_and_complete_counts(tmp_path, monkeypatch):
    from tools.tier_d_nonrl import progress_io

    root, protocol, (arm,) = setup_attempt(tmp_path)
    reader = ProgressReader(root, expected_protocol=protocol)
    calls, observed = [], []
    real_fsync = progress_io.os.fsync

    def fsync(fd):
        real_fsync(fd)
        calls.append("durable")

    def changed():
        assert calls[-1] == "durable"
        calls.append("notification")
        observed.extend(reader.poll())

    monkeypatch.setattr(progress_io.os, "fsync", fsync)
    with ProgressJournal(root, arm, on_change=changed) as journal:
        for start in range(10, 15):
            journal.advance("actor" if start < 13 else "reserve", start, start + 1)
    assert [item["event"] for item in observed] == [
        "native_started",
        *("native_advanced" for _ in range(5)),
        "native_returned",
    ]
    assert [item["count"] for item in observed] == [0, 1, 2, 3, 4, 5, 5]
    assert all(item["configuration"] == "C" for item in observed)
    assert calls == ["durable", "notification"] * 7
    assert reader.states[arm.arm_id]["event"] == "native_returned"
    assert reader.stalled() == []
    saved = reader.states
    saved[arm.arm_id]["count"] = 0
    assert reader.states[arm.arm_id]["count"] == 5
    before = (root / "native-progress" / f"{arm.arm_id}.jsonl").read_bytes()
    with pytest.raises(FileExistsError):
        with ProgressJournal(root, arm):
            pytest.fail("old arm journal reopened")
    assert (root / "native-progress" / f"{arm.arm_id}.jsonl").read_bytes() == before


def test_no_native_start_is_not_stalled_and_other_arm_progress_does_not_reset(tmp_path):
    now = [0.0]
    root, protocol, (a, b) = setup_attempt(tmp_path, accounts=("A", "B"))
    reader = ProgressReader(root, clock=lambda: now[0])
    assert list(reader.poll()) == [] and reader.stalled() == []
    append(root, a, row(protocol, a))
    append(root, b, row(protocol, b))
    assert len(list(reader.poll())) == 2
    now[0] = 2
    append(root, a, row(protocol, a, "native_advanced", 1))
    list(reader.poll())
    now[0] = 9
    append(root, b, row(protocol, b, "native_advanced", 1))
    list(reader.poll())
    now[0] = 12
    stalled = reader.stalled()
    assert len(stalled) == 1 and stalled[0]["arm_id"] == a.arm_id
    assert stalled[0]["idle_seconds"] == 10
    assert stalled[0]["count"] == 1 and stalled[0]["target"] == 5


def test_completed_frontier_or_failed_end_is_excluded_from_stall(tmp_path):
    root, protocol, (a, b) = setup_attempt(tmp_path, accounts=("A", "B"))
    now = [0]
    reader = ProgressReader(root, clock=lambda: now[0])
    append(root, a, row(protocol, a))
    for count in range(1, 6):
        append(root, a, row(protocol, a, "native_advanced", count))
    append(root, b, row(protocol, b))
    append(root, b, row(protocol, b, "native_failed"))
    list(reader.poll())
    now[0] = 100
    assert reader.stalled() == []


def test_short_native_return_ends_watchdog_without_claiming_full_completion(tmp_path):
    root, _, (arm,) = setup_attempt(tmp_path)
    now = [0]
    reader = ProgressReader(root, clock=lambda: now[0])
    with ProgressJournal(root, arm) as journal:
        journal.advance("actor", 10, 11)
    events = list(reader.poll())
    assert events[-1]["event"] == "native_returned"
    assert events[-1]["count"] == 1 and events[-1]["target"] == 5
    assert "qualified" not in events[-1] and "completed" not in events[-1]
    now[0] = 100
    assert reader.stalled() == []


def test_incomplete_eof_is_retained_until_newline_without_read_side_effects(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    path = append(root, arm, row(protocol, arm), newline=False)
    reader = ProgressReader(root, expected_policy=POLICY)
    before = path.read_bytes()
    assert list(reader.poll()) == [] and reader.states == {}
    assert path.read_bytes() == before
    with path.open("ab") as output:
        output.write(b"\n")
    assert len(list(reader.poll())) == 1
    assert list(reader.poll()) == []


def test_enrichment_keeps_event_clock_and_separately_binds_arm_origin(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    append(root, arm, row(protocol, arm))
    append(root, arm, row(protocol, arm, "native_advanced", 1))
    append(root, arm, row(protocol, arm, "native_advanced", 2))
    records = list(ProgressReader(root).poll())
    assert records[-1]["start_index"] == 11
    assert records[-1]["processing_index"] == 12
    assert records[-1]["arm_start_index"] == 10


@pytest.mark.parametrize(
    "changes",
    [
        {"seq": True},
        {"count": 2},
        {"count": True},
        {"total": 5.0},
        {"phase": "reserve"},
        {"plan_digest": "b" * 64},
        {"protocol_digest": "b" * 64},
        {"event": "heartbeat"},
        {"event": []},
        {"extra": 1},
    ],
)
def test_reader_rejects_malformed_gap_phase_and_foreign_binding(tmp_path, changes):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    append(root, arm, row(protocol, arm))
    append(root, arm, row(protocol, arm, "native_advanced", 1) | changes)
    with pytest.raises(ValueError):
        list(ProgressReader(root).poll())


def test_duplicate_does_not_refresh_last_valid_advance_clock(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    now = [0]
    reader = ProgressReader(root, clock=lambda: now[0])
    record = row(protocol, arm)
    append(root, arm, record)
    list(reader.poll())
    now[0] = 9
    append(root, arm, record)
    with pytest.raises(ValueError):
        list(reader.poll())
    now[0] = 10
    assert reader.stalled()[0]["idle_seconds"] == 10


@pytest.mark.parametrize("name", ["protocol.json", "native-progress-roster.json"])
def test_control_bytes_cannot_change_after_observation(tmp_path, name):
    root, _, _ = setup_attempt(tmp_path)
    reader = ProgressReader(root)
    list(reader.poll())
    with (root / name).open("ab") as output:
        output.write(b" ")
    with pytest.raises(ValueError, match="changed"):
        list(reader.poll())


def test_foreign_path_and_resolved_escape_reject_before_content_read(
    tmp_path, monkeypatch
):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    path = append(root, arm, row(protocol, arm))
    (path.parent / "foreign.jsonl").write_text("{}\n")
    with pytest.raises(ValueError, match="foreign"):
        list(ProgressReader(root).poll())
    (path.parent / "foreign.jsonl").unlink()
    original = Path.resolve

    def resolve(self, *args, **kwargs):
        return (
            tmp_path / "outside.jsonl"
            if self == path
            else original(self, *args, **kwargs)
        )

    monkeypatch.setattr(Path, "resolve", resolve)
    with pytest.raises(ValueError, match="outside"):
        list(ProgressReader(root).poll())


def test_end_publication_failure_never_masks_primary_native_error(
    tmp_path, monkeypatch
):
    from tools.tier_d_nonrl import progress_io

    root, _, (arm,) = setup_attempt(tmp_path)
    original, calls = progress_io.os.fsync, []

    def fsync(fd):
        calls.append(True)
        if len(calls) == 2:
            raise OSError("end publication failed")
        original(fd)

    monkeypatch.setattr(progress_io.os, "fsync", fsync)
    error = LookupError("native primary failure")
    with pytest.raises(LookupError) as caught:
        with ProgressJournal(root, arm):
            raise error
    assert caught.value is error
    assert len(calls) == 2


def test_split_json_and_bounded_line_corruption_preserve_known_prefix(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    path = append(root, arm, row(protocol, arm))
    reader = ProgressReader(root)
    list(reader.poll())
    next_line = json.dumps(row(protocol, arm, "native_advanced", 1)).encode() + b"\n"
    with path.open("ab") as output:
        output.write(next_line[:20])
    assert list(reader.poll()) == []
    with path.open("ab") as output:
        output.write(next_line[20:])
    assert [item["count"] for item in reader.poll()] == [1]
    with path.open("ab") as output:
        output.write(b"x" * (16 * 1024 + 1))
    with pytest.raises(ValueError, match="bounded"):
        list(reader.poll())
    assert reader.states[arm.arm_id]["count"] == 1


def test_duplicate_json_keys_and_journal_truncation_fail_closed(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    path = append(root, arm, row(protocol, arm))
    reader = ProgressReader(root)
    list(reader.poll())
    with path.open("ab") as output:
        output.write(b'{"event":"native_failed","event":"native_returned"}\n')
    with pytest.raises(ValueError, match="duplicate"):
        list(reader.poll())
    path.write_bytes(b"")
    with pytest.raises(ValueError, match="truncated"):
        list(reader.poll())


def test_wrong_journal_phase_rejects_before_publication_and_retains_failed_prefix(
    tmp_path,
):
    root, _, (arm,) = setup_attempt(tmp_path)
    error = LookupError("native body stopped")
    with pytest.raises(LookupError):
        with ProgressJournal(root, arm) as journal:
            journal.advance("actor", 10, 11)
            path = root / "native-progress" / f"{arm.arm_id}.jsonl"
            before = path.read_bytes()
            with pytest.raises(ValueError):
                journal.advance("reserve", 11, 12)
            assert path.read_bytes() == before
            raise error
    records = list(ProgressReader(root).poll())
    assert records[-1]["event"] == "native_failed" and records[-1]["count"] == 1


def test_missing_preparation_and_mismatched_expected_bindings_are_not_active(tmp_path):
    reader = ProgressReader(tmp_path / "not-created", expected_policy=POLICY)
    assert list(reader.poll()) == [] and reader.stalled() == []
    root, protocol, _ = setup_attempt(tmp_path)
    with pytest.raises(ValueError, match="expected"):
        list(ProgressReader(root, expected_protocol=protocol | {"extra": 1}).poll())
    with pytest.raises(ValueError, match="expected"):
        list(
            ProgressReader(root, expected_policy=POLICY | {"sample_seconds": 2}).poll()
        )


def test_roster_publication_is_durable_atomic_and_exclusive(tmp_path, monkeypatch):
    from tools.tier_d_nonrl import progress_io

    root, protocol, (arm,) = setup_attempt(tmp_path)
    roster_path = root / "native-progress-roster.json"
    roster_path.unlink()
    reader, observations = ProgressReader(root), []
    original = progress_io.os.fsync

    def staged_fsync(fd):
        observations.append(roster_path.exists())
        assert list(reader.poll()) == [] and reader.states == {}
        original(fd)

    with monkeypatch.context() as patch:
        patch.setattr(progress_io.os, "fsync", staged_fsync)
        publish_roster(root, protocol, (arm,))
    assert observations == [False]
    retained = roster_path.read_bytes()
    with ProgressJournal(root, arm):
        pass
    assert list(reader.poll())[-1]["event"] == "native_returned"
    with pytest.raises(FileExistsError):
        publish_roster(root, protocol, (arm,))
    assert roster_path.read_bytes() == retained


def test_preparing_partial_protocol_does_not_establish_immutable_binding(tmp_path):
    root = tmp_path / "attempt"
    root.mkdir()
    (root / "protocol.json").write_bytes(b"{")
    reader = ProgressReader(root)
    assert list(reader.poll()) == []
    (root / "protocol.json").unlink()
    root.rmdir()
    setup_attempt(tmp_path)
    assert list(reader.poll()) == []


def test_dangling_link_presence_is_not_silently_treated_as_unstarted(
    tmp_path, monkeypatch
):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    path = append(root, arm, row(protocol, arm))
    exists, symlink = Path.exists, Path.is_symlink
    monkeypatch.setattr(
        Path, "exists", lambda self: False if self == path else exists(self)
    )
    monkeypatch.setattr(
        Path, "is_symlink", lambda self: True if self == path else symlink(self)
    )
    with pytest.raises(ValueError, match="non-symlink"):
        list(ProgressReader(root).poll())


def test_observed_directory_disappearance_rejects_and_keeps_known_states(tmp_path):
    root, protocol, (arm,) = setup_attempt(tmp_path)
    append(root, arm, row(protocol, arm))
    reader = ProgressReader(root)
    list(reader.poll())
    before = reader.states
    directory = root / "native-progress"
    retained = root / "retained-native-progress"
    assert directory.resolve().is_relative_to(tmp_path.resolve())
    assert retained.resolve().is_relative_to(tmp_path.resolve())
    directory.rename(retained)
    with pytest.raises(ValueError, match="disappeared"):
        list(reader.poll())
    assert reader.states == before
    assert (retained / f"{arm.arm_id}.jsonl").is_file()
