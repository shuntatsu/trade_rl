import hashlib
import sqlite3
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import pytest

from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.store import PaperJournal


def initialized(tmp_path):
    root = tmp_path / "account"
    digest = PaperJournal.initialize(
        root, {"purpose": "test only", "initial_capital": 10000}
    )
    return root, digest, PaperJournal(root, expected_protocol_sha256=digest)


def test_protocol_bound_restart_recovers_exact_committed_events(tmp_path):
    root, digest, journal = initialized(tmp_path)
    assert journal.events() == []
    first = journal.append(
        "intent-1", "decision", {"lots": [17, -17]}, expected_parent=digest
    )
    second = journal.append(
        "observation-1",
        "observation",
        {"snapshot_sha256": "a" * 64},
        expected_parent=first["sha256"],
    )
    resumed = PaperJournal(root, expected_protocol_sha256=digest)
    assert resumed.events() == [first, second]
    assert first["event"]["sequence"] == 0 and second["event"]["sequence"] == 1
    assert second["event"]["parent"] == first["sha256"]
    assert (
        first["sha256"]
        == hashlib.sha256(canonical_json_bytes(first["event"])).hexdigest()
    )


def test_identical_command_retries_do_not_create_additional_events(tmp_path):
    root, digest, journal = initialized(tmp_path)
    first = journal.append(
        "intent-1", "decision", {"lots": [17, -17]}, expected_parent=digest
    )
    second = journal.append("obs-1", "observation", {}, expected_parent=first["sha256"])
    assert (
        journal.append(
            "intent-1", "decision", {"lots": [17, -17]}, expected_parent=digest
        )
        == first
    )
    assert PaperJournal(root, expected_protocol_sha256=digest).events() == [
        first,
        second,
    ]
    with pytest.raises(ValueError, match="idempotency"):
        journal.append(
            "intent-1", "decision", {"lots": [18, -18]}, expected_parent=digest
        )
    with pytest.raises(ValueError, match="idempotency"):
        journal.append(
            "intent-1",
            "decision",
            {"lots": [17, -17]},
            expected_parent=second["sha256"],
        )


def test_stale_parent_never_appends(tmp_path):
    _, digest, journal = initialized(tmp_path)
    first = journal.append("one", "decision", {}, expected_parent=digest)
    with pytest.raises(ValueError, match="parent"):
        journal.append("two", "observation", {}, expected_parent=digest)
    assert journal.events() == [first]


def test_two_writers_from_same_tip_have_one_winner(tmp_path):
    root, digest, journal = initialized(tmp_path)

    def write(key):
        try:
            return PaperJournal(root, expected_protocol_sha256=digest).append(
                key, "decision", {}, expected_parent=digest
            )
        except ValueError:
            return None

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(write, ["one", "two"]))
    assert sum(value is not None for value in values) == 1
    assert len(journal.events()) == 1


def test_concurrent_exact_retries_share_one_event(tmp_path):
    root, digest, journal = initialized(tmp_path)

    def write(_):
        return PaperJournal(root, expected_protocol_sha256=digest).append(
            "one", "decision", {}, expected_parent=digest
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        values = list(pool.map(write, range(2)))
    assert values[0] == values[1] and journal.events() == [values[0]]


def test_initialization_and_open_never_overwrite_or_create_wrong_root(tmp_path):
    root, digest, journal = initialized(tmp_path)
    before = (root / "protocol.json").read_bytes()
    with pytest.raises(FileExistsError):
        PaperJournal.initialize(root, {"changed": True})
    with pytest.raises(ValueError, match="protocol"):
        PaperJournal(root, expected_protocol_sha256="0" * 64)
    missing = tmp_path / "missing"
    with pytest.raises(ValueError):
        PaperJournal(missing, expected_protocol_sha256=digest)
    assert not missing.exists()
    assert (root / "protocol.json").read_bytes() == before and journal.events() == []


def test_tampered_protocol_fails_before_database_write(tmp_path):
    root, digest, journal = initialized(tmp_path)
    (root / "protocol.json").write_text("{}")
    with pytest.raises(ValueError, match="protocol"):
        journal.append("one", "decision", {}, expected_parent=digest)


def test_updates_and_deletes_are_rejected_by_database(tmp_path):
    root, digest, journal = initialized(tmp_path)
    journal.append("one", "decision", {}, expected_parent=digest)
    with sqlite3.connect(root / "journal.sqlite") as db:
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("DELETE FROM events")
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("UPDATE events SET sha256='bad'")
    assert len(journal.events()) == 1


def test_database_rejects_gap_insertion_and_sql_replace(tmp_path):
    root, digest, journal = initialized(tmp_path)
    first = journal.append("one", "decision", {}, expected_parent=digest)
    with sqlite3.connect(root / "journal.sqlite") as db:
        with pytest.raises(sqlite3.DatabaseError):
            db.execute("INSERT INTO events VALUES(5,?,?,?)", ("gap", "a" * 64, b"{}"))
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(
                "INSERT OR REPLACE INTO events VALUES(0,?,?,?)",
                ("one", "a" * 64, b"{}"),
            )
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(
                "INSERT OR REPLACE INTO events VALUES(1,?,?,?)",
                ("one", "a" * 64, b"{}"),
            )
        with pytest.raises(sqlite3.DatabaseError):
            db.execute(
                "INSERT OR REPLACE INTO events VALUES(1,?,?,?)",
                ("two", first["sha256"], b"{}"),
            )
    assert journal.events() == [first]


def test_external_event_corruption_is_detected_on_rebuild(tmp_path):
    root, digest, journal = initialized(tmp_path)
    journal.append("one", "decision", {}, expected_parent=digest)
    with sqlite3.connect(root / "journal.sqlite") as db:
        db.execute("DROP TRIGGER events_no_update")
        db.execute("UPDATE events SET body=?", (b"{}",))
    with pytest.raises(ValueError, match="event|digest"):
        journal.events()


def test_process_death_before_commit_leaves_only_committed_prefix(tmp_path):
    root, digest, journal = initialized(tmp_path)
    first = journal.append("one", "decision", {}, expected_parent=digest)
    code = """
import os,sqlite3,sys
db=sqlite3.connect(sys.argv[1],isolation_level=None)
db.execute('BEGIN IMMEDIATE')
db.execute('INSERT INTO events(sequence,event_key,sha256,body) VALUES(1,?,?,?)',('interrupted','x'*64,b'{}'))
os._exit(7)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root / "journal.sqlite")],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    assert journal.events() == [first]
    second = journal.append(
        "interrupted", "observation", {}, expected_parent=first["sha256"]
    )
    assert second["event"]["sequence"] == 1
    assert len(PaperJournal(root, expected_protocol_sha256=digest).events()) == 2


def test_restart_recovers_hot_rollback_journal_after_dirty_pages_reach_disk(tmp_path):
    root, digest, journal = initialized(tmp_path)
    first = journal.append("one", "decision", {}, expected_parent=digest)
    code = """
import os,sqlite3,sys
db=sqlite3.connect(sys.argv[1],isolation_level=None)
db.execute('PRAGMA cache_size=1')
db.execute('BEGIN IMMEDIATE')
db.execute('INSERT INTO events(sequence,event_key,sha256,body) VALUES(1,?,?,?)',('interrupted','x'*64,b'x'*1000000))
os._exit(7)
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(root / "journal.sqlite")],
        capture_output=True,
        check=False,
    )
    assert result.returncode == 7, result.stderr
    assert (root / "journal.sqlite-journal").stat().st_size > 512
    resumed = PaperJournal(root, expected_protocol_sha256=digest)
    assert resumed.events() == [first]
    resumed.append("interrupted", "observation", {}, expected_parent=first["sha256"])
    assert len(resumed.events()) == 2


@pytest.mark.parametrize(
    "key,kind,payload",
    [
        ("", "decision", {}),
        ("x", "", {}),
        ("x", "INVALID", {}),
        ("x", "decision", {"bad": float("nan")}),
        ("x", "decision", []),
    ],
)
def test_invalid_event_cannot_modify_journal(tmp_path, key, kind, payload):
    _, digest, journal = initialized(tmp_path)
    with pytest.raises((TypeError, ValueError)):
        journal.append(key, kind, payload, expected_parent=digest)
    assert journal.events() == []
