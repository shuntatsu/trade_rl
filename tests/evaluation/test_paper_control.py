import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from trade_rl.evaluation.paper.control import CollectionControl

_DIGEST = "a" * 64


def _create_legacy_control(
    root, *, completed_cycles: int, pending_cycles: int = 0
) -> None:
    root.mkdir()
    with sqlite3.connect(root / "collection-control.sqlite") as connection:
        connection.execute("CREATE TABLE identity(digest TEXT NOT NULL)")
        connection.execute("INSERT INTO identity VALUES(?)", (_DIGEST,))
        connection.execute(
            "CREATE TABLE cycles(id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, "
            "complete INTEGER NOT NULL CHECK(complete IN (0,1)))"
        )
        rows = [
            (cycle, f"completed-{cycle}", 1) for cycle in range(1, completed_cycles + 1)
        ]
        rows.extend(
            (cycle, f"pending-{cycle}", 0)
            for cycle in range(
                completed_cycles + 1, completed_cycles + pending_cycles + 1
            )
        )
        connection.executemany(
            "INSERT INTO cycles(id,started_at,complete) VALUES(?,?,?)", rows
        )


def test_legacy_completed_history_uses_incomplete_only_index_and_reopens(tmp_path):
    root = tmp_path / "collection"
    _create_legacy_control(root, completed_cycles=4096)

    control = CollectionControl(root, _DIGEST)
    try:
        plan = " ".join(
            row[3]
            for row in control.connection.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM cycles WHERE complete=0 LIMIT 1"
            )
        )
        assert "cycles_incomplete_idx" in plan
        indexes = {
            row[1]: row
            for row in control.connection.execute("PRAGMA index_list(cycles)")
        }
        index = indexes.get("cycles_incomplete_idx")
        assert index is not None
        assert index[4] == 1
        assert [
            row[2]
            for row in control.connection.execute(
                "PRAGMA index_info(cycles_incomplete_idx)"
            )
        ] == ["id"]
        assert control.connection.execute(
            "SELECT COUNT(*) FROM cycles WHERE complete=0"
        ).fetchone() == (0,)

        started = datetime(2026, 1, 1, tzinfo=UTC)
        cycle = control.begin(started)
        with pytest.raises(RuntimeError, match="interrupted collection cycle"):
            control.begin(started + timedelta(seconds=1))
        control.finish(cycle)
    finally:
        control.close()

    reopened = CollectionControl(root, _DIGEST)
    try:
        cycle = reopened.begin(datetime(2026, 1, 1, 0, 1, tzinfo=UTC))
        reopened.finish(cycle)
    finally:
        reopened.close()


def test_legacy_pending_rows_still_halt_open_and_remain_unacknowledged(tmp_path):
    root = tmp_path / "collection"
    _create_legacy_control(root, completed_cycles=16, pending_cycles=2)

    with pytest.raises(RuntimeError, match="interrupted collection cycle"):
        CollectionControl(root, _DIGEST)

    with sqlite3.connect(root / "collection-control.sqlite") as connection:
        assert connection.execute(
            "SELECT COUNT(*) FROM cycles WHERE complete=0"
        ).fetchone() == (2,)
        plan = " ".join(
            row[3]
            for row in connection.execute(
                "EXPLAIN QUERY PLAN SELECT id FROM cycles WHERE complete=0 LIMIT 1"
            )
        )
        assert "cycles_incomplete_idx" in plan


def test_registry_tables_are_added_to_an_existing_database_and_survive_reopen(tmp_path):
    root = tmp_path / "collection"
    _create_legacy_control(root, completed_cycles=2)

    control = CollectionControl(root, _DIGEST)
    assert control.skipped_slots() == []
    cycle = control.begin(datetime(2026, 1, 1, tzinfo=UTC))
    ordinal = control.register_attempt(
        cycle,
        slot=7,
        purpose="snapshot",
        path="sources/2026-01-01/a",
        outcome="failed",
        error_type="TimeoutError",
        error="x" * 900,
    )
    assert ordinal == 1
    assert control.register_skips(cycle, [7], "snapshot_unavailable") == [7]
    control.finish(cycle)
    control.close()

    reopened = CollectionControl(root, _DIGEST)
    assert reopened.skipped_slots() == [7]
    row = reopened.connection.execute("SELECT error, ordinal FROM attempts").fetchone()
    assert len(row[0]) == 500 and row[1] == 1
    reopened.close()


def test_a_capture_directory_is_registered_once_and_the_first_skip_reason_wins(
    tmp_path,
):
    root = tmp_path / "collection"
    _create_legacy_control(root, completed_cycles=0)
    control = CollectionControl(root, _DIGEST)
    cycle = control.begin(datetime(2026, 1, 1, tzinfo=UTC))
    kwargs = dict(slot=1, purpose="rules", outcome="failed", error_type="E", error="e")
    assert control.register_attempt(cycle, path="sources/d/a", **kwargs) == 1
    assert control.register_attempt(cycle, path="sources/d/b", **kwargs) == 2
    with pytest.raises(sqlite3.IntegrityError):
        control.register_attempt(cycle, path="sources/d/a", **kwargs)
    with pytest.raises(ValueError, match="outcome"):
        control.register_attempt(
            cycle, path="sources/d/c", **{**kwargs, "outcome": "ok"}
        )

    assert control.register_skips(cycle, [3, 4], "missed_slot") == [3, 4]
    assert control.register_skips(cycle, [4, 5], "snapshot_unavailable") == [3, 4, 5]
    reasons = dict(
        control.connection.execute("SELECT slot, reason FROM skips").fetchall()
    )
    assert reasons == {3: "missed_slot", 4: "missed_slot", 5: "snapshot_unavailable"}
    control.close()


def test_previous_cycle_start_is_the_prior_acknowledged_or_open_cycle(tmp_path):
    root = tmp_path / "collection"
    _create_legacy_control(root, completed_cycles=0)
    control = CollectionControl(root, _DIGEST)
    first_at = datetime(2026, 1, 1, tzinfo=UTC)
    first = control.begin(first_at)
    control.finish(first)
    second = control.begin(first_at + timedelta(seconds=60))
    assert control.previous_cycle_start(first) is None
    assert control.previous_cycle_start(second) == first_at
    control.close()
