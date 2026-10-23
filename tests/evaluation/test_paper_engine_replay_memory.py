import hashlib
import json
import shutil
import sqlite3
from datetime import UTC, datetime, timedelta

import pytest

from tests.evaluation.test_paper_engine import NOW, market, setup
from trade_rl.artifacts import canonical_json_bytes
from trade_rl.evaluation.paper.engine import PaperEngine, PaperSettings


def initialized(root):
    start_at = datetime(2025, 1, 1, tzinfo=UTC)
    settings = PaperSettings(start_at=start_at, close_at=start_at + timedelta(days=2))
    digest = PaperEngine.initialize(root, settings=settings, study={"purpose": "test"})
    return PaperEngine(root, expected_protocol_sha256=digest), digest, start_at


def test_known_key_retry_returns_original_record_after_restart(tmp_path):
    root = tmp_path / "paper"
    engine, digest, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    original = engine.gap("gap-1", at=at, reason="source unavailable")

    reopened = PaperEngine(root, expected_protocol_sha256=digest)

    assert reopened.gap("gap-1", at=at, reason="source unavailable") == original
    assert reopened.status() == engine.status()


def test_warm_decision_retry_does_not_read_removed_source_files(tmp_path):
    root = tmp_path / "paper"
    engine, rules, _ = setup(root)
    reference = market(root, "m1", 0)
    at = NOW + timedelta(seconds=0.1)
    original = engine.decide("decision-1", market=reference, rules=rules, at=at)
    before = engine.status()
    shutil.rmtree(root / "m1")
    shutil.rmtree(root / "rules")

    assert engine.decide("decision-1", market=reference, rules=rules, at=at) == original
    assert engine.status() == before


def test_known_key_retry_rejects_changed_request_without_changing_state(tmp_path):
    root = tmp_path / "paper"
    engine, _, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    engine.gap("gap-1", at=at, reason="source unavailable")
    before = engine.status()

    with pytest.raises(ValueError, match="idempotency"):
        engine.gap("gap-1", at=at, reason="different reason")

    assert engine.status() == before


def test_engine_unseen_external_key_keeps_apply_then_append_semantics(tmp_path):
    root = tmp_path / "paper"
    engine, digest, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    external_engine, external_digest, _ = initialized(tmp_path / "external")
    assert external_digest == digest
    external = external_engine.gap("gap-1", at=at, reason="source unavailable")
    external = engine.journal.append(
        "gap-1", "gap", external["event"]["payload"], expected_parent=digest
    )
    before = engine.status()

    with pytest.raises(ValueError, match="follow the previous command"):
        engine.gap("gap-1", at=start_at, reason="source unavailable")

    assert engine.status() == before

    result = engine.gap("gap-1", at=at, reason="source unavailable")

    assert result == external
    assert engine.status()["events"] == 1
    assert engine.status() == external_engine.status()


def test_known_key_retry_fails_closed_when_trusted_row_changes(tmp_path):
    root = tmp_path / "paper"
    engine, _, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    engine.gap("gap-1", at=at, reason="source unavailable")
    before = engine.status()
    with sqlite3.connect(root / "journal.sqlite") as connection:
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute("UPDATE events SET body=? WHERE sequence=0", (b"{}",))

    with pytest.raises(ValueError, match="digest"):
        engine.gap("gap-1", at=at, reason="source unavailable")

    assert engine.status() == before


def test_known_key_retry_checks_trusted_digest_after_valid_rewrite(tmp_path):
    root = tmp_path / "paper"
    engine, _, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    engine.gap("gap-1", at=at, reason="source unavailable")
    before = engine.status()
    with sqlite3.connect(root / "journal.sqlite") as connection:
        row = connection.execute("SELECT body FROM events WHERE sequence=0").fetchone()
        event = json.loads(row[0])
        event["payload"]["result"]["quality_failures"] = ["rewritten"]
        body = canonical_json_bytes(event)
        connection.execute("DROP TRIGGER events_no_update")
        connection.execute(
            "UPDATE events SET sha256=?, body=? WHERE sequence=0",
            (hashlib.sha256(body).hexdigest(), body),
        )

    with pytest.raises(ValueError, match="expected digest"):
        engine.gap("gap-1", at=at, reason="source unavailable")

    assert engine.status() == before


def test_known_key_retry_fails_closed_when_trusted_row_is_missing(tmp_path):
    root = tmp_path / "paper"
    engine, _, start_at = initialized(root)
    at = start_at + timedelta(seconds=1)
    engine.gap("gap-1", at=at, reason="source unavailable")
    before = engine.status()
    with sqlite3.connect(root / "journal.sqlite") as connection:
        connection.execute("DROP TRIGGER events_no_delete")
        connection.execute("DELETE FROM events WHERE sequence=0")

    with pytest.raises(ValueError, match="indexed event is missing"):
        engine.gap("gap-1", at=at, reason="source unavailable")

    assert engine.status() == before
