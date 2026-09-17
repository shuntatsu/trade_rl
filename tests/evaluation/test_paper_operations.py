import json
from datetime import timedelta
from fractions import Fraction

import pytest

from tests.evaluation.test_paper_supervisor import collector, setup
from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import operations
from trade_rl.evaluation.paper.engine import PaperEngine


def test_economic_seal_fixes_duration_costs_and_requires_future_notice(tmp_path):
    root = tmp_path / "study"
    start = NOW + timedelta(minutes=5)
    digest = operations.seal_paper_study(root, start_at=start, clock=lambda: NOW)
    protocol = json.loads((root / "protocol.json").read_bytes())["protocol"]
    assert len(digest) == 64
    assert protocol["settings"]["close_at"] == (start + timedelta(days=90)).isoformat()
    assert protocol["settings"]["initial_capital"] == 10000
    assert protocol["settings"]["spot_fee_bps"] == 10
    assert protocol["settings"]["perpetual_fee_bps"] == 5
    assert protocol["study"]["research_plan"]["schema"] == "carry_paper_screen_v1"
    with pytest.raises(ValueError, match="five minutes"):
        operations.seal_paper_study(
            tmp_path / "late", start_at=start, clock=lambda: NOW + timedelta(seconds=1)
        )
    assert not (tmp_path / "late").exists()


def test_seal_normalizes_local_start_to_ninety_utc_days(tmp_path):
    from datetime import timezone

    start = (NOW + timedelta(minutes=5)).astimezone(timezone(timedelta(hours=9)))
    operations.seal_paper_study(tmp_path / "study", start_at=start, clock=lambda: NOW)
    settings = json.loads((tmp_path / "study/protocol.json").read_bytes())["protocol"][
        "settings"
    ]
    assert settings["start_at"] == (NOW + timedelta(minutes=5)).isoformat()
    assert settings["close_at"] == (NOW + timedelta(minutes=5, days=90)).isoformat()


def test_runner_uses_future_slots_and_preserves_real_round_trip(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    published, sleeps = [], []

    def wait(seconds):
        assert 0 < seconds <= 60
        sleeps.append(seconds)
        clock.value += timedelta(seconds=seconds)

    with collector(root, digest, clock, feed) as active:
        operations.collect_until_finished(
            active, clock=clock, sleep=wait, publish=published.append
        )
    assert published[0]["phase"] == "waiting"
    assert published[-1]["phase"] == "finished"
    assert published[-1]["status"]["terminal_flat"]
    assert published[-1]["status"]["account"]["fill_count"] == 8
    assert len(sleeps) >= 4
    assert not published[-1]["status"]["quality_failures"]
    state = operations.collection_status(root, expected_protocol_sha256=digest)
    assert state["verification"] == "journal_chain_only"
    assert state["economic_decision"] == "NOT_EVALUATED"
    assert not state["production_eligible"]
    assert state["status"]["account"] == published[-1]["status"]["account"]


def test_runner_never_retries_a_failed_request(tmp_path, monkeypatch):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    clock.value = NOW + timedelta(seconds=2)
    feed.fail = True
    with pytest.raises(RuntimeError, match="source unavailable"):
        with collector(root, digest, clock, feed) as active:
            operations.collect_until_finished(
                active,
                clock=clock,
                sleep=lambda _: pytest.fail("retry sleep"),
                publish=lambda _: None,
            )
    assert len(feed.urls) == 1
    assert (root / "collection-failure.json").is_file()


def test_cadence_clock_reversal_halts_instead_of_waiting_away_the_evidence(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch)
    clock.value = NOW + timedelta(seconds=2)
    shifted = False

    def wait(seconds):
        nonlocal shifted
        if not shifted:
            clock.value -= timedelta(seconds=120)
            shifted = True
        else:
            clock.value += timedelta(seconds=seconds)

    with pytest.raises(ValueError, match="backwards"):
        with collector(root, digest, clock, feed) as active:
            operations.collect_until_finished(
                active, clock=clock, sleep=wait, publish=lambda _: None
            )
    assert (root / "collection-failure.json").is_file()
    assert (
        json.loads((root / "collection-failure.json").read_bytes())["error_type"]
        == "ValueError"
    )


def test_status_empty_journal_preserves_initial_capital(tmp_path):
    root = tmp_path / "study"
    digest = operations.seal_paper_study(
        root, start_at=NOW + timedelta(minutes=5), clock=lambda: NOW
    )
    state = operations.collection_status(root, expected_protocol_sha256=digest)
    assert state["events"] == 0 and state["tip"] == digest
    assert state["status"]["account"]["equity"] == 10000
    assert state["status"]["observed_at"] is None


def test_runner_stops_a_failed_screen_only_after_actual_flat_exit(
    tmp_path, monkeypatch
):
    root, digest, clock, feed = setup(tmp_path, monkeypatch, close_seconds=3600)
    published = []

    def publish(row):
        published.append(row)
        if row["status"]["account"]["fill_count"] == 4:
            clock.value += timedelta(seconds=200)

    def wait(seconds):
        clock.value += timedelta(seconds=seconds)

    with collector(root, digest, clock, feed) as active:
        last = operations.collect_until_finished(
            active, clock=clock, sleep=wait, publish=publish
        )
    assert last["phase"] == "rejected"
    assert last["status"]["terminal_flat"]
    assert last["status"]["account"]["fill_count"] == 8
    assert "observation_gap" in last["status"]["quality_failures"]
    assert clock.value < NOW + timedelta(seconds=3600)


def test_engine_reports_unpaid_announced_funding_for_held_quantity(tmp_path):
    from tests.evaluation.test_paper_engine import entry, market

    root = tmp_path / "paper"
    engine, rule, digest, _, _ = entry(root)
    engine.decide(
        "scheduled",
        market=market(root, "m3", 2, next_funding=3),
        rules=rule,
        at=NOW + timedelta(seconds=2.1),
    )
    assert engine.unsettled_funding() == []
    engine.decide(
        "due", market=market(root, "m4", 4), rules=rule, at=NOW + timedelta(seconds=4.1)
    )
    unpaid = engine.unsettled_funding()
    assert [row["symbol"] for row in unpaid] == ["BTCUSDT", "ETHUSDT"]
    assert all(
        row["funding_at"] == (NOW + timedelta(seconds=3)).isoformat() for row in unpaid
    )
    assert all(Fraction(row["quantity"]) < 0 for row in unpaid)
    assert not engine.status()["quality_failures"]
    assert (
        PaperEngine(root, expected_protocol_sha256=digest).unsettled_funding() == unpaid
    )
    engine.decide(
        "paid",
        market=market(root, "m5", 5, funding=(NOW + timedelta(seconds=3), 0.001)),
        rules=rule,
        at=NOW + timedelta(seconds=5.1),
    )
    assert engine.unsettled_funding() == []
