"""Storage probe: fake clock and transport only; no socket is ever opened."""

from __future__ import annotations

import json
import math
import os
import socket
from datetime import timedelta
from pathlib import Path

import pytest

from tests.evaluation.test_paper_supervisor import Clock, FeedTransport
from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import attempt2, storage_probe, supervisor
from trade_rl.evaluation.paper.assessment import evaluate_paper_study
from trade_rl.evaluation.paper.attempt2 import Attempt2Refusal
from trade_rl.evaluation.paper.operations import STORAGE_PROBE_PLAN_SCHEMA
from trade_rl.evaluation.paper.storage_probe import measure_storage, storage_breakdown

AUTH = "https://github.com/shuntatsu/trade_rl/issues/810#issuecomment-4242424242"


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    def refuse(*_args, **_kwargs):
        raise AssertionError("the probe tests must not use the network")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)
    monkeypatch.setattr(
        supervisor, "build_candidate_run_provenance", lambda: {"source": "original"}
    )


class Sleeper:
    def __init__(self, clock):
        self.clock, self.calls = clock, 0

    def __call__(self, seconds):
        self.calls += 1
        self.clock.value += timedelta(seconds=seconds)


def probe(tmp_path, **overrides):
    clock = overrides.pop("clock", None) or Clock()
    transport = overrides.pop("transport", None) or FeedTransport(clock)
    sleeper = Sleeper(clock)
    options = dict(
        authorization=AUTH,
        allow_network=True,
        slots=10,
        clock=clock,
        transport=transport,
        monotonic=lambda: 0,
        sleep=sleeper,
        repository_root=tmp_path / "repository",
    )
    options.update(overrides)
    return measure_storage(tmp_path / "scratch" / "probe", **options), transport


def refusal_of(tmp_path, **overrides):
    with pytest.raises(Attempt2Refusal) as error:
        probe(tmp_path, **overrides)
    return error.value.reasons


# --------------------------------------------------------------- refusals


def test_probe_needs_authorization_the_network_flag_and_a_bounded_length(tmp_path):
    assert refusal_of(tmp_path, authorization=None) == ["authorization_missing"]
    assert refusal_of(tmp_path, authorization="https://example.com/x") == [
        "authorization_malformed"
    ]
    assert refusal_of(tmp_path, allow_network=False) == ["network_not_allowed"]
    for slots in (0, 9, 61, True):
        assert refusal_of(tmp_path, slots=slots) == ["probe_slots_out_of_range"]
    everything = refusal_of(tmp_path, authorization=None, allow_network=False, slots=1)
    assert everything == [
        "authorization_missing",
        "network_not_allowed",
        "probe_slots_out_of_range",
    ]
    assert not (tmp_path / "scratch").exists()


def test_refused_probe_never_touches_the_transport(tmp_path):
    clock = Clock()
    transport = FeedTransport(clock)
    refusal_of(tmp_path, clock=clock, transport=transport, allow_network=False)
    assert transport.urls == []


def test_probe_root_must_be_new_outside_the_repository_and_any_sealed_root(tmp_path):
    existing = tmp_path / "scratch" / "probe"
    existing.mkdir(parents=True)
    assert refusal_of(tmp_path) == ["probe_root_exists"]

    inside = tmp_path / "repository"
    with pytest.raises(Attempt2Refusal) as error:
        measure_storage(
            inside / "scratch" / "probe",
            authorization=AUTH,
            allow_network=True,
            repository_root=inside,
        )
    assert error.value.reasons == ["probe_root_inside_repository"]
    assert not (inside / "scratch").exists()

    sealed = tmp_path / "sealed"
    sealed.mkdir()
    (sealed / "protocol.json").write_text("{}", encoding="utf-8")
    with pytest.raises(Attempt2Refusal) as error:
        measure_storage(
            sealed / "child",
            authorization=AUTH,
            allow_network=True,
            repository_root=tmp_path / "repository",
        )
    assert error.value.reasons == ["probe_root_inside_a_sealed_root"]
    assert not (sealed / "child").exists()


def test_default_repository_root_is_this_checkout():
    assert (storage_probe.REPOSITORY_ROOT / "trade_rl" / "evaluation").is_dir()
    assert (storage_probe.REPOSITORY_ROOT / "pyproject.toml").is_file()


def test_a_source_failure_is_a_refusal_not_a_rate(tmp_path):
    clock = Clock()
    transport = FeedTransport(clock)
    transport.fail = True
    reasons = refusal_of(tmp_path, clock=clock, transport=transport)
    assert len(reasons) == 1
    assert reasons[0].startswith("probe_collection_failed:RuntimeError")


# ----------------------------------------------------------------- success


def test_probe_reports_bytes_per_observation_from_the_real_collector(tmp_path):
    report, transport = probe(tmp_path)
    root = Path(report["root"])
    assert root == (tmp_path / "scratch" / "probe").absolute()
    assert report["schema"] == storage_probe.PROBE_REPORT_SCHEMA
    assert report["probe_only"] is True
    assert report["sealable_as_attempt2"] is False and report["evaluable"] is False
    assert report["production_eligible"] is False
    assert report["authorization"] == AUTH
    assert report["observations"] >= storage_probe.PROBE_MINIMUM_OBSERVATIONS
    assert any("exchangeInfo" in url for url in transport.urls)

    on_disk = sum(
        os.path.getsize(os.path.join(folder, name))
        for folder, _dirs, files in os.walk(root)
        for name in files
    )
    storage = report["storage"]
    assert storage["total_bytes"] == on_disk > 0
    assert storage["total_bytes"] == (
        storage["rule_capture_bytes"]
        + storage["snapshot_capture_bytes"]
        + storage["other_bytes"]
    )
    assert storage["rule_captures"] >= 1 and storage["snapshot_captures"] >= 1
    measured = storage["total_bytes"] / report["observations"]
    assert report["bytes_per_observation"] == math.ceil(measured)
    assert (
        report["conservative_bytes_per_observation"] > report["bytes_per_observation"]
    )
    assert report["required_bytes_at_conservative_rate"] >= math.ceil(
        attempt2.SCREEN_SLOTS * report["bytes_per_observation"] * 1.25
    )
    assert report["screen_minute_slots"] == 129_604
    json.dumps(report, allow_nan=False)


def test_the_v1_status_reader_exposes_the_fields_the_lineage_check_compares(tmp_path):
    from trade_rl.evaluation.paper.operations import collection_status

    report, _ = probe(tmp_path)
    status = collection_status(
        report["root"], expected_protocol_sha256=report["probe_protocol_sha256"]
    )
    assert status["events"] == report["events"]
    assert status["status"]["observed_at"] is not None
    assert len(status["tip"]) == 64


def test_probe_root_is_marked_as_a_probe_and_carries_no_attempt_lineage(tmp_path):
    report, _ = probe(tmp_path)
    protocol = json.loads((Path(report["root"]) / "protocol.json").read_bytes())
    plan = protocol["protocol"]["study"]["research_plan"]
    assert plan["schema"] == STORAGE_PROBE_PLAN_SCHEMA
    assert plan["probe_only"] is True and plan["attempt_lineage"] is None
    assert plan["screen_eligible"] is False and plan["evaluable"] is False
    assert (
        protocol["protocol"]["settings"]["close_at"]
        != (protocol["protocol"]["settings"]["start_at"])
    )


def test_a_probe_root_cannot_be_evaluated_launched_or_resealed(tmp_path):
    report, _ = probe(tmp_path)
    root = Path(report["root"])
    digest = report["probe_protocol_sha256"]
    with pytest.raises(ValueError, match="storage probe"):
        evaluate_paper_study(
            root, expected_protocol_sha256=digest, expected_tip="0" * 64
        )
    reasons = attempt2._protocol_reasons(root, digest, NOW + timedelta(seconds=60))
    assert "protocol_is_a_storage_probe" in reasons
    assert "protocol_not_attempt2_lineage" in reasons
    # Attempt-2 seal never writes into an existing root, so a probe root cannot
    # become a screen either.
    from datetime import UTC, datetime

    from trade_rl.evaluation.forward_study.seal import CodeIdentity

    with pytest.raises(Attempt2Refusal) as error:
        attempt2.seal_attempt2(
            root,
            start_at=datetime.now(UTC) + timedelta(days=1),
            authorization=AUTH,
            bytes_per_observation=1_000_000,
            predecessor_root_status="unavailable",
            code_identity=lambda: CodeIdentity("a" * 40, True),
            disk_usage=lambda _p: type("U", (), {"free": 10**13})(),
        )
    assert "root_exists" in error.value.reasons


def test_the_probe_runs_a_bounded_number_of_slots(tmp_path):
    report, _ = probe(tmp_path, slots=10)
    # 10 slots plus the 180 second terminal grace, one observation per minute.
    assert 10 <= report["observations"] <= 14


def test_storage_breakdown_splits_rule_and_snapshot_captures(tmp_path):
    (tmp_path / "sources" / "d" / "a").mkdir(parents=True)
    (tmp_path / "sources" / "d" / "b").mkdir(parents=True)
    (tmp_path / "sources" / "d" / "a" / "rules.json").write_bytes(b"x" * 10)
    (tmp_path / "sources" / "d" / "a" / "extra.bin").write_bytes(b"x" * 5)
    (tmp_path / "sources" / "d" / "b" / "snapshot.json").write_bytes(b"x" * 7)
    (tmp_path / "journal.sqlite").write_bytes(b"x" * 3)
    assert storage_breakdown(tmp_path) == dict(
        rule_capture_bytes=15,
        snapshot_capture_bytes=7,
        other_bytes=3,
        total_bytes=25,
        rule_captures=1,
        snapshot_captures=1,
        journal_bytes=3,
    )


# --------------------------------------------------------------------- CLI


def test_cli_refuses_without_authorization_or_network_and_creates_nothing(
    tmp_path, capsys
):
    root = tmp_path / "scratch" / "probe"
    assert storage_probe.main(["--root", str(root)]) == 2
    refused = json.loads(capsys.readouterr().err)["refused"]
    assert {"authorization_missing", "network_not_allowed"} <= set(refused)
    assert not root.parent.exists()


def test_cli_output_is_write_once(tmp_path, capsys):
    output = tmp_path / "report.json"
    output.write_text("{}", encoding="utf-8")
    code = storage_probe.main(
        [
            "--root",
            str(tmp_path / "scratch" / "probe"),
            "--authorization",
            AUTH,
            "--allow-network",
            "--output",
            str(output),
        ]
    )
    assert code == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"
    assert not (tmp_path / "scratch").exists()


def test_conservative_rate_formula_is_pinned_exactly():
    usage = dict(
        rule_capture_bytes=1_000,
        snapshot_capture_bytes=30_000,
        other_bytes=9_000,
        total_bytes=40_000,
        rule_captures=1,
        snapshot_captures=15,
        journal_bytes=6_000,
    )
    # total / observations + mean snapshot + journal bytes per event
    assert storage_probe.conservative_rate(usage, observations=20, events=10) == (
        40_000 / 20 + 30_000 / 15 + 6_000 / 10
    )
    assert storage_probe.conservative_rate(usage, observations=20, events=10) == 4_600
    # zero captures or events cannot divide by zero
    empty = dict(usage, snapshot_captures=0, journal_bytes=0)
    assert storage_probe.conservative_rate(empty, observations=20, events=0) == (
        40_000 / 20 + 30_000
    )


def test_report_rate_is_the_pinned_formula_applied_to_the_report(tmp_path):
    report, _ = probe(tmp_path)
    storage = report["storage"]
    expected = (
        storage["total_bytes"] / report["observations"]
        + storage["snapshot_capture_bytes"] / storage["snapshot_captures"]
        + storage["journal_bytes"] / report["events"]
    )
    assert (
        storage["journal_bytes"]
        == (Path(report["root"]) / "journal.sqlite").stat().st_size
    )
    assert report["conservative_bytes_per_observation"] == math.ceil(expected)
    assert report["required_bytes_at_conservative_rate"] == math.ceil(
        attempt2.SCREEN_SLOTS * expected * 1.25
    )


def test_generic_run_and_supervise_refuse_a_probe_root_and_write_nothing(
    tmp_path, capsys
):
    from trade_rl.evaluation.paper import cli

    report, _ = probe(tmp_path)
    root = Path(report["root"])
    digest = report["probe_protocol_sha256"]
    before = sorted(p.name for p in root.iterdir())
    for command in ("run", "supervise"):
        arguments = [command, "--root", str(root), "--protocol-sha256", digest]
        if command == "supervise":
            arguments += ["--log-file", str(tmp_path / "child.log")]
        assert cli.main(arguments) == 2
        assert "storage probe" in json.loads(capsys.readouterr().err)["error"]
    assert sorted(p.name for p in root.iterdir()) == before
    assert not (tmp_path / "child.log").exists()
