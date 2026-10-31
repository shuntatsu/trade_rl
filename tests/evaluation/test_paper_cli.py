import json
from datetime import timedelta

import pytest

from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import cli
from trade_rl.evaluation.paper.operations import seal_paper_study

FIRST_ATTEMPT_LINEAGE = {
    "schema": "carry_paper_attempt_lineage_v2",
    "attempt_number": 1,
    "predecessor": None,
}


def test_cli_status_is_machine_readable_and_explicitly_not_economic_evidence(
    tmp_path, capsys
):
    root = tmp_path / "study"
    digest = seal_paper_study(
        root,
        start_at=NOW + timedelta(minutes=5),
        attempt_lineage=FIRST_ATTEMPT_LINEAGE,
        clock=lambda: NOW,
    )
    assert cli.main(["status", "--root", str(root), "--protocol-sha256", digest]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["tip"] == digest and output["events"] == 0
    assert output["economic_decision"] == "NOT_EVALUATED"
    assert output["production_eligible"] is False


def test_cli_wrong_anchor_fails_and_never_replaces_existing_output(tmp_path, capsys):
    root = tmp_path / "study"
    seal_paper_study(
        root,
        start_at=NOW + timedelta(minutes=5),
        attempt_lineage=FIRST_ATTEMPT_LINEAGE,
        clock=lambda: NOW,
    )
    assert cli.main(["status", "--root", str(root), "--protocol-sha256", "a" * 64]) == 2
    assert "digest" in json.loads(capsys.readouterr().err)["error"]


def test_cli_seal_publishes_the_external_anchor_and_refuses_root_reuse(
    tmp_path, capsys
):
    from datetime import UTC, datetime

    root = tmp_path / "study"
    lineage_path = tmp_path / "attempt-lineage.json"
    lineage_path.write_text(json.dumps(FIRST_ATTEMPT_LINEAGE), encoding="utf-8")
    args = [
        "seal",
        "--root",
        str(root),
        "--start-at",
        (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
        "--attempt-lineage-json",
        str(lineage_path),
    ]
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["protocol_sha256"]) == 64
    assert output["production_eligible"] is False
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"


def test_cli_seal_requires_explicit_attempt_lineage(tmp_path):
    from datetime import UTC, datetime

    root = tmp_path / "study"
    with pytest.raises(SystemExit):
        cli.main(
            [
                "seal",
                "--root",
                str(root),
                "--start-at",
                (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
            ]
        )
    assert not root.exists()


def test_terminal_window_alone_is_not_success_when_positions_remain(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext

    monkeypatch.setattr(
        cli, "PaperCollector", lambda *args, **kwargs: nullcontext(object())
    )
    monkeypatch.setattr(cli, "install_termination_handlers", lambda *_: [])
    monkeypatch.setattr(
        cli,
        "collect_until_finished",
        lambda *args, **kwargs: dict(
            phase="finished",
            status=dict(terminal_flat=False, quality_failures=[], pending=False),
        ),
    )
    assert (
        cli.main(["run", "--root", str(tmp_path), "--protocol-sha256", "a" * 64]) == 1
    )


def _sealed(tmp_path):
    root = tmp_path / "study"
    digest = seal_paper_study(
        root,
        start_at=NOW + timedelta(minutes=5),
        attempt_lineage=FIRST_ATTEMPT_LINEAGE,
        clock=lambda: NOW,
    )
    return root, digest


def test_cli_check_exits_three_when_it_records_a_stale_failure(
    tmp_path, capsys, monkeypatch
):
    from trade_rl.evaluation.paper import watchdog

    root, digest = _sealed(tmp_path)
    late = NOW + timedelta(days=2)
    original = watchdog.check_collection
    monkeypatch.setattr(
        cli,
        "check_collection",
        lambda *a, **k: original(*a, clock=lambda: late, **k),
    )
    code = cli.main(["check", "--root", str(root), "--protocol-sha256", digest])
    assert code == 3
    assert json.loads(capsys.readouterr().out)["state"] == "failed_never_started"
    assert (root / "collection-failure.json").is_file()


def test_cli_check_reports_healthy_before_start_and_rejects_a_wrong_anchor(
    tmp_path, capsys, monkeypatch
):
    from trade_rl.evaluation.paper import watchdog

    root, digest = _sealed(tmp_path)
    early = NOW
    original = watchdog.check_collection
    monkeypatch.setattr(
        cli,
        "check_collection",
        lambda *a, **k: original(*a, clock=lambda: early, **k),
    )
    args = ["check", "--root", str(root), "--protocol-sha256", digest]
    assert cli.main(args) == 0
    assert json.loads(capsys.readouterr().out)["state"] == "not_started"
    assert not (root / "collection-failure.json").exists()
    assert cli.main([*args[:-1], "a" * 64]) == 2
    assert "digest" in json.loads(capsys.readouterr().err)["error"]


def test_cli_supervise_maps_the_outcome_to_an_exit_code(tmp_path, capsys, monkeypatch):
    root, digest = _sealed(tmp_path)
    launched = []

    def fake_launcher(*args):
        launched.append(args)
        return lambda: pytest.fail("no child")

    monkeypatch.setattr(cli, "popen_launcher", fake_launcher)
    monkeypatch.setattr(
        cli,
        "supervise_collector",
        lambda *a, **k: dict(outcome="failed", exit_code=3),
    )
    args = [
        "supervise",
        "--root",
        str(root),
        "--protocol-sha256",
        digest,
        "--log-file",
        str(tmp_path / "child.log"),
    ]
    assert cli.main(args) == 3
    assert launched == [(root, digest, tmp_path / "child.log")]


def test_cli_supervise_refuses_a_log_inside_the_sealed_root(tmp_path, capsys):
    root, digest = _sealed(tmp_path)
    args = [
        "supervise",
        "--root",
        str(root),
        "--protocol-sha256",
        digest,
        "--log-file",
        str(root / "child.log"),
    ]
    assert cli.main(args) == 2
    assert "outside the sealed root" in json.loads(capsys.readouterr().err)["error"]


def test_cli_run_reports_signal_termination_as_exit_three(
    tmp_path, capsys, monkeypatch
):
    from contextlib import nullcontext

    from trade_rl.evaluation.paper.watchdog import CollectorTerminated

    monkeypatch.setattr(
        cli, "PaperCollector", lambda *args, **kwargs: nullcontext(object())
    )
    monkeypatch.setattr(cli, "install_termination_handlers", lambda *_: [])

    def terminated(*args, **kwargs):
        raise CollectorTerminated("terminated by signal 15")

    monkeypatch.setattr(cli, "collect_until_finished", terminated)
    code = cli.main(["run", "--root", str(tmp_path), "--protocol-sha256", "a" * 64])
    assert code == 3
    assert json.loads(capsys.readouterr().err)["error_type"] == "CollectorTerminated"


def test_cli_run_wires_heartbeat_and_disarms_the_exit_guard(tmp_path, monkeypatch):
    import atexit
    from contextlib import nullcontext

    registered = []
    monkeypatch.setattr(atexit, "register", registered.append)
    monkeypatch.setattr(atexit, "unregister", registered.remove)
    monkeypatch.setattr(
        cli, "PaperCollector", lambda *args, **kwargs: nullcontext(object())
    )
    hooks = []

    def install(hook):
        hooks.append(hook)
        return []

    monkeypatch.setattr(cli, "install_termination_handlers", install)
    seen = {}

    def finished(collector, **kwargs):
        seen.update(kwargs)
        assert len(registered) == 1 and registered[0].armed
        return dict(
            phase="finished",
            status=dict(terminal_flat=True, quality_failures=[], pending=False),
        )

    monkeypatch.setattr(cli, "collect_until_finished", finished)
    code = cli.main(["run", "--root", str(tmp_path), "--protocol-sha256", "a" * 64])
    assert code == 0 and registered == []
    assert callable(seen["heartbeat"]) and seen["heartbeat_interval_seconds"] == 15
    assert hooks and callable(hooks[0])
