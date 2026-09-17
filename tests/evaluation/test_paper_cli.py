import json
from datetime import timedelta

from tests.integrations.test_binance_forward import NOW
from trade_rl.evaluation.paper import cli
from trade_rl.evaluation.paper.operations import seal_paper_study


def test_cli_status_is_machine_readable_and_explicitly_not_economic_evidence(
    tmp_path, capsys
):
    root = tmp_path / "study"
    digest = seal_paper_study(
        root, start_at=NOW + timedelta(minutes=5), clock=lambda: NOW
    )
    assert cli.main(["status", "--root", str(root), "--protocol-sha256", digest]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["tip"] == digest and output["events"] == 0
    assert output["economic_decision"] == "NOT_EVALUATED"
    assert output["production_eligible"] is False


def test_cli_wrong_anchor_fails_and_never_replaces_existing_output(tmp_path, capsys):
    root = tmp_path / "study"
    seal_paper_study(root, start_at=NOW + timedelta(minutes=5), clock=lambda: NOW)
    assert cli.main(["status", "--root", str(root), "--protocol-sha256", "a" * 64]) == 2
    assert "digest" in json.loads(capsys.readouterr().err)["error"]


def test_cli_seal_publishes_the_external_anchor_and_refuses_root_reuse(
    tmp_path, capsys
):
    from datetime import UTC, datetime

    root = tmp_path / "study"
    args = [
        "seal",
        "--root",
        str(root),
        "--start-at",
        (datetime.now(UTC) + timedelta(minutes=10)).isoformat(),
    ]
    assert cli.main(args) == 0
    output = json.loads(capsys.readouterr().out)
    assert len(output["protocol_sha256"]) == 64
    assert output["production_eligible"] is False
    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().err)["error_type"] == "FileExistsError"


def test_terminal_window_alone_is_not_success_when_positions_remain(
    tmp_path, monkeypatch
):
    from contextlib import nullcontext

    monkeypatch.setattr(
        cli, "PaperCollector", lambda *args, **kwargs: nullcontext(object())
    )
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
