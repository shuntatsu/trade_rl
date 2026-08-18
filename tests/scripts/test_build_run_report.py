from __future__ import annotations

import json
from pathlib import Path

from scripts import build_run_report as module
from trade_rl.reporting.run_report import (
    RUN_REPORT_STAGE_ORDER,
    RunReport,
    RunStageReport,
    RunStageStatus,
)


def _report(root: Path) -> RunReport:
    return RunReport(
        root=str(root),
        identities={"run_manifest_digest": "a" * 64},
        stages=tuple(
            RunStageReport(name=name, status=RunStageStatus.MISSING)
            for name in RUN_REPORT_STAGE_ORDER
        ),
    )


def test_cli_chat_profile_writes_markdown_to_stdout(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setattr(module, "build_run_report", lambda root: _report(Path(root)))

    exit_code = module.main(
        ["--root", str(tmp_path), "--profile", "chat", "--output", "-"]
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert output.startswith("# Machine Run Report\n")
    assert "| signal | MISSING |" in output


def test_cli_json_profile_writes_deterministic_payload_outside_source_root(
    monkeypatch, tmp_path: Path
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(module, "build_run_report", lambda root: _report(Path(root)))
    output = tmp_path / "report.json"

    exit_code = module.main(
        [
            "--root",
            str(run_root),
            "--profile",
            "json",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "run_report_v1"
    assert [stage["name"] for stage in payload["stages"]] == list(
        RUN_REPORT_STAGE_ORDER
    )


def test_cli_rejects_output_inside_source_root(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    run_root = tmp_path / "run"
    run_root.mkdir()
    monkeypatch.setattr(module, "build_run_report", lambda root: _report(Path(root)))
    output = run_root / "report.md"

    exit_code = module.main(
        ["--root", str(run_root), "--profile", "chat", "--output", str(output)]
    )

    assert exit_code == 2
    assert not output.exists()
    assert "outside" in capsys.readouterr().err.lower()


def test_cli_rejects_missing_root_without_creating_it(tmp_path: Path, capsys) -> None:
    missing = tmp_path / "does-not-exist"

    exit_code = module.main(["--root", str(missing), "--output", "-"])

    assert exit_code == 2
    assert not missing.exists()
    assert "does not exist" in capsys.readouterr().err.lower()
