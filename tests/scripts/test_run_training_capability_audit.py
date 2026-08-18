from __future__ import annotations

import json
import sys
from pathlib import Path

from scripts import run_training_capability_audit as cli


def test_main_delegates_to_public_operation_and_preserves_stdout(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    output = tmp_path / "audit"
    report = {
        "digest": "d" * 64,
        "schema_version": "full_training_capability_audit_v1",
    }
    observed: list[Path] = []

    def run(path: Path) -> dict[str, object]:
        observed.append(path)
        return report

    monkeypatch.setattr(cli, "run_training_capability_audit", run)
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_training_capability_audit.py", "--output", str(output)],
    )

    assert cli.main() == 0
    assert observed == [output]
    assert capsys.readouterr().out == json.dumps(report, sort_keys=True) + "\n"


def test_main_preserves_default_output(monkeypatch, capsys) -> None:
    observed: list[Path] = []

    def run(path: Path) -> dict[str, object]:
        observed.append(path)
        return {"schema_version": "full_training_capability_audit_v1"}

    monkeypatch.setattr(cli, "run_training_capability_audit", run)
    monkeypatch.setattr(sys, "argv", ["run_training_capability_audit.py"])

    assert cli.main() == 0
    assert observed == [Path("var/training-capability-audit")]
    capsys.readouterr()
