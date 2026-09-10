from __future__ import annotations

import json
from pathlib import Path

import pytest

from trade_rl.evaluation.experiments.bootstrap import cli
from trade_rl.evaluation.experiments.bootstrap.workflow import CanonicalM2BootstrapResult


def _result(root: Path) -> CanonicalM2BootstrapResult:
    return CanonicalM2BootstrapResult(
        root=root,
        config_digest="1" * 64,
        bootstrap_digest="2" * 64,
        dataset_id="3" * 64,
        dataset_artifact_digest="4" * 64,
        study_digest="5" * 64,
    )


def test_cli_calls_bootstrap_once_and_prints_stable_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = tmp_path / "config.json"
    output = tmp_path / "out"
    calls: list[tuple[Path, Path]] = []

    def fake_bootstrap(config_path: str | Path, output_root: str | Path):
        calls.append((Path(config_path), Path(output_root)))
        return _result(Path(output_root))

    monkeypatch.setattr(cli, "bootstrap_canonical_m2_study", fake_bootstrap)

    assert cli.main(["--config", str(config), "--output", str(output)]) == 0
    assert calls == [(config, output)]
    assert json.loads(capsys.readouterr().out) == {
        "bootstrap_digest": "2" * 64,
        "config_digest": "1" * 64,
        "dataset_artifact_digest": "4" * 64,
        "dataset_id": "3" * 64,
        "root": str(output),
        "study_digest": "5" * 64,
    }


def test_cli_rejects_unknown_arguments() -> None:
    with pytest.raises(SystemExit) as error:
        cli.main(["--config", "config.json", "--output", "out", "--surprise"])
    assert error.value.code == 2


def test_cli_does_not_hide_bootstrap_failures(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(_config: str | Path, _output: str | Path):
        raise FileExistsError("already exists")

    monkeypatch.setattr(cli, "bootstrap_canonical_m2_study", fail)
    with pytest.raises(FileExistsError, match="already exists"):
        cli.main(["--config", "config.json", "--output", "out"])
