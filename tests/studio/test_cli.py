from __future__ import annotations

from pathlib import Path

import pytest

from trade_rl.cli import main as trade_rl_main
from trade_rl.studio import cli


def test_studio_cli_binds_loopback_and_constructs_app(tmp_path: Path) -> None:
    calls: list[dict[str, object]] = []

    def runner(app, **kwargs):
        calls.append({"app": app, **kwargs})

    result = cli.main(
        ["start", "--project-root", str(tmp_path), "--port", "9876"],
        runner=runner,
    )

    assert result == 0
    assert calls[0]["host"] == "127.0.0.1"
    assert calls[0]["port"] == 9876
    assert calls[0]["app"].title == "Trade RL Studio API"


def test_studio_cli_rejects_remote_binding_without_explicit_override(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="--allow-remote"):
        cli.main(
            [
                "start",
                "--project-root",
                str(tmp_path),
                "--host",
                "0.0.0.0",
            ],
            runner=lambda *args, **kwargs: None,
        )


def test_top_level_cli_dispatches_studio_without_loading_full_research_parser(
    monkeypatch,
) -> None:
    captured: list[list[str]] = []

    def fake(arguments):
        captured.append(list(arguments))
        return 7

    monkeypatch.setattr("trade_rl.studio.cli.main", fake)

    result = trade_rl_main(["studio", "start"])

    assert result == 7
    assert captured == [["start"]]
