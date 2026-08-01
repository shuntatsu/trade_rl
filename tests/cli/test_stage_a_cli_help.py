from __future__ import annotations

from io import StringIO

from trade_rl.cli import main


def test_top_level_help_lists_stage_a() -> None:
    stdout = StringIO()

    assert main(["--help"], stdout=stdout) == 0
    assert "stage-a" in stdout.getvalue()


def test_empty_arguments_print_authoritative_help() -> None:
    stdout = StringIO()

    assert main([], stdout=stdout) == 2
    assert "stage-a" in stdout.getvalue()
