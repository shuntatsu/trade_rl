from __future__ import annotations

import sys
from io import StringIO
from types import ModuleType

import pytest

from trade_rl.cli import build_parser, main


_COMMON_ARGS = [
    "--plan",
    "plan.json",
    "--manifest",
    "manifest.json",
    "--execution-store",
    "execution-store",
    "--baseline-config-digest",
    "a" * 64,
    "--output-root",
    "output",
]


def test_parser_exposes_stage_a_commands() -> None:
    parser = build_parser()

    validation = parser.parse_args(["stage-a", "validation", *_COMMON_ARGS])
    assert validation.stage_a_command == "validation"

    sealed = parser.parse_args(
        [
            "stage-a",
            "sealed-test",
            *_COMMON_ARGS,
            "--validation-package",
            "output/validation",
            "--database-url",
            "postgresql://example",
        ]
    )
    assert sealed.stage_a_command == "sealed-test"

    complete = parser.parse_args(
        [
            "stage-a",
            "run",
            *_COMMON_ARGS,
            "--database-url",
            "postgresql://example",
        ]
    )
    assert complete.stage_a_command == "run"


def test_top_level_cli_routes_stage_a_without_importing_application(monkeypatch) -> None:
    calls: list[tuple[list[str], object, object]] = []
    fake = ModuleType("trade_rl.cli.stage_a")

    def fake_main(argv, *, stdout, stderr):
        calls.append((list(argv), stdout, stderr))
        return 17

    fake.main = fake_main  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "trade_rl.cli.stage_a", fake)
    stdout = StringIO()
    stderr = StringIO()

    assert main(["stage-a", "validation"], stdout=stdout, stderr=stderr) == 17
    assert calls == [(["validation"], stdout, stderr)]


def test_stage_a_subcommands_require_common_identity_inputs() -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["stage-a", "validation"])
