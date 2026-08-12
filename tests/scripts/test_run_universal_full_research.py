from __future__ import annotations

import pytest

from scripts.run_universal_full_research import _parser


def test_parser_requires_runtime_manifest_and_defaults_concrete_factory() -> None:
    parser = _parser()
    help_text = parser.format_help()
    assert "--runtime-manifest" in help_text
    factory_action = next(
        action for action in parser._actions if action.dest == "runtime_factory"
    )
    assert factory_action.default == (
        "trade_rl.workflows.binance_universal_runtime:build_runtime"
    )
    with pytest.raises(SystemExit):
        parser.parse_args([])
