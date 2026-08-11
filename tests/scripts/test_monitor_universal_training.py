from __future__ import annotations

from scripts.monitor_universal_training import _parser


def test_monitor_cli_requires_generation_and_output_roots() -> None:
    help_text = _parser().format_help()
    assert "--generation-root" in help_text
    assert "--output-root" in help_text
    assert "--container" in help_text
