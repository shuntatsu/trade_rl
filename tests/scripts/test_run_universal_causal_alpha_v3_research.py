from __future__ import annotations

import pytest

from scripts.run_universal_causal_alpha_v3_research import _parser


def test_v3_research_parser_requires_artifact_bound_runtime_inputs() -> None:
    parser = _parser()
    help_text = parser.format_help()

    assert "--runtime-manifest" in help_text
    assert "--frozen-metadata-root" in help_text
    assert "--run-config" in help_text
    assert "--output-root" in help_text
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_v3_research_parser_limits_stage_order() -> None:
    parser = _parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--runtime-manifest",
                "manifest.json",
                "--frozen-metadata-root",
                "metadata",
                "--run-config",
                "config.json",
                "--output-root",
                "output",
                "--stage-limit",
                "rl-before-admission",
            ]
        )
