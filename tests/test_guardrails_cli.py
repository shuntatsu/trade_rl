"""Tests for guardrails CLI entry point."""

import json

from mars_lite.trading.guardrails import main


def test_guardrails_cli_flatten(capsys):
    ret = main(["--action", "flatten", "--reason", "emergency stop", "--output-format", "json"])
    assert ret == 0
    out, _ = capsys.readouterr()
    data = json.loads(out.strip())
    assert data["action"] == "flatten"
    assert data["scale"] == 0.0
    assert "emergency stop" in data["triggered"]


def test_guardrails_cli_scale(capsys):
    ret = main(["--action", "scale", "--scale", "0.5", "--output-format", "json"])
    assert ret == 0
    out, _ = capsys.readouterr()
    data = json.loads(out.strip())
    assert data["action"] == "scale"
    assert data["scale"] == 0.5


def test_guardrails_cli_text_format(capsys):
    ret = main(["--action", "flatten", "--output-format", "text"])
    assert ret == 0
    out, _ = capsys.readouterr()
    assert "Action: FLATTEN" in out
