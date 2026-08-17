from __future__ import annotations

import json
from pathlib import Path

from scripts import analyze_universal_causal_alpha_v3_signal as cli_module
from tests.workflows.test_universal_causal_alpha_v3_signal_forensics import _build_run
from tests.workflows.test_universal_causal_alpha_v3_signal_forensics_v2_sidecars import (
    _complete_sidecars,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics import (
    load_causal_alpha_v3_signal_forensics,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2 import (
    load_causal_alpha_v3_signal_forensics_v2,
)


def _stdout_payload(capsys) -> dict[str, object]:
    payload = json.loads(capsys.readouterr().out)
    assert isinstance(payload, dict)
    return payload


def test_cli_default_v1_matches_direct_historical_report(
    tmp_path: Path, capsys
) -> None:
    _build_run(tmp_path)
    expected = load_causal_alpha_v3_signal_forensics(tmp_path).to_payload()

    assert cli_module.main([str(tmp_path)]) == 0

    assert _stdout_payload(capsys) == expected


def test_cli_v2_historical_mode_matches_direct_report(tmp_path: Path, capsys) -> None:
    _build_run(tmp_path)
    expected = load_causal_alpha_v3_signal_forensics_v2(tmp_path).to_payload()

    assert cli_module.main([str(tmp_path), "--schema", "v2"]) == 0

    payload = _stdout_payload(capsys)
    assert payload == expected
    assert payload["schema_version"] == "causal_alpha_v3_signal_forensics_v2"
    assert payload["sidecar_mode"] == "historical_unavailable"


def test_cli_v2_complete_mode_matches_direct_sidecar_report(
    tmp_path: Path, capsys
) -> None:
    _complete_sidecars(tmp_path)
    expected = load_causal_alpha_v3_signal_forensics_v2(tmp_path).to_payload()

    assert cli_module.main([str(tmp_path), "--schema", "v2"]) == 0

    payload = _stdout_payload(capsys)
    assert payload == expected
    assert payload["schema_version"] == "causal_alpha_v3_signal_forensics_v2"
    assert payload["sidecar_mode"] == "sidecar_complete"
    assert payload["research_only"] is True
    assert payload["promotion_eligible"] is False
