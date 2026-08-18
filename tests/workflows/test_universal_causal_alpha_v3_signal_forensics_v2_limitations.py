from __future__ import annotations

from pathlib import Path

from tests.workflows.test_universal_causal_alpha_v3_signal_forensics_v2_sidecars import (
    _complete_sidecars,
)
from trade_rl.workflows.universal_causal_alpha_v3_signal_forensics_v2 import (
    load_causal_alpha_v3_signal_forensics_v2,
)


def test_v2_complete_report_disclaims_availability_error_causality(
    tmp_path: Path,
) -> None:
    _complete_sidecars(tmp_path)

    report = load_causal_alpha_v3_signal_forensics_v2(tmp_path)

    assert "availability_error_causal_attribution" in {
        item.analysis for item in report.unavailable_analyses
    }
