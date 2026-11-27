"""The study-specific calibration harness belongs to the research repository."""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

from trade_rl.evaluation.signal_measurement import ewc_mean_test

ROOT = Path(__file__).resolve().parents[2]
HISTORY = "fb126a9a9b1a7f337041548ee92a27e8230e02f3"
RETIRED = {
    "tools/issue1146_significance_calibration.py": "2b1e46f69045bd46607e02c52f79e5aa45534baf236d8dc8f660d9e1f8740946",
    "tests/tools/test_issue1146_significance_calibration.py": "7b0167ddcd65d1bfda72a920de347a9053c8c341aed5640c2a9c3704241cc453",
    "tests/evaluation/test_signal_measurement_calibration.py": "1f9af42a998521fa0d648f9e6991bfc3efed6ef4013748cc2d04b832ea0daa29",
}


def test_study_harness_and_duplicate_tests_are_retired() -> None:
    for path in RETIRED:
        assert not (ROOT / path).exists(), path


def test_original_calibration_evidence_is_byte_preserved() -> None:
    for path, expected in RETIRED.items():
        raw = subprocess.check_output(["git", "show", f"{HISTORY}:{path}"], cwd=ROOT)
        assert hashlib.sha256(raw).hexdigest() == expected, path


def test_core_keeps_the_measurement_contract() -> None:
    assert (
        ewc_mean_test.__module__
        == "trade_rl.evaluation.signal_measurement.significance"
    )
    assert (ROOT / "tests/evaluation/test_signal_measurement_significance.py").is_file()
