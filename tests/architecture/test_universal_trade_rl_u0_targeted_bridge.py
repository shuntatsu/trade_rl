from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_universal_trade_rl_u0_documentation_matches_phase_firewall() -> None:
    root = Path(__file__).resolve().parents[2]
    documentation = (root / "docs/UNIVERSAL_TRADE_RL.md").read_text(encoding="utf-8")

    assert "TRAIN\n  evaluate = none\n  fit      = Train" in documentation
    assert "DEVELOPMENT\n  evaluate = Development\n  fit      = Train" in documentation
    assert "ADMISSION\n  evaluate = Admission\n  fit      = none" in documentation
    assert "Admissionを開いた後でも`fit_symbols`は空" in documentation
    assert "evaluate = Train + Development" not in documentation
    assert "fit_symbols`はTrainだけ" not in documentation


def test_universal_trade_rl_u0_targeted_suite() -> None:
    root = Path(__file__).resolve().parents[2]
    candidates = (
        root / "tests/domain/test_universal_trade_rl_universe.py",
        root / "tests/workflows/test_universal_trade_rl_universe_config.py",
        root / "tests/workflows/test_universal_trade_rl_universe_manifest.py",
        root / "tests/workflows/test_universal_trade_rl_universe_access.py",
        root / "tests/workflows/test_universal_trade_rl_data_provenance.py",
        root / "tests/workflows/test_universal_trade_rl_run_identity.py",
        root / "tests/workflows/test_universal_trade_rl_universe_runner.py",
        root / "tests/workflows/test_universal_trade_rl_universe_isolation.py",
    )
    tests = [str(path.relative_to(root)) for path in candidates if path.is_file()]
    assert tests, "U0 targeted test bridge requires at least one U0 test file"

    completed = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests],
        cwd=root,
        check=False,
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, (
        "U0 targeted suite failed:\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )
