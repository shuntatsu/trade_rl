from __future__ import annotations

from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
MODULE = PACKAGE / "simulation" / "depth_calibration.py"


def test_depth_calibration_is_pure_and_imports_no_upper_layer() -> None:
    imports = ImportCollector(PACKAGE).collect(MODULE)
    internal = {name for name in imports if within_module(name, "trade_rl")}
    assert internal <= {"trade_rl._validation", "trade_rl.artifacts.hashing"}
    for forbidden in (
        "trade_rl.integrations",
        "trade_rl.data",
        "trade_rl.strategies",
        "trade_rl.evaluation",
        "trade_rl.risk",
        "numpy",
        "psycopg",
        "requests",
        "urllib",
        "socket",
    ):
        assert not any(within_module(name, forbidden) for name in imports), forbidden


def test_depth_calibration_is_not_reexported_by_the_simulation_facade() -> None:
    facade = (PACKAGE / "simulation" / "__init__.py").read_text(encoding="utf-8")
    assert "depth_calibration" not in facade


def test_depth_calibration_uses_no_float_arithmetic_or_division_by_input() -> None:
    source = MODULE.read_text(encoding="utf-8")
    assert "import numpy" not in source
    assert "float(" not in source
    assert "math.exp" not in source and "** " not in source.replace("10**", "")
