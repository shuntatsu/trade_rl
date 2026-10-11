"""Settled timing declarations belong below evaluation and the optional learner."""

from importlib import import_module
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"


def test_settled_clock_has_no_runtime_or_optional_learner_dependency():
    path = PACKAGE / "strategies/rl/allocation_settled_clock.py"
    assert path.is_file(), "the pure settled allocation clock owner is missing"
    imports = ImportCollector(PACKAGE).collect(path)
    assert all(
        not name.startswith("trade_rl.") or name.startswith("trade_rl.artifacts.")
        for name in imports
    )
    assert not any(
        name.split(".")[0] in {"numpy", "gymnasium", "torch", "stable_baselines3"}
        for name in imports
    )
    module = import_module("trade_rl.strategies.rl.allocation_settled_clock")
    assert module.__all__ == ["SettledAllocationClockContract"]
