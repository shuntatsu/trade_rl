from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
RUNS = PACKAGE / "evaluation" / "runs"
V1_OWNERS = (
    "trade_rl.evaluation.runs.portfolio_family_attempt",
    "trade_rl.evaluation.runs.portfolio_family_result_artifact",
)
V2_OWNERS = (
    "trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2",
    "trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2",
)
FORBIDDEN_IMPORTS = (
    "trade_rl.research",
    "trade_rl.strategies.rl.portfolio_training",
)


def test_scenario_attempt_v2_owners_stay_private_and_outside_research() -> None:
    collector = ImportCollector(PACKAGE)
    for module, relative in zip(
        V2_OWNERS,
        (
            "portfolio_family_scenario_attempt_v2.py",
            "portfolio_family_scenario_result_artifact_v2.py",
        ),
        strict=True,
    ):
        imports = collector.collect_direct(RUNS / relative)
        assert not {
            imported
            for imported in imports
            if any(within_module(imported, owner) for owner in FORBIDDEN_IMPORTS)
        }, module
    for owner, relative in zip(
        V1_OWNERS,
        ("portfolio_family_attempt.py", "portfolio_family_result_artifact.py"),
        strict=True,
    ):
        imports = collector.collect_direct(RUNS / relative)
        assert not {
            imported
            for imported in imports
            if any(within_module(imported, v2) for v2 in V2_OWNERS)
        }, owner


def test_v2_durable_owners_are_not_facade_exports_or_eager_rl_imports() -> None:
    script = (
        "import sys; import trade_rl.evaluation.runs as runs; "
        "import trade_rl.evaluation.runs.portfolio_family_scenario_attempt_v2 as attempt; "
        "import trade_rl.evaluation.runs.portfolio_family_scenario_result_artifact_v2 as result; "
        "assert not hasattr(runs, 'run_four_family_portfolio_scenario_attempt_v2'); "
        "assert not hasattr(runs, 'load_four_family_portfolio_scenario_result_artifact_v2'); "
        "assert attempt.__name__.endswith('portfolio_family_scenario_attempt_v2'); "
        "assert result.__name__.endswith('portfolio_family_scenario_result_artifact_v2'); "
        "assert 'torch' not in sys.modules; "
        "assert not any(name == 'stable_baselines3' or "
        "name.startswith('stable_baselines3.') for name in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-B", "-c", script],
        cwd=ROOT,
        capture_output=True,
        check=False,
        text=True,
    )
    assert result.returncode == 0, result.stderr
