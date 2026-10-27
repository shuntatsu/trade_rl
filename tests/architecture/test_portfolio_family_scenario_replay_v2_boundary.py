from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from tools.agent_repo.source_index import ImportCollector, within_module

ROOT = Path(__file__).resolve().parents[2]
PACKAGE = ROOT / "trade_rl"
OWNER = PACKAGE / "evaluation" / "runs" / "portfolio_family_scenario_replay_v2.py"
FORBIDDEN_IMPORTS = (
    "trade_rl.evaluation.runs.portfolio_family_attempt",
    "trade_rl.evaluation.runs.portfolio_family_result_artifact",
    "trade_rl.research",
    "trade_rl.strategies.rl.portfolio_training",
)


def test_v2_scenario_replay_stays_outside_research_training_and_durable_owners() -> (
    None
):
    imports = ImportCollector(PACKAGE).collect_direct(OWNER)
    assert not {
        imported
        for imported in imports
        if any(within_module(imported, owner) for owner in FORBIDDEN_IMPORTS)
    }


def test_v2_scenario_replay_owner_is_not_added_to_the_runs_facade() -> None:
    script = (
        "import sys; import trade_rl.evaluation.runs as runs; "
        "import trade_rl.evaluation.runs.portfolio_family_scenario_replay_v2 as owner; "
        "assert not hasattr(runs, 'run_four_family_shared_portfolio_scenario_replay_v2'); "
        "assert not hasattr(runs, 'FourFamilyPortfolioScenarioReplayV2Result'); "
        "assert owner.__name__.endswith('portfolio_family_scenario_replay_v2'); "
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
