from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENVIRONMENT = ROOT / "trade_rl" / "rl" / "environment.py"
ASSEMBLY = ROOT / "trade_rl" / "rl" / "environment_assembly.py"
SERVICE_PATHS = (
    ROOT / "trade_rl" / "rl" / "environment_decision.py",
    ROOT / "trade_rl" / "rl" / "environment_risk.py",
    ROOT / "trade_rl" / "rl" / "environment_reward.py",
    ROOT / "trade_rl" / "rl" / "environment_info.py",
)
SERVICE_ATTRIBUTES = (
    ("EnvironmentDecisionPlanner", "self._decision_planner"),
    ("EnvironmentRiskProjector", "self._risk_projector"),
    ("EnvironmentRewardCoordinator", "self._reward_coordinator"),
    ("EnvironmentInfoBuilder", "self._info_builder"),
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_environment_step_services_have_dedicated_modules() -> None:
    missing = [
        path.relative_to(ROOT).as_posix()
        for path in SERVICE_PATHS
        if not path.is_file()
    ]

    assert missing == []


def test_environment_assembly_constructs_all_step_services() -> None:
    environment_source = _source(ENVIRONMENT)
    assembly_source = _source(ASSEMBLY)

    assert "EnvironmentServiceAssembler.assemble(" in environment_source
    for symbol, attribute in SERVICE_ATTRIBUTES:
        assert symbol in assembly_source
        assert attribute in environment_source


def test_environment_step_delegates_action_risk_reward_and_info() -> None:
    source = _source(ENVIRONMENT)
    step_source = source.split("    def step(\n", maxsplit=1)[1]

    assert "self._decision_planner.plan(" in step_source
    assert "self._risk_projector.project(" in step_source
    assert "self._reward_coordinator.step(" in step_source
    assert "self._info_builder.step_info(" in step_source


def test_environment_step_no_longer_owns_extracted_policy_logic() -> None:
    source = _source(ENVIRONMENT)
    step_source = source.split("    def step(\n", maxsplit=1)[1]

    assert "self.composer.compose(" not in step_source
    assert "self.reward_tracker.step(" not in step_source
    assert "self.emergency_risk_monitor.assess(" not in step_source
    assert "info: dict[str, object] = {" not in step_source


def test_terminal_information_is_built_by_info_service() -> None:
    source = _source(ENVIRONMENT)

    assert "self._info_builder.terminal_info(" in source
    assert "def _book_metrics(" not in source
