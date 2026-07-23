from __future__ import annotations

import ast
import importlib.util
import inspect
import tomllib
from pathlib import Path

from trade_rl.rl.environment import ResidualMarketEnv

ROOT = Path(__file__).resolve().parents[2]
REQUIRED_MODULES = {
    "trade_rl.rl.environment_dependencies": "EnvironmentDependencyResolver",
    "trade_rl.rl.environment_observation_contract": (
        "EnvironmentObservationContractFactory"
    ),
    "trade_rl.rl.environment_assembly": "EnvironmentServiceAssembler",
    "trade_rl.rl.environment_state": "EnvironmentInitialStateFactory",
}
CONSTRUCTION_PATHS = (
    "trade_rl/rl/environment_dependencies.py",
    "trade_rl/rl/environment_observation_contract.py",
    "trade_rl/rl/environment_assembly.py",
    "trade_rl/rl/environment_state.py",
)
FORBIDDEN_CONSTRUCTOR_SYMBOLS = {
    "BookState.zero",
    "CausalEmergencyRiskMonitor",
    "EnvironmentDecisionPlanner",
    "EnvironmentExecutionCoordinator",
    "EnvironmentInfoBuilder",
    "EnvironmentObservationAssembler",
    "EnvironmentRewardCoordinator",
    "EnvironmentRiskProjector",
    "EnvironmentTerminationCoordinator",
    "EpisodeContractSampler",
    "MarketExecutor",
    "ObservationBuilder",
    "SequenceObservationBuilder",
    "build_sequence_policy_plane",
}


def _constructor_source() -> str:
    return inspect.getsource(ResidualMarketEnv.__init__)


def _module_source(module_name: str) -> str:
    path = ROOT / (module_name.replace(".", "/") + ".py")
    return path.read_text(encoding="utf-8")


def test_environment_construction_modules_exist_with_typed_owners() -> None:
    for module_name, owner_name in REQUIRED_MODULES.items():
        assert importlib.util.find_spec(module_name) is not None, module_name
        source = _module_source(module_name)
        assert "from trade_rl.rl.environment import" not in source, module_name
        tree = ast.parse(source)
        owners = {
            node.name
            for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef))
        }
        assert owner_name in owners, module_name
        assert any(
            isinstance(node, ast.ClassDef)
            and any(
                isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Name)
                and decorator.func.id == "dataclass"
                and any(
                    keyword.arg == "frozen"
                    and isinstance(keyword.value, ast.Constant)
                    and keyword.value.value is True
                    for keyword in decorator.keywords
                )
                for decorator in node.decorator_list
            )
            for node in tree.body
        ), module_name


def test_environment_constructor_is_bounded_orchestration() -> None:
    lines, _ = inspect.getsourcelines(ResidualMarketEnv.__init__)
    source = "".join(lines)

    assert len(lines) <= 180
    for delegation in (
        "EnvironmentDependencyResolver.resolve",
        "EnvironmentObservationContractFactory.build",
        "EnvironmentServiceAssembler.assemble",
        "EnvironmentInitialStateFactory.create",
    ):
        assert delegation in source
    for symbol in FORBIDDEN_CONSTRUCTOR_SYMBOLS:
        assert symbol not in source


def test_environment_facade_keeps_public_identity_and_mutable_state() -> None:
    assert ResidualMarketEnv.__module__ == "trade_rl.rl.environment"
    source = _constructor_source()
    for field in (
        "self.hybrid",
        "self.shadow",
        "self._hybrid_order_book",
        "self._shadow_order_book",
        "self._previous_action",
        "self._pending_hybrid_target",
        "self._pending_shadow_target",
        "self._execution_state",
        "self._action_diagnostics",
        "self._has_reset",
    ):
        assert field in source
    assert "self.__dict__.update" not in source


def test_environment_construction_coverage_tracks_behavior_owners() -> None:
    configuration = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    group = configuration["tool"]["trade_rl"]["critical_coverage"]["groups"][
        "environment_construction"
    ]

    assert group["minimum"] == 64.0
    assert tuple(group["paths"]) == CONSTRUCTION_PATHS
