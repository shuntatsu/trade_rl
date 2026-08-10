from __future__ import annotations

import ast
import importlib
from pathlib import Path

from tests.architecture.repository_paths import PYTHON_SOURCE_ROOT

PACKAGE_ROOT = PYTHON_SOURCE_ROOT
TRAINING_PATH = PACKAGE_ROOT / "integrations/sb3_training.py"
OWNER_PATHS = {
    "trade_rl.integrations.sb3_runtime": (
        PACKAGE_ROOT / "integrations/sb3_runtime.py"
    ),
    "trade_rl.integrations.sb3_environment": (
        PACKAGE_ROOT / "integrations/sb3_environment.py"
    ),
    "trade_rl.integrations.sb3_behavior_cloning": (
        PACKAGE_ROOT / "integrations/sb3_behavior_cloning.py"
    ),
}

RUNTIME_HELPERS = frozenset(
    {
        "_lagrangian_probe_worker_count",
        "_oracle_solver_config",
        "_oracle_accelerator_backend",
        "_teacher_worker_count",
        "_oracle_episode_sampling_config",
        "_configure_torch_cuda_runtime",
        "_configure_sequence_runtime",
    }
)

ENVIRONMENT_HELPERS = frozenset(
    {
        "_HEAVY_TRAINING_INFO_KEYS",
        "_compact_training_info",
        "_TrainingInfoFilter",
        "_filtered_training_environment",
        "_build_training_environment",
        "_effective_vector_environment_kind",
        "_compact_filtered_training_environment",
        "_build_parallel_sequence_training_environment",
        "_reset_observation_for_export",
    }
)

BEHAVIOR_CLONING_HELPERS = frozenset(
    {
        "_teacher_cache_key",
        "_TeacherIdentity",
        "_behavior_cloning_quality",
        "_resolve_behavior_cloning_seed",
        "_save_behavior_cloning_policy_candidate",
        "_restore_member_seed_after_behavior_cloning",
        "_required_hierarchical_config",
        "_teacher_change_labels",
        "_uses_hierarchical_actor_head",
        "_hierarchical_teacher_labels",
        "_hierarchical_behavior_cloning_config",
        "_behavior_cloning_gate_thresholds",
        "_evaluate_hierarchical_behavior_cloning_gate",
        "_enforce_behavior_cloning_gates",
    }
)

HELPERS_BY_MODULE = {
    "trade_rl.integrations.sb3_runtime": RUNTIME_HELPERS,
    "trade_rl.integrations.sb3_environment": ENVIRONMENT_HELPERS,
    "trade_rl.integrations.sb3_behavior_cloning": BEHAVIOR_CLONING_HELPERS,
}
ALL_EXTRACTED_HELPERS = frozenset().union(*HELPERS_BY_MODULE.values())


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _defined_names(path: Path) -> frozenset[str]:
    names: set[str] = set()
    for node in _tree(path).body:
        if isinstance(node, ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(
                target.id for target in node.targets if isinstance(target, ast.Name)
            )
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
    return frozenset(names)


def _import_targets(path: Path) -> frozenset[str]:
    targets: set[str] = set()
    for node in ast.walk(_tree(path)):
        if isinstance(node, ast.Import):
            targets.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            targets.add(node.module)
    return frozenset(targets)


def test_sb3_helpers_are_owned_by_focused_modules() -> None:
    for module_name, helpers in HELPERS_BY_MODULE.items():
        path = OWNER_PATHS[module_name]
        assert path.is_file(), f"missing owner module: {module_name}"
        assert helpers <= _defined_names(path)

    assert not ALL_EXTRACTED_HELPERS & _defined_names(TRAINING_PATH)


def test_sb3_training_keeps_direct_compatibility_aliases() -> None:
    coordinator = importlib.import_module("trade_rl.integrations.sb3_training")
    for module_name, helpers in HELPERS_BY_MODULE.items():
        owner = importlib.import_module(module_name)
        for helper in helpers:
            assert getattr(coordinator, helper) is getattr(owner, helper)


def test_sb3_helper_modules_never_depend_on_training_coordinator() -> None:
    forbidden = "trade_rl.integrations.sb3_training"
    for module_name, path in OWNER_PATHS.items():
        assert path.is_file(), f"missing owner module: {module_name}"
        assert all(
            target != forbidden and not target.startswith(f"{forbidden}.")
            for target in _import_targets(path)
        )


def test_sb3_training_coordinator_stays_below_sixty_kibibytes() -> None:
    assert TRAINING_PATH.stat().st_size < 61_440
