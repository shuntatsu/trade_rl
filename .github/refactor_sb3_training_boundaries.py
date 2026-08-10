from __future__ import annotations

from pathlib import Path
from textwrap import dedent


TRAINING_PATH = Path("trade_rl/integrations/sb3_training.py")


def _replace_exact(
    path: Path,
    old: str,
    new: str,
    *,
    expected_count: int = 1,
) -> None:
    source = path.read_text(encoding="utf-8")
    actual_count = source.count(old)
    if actual_count != expected_count:
        raise RuntimeError(
            f"{path}: expected {expected_count} occurrences, found {actual_count}: {old!r}"
        )
    path.write_text(source.replace(old, new), encoding="utf-8")


def _extract_modules() -> None:
    source = TRAINING_PATH.read_text(encoding="utf-8")
    markers = {
        "runtime_start": "\ndef _lagrangian_probe_worker_count(",
        "bc_first_start": "\ndef _teacher_cache_key(",
        "environment_first_start": "\n_HEAVY_TRAINING_INFO_KEYS = (",
        "bc_second_start": "\ndef _required_hierarchical_config(",
        "environment_second_start": "\ndef _build_training_environment(",
        "backend_start": "\nclass StableBaselines3Backend:",
    }
    for name, marker in markers.items():
        count = source.count(marker)
        if count != 1:
            raise RuntimeError(f"expected one {name} marker, found {count}")

    positions = {name: source.index(marker) for name, marker in markers.items()}
    teacher_methods_marker = "\n    def _oracle_episode_batch("
    if source.count(teacher_methods_marker) != 1:
        raise RuntimeError("teacher pipeline start marker changed")
    positions["teacher_methods_start"] = source.index(
        teacher_methods_marker,
        positions["backend_start"],
    )
    positions["train_method_start"] = source.index(
        "\n    def train(",
        positions["teacher_methods_start"],
    )

    ordered = [
        positions["runtime_start"],
        positions["bc_first_start"],
        positions["environment_first_start"],
        positions["bc_second_start"],
        positions["environment_second_start"],
        positions["backend_start"],
        positions["teacher_methods_start"],
        positions["train_method_start"],
    ]
    if ordered != sorted(ordered):
        raise RuntimeError("SB3 training extraction markers are out of order")

    def block(start_name: str, end_name: str) -> str:
        return (
            source[positions[start_name] : positions[end_name]].lstrip("\n").rstrip()
            + "\n"
        )

    runtime_block = block("runtime_start", "bc_first_start")
    bc_first_block = block("bc_first_start", "environment_first_start")
    environment_first_block = block("environment_first_start", "bc_second_start")
    bc_second_block = block("bc_second_start", "environment_second_start")
    environment_second_block = block("environment_second_start", "backend_start")
    teacher_methods_block = block("teacher_methods_start", "train_method_start")

    runtime_header = dedent(
        '''
        """Runtime and resource policy for Stable-Baselines3 training."""

        from __future__ import annotations

        import os
        from typing import Any, cast

        import numpy as np

        from trade_rl.learning.episode_oracle_bc import oracle_episode_sampling_config
        from trade_rl.learning.episode_oracle_teacher import OracleEpisodeSamplingConfig
        from trade_rl.learning.oracle_bellman_contracts import (
            CompileMode,
            OracleSolverConfig,
            SolverSelection,
        )
        from trade_rl.learning.oracle_solver import OracleBatchBackend
        from trade_rl.rl.training import ResidualTrainingConfig
        from trade_rl.rl.training_modes import CudaRuntimeMode

        '''
    ).lstrip()
    environment_header = dedent(
        '''
        """Training environment adapters for Stable-Baselines3."""

        from __future__ import annotations

        from collections.abc import Callable, Mapping
        from functools import partial
        from typing import Any

        import gymnasium as gym
        import numpy as np

        from trade_rl.rl.training import ResidualTrainingConfig

        '''
    ).lstrip()
    behavior_cloning_header = dedent(
        '''
        """Behavior-cloning helpers used by Stable-Baselines3 orchestration."""

        from __future__ import annotations

        from collections.abc import Mapping
        from dataclasses import dataclass
        from pathlib import Path
        from typing import Any

        import numpy as np

        from trade_rl.artifacts.hashing import content_digest
        from trade_rl.artifacts.verified_file import file_digest
        from trade_rl.learning import (
            BehaviorCloningConfig,
            BehaviorCloningGateEvaluation,
            BehaviorCloningGateThresholds,
            SupervisedPolicyDataset,
            evaluate_behavior_cloning_gates,
        )
        from trade_rl.learning.hierarchical_teacher_labels import (
            HierarchicalTeacherLabels,
            build_hierarchical_teacher_labels,
        )
        from trade_rl.rl.checkpointing import save_policy_without_runtime_state
        from trade_rl.rl.training import ResidualTrainingConfig

        '''
    ).lstrip()
    teacher_pipeline_header = dedent(
        '''
        """Teacher generation, caching, and artifact lifecycle for SB3 training."""

        from __future__ import annotations

        import shutil
        import tempfile
        from collections.abc import Callable
        from pathlib import Path
        from typing import Any

        import numpy as np

        from trade_rl.artifacts.hashing import content_digest
        from trade_rl.catalog.contracts import ArtifactKind
        from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
        from trade_rl.integrations.sb3_behavior_cloning import (
            _TeacherIdentity,
            _teacher_cache_key,
        )
        from trade_rl.integrations.sb3_runtime import _oracle_accelerator_backend
        from trade_rl.learning import (
            OracleTeacherConfig,
            SupervisedPolicyDataset,
            collect_teacher_rollout,
            load_teacher_artifact,
            oracle_target_path,
            write_teacher_artifact,
        )
        from trade_rl.learning.episode_oracle_bc import resolve_episode_initial_weights
        from trade_rl.learning.episode_oracle_teacher import (
            EpisodeOracleBatch,
            OracleEpisodeSamplingConfig,
            build_episode_oracle_batch,
        )
        from trade_rl.learning.episode_teacher_artifact import (
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
            EpisodeSupervisedPolicyDataset,
            collect_episode_teacher_rollout,
            collect_episode_teacher_rollout_parallel,
            load_episode_teacher_artifact,
            write_episode_teacher_artifact,
        )
        from trade_rl.learning.oracle_bellman_contracts import OracleSolverConfig
        from trade_rl.learning.teacher_cache import (
            teacher_cache_identity,
            teacher_cache_identity_v2,
        )


        class _StableBaselines3TeacherPipeline:
            """Own teacher computation and immutable cache publication."""

            environment_factory: Callable[[], Any]
            teacher_cache_root: Path | None
            reusable_artifact_index: ReusableArtifactIndex | None
            _oracle_target_cache: dict[tuple[str, int, int, str], np.ndarray]
            _oracle_episode_batch_cache: dict[
                tuple[str, int, int, str, str, str], EpisodeOracleBatch
            ]
            _trend_target_cache: dict[tuple[str, int, int, str], np.ndarray]
            _teacher_dataset_cache: dict[
                tuple[str, int, int, str, str, str], SupervisedPolicyDataset
            ]
            _episode_teacher_dataset_cache: dict[
                tuple[str, int, int, str, str, str],
                EpisodeSupervisedPolicyDataset,
            ]

        '''
    ).lstrip()

    Path("trade_rl/integrations/sb3_runtime.py").write_text(
        runtime_header + runtime_block,
        encoding="utf-8",
    )
    Path("trade_rl/integrations/sb3_environment.py").write_text(
        environment_header + environment_first_block + "\n" + environment_second_block,
        encoding="utf-8",
    )
    Path("trade_rl/integrations/sb3_behavior_cloning.py").write_text(
        behavior_cloning_header + bc_first_block + "\n" + bc_second_block,
        encoding="utf-8",
    )
    Path("trade_rl/integrations/sb3_teacher_pipeline.py").write_text(
        teacher_pipeline_header + teacher_methods_block,
        encoding="utf-8",
    )

    ranges = [
        (positions["runtime_start"], positions["bc_first_start"]),
        (positions["bc_first_start"], positions["environment_first_start"]),
        (positions["environment_first_start"], positions["bc_second_start"]),
        (positions["bc_second_start"], positions["environment_second_start"]),
        (positions["environment_second_start"], positions["backend_start"]),
        (positions["teacher_methods_start"], positions["train_method_start"]),
    ]
    reduced = source
    for start, end in reversed(ranges):
        reduced = reduced[:start] + reduced[end:]

    compatibility_imports = dedent(
        '''
        from trade_rl.integrations.sb3_behavior_cloning import (
            _TeacherIdentity as _TeacherIdentity,
            _behavior_cloning_gate_thresholds as _behavior_cloning_gate_thresholds,
            _behavior_cloning_quality as _behavior_cloning_quality,
            _enforce_behavior_cloning_gates as _enforce_behavior_cloning_gates,
            _evaluate_hierarchical_behavior_cloning_gate as _evaluate_hierarchical_behavior_cloning_gate,
            _hierarchical_behavior_cloning_config as _hierarchical_behavior_cloning_config,
            _hierarchical_teacher_labels as _hierarchical_teacher_labels,
            _required_hierarchical_config as _required_hierarchical_config,
            _resolve_behavior_cloning_seed as _resolve_behavior_cloning_seed,
            _restore_member_seed_after_behavior_cloning as _restore_member_seed_after_behavior_cloning,
            _save_behavior_cloning_policy_candidate as _save_behavior_cloning_policy_candidate,
            _teacher_cache_key as _teacher_cache_key,
            _teacher_change_labels as _teacher_change_labels,
            _uses_hierarchical_actor_head as _uses_hierarchical_actor_head,
        )
        from trade_rl.integrations.sb3_environment import (
            _HEAVY_TRAINING_INFO_KEYS as _HEAVY_TRAINING_INFO_KEYS,
            _TrainingInfoFilter as _TrainingInfoFilter,
            _build_parallel_sequence_training_environment as _build_parallel_sequence_training_environment,
            _build_training_environment as _build_training_environment,
            _compact_filtered_training_environment as _compact_filtered_training_environment,
            _compact_training_info as _compact_training_info,
            _effective_vector_environment_kind as _effective_vector_environment_kind,
            _filtered_training_environment as _filtered_training_environment,
            _reset_observation_for_export as _reset_observation_for_export,
        )
        from trade_rl.integrations.sb3_runtime import (
            _configure_sequence_runtime as _configure_sequence_runtime,
            _configure_torch_cuda_runtime as _configure_torch_cuda_runtime,
            _lagrangian_probe_worker_count as _lagrangian_probe_worker_count,
            _oracle_accelerator_backend as _oracle_accelerator_backend,
            _oracle_episode_sampling_config as _oracle_episode_sampling_config,
            _oracle_solver_config as _oracle_solver_config,
            _teacher_worker_count as _teacher_worker_count,
        )
        from trade_rl.integrations.sb3_teacher_pipeline import (
            _StableBaselines3TeacherPipeline,
        )
        '''
    ).lstrip()
    insertion_marker = "from trade_rl.integrations.behavior_cloning import pretrain_policy\n"
    if reduced.count(insertion_marker) != 1:
        raise RuntimeError("behavior-cloning import insertion marker changed")
    reduced = reduced.replace(
        insertion_marker,
        insertion_marker + compatibility_imports,
        1,
    )
    backend_marker = "class StableBaselines3Backend:"
    if reduced.count(backend_marker) != 1:
        raise RuntimeError("StableBaselines3Backend marker changed")
    reduced = reduced.replace(
        backend_marker,
        "class StableBaselines3Backend(_StableBaselines3TeacherPipeline):",
        1,
    )
    TRAINING_PATH.write_text(reduced, encoding="utf-8")

    expected_blocks = {
        "runtime": (
            runtime_block,
            {
                "_lagrangian_probe_worker_count",
                "_oracle_solver_config",
                "_oracle_accelerator_backend",
                "_teacher_worker_count",
                "_oracle_episode_sampling_config",
                "_configure_torch_cuda_runtime",
                "_configure_sequence_runtime",
            },
        ),
        "environment": (
            environment_first_block + environment_second_block,
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
            },
        ),
        "behavior cloning": (
            bc_first_block + bc_second_block,
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
            },
        ),
    }
    for label, (extracted, expected_names) in expected_blocks.items():
        for name in expected_names:
            if extracted.count(name) < 1:
                raise RuntimeError(f"{label} extraction omitted {name}")
    for name in {
        "_oracle_episode_batch",
        "_episode_teacher_dataset",
        "_oracle_targets",
        "_trend_baseline_targets",
        "_teacher_dataset",
    }:
        if teacher_methods_block.count(f"def {name}(") != 1:
            raise RuntimeError(f"teacher pipeline extraction omitted {name}")


def _migrate_tests_to_canonical_owner() -> None:
    sb3_test = Path("tests/integrations/test_sb3_training.py")
    _replace_exact(
        sb3_test,
        "from trade_rl.integrations import sb3_training\n",
        "from trade_rl.integrations import sb3_training\n"
        "import trade_rl.integrations.sb3_teacher_pipeline as sb3_teacher_pipeline\n",
    )
    _replace_exact(
        sb3_test,
        "from trade_rl.learning import OracleTeacherConfig\n",
        "from trade_rl.learning import OracleTeacherConfig\n"
        "from trade_rl.learning.episode_oracle_teacher import "
        "OracleEpisodeSamplingConfig\n"
        "from trade_rl.learning.oracle_bellman_contracts import OracleSolverConfig\n",
    )
    _replace_exact(
        sb3_test,
        'monkeypatch.setattr(sb3_training, "oracle_target_path", calculate)',
        'monkeypatch.setattr(sb3_teacher_pipeline, "oracle_target_path", calculate)',
    )
    _replace_exact(
        sb3_test,
        'monkeypatch.setattr(sb3_training, "collect_teacher_rollout", collect)',
        'monkeypatch.setattr(sb3_teacher_pipeline, "collect_teacher_rollout", collect)',
        expected_count=2,
    )
    _replace_exact(
        sb3_test,
        "    monkeypatch.setattr(\n"
        "        sb3_training,\n"
        '        "build_episode_oracle_batch",\n'
        "        fake_build_episode_oracle_batch,\n"
        "    )",
        "    monkeypatch.setattr(\n"
        "        sb3_teacher_pipeline,\n"
        '        "build_episode_oracle_batch",\n'
        "        fake_build_episode_oracle_batch,\n"
        "    )",
    )
    _replace_exact(
        sb3_test,
        "sb3_training.OracleSolverConfig",
        "OracleSolverConfig",
        expected_count=4,
    )
    _replace_exact(
        sb3_test,
        "sb3_training.OracleEpisodeSamplingConfig",
        "OracleEpisodeSamplingConfig",
    )

    episode_test = Path("tests/learning/test_episode_teacher_integration.py")
    _replace_exact(
        episode_test,
        "from trade_rl.integrations import sb3_training\n",
        "from trade_rl.integrations import sb3_training\n"
        "import trade_rl.integrations.sb3_teacher_pipeline as sb3_teacher_pipeline\n",
    )
    _replace_exact(
        episode_test,
        'monkeypatch.setattr(sb3_training, "build_episode_oracle_batch", build)',
        'monkeypatch.setattr(sb3_teacher_pipeline, "build_episode_oracle_batch", build)',
    )

    ownership_test = Path("tests/architecture/test_ownership_boundaries.py")
    _replace_exact(
        ownership_test,
        '    sb3_source = (PACKAGE_ROOT / "integrations/sb3_training.py").read_text(\n'
        '        encoding="utf-8"\n'
        "    )",
        "    teacher_pipeline_source = (\n"
        '        PACKAGE_ROOT / "integrations/sb3_teacher_pipeline.py"\n'
        '    ).read_text(encoding="utf-8")',
    )
    _replace_exact(
        ownership_test,
        '    assert "accelerator_backend=_oracle_accelerator_backend(" in sb3_source',
        "    assert (\n"
        '        "accelerator_backend=_oracle_accelerator_backend("\n'
        "        in teacher_pipeline_source\n"
        "    )",
    )


def main() -> None:
    _extract_modules()
    _migrate_tests_to_canonical_owner()


if __name__ == "__main__":
    main()
