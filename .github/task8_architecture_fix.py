from pathlib import Path

# Keep catalog below learning: move teacher-specific identity/backfill ownership upward.
path = Path("trade_rl/catalog/reusable_artifacts.py")
text = path.read_text(encoding="utf-8")
text = text.replace("import json\n", "")
text = text.replace(
    "from trade_rl.learning.oracle_bellman_contracts import OracleSolverProvenance\n",
    "",
)
text = text.replace(
    "from trade_rl.learning.oracle_market_tape import ORACLE_MARKET_TAPE_SCHEMA\n",
    "",
)
start = text.index("\ndef teacher_cache_identity(")
end = text.index("\n\nclass ReusableArtifactIndex", start)
text = text[:start] + text[end:]
start = text.index("\ndef backfill_teacher_cache(")
end = text.index("\n\n__all__ =", start)
text = text[:start] + text[end:]
text = text.replace(
    '__all__ = [\n    "ReusableArtifactIndex",\n    "backfill_teacher_cache",\n    "teacher_cache_identity",\n    "teacher_cache_identity_v2",\n]\n',
    '__all__ = ["ReusableArtifactIndex"]\n',
)
path.write_text(text, encoding="utf-8")

Path("trade_rl/learning/teacher_cache.py").write_text(
    '''"""Oracle teacher cache identities and artifact backfill orchestration."""

from __future__ import annotations

import json

from trade_rl.catalog.contracts import ArtifactKind
from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.learning.episode_teacher_artifact import (
    EPISODE_TEACHER_ARTIFACT_SCHEMA,
    EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
    load_episode_teacher_artifact,
)
from trade_rl.learning.oracle_bellman_contracts import OracleSolverProvenance
from trade_rl.learning.oracle_market_tape import ORACLE_MARKET_TAPE_SCHEMA
from trade_rl.learning.teacher_artifact import load_teacher_artifact


def teacher_cache_identity(
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    environment_digest: str,
    action_spec_digest: str,
    teacher_config_digest: str,
) -> dict[str, object]:
    return {
        "action_spec_digest": action_spec_digest,
        "dataset_id": dataset_id,
        "environment_digest": environment_digest,
        "schema_version": "teacher_cache_identity_v1",
        "teacher_config_digest": teacher_config_digest,
        "train_range": train_range,
    }


def teacher_cache_identity_v2(
    *,
    dataset_id: str,
    train_range: tuple[int, int],
    environment_digest: str,
    action_spec_digest: str,
    teacher_config_digest: str,
    solver_provenance: OracleSolverProvenance,
) -> dict[str, object]:
    """Return stable solver-aware cache identity for newly generated teachers."""

    if not isinstance(solver_provenance, OracleSolverProvenance):
        raise ValueError("solver_provenance must be OracleSolverProvenance")
    return {
        "action_spec_digest": action_spec_digest,
        "compile_chunk_size": solver_provenance.compile_chunk_size,
        "compile_mode": solver_provenance.compile_mode,
        "dataset_id": dataset_id,
        "environment_digest": environment_digest,
        "episode_batch_size": solver_provenance.episode_batch_size,
        "fallback_reason": solver_provenance.fallback_reason,
        "market_tape_digest": solver_provenance.market_tape_digest,
        "market_tape_schema": ORACLE_MARKET_TAPE_SCHEMA,
        "numeric_dtype": solver_provenance.numeric_dtype,
        "oom_retry_performed": solver_provenance.oom_retry_performed,
        "schema_version": "teacher_cache_identity_v2",
        "solver_backend": solver_provenance.backend,
        "solver_contract": solver_provenance.solver_contract,
        "target_state_block_size": solver_provenance.target_state_block_size,
        "teacher_config_digest": teacher_config_digest,
        "tie_break_contract": solver_provenance.tie_break_contract,
        "tie_tolerance": solver_provenance.tie_tolerance,
        "train_range": train_range,
    }


def backfill_teacher_cache(index: ReusableArtifactIndex) -> int:
    """Validate and index completed Teacher directories already on the volume."""

    registered = 0
    for path in sorted(index.storage_root.iterdir()):
        manifest_path = path / "manifest.json"
        if (
            not path.is_dir()
            or path.name.startswith(".")
            or not manifest_path.is_file()
        ):
            continue
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("teacher cache manifest must be an object")
        schema_version = str(raw.get("schema_version", ""))
        if schema_version in {
            EPISODE_TEACHER_ARTIFACT_SCHEMA_V1,
            EPISODE_TEACHER_ARTIFACT_SCHEMA,
        }:
            episode_manifest, _ = load_episode_teacher_artifact(path)
            artifact_digest = episode_manifest.artifact_digest
            manifest_schema = episode_manifest.schema_version
            dataset_id = episode_manifest.dataset_id
            train_range = (episode_manifest.train_start, episode_manifest.train_stop)
            environment_digest = episode_manifest.environment_digest
            action_spec_digest = episode_manifest.action_spec_digest
            teacher_config_digest = episode_manifest.teacher_config_digest
            solver_provenance = episode_manifest.solver_provenance
            metadata: dict[str, object] = {
                "episode_count": episode_manifest.episode_count,
                "sample_count": episode_manifest.sample_count,
            }
            if solver_provenance is not None:
                metadata["solver_provenance"] = solver_provenance.serialized_payload()
        else:
            teacher_manifest, _ = load_teacher_artifact(path)
            artifact_digest = teacher_manifest.artifact_digest
            manifest_schema = teacher_manifest.schema_version
            dataset_id = teacher_manifest.dataset_id
            train_range = (teacher_manifest.train_start, teacher_manifest.train_stop)
            environment_digest = teacher_manifest.environment_digest
            action_spec_digest = teacher_manifest.action_spec_digest
            teacher_config_digest = teacher_manifest.teacher_config_digest
            solver_provenance = None
            metadata = {"sample_count": teacher_manifest.sample_count}
        cache_key = (
            teacher_cache_identity(
                dataset_id=dataset_id,
                train_range=train_range,
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config_digest,
            )
            if solver_provenance is None
            else teacher_cache_identity_v2(
                dataset_id=dataset_id,
                train_range=train_range,
                environment_digest=environment_digest,
                action_spec_digest=action_spec_digest,
                teacher_config_digest=teacher_config_digest,
                solver_provenance=solver_provenance,
            )
        )
        index.register_directory(
            artifact_digest=artifact_digest,
            artifact_kind=ArtifactKind.ORACLE_TEACHER,
            schema_version=manifest_schema,
            dataset_id=dataset_id,
            cache_key=cache_key,
            metadata=metadata,
            location=path,
        )
        registered += 1
    return registered


__all__ = [
    "backfill_teacher_cache",
    "teacher_cache_identity",
    "teacher_cache_identity_v2",
]
''',
    encoding="utf-8",
)

path = Path("trade_rl/integrations/sb3_training.py")
text = path.read_text(encoding="utf-8")
old = '''from trade_rl.catalog.reusable_artifacts import (
    ReusableArtifactIndex,
    teacher_cache_identity,
    teacher_cache_identity_v2,
)'''
new = '''from trade_rl.catalog.reusable_artifacts import ReusableArtifactIndex
from trade_rl.learning.teacher_cache import (
    teacher_cache_identity,
    teacher_cache_identity_v2,
)'''
if text.count(old) != 1:
    raise SystemExit("SB3 reusable-artifact import changed unexpectedly")
path.write_text(text.replace(old, new), encoding="utf-8")

path = Path("tests/catalog/test_reusable_artifacts.py")
text = path.read_text(encoding="utf-8")
text = text.replace(
    "from trade_rl.catalog.reusable_artifacts import teacher_cache_identity_v2",
    "from trade_rl.learning.teacher_cache import teacher_cache_identity_v2",
)
text = text.replace(
    "from trade_rl.catalog.reusable_artifacts import backfill_teacher_cache",
    "from trade_rl.learning.teacher_cache import backfill_teacher_cache",
)
path.write_text(text, encoding="utf-8")

# Preserve the framework-neutral learning contract with a narrow numerical-backend exception.
path = Path(".importlinter")
text = path.read_text(encoding="utf-8")
anchor = '''forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
'''
replacement = '''forbidden_modules =
    stable_baselines3
    sb3_contrib
    torch
allow_indirect_imports = True
ignore_imports =
    trade_rl.learning.oracle_bellman_torch -> torch
    trade_rl.learning.oracle_transition_torch -> torch
'''
if text.count(anchor) != 1:
    raise SystemExit("learning framework contract changed unexpectedly")
path.write_text(text.replace(anchor, replacement, 1), encoding="utf-8")

Path("tests/architecture/test_oracle_torch_backend_boundary.py").write_text(
    '''from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
APPROVED_TORCH_MODULES = {
    "oracle_bellman_torch.py",
    "oracle_transition_torch.py",
}


def _imports_torch(path: Path) -> bool:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(
                alias.name == "torch" or alias.name.startswith("torch.")
                for alias in node.names
            ):
                return True
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module == "torch" or module.startswith("torch."):
                return True
    return False


def test_learning_torch_dependency_is_limited_to_explicit_oracle_backend() -> None:
    learning = ROOT / "trade_rl" / "learning"
    torch_modules = {
        path.name for path in learning.glob("*.py") if _imports_torch(path)
    }
    assert torch_modules == APPROVED_TORCH_MODULES

    contract = (ROOT / ".importlinter").read_text(encoding="utf-8")
    block = contract.split(
        "[importlinter:contract:learning-frameworks]", maxsplit=1
    )[1].split("[importlinter:", maxsplit=1)[0]
    assert "allow_indirect_imports = True" in block
    for module in APPROVED_TORCH_MODULES:
        qualified = f"trade_rl.learning.{module.removesuffix('.py')} -> torch"
        assert qualified in block


def test_catalog_reusable_artifacts_remains_below_learning() -> None:
    source = (ROOT / "trade_rl" / "catalog" / "reusable_artifacts.py").read_text(
        encoding="utf-8"
    )
    assert "trade_rl.learning" not in source
''',
    encoding="utf-8",
)
