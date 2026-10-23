"""Write-once allocation bundles, verified before optional PPO deserialization."""

from __future__ import annotations

import importlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

from trade_rl._validation import require_sha256
from trade_rl.artifacts import canonical_json_bytes, content_digest
from trade_rl.artifacts.atomic_write import atomic_rename_directory
from trade_rl.artifacts.verified_file import (
    file_digest,
    open_regular_binary,
    verified_private_copy,
)
from trade_rl.strategies.rl.allocation_manifest import validate_allocation_manifest
from trade_rl.strategies.rl.allocation_model import (
    AllocationPPOPolicy,
    validate_allocation_model,
)
from trade_rl.strategies.rl.allocation_preprocessing import (
    AllocationFeaturePreprocessing,
)
from trade_rl.strategies.rl.allocation_training_schedule import (
    AllocationTrainingSchedule,
)


def save_allocation_policy(root: Path, policy: AllocationPPOPolicy) -> str:
    """Publish immutable bytes; callers retain the returned external content pin."""
    root = Path(root)
    if root.exists() or root.is_symlink():
        raise FileExistsError(f"allocation policy destination already exists: {root}")
    manifest = policy.manifest
    validate_allocation_model(policy.model, manifest)
    root.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{root.name}.staging-", dir=root.parent))
    try:
        policy_path = staging / "policy.zip"
        policy.model.save(str(policy_path))
        manifest["policy_sha256"] = file_digest(policy_path, field="allocation policy")
        validate_allocation_manifest(manifest, require_policy=True)
        with (staging / "manifest.json").open("xb") as stream:
            stream.write(canonical_json_bytes(manifest))
        atomic_rename_directory(staging, root)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return content_digest(manifest)


def _load_policy(path: Path) -> Any:
    torch = importlib.import_module("torch")
    getattr(torch, "set_num_threads")(1)
    sb3 = importlib.import_module("stable_baselines3")
    return getattr(sb3, "PPO").load(str(path), device="cpu")


def read_allocation_policy_manifest(
    root: Path,
    *,
    expected_digest: str,
    expected_recipe_digest: str,
    training_cutoff_ns: int | None = None,
) -> dict[str, Any]:
    """Read pinned native metadata without importing/deserializing the backend.

    The opt-in cutoff checks scheduled source claims, not fit authenticity.
    Policy bytes and actual model attributes remain the loader's responsibility.
    """
    if training_cutoff_ns is not None and (
        type(training_cutoff_ns) is not int or not -(2**63) < training_cutoff_ns < 2**63
    ):
        raise ValueError("training cutoff must be native non-NaT nanoseconds")
    require_sha256(expected_digest, field="expected_digest")
    require_sha256(expected_recipe_digest, field="expected_recipe_digest")
    root = Path(root)
    if (
        root.is_symlink()
        or not root.is_dir()
        or {path.name for path in root.iterdir()} != {"manifest.json", "policy.zip"}
    ):
        raise ValueError("allocation bundle must be an exact regular directory")
    with open_regular_binary(
        root / "manifest.json", field="allocation manifest"
    ) as stream:
        raw = stream.read()
    try:
        manifest = json.loads(raw)
    except (ValueError, UnicodeDecodeError) as error:
        raise ValueError("allocation manifest is malformed") from error
    if (
        canonical_json_bytes(manifest) != raw
        or content_digest(manifest) != expected_digest
    ):
        raise ValueError("allocation manifest differs from its pinned canonical digest")
    manifest = validate_allocation_manifest(manifest, require_policy=True)
    if manifest["recipe_digest"] != expected_recipe_digest:
        raise ValueError("allocation recipe differs from the caller's pinned recipe")
    if training_cutoff_ns is not None:
        if manifest["schema"] != "allocation_ppo_inference_bundle_v5":
            raise ValueError("global cutoff requires explicit scheduled bundle v5")
        schedule = AllocationTrainingSchedule.from_payload(
            manifest["training"]["schedule"]
        )
        if any(
            w.terminal_time_ns >= training_cutoff_ns for w in schedule.training_windows
        ):
            raise ValueError("training terminal must strictly precede global OOS start")
        recipe = manifest["recipe"]
        if recipe["schema"] == "allocation_ppo_recipe_v3":
            frozen = AllocationFeaturePreprocessing.from_payload(
                recipe["observation"]["feature_preprocessing"]
            )
            if (
                max(frozen.fit_last_event_time_ns, frozen.fit_as_of_ns)
                >= training_cutoff_ns
            ):
                raise ValueError(
                    "preprocessing fit must strictly precede global OOS start"
                )
    return manifest


def load_allocation_policy(
    root: Path,
    *,
    expected_digest: str,
    expected_recipe_digest: str,
    training_cutoff_ns: int | None = None,
) -> AllocationPPOPolicy:
    """Recheck pinned metadata, copy/hash policy bytes, then execute the loader."""
    root = Path(root)
    manifest = read_allocation_policy_manifest(
        root,
        expected_digest=expected_digest,
        expected_recipe_digest=expected_recipe_digest,
        training_cutoff_ns=training_cutoff_ns,
    )
    with verified_private_copy(
        root / "policy.zip",
        expected_digest=manifest["policy_sha256"],
        field="allocation policy",
        filename="policy.zip",
    ) as verified:
        model = _load_policy(verified)
    return AllocationPPOPolicy(model, manifest)
