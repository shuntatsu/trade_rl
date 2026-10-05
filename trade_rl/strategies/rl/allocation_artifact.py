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


def load_allocation_policy(
    root: Path, *, expected_digest: str, expected_recipe_digest: str
) -> AllocationPPOPolicy:
    """Verify canonical receipt and copy/hash policy bytes before executing loader."""
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
    with verified_private_copy(
        root / "policy.zip",
        expected_digest=manifest["policy_sha256"],
        field="allocation policy",
        filename="policy.zip",
    ) as verified:
        model = _load_policy(verified)
    return AllocationPPOPolicy(model, manifest)
