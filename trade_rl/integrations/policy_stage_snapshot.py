"""Immutable policy snapshots for causal pretraining-stage attribution."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest

_STAGES = frozenset({"random", "behavior_cloning", "behavior_cloning_critic"})


def write_policy_stage_snapshot(
    policy: Any,
    *,
    output_root: Path,
    stage: str,
    member_seed: int,
) -> dict[str, object]:
    """Save one policy-only checkpoint before later stages can mutate it."""

    if stage not in _STAGES:
        raise ValueError("unsupported policy stage")
    save = getattr(policy, "save", None)
    if not callable(save):
        raise TypeError("policy stage snapshot requires a save-capable policy")
    if isinstance(member_seed, bool) or not isinstance(member_seed, int) or member_seed < 0:
        raise ValueError("member_seed must be a non-negative integer")
    root = output_root / "policy-stages" / stage
    if root.exists():
        raise FileExistsError(f"policy stage already exists: {stage}")
    root.mkdir(parents=True)
    policy_path = root / "policy.zip"
    save(str(policy_path))
    if not policy_path.is_file() or policy_path.stat().st_size <= 0:
        raise RuntimeError("policy stage snapshot was not created")
    file_digest = hashlib.sha256(policy_path.read_bytes()).hexdigest()
    payload: dict[str, object] = {
        "file_digest": file_digest,
        "member_seed": member_seed,
        "policy_file": "policy.zip",
        "schema_version": "policy_stage_snapshot_v1",
        "stage": stage,
    }
    manifest = {**payload, "artifact_digest": content_digest(payload)}
    atomic_write_bytes(
        root / "manifest.json",
        canonical_json_bytes(manifest) + b"\n",
    )
    return manifest


__all__ = ["write_policy_stage_snapshot"]
