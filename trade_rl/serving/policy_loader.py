"""Canonical serving loader selection for structured policy ensembles."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path, PurePosixPath
from typing import Any, Final, TypedDict

import numpy as np

from trade_rl.artifacts.atomic_pointer import atomic_replace_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.structured_policy_contract import (
    STRUCTURED_EXPORT_MANIFEST_NAME,
    STRUCTURED_EXPORT_MODEL_NAME,
    load_structured_export_manifest,
)
from trade_rl.domain.common import require_sha256
from trade_rl.rl.sequence_observations import SEQUENCE_OBSERVATION_SCHEMA
from trade_rl.serving.bundle import ServingBundle, ServingBundleManifest

STRUCTURED_POLICY_LOADER_NAME: Final = "structured-policy-loader.json"
STRUCTURED_POLICY_LOADER_SCHEMA: Final = "structured_policy_loader_v1"


class StructuredPolicyLoaderMember(TypedDict):
    manifest: str
    manifest_digest: str
    model: str
    model_digest: str


class StructuredPolicyLoaderManifest(TypedDict):
    action_size: int
    architecture_digest: str
    digest: str
    members: tuple[StructuredPolicyLoaderMember, ...]
    schema_version: str


def _relative_path(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{field} contains an unsafe path")
    return path.as_posix()


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def build_structured_policy_loader_payload(
    root: Path,
    *,
    expected_members: int,
) -> dict[str, object]:
    """Build one strict loader manifest from member export manifests."""

    if (
        isinstance(expected_members, bool)
        or not isinstance(expected_members, int)
        or expected_members <= 0
    ):
        raise ValueError("expected_members must be a positive integer")
    root = Path(root)
    members: list[dict[str, object]] = []
    architecture_digest: str | None = None
    action_size: int | None = None
    policy_identity_digest: str | None = None
    input_contract: tuple[tuple[str, tuple[int, ...], str], ...] | None = None
    for index in range(expected_members):
        directory = root / "members" / f"member-{index:03d}"
        manifest_path = directory / STRUCTURED_EXPORT_MANIFEST_NAME
        model_path = directory / STRUCTURED_EXPORT_MODEL_NAME
        manifest = load_structured_export_manifest(manifest_path)
        if not model_path.is_file():
            raise FileNotFoundError(f"structured export model is missing: {model_path}")
        observed_contract = tuple(
            (item.name, item.shape, item.dtype) for item in manifest.inputs
        )
        if architecture_digest is None:
            architecture_digest = manifest.architecture_digest
            action_size = manifest.action_size
            policy_identity_digest = manifest.policy_identity_digest
            input_contract = observed_contract
        elif (
            manifest.architecture_digest != architecture_digest
            or manifest.action_size != action_size
            or manifest.policy_identity_digest != policy_identity_digest
            or observed_contract != input_contract
        ):
            raise ValueError("structured export members have inconsistent identity")
        manifest_relative = manifest_path.relative_to(root).as_posix()
        model_relative = model_path.relative_to(root).as_posix()
        members.append(
            {
                "manifest": manifest_relative,
                "manifest_digest": manifest.digest,
                "model": model_relative,
                "model_digest": manifest.model_digest,
            }
        )
    if architecture_digest is None or action_size is None:
        raise RuntimeError("structured export identity was not resolved")
    payload: dict[str, object] = {
        "action_size": action_size,
        "architecture_digest": architecture_digest,
        "members": tuple(members),
        "schema_version": STRUCTURED_POLICY_LOADER_SCHEMA,
    }
    return {"digest": content_digest(payload), **payload}


def write_structured_policy_loader_manifest(
    root: Path,
    *,
    expected_members: int,
) -> Path:
    payload = build_structured_policy_loader_payload(
        root,
        expected_members=expected_members,
    )
    path = Path(root) / STRUCTURED_POLICY_LOADER_NAME
    atomic_replace_bytes(path, canonical_json_bytes(payload))
    return path


def load_structured_policy_loader_manifest(
    path: Path,
) -> StructuredPolicyLoaderManifest:
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = _mapping(raw, field="structured policy loader manifest")
    expected = {
        "action_size",
        "architecture_digest",
        "digest",
        "members",
        "schema_version",
    }
    if set(payload) != expected:
        raise ValueError("structured policy loader manifest fields are invalid")
    if payload.get("schema_version") != STRUCTURED_POLICY_LOADER_SCHEMA:
        raise ValueError("unsupported structured policy loader schema")
    architecture_digest = payload.get("architecture_digest")
    if not isinstance(architecture_digest, str):
        raise ValueError("structured policy loader architecture digest is invalid")
    require_sha256(architecture_digest, field="architecture_digest")
    action_size = payload.get("action_size")
    if (
        isinstance(action_size, bool)
        or not isinstance(action_size, int)
        or action_size <= 0
    ):
        raise ValueError("structured policy loader action_size is invalid")
    raw_members = payload.get("members")
    if not isinstance(raw_members, list) or not raw_members:
        raise ValueError("structured policy loader members must be a non-empty list")
    members: list[StructuredPolicyLoaderMember] = []
    for index, raw_member in enumerate(raw_members):
        member = _mapping(raw_member, field=f"members[{index}]")
        if set(member) != {
            "manifest",
            "manifest_digest",
            "model",
            "model_digest",
        }:
            raise ValueError("structured policy loader member fields are invalid")
        manifest = _relative_path(member.get("manifest"), field="manifest")
        model = _relative_path(member.get("model"), field="model")
        if not manifest.endswith(f"/{STRUCTURED_EXPORT_MANIFEST_NAME}"):
            raise ValueError("structured member manifest path is invalid")
        if not model.endswith(f"/{STRUCTURED_EXPORT_MODEL_NAME}"):
            raise ValueError("structured member model path is invalid")
        manifest_digest = member.get("manifest_digest")
        model_digest = member.get("model_digest")
        if not isinstance(manifest_digest, str):
            raise ValueError("structured member manifest digest is invalid")
        if not isinstance(model_digest, str):
            raise ValueError("structured member model digest is invalid")
        require_sha256(manifest_digest, field="manifest_digest")
        require_sha256(model_digest, field="model_digest")
        members.append(
            {
                "manifest": manifest,
                "manifest_digest": manifest_digest,
                "model": model,
                "model_digest": model_digest,
            }
        )
    digest_payload = {
        "action_size": action_size,
        "architecture_digest": architecture_digest,
        "members": tuple(members),
        "schema_version": STRUCTURED_POLICY_LOADER_SCHEMA,
    }
    digest = payload.get("digest")
    if not isinstance(digest, str):
        raise ValueError("structured policy loader digest is invalid")
    require_sha256(digest, field="structured_policy_loader.digest")
    if digest != content_digest(digest_payload):
        raise ValueError("structured policy loader digest mismatch")
    return {
        "action_size": action_size,
        "architecture_digest": architecture_digest,
        "digest": digest,
        "members": tuple(members),
        "schema_version": STRUCTURED_POLICY_LOADER_SCHEMA,
    }


class StructuredTorchScriptEnsemblePolicy:
    """Deterministic mean ensemble of validated structured TorchScript members."""

    def __init__(self, members: tuple[Any, ...], *, action_size: int) -> None:
        if not members:
            raise ValueError("structured policy ensemble must contain members")
        self.members = members
        self.action_size = action_size

    def predict(self, observation: Mapping[str, np.ndarray]) -> np.ndarray:
        actions = tuple(
            np.asarray(member.predict(observation)) for member in self.members
        )
        matrix = np.stack(actions, axis=0)
        if matrix.shape != (len(self.members), self.action_size):
            raise ValueError("structured ensemble member action shape mismatch")
        result = matrix.mean(axis=0, dtype=np.float64).astype(np.float32)
        if not np.isfinite(result).all():
            raise ValueError("structured ensemble action must be finite")
        return result

    def smoke_observation(self) -> dict[str, np.ndarray]:
        factory = getattr(self.members[0], "smoke_observation", None)
        if not callable(factory):
            raise TypeError("structured ensemble member lacks smoke observation")
        return factory()


class StructuredTorchScriptEnsembleLoader:
    """Load the canonical structured export ensemble without Stable-Baselines3."""

    def __init__(self, *, expected_architecture_digest: str) -> None:
        require_sha256(
            expected_architecture_digest,
            field="expected_architecture_digest",
        )
        self.expected_architecture_digest = expected_architecture_digest

    def load(self, bundle: ServingBundle) -> StructuredTorchScriptEnsemblePolicy:
        if bundle.manifest.observation_schema != SEQUENCE_OBSERVATION_SCHEMA:
            raise ValueError("structured loader requires sequence observation schema")
        path = bundle.root / STRUCTURED_POLICY_LOADER_NAME
        declared = {item.path: item for item in bundle.manifest.files}
        declared_loader = declared.get(STRUCTURED_POLICY_LOADER_NAME)
        if declared_loader is None or not path.is_file():
            raise ValueError("structured policy loader manifest is not bound to bundle")
        if path.stat().st_size != declared_loader.size_bytes:
            raise ValueError("structured policy loader manifest size mismatch")
        from trade_rl.serving.structured_policy import StructuredTorchScriptPolicy

        loader_payload = load_structured_policy_loader_manifest(path)
        if loader_payload["architecture_digest"] != self.expected_architecture_digest:
            raise ValueError("structured policy architecture does not match runtime")
        if loader_payload["action_size"] != bundle.manifest.action_size:
            raise ValueError("structured policy action size does not match bundle")
        members: list[StructuredTorchScriptPolicy] = []
        for raw_member in loader_payload["members"]:
            if not isinstance(raw_member, Mapping):
                raise ValueError("structured loader member is invalid")
            manifest_relative = str(raw_member["manifest"])
            model_relative = str(raw_member["model"])
            if manifest_relative not in declared or model_relative not in declared:
                raise ValueError("structured member files are not bound to bundle")
            manifest = load_structured_export_manifest(bundle.root / manifest_relative)
            if manifest.digest != raw_member["manifest_digest"]:
                raise ValueError("structured member manifest digest mismatch")
            if manifest.model_digest != raw_member["model_digest"]:
                raise ValueError("structured member model digest mismatch")
            if manifest.architecture_digest != self.expected_architecture_digest:
                raise ValueError("structured member architecture mismatch")
            members.append(
                StructuredTorchScriptPolicy(
                    root=(bundle.root / manifest_relative).parent,
                    manifest=manifest,
                )
            )
        return StructuredTorchScriptEnsemblePolicy(
            tuple(members),
            action_size=bundle.manifest.action_size,
        )


def canonical_policy_loader(
    *,
    manifest: ServingBundleManifest,
    architecture_digest: str | None,
    fallback: object | None = None,
) -> Any:
    """Resolve the only supported loader for a deployment identity."""

    if manifest.observation_schema == SEQUENCE_OBSERVATION_SCHEMA:
        if architecture_digest is None:
            raise ValueError("structured serving requires architecture identity")
        if manifest.architecture_digest != architecture_digest:
            raise ValueError("serving architecture identity mismatch")
        return StructuredTorchScriptEnsembleLoader(
            expected_architecture_digest=architecture_digest
        )
    if architecture_digest is not None:
        raise ValueError("flat serving cannot declare architecture identity")
    if fallback is None:
        raise RuntimeError("flat residual serving requires an explicit policy loader")
    return fallback


__all__ = [
    "STRUCTURED_POLICY_LOADER_NAME",
    "STRUCTURED_POLICY_LOADER_SCHEMA",
    "StructuredTorchScriptEnsembleLoader",
    "StructuredTorchScriptEnsemblePolicy",
    "build_structured_policy_loader_payload",
    "canonical_policy_loader",
    "load_structured_policy_loader_manifest",
    "write_structured_policy_loader_manifest",
]
