"""Immutable serving bundles with complete evidence-chain validation."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.hashing import content_digest
from trade_rl.domain.common import (
    require_aware_datetime,
    require_non_empty,
    require_sha256,
)
from trade_rl.domain.selection import PolicyMode
from trade_rl.release.attestation import (
    ReleaseAttestation,
    default_attestation_path,
    load_release_attestation,
)
from trade_rl.serving.normalizer import (
    NORMALIZER_ARTIFACT_NAME,
    load_observation_normalizer,
)

BUNDLE_MANIFEST_NAME = "bundle.json"
SERVING_BUNDLE_SCHEMA = "serving_bundle_v5"
_SELECTED_FINAL = "research_selected_final"
_BASELINE_RELEASE = "baseline_release"


def _relative_path(value: str) -> str:
    normalized = PurePosixPath(value)
    if normalized.is_absolute() or not normalized.parts:
        raise ValueError("bundle artifact path must be relative")
    if any(part in {"", ".", ".."} for part in normalized.parts):
        raise ValueError("bundle artifact path contains an unsafe segment")
    path = normalized.as_posix()
    if path == BUNDLE_MANIFEST_NAME:
        raise ValueError("bundle manifest cannot include itself as an artifact")
    return path


def _file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _optional_digest(value: str | None, *, field: str) -> None:
    if value is not None:
        require_sha256(value, field=field)


@dataclass(frozen=True, slots=True)
class BundleFile:
    path: str
    digest: str
    size_bytes: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "path", _relative_path(self.path))
        require_sha256(self.digest, field="file.digest")
        if isinstance(self.size_bytes, bool) or self.size_bytes < 0:
            raise ValueError("file size_bytes must be non-negative")


@dataclass(frozen=True, slots=True)
class ServingBundleManifest:
    bundle_digest: str
    dataset_id: str
    action_schema: str
    observation_schema: str
    observation_size: int
    environment_digest: str
    initial_capital: float
    policy_mode: PolicyMode
    policy_digest: str | None
    signal_digest: str
    selection_digest: str
    training_run_digest: str | None
    run_kind: str
    selection_proposal_digest: str | None
    selection_authorization_digest: str | None
    walk_forward_run_digest: str | None
    gate_evidence_digest: str | None
    confirmation_evidence_digest: str | None
    files: tuple[BundleFile, ...]
    created_at: datetime
    action_size: int = 2
    action_names: tuple[str, ...] = ()
    action_spec_digest: str | None = None
    alpha_artifact_digest: str | None = None
    factor_artifact_digest: str | None = None
    normalizer_digest: str | None = None
    schema_version: str = SERVING_BUNDLE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != SERVING_BUNDLE_SCHEMA:
            raise ValueError("unsupported serving bundle schema_version")
        require_sha256(self.bundle_digest, field="bundle_digest")
        require_sha256(self.dataset_id, field="dataset_id")
        require_non_empty(self.action_schema, field="action_schema")
        require_non_empty(self.observation_schema, field="observation_schema")
        if (
            isinstance(self.observation_size, bool)
            or not isinstance(self.observation_size, int)
            or self.observation_size <= 0
        ):
            raise ValueError("observation_size must be a positive integer")
        if (
            isinstance(self.action_size, bool)
            or not isinstance(self.action_size, int)
            or self.action_size <= 0
        ):
            raise ValueError("action_size must be a positive integer")
        if len(self.action_names) != self.action_size:
            raise ValueError("action_names must match action_size")
        if len(set(self.action_names)) != len(self.action_names) or any(
            not name for name in self.action_names
        ):
            raise ValueError("action_names must be unique and non-empty")
        if self.action_spec_digest is None:
            raise ValueError("serving bundle requires action_spec_digest")
        require_sha256(self.action_spec_digest, field="action_spec_digest")
        require_sha256(self.environment_digest, field="environment_digest")
        if not math.isfinite(self.initial_capital) or self.initial_capital <= 0.0:
            raise ValueError("initial_capital must be finite and positive")
        require_sha256(self.signal_digest, field="signal_digest")
        require_sha256(self.selection_digest, field="selection_digest")
        for field, value in (
            ("training_run_digest", self.training_run_digest),
            ("selection_proposal_digest", self.selection_proposal_digest),
            ("selection_authorization_digest", self.selection_authorization_digest),
            ("walk_forward_run_digest", self.walk_forward_run_digest),
            ("gate_evidence_digest", self.gate_evidence_digest),
            ("confirmation_evidence_digest", self.confirmation_evidence_digest),
            ("alpha_artifact_digest", self.alpha_artifact_digest),
            ("factor_artifact_digest", self.factor_artifact_digest),
            ("normalizer_digest", self.normalizer_digest),
        ):
            _optional_digest(value, field=field)
        if self.policy_mode is PolicyMode.BASELINE_ONLY:
            if self.policy_digest is not None:
                raise ValueError("baseline_only bundle cannot contain a policy digest")
            if self.run_kind != _BASELINE_RELEASE:
                raise ValueError(
                    "baseline_only bundle requires baseline_release run_kind"
                )
            if any(
                value is not None
                for value in (
                    self.training_run_digest,
                    self.selection_proposal_digest,
                    self.selection_authorization_digest,
                    self.walk_forward_run_digest,
                    self.gate_evidence_digest,
                    self.confirmation_evidence_digest,
                )
            ):
                raise ValueError("baseline release cannot contain training evidence")
        else:
            if self.policy_digest is None:
                raise ValueError("residual policy bundle requires a policy digest")
            require_sha256(self.policy_digest, field="policy_digest")
            if self.run_kind != _SELECTED_FINAL:
                raise ValueError(
                    "residual policy bundle requires selected-final run_kind"
                )
            if any(
                value is None
                for value in (
                    self.training_run_digest,
                    self.selection_proposal_digest,
                    self.selection_authorization_digest,
                    self.walk_forward_run_digest,
                    self.gate_evidence_digest,
                    self.confirmation_evidence_digest,
                )
            ):
                raise ValueError(
                    "residual policy bundle requires the complete authorization chain"
                )
        if not self.files:
            raise ValueError("serving bundle must contain artifact files")
        paths = tuple(file.path for file in self.files)
        if len(set(paths)) != len(paths):
            raise ValueError("serving bundle artifact paths must be unique")
        require_aware_datetime(self.created_at, field="created_at")
        if self.bundle_digest != content_digest(self.digest_payload()):
            raise ValueError("bundle digest does not match manifest content")

    def digest_payload(self) -> dict[str, object]:
        return {
            "action_names": self.action_names,
            "action_schema": self.action_schema,
            "action_size": self.action_size,
            "action_spec_digest": self.action_spec_digest,
            "alpha_artifact_digest": self.alpha_artifact_digest,
            "confirmation_evidence_digest": self.confirmation_evidence_digest,
            "created_at": self.created_at,
            "dataset_id": self.dataset_id,
            "environment_digest": self.environment_digest,
            "factor_artifact_digest": self.factor_artifact_digest,
            "files": self.files,
            "gate_evidence_digest": self.gate_evidence_digest,
            "initial_capital": self.initial_capital,
            "normalizer_digest": self.normalizer_digest,
            "observation_schema": self.observation_schema,
            "observation_size": self.observation_size,
            "policy_digest": self.policy_digest,
            "policy_mode": self.policy_mode,
            "run_kind": self.run_kind,
            "schema_version": self.schema_version,
            "selection_authorization_digest": self.selection_authorization_digest,
            "selection_digest": self.selection_digest,
            "selection_proposal_digest": self.selection_proposal_digest,
            "signal_digest": self.signal_digest,
            "training_run_digest": self.training_run_digest,
            "walk_forward_run_digest": self.walk_forward_run_digest,
        }

    @classmethod
    def build(
        cls,
        *,
        root: Path,
        dataset_id: str,
        action_schema: str,
        observation_schema: str,
        observation_size: int,
        environment_digest: str,
        initial_capital: float,
        policy_mode: PolicyMode,
        policy_digest: str | None,
        signal_digest: str,
        selection_digest: str,
        artifact_paths: tuple[str, ...],
        created_at: datetime,
        action_size: int = 2,
        action_names: tuple[str, ...] = (),
        action_spec_digest: str | None = None,
        alpha_artifact_digest: str | None = None,
        factor_artifact_digest: str | None = None,
        normalizer_digest: str | None = None,
        training_run_digest: str | None = None,
        run_kind: str | None = None,
        selection_proposal_digest: str | None = None,
        selection_authorization_digest: str | None = None,
        walk_forward_run_digest: str | None = None,
        gate_evidence_digest: str | None = None,
        confirmation_evidence_digest: str | None = None,
        release_digest: str | None = None,
    ) -> ServingBundleManifest:
        if release_digest is not None:
            raise ValueError(
                "release attestations are external to serving bundle identity"
            )
        resolved_run_kind = run_kind or (
            _BASELINE_RELEASE
            if policy_mode is PolicyMode.BASELINE_ONLY
            else _SELECTED_FINAL
        )
        files: list[BundleFile] = []
        for raw_path in artifact_paths:
            relative = _relative_path(raw_path)
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(f"bundle artifact does not exist: {relative}")
            files.append(
                BundleFile(
                    path=relative,
                    digest=_file_digest(path),
                    size_bytes=path.stat().st_size,
                )
            )
        ordered = tuple(sorted(files, key=lambda item: item.path))
        payload = {
            "action_names": action_names,
            "action_schema": action_schema,
            "action_size": action_size,
            "action_spec_digest": action_spec_digest,
            "alpha_artifact_digest": alpha_artifact_digest,
            "confirmation_evidence_digest": confirmation_evidence_digest,
            "created_at": created_at,
            "dataset_id": dataset_id,
            "environment_digest": environment_digest,
            "factor_artifact_digest": factor_artifact_digest,
            "files": ordered,
            "gate_evidence_digest": gate_evidence_digest,
            "initial_capital": initial_capital,
            "normalizer_digest": normalizer_digest,
            "observation_schema": observation_schema,
            "observation_size": observation_size,
            "policy_digest": policy_digest,
            "policy_mode": policy_mode,
            "run_kind": resolved_run_kind,
            "schema_version": SERVING_BUNDLE_SCHEMA,
            "selection_authorization_digest": selection_authorization_digest,
            "selection_digest": selection_digest,
            "selection_proposal_digest": selection_proposal_digest,
            "signal_digest": signal_digest,
            "training_run_digest": training_run_digest,
            "walk_forward_run_digest": walk_forward_run_digest,
        }
        return cls(
            bundle_digest=content_digest(payload),
            dataset_id=dataset_id,
            action_schema=action_schema,
            observation_schema=observation_schema,
            observation_size=observation_size,
            environment_digest=environment_digest,
            initial_capital=initial_capital,
            policy_mode=policy_mode,
            policy_digest=policy_digest,
            signal_digest=signal_digest,
            selection_digest=selection_digest,
            training_run_digest=training_run_digest,
            run_kind=resolved_run_kind,
            selection_proposal_digest=selection_proposal_digest,
            selection_authorization_digest=selection_authorization_digest,
            walk_forward_run_digest=walk_forward_run_digest,
            gate_evidence_digest=gate_evidence_digest,
            confirmation_evidence_digest=confirmation_evidence_digest,
            files=ordered,
            created_at=created_at,
            action_size=action_size,
            action_names=action_names,
            action_spec_digest=action_spec_digest,
            alpha_artifact_digest=alpha_artifact_digest,
            factor_artifact_digest=factor_artifact_digest,
            normalizer_digest=normalizer_digest,
            schema_version=SERVING_BUNDLE_SCHEMA,
        )


@dataclass(frozen=True, slots=True)
class ServingBundle:
    root: Path
    manifest: ServingBundleManifest
    release: ReleaseAttestation | None = None
    normalizer: object | None = None


def write_serving_bundle_manifest(root: Path, manifest: ServingBundleManifest) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / BUNDLE_MANIFEST_NAME
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_bytes(canonical_json_bytes(manifest))
    temporary.replace(path)
    return path


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be a mapping")
    return value


def _string(value: object, *, field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    return value


def _optional_string(value: object, *, field: str) -> str | None:
    if value is None:
        return None
    return _string(value, field=field)


def _string_tuple(value: object, *, field: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f"{field} must be a list of strings")
    return tuple(value)


def _integer(value: object, *, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer")
    return value


def _number(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{field} must be a number")
    return float(value)


def _parse_manifest(payload: Mapping[str, object]) -> ServingBundleManifest:
    expected_fields = {
        "action_names",
        "action_schema",
        "action_size",
        "action_spec_digest",
        "alpha_artifact_digest",
        "bundle_digest",
        "confirmation_evidence_digest",
        "created_at",
        "dataset_id",
        "environment_digest",
        "factor_artifact_digest",
        "files",
        "gate_evidence_digest",
        "initial_capital",
        "normalizer_digest",
        "observation_schema",
        "observation_size",
        "policy_digest",
        "policy_mode",
        "run_kind",
        "schema_version",
        "selection_authorization_digest",
        "selection_digest",
        "selection_proposal_digest",
        "signal_digest",
        "training_run_digest",
        "walk_forward_run_digest",
    }
    if set(payload) != expected_fields:
        raise ValueError("serving bundle manifest fields are invalid")
    if payload.get("schema_version") != SERVING_BUNDLE_SCHEMA:
        raise ValueError("unsupported serving bundle schema_version")
    raw_files = payload.get("files")
    if not isinstance(raw_files, list):
        raise ValueError("files must be a list")
    files: list[BundleFile] = []
    for index, raw_file in enumerate(raw_files):
        item = _mapping(raw_file, field=f"files[{index}]")
        files.append(
            BundleFile(
                path=_string(item.get("path"), field=f"files[{index}].path"),
                digest=_string(item.get("digest"), field=f"files[{index}].digest"),
                size_bytes=_integer(
                    item.get("size_bytes"), field=f"files[{index}].size_bytes"
                ),
            )
        )
    created_at = datetime.fromisoformat(
        _string(payload.get("created_at"), field="created_at").replace("Z", "+00:00")
    )
    return ServingBundleManifest(
        bundle_digest=_string(payload.get("bundle_digest"), field="bundle_digest"),
        dataset_id=_string(payload.get("dataset_id"), field="dataset_id"),
        action_schema=_string(payload.get("action_schema"), field="action_schema"),
        observation_schema=_string(
            payload.get("observation_schema"), field="observation_schema"
        ),
        observation_size=_integer(
            payload.get("observation_size"), field="observation_size"
        ),
        environment_digest=_string(
            payload.get("environment_digest"), field="environment_digest"
        ),
        initial_capital=_number(
            payload.get("initial_capital"), field="initial_capital"
        ),
        policy_mode=PolicyMode(
            _string(payload.get("policy_mode"), field="policy_mode")
        ),
        policy_digest=_optional_string(
            payload.get("policy_digest"), field="policy_digest"
        ),
        signal_digest=_string(payload.get("signal_digest"), field="signal_digest"),
        selection_digest=_string(
            payload.get("selection_digest"), field="selection_digest"
        ),
        training_run_digest=_optional_string(
            payload.get("training_run_digest"), field="training_run_digest"
        ),
        run_kind=_string(payload.get("run_kind"), field="run_kind"),
        selection_proposal_digest=_optional_string(
            payload.get("selection_proposal_digest"), field="selection_proposal_digest"
        ),
        selection_authorization_digest=_optional_string(
            payload.get("selection_authorization_digest"),
            field="selection_authorization_digest",
        ),
        walk_forward_run_digest=_optional_string(
            payload.get("walk_forward_run_digest"), field="walk_forward_run_digest"
        ),
        gate_evidence_digest=_optional_string(
            payload.get("gate_evidence_digest"), field="gate_evidence_digest"
        ),
        confirmation_evidence_digest=_optional_string(
            payload.get("confirmation_evidence_digest"),
            field="confirmation_evidence_digest",
        ),
        files=tuple(files),
        created_at=created_at,
        action_size=_integer(payload.get("action_size"), field="action_size"),
        action_names=_string_tuple(payload.get("action_names"), field="action_names"),
        action_spec_digest=_optional_string(
            payload.get("action_spec_digest"), field="action_spec_digest"
        ),
        alpha_artifact_digest=_optional_string(
            payload.get("alpha_artifact_digest"), field="alpha_artifact_digest"
        ),
        factor_artifact_digest=_optional_string(
            payload.get("factor_artifact_digest"), field="factor_artifact_digest"
        ),
        normalizer_digest=_optional_string(
            payload.get("normalizer_digest"), field="normalizer_digest"
        ),
        schema_version=SERVING_BUNDLE_SCHEMA,
    )


def load_serving_bundle(root: Path) -> ServingBundle:
    root = Path(root)
    manifest_path = root / BUNDLE_MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"serving bundle manifest is missing: {manifest_path}")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest = _parse_manifest(_mapping(payload, field="bundle manifest"))
    root_resolved = root.resolve()
    declared = {BUNDLE_MANIFEST_NAME}
    normalizer = None
    declared_files = {item.path for item in manifest.files}
    if manifest.normalizer_digest is not None:
        if NORMALIZER_ARTIFACT_NAME not in declared_files:
            raise ValueError("serving bundle does not declare its normalizer sidecar")
        normalizer = load_observation_normalizer(root)
        if normalizer.digest != manifest.normalizer_digest:
            raise ValueError("serving bundle normalizer digest mismatch")
    elif NORMALIZER_ARTIFACT_NAME in declared_files:
        raise ValueError("serving bundle declares an unbound normalizer sidecar")
    release = None
    external_path = default_attestation_path(root)
    if external_path.is_file():
        release = load_release_attestation(external_path)
        release.require_bundle_identity(
            bundle_digest=manifest.bundle_digest,
            dataset_id=manifest.dataset_id,
            training_run_digest=manifest.training_run_digest,
            run_kind=manifest.run_kind,
            selection_proposal_digest=manifest.selection_proposal_digest,
            selection_authorization_digest=manifest.selection_authorization_digest,
            walk_forward_run_digest=manifest.walk_forward_run_digest,
            gate_evidence_digest=manifest.gate_evidence_digest,
            confirmation_evidence_digest=manifest.confirmation_evidence_digest,
            selected_policy_digest=manifest.policy_digest,
        )
    for file in manifest.files:
        path = root / file.path
        declared.add(file.path)
        if path.is_symlink():
            raise ValueError(f"bundle artifact cannot be a symlink: {file.path}")
        resolved = path.resolve()
        if not resolved.is_relative_to(root_resolved):
            raise ValueError(f"bundle artifact escapes bundle root: {file.path}")
        if not path.is_file():
            raise ValueError(f"bundle artifact is missing: {file.path}")
        if path.stat().st_size != file.size_bytes:
            raise ValueError(f"bundle artifact size mismatch: {file.path}")
        if _file_digest(path) != file.digest:
            raise ValueError(f"bundle artifact digest mismatch: {file.path}")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() or path.is_symlink()
    }
    undeclared = sorted(actual - declared)
    missing = sorted(declared - actual)
    if undeclared:
        raise ValueError(f"serving bundle contains undeclared files: {undeclared}")
    if missing:
        raise ValueError(f"serving bundle is missing declared files: {missing}")
    return ServingBundle(
        root=root, manifest=manifest, release=release, normalizer=normalizer
    )


__all__ = [
    "BUNDLE_MANIFEST_NAME",
    "SERVING_BUNDLE_SCHEMA",
    "BundleFile",
    "ServingBundle",
    "ServingBundleManifest",
    "load_serving_bundle",
    "write_serving_bundle_manifest",
]
