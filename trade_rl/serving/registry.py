"""Atomic activation registry for validated serving bundles."""

from __future__ import annotations

import json
import os
import shutil
from collections.abc import Mapping
from pathlib import Path

from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.domain.common import require_sha256
from trade_rl.release.attestation import ReleaseAttestation, default_attestation_path
from trade_rl.serving.bundle import ServingBundle, load_serving_bundle


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)
    _fsync_directory(path.parent)


class ServingRegistry:
    """Validated immutable bundle versions with an atomic active pointer."""

    def __init__(
        self,
        root: Path,
        *,
        allow_unreleased: bool = False,
        trusted_attestation_keys: Mapping[str, bytes | bytearray | memoryview]
        | None = None,
    ) -> None:
        self.root = root
        self.allow_unreleased = allow_unreleased
        self.trusted_attestation_keys = dict(trusted_attestation_keys or {})
        self.staging_root = root / ".staging"
        self.versions_root = root / "versions"
        self.active_pointer = root / "active.json"
        for path in (self.root, self.staging_root, self.versions_root):
            path.mkdir(parents=True, exist_ok=True)

    def _require_activatable(self, bundle: ServingBundle) -> None:
        if bundle.release is None:
            if not self.allow_unreleased:
                raise ValueError(
                    "serving bundle requires a verified release attestation"
                )
            return
        if isinstance(bundle.release, ReleaseAttestation):
            bundle.release.verify(self.trusted_attestation_keys)
            return
        if not self.allow_unreleased:
            raise ValueError(
                "legacy release metadata is not an authenticated signed attestation"
            )

    def activate(self, source: Path) -> ServingBundle:
        """Validate a source bundle before copying and replacing active identity."""

        source_bundle = load_serving_bundle(source)
        self._require_activatable(source_bundle)
        digest = source_bundle.manifest.bundle_digest
        require_sha256(digest, field="bundle_digest")
        destination = self.versions_root / digest

        if destination.exists():
            installed = load_serving_bundle(destination)
            self._require_activatable(installed)
            if installed.manifest.bundle_digest != digest:
                raise ValueError(
                    "installed bundle digest does not match directory identity"
                )
        else:
            stage = self.staging_root / digest
            stage_attestation = default_attestation_path(stage)
            destination_attestation = default_attestation_path(destination)
            source_attestation = default_attestation_path(source)
            if stage.exists():
                shutil.rmtree(stage)
            stage_attestation.unlink(missing_ok=True)
            try:
                shutil.copytree(source, stage)
                if source_attestation.is_file():
                    shutil.copy2(source_attestation, stage_attestation)
                staged = load_serving_bundle(stage)
                self._require_activatable(staged)
                if staged.manifest.bundle_digest != digest:
                    raise ValueError(
                        "staged bundle digest changed during registry copy"
                    )
                os.replace(stage, destination)
                if stage_attestation.is_file():
                    os.replace(stage_attestation, destination_attestation)
                _fsync_directory(self.versions_root)
                installed = load_serving_bundle(destination)
            except Exception:
                shutil.rmtree(stage, ignore_errors=True)
                stage_attestation.unlink(missing_ok=True)
                raise

        pointer = {
            "bundle_digest": digest,
            "path": destination.relative_to(self.root).as_posix(),
            "schema": "serving_registry_pointer_v1",
        }
        _atomic_write(self.active_pointer, canonical_json_bytes(pointer))
        return installed

    def active_bundle(self) -> ServingBundle:
        """Load and revalidate the currently active bundle."""

        if not self.active_pointer.is_file():
            raise FileNotFoundError("serving registry has no active bundle")
        payload = json.loads(self.active_pointer.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("active registry pointer must be a mapping")
        digest = payload.get("bundle_digest")
        relative_path = payload.get("path")
        schema = payload.get("schema")
        if not isinstance(digest, str):
            raise ValueError("active registry bundle_digest must be a string")
        require_sha256(digest, field="bundle_digest")
        if not isinstance(relative_path, str):
            raise ValueError("active registry path must be a string")
        if schema != "serving_registry_pointer_v1":
            raise ValueError("active registry pointer schema is unsupported")
        path = self.root / relative_path
        resolved_root = self.root.resolve()
        resolved_path = path.resolve()
        if resolved_root not in resolved_path.parents:
            raise ValueError("active registry pointer escapes the registry root")
        bundle = load_serving_bundle(path)
        self._require_activatable(bundle)
        if bundle.manifest.bundle_digest != digest:
            raise ValueError("active registry pointer digest mismatch")
        return bundle
