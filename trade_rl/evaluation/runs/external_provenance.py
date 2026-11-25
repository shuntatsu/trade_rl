"""Opt-in, exact package-source closure for external research consumers.

The caller obtains the expected pin digests from independently reviewed
source/build records. This module checks the bytes; it cannot authenticate
those records or establish which Python objects have actually executed.
"""

from __future__ import annotations

import re
import stat
import zipfile
from dataclasses import dataclass, field
from hashlib import sha256
from pathlib import Path
from typing import Final

from trade_rl._validation import require_git_sha, require_sha256
from trade_rl.artifacts.hashing import content_digest
from trade_rl.artifacts.verified_file import read_verified_bytes, verified_private_copy
from trade_rl.evaluation.runs.provenance import runtime_environment_manifest

EXTERNAL_SOURCE_PROVENANCE_SCHEMA: Final = "external_source_provenance_v1"
_MAX_WHEEL_BYTES = 128 * 1024 * 1024
_MAX_PAYLOAD_BYTES = 128 * 1024 * 1024
_MAX_MEMBERS = 10_000
_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.-]*")
_REPOSITORY = re.compile(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+")
_PACKAGE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def _safe_parts(path: str) -> tuple[str, ...]:
    if type(path) is not str or len(path) > 1024:
        raise ValueError("invalid package path")
    parts = tuple(path.split("/"))
    if any(
        not _SAFE_COMPONENT.fullmatch(part)
        or part.endswith(".")
        or part.split(".")[0].upper() in _RESERVED
        or part.casefold() == "__pycache__"
        for part in parts
    ):
        raise ValueError("unsafe or unsupported package path")
    return parts


def _check_path_roster(entries: list[tuple[str, bool]]) -> None:
    """Refuse duplicate, case-alias and file/directory-prefix collisions."""
    seen: set[str] = set()
    prefixes: dict[tuple[str, ...], tuple[tuple[str, ...], bool]] = {}
    for name, is_file in entries:
        parts = _safe_parts(name)
        if name in seen:
            raise ValueError("duplicate package path")
        seen.add(name)
        for end in range(1, len(parts) + 1):
            prefix = parts[:end]
            key = tuple(part.casefold() for part in prefix)
            value = prefix, is_file and end == len(parts)
            previous = prefixes.get(key)
            if previous is not None and previous != value:
                raise ValueError("aliased package path or file/directory collision")
            prefixes[key] = value


def _check_digest(value: str, *, name: str) -> None:
    if type(value) is not str:
        raise ValueError(f"{name} must be a native SHA-256 digest")
    require_sha256(value, field=name)


def _check_size(value: int, *, maximum: int, minimum: int = 0) -> None:
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("package size outside supported bounds")


@dataclass(frozen=True, slots=True)
class PackageFilePin:
    """Exact file bytes, relative to one reviewed top-level package."""

    path: str
    size_bytes: int
    sha256: str

    def __post_init__(self) -> None:
        parts = _safe_parts(self.path)
        if parts[-1].casefold().endswith((".pyc", ".pyo")):
            raise ValueError("compiled bytecode is not admitted")
        _check_size(self.size_bytes, maximum=_MAX_PAYLOAD_BYTES)
        _check_digest(self.sha256, name="package file digest")


@dataclass(frozen=True, slots=True)
class PackageSourcePin:
    """Externally reviewed Git and built-wheel identities plus source manifest.

    Include every distributed file under the package, not just Python source.
    Git commit/tree claims must have been independently established by the caller.
    """

    repository: str
    commit: str
    tree: str
    package: str
    wheel_sha256: str
    wheel_size_bytes: int
    files: tuple[PackageFilePin, ...]

    def __post_init__(self) -> None:
        if (
            type(self.repository) is not str
            or not _REPOSITORY.fullmatch(self.repository)
            or any(p in (".", "..") for p in self.repository.split("/"))
        ):
            raise ValueError("expected an explicit owner/repository")
        if type(self.commit) is not str or type(self.tree) is not str:
            raise ValueError("Git identities must be native strings")
        require_git_sha(self.commit, field="commit")
        require_git_sha(self.tree, field="tree")
        if type(self.package) is not str or not _PACKAGE.fullmatch(self.package):
            raise ValueError("expected one top-level Python package")
        _safe_parts(self.package)
        _check_digest(self.wheel_sha256, name="wheel digest")
        _check_size(self.wheel_size_bytes, minimum=1, maximum=_MAX_WHEEL_BYTES)
        if (
            type(self.files) is not tuple
            or not 0 < len(self.files) <= _MAX_MEMBERS
            or any(type(item) is not PackageFilePin for item in self.files)
        ):
            raise ValueError("expected a bounded immutable file manifest")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or "__init__.py" not in paths:
            raise ValueError("expected a sorted regular-package file manifest")
        _check_path_roster([(path, True) for path in paths])
        _check_size(
            sum(item.size_bytes for item in self.files), maximum=_MAX_PAYLOAD_BYTES
        )


@dataclass(frozen=True, slots=True)
class ExternalPackageInput:
    """Locations for verification; not part of the canonical evidence record."""

    pin: PackageSourcePin
    wheel: Path
    installed_package: Path

    def __post_init__(self) -> None:
        if type(self.pin) is not PackageSourcePin:
            raise ValueError("expected a package source pin")
        for path in (self.wheel, self.installed_package):
            if (
                not isinstance(path, Path)
                or not path.is_absolute()
                or ".." in path.parts
            ):
                raise ValueError(
                    "verification paths must be absolute without parent traversal"
                )
        if (
            self.wheel.suffix != ".whl"
            or self.installed_package.name != self.pin.package
        ):
            raise ValueError("wheel or installed package name mismatch")


@dataclass(frozen=True, slots=True)
class ExternalSourceProvenance:
    """A verified content snapshot, not an authorization or runtime attestation."""

    core: PackageSourcePin
    research: PackageSourcePin
    runtime_config_digest: str
    runtime_environment_digest: str
    schema_version: str = field(default=EXTERNAL_SOURCE_PROVENANCE_SCHEMA, init=False)

    @property
    def digest(self) -> str:
        return content_digest(self)


def _verify_wheel(source: ExternalPackageInput) -> None:
    pin = source.pin
    manifest = {item.path: item for item in pin.files}
    with verified_private_copy(
        source.wheel,
        expected_digest=pin.wheel_sha256,
        expected_size_bytes=pin.wheel_size_bytes,
        field="external source wheel",
        filename="package.whl",
    ) as snapshot:
        try:
            with zipfile.ZipFile(snapshot) as wheel:
                members = wheel.infolist()
                if not 0 < len(members) <= _MAX_MEMBERS:
                    raise ValueError("wheel member count outside supported bounds")
                _check_path_roster(
                    [
                        (info.filename.removesuffix("/"), not info.is_dir())
                        for info in members
                    ]
                )
                observed: set[str] = set()
                metadata_roots: set[str] = set()
                metadata_files: set[str] = set()
                for info in members:
                    if info.orig_filename != info.filename:
                        raise ValueError("ambiguous wheel member name")
                    name = info.filename.removesuffix("/")
                    parts = _safe_parts(name)
                    kind = stat.S_IFMT(info.external_attr >> 16)
                    if (
                        kind not in (0, stat.S_IFDIR if info.is_dir() else stat.S_IFREG)
                        or info.flag_bits & 1
                    ):
                        raise ValueError(
                            "wheel member is not a regular unencrypted entry"
                        )
                    if parts[0] != pin.package:
                        if not parts[0].endswith(".dist-info"):
                            raise ValueError("unexpected wheel payload outside package")
                        metadata_roots.add(parts[0])
                        if not info.is_dir():
                            if len(parts) < 2:
                                raise ValueError(
                                    "wheel metadata root must be a directory"
                                )
                            metadata_files.add(name)
                        continue
                    if info.is_dir():
                        if len(parts) > 1 and not any(
                            p.startswith("/".join(parts[1:]) + "/") for p in manifest
                        ):
                            raise ValueError("wheel directory outside reviewed roster")
                        continue
                    relative = "/".join(parts[1:])
                    expected = manifest.get(relative)
                    if expected is None:
                        raise ValueError("wheel file outside reviewed roster")
                    if info.file_size != expected.size_bytes:
                        raise ValueError("wheel source size mismatch")
                    with wheel.open(info) as handle:
                        raw = handle.read(expected.size_bytes + 1)
                    if (
                        len(raw) != expected.size_bytes
                        or sha256(raw).hexdigest() != expected.sha256
                    ):
                        raise ValueError("wheel source digest mismatch")
                    observed.add(relative)
                if observed != manifest.keys() or len(metadata_roots) != 1:
                    raise ValueError("incomplete wheel package roster")
                metadata_root = next(iter(metadata_roots))
                if (
                    not {
                        f"{metadata_root}/METADATA",
                        f"{metadata_root}/WHEEL",
                        f"{metadata_root}/RECORD",
                    }
                    <= metadata_files
                ):
                    raise ValueError("incomplete wheel metadata roster")
        except zipfile.BadZipFile as exc:
            raise ValueError("invalid pinned wheel archive") from exc


def _verify_installed(source: ExternalPackageInput) -> None:
    root = source.installed_package
    if any(p.is_symlink() or p.is_junction() for p in (root, *root.parents)):
        raise ValueError("installed package must not traverse symlinks or junctions")
    if not root.is_dir():
        raise ValueError("installed package directory missing")
    files = {item.path: item for item in source.pin.files}
    directories = {
        "/".join(parts[:end])
        for path in files
        for parts in [_safe_parts(path)]
        for end in range(1, len(parts))
    }
    observed: set[str] = set()
    pending = [root]
    while pending:
        for path in pending.pop().iterdir():
            name = path.relative_to(root).as_posix()
            _safe_parts(name)
            if path.is_symlink() or path.is_junction():
                raise ValueError("installed package symlink or junction")
            mode = path.lstat().st_mode
            if stat.S_ISDIR(mode):
                if name not in directories:
                    raise ValueError("installed directory outside reviewed roster")
                pending.append(path)
                continue
            expected = files.get(name)
            if not stat.S_ISREG(mode) or expected is None:
                raise ValueError("installed file outside reviewed roster")
            read_verified_bytes(
                path,
                expected_digest=expected.sha256,
                expected_size_bytes=expected.size_bytes,
                field="installed package source",
            )
            observed.add(name)
    if observed != files.keys():
        raise ValueError("incomplete installed package roster")


def verify_external_source_provenance(
    core: ExternalPackageInput,
    research: ExternalPackageInput,
    *,
    runtime_config: object,
    expected_core_pin_digest: str,
    expected_research_pin_digest: str,
    expected_runtime_config_digest: str,
    expected_provenance_digest: str | None = None,
) -> ExternalSourceProvenance:
    """Verify two pinned wheel/source closures and bind the current runtime identity.

    Use independently trusted pin/config digests, a fresh noneditable install
    with no generated bytecode (run Python with -B), and repeat with the first
    record digest at the end of the run. Missing or extra files, even resources,
    are refused. Paths are not serialized; exact raw bytes are hashed.

    This is an observation of two explicit directories and wheel archives.
    It does not establish import resolution, metadata outside the package,
    dependency bytes, Git/approval authenticity, concurrent or temporary
    mutation during execution, or research/data/production authorization.
    """
    if (
        type(core) is not ExternalPackageInput
        or type(research) is not ExternalPackageInput
        or core.pin.package != "trade_rl"
        or research.pin.package.casefold() == core.pin.package.casefold()
        or research.pin.repository.casefold() == core.pin.repository.casefold()
    ):
        raise ValueError("expected distinct core and external research packages")
    for expected, actual, name in (
        (expected_core_pin_digest, content_digest(core.pin), "core pin digest"),
        (
            expected_research_pin_digest,
            content_digest(research.pin),
            "research pin digest",
        ),
        (
            expected_runtime_config_digest,
            content_digest(runtime_config),
            "runtime config digest",
        ),
    ):
        _check_digest(expected, name=name)
        if expected != actual:
            raise ValueError(f"{name} mismatch")
    if expected_provenance_digest is not None:
        _check_digest(expected_provenance_digest, name="provenance digest")
    for source in (core, research):
        _verify_wheel(source)
        _verify_installed(source)
    record = ExternalSourceProvenance(
        core=core.pin,
        research=research.pin,
        runtime_config_digest=expected_runtime_config_digest,
        runtime_environment_digest=content_digest(runtime_environment_manifest()),
    )
    if (
        expected_provenance_digest is not None
        and record.digest != expected_provenance_digest
    ):
        raise ValueError("provenance digest mismatch")
    return record


__all__ = [
    "EXTERNAL_SOURCE_PROVENANCE_SCHEMA",
    "ExternalPackageInput",
    "ExternalSourceProvenance",
    "PackageFilePin",
    "PackageSourcePin",
    "verify_external_source_provenance",
]
