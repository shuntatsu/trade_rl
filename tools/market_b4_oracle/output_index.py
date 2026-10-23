"""Bind H's original, self-excluding output manifest to its exact file tree.

This is a caller-pinned file inventory. It authenticates neither a native launch
nor the manifest's authorship, completeness as a financial run, or consumption.
The caller must obtain its manifest pin from the independent outer witness.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Literal

from tools.market_b4_oracle.receipt_stream import (
    PinnedReceiptRows,
    ReceiptLimits,
    ReceiptPin,
)

_MANIFEST = "output-manifest.json"
_MANIFEST_LIMIT = 4 * 1024**2
_FILE_LIMIT = 8 * 1024**3
_FILE_COUNT = 16384
_TREE_ENTRIES = 32768
_CHUNK = 65536
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_DEVICES = {"CON", "PRN", "AUX", "NUL"} | {
    prefix + str(number) for prefix in ("COM", "LPT") for number in range(1, 10)
}


@dataclass(frozen=True, slots=True)
class ArtifactPin:
    bytes: int
    sha256: str

    def __post_init__(self) -> None:
        if type(self.bytes) is not int or not 0 <= self.bytes <= _FILE_LIMIT:
            raise ValueError("artifact pin requires a bounded nonnegative integer")
        if type(self.sha256) is not str or not _SHA256.fullmatch(self.sha256):
            raise ValueError("artifact pin requires a lowercase SHA-256")


def _name(value: object) -> str:
    if type(value) is not str or not value or "\\" in value or ":" in value:
        raise ValueError("artifact path must be a root-relative POSIX path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or path.as_posix() != value
        or any(
            part in {".", ".."}
            or part.endswith((".", " "))
            or part.split(".", 1)[0].upper() in _DEVICES
            or any(unicodedata.category(char) == "Cc" for char in part)
            for part in path.parts
        )
    ):
        raise ValueError("artifact path has an unsafe or normalized alias")
    if path.name.casefold() == _MANIFEST.casefold():
        raise ValueError("output manifest must be self-excluding")
    return value


def _information(path: Path, *, directory: bool = False) -> os.stat_result:
    information = path.lstat()
    if (
        stat.S_ISLNK(information.st_mode)
        or getattr(information, "st_file_attributes", 0) & 0x400
    ):
        raise ValueError("output tree contains a link or reparse point")
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(information.st_mode):
        raise ValueError("output tree contains an unsupported file type")
    if not directory and information.st_nlink != 1:
        raise ValueError("output file has a hard-link alias")
    return information


def _root(path: Path) -> Path:
    if not path.is_absolute() or ".." in path.parts:
        raise ValueError("attempt root must be an absolute non-aliased path")
    for parent in (*reversed(path.parents), path):
        _information(parent, directory=True)
    return path


def _identity(information: os.stat_result) -> tuple[int, int]:
    return information.st_dev, information.st_ino


def _stamp(information: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        information.st_dev,
        information.st_ino,
        information.st_size,
        information.st_mtime_ns,
        information.st_ctime_ns,
    )


def _inventory(root: Path) -> dict[str, tuple[int, int, int, int, int]]:
    result = {}
    pending = [root]
    total = 0
    while pending:
        directory = pending.pop()
        _information(directory, directory=True)
        with os.scandir(directory) as entries:
            for entry in entries:
                total += 1
                if total > _TREE_ENTRIES:
                    raise ValueError("output tree exceeds its entry limit")
                path = Path(entry.path)
                information = path.lstat()
                if stat.S_ISDIR(information.st_mode):
                    _information(path, directory=True)
                    pending.append(path)
                else:
                    information = _information(path)
                    name = path.relative_to(root).as_posix()
                    if name != _MANIFEST:
                        _name(name)
                    result[name] = _stamp(information)
                    if len(result) > _FILE_COUNT + 1:
                        raise ValueError("output tree exceeds its file limit")
    if len({value[:2] for value in result.values()}) != len(result):
        raise ValueError("output files alias the same file identity")
    return result


def _read(
    path: Path,
    pin: ArtifactPin,
    identity: tuple[int, int, int, int, int],
    *,
    retain: bool = False,
) -> bytes:
    before = _information(path)
    if _stamp(before) != identity or before.st_size != pin.bytes:
        raise ValueError("output file identity or byte length changed")
    digest, size = hashlib.sha256(), 0
    chunks = []
    with path.open("rb") as stream:
        opened = _stamp(os.fstat(stream.fileno()))
        # Python 3.12 Windows exposes different ctime meanings through lstat
        # and fstat. Compare each interface's ctime to its own opening value;
        # device, inode, size and nanosecond mtime agree across interfaces.
        if opened[:4] != identity[:4]:
            raise ValueError("output file replaced while opening")
        while chunk := stream.read(_CHUNK):
            size += len(chunk)
            if size > pin.bytes:
                raise ValueError("output file grew beyond its pin")
            digest.update(chunk)
            if retain:
                chunks.append(chunk)
        if _stamp(os.fstat(stream.fileno())) != opened:
            raise ValueError("output file identity changed while reading")
    if (
        size != pin.bytes
        or digest.hexdigest() != pin.sha256
        or _stamp(_information(path)) != identity
    ):
        raise ValueError("original output file differs from its pin")
    return b"".join(chunks)


def _pairs(values: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in values:
        if key in result:
            raise ValueError("duplicate output-manifest JSON key")
        result[key] = value
    return result


def _constant(value: str) -> object:
    raise ValueError("output manifest has a non-finite JSON value")


def _entries(raw: bytes) -> tuple[tuple[str, ArtifactPin], ...]:
    value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    if type(value) is not dict or not 0 < len(value) <= _FILE_COUNT:
        raise ValueError("output manifest requires a bounded nonempty file mapping")
    entries = []
    aliases = set()
    total = 0
    for name, row in value.items():
        name = _name(name)
        if name.casefold() in aliases:
            raise ValueError("output manifest has case aliases")
        aliases.add(name.casefold())
        if type(row) is not dict or set(row) != {"bytes", "sha256"}:
            raise ValueError("output manifest file pin has missing or foreign fields")
        pin = ArtifactPin(row["bytes"], row["sha256"])
        total += pin.bytes
        if total > _FILE_LIMIT:
            raise ValueError("output files exceed their aggregate read budget")
        entries.append((name, pin))
    return tuple(sorted(entries))


@dataclass(frozen=True, slots=True)
class NativeOutputIndex:
    """Immutable file pins, subject to explicit recheck; never a locked archive."""

    _root: Path
    manifest_pin: ArtifactPin
    _entries: tuple[tuple[str, ArtifactPin], ...]
    _identities: tuple[tuple[str, tuple[int, int, int, int, int]], ...]
    _root_identity: tuple[int, int]

    @property
    def files(self) -> tuple[str, ...]:
        return tuple(name for name, _ in self._entries)

    @property
    def scope(self) -> str:
        return "PINNED_NATIVE_OUTPUT_FILES_ONLY"

    def pin(self, name: str) -> ArtifactPin:
        try:
            return dict(self._entries)[_name(name)]
        except KeyError as error:
            raise ValueError("artifact absent from pinned output index") from error

    def path(self, name: str) -> Path:
        """Return a confined path; consumers must verify its pin when reading."""
        self.pin(name)
        try:
            _root(self._root)
            if (
                _identity(_information(self._root, directory=True))
                != self._root_identity
            ):
                raise ValueError("original attempt root has been replaced")
            path = self._root / name
            for parent in path.parents:
                if parent == self._root:
                    break
                _information(parent, directory=True)
            if _stamp(_information(path)) != dict(self._identities)[name]:
                raise ValueError("indexed artifact has been replaced")
            return path
        except OSError as error:
            raise ValueError("indexed artifact is no longer present") from error

    def read_pinned_bytes(self, name: str, *, maximum: int) -> bytes:
        """Read one indexed file with a caller-specific hard byte ceiling."""
        if type(maximum) is not int or maximum < 0:
            raise ValueError("indexed read maximum must be a nonnegative integer")
        pin = self.pin(name)
        if pin.bytes > maximum:
            raise ValueError("indexed artifact exceeds its caller read limit")
        identity = dict(self._identities)[_name(name)]
        return _read(self.path(name), pin, identity, retain=True)

    def receipt_rows(
        self,
        name: str,
        *,
        phase: Literal["actor", "reserve"],
        limits: ReceiptLimits = ReceiptLimits(),
    ) -> PinnedReceiptRows:
        """Scan a manifest-bound original; both phases retain the same raw pin.

        The returned iterator still yields provisional rows. Its own complete
        structural scan, outer-witness checks and financial replay are separate.
        """
        pin = self.pin(name)
        return PinnedReceiptRows(
            self.path(name),
            ReceiptPin(pin.bytes, pin.sha256),
            phase=phase,
            limits=limits,
        )

    def recheck(self) -> None:
        """Check all observed original files again, publishing no partial index."""
        try:
            root = _root(self._root)
            identities = dict(self._identities)
            if (
                _identity(_information(root, directory=True)) != self._root_identity
                or _inventory(root) != identities
            ):
                raise ValueError("output file roster or identity changed")
            _read(root / _MANIFEST, self.manifest_pin, identities[_MANIFEST])
            for name, pin in self._entries:
                _read(self.path(name), pin, identities[name])
            _read(root / _MANIFEST, self.manifest_pin, identities[_MANIFEST])
            if _inventory(root) != identities:
                raise ValueError("output tree changed during verification")
        except OSError as error:
            raise ValueError("output tree is missing or changed") from error


def load_pinned_output_index(
    root: Path, manifest_pin: ArtifactPin
) -> NativeOutputIndex:
    """Verify H's flat output-manifest format; its independent pin is mandatory.

    Limits here bound reading only; they do not change native execution budgets.
    Hashes and OS identities detect observed changes, without freezing future
    files or proving the process which produced them. No file is extracted.
    """
    if type(manifest_pin) is not ArtifactPin:
        raise ValueError("output index requires an exact caller-supplied pin")
    manifest_pin.__post_init__()
    if not 0 < manifest_pin.bytes <= _MANIFEST_LIMIT:
        raise ValueError("output manifest exceeds its bounded metadata limit")
    try:
        root = _root(Path(root))
        root_identity = _identity(_information(root, directory=True))
        identities = _inventory(root)
        if _MANIFEST not in identities:
            raise ValueError("original output manifest is absent")
        raw = _read(root / _MANIFEST, manifest_pin, identities[_MANIFEST], retain=True)
        entries = _entries(raw)
        if {name for name, _ in entries} | {_MANIFEST} != set(identities):
            raise ValueError("output manifest does not cover the exact file roster")
        result = NativeOutputIndex(
            root,
            manifest_pin,
            entries,
            tuple(sorted(identities.items())),
            root_identity,
        )
        result.recheck()
        return result
    except (OSError, UnicodeError, json.JSONDecodeError, RecursionError) as error:
        raise ValueError("invalid or changed original output manifest") from error


def capture_output_manifest_pin(root: Path) -> ArtifactPin:
    """Observe the current manifest through a bounded read for an outer owner.

    This returns only the exact bytes seen at this path. It does not prove which
    process wrote them; callers must bind the result to an independently
    observed attempt before treating it as an output index.
    """
    try:
        root = _root(Path(root))
        path = root / _MANIFEST
        before = _information(path)
        if not 0 < before.st_size <= _MANIFEST_LIMIT:
            raise ValueError("output manifest exceeds its bounded metadata limit")
        identity = _stamp(before)
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            opened = _stamp(os.fstat(stream.fileno()))
            if opened[:4] != identity[:4]:
                raise ValueError("output manifest changed while opening")
            while chunk := stream.read(_CHUNK):
                size += len(chunk)
                if size > min(before.st_size, _MANIFEST_LIMIT):
                    raise ValueError("output manifest grew beyond its bound")
                digest.update(chunk)
            if _stamp(os.fstat(stream.fileno())) != opened:
                raise ValueError("output manifest changed while reading")
        if size != before.st_size or _stamp(_information(path)) != identity:
            raise ValueError("output manifest changed during outer readback")
        return ArtifactPin(size, digest.hexdigest())
    except OSError as error:
        raise ValueError(
            "output manifest is unavailable for bounded pinning"
        ) from error
