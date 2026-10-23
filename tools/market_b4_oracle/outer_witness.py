"""Read back the root-owned start/finish records around one observed attempt.

This module contains only immutable evidence parsing. Process creation and OS
observation live in ``tools.market_b4_outer_runner`` outside this comparison
package.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import stat
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_MAX_WITNESS_BYTES = 1024 * 1024
_ROOT_OBSERVER_TOKEN = object()
_FROZEN_H_RUNTIME_PIN_TOKEN = object()


@dataclass(frozen=True, slots=True)
class WitnessFilePin:
    """Exact raw-byte pin for one exclusive root-owned witness record."""

    byte_count: int
    sha256: str

    def __post_init__(self) -> None:
        if (
            type(self.byte_count) is not int
            or not 0 < self.byte_count <= _MAX_WITNESS_BYTES
        ):
            raise ValueError("witness pin byte count is outside its bounded limit")
        if (
            type(self.sha256) is not str
            or len(self.sha256) != 64
            or any(character not in "0123456789abcdef" for character in self.sha256)
        ):
            raise ValueError("witness pin requires a lowercase SHA-256")


@dataclass(frozen=True, slots=True)
class FrozenHRuntimePin:
    """Independent, read-only runtime facts for the dedicated frozen-H venv."""

    source_root: Path
    python_executable: Path
    python_executable_sha256: str
    runtime_manifest_canonical: bytes
    runtime_environment_sha256: str
    probe_facts_canonical: bytes
    probe_facts_sha256: str
    _runtime_pin_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            self._runtime_pin_token is not _FROZEN_H_RUNTIME_PIN_TOKEN
            or not isinstance(self.source_root, Path)
            or not self.source_root.is_absolute()
            or not isinstance(self.python_executable, Path)
            or not self.python_executable.is_absolute()
            or self.python_executable
            != self.source_root / ".venv" / "Scripts" / "python.exe"
            or not _is_sha256(self.python_executable_sha256)
            or not _is_sha256(self.runtime_environment_sha256)
            or not _is_sha256(self.probe_facts_sha256)
            or type(self.runtime_manifest_canonical) is not bytes
            or not 0 < len(self.runtime_manifest_canonical) <= 64 * 1024
            or type(self.probe_facts_canonical) is not bytes
            or not 0 < len(self.probe_facts_canonical) <= 64 * 1024
        ):
            raise ValueError("frozen-H runtime pin is malformed or caller-created")
        runtime = _parse_canonical_object(self.runtime_manifest_canonical)
        probe = _parse_canonical_object(self.probe_facts_canonical)
        if (
            hashlib.sha256(self.runtime_manifest_canonical).hexdigest()
            != self.runtime_environment_sha256
            or hashlib.sha256(self.probe_facts_canonical).hexdigest()
            != self.probe_facts_sha256
            or runtime.get("schema_version") != "candidate_run_runtime_v1"
            or type(runtime.get("python")) is not dict
            or type(runtime.get("os")) is not dict
            or runtime["os"].get("family") != "Windows"
            or runtime["python"].get("implementation") != "CPython"
            or not str(runtime["python"].get("version", "")).startswith("3.12.")
            or type(runtime.get("packages")) is not dict
            or any(
                type(runtime["packages"].get(name)) is not str
                for name in ("trade-rl", "numpy", "gymnasium", "lightgbm")
            )
            or probe.get("schema") != "issue810_frozen_h_runtime_probe_v1"
            or probe.get("source_root") != str(self.source_root)
            or probe.get("python_executable") != str(self.python_executable)
            or probe.get("python_executable_sha256") != self.python_executable_sha256
            or probe.get("runtime_environment_sha256")
            != self.runtime_environment_sha256
        ):
            raise ValueError("frozen-H runtime facts do not match their exact pins")

    def authority_payload(self) -> dict[str, Any]:
        """Return only the bounded runtime facts bound by this pin."""
        runtime = _parse_canonical_object(self.runtime_manifest_canonical)
        probe = _parse_canonical_object(self.probe_facts_canonical)
        return {
            "schema": "issue810_frozen_h_runtime_pin_v1",
            "source_root": str(self.source_root),
            "python_executable": str(self.python_executable),
            "python_executable_sha256": self.python_executable_sha256,
            "runtime_environment": runtime,
            "runtime_environment_sha256": self.runtime_environment_sha256,
            "probe_facts": probe,
            "probe_facts_sha256": self.probe_facts_sha256,
        }


def _make_frozen_h_runtime_pin(
    *,
    source_root: Path,
    python_executable: Path,
    python_executable_sha256: str,
    runtime_manifest_canonical: bytes,
    runtime_environment_sha256: str,
    probe_facts_canonical: bytes,
    probe_facts_sha256: str,
) -> FrozenHRuntimePin:
    """Create a runtime pin after the flat observer verifies the H venv."""
    return FrozenHRuntimePin(
        source_root=source_root,
        python_executable=python_executable,
        python_executable_sha256=python_executable_sha256,
        runtime_manifest_canonical=runtime_manifest_canonical,
        runtime_environment_sha256=runtime_environment_sha256,
        probe_facts_canonical=probe_facts_canonical,
        probe_facts_sha256=probe_facts_sha256,
        _runtime_pin_token=_FROZEN_H_RUNTIME_PIN_TOKEN,
    )


@dataclass(frozen=True, slots=True)
class RootObservedAttempt:
    """A start/finish pair emitted by the root process that owned the child."""

    start_path: Path
    finish_path: Path
    repository_head: str
    implementation_source_sha256: str
    _observer_token: object = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.start_path, Path)
            or not isinstance(self.finish_path, Path)
            or not self.start_path.is_absolute()
            or not self.finish_path.is_absolute()
            or len(self.repository_head) != 40
            or any(
                character not in "0123456789abcdef"
                for character in self.repository_head
            )
            or len(self.implementation_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in self.implementation_source_sha256
            )
            or self.start_path == self.finish_path
            or self._observer_token is not _ROOT_OBSERVER_TOKEN
        ):
            raise ValueError("root-observed witness must come from the outer observer")

    def readback(
        self,
        *,
        expected_start_pin: WitnessFilePin,
        expected_finish_pin: WitnessFilePin,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if (
            type(expected_start_pin) is not WitnessFilePin
            or type(expected_finish_pin) is not WitnessFilePin
        ):
            raise ValueError("root-observed start and finish pins are required")
        start = _read_pinned_record(self.start_path, expected_start_pin)
        finish = _read_pinned_record(self.finish_path, expected_finish_pin)
        _validate_pair(start, finish)
        return start, finish


def _make_root_observed_attempt(
    start_path: Path,
    finish_path: Path,
    *,
    repository_head: str,
    implementation_source_sha256: str,
) -> RootObservedAttempt:
    """Private factory used only after the outer process observer completes."""
    return RootObservedAttempt(
        Path(start_path).absolute(),
        Path(finish_path).absolute(),
        repository_head,
        implementation_source_sha256,
        _ROOT_OBSERVER_TOKEN,
    )


def _source_components(
    path: Path, *, directory: bool | None = False
) -> tuple[tuple[Path, os.stat_result], ...]:
    if not isinstance(path, Path) or not path.is_absolute() or ".." in path.parts:
        raise ValueError("comparison source path is not an absolute non-aliased path")
    components: list[tuple[Path, os.stat_result]] = []
    ordered = (*reversed(path.parents), path)
    for index, component in enumerate(ordered):
        information = component.lstat()
        if stat.S_ISLNK(information.st_mode) or (
            getattr(information, "st_file_attributes", 0) & 0x400
        ):
            raise ValueError("comparison source tree contains a link or reparse point")
        final_component = index == len(ordered) - 1
        expected_directory = not final_component or directory is True
        if expected_directory and not stat.S_ISDIR(information.st_mode):
            raise ValueError("comparison source tree contains an unsupported entry")
        if (
            final_component
            and directory is None
            and not (
                stat.S_ISDIR(information.st_mode) or stat.S_ISREG(information.st_mode)
            )
        ):
            raise ValueError("comparison source tree contains an unsupported entry")
        if (
            final_component
            and directory is False
            and not stat.S_ISREG(information.st_mode)
        ):
            raise ValueError("comparison source tree contains an unsupported entry")
        if stat.S_ISREG(information.st_mode) and information.st_nlink != 1:
            raise ValueError("comparison source file has a hard-link alias")
        components.append((component, information))
    return tuple(components)


def _component_stamp(
    components: tuple[tuple[Path, os.stat_result], ...],
) -> tuple[tuple[str, int, int, int, int, int, int], ...]:
    return tuple(
        (
            str(path),
            information.st_dev,
            information.st_ino,
            information.st_mode,
            information.st_size,
            information.st_mtime_ns,
            information.st_ctime_ns,
        )
        for path, information in components
    )


def _read_source_file(path: Path) -> tuple[os.stat_result, bytes]:
    components_before = _source_components(path, directory=False)
    before = components_before[-1][1]
    maximum = 4 * 1024**2
    if before.st_size > maximum:
        raise ValueError(
            "comparison implementation source exceeds its per-file byte limit"
        )
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or (
                opened.st_dev,
                opened.st_ino,
                opened.st_size,
                opened.st_mtime_ns,
            ) != (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ):
                raise ValueError("comparison source changed while opening")
            chunks = []
            size = 0
            while chunk := stream.read(min(65536, maximum - size + 1)):
                size += len(chunk)
                if size > maximum or size > before.st_size:
                    raise ValueError("comparison source grew beyond its bounded read")
                chunks.append(chunk)
            after_open = os.fstat(stream.fileno())
        components_after = _source_components(path)
    except OSError as error:
        raise ValueError("comparison source could not be read safely") from error
    if (
        size != before.st_size
        or not _same_file_stamp(opened, after_open)
        or _component_stamp(components_before) != _component_stamp(components_after)
    ):
        raise ValueError("comparison implementation source changed while pinning")
    return before, b"".join(chunks)


def implementation_source_identity(root: Path | None = None) -> dict[str, Any]:
    """Pin comparison source bytes and runtime without process or Git access."""
    repository = (
        Path(root).absolute()
        if root is not None
        else Path(__file__).absolute().parents[2]
    )
    if root is not None and ".." in Path(root).parts:
        raise ValueError("comparison repository path cannot contain parent aliases")
    _source_components(repository, directory=True)
    roots = (
        repository / "tools" / "market_b4_oracle",
        repository / "tools" / "market_replay_oracle",
        repository / "trade_rl" / "artifacts",
        repository / "tools" / "market_b4_outer_runner.py",
        repository / "tools" / "tier_d_nonrl" / "tier_d_process_tree.py",
    )
    paths: list[Path] = []
    entry_count = 0
    for source_root in roots:
        information = _source_components(source_root, directory=None)[-1][1]
        if stat.S_ISREG(information.st_mode):
            paths.append(source_root)
            continue
        if not stat.S_ISDIR(information.st_mode):
            raise ValueError("comparison implementation source is incomplete")
        pending = [source_root]
        while pending:
            directory = pending.pop()
            _source_components(directory, directory=True)
            with os.scandir(directory) as directory_entries:
                children = sorted(
                    (Path(entry.path) for entry in directory_entries), key=str
                )
            for child in children:
                entry_count += 1
                if entry_count > 16384:
                    raise ValueError(
                        "comparison implementation source roster is too large"
                    )
                child_information = _source_components(child, directory=None)[-1][1]
                if stat.S_ISDIR(child_information.st_mode):
                    if child.name != "__pycache__":
                        pending.append(child)
                elif stat.S_ISREG(child_information.st_mode):
                    if child.suffix == ".py":
                        paths.append(child)
                else:
                    raise ValueError(
                        "comparison implementation source has an unsupported entry"
                    )
    relative_paths = [path.relative_to(repository).as_posix() for path in paths]
    if len(relative_paths) != len(set(relative_paths)) or len(paths) > 2048:
        raise ValueError("comparison implementation source roster is invalid")
    entries = []
    total = 0
    for path, relative in sorted(zip(paths, relative_paths, strict=True)):
        before, raw = _read_source_file(path)
        if before.st_size > 4 * 1024**2:
            raise ValueError(
                "comparison implementation source is not bounded regular Python"
            )
        total += len(raw)
        if total > 64 * 1024**2:
            raise ValueError("comparison implementation source exceeds its byte budget")
        entries.append(
            {
                "path": relative,
                "bytes": len(raw),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    packages = {
        "numpy": importlib.metadata.version("numpy"),
        "trade-rl": importlib.metadata.version("trade-rl"),
    }
    runtime = {
        "python_executable": str(Path(sys.executable).resolve()),
        "python_implementation": sys.implementation.name,
        "python_version": sys.version,
        "platform": platform.platform(),
        "packages": packages,
    }
    closure = {"files": entries, "runtime": runtime}
    return {
        "closure_sha256": hashlib.sha256(_canonical(closure)).hexdigest(),
        "files": entries,
        "runtime": runtime,
    }


def _pin(raw: bytes) -> WitnessFilePin:
    return WitnessFilePin(len(raw), hashlib.sha256(raw).hexdigest())


def _is_sha256(value: object) -> bool:
    return (
        type(value) is str
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _parse_canonical_object(raw: bytes) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("frozen-H runtime facts are not canonical JSON") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("frozen-H runtime facts are not canonical JSON objects")
    return value


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _pairs(rows: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in rows:
        if key in result:
            raise ValueError("root-owned witness has a duplicate JSON key")
        result[key] = value
    return result


def _constant(_: str) -> None:
    raise ValueError("root-owned witness has a non-finite JSON value")


def _same_file_stamp(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        left.st_dev,
        left.st_ino,
        left.st_size,
        left.st_mtime_ns,
    ) == (
        right.st_dev,
        right.st_ino,
        right.st_size,
        right.st_mtime_ns,
    )


def _read_pinned_record(path: Path, pin: WitnessFilePin) -> dict[str, Any]:
    if not isinstance(path, Path) or type(pin) is not WitnessFilePin:
        raise ValueError("root-owned witness path or expected pin is malformed")
    try:
        components_before = _source_components(path, directory=False)
        before = components_before[-1][1]
        if (
            not stat.S_ISREG(before.st_mode)
            or stat.S_ISLNK(before.st_mode)
            or getattr(before, "st_file_attributes", 0) & 0x400
            or before.st_size != pin.byte_count
            or before.st_size > _MAX_WITNESS_BYTES
        ):
            raise ValueError("root-owned witness file is not a bounded regular file")
        flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if not stat.S_ISREG(opened.st_mode) or not _same_file_stamp(before, opened):
                raise ValueError("root-owned witness changed while opening")
            raw = stream.read(_MAX_WITNESS_BYTES + 1)
            after_open = os.fstat(stream.fileno())
        components_after = _source_components(path, directory=False)
    except OSError as error:
        raise ValueError("root-owned witness file is unavailable") from error
    if (
        len(raw) != pin.byte_count
        or len(raw) > _MAX_WITNESS_BYTES
        or not _same_file_stamp(opened, after_open)
        or _component_stamp(components_before) != _component_stamp(components_after)
        or hashlib.sha256(raw).hexdigest() != pin.sha256
    ):
        raise ValueError("root-owned witness file differs from its external pin")
    try:
        value = json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("root-owned witness is not strict UTF-8 JSON") from error
    if type(value) is not dict or _canonical(value) != raw:
        raise ValueError("root-owned witness is not a canonical JSON object")
    return value


def _integer(value: object, name: str, *, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise ValueError(f"root-owned witness {name} is invalid")
    return value


def _sha(value: object, name: str) -> str:
    if (
        type(value) is not str
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"root-owned witness {name} is invalid")
    return value


def _validate_pair(start: dict[str, Any], finish: dict[str, Any]) -> None:
    if (
        start.get("schema") != "issue810_root_owned_attempt_start_v1"
        or finish.get("schema") != "issue810_root_owned_attempt_finish_v1"
    ):
        raise ValueError("root-owned witness schemas are unsupported")
    attempt_uuid = start.get("attempt_uuid")
    if (
        type(attempt_uuid) is not str
        or len(attempt_uuid) != 32
        or any(character not in "0123456789abcdef" for character in attempt_uuid)
        or finish.get("attempt_uuid") != attempt_uuid
    ):
        raise ValueError("root-owned witness UUID differs between start and finish")
    start_clock, finish_clock = start.get("clock"), finish.get("clock")
    if type(start_clock) is not dict or type(finish_clock) is not dict:
        raise ValueError("root-owned witness clock is missing")
    start_wall = _integer(start_clock.get("wall_utc_ns"), "start wall clock")
    start_mono = _integer(start_clock.get("monotonic_ns"), "start monotonic clock")
    finish_wall = _integer(finish_clock.get("wall_utc_ns"), "finish wall clock")
    finish_mono = _integer(finish_clock.get("monotonic_ns"), "finish monotonic clock")
    if finish_wall < start_wall or finish_mono < start_mono:
        raise ValueError("root-owned witness finish clock precedes its start")
    argv = start.get("argv")
    if (
        type(argv) is not list
        or not argv
        or any(type(value) is not str or not value for value in argv)
        or finish.get("argv_sha256") != hashlib.sha256(_canonical(argv)).hexdigest()
    ):
        raise ValueError("root-owned witness argv differs between start and finish")
    output_root = start.get("output_root")
    if (
        type(output_root) is not str
        or not Path(output_root).is_absolute()
        or finish.get("output_root") != output_root
    ):
        raise ValueError("root-owned witness output root differs")
    source_start, source_finish = (
        start.get("implementation_source"),
        finish.get("implementation_source"),
    )
    if (
        type(source_start) is not dict
        or type(source_finish) is not dict
        or _sha(source_start.get("closure_sha256"), "start source closure")
        != _sha(source_finish.get("closure_sha256"), "finish source closure")
        or source_start.get("repository_head") != source_finish.get("repository_head")
        or source_start.get("working_tree_clean")
        != source_finish.get("working_tree_clean")
        or source_start.get("git_status_sha256")
        != source_finish.get("git_status_sha256")
    ):
        raise ValueError("root-owned witness implementation source changed")
    child = finish.get("child")
    if type(child) is not dict:
        raise ValueError("root-owned witness child observation is missing")
    root_pid = _integer(child.get("pid"), "child PID", minimum=1)
    birth = _integer(child.get("birth_ticks"), "child birth identity", minimum=1)
    _integer(child.get("exit_code"), "child exit code")
    identities = child.get("observed_process_identities")
    if (
        type(child.get("owned_tree_empty")) is not bool
        or child.get("owned_tree_empty") is not True
        or type(identities) is not list
        or [root_pid, birth] not in identities
    ):
        raise ValueError(
            "root-owned witness did not observe a fully stopped process tree"
        )
    manifest = finish.get("output_manifest")
    if type(manifest) is not dict:
        raise ValueError("root-owned witness output-manifest pin is missing")
    _integer(manifest.get("bytes"), "output manifest bytes", minimum=1)
    _sha(manifest.get("sha256"), "output manifest SHA-256")
