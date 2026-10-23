"""Exclusive durable arm journals and bounded read-only incremental diagnostics."""

from __future__ import annotations

import json
import os
import stat
import tempfile
import time
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from types import TracebackType
from typing import Any, BinaryIO, Literal

from .progress import (
    ArmProgressSpec,
    _closed,
    roster_payload,
    validate_progress_policy,
    validate_record,
)
from .tier_d_protocol import json_sha256
from .tier_d_records import _constant, _pairs

_CONTROL_LIMIT = 1024 * 1024
_LINE_LIMIT = 16 * 1024


def _present(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _decode(raw: bytes) -> Any:
    return json.loads(raw, object_pairs_hook=_pairs, parse_constant=_constant)


def _regular(root: Path, path: Path) -> Path:
    canonical = root.resolve(strict=True)
    if path.is_symlink() or not stat.S_ISREG(path.lstat().st_mode):
        raise ValueError("progress path must be a regular non-symlink file")
    resolved = path.resolve(strict=True)
    if not resolved.is_relative_to(canonical):
        raise ValueError("progress path resolves outside attempt root")
    return resolved


def _read_control(root: Path, path: Path) -> bytes:
    resolved = _regular(root, path)
    with resolved.open("rb") as source:
        raw = source.read(_CONTROL_LIMIT + 1)
    if len(raw) > _CONTROL_LIMIT:
        raise ValueError("progress control JSON exceeds bounded size")
    return raw


def _bindings(root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = _decode(_read_control(root, root / "protocol.json"))
    roster = _closed(
        _decode(_read_control(root, root / "native-progress-roster.json")),
        {"schema", "protocol_digest", "progress_policy", "arms"},
    )
    if type(roster["arms"]) is not list:
        raise ValueError("progress roster arms must be a list")
    specs = tuple(ArmProgressSpec.from_payload(item) for item in roster["arms"])
    if json_sha256(roster_payload(protocol, specs)) != json_sha256(roster):
        raise ValueError("progress roster/protocol binding differs")
    return protocol, roster


def publish_roster(
    root: Path, protocol: dict[str, Any], specs: Sequence[ArmProgressSpec]
) -> dict[str, Any]:
    payload = roster_payload(protocol, tuple(specs))
    if json_sha256(_decode(_read_control(root, root / "protocol.json"))) != json_sha256(
        protocol
    ):
        raise ValueError("progress protocol differs from retained attempt")
    raw = (json.dumps(payload, sort_keys=True, allow_nan=False) + "\n").encode()
    if len(raw) > _CONTROL_LIMIT:
        raise ValueError("progress control JSON exceeds bounded size")
    staging = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=root, prefix=".native-progress-roster.", suffix=".tmp", delete=False
        ) as output:
            staging = Path(output.name)
            output.write(raw)
            output.flush()
            os.fsync(output.fileno())
        # Same-filesystem hard link publishes a complete closed file atomically,
        # and fails if the immutable destination already exists. No replacement.
        os.link(staging, root / "native-progress-roster.json")
    finally:
        if staging is not None:
            staging.unlink(missing_ok=True)
    return payload


class ProgressJournal:
    def __init__(
        self,
        root: Path,
        spec: ArmProgressSpec,
        on_change: Callable[[], None] | None = None,
    ):
        if type(spec) is not ArmProgressSpec or (
            on_change is not None and not callable(on_change)
        ):
            raise ValueError(
                "progress journal requires an exact spec and optional callable"
            )
        self.root, self.spec, self.on_change = root, spec, on_change
        self._stream: BinaryIO | None = None
        self._last: dict[str, Any] | None = None
        self._broken = False

    def __enter__(self) -> ProgressJournal:
        if self._stream is not None or self._last is not None:
            raise ValueError("progress journal cannot be entered twice")
        _, roster = _bindings(self.root)
        if self.spec.payload not in roster["arms"]:
            raise ValueError("progress journal spec is foreign to frozen roster")
        self._protocol_digest = roster["protocol_digest"]
        directory = self.root / "native-progress"
        directory.mkdir(exist_ok=True)
        if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(
            self.root.resolve(strict=True)
        ):
            raise ValueError("progress journal directory resolves outside attempt root")
        self._stream = (directory / f"{self.spec.arm_id}.jsonl").open("xb")
        try:
            self._record(
                "native_started", None, self.spec.start_index, self.spec.start_index, 0
            )
        except BaseException:
            self._stream.close()
            raise
        return self

    def _record(
        self, event: str, phase: str | None, start: int, processing: int, count: int
    ) -> None:
        if self._stream is None or self._broken:
            raise ValueError("progress journal is not writable")
        raw = validate_record(
            {
                "schema": "issue810_native_progress_record_v1",
                "protocol_digest": self._protocol_digest,
                "arm_id": self.spec.arm_id,
                "plan_digest": self.spec.plan_digest,
                "event": event,
                "seq": 0 if self._last is None else self._last["seq"] + 1,
                "count": count,
                "total": self.spec.target,
                "phase": phase,
                "start_index": start,
                "processing_index": processing,
            },
            self.spec,
            self._protocol_digest,
            self._last,
        )
        data = (json.dumps(raw, sort_keys=True, allow_nan=False) + "\n").encode()
        try:
            self._stream.write(data)
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            self._broken = True
            raise
        self._last = raw
        if self.on_change is not None:
            self.on_change()

    def advance(self, phase: str, start: int, processing: int) -> None:
        if self._last is None:
            raise ValueError("progress journal has no native start")
        self._record(
            "native_advanced", phase, start, processing, self._last["count"] + 1
        )

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        trace: TracebackType | None,
    ) -> Literal[False]:
        publication_error = None
        try:
            if not self._broken and self._last is not None:
                count = self._last["count"]
                index = self.spec.start_index + count
                self._record(
                    "native_returned" if error is None else "native_failed",
                    None,
                    index,
                    index,
                    count,
                )
        except Exception as failure:
            publication_error = failure
        finally:
            if self._stream is not None:
                try:
                    self._stream.close()
                except Exception as failure:
                    publication_error = publication_error or failure
        if publication_error is not None:
            if error is None:
                raise publication_error
            try:
                setattr(
                    error, "native_progress_publication_failure", str(publication_error)
                )
            except Exception:
                pass
        return False


class ProgressReader:
    def __init__(
        self,
        root: Path,
        expected_protocol: dict[str, Any] | None = None,
        expected_policy: dict[str, Any] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ):
        self.root, self.clock = root, clock
        self._expected = (
            None if expected_protocol is None else json_sha256(expected_protocol)
        )
        self._expected_policy = (
            None
            if expected_policy is None
            else validate_progress_policy(expected_policy)
        )
        self._controls: dict[str, bytes] = {}
        self._states: dict[str, dict[str, Any]] = {}
        self._last_advance: dict[str, float] = {}
        self._offsets: dict[str, int] = {}
        self._identities: dict[str, tuple[int, int]] = {}
        self._specs: dict[str, ArmProgressSpec] = {}
        self._policy: dict[str, Any] = {}

    @property
    def states(self) -> dict[str, dict[str, Any]]:
        return {arm: dict(row) for arm, row in self._states.items()}

    def _prepare(self) -> bool:
        if not all(
            _present(self.root / name)
            for name in ("protocol.json", "native-progress-roster.json")
        ):
            if self._controls:
                raise ValueError("immutable progress control disappeared/changed")
            return False
        for name in ("protocol.json", "native-progress-roster.json"):
            path = self.root / name
            raw = _read_control(self.root, path)
            if name in self._controls and raw != self._controls[name]:
                raise ValueError("immutable progress control bytes changed")
            self._controls[name] = raw
        protocol, roster = _bindings(self.root)
        policy = validate_progress_policy(protocol["progress_policy"])
        if (self._expected is not None and json_sha256(protocol) != self._expected) or (
            self._expected_policy is not None and policy != self._expected_policy
        ):
            raise ValueError("progress protocol/policy differs from expected binding")
        self._policy, self._protocol_digest = policy, roster["protocol_digest"]
        self._specs = {
            row["arm_id"]: ArmProgressSpec.from_payload(row) for row in roster["arms"]
        }
        return True

    def poll(self) -> Iterator[dict[str, Any]]:
        if not self._prepare():
            return
        directory = self.root / "native-progress"
        if not _present(directory):
            if self._states:
                raise ValueError("observed progress journal directory disappeared")
            return
        if directory.is_symlink() or not directory.resolve(strict=True).is_relative_to(
            self.root.resolve(strict=True)
        ):
            raise ValueError("progress directory resolves outside attempt root")
        names = {f"{arm}.jsonl" for arm in self._specs}
        if any(path.name not in names for path in directory.iterdir()):
            raise ValueError("foreign progress journal path")
        for arm, spec in self._specs.items():
            path = directory / f"{arm}.jsonl"
            if not _present(path):
                if arm in self._offsets:
                    raise ValueError("observed progress journal disappeared")
                continue
            resolved = _regular(self.root, path)
            info, offset = resolved.stat(), self._offsets.get(arm, 0)
            identity = (info.st_dev, info.st_ino)
            if info.st_size < offset or (
                arm in self._identities and self._identities[arm] != identity
            ):
                raise ValueError("progress journal was replaced/truncated")
            self._identities[arm] = identity
            with resolved.open("rb") as source:
                source.seek(offset)
                while source.tell() < info.st_size:
                    line = source.readline(
                        min(_LINE_LIMIT + 1, info.st_size - source.tell())
                    )
                    if len(line) > _LINE_LIMIT:
                        raise ValueError("progress journal line exceeds bounded size")
                    if not line.endswith(b"\n"):
                        break
                    previous = self._states.get(arm)
                    record = validate_record(
                        _decode(line), spec, self._protocol_digest, previous
                    )
                    self._offsets[arm] = source.tell()
                    self._states[arm] = (
                        spec.payload | record | {"arm_start_index": spec.start_index}
                    )
                    if record["event"] in {"native_started", "native_advanced"}:
                        self._last_advance[arm] = self.clock()
                    yield dict(self._states[arm])

    def stalled(self) -> list[dict[str, Any]]:
        now = self.clock()
        return [
            row | {"idle_seconds": now - self._last_advance[arm]}
            for arm, row in self._states.items()
            if row["event"] not in {"native_returned", "native_failed"}
            and row["count"] < row["target"]
            and now - self._last_advance[arm] >= self._policy["no_progress_seconds"]
        ]
