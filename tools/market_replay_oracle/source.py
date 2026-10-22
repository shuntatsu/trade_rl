"""Static, byte-pinned verification of the supported native source tree.

This module does not import or execute native trading code. Its result attests
only to the source bytes selected by a caller-provided Git HEAD and snapshot
pin; it is not a runtime or research authorization attestation.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

SOURCE_ROOTS = ("trade_rl", "tools/tier_d_nonrl")
MAX_SOURCE_FILES = 4096
MAX_SOURCE_FILE_BYTES = 2 * 1024 * 1024
MAX_TOTAL_SOURCE_BYTES = 16 * 1024 * 1024
MAX_WALK_ENTRIES = 100_000

# These fingerprints bind the static policy derivation used by the supported
# profile. model.py has a bounded set of reviewed source variants with the same
# execution_policy_digest AST; broader order/event/replay behavior is not thereby
# asserted equivalent.
SUPPORTED_PROFILE_SOURCES: dict[str, frozenset[str]] = {
    "tools/tier_d_nonrl/tier_d_protocol.py": frozenset(
        {"9d679316cc4ffe614bdd8237271610c3168915f4b66993a16010b2fa812c6641"}
    ),
    "tools/tier_d_nonrl/tier_d_native.py": frozenset(
        {"dcf6dacf2d82ecec6ef957d5df13006fa4df972cb96e3d2d2ae03042afa1658e"}
    ),
    "trade_rl/simulation/execution.py": frozenset(
        {"6ce371ba1ab4d7221c214a0d722062d3d1e7791caf5da71704fecdd06f648d63"}
    ),
    "trade_rl/strategies/rl/allocation_policy.py": frozenset(
        {"627d146cffb240947b5d0322af613c7d7ddcc9f13dc01a322a9c88b328856576"}
    ),
    "trade_rl/strategies/rl/allocation_recipe_v2.py": frozenset(
        {"fee8318929cc9fd9c16f0eaae82caff96d3658a0f75d131ac1792b637311f2da"}
    ),
    "trade_rl/simulation/orders/model.py": frozenset(
        {
            "cb1c2f9e71a8ee556ffb9058f8564109cb7699e102a729b5a73afaa80c2c3a79",
            "07a01341e51d2ac54aa0ea316d59fd6ed58046309b40f73db51390503abe741d",
            "13ccc2feaaffa6225c9a8a1eda6902e130a77cfec5ef0f87383bdd0352ddecf4",
        }
    ),
    "trade_rl/artifacts/hashing.py": frozenset(
        {"cb1593dda363fea3276184a699ce730380646ad764969fec47196161d388b4fa"}
    ),
    "trade_rl/artifacts/canonical.py": frozenset(
        {"c827358950f642f6258b0e00f1c9c16ce430cb493e53f3591ba7eed570952e3f"}
    ),
    "trade_rl/risk/pretrade.py": frozenset(
        {"2124d298b18fe5332fbe0ff07b03d67ece1251029cb18c32a1f67203e0f9cc49"}
    ),
    "trade_rl/strategies/allocation.py": frozenset(
        {"7294e78ba9c3d95501fd5ef9e1c87ac1e2ca95b971c70654e2c0913c4ea9f014"}
    ),
    "trade_rl/strategies/allocation_action.py": frozenset(
        {"07792791cb4bf6e6e0b2a9c375b420f6f547c766b7cbe03315e697da596f3d43"}
    ),
    "trade_rl/evaluation/rl_allocation/env.py": frozenset(
        {"e566ffb69c5268fa4b2e8ea379f48dc4fe0d99308961f02d5a878cc5f24583a1"}
    ),
    "trade_rl/evaluation/rl_allocation/binding.py": frozenset(
        {"9750be33cda4a6622ee60fb8c3f01b67978e69f333186bc30017ecec9a0ba791"}
    ),
    "trade_rl/evaluation/objectives/clock.py": frozenset(
        {"d4f8ff01593779c04a477b628048113bc70b44c0a3819b748b8749766a846a3e"}
    ),
    "trade_rl/strategies/rl/allocation_observation_v2.py": frozenset(
        {"27e66b0b59ca15606a5ca4a081ea0254a01475afacc7fad9e578948c7a1c79a2"}
    ),
    "trade_rl/data/market_order_rules.py": frozenset(
        {"a13570349844f7a1b838273e340171c04d97b7313a0818236551f49e49023e7c"}
    ),
}


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _valid_digest(value: object, width: int, label: str) -> str:
    if (
        not isinstance(value, str)
        or re.fullmatch(rf"[0-9a-f]{{{width}}}", value) is None
    ):
        raise ValueError(
            f"{label} must be lowercase hexadecimal SHA with width {width}"
        )
    return value


@dataclass(frozen=True, slots=True)
class NativeSourcePin:
    """Caller-provided exact Git HEAD and canonical raw-source snapshot pin."""

    head_sha: str
    snapshot_sha256: str

    def __post_init__(self) -> None:
        _valid_digest(self.head_sha, 40, "HEAD SHA")
        _valid_digest(self.snapshot_sha256, 64, "snapshot SHA-256")


def _git(root: Path, *args: str, input_bytes: bytes | None = None) -> bytes:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), *args],
            input=input_bytes,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        raise ValueError(
            f"Git command could not verify source tree: {args[0]}"
        ) from error
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(f"Git source verification failed ({args[0]}): {detail[:300]}")
    return result.stdout


def _is_reparse(info: os.stat_result) -> bool:
    return stat.S_ISLNK(info.st_mode) or bool(
        getattr(info, "st_file_attributes", 0) & 0x400
    )


def _require_directory(path: Path) -> None:
    try:
        info = path.lstat()
    except OSError as error:
        raise ValueError(f"source directory is missing: {path}") from error
    if _is_reparse(info) or not stat.S_ISDIR(info.st_mode):
        raise ValueError(
            f"source directory must not be a symlink or reparse point: {path}"
        )


def _validate_root(root: Path) -> Path:
    root = Path(os.path.abspath(root))
    for path in reversed((root, *root.parents)):
        _require_directory(path)
    top = _git(root, "rev-parse", "--show-toplevel").decode("utf-8", "strict").strip()
    if os.path.normcase(os.path.normpath(top)) != os.path.normcase(
        os.path.normpath(str(root))
    ):
        raise ValueError("root must be the Git repository top level")
    return root


def _safe_relative(raw: bytes) -> str:
    try:
        value = raw.decode("utf-8", "strict")
    except UnicodeDecodeError as error:
        raise ValueError("source path is not valid UTF-8") from error
    pieces = value.split("/")
    if (
        not value
        or value.startswith("/")
        or "\\" in value
        or any(part in {"", ".", ".."} for part in pieces)
        or any(ord(char) < 32 or char in '<>:"|?*' for char in value)
        or any(part.endswith((".", " ")) for part in pieces)
        or any(
            part.split(".", 1)[0].upper()
            in {
                "CON",
                "PRN",
                "AUX",
                "NUL",
                *(f"COM{i}" for i in range(1, 10)),
                *(f"LPT{i}" for i in range(1, 10)),
            }
            for part in pieces
        )
    ):
        raise ValueError("source path is unsafe or not portable")
    if not any(value.startswith(f"{prefix}/") for prefix in SOURCE_ROOTS):
        raise ValueError("Git returned a path outside native source namespaces")
    return value


def _tree_entries(root: Path) -> list[tuple[str, str, int]]:
    raw = _git(
        root, "ls-tree", "-r", "-l", "-z", "--full-tree", "HEAD", "--", *SOURCE_ROOTS
    )
    entries: list[tuple[str, str, int]] = []
    total = 0
    for record in raw.split(b"\0"):
        if not record:
            continue
        metadata, separator, path_raw = record.partition(b"\t")
        fields = metadata.decode("ascii", "strict").split()
        if not separator or len(fields) != 4:
            raise ValueError("Git source tree entry is malformed")
        mode, object_type, oid, size_text = fields
        path = _safe_relative(path_raw)
        if mode not in {"100644", "100755"} or object_type != "blob":
            raise ValueError(
                f"native source tree contains a non-regular Git entry: {path}"
            )
        size = int(size_text)
        if size < 0 or size > MAX_SOURCE_FILE_BYTES:
            raise ValueError(f"native source file exceeds bounded size: {path}")
        total += size
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("native source tree exceeds bounded total size")
        if not re.fullmatch(r"[0-9a-f]{40,64}", oid):
            raise ValueError("Git source blob object ID is malformed")
        entries.append((path, oid, size))
        if len(entries) > MAX_SOURCE_FILES:
            raise ValueError("native source tree contains too many tracked files")
    if not entries:
        raise ValueError("native source tree is empty")
    return sorted(entries, key=lambda entry: entry[0])


def _blob_contents(root: Path, entries: list[tuple[str, str, int]]) -> dict[str, bytes]:
    request = b"".join(oid.encode("ascii") + b"\n" for _, oid, _ in entries)
    raw = _git(root, "cat-file", "--batch", input_bytes=request)
    offset = 0
    contents: dict[str, bytes] = {}
    for path, expected_oid, expected_size in entries:
        newline = raw.find(b"\n", offset)
        if newline < 0:
            raise ValueError("Git blob batch response is truncated")
        header = raw[offset:newline].decode("ascii", "strict").split()
        if len(header) != 3:
            raise ValueError("Git blob batch header is malformed")
        oid, object_type, size_text = header
        size = int(size_text)
        if oid != expected_oid or object_type != "blob" or size != expected_size:
            raise ValueError(f"Git source blob identity differs: {path}")
        start = newline + 1
        end = start + size
        if end >= len(raw) or raw[end : end + 1] != b"\n":
            raise ValueError(f"Git source blob bytes are truncated: {path}")
        contents[path] = raw[start:end]
        offset = end + 1
    if offset != len(raw):
        raise ValueError("Git blob batch response has trailing bytes")
    return contents


def _check_namespace_tree(root: Path) -> None:
    seen = 0
    for namespace in SOURCE_ROOTS:
        base = root
        for component in namespace.split("/"):
            base = base / component
            _require_directory(base)
        pending = [base]
        while pending:
            directory = pending.pop()
            try:
                with os.scandir(directory) as children:
                    for child in children:
                        seen += 1
                        if seen > MAX_WALK_ENTRIES:
                            raise ValueError(
                                "native source namespace has too many filesystem entries"
                            )
                        info = child.stat(follow_symlinks=False)
                        if _is_reparse(info):
                            raise ValueError(
                                f"native source namespace contains a link: {child.path}"
                            )
                        if stat.S_ISDIR(info.st_mode):
                            if child.name != "__pycache__":
                                pending.append(Path(child.path))
                        elif not stat.S_ISREG(info.st_mode):
                            raise ValueError(
                                f"native source namespace contains a special file: {child.path}"
                            )
            except OSError as error:
                raise ValueError(
                    f"native source namespace cannot be inspected: {directory}"
                ) from error


def _untracked_python_paths(root: Path, *, ignored: bool) -> list[str]:
    args = ["ls-files", "--others"]
    if ignored:
        args.append("--ignored")
    args.extend(("--exclude-standard", "-z", "--", *SOURCE_ROOTS))
    raw = _git(root, *args)
    rejected: list[str] = []
    for path_raw in raw.split(b"\0"):
        if not path_raw:
            continue
        path = _safe_relative(path_raw)
        pieces = path.split("/")
        if "__pycache__" not in pieces and Path(path).suffix.casefold() == ".py":
            rejected.append(path)
    return rejected


def _check_source_cleanliness(root: Path) -> None:
    staged = subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "diff",
            "--cached",
            "--quiet",
            "HEAD",
            "--",
            *SOURCE_ROOTS,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=30,
    )
    if staged.returncode != 0:
        raise ValueError("staged changes exist in native source namespaces")
    _check_namespace_tree(root)
    for ignored in (False, True):
        found = _untracked_python_paths(root, ignored=ignored)
        if found:
            label = "ignored" if ignored else "untracked"
            raise ValueError(
                f"{label} Python source exists in native namespaces: {found[0]}"
            )


def _read_regular(path: Path, relative: str) -> bytes:
    try:
        before = path.lstat()
    except OSError as error:
        raise ValueError(f"tracked native source is missing: {relative}") from error
    if _is_reparse(before) or not stat.S_ISREG(before.st_mode):
        raise ValueError(f"tracked native source is not a regular file: {relative}")
    if before.st_size > MAX_SOURCE_FILE_BYTES:
        raise ValueError(f"tracked native source exceeds bounded size: {relative}")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
        with os.fdopen(descriptor, "rb") as stream:
            opened = os.fstat(stream.fileno())
            if _is_reparse(opened) or not stat.S_ISREG(opened.st_mode):
                raise ValueError(
                    f"tracked native source changed type while opening: {relative}"
                )
            raw = stream.read(MAX_SOURCE_FILE_BYTES + 1)
            after_open = os.fstat(stream.fileno())
        after = path.lstat()
    except OSError as error:
        raise ValueError(f"tracked native source cannot be read: {relative}") from error
    if (
        len(raw) > MAX_SOURCE_FILE_BYTES
        or len(raw) != before.st_size
        or before.st_size != opened.st_size
        or opened.st_size != after_open.st_size
        or after_open.st_size != after.st_size
        or _is_reparse(after)
        or not stat.S_ISREG(after.st_mode)
    ):
        raise ValueError(f"tracked native source changed while reading: {relative}")
    return raw


def verify_native_source(root: Path, pin: NativeSourcePin) -> dict[str, Any]:
    """Verify bounded native source bytes against exact Git and caller pins."""
    if not isinstance(pin, NativeSourcePin):
        raise ValueError("NativeSourcePin is required")
    root = _validate_root(Path(root))
    head = _git(root, "rev-parse", "--verify", "HEAD").decode("ascii", "strict").strip()
    if head != pin.head_sha:
        raise ValueError("Git HEAD differs from pinned native source HEAD")
    _check_source_cleanliness(root)
    entries = _tree_entries(root)
    blobs = _blob_contents(root, entries)
    rows: list[dict[str, Any]] = []
    snapshot_rows: list[dict[str, Any]] = []
    profile_blobs: dict[str, str] = {}
    total = 0
    for relative, oid, expected_size in entries:
        path = root.joinpath(*PurePosixPath(relative).parts)
        for parent in path.parents:
            if parent == root:
                break
            _require_directory(parent)
        git_blob = blobs[relative]
        if len(git_blob) != expected_size:
            raise ValueError(f"Git source blob byte count differs: {relative}")
        raw = _read_regular(path, relative)
        total += len(raw)
        if total > MAX_TOTAL_SOURCE_BYTES:
            raise ValueError("native source worktree exceeds bounded total size")
        normalized = raw.replace(b"\r\n", b"\n")
        if normalized != git_blob:
            raise ValueError(f"live native source differs from Git blob: {relative}")
        raw_sha = _sha(raw)
        blob_sha = _sha(git_blob)
        if relative in SUPPORTED_PROFILE_SOURCES:
            if blob_sha not in SUPPORTED_PROFILE_SOURCES[relative]:
                raise ValueError(
                    f"unsupported static profile source fingerprint: {relative}"
                )
            profile_blobs[relative] = blob_sha
        row = {
            "path": relative,
            "bytes": len(raw),
            "sha256": raw_sha,
            "git_blob_oid": oid,
            "git_blob_bytes": len(git_blob),
            "git_blob_sha256": blob_sha,
            "normalized_sha256": _sha(normalized),
            "crlf_count": raw.count(b"\r\n"),
            "lf_count": raw.count(b"\n") - raw.count(b"\r\n"),
        }
        rows.append(row)
        snapshot_rows.append({"path": relative, "bytes": len(raw), "sha256": raw_sha})
    missing_profiles = set(SUPPORTED_PROFILE_SOURCES) - profile_blobs.keys()
    if missing_profiles:
        raise ValueError(
            f"supported static profile source is missing: {sorted(missing_profiles)[0]}"
        )
    snapshot = _sha(_canonical(snapshot_rows))
    if snapshot != pin.snapshot_sha256:
        raise ValueError("native source snapshot digest differs from caller pin")
    return {
        "head_sha": head,
        "snapshot_sha256": snapshot,
        "files": rows,
        "profile_source_sha256": profile_blobs,
        "scope": "STATIC_SOURCE_BYTES_NOT_RUNTIME_ATTESTATION",
    }
