"""Build an exact source-only packet from two immutable Git commits."""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import subprocess
import tarfile
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterator

_HEX = re.compile(r"^[0-9a-fA-F]+$")
_DIFF_FLAGS = (
    "--full-index",
    "--binary",
    "--no-renames",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--diff-algorithm=myers",
    "--no-indent-heuristic",
    "--unified=3",
    "--inter-hunk-context=0",
    "--ignore-submodules=none",
    "--src-prefix=a/",
    "--dst-prefix=b/",
)
_NAME_STATUS_FLAGS = (
    "--name-status",
    "--no-renames",
    "--no-ext-diff",
    "--no-textconv",
    "--no-color",
    "--diff-algorithm=myers",
    "--no-indent-heuristic",
    "--unified=3",
    "--inter-hunk-context=0",
    "--ignore-submodules=none",
    "-z",
)
_WINDOWS_RESERVED = frozenset(
    {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "CONIN$",
        "CONOUT$",
        *(f"COM{i}" for i in range(1, 10)),
        *(f"COM{suffix}" for suffix in "¹²³"),
        *(f"LPT{i}" for i in range(1, 10)),
        *(f"LPT{suffix}" for suffix in "¹²³"),
    }
)


@dataclass(frozen=True, slots=True)
class _TreeEntry:
    mode: str
    object_type: str
    oid: str


@dataclass(frozen=True, slots=True)
class _PatchRenderer:
    repository: Path
    git_version: str
    attribute_tree: str
    global_attributes: Path


def _git(
    repository: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> bytes:
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment.update(
        {
            "GIT_ATTR_NOSYSTEM": "1",
            "GIT_CONFIG_GLOBAL": os.devnull,
            "GIT_CONFIG_NOSYSTEM": "1",
            "GIT_NO_LAZY_FETCH": "1",
            "GIT_NO_REPLACE_OBJECTS": "1",
            "GIT_TERMINAL_PROMPT": "0",
        }
    )
    result = subprocess.run(
        ["git", *args],
        cwd=repository,
        input=input_bytes,
        check=False,
        capture_output=True,
        env=environment,
    )
    if result.returncode:
        detail = result.stderr.decode("utf-8", errors="replace").strip()
        raise ValueError(detail or f"git {args[0]} failed")
    return result.stdout


def _git_text(
    repository: Path,
    *args: str,
    input_bytes: bytes | None = None,
) -> str:
    return (
        _git(repository, *args, input_bytes=input_bytes).decode("ascii").rstrip("\r\n")
    )


def _outside_repository_and_worktrees(
    path: Path,
    repository: Path,
    worktree_roots: tuple[Path, ...],
) -> bool:
    return not (
        path == repository
        or path.is_relative_to(repository)
        or any(
            path == worktree or path.is_relative_to(worktree)
            for worktree in worktree_roots
        )
    )


@contextmanager
def _canonical_renderer(
    repository: Path,
    object_format: str,
    worktree_roots: tuple[Path, ...],
) -> Iterator[_PatchRenderer]:
    temporary_parent = Path(tempfile.gettempdir()).resolve(strict=True)
    if not _outside_repository_and_worktrees(
        temporary_parent, repository, worktree_roots
    ):
        raise ValueError("temporary renderer parent must be outside every worktree")

    object_directory_value = _git_text(repository, "rev-parse", "--git-path", "objects")
    object_directory = Path(object_directory_value)
    if not object_directory.is_absolute():
        object_directory = repository / object_directory
    object_directory = object_directory.resolve(strict=True)

    with tempfile.TemporaryDirectory(
        prefix="agent-repo-source-packet-", dir=temporary_parent
    ) as temporary_directory:
        temporary_root = Path(temporary_directory).resolve(strict=True)
        if not _outside_repository_and_worktrees(
            temporary_root, repository, worktree_roots
        ):
            raise ValueError("temporary renderer must be outside every worktree")
        renderer_repository = temporary_root / "renderer.git"
        _git(
            repository,
            "init",
            "--quiet",
            "--bare",
            "--template=",
            f"--object-format={object_format}",
            os.fspath(renderer_repository),
        )
        alternates = renderer_repository / "objects" / "info" / "alternates"
        (renderer_repository / "info").mkdir(exist_ok=True)
        alternates.write_bytes(os.fsencode(object_directory) + b"\n")
        (renderer_repository / "info" / "attributes").write_bytes(b"")
        global_attributes = temporary_root / "global-attributes"
        global_attributes.write_bytes(b"")
        attribute_tree = _git_text(
            renderer_repository,
            "hash-object",
            "-t",
            "tree",
            "-w",
            "--stdin",
            input_bytes=b"",
        )
        yield _PatchRenderer(
            repository=renderer_repository,
            git_version=_git_text(repository, "--version"),
            attribute_tree=attribute_tree,
            global_attributes=global_attributes,
        )


def _renderer_config(renderer: _PatchRenderer) -> tuple[str, ...]:
    return (
        "-c",
        "core.quotePath=true",
        "-c",
        f"core.attributesFile={renderer.global_attributes}",
    )


def _repository_root(repository: Path) -> Path:
    raw = _git(repository, "rev-parse", "--show-toplevel").rstrip(b"\r\n")
    return Path(os.fsdecode(raw)).resolve(strict=True)


def _resolve_commit(repository: Path, value: str, object_format: str) -> str:
    if object_format not in {"sha1", "sha256"}:
        raise ValueError(f"unsupported Git object format: {object_format}")
    expected_length = 40 if object_format == "sha1" else 64
    if not _HEX.fullmatch(value) or len(value) != expected_length:
        raise ValueError("base and head must be full immutable commit object IDs")
    resolved = _git_text(repository, "rev-parse", "--verify", f"{value}^{{commit}}")
    if resolved.lower() != value.lower():
        raise ValueError("commit object ID did not resolve to the exact commit")
    if _git_text(repository, "cat-file", "-t", resolved) != "commit":
        raise ValueError("base and head must identify commit objects")
    return resolved


def _object_oid(raw: bytes, object_type: str, object_format: str) -> str:
    git_hash = hashlib.new(object_format)
    git_hash.update(f"{object_type} {len(raw)}\0".encode("ascii"))
    git_hash.update(raw)
    return git_hash.hexdigest()


def _commit_metadata(
    repository: Path, oid: str, object_format: str
) -> dict[str, object]:
    raw = _git(repository, "cat-file", "commit", oid)
    if _object_oid(raw, "commit", object_format) != oid:
        raise ValueError("raw Git commit bytes did not match the exact object ID")
    header, separator, _message = raw.partition(b"\n\n")
    if not separator:
        raise ValueError("commit object has no header separator")
    tree: str | None = None
    parents: list[str] = []
    for line in header.split(b"\n"):
        if line.startswith(b"tree "):
            if tree is not None:
                raise ValueError("commit object has multiple tree headers")
            tree = line[5:].decode("ascii")
        elif line.startswith(b"parent "):
            parents.append(line[7:].decode("ascii"))
    oid_length = 40 if object_format == "sha1" else 64
    if (
        tree is None
        or len(tree) != oid_length
        or any(len(parent) != oid_length for parent in parents)
    ):
        raise ValueError("commit object has invalid tree or parent headers")
    return {"commit": oid, "tree": tree, "parents": parents}


def _tree_entries(repository: Path, commit: str) -> dict[bytes, _TreeEntry]:
    raw = _git(repository, "ls-tree", "-rz", "--full-tree", commit)
    entries: dict[bytes, _TreeEntry] = {}
    for record in raw.split(b"\0"):
        if not record:
            continue
        try:
            header, path = record.split(b"\t", 1)
            mode, object_type, oid = header.decode("ascii").split(" ")
        except (UnicodeDecodeError, ValueError) as error:
            raise ValueError("Git tree contains an invalid entry") from error
        if path in entries:
            raise ValueError("Git tree contains a duplicate path")
        entries[path] = _TreeEntry(mode, object_type, oid)
    return entries


def _safe_path(raw: bytes) -> str:
    try:
        path = raw.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("changed Git paths must be valid UTF-8") from error
    parts = path.split("/")
    if (
        not path
        or path.startswith("/")
        or "\\" in path
        or any(part in {"", ".", ".."} for part in parts)
        or any(ord(char) < 32 or ord(char) == 127 for char in path)
        or any(char in '<>:"|?*' for char in path)
        or any(part.endswith((".", " ")) for part in parts)
        or any(
            part.split(".", 1)[0].rstrip(" ").upper() in _WINDOWS_RESERVED
            for part in parts
        )
        or PurePosixPath(path).is_absolute()
        or PurePosixPath(path).as_posix() != path
    ):
        raise ValueError(f"unsafe changed Git path: {path!r}")
    return path


def _changed_paths(
    renderer: _PatchRenderer, base: str, head: str
) -> tuple[tuple[str, bytes], ...]:
    raw = _git(
        renderer.repository,
        f"--attr-source={renderer.attribute_tree}",
        *_renderer_config(renderer),
        "diff",
        *_NAME_STATUS_FLAGS,
        base,
        head,
        "--",
    )
    fields = raw.split(b"\0")
    if fields and not fields[-1]:
        fields.pop()
    if len(fields) % 2:
        raise ValueError("Git returned an invalid name-status stream")
    result: list[tuple[str, bytes]] = []
    seen_paths: set[bytes] = set()
    windows_names: set[str] = set()
    for index in range(0, len(fields), 2):
        status = fields[index].decode("ascii")
        path = fields[index + 1]
        if status not in {"A", "D", "M", "T"}:
            raise ValueError(f"unsupported Git diff status: {status}")
        safe_path = _safe_path(path)
        folded_path = safe_path.casefold()
        if path in seen_paths or folded_path in windows_names:
            raise ValueError("changed Git paths are not uniquely portable")
        seen_paths.add(path)
        windows_names.add(folded_path)
        result.append((status, path))
    return tuple(sorted(result, key=lambda item: item[1]))


def _entry_payload(entry: _TreeEntry) -> dict[str, str]:
    return {"mode": entry.mode, "type": entry.object_type, "oid": entry.oid}


def _raw_blob(repository: Path, oid: str, object_format: str) -> bytes:
    raw = _git(repository, "cat-file", "blob", oid)
    if _object_oid(raw, "blob", object_format) != oid:
        raise ValueError("raw Git blob bytes did not match the tree object ID")
    return raw


def _make_tar(members: dict[str, tuple[bytes, int]]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as archive:
        for name, (raw, mode) in sorted(members.items()):
            info = tarfile.TarInfo(name=f"files/{name}")
            info.size = len(raw)
            info.mode = mode
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            archive.addfile(info, io.BytesIO(raw))
    return buffer.getvalue()


def _verify_tar(raw_tar: bytes, expected: dict[str, bytes]) -> None:
    try:
        with tarfile.open(fileobj=io.BytesIO(raw_tar), mode="r:") as archive:
            members = archive.getmembers()
            if [member.name for member in members] != sorted(expected):
                raise ValueError("source TAR member inventory mismatch")
            for member in members:
                if not member.isfile():
                    raise ValueError("source TAR contains a non-regular member")
                stream = archive.extractfile(member)
                if stream is None or stream.read() != expected[member.name]:
                    raise ValueError("source TAR member bytes do not match Git blobs")
    except tarfile.TarError as error:
        raise ValueError("source TAR could not be reopened") from error


def _new_output_directory(
    repository: Path,
    output: Path,
    worktree_roots: tuple[Path, ...],
) -> Path:
    candidate = Path(
        os.path.abspath(output if output.is_absolute() else Path.cwd() / output)
    )
    cursor = Path(candidate.anchor)
    for part in candidate.parts[1:-1]:
        cursor = cursor / part
        if cursor.is_symlink() or getattr(cursor, "is_junction", lambda: False)():
            raise ValueError("output path may not contain symlinks or junctions")
    parent = candidate.parent
    if not parent.is_dir():
        raise ValueError("output parent must be an existing directory")
    resolved_parent = parent.resolve(strict=True)
    target = resolved_parent / candidate.name
    if target.is_relative_to(repository) or target == repository:
        raise ValueError("source packet output must be outside the repository")
    if not _outside_repository_and_worktrees(target, repository, worktree_roots):
        raise ValueError(
            "source packet output must be outside every registered worktree"
        )
    if target.exists() or target.is_symlink():
        raise ValueError("source packet output directory already exists")
    return target


def _registered_worktree_roots(repository: Path) -> tuple[Path, ...]:
    raw = _git(repository, "worktree", "list", "--porcelain", "-z")
    roots: set[Path] = set()
    for record in raw.split(b"\0\0"):
        if not record:
            continue
        fields = record.rstrip(b"\0").split(b"\0")
        worktree_field = fields[0]
        prefix = b"worktree "
        if not worktree_field.startswith(prefix) or len(worktree_field) == len(prefix):
            raise ValueError("Git returned an invalid registered worktree inventory")
        root = Path(os.fsdecode(worktree_field[len(prefix) :]))
        if not root.is_absolute():
            raise ValueError("Git returned a non-absolute registered worktree path")
        try:
            roots.add(root.resolve(strict=False))
        except (OSError, RuntimeError) as error:
            raise ValueError(
                "registered worktree path could not be resolved"
            ) from error
    if not roots:
        raise ValueError("Git returned no registered worktree roots")
    return tuple(sorted(roots, key=os.fspath))


def create_source_packet(
    repository: Path,
    *,
    base: str,
    head: str,
    output: Path,
) -> dict[str, str]:
    """Write a deterministic packet whose payload bytes come directly from Git."""

    root = _repository_root(repository)
    worktree_roots = _registered_worktree_roots(root)
    target = _new_output_directory(root, output, worktree_roots)
    object_format = _git_text(root, "rev-parse", "--show-object-format")
    base_oid = _resolve_commit(root, base, object_format)
    head_oid = _resolve_commit(root, head, object_format)
    base_metadata = _commit_metadata(root, base_oid, object_format)
    head_metadata = _commit_metadata(root, head_oid, object_format)
    base_entries = _tree_entries(root, base_oid)
    head_entries = _tree_entries(root, head_oid)
    with _canonical_renderer(root, object_format, worktree_roots) as renderer:
        paths = _changed_paths(renderer, base_oid, head_oid)
        patch = _git(
            renderer.repository,
            f"--attr-source={renderer.attribute_tree}",
            *_renderer_config(renderer),
            "diff",
            *_DIFF_FLAGS,
            base_oid,
            head_oid,
            "--",
        )
        git_version = renderer.git_version
        attribute_tree = renderer.attribute_tree

    inventory: list[dict[str, object]] = []
    members: dict[str, tuple[bytes, int]] = {}
    expected_tar: dict[str, bytes] = {}
    for status, path_bytes in paths:
        path = _safe_path(path_bytes)
        old_entry = base_entries.get(path_bytes)
        new_entry = head_entries.get(path_bytes)
        if (
            (status == "A" and (old_entry is not None or new_entry is None))
            or (status == "D" and (old_entry is None or new_entry is not None))
            or (status in {"M", "T"} and (old_entry is None or new_entry is None))
        ):
            raise ValueError("Git name-status inventory disagrees with commit trees")
        for entry in (old_entry, new_entry):
            if entry is not None and (
                entry.mode not in {"100644", "100755"} or entry.object_type != "blob"
            ):
                raise ValueError("changed symlink and gitlink entries are unsupported")

        new_payload: dict[str, object] | None = None
        if new_entry is not None:
            raw_blob = _raw_blob(root, new_entry.oid, object_format)
            tar_name = f"files/{path}"
            expected_tar[tar_name] = raw_blob
            members[path] = (raw_blob, int(new_entry.mode, 8))
            new_payload = {
                **_entry_payload(new_entry),
                "size": len(raw_blob),
                "sha256": hashlib.sha256(raw_blob).hexdigest(),
                "tar_member": tar_name,
            }
        inventory.append(
            {
                "path": path,
                "status": status,
                "base": None if old_entry is None else _entry_payload(old_entry),
                "head": new_payload,
            }
        )

    diff_flags = _DIFF_FLAGS
    name_status_flags = _NAME_STATUS_FLAGS
    source_tar = _make_tar(members)
    _verify_tar(source_tar, expected_tar)
    manifest = {
        "schema_version": "agent_repo_source_packet_v2",
        "scope": "source_transport_only",
        "object_format": object_format,
        "base": base_metadata,
        "head": head_metadata,
        "diff": {
            "path": "source.patch",
            "flags": list(diff_flags),
            "pathspec": "all tracked paths",
            "size": len(patch),
            "sha256": hashlib.sha256(patch).hexdigest(),
            "renderer": {
                "implementation": "git diff in an isolated bare repository",
                "git_version": git_version,
                "config": {
                    "core.quotePath": "true",
                    "core.attributesFile": "empty renderer-owned file",
                },
                "attributes": {
                    "source": "empty-tree",
                    "tree": attribute_tree,
                    "info_attributes": "empty",
                    "global_attributes": "empty",
                    "system_attributes": "disabled",
                },
                "global_options": [f"--attr-source={attribute_tree}"],
                "environment": {
                    "global_config": "disabled",
                    "system_config": "disabled",
                    "replace_refs": "disabled",
                    "lazy_fetch": "disabled",
                },
                "object_access": "temporary object database with source objects as an alternate",
                "name_status_flags": list(name_status_flags),
                "cross_version_byte_identity": "not_claimed",
            },
        },
        "source_tar": {
            "path": "source.tar",
            "format": "pax",
            "members": sorted(expected_tar),
            "size": len(source_tar),
            "sha256": hashlib.sha256(source_tar).hexdigest(),
        },
        "changed_paths": inventory,
    }
    manifest_raw = (
        json.dumps(manifest, ensure_ascii=True, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")

    target.mkdir(exist_ok=False)
    (target / "source.patch").write_bytes(patch)
    (target / "source.tar").write_bytes(source_tar)
    (target / "manifest.json").write_bytes(manifest_raw)
    return {
        "output_directory": str(target),
        "manifest_sha256": hashlib.sha256(manifest_raw).hexdigest(),
        "patch_sha256": hashlib.sha256(patch).hexdigest(),
        "tar_sha256": hashlib.sha256(source_tar).hexdigest(),
    }


__all__ = ["create_source_packet"]
