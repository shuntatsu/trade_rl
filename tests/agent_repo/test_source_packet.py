from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tarfile
from io import BytesIO
from pathlib import Path

import pytest

from tools.agent_repo.source_packet import _safe_path

ROOT = Path(__file__).resolve().parents[2]


def _git(
    root: Path,
    *args: str,
    text: bool = True,
    input_bytes: bytes | None = None,
) -> str | bytes:
    result = subprocess.run(
        ["git", *args],
        cwd=root,
        input=input_bytes,
        check=True,
        capture_output=True,
        text=text,
    )
    return result.stdout.strip() if text else result.stdout


def _write(root: Path, relative: str, value: bytes) -> None:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value)


def _repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")
    _git(root, "config", "core.autocrlf", "true")
    _write(root, ".gitattributes", b"*.txt text eol=crlf\n")
    _write(root, "tracked.txt", b"alpha\nbeta\n")
    _write(root, "deleted.txt", b"remove me\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    base = str(_git(root, "rev-parse", "HEAD"))
    _write(root, "tracked.txt", b"alpha\r\nbeta\r\n")
    assert (root / "tracked.txt").read_bytes() == b"alpha\r\nbeta\r\n"

    _git(root, "switch", "-c", "side")
    _write(root, "side.txt", b"second parent\n")
    _git(root, "add", "side.txt")
    _git(root, "commit", "-m", "side parent")
    side_parent = str(_git(root, "rev-parse", "HEAD"))
    _git(root, "switch", "main")

    _write(root, "tracked.txt", b"alpha\r\nbeta\r\ngamma\r\n")
    (root / "deleted.txt").unlink()
    _write(root, "binary.dat", b"\x00raw\xff\x01")
    _git(root, "add", "-A")
    _git(root, "commit", "-m", "main parent")
    main_parent = str(_git(root, "rev-parse", "HEAD"))
    _git(root, "merge", "--no-ff", "--no-edit", "side")
    head = str(_git(root, "rev-parse", "HEAD"))
    assert str(_git(root, "rev-list", "--parents", "-n", "1", head)).split()[1:] == [
        main_parent,
        side_parent,
    ]
    return base, head


def _run(root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "source-packet", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _tar_member(raw: bytes, name: str) -> bytes:
    with tarfile.open(fileobj=BytesIO(raw), mode="r:") as archive:
        member = archive.getmember(name)
        stream = archive.extractfile(member)
        assert stream is not None
        return stream.read()


def test_source_packet_binds_raw_git_bytes_and_exact_diff_metadata(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    base, head = _repository(repository)
    before = str(_git(repository, "status", "--porcelain=v1"))
    first_output = tmp_path / "packet-one"

    result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(first_output),
    )

    assert result.returncode == 0, result.stderr
    receipt = json.loads(result.stdout)
    manifest_raw = (first_output / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    patch = (first_output / "source.patch").read_bytes()
    source_tar = (first_output / "source.tar").read_bytes()
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_raw).hexdigest()
    assert receipt["patch_sha256"] == hashlib.sha256(patch).hexdigest()
    assert receipt["tar_sha256"] == hashlib.sha256(source_tar).hexdigest()

    assert manifest["base"]["commit"] == base
    assert manifest["head"]["commit"] == head
    assert (
        manifest["head"]["parents"]
        == str(_git(repository, "rev-list", "--parents", "-n", "1", head)).split()[1:]
    )
    assert [entry["path"] for entry in manifest["changed_paths"]] == [
        "binary.dat",
        "deleted.txt",
        "side.txt",
        "tracked.txt",
    ]
    deleted = next(
        entry for entry in manifest["changed_paths"] if entry["path"] == "deleted.txt"
    )
    assert deleted["status"] == "D"
    assert deleted["head"] is None

    head_oid = str(_git(repository, "rev-parse", f"{head}:tracked.txt"))
    raw_blob = _git(repository, "cat-file", "blob", head_oid, text=False)
    assert isinstance(raw_blob, bytes)
    tracked = next(
        entry for entry in manifest["changed_paths"] if entry["path"] == "tracked.txt"
    )
    assert tracked["head"]["oid"] == head_oid
    assert tracked["head"]["size"] == len(raw_blob)
    assert tracked["head"]["sha256"] == hashlib.sha256(raw_blob).hexdigest()
    assert (
        _tar_member(source_tar, "files/tracked.txt")
        == raw_blob
        == b"alpha\nbeta\ngamma\n"
    )
    assert (repository / "tracked.txt").read_bytes() == b"alpha\r\nbeta\r\ngamma\r\n"
    with tarfile.open(fileobj=BytesIO(source_tar), mode="r:") as archive:
        assert "files/deleted.txt" not in archive.getnames()
        assert archive.getnames() == [
            "files/binary.dat",
            "files/side.txt",
            "files/tracked.txt",
        ]

    assert patch.startswith(b"diff --git ")
    assert b"GIT binary patch" in patch
    binary_oid = str(_git(repository, "rev-parse", f"{head}:binary.dat"))
    binary_index = next(
        line
        for line in patch.splitlines()
        if line.startswith(b"index ") and binary_oid.encode() in line
    )
    assert binary_index.split()[1] == f"{'0' * len(binary_oid)}..{binary_oid}".encode()
    if os.name == "nt":
        git_archive = _git(
            repository,
            "archive",
            "--format=tar",
            head,
            "--",
            "tracked.txt",
            text=False,
        )
        assert isinstance(git_archive, bytes)
        assert _tar_member(git_archive, "tracked.txt") != raw_blob

    second_output = tmp_path / "packet-two"
    repeated = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(second_output),
    )
    assert repeated.returncode == 0, repeated.stderr
    for name in ("manifest.json", "source.patch", "source.tar"):
        assert (second_output / name).read_bytes() == (first_output / name).read_bytes()
    assert str(_git(repository, "status", "--porcelain=v1")) == before


def test_source_packet_rejects_moving_refs_and_existing_or_in_repo_output(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    base, head = _repository(repository)
    inside = repository / "packet"
    existing = tmp_path / "existing"
    existing.mkdir()
    marker = existing / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    for base_ref, output in (
        ("main", tmp_path / "bad-ref"),
        (base, inside),
        (base, existing),
    ):
        result = _run(
            repository,
            "--base",
            base_ref,
            "--head",
            head,
            "--output",
            str(output),
        )
        assert result.returncode != 0
        assert result.stdout == ""

    assert not inside.exists()
    assert marker.read_text(encoding="utf-8") == "keep"
    assert str(_git(repository, "status", "--porcelain=v1")) == ""


def test_source_packet_reads_exact_commit_objects_when_replace_refs_exist(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    base, head = _repository(repository)
    base_tree = str(_git(repository, "rev-parse", f"{base}^{{tree}}"))
    replacement_tree = str(_git(repository, "rev-parse", f"{head}^{{tree}}"))
    replacement = str(
        _git(repository, "commit-tree", replacement_tree, "-m", "replacement")
    )
    _git(repository, "replace", base, replacement)
    output = tmp_path / "packet"

    result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(output),
    )

    assert result.returncode == 0, result.stderr
    manifest = json.loads((output / "manifest.json").read_bytes())
    assert manifest["base"]["tree"] == base_tree
    assert [entry["path"] for entry in manifest["changed_paths"]]
    assert manifest["changed_paths"]


def test_source_packet_rejects_changed_symlink_mode(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    base, _ = _repository(repository)
    link_blob = _git(
        repository,
        "hash-object",
        "-w",
        "--stdin",
        text=False,
        input_bytes=b"unsafe-target",
    )
    assert isinstance(link_blob, bytes)
    output = tmp_path / "packet"

    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"120000,{link_blob.decode('ascii').strip()},unsafe-link",
    )
    _git(repository, "commit", "-m", "symlink mode")
    head = str(_git(repository, "rev-parse", "HEAD"))
    result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert not output.exists()


def test_source_packet_rejects_changed_gitlink_mode(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    base, _ = _repository(repository)
    _git(
        repository,
        "update-index",
        "--add",
        "--cacheinfo",
        f"160000,{base},submodule",
    )
    _git(repository, "commit", "-m", "gitlink mode")
    head = str(_git(repository, "rev-parse", "HEAD"))
    output = tmp_path / "packet"

    result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(output),
    )

    assert result.returncode != 0
    assert result.stdout == ""
    assert not output.exists()


@pytest.mark.parametrize(
    "path",
    (b"../escape.txt", b"/absolute.txt", b"folder\\escape.txt", b"unsafe:stream"),
)
def test_source_packet_rejects_nonportable_git_paths(path: bytes) -> None:
    with pytest.raises(ValueError, match="unsafe changed Git path"):
        _safe_path(path)
