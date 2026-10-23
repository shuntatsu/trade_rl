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


def _run(
    root: Path,
    *args: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    env = dict(os.environ)
    current = env.get("PYTHONPATH")
    env["PYTHONPATH"] = str(ROOT) if not current else f"{ROOT}{os.pathsep}{current}"
    if extra_env is not None:
        env.update(extra_env)
    return subprocess.run(
        [sys.executable, "-m", "tools.agent_repo", "source-packet", *args],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def _renderer_repository(root: Path) -> tuple[str, str]:
    root.mkdir()
    _git(root, "init", "-b", "main")
    _git(root, "config", "user.name", "Test User")
    _git(root, "config", "user.email", "test@example.com")

    context_lines = [f"line {index:02d}\n" for index in range(40)]
    inter_hunk_lines = [f"entry {index:02d}\n" for index in range(30)]
    algorithm_before = ["D", "D", "C", "C", "A", "B", "A", "C", "B", "C", "C", "B"]
    algorithm_after = ["D", "D", "C", "C", "C", "A", "B", "B", "C", "B", "C", "B"]
    indent_before = """\
        nested()
        return 1
    def method(self):
        nested()
    else:
        call_c()
        call_d()
class C:
        nested()
        call_b()
        nested()
    else:
        return 2
    cleanup()
        call_b()
class C:
        call_b()
        call_d()
        call_c()
        call_c()
    cleanup()
        nested()""".splitlines()
    indent_after = indent_before.copy()
    indent_after[14:14] = ["    def method(self):"]
    indent_after[18:18] = ["        call_b()"]

    _write(root, "context.txt", "".join(context_lines).encode())
    _write(root, "interhunk.txt", "".join(inter_hunk_lines).encode())
    _write(root, "café.txt", b"before\n")
    _write(root, "algorithm.txt", ("\n".join(algorithm_before) + "\n").encode())
    _write(root, "indent.py", ("\n".join(indent_before) + "\n").encode())
    _write(root, "folder/data.txt", b"before attributes\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "base")
    base = str(_git(root, "rev-parse", "HEAD"))

    context_lines[2] = "changed near start\n"
    context_lines[35] = "changed near end\n"
    inter_hunk_lines[3] = "changed hunk one\n"
    inter_hunk_lines[13] = "changed hunk two\n"
    _write(root, "context.txt", "".join(context_lines).encode())
    _write(root, "interhunk.txt", "".join(inter_hunk_lines).encode())
    _write(root, "café.txt", b"after\n")
    _write(root, "algorithm.txt", ("\n".join(algorithm_after) + "\n").encode())
    _write(root, "indent.py", ("\n".join(indent_after) + "\n").encode())
    _write(root, "folder/data.txt", b"after attributes\n")
    _git(root, "add", ".")
    _git(root, "commit", "-m", "head")
    return base, str(_git(root, "rev-parse", "HEAD"))


def _renderer_poison(
    root: Path,
    tmp_path: Path,
    poison: str,
) -> dict[str, str]:
    if poison == "local color":
        _git(root, "config", "color.diff", "always")
    elif poison == "local context":
        _git(root, "config", "diff.context", "0")
    elif poison == "local inter-hunk context":
        _git(root, "config", "diff.interHunkContext", "10")
    elif poison == "local path quoting":
        _git(root, "config", "core.quotePath", "false")
    elif poison == "local algorithm":
        _git(root, "config", "diff.algorithm", "histogram")
    elif poison == "local indent heuristic":
        _git(root, "config", "diff.indentHeuristic", "false")
    elif poison == "diff options environment":
        return {"GIT_DIFF_OPTS": "--unified=0"}
    elif poison == "global config environment":
        config = tmp_path / "hostile-global.config"
        config.write_text(
            "[color]\n diff = always\n[diff]\n context = 0\n", encoding="utf-8"
        )
        return {"GIT_CONFIG_GLOBAL": str(config)}
    elif poison == "global attribute file environment":
        config_directory = tmp_path / "xdg-config"
        attributes = config_directory / "git" / "attributes"
        attributes.parent.mkdir(parents=True)
        attributes.write_bytes(b"folder/data.txt -diff\n")
        return {"XDG_CONFIG_HOME": str(config_directory)}
    elif poison == "system config environment":
        config = tmp_path / "hostile-system.config"
        config.write_text("[diff]\n interHunkContext = 10\n", encoding="utf-8")
        return {"GIT_CONFIG_NOSYSTEM": "0", "GIT_CONFIG_SYSTEM": str(config)}
    elif poison == "config injection environment":
        return {
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "diff.context",
            "GIT_CONFIG_VALUE_0": "0",
        }
    elif poison == "worktree attributes":
        _write(root, ".gitattributes", b"folder/data.txt -diff\n")
    elif poison == "info attributes":
        attributes = root / ".git" / "info" / "attributes"
        attributes.parent.mkdir(parents=True, exist_ok=True)
        attributes.write_bytes(b"folder/data.txt -diff\n")
    else:
        raise AssertionError(f"unknown renderer poison: {poison}")
    return {}


def _packet_contents(output: Path) -> tuple[bytes, bytes, bytes]:
    return (
        (output / "manifest.json").read_bytes(),
        (output / "source.patch").read_bytes(),
        (output / "source.tar").read_bytes(),
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
    assert "--ignore-submodules=none" in manifest["diff"]["flags"]
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


@pytest.mark.parametrize(
    "poison",
    (
        "local color",
        "local context",
        "local inter-hunk context",
        "local path quoting",
        "local algorithm",
        "local indent heuristic",
        "diff options environment",
        "global config environment",
        "global attribute file environment",
        "system config environment",
        "config injection environment",
        "worktree attributes",
        "info attributes",
    ),
)
def test_source_packet_is_independent_of_git_rendering_environment(
    tmp_path: Path,
    poison: str,
) -> None:
    repository = tmp_path / "repo"
    base, head = _renderer_repository(repository)
    baseline = tmp_path / "baseline"
    baseline_result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(baseline),
    )
    assert baseline_result.returncode == 0, baseline_result.stderr
    baseline_contents = _packet_contents(baseline)

    extra_env = _renderer_poison(repository, tmp_path, poison)
    poisoned = tmp_path / "poisoned"
    poisoned_result = _run(
        repository,
        "--base",
        base,
        "--head",
        head,
        "--output",
        str(poisoned),
        extra_env=extra_env,
    )

    assert poisoned_result.returncode == 0, poisoned_result.stderr
    assert _packet_contents(poisoned) == baseline_contents, poison
    manifest = json.loads((poisoned / "manifest.json").read_bytes())
    assert manifest["diff"]["renderer"]["git_version"].startswith("git version ")
    assert manifest["diff"]["renderer"]["attributes"]["source"] == "empty-tree"
    assert manifest["diff"]["renderer"]["attributes"]["info_attributes"] == "empty"
    assert manifest["diff"]["renderer"]["attributes"]["global_attributes"] == "empty"
    assert manifest["diff"]["renderer"]["environment"]["lazy_fetch"] == "disabled"
    assert "--no-color" in manifest["diff"]["flags"]
    assert "--no-color" in manifest["diff"]["renderer"]["name_status_flags"]

    if poison == "global config environment":
        base_worktree = tmp_path / "base-worktree"
        _git(repository, "worktree", "add", "--detach", str(base_worktree), base)
        apply_env = dict(os.environ)
        apply_env.update(extra_env)
        apply_result = subprocess.run(
            ["git", "apply", "--check", str(poisoned / "source.patch")],
            cwd=base_worktree,
            env=apply_env,
            capture_output=True,
            text=True,
            check=False,
        )
        assert apply_result.returncode == 0, apply_result.stderr


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
    _git(repository, "config", "diff.ignoreSubmodules", "all")
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
    assert "changed symlink and gitlink entries are unsupported" in result.stderr
    assert result.stdout == ""
    assert not output.exists()


def test_source_packet_rejects_output_inside_registered_sibling_worktree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    base, head = _repository(repository)
    sibling = tmp_path / "sibling worktree"
    _git(repository, "worktree", "add", "--detach", str(sibling), head)
    output = sibling / "packet"

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
    assert "outside every registered worktree" in result.stderr
    assert result.stdout == ""
    assert not output.exists()
    assert str(_git(repository, "status", "--porcelain=v1")) == ""
    assert str(_git(sibling, "status", "--porcelain=v1")) == ""


@pytest.mark.parametrize(
    "path",
    (b"../escape.txt", b"/absolute.txt", b"folder\\escape.txt", b"unsafe:stream"),
)
def test_source_packet_rejects_nonportable_git_paths(path: bytes) -> None:
    with pytest.raises(ValueError, match="unsafe changed Git path"):
        _safe_path(path)


@pytest.mark.parametrize(
    "path",
    (
        b"COM\xc2\xb9.txt",
        b"cOnIn$.txt",
        b"LPT\xc2\xb2",
        b"CONIN$",
        b"CONOUT$",
        b"CON .txt",
        b"folder/COM\xc2\xb9 .txt",
    ),
)
def test_source_packet_rejects_windows_reserved_device_names(path: bytes) -> None:
    with pytest.raises(ValueError, match="unsafe changed Git path"):
        _safe_path(path)


@pytest.mark.parametrize("path", ("café.txt", "COM10.txt", "LPT10"))
def test_source_packet_accepts_unicode_and_unreserved_device_prefixes(
    path: str,
) -> None:
    assert _safe_path(path.encode("utf-8")) == path


def test_source_packet_rejects_reserved_device_name_in_raw_git_tree(
    tmp_path: Path,
) -> None:
    repository = tmp_path / "repo"
    base, _ = _repository(repository)
    base_entries = _git(repository, "ls-tree", "-rz", "--full-tree", base, text=False)
    assert isinstance(base_entries, bytes)
    tracked_blob = str(_git(repository, "rev-parse", f"{base}:tracked.txt"))
    reserved_entry = (
        f"100644 blob {tracked_blob}\t".encode("ascii")
        + "COM¹.txt".encode("utf-8")
        + b"\0"
    )
    tree_bytes = _git(
        repository,
        "mktree",
        "-z",
        text=False,
        input_bytes=base_entries + reserved_entry,
    )
    assert isinstance(tree_bytes, bytes)
    tree = tree_bytes.decode("ascii").strip()
    head = str(_git(repository, "commit-tree", tree, "-p", base, "-m", "reserved path"))
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
    assert "unsafe changed Git path" in result.stderr
    assert result.stdout == ""
    assert not output.exists()
