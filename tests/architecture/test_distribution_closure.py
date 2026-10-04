from __future__ import annotations

import importlib
import io
import stat
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

SOURCE = "trade_rl/__init__.py"
CONTENT = b'__version__ = "1.0"\n'
ACTIVATION_RESOURCE = "trade_rl/evaluation/ppo_normalization_activation.json"
ACTIVATION_CONTENT = (
    b'{"activation_sha256":null,'
    b'"schema":"ppo_normalization_execution_activation_authority_v1"}'
)


def _git(root: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)


@pytest.fixture
def checkout(tmp_path: Path) -> Path:
    root = tmp_path / "repo"
    (root / "trade_rl").mkdir(parents=True)
    (root / SOURCE).write_bytes(CONTENT)
    _git(root, "init")
    _git(root, "add", SOURCE)
    _git(
        root,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "source fixture",
    )
    return root


def _add_activation_resource(checkout: Path) -> None:
    path = checkout / ACTIVATION_RESOURCE
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(ACTIVATION_CONTENT)
    _git(checkout, "add", ACTIVATION_RESOURCE)
    _git(
        checkout,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "activation resource fixture",
    )


def _archive(
    path: Path, entries: list[tuple[str, bytes]], *, symlink: bool = False
) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path, "w") as archive:
            for name, content in entries:
                info = zipfile.ZipInfo(name)
                info.create_system = 3
                info.external_attr = (stat.S_IFLNK if symlink else stat.S_IFREG) << 16
                archive.writestr(info, content)
    else:
        with tarfile.open(path, "w:gz") as archive:
            for name, content in entries:
                info = tarfile.TarInfo("trade_rl-1.0/" + name)
                if symlink:
                    info.type = tarfile.SYMTYPE
                    info.linkname = "elsewhere"
                    archive.addfile(info)
                else:
                    info.size = len(content)
                    archive.addfile(info, io.BytesIO(content))


def _verify(checkout: Path, archive: Path) -> None:
    module = importlib.import_module("tests.architecture.distribution")
    module.verify_distribution(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_exact_tracked_sources_match_distributions(
    checkout: Path, extension: str
) -> None:
    archive = checkout.parent / ("package" + extension)
    _archive(archive, [(SOURCE, CONTENT)])
    _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
@pytest.mark.parametrize("kind", ["missing", "extra", "changed"])
def test_distribution_source_drift_fails_closed(
    checkout: Path, extension: str, kind: str
) -> None:
    entries = [(SOURCE, CONTENT)]
    if kind == "missing":
        entries = []
    elif kind == "extra":
        entries.append(("trade_rl/shadow.py", b"pass\n"))
    else:
        entries = [(SOURCE, b'__version__ = "2.0"\n')]
    archive = checkout.parent / ("package" + extension)
    _archive(archive, entries)
    with pytest.raises(ValueError, match="source|empty"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_duplicate_archive_sources_are_rejected(checkout: Path, extension: str) -> None:
    archive = checkout.parent / ("package" + extension)
    if extension == ".whl":
        with pytest.warns(UserWarning, match="Duplicate"):
            _archive(archive, [(SOURCE, CONTENT), (SOURCE, CONTENT)])
    else:
        _archive(archive, [(SOURCE, CONTENT), (SOURCE, CONTENT)])
    with pytest.raises(ValueError, match="duplicate"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_symlink_archive_sources_are_rejected(checkout: Path, extension: str) -> None:
    archive = checkout.parent / ("package" + extension)
    _archive(archive, [(SOURCE, CONTENT)], symlink=True)
    with pytest.raises(ValueError, match="regular|symlink"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_unsafe_archive_paths_are_rejected(checkout: Path, extension: str) -> None:
    archive = checkout.parent / ("package" + extension)
    _archive(archive, [(SOURCE, CONTENT), ("../trade_rl/hidden.py", b"pass\n")])
    with pytest.raises(ValueError, match="path"):
        _verify(checkout, archive)


def test_ignored_untracked_source_is_not_packaged_as_authority(checkout: Path) -> None:
    (checkout / ".gitignore").write_text("build/\n")
    nested = checkout / "trade_rl/build"
    nested.mkdir()
    (nested / "hidden.py").write_text("pass\n")
    archive = checkout.parent / "package.whl"
    _archive(archive, [(SOURCE, CONTENT)])
    with pytest.raises(ValueError, match="untracked|source"):
        _verify(checkout, archive)


def test_worktree_source_drift_is_not_confused_with_git_source(checkout: Path) -> None:
    (checkout / SOURCE).write_text("changed = True\n")
    archive = checkout.parent / "package.whl"
    _archive(archive, [(SOURCE, b"changed = True\n")])
    with pytest.raises(ValueError, match="worktree|source"):
        _verify(checkout, archive)


def test_source_hidden_in_wheel_data_scheme_is_rejected(checkout: Path) -> None:
    archive = checkout.parent / "package.whl"
    _archive(
        archive, [(SOURCE, CONTENT), ("pkg.data/purelib/trade_rl/hidden.py", b"pass\n")]
    )
    with pytest.raises(ValueError, match="source|path"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_archive_symlink_package_parent_is_rejected(
    checkout: Path, extension: str
) -> None:
    archive = checkout.parent / ("package" + extension)
    if extension == ".whl":
        with zipfile.ZipFile(archive, "w") as stream:
            info = zipfile.ZipInfo("trade_rl")
            info.create_system = 3
            info.external_attr = stat.S_IFLNK << 16
            stream.writestr(info, b"../outside")
            stream.writestr(SOURCE, CONTENT)
    else:
        with tarfile.open(archive, "w:gz") as stream:
            parent = tarfile.TarInfo("trade_rl-1.0/trade_rl")
            parent.type = tarfile.SYMTYPE
            parent.linkname = "../outside"
            stream.addfile(parent)
            info = tarfile.TarInfo("trade_rl-1.0/" + SOURCE)
            info.size = len(CONTENT)
            stream.addfile(info, io.BytesIO(CONTENT))
    with pytest.raises(ValueError, match="regular|symlink"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_activation_resource_matches_checkout_and_distribution(
    checkout: Path,
    extension: str,
) -> None:
    _add_activation_resource(checkout)
    archive = checkout.parent / ("activation" + extension)
    _archive(
        archive,
        [(SOURCE, CONTENT), (ACTIVATION_RESOURCE, ACTIVATION_CONTENT)],
    )
    _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
@pytest.mark.parametrize("kind", ["missing", "changed"])
def test_activation_resource_distribution_drift_fails_closed(
    checkout: Path,
    extension: str,
    kind: str,
) -> None:
    _add_activation_resource(checkout)
    entries = [(SOURCE, CONTENT)]
    if kind == "changed":
        changed = (
            b'{"activation_sha256":"'
            + b"a" * 64
            + b'","schema":"ppo_normalization_execution_activation_authority_v1"}'
        )
        entries.append((ACTIVATION_RESOURCE, changed))
    archive = checkout.parent / ("activation-drift" + extension)
    _archive(archive, entries)
    with pytest.raises(ValueError, match="source|mismatch|activation"):
        _verify(checkout, archive)


@pytest.mark.parametrize("extension", [".whl", ".tar.gz"])
def test_activation_resource_archive_symlink_is_rejected(
    checkout: Path,
    extension: str,
) -> None:
    _add_activation_resource(checkout)
    archive = checkout.parent / ("activation-symlink" + extension)
    if extension == ".whl":
        with zipfile.ZipFile(archive, "w") as stream:
            source = zipfile.ZipInfo(SOURCE)
            source.create_system = 3
            source.external_attr = stat.S_IFREG << 16
            stream.writestr(source, CONTENT)
            resource = zipfile.ZipInfo(ACTIVATION_RESOURCE)
            resource.create_system = 3
            resource.external_attr = stat.S_IFLNK << 16
            stream.writestr(resource, b"elsewhere")
    else:
        with tarfile.open(archive, "w:gz") as stream:
            source = tarfile.TarInfo("trade_rl-1.0/" + SOURCE)
            source.size = len(CONTENT)
            stream.addfile(source, io.BytesIO(CONTENT))
            resource = tarfile.TarInfo("trade_rl-1.0/" + ACTIVATION_RESOURCE)
            resource.type = tarfile.SYMTYPE
            resource.linkname = "elsewhere"
            stream.addfile(resource)
    with pytest.raises(ValueError, match="symlink|regular"):
        _verify(checkout, archive)


@pytest.mark.parametrize("kind", ["missing", "changed", "symlink"])
def test_activation_resource_checkout_drift_fails_closed(
    checkout: Path,
    kind: str,
) -> None:
    _add_activation_resource(checkout)
    resource = checkout / ACTIVATION_RESOURCE
    if kind == "missing":
        resource.unlink()
    elif kind == "changed":
        resource.write_bytes(
            b'{"activation_sha256":"'
            + b"a" * 64
            + b'","schema":"ppo_normalization_execution_activation_authority_v1"}'
        )
    else:
        target = checkout / "activation-target.json"
        target.write_bytes(ACTIVATION_CONTENT)
        resource.unlink()
        try:
            resource.symlink_to(target)
        except OSError:
            pytest.skip("symlinks are unavailable on this platform")
    archive = checkout.parent / "activation-checkout.whl"
    _archive(
        archive,
        [(SOURCE, CONTENT), (ACTIVATION_RESOURCE, ACTIVATION_CONTENT)],
    )
    with pytest.raises(ValueError, match="worktree|source|regular|symlink"):
        _verify(checkout, archive)
