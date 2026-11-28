"""Generate bounded Python lockfile proposals from an immutable Git checkout.

The resolver runs only in an isolated archive. The caller's lockfile and
manifest are never written; publishing a candidate does not approve a PR.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import tomllib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlsplit

from tools.uv_advisory_audit import locked_packages, required_uv_version

_NAME = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_RELEASE = re.compile(
    r"(?:0|[1-9][0-9]{0,8})(?:\.(?:0|[1-9][0-9]{0,8})){0,3}"
    r"(?:\.post(?:0|[1-9][0-9]{0,8}))?"
)
_SHA = re.compile(r"[0-9a-f]{40}")
_FILE_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _release_key(version: str) -> tuple[int, ...]:
    # Admit only the stable numeric release grammar used by the current lock.
    # An epoch, local, development or prerelease version needs separate review.
    if not _RELEASE.fullmatch(version):
        raise ValueError("unsupported release version")
    release, separator, post = version.partition(".post")
    parts = tuple(int(value) for value in release.split("."))
    return (*parts, *((0,) * (4 - len(parts))), int(post) if separator else -1)


def _artifact(artifact: object) -> None:
    if not isinstance(artifact, dict):
        raise ValueError("malformed registry artifact")
    url, digest, size = (
        artifact.get("url"),
        artifact.get("hash"),
        artifact.get("size"),
    )
    if not isinstance(url, str) or not isinstance(digest, str):
        raise ValueError("missing registry artifact identity")
    parsed = urlsplit(url)
    if (
        parsed.scheme != "https"
        or parsed.netloc != "files.pythonhosted.org"
        or not parsed.path.startswith("/packages/")
        or parsed.query
        or parsed.fragment
        or not _FILE_HASH.fullmatch(digest)
        or type(size) is not int
        or size <= 0
    ):
        raise ValueError("unsupported registry artifact")


@dataclass(frozen=True)
class LockSnapshot:
    header: dict[str, object]
    root: dict[str, object]
    packages: dict[str, dict[str, object]]
    versions: dict[str, str]


def read_lock(raw: str) -> LockSnapshot:
    identities = locked_packages(raw)
    versions = dict(identities)
    if len(versions) != len(identities):
        raise ValueError("multiple locked versions need separate review")
    for name, version in versions.items():
        if not _NAME.fullmatch(name):
            raise ValueError("noncanonical package name")
        _release_key(version)
    payload = tomllib.loads(raw)
    rows = payload.pop("package")
    root: dict[str, object] = {}
    packages: dict[str, dict[str, object]] = {}
    for row in rows:
        name = row["name"]
        if name == "trade-rl" and row.get("source") == {"editable": "."}:
            root = row
            continue
        if name in packages or name not in versions:
            raise ValueError("ambiguous package roster")
        wheels = row.get("wheels", [])
        if not isinstance(wheels, list):
            raise ValueError("malformed wheel roster")
        artifacts = [*wheels]
        if "sdist" in row:
            artifacts.append(row["sdist"])
        if not artifacts:
            raise ValueError("missing registry artifacts")
        for artifact in artifacts:
            _artifact(artifact)
        packages[name] = row
    if not root or set(packages) != set(versions):
        raise ValueError("incomplete lock roster")
    return LockSnapshot(payload, root, packages, versions)


def upgrade_targets(
    snapshot: LockSnapshot, requested: tuple[str, ...]
) -> tuple[str, ...]:
    names = requested or tuple(snapshot.packages)
    if len(names) != len(set(names)) or any(
        name not in snapshot.packages for name in names
    ):
        raise ValueError("unknown or duplicate target package")
    return tuple(
        f"{name}>={snapshot.versions[name]},<{_release_key(snapshot.versions[name])[0] + 1}"
        for name in sorted(names)
    )


def compare_locks(
    before: str, after: str, requested: tuple[str, ...]
) -> tuple[tuple[str, str, str], ...]:
    old, new = read_lock(before), read_lock(after)
    upgrade_targets(old, requested)
    if old.header != new.header or old.root != new.root:
        raise ValueError("lock metadata or project contract changed")
    if old.packages.keys() != new.packages.keys():
        raise ValueError("package roster changed")
    changes: list[tuple[str, str, str]] = []
    for name, previous in sorted(old.versions.items()):
        current = new.versions[name]
        if current == previous:
            if old.packages[name] != new.packages[name]:
                raise ValueError("same-release metadata changed")
            continue
        old_key, new_key = _release_key(previous), _release_key(current)
        if (
            (requested and name not in requested)
            or new_key[0] != old_key[0]
            or new_key <= old_key
        ):
            raise ValueError("unapproved release change")
        changes.append((name, previous, current))
    if not changes and before != after:
        raise ValueError("unexplained lockfile rewrite")
    return tuple(changes)


def resolve_candidate(project: Path, pin: str, targets: tuple[str, ...]) -> None:
    executable = shutil.which("uv")
    if executable is None:
        raise ValueError("pinned uv executable is unavailable")
    # Do not inherit registry configuration, credentials, or uv overrides.
    # The runner must provide Python 3.12 and network access to public PyPI.
    env = {
        key: os.environ[key]
        for key in ("PATH", "SYSTEMROOT", "WINDIR", "TMP", "TEMP", "LANG")
        if key in os.environ
    }
    with tempfile.TemporaryDirectory(prefix="uv-proposal-config-") as directory:
        private = Path(directory)
        config = private / "uv.toml"
        config.write_text(f'required-version = "=={pin}"\n', encoding="utf-8")
        env.update(
            HOME=directory,
            XDG_CONFIG_HOME=directory,
            XDG_CACHE_HOME=str(private / "cache"),
            UV_CACHE_DIR=str(private / "uv-cache"),
            UV_PYTHON_PREFERENCE="only-system",
            UV_PYTHON_DOWNLOADS="never",
        )
        version = subprocess.run(
            [executable, "--version"],
            cwd=project,
            env=env,
            capture_output=True,
            text=True,
            timeout=15,
            check=False,
        )
        match = re.match(r"^uv ([0-9]+\.[0-9]+\.[0-9]+)(?:\s|$)", version.stdout)
        if version.returncode or not match or match.group(1) != pin:
            raise ValueError("actual uv version differs from project pin")
        command = [
            executable,
            "--config-file",
            str(config),
            "lock",
            "--no-build",
            "--python",
            sys.executable,
            "--default-index",
            "https://pypi.org/simple",
        ]
        upgrade = [
            argument for target in targets for argument in ("--upgrade-package", target)
        ]
        for arguments in (["--check"], upgrade, ["--check"]):
            subprocess.run(
                [*command, *arguments],
                cwd=project,
                env=env,
                capture_output=True,
                check=True,
                timeout=300,
            )


def _git(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], text=True, stderr=subprocess.PIPE, timeout=30
    ).strip()


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def generate(
    output: Path, requested: tuple[str, ...], expected_head: str
) -> dict[str, object]:
    root = Path.cwd().resolve()
    if Path(_git("rev-parse", "--show-toplevel")).resolve() != root:
        raise ValueError("run from the repository root")
    head = _git("rev-parse", "HEAD")
    if not _SHA.fullmatch(head) or head != expected_head:
        raise ValueError("source HEAD differs from requested revision")
    if output.is_symlink():
        raise ValueError("output cannot be a symbolic link")
    output = output.resolve()
    if output.is_relative_to(root) or output.exists():
        raise ValueError("require fresh output outside the repository")
    manifest, lock = root / "pyproject.toml", root / "uv.lock"
    if manifest.is_symlink() or lock.is_symlink():
        raise ValueError("require regular manifest and lock files")
    before = manifest.read_bytes(), lock.read_bytes()
    pin = required_uv_version(before[0])
    targets = upgrade_targets(read_lock(before[1].decode("utf-8")), requested)
    with tempfile.TemporaryDirectory(prefix="trade-rl-lock-proposal-") as directory:
        temporary = Path(directory)
        archive = temporary / "source.tar"
        project = temporary / "source"
        project.mkdir()
        subprocess.run(
            ["git", "archive", "--format=tar", f"--output={archive}", head],
            check=True,
            capture_output=True,
            timeout=60,
        )
        with tarfile.open(archive) as source:
            source.extractall(project, filter="data")
        archived_manifest, archived_lock = (
            project / "pyproject.toml",
            project / "uv.lock",
        )
        if (
            (archived_manifest.read_bytes(), archived_lock.read_bytes()) != before
            or (project / "uv.toml").exists()
            or (project / "uv.toml").is_symlink()
        ):
            raise ValueError("manifest or lock differs from committed source")
        resolve_candidate(project, pin, targets)
        candidate = archived_lock.read_bytes()
        if (
            archived_manifest.read_bytes() != before[0]
            or (manifest.read_bytes(), lock.read_bytes()) != before
            or _git("rev-parse", "HEAD") != head
        ):
            raise ValueError("source or manifest changed during resolution")
        changes = compare_locks(
            before[1].decode("utf-8"), candidate.decode("utf-8"), requested
        )

    report: dict[str, object] = {
        "schema": "uv_lock_proposal_v1",
        "source_head": head,
        "uv_version": pin,
        "manifest_sha256": _digest(before[0]),
        "source_lock_sha256": _digest(before[1]),
        "candidate_lock_sha256": _digest(candidate),
        "status": "CANDIDATE" if changes else "NO_CHANGES",
        "changes": changes,
        "requested_packages": requested,
    }
    output.mkdir()
    if changes:
        (output / "uv.lock").write_bytes(candidate)
        patch = "".join(
            difflib.unified_diff(
                before[1].decode("utf-8").splitlines(keepends=True),
                candidate.decode("utf-8").splitlines(keepends=True),
                fromfile="a/uv.lock",
                tofile="b/uv.lock",
            )
        )
        (output / "uv-lock.patch").write_bytes(patch.encode("utf-8"))
    (output / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--packages",
        default="",
        help="space-separated locked package names; empty means all",
    )
    args = parser.parse_args(argv)
    try:
        report = generate(args.output, tuple(args.packages.split()), args.head)
    except (
        ValueError,
        KeyError,
        TypeError,
        OSError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ):
        print(
            "PROPOSAL_FAILED: unsupported, incomplete or unavailable resolution; no update approved"
        )
        return 2
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
