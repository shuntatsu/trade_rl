from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from trade_rl.artifacts.atomic_write import atomic_write_bytes
from trade_rl.artifacts.codec import canonical_json_bytes
from trade_rl.artifacts.provenance import source_tree_digest
from trade_rl.workflows.universal_runtime_manifest import (
    load_universal_runtime_manifest,
)

_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{2,95}$")


@dataclass(frozen=True, slots=True)
class UniversalTrainingLaunch:
    generation: str
    container_name: str
    image: str
    generation_root: Path
    git_commit: str
    source_tree_digest: str
    lockfile_digest: str
    runtime_manifest_digest: str

    def to_json_dict(self) -> dict[str, object]:
        return {
            "container_name": self.container_name,
            "generation": self.generation,
            "generation_root": str(self.generation_root),
            "git_commit": self.git_commit,
            "image": self.image,
            "lockfile_digest": self.lockfile_digest,
            "runtime_manifest_digest": self.runtime_manifest_digest,
            "schema_version": "universal_training_launch_v1",
            "source_tree_digest": self.source_tree_digest,
        }


def _run(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str] | None = None
) -> None:
    subprocess.run(tuple(command), cwd=cwd, env=env, check=True)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(
        ("git", "-C", str(root), *args),
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_status(root: Path) -> str:
    return _git(root, "status", "--porcelain", "--untracked-files=all")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _container_exists(name: str) -> bool:
    completed = subprocess.run(
        ("docker", "container", "inspect", name),
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def launch_generation(
    *,
    project_root: str | Path,
    generation: str,
    compose_file: str | Path,
    runtime_manifest: str | Path,
) -> UniversalTrainingLaunch:
    """Build, preflight, and start one immutable full-training generation."""

    root = Path(project_root).resolve()
    if not _GENERATION.fullmatch(generation):
        raise ValueError("generation must contain only letters, digits, and hyphens")
    if _git_status(root):
        raise RuntimeError("Universal launch requires a clean Git tree")
    commit = _git(root, "rev-parse", "HEAD")
    source_digest = source_tree_digest(root)
    lock_digest = _sha256_file(root / "uv.lock")
    manifest_path = Path(runtime_manifest).resolve()
    manifest_digest = load_universal_runtime_manifest(manifest_path).manifest_digest
    container_name = f"trade-rl-{generation}"
    if _container_exists(container_name):
        raise FileExistsError(f"container already exists: {container_name}")
    generation_root = root / "artifacts" / "universal" / "launch" / generation
    if generation_root.exists():
        raise FileExistsError(f"generation evidence already exists: {generation_root}")
    image = f"trade-rl-universal:{commit[:12]}-{manifest_digest[:12]}"
    environment = {
        **os.environ,
        "TRADE_RL_GIT_COMMIT": commit,
        "TRADE_RL_GIT_DIRTY": "false",
        "TRADE_RL_SOURCE_TREE_DIGEST": source_digest,
        "TRADE_RL_LOCKFILE_DIGEST": lock_digest,
        "TRADE_RL_RUNTIME_MANIFEST_DIGEST": manifest_digest,
        "TRADE_RL_RUN_GENERATION": generation,
        "TRADE_RL_UNIVERSAL_IMAGE": image,
        "TRADE_RL_UNIVERSAL_ARTIFACT_ROOT": str(manifest_path.parent),
    }
    build_args = (
        "TRADE_RL_GIT_COMMIT",
        "TRADE_RL_GIT_DIRTY",
        "TRADE_RL_SOURCE_TREE_DIGEST",
        "TRADE_RL_LOCKFILE_DIGEST",
        "TRADE_RL_RUNTIME_MANIFEST_DIGEST",
    )
    command = ["docker", "build", "--target", "training-runtime", "-f", "Dockerfile.training", "-t", image]
    for name in build_args:
        command.extend(("--build-arg", f"{name}={environment[name]}"))
    command.append(".")
    _run(command, cwd=root, env=environment)
    compose = str(Path(compose_file).resolve())
    _run(
        (
            "docker", "compose", "-f", compose, "run", "--rm", "--no-deps",
            "trainer", "python", "-c",
            "from trade_rl.workflows.universal_runtime_manifest import load_universal_runtime_manifest as load; print(load('/workspace/var/universal/runtime-manifest.json').manifest_digest)",
        ),
        cwd=root,
        env=environment,
    )
    launch = UniversalTrainingLaunch(
        generation=generation,
        container_name=container_name,
        image=image,
        generation_root=generation_root,
        git_commit=commit,
        source_tree_digest=source_digest,
        lockfile_digest=lock_digest,
        runtime_manifest_digest=manifest_digest,
    )
    generation_root.mkdir(parents=True)
    atomic_write_bytes(
        generation_root / "launch-manifest.json",
        canonical_json_bytes(launch.to_json_dict()) + b"\n",
    )
    _run(
        (
            "docker", "compose", "-f", compose, "run", "--detach", "--name",
            container_name, "trainer",
        ),
        cwd=root,
        env=environment,
    )
    return launch


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch immutable Universal U6 training")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--compose-file", type=Path, default=Path("compose.universal-training.yaml"))
    parser.add_argument("--runtime-manifest", required=True, type=Path)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    result = launch_generation(
        project_root=args.project_root,
        generation=args.generation,
        compose_file=args.compose_file,
        runtime_manifest=args.runtime_manifest,
    )
    print(json.dumps(result.to_json_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
