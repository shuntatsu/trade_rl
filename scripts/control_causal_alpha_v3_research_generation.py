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
from trade_rl.domain.common import require_sha256
from trade_rl.workflows.universal_runtime_manifest import (
    load_universal_runtime_manifest,
)

_GENERATION = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{2,95}$")
_LAUNCH_SCHEMA = "causal_alpha_v3_launch_v1"
_STATUS_SCHEMA = "causal_alpha_v3_status_v1"
_RESULT_SCHEMA = "causal_alpha_v3_research_result_v1"
_RESEARCH_CONFIG = Path("examples/binance/universal-causal-alpha-v3-research.json")
_RUN_CONFIG = Path("examples/binance-multitimeframe/universal-u6-ppo.json")
_CONTAINER_OUTPUT_PREFIX = Path("/workspace/var/runs")
_REQUIRED_VOLUME = "trade-rl-training-data"
_REQUIRED_NETWORK = "trade_rl_default"


def validate_generation(value: str) -> str:
    if (
        not isinstance(value, str)
        or not _GENERATION.fullmatch(value)
        or Path(value).name != value
        or value in {".", ".."}
    ):
        raise ValueError("generation is not a safe stable identifier")
    return value


def classify_research_outcome(
    exit_code: int, *, operator_stopped: bool = False
) -> tuple[str, str]:
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code < 0:
        raise ValueError("container exit code must be a non-negative integer")
    if operator_stopped:
        return "operator_stopped", "unavailable"
    outcomes = {
        0: "admitted",
        2: "signal_rejected",
        3: "selection_rejected",
        4: "admission_rejected",
    }
    outcome = outcomes.get(exit_code)
    return ("completed", outcome) if outcome is not None else ("failed", "unavailable")


def _require_text(value: object, *, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty string")
    return value


def _require_git_commit(value: object, *, field: str) -> str:
    text = _require_text(value, field=field)
    if re.fullmatch(r"[0-9a-f]{40}", text) is None:
        raise ValueError(f"{field} must be a full Git commit SHA")
    return text


@dataclass(frozen=True, slots=True)
class CausalAlphaV3Launch:
    generation: str
    container_name: str
    image: str
    image_id: str
    git_commit: str
    source_tree_digest: str
    lockfile_digest: str
    runtime_manifest_digest: str
    research_config_digest: str
    run_config_digest: str
    output_path: str

    def __post_init__(self) -> None:
        validate_generation(self.generation)
        for name in ("container_name", "image", "output_path"):
            _require_text(getattr(self, name), field=name)
        require_sha256(self.image_id, field="image_id")
        _require_git_commit(self.git_commit, field="git_commit")
        for name in (
            "source_tree_digest",
            "lockfile_digest",
            "runtime_manifest_digest",
            "research_config_digest",
            "run_config_digest",
        ):
            require_sha256(getattr(self, name), field=name)
        expected_output = (_CONTAINER_OUTPUT_PREFIX / self.generation).as_posix()
        if self.output_path != expected_output:
            raise ValueError("output_path does not match generation identity")

    def to_payload(self) -> dict[str, object]:
        return {
            "container_name": self.container_name,
            "generation": self.generation,
            "git_commit": self.git_commit,
            "image": self.image,
            "image_id": self.image_id,
            "lockfile_digest": self.lockfile_digest,
            "output_path": self.output_path,
            "research_config_digest": self.research_config_digest,
            "run_config_digest": self.run_config_digest,
            "runtime_manifest_digest": self.runtime_manifest_digest,
            "schema_version": _LAUNCH_SCHEMA,
            "source_tree_digest": self.source_tree_digest,
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> CausalAlphaV3Launch:
        values = dict(payload)
        expected = {
            "container_name",
            "generation",
            "git_commit",
            "image",
            "image_id",
            "lockfile_digest",
            "output_path",
            "research_config_digest",
            "run_config_digest",
            "runtime_manifest_digest",
            "schema_version",
            "source_tree_digest",
        }
        if set(values) != expected:
            raise ValueError("launch manifest fields are invalid")
        if values["schema_version"] != _LAUNCH_SCHEMA:
            raise ValueError("launch manifest schema is unsupported")
        return cls(
            generation=_require_text(values["generation"], field="generation"),
            container_name=_require_text(
                values["container_name"], field="container_name"
            ),
            image=_require_text(values["image"], field="image"),
            image_id=_require_text(values["image_id"], field="image_id"),
            git_commit=_require_text(values["git_commit"], field="git_commit"),
            source_tree_digest=_require_text(
                values["source_tree_digest"], field="source_tree_digest"
            ),
            lockfile_digest=_require_text(
                values["lockfile_digest"], field="lockfile_digest"
            ),
            runtime_manifest_digest=_require_text(
                values["runtime_manifest_digest"], field="runtime_manifest_digest"
            ),
            research_config_digest=_require_text(
                values["research_config_digest"], field="research_config_digest"
            ),
            run_config_digest=_require_text(
                values["run_config_digest"], field="run_config_digest"
            ),
            output_path=_require_text(values["output_path"], field="output_path"),
        )


@dataclass(frozen=True, slots=True)
class ContainerState:
    running: bool
    oom_killed: bool
    exit_code: int

    def __post_init__(self) -> None:
        if not isinstance(self.running, bool) or not isinstance(self.oom_killed, bool):
            raise TypeError("container state flags must be booleans")
        if (
            isinstance(self.exit_code, bool)
            or not isinstance(self.exit_code, int)
            or self.exit_code < 0
        ):
            raise ValueError("container exit code must be a non-negative integer")


def _run(
    command: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> None:
    subprocess.run(tuple(command), cwd=cwd, env=env, check=True)


def _run_capture(
    command: Sequence[str], *, cwd: Path | None = None, env: Mapping[str, str] | None = None
) -> str:
    return subprocess.run(
        tuple(command),
        cwd=cwd,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git(root: Path, *args: str) -> str:
    return _run_capture(("git", "-C", str(root), *args))


def _git_status(root: Path) -> str:
    return _git(root, "status", "--porcelain", "--untracked-files=all")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _state_directory(state_root: Path, generation: str) -> Path:
    return Path(state_root).expanduser().resolve() / validate_generation(generation)


def _launch_path(state_root: Path, generation: str) -> Path:
    return _state_directory(state_root, generation) / "launch-manifest.json"


def _load_launch(*, state_root: Path, generation: str) -> CausalAlphaV3Launch:
    path = _launch_path(state_root, generation)
    if not path.is_file():
        raise FileNotFoundError(f"launch manifest is missing: {path}")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, Mapping):
        raise ValueError("launch manifest root must be an object")
    launch = CausalAlphaV3Launch.from_payload(raw)
    if launch.generation != generation:
        raise RuntimeError("launch manifest generation identity drifted")
    return launch


def _container_exists(name: str) -> bool:
    completed = subprocess.run(
        ("docker", "container", "inspect", name),
        check=False,
        capture_output=True,
    )
    return completed.returncode == 0


def _parse_inspect_object(raw: str, *, kind: str) -> Mapping[str, object]:
    payload = json.loads(raw)
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise RuntimeError(f"{kind} inspect payload is invalid")
        return dict(payload[0])
    if isinstance(payload, Mapping):
        return dict(payload)
    raise RuntimeError(f"{kind} inspect payload is invalid")


def _mapping(value: object, *, field: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{field} is unavailable")
    return {str(key): item for key, item in value.items()}


def _inspect_container(launch: CausalAlphaV3Launch) -> ContainerState:
    payload = _parse_inspect_object(
        _run_capture(("docker", "container", "inspect", launch.container_name)),
        kind="container",
    )
    config = _mapping(payload.get("Config"), field="container Config")
    labels = _mapping(config.get("Labels"), field="container labels")
    expected_labels = {
        "trade-rl.kind": "causal-alpha-v3",
        "trade-rl.generation": launch.generation,
        "trade-rl.git-commit": launch.git_commit,
        "trade-rl.runtime-manifest-digest": launch.runtime_manifest_digest,
    }
    if config.get("Image") != launch.image or any(
        labels.get(key) != value for key, value in expected_labels.items()
    ):
        raise RuntimeError("container identity does not match launch manifest")
    raw_image_id = payload.get("Image")
    if isinstance(raw_image_id, str) and raw_image_id:
        normalized = raw_image_id.removeprefix("sha256:")
        if re.fullmatch(r"[0-9a-f]{64}", normalized) and normalized != launch.image_id:
            raise RuntimeError("container image identity does not match launch manifest")
    state = _mapping(payload.get("State"), field="container State")
    running = state.get("Running")
    oom_killed = state.get("OOMKilled")
    exit_code = state.get("ExitCode")
    if not isinstance(running, bool) or not isinstance(oom_killed, bool):
        raise RuntimeError("container state flags are invalid")
    if isinstance(exit_code, bool) or not isinstance(exit_code, int) or exit_code < 0:
        raise RuntimeError("container exit code is invalid")
    return ContainerState(
        running=running,
        oom_killed=oom_killed,
        exit_code=exit_code,
    )


def _container_logs(launch: CausalAlphaV3Launch) -> str:
    return _run_capture(("docker", "container", "logs", launch.container_name)) + "\n"


def _inspect_image(image: str) -> tuple[str, Mapping[str, object]]:
    payload = _parse_inspect_object(
        _run_capture(("docker", "image", "inspect", image)), kind="image"
    )
    raw_id = payload.get("Id")
    if not isinstance(raw_id, str):
        raise RuntimeError("training image ID is unavailable")
    image_id = raw_id.removeprefix("sha256:")
    require_sha256(image_id, field="training image ID")
    config = _mapping(payload.get("Config"), field="training image Config")
    labels = _mapping(config.get("Labels"), field="training image labels")
    return image_id, labels


def _start_environment(
    *,
    commit: str,
    source_digest: str,
    lock_digest: str,
    runtime_manifest_digest: str,
    generation: str,
    image: str,
    runtime_artifact_root: Path,
) -> dict[str, str]:
    return {
        **os.environ,
        "TRADE_RL_GIT_COMMIT": commit,
        "TRADE_RL_GIT_DIRTY": "false",
        "TRADE_RL_SOURCE_TREE_DIGEST": source_digest,
        "TRADE_RL_LOCKFILE_DIGEST": lock_digest,
        "TRADE_RL_RUNTIME_MANIFEST_DIGEST": runtime_manifest_digest,
        "TRADE_RL_RUN_GENERATION": generation,
        "TRADE_RL_CAUSAL_ALPHA_V3_IMAGE": image,
        "TRADE_RL_UNIVERSAL_ARTIFACT_ROOT": str(runtime_artifact_root),
    }


def start_generation(
    *,
    project_root: str | Path,
    generation: str,
    compose_file: str | Path,
    runtime_artifact_root: str | Path,
    state_root: str | Path,
) -> CausalAlphaV3Launch:
    root = Path(project_root).resolve()
    name = validate_generation(generation)
    if _git_status(root):
        raise RuntimeError("Causal Alpha V3 launch requires a clean Git tree")
    state_directory = _state_directory(Path(state_root), name)
    if state_directory.exists():
        raise FileExistsError(f"generation state already exists: {state_directory}")

    commit = _require_git_commit(_git(root, "rev-parse", "HEAD"), field="Git HEAD")
    expected_head = os.environ.get("GITHUB_SHA", "").strip()
    if expected_head and commit != expected_head:
        raise RuntimeError("Git HEAD does not match GITHUB_SHA")

    compose_path = Path(compose_file)
    if not compose_path.is_absolute():
        compose_path = root / compose_path
    compose_path = compose_path.resolve()
    if not compose_path.is_file():
        raise FileNotFoundError(f"V3 Compose file is missing: {compose_path}")

    artifact_root = Path(runtime_artifact_root).expanduser().resolve()
    runtime_manifest_path = artifact_root / "runtime-manifest.json"
    if not runtime_manifest_path.is_file():
        raise FileNotFoundError(
            f"Universal runtime manifest is missing: {runtime_manifest_path}"
        )
    research_config = root / _RESEARCH_CONFIG
    run_config = root / _RUN_CONFIG
    for path in (research_config, run_config, root / "uv.lock"):
        if not path.is_file():
            raise FileNotFoundError(f"required launch input is missing: {path}")

    runtime_manifest_digest = load_universal_runtime_manifest(
        runtime_manifest_path
    ).manifest_digest
    source_digest = source_tree_digest(root)
    lock_digest = _sha256_file(root / "uv.lock")
    research_config_digest = _sha256_file(research_config)
    run_config_digest = _sha256_file(run_config)
    for digest, field in (
        (runtime_manifest_digest, "runtime manifest digest"),
        (source_digest, "source tree digest"),
        (lock_digest, "lockfile digest"),
        (research_config_digest, "research config digest"),
        (run_config_digest, "run config digest"),
    ):
        require_sha256(digest, field=field)

    container_name = f"trade-rl-causal-alpha-v3-{name}"
    if _container_exists(container_name):
        raise FileExistsError(f"container already exists: {container_name}")

    _run(("docker", "volume", "inspect", _REQUIRED_VOLUME), cwd=root)
    _run(("docker", "network", "inspect", _REQUIRED_NETWORK), cwd=root)

    image = f"trade-rl-causal-alpha-v3:{commit[:12]}-{runtime_manifest_digest[:12]}"
    environment = _start_environment(
        commit=commit,
        source_digest=source_digest,
        lock_digest=lock_digest,
        runtime_manifest_digest=runtime_manifest_digest,
        generation=name,
        image=image,
        runtime_artifact_root=artifact_root,
    )
    build_command = [
        "docker",
        "build",
        "--target",
        "training-runtime",
        "-f",
        "docker/Dockerfile.training",
        "-t",
        image,
    ]
    for key in (
        "TRADE_RL_GIT_COMMIT",
        "TRADE_RL_GIT_DIRTY",
        "TRADE_RL_SOURCE_TREE_DIGEST",
        "TRADE_RL_LOCKFILE_DIGEST",
        "TRADE_RL_RUNTIME_MANIFEST_DIGEST",
    ):
        build_command.extend(("--build-arg", f"{key}={environment[key]}"))
    build_command.append(".")
    _run(build_command, cwd=root, env=environment)

    image_id, labels = _inspect_image(image)
    expected_image_labels = {
        "org.opencontainers.image.revision": commit,
        "io.trade-rl.source-tree-digest": source_digest,
        "io.trade-rl.lockfile-digest": lock_digest,
        "io.trade-rl.runtime-manifest-digest": runtime_manifest_digest,
    }
    if any(labels.get(key) != value for key, value in expected_image_labels.items()):
        raise RuntimeError("training image provenance labels do not match launch inputs")

    output_path = (_CONTAINER_OUTPUT_PREFIX / name).as_posix()
    preflight_code = (
        "from pathlib import Path; "
        "from trade_rl.workflows.universal_runtime_manifest import "
        "load_universal_runtime_manifest as load; "
        f"assert load('/workspace/var/universal/runtime-manifest.json').manifest_digest == '{runtime_manifest_digest}'; "
        f"assert not Path('{output_path}').exists(), 'V3 output root already exists'"
    )
    _run(
        (
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "run",
            "--rm",
            "--no-deps",
            "research",
            "python",
            "-c",
            preflight_code,
        ),
        cwd=root,
        env=environment,
    )

    launch = CausalAlphaV3Launch(
        generation=name,
        container_name=container_name,
        image=image,
        image_id=image_id,
        git_commit=commit,
        source_tree_digest=source_digest,
        lockfile_digest=lock_digest,
        runtime_manifest_digest=runtime_manifest_digest,
        research_config_digest=research_config_digest,
        run_config_digest=run_config_digest,
        output_path=output_path,
    )
    state_directory.mkdir(parents=True)
    atomic_write_bytes(
        state_directory / "launch-manifest.json",
        canonical_json_bytes(launch.to_payload()) + b"\n",
    )
    _run(
        (
            "docker",
            "compose",
            "-f",
            str(compose_path),
            "run",
            "--detach",
            "--name",
            container_name,
            "research",
        ),
        cwd=root,
        env=environment,
    )
    return launch


def status_generation(
    *, generation: str, state_root: str | Path
) -> dict[str, object]:
    name = validate_generation(generation)
    launch = _load_launch(state_root=Path(state_root), generation=name)
    state = _inspect_container(launch)
    return {
        "container_exit_code": None if state.running else state.exit_code,
        "container_status": "running" if state.running else "exited",
        "generation": name,
        "launch": launch.to_payload(),
        "oom_killed": state.oom_killed,
        "schema_version": _STATUS_SCHEMA,
    }


def _retained_directory(retained_root: Path, generation: str) -> Path:
    root = Path(retained_root).expanduser().resolve()
    directory = root / validate_generation(generation)
    if directory.exists():
        raise FileExistsError(f"retained generation already exists: {directory}")
    directory.mkdir(parents=True)
    return directory


def _retain_terminal_evidence(
    *,
    launch: CausalAlphaV3Launch,
    state: ContainerState,
    retained_root: Path,
    operator_stopped: bool,
) -> dict[str, object]:
    retained = _retained_directory(retained_root, launch.generation)
    logs = _container_logs(launch)
    (retained / "container.log").write_text(logs, encoding="utf-8")
    run_copy = retained / "run"
    run_copy.mkdir()
    _run(
        (
            "docker",
            "cp",
            f"{launch.container_name}:{launch.output_path}/.",
            str(run_copy),
        )
    )
    atomic_write_bytes(
        retained / "launch-manifest.json",
        canonical_json_bytes(launch.to_payload()) + b"\n",
    )
    execution_status, research_outcome = classify_research_outcome(
        state.exit_code, operator_stopped=operator_stopped
    )
    if state.oom_killed and not operator_stopped:
        execution_status, research_outcome = "failed", "unavailable"
    result: dict[str, object] = {
        "container_exit_code": state.exit_code,
        "execution_status": execution_status,
        "generation": launch.generation,
        "launch": launch.to_payload(),
        "oom_killed": state.oom_killed,
        "research_outcome": research_outcome,
        "schema_version": _RESULT_SCHEMA,
    }
    atomic_write_bytes(
        retained / "research-result.json",
        canonical_json_bytes(result) + b"\n",
    )
    return result


def collect_generation(
    *,
    generation: str,
    state_root: str | Path,
    retained_root: str | Path,
) -> dict[str, object]:
    name = validate_generation(generation)
    launch = _load_launch(state_root=Path(state_root), generation=name)
    state = _inspect_container(launch)
    if state.running:
        raise RuntimeError("cannot collect a running V3 research container")
    return _retain_terminal_evidence(
        launch=launch,
        state=state,
        retained_root=Path(retained_root),
        operator_stopped=False,
    )


def stop_generation(
    *,
    generation: str,
    state_root: str | Path,
    retained_root: str | Path,
) -> dict[str, object]:
    name = validate_generation(generation)
    launch = _load_launch(state_root=Path(state_root), generation=name)
    state = _inspect_container(launch)
    if state.running:
        _run(("docker", "container", "stop", launch.container_name))
        state = _inspect_container(launch)
    if state.running:
        raise RuntimeError("V3 research container remained running after stop")
    return _retain_terminal_evidence(
        launch=launch,
        state=state,
        retained_root=Path(retained_root),
        operator_stopped=True,
    )


def _default_state_root() -> Path:
    configured = os.environ.get("TRADE_RL_CAUSAL_ALPHA_V3_STATE_ROOT", "").strip()
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local/state/trade-rl/causal-alpha-v3"


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Control an immutable Causal Alpha V3 research generation"
    )
    parser.add_argument(
        "--operation",
        required=True,
        choices=("start", "status", "collect", "stop"),
    )
    parser.add_argument("--generation", required=True)
    parser.add_argument("--project-root", type=Path, default=Path.cwd())
    parser.add_argument(
        "--compose-file",
        type=Path,
        default=Path("docker/compose.causal-alpha-v3-research.yaml"),
    )
    parser.add_argument(
        "--runtime-artifact-root",
        type=Path,
        default=(
            Path(os.environ["TRADE_RL_UNIVERSAL_ARTIFACT_ROOT"])
            if os.environ.get("TRADE_RL_UNIVERSAL_ARTIFACT_ROOT", "").strip()
            else None
        ),
    )
    parser.add_argument("--state-root", type=Path, default=_default_state_root())
    parser.add_argument(
        "--retained-root",
        type=Path,
        default=Path("retained-causal-alpha-v3"),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    operation = args.operation
    if operation == "start":
        if args.runtime_artifact_root is None:
            raise ValueError(
                "start requires TRADE_RL_UNIVERSAL_ARTIFACT_ROOT or --runtime-artifact-root"
            )
        payload: Mapping[str, object] = start_generation(
            project_root=args.project_root,
            generation=args.generation,
            compose_file=args.compose_file,
            runtime_artifact_root=args.runtime_artifact_root,
            state_root=args.state_root,
        ).to_payload()
    elif operation == "status":
        payload = status_generation(
            generation=args.generation,
            state_root=args.state_root,
        )
    elif operation == "collect":
        payload = collect_generation(
            generation=args.generation,
            state_root=args.state_root,
            retained_root=args.retained_root,
        )
    else:
        payload = stop_generation(
            generation=args.generation,
            state_root=args.state_root,
            retained_root=args.retained_root,
        )
    print(json.dumps(dict(payload), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
