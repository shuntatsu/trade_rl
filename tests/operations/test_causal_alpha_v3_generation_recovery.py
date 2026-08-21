from __future__ import annotations

import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest


def _module():
    return importlib.import_module("trade_rl.operations.causal_alpha_v3_generation")


def _launch(module):
    commit = "a" * 40
    runtime_digest = "d" * 64
    generation = "causal-alpha-v3-v2-r1"
    return module.CausalAlphaV3Launch(
        generation=generation,
        container_name=f"trade-rl-causal-alpha-v3-{generation}",
        image=f"trade-rl-causal-alpha-v3:{commit[:12]}-{runtime_digest[:12]}",
        image_id="1" * 64,
        git_commit=commit,
        source_tree_digest="b" * 64,
        lockfile_digest="c" * 64,
        runtime_manifest_digest=runtime_digest,
        research_config_digest="e" * 64,
        run_config_digest="f" * 64,
        output_path=f"/workspace/var/runs/{generation}",
    )


def _canonical(payload: object) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode() + b"\n"


def _write_launch(path: Path, launch) -> bytes:
    payload = _canonical(launch.to_payload())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return payload


def test_start_retries_same_persisted_identity_when_detach_previously_failed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    root = tmp_path / "project"
    state_root = tmp_path / "state"
    runtime_root = tmp_path / "runtime"
    compose = root / "docker/compose.causal-alpha-v3-research.yaml"
    research = root / "examples/binance/universal-causal-alpha-v3-research.json"
    run_config = root / "examples/binance-multitimeframe/universal-u6-ppo.json"
    for path in (compose, research, run_config, root / "uv.lock"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    (runtime_root / "runtime-manifest.json").parent.mkdir(parents=True)
    (runtime_root / "runtime-manifest.json").write_text("{}\n", encoding="utf-8")
    launch_path = state_root / launch.generation / "launch-manifest.json"
    original_bytes = _write_launch(launch_path, launch)
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(module, "_git_status", lambda _root: "")
    monkeypatch.setattr(module, "_git", lambda *_args: launch.git_commit)
    monkeypatch.setattr(
        module,
        "load_universal_runtime_manifest",
        lambda _path: SimpleNamespace(manifest_digest=launch.runtime_manifest_digest),
    )
    monkeypatch.setattr(
        module, "source_tree_digest", lambda _root: launch.source_tree_digest
    )

    def digest(path: Path) -> str:
        values = {
            "uv.lock": launch.lockfile_digest,
            "universal-causal-alpha-v3-research.json": (launch.research_config_digest),
            "universal-u6-ppo.json": launch.run_config_digest,
        }
        return values[path.name]

    monkeypatch.setattr(module, "_sha256_file", digest)
    monkeypatch.setattr(module, "_container_exists", lambda _name: False)
    monkeypatch.setattr(
        module,
        "_inspect_image",
        lambda _image: (
            launch.image_id,
            {
                "org.opencontainers.image.revision": launch.git_commit,
                "io.trade-rl.source-tree-digest": launch.source_tree_digest,
                "io.trade-rl.lockfile-digest": launch.lockfile_digest,
                "io.trade-rl.runtime-manifest-digest": (launch.runtime_manifest_digest),
            },
        ),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: commands.append(tuple(command)),
    )

    result = module.start_generation(
        project_root=root,
        generation=launch.generation,
        compose_file=compose,
        runtime_artifact_root=runtime_root,
        state_root=state_root,
    )

    assert result == launch
    assert launch_path.read_bytes() == original_bytes
    assert sum("--detach" in command for command in commands) == 1


def test_collect_resumes_incomplete_copy_and_then_is_idempotent(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    state_root = tmp_path / "state"
    retained_root = tmp_path / "retained"
    _write_launch(
        state_root / launch.generation / "launch-manifest.json",
        launch,
    )
    retained = retained_root / launch.generation
    retained.mkdir(parents=True)
    _write_launch(retained / "launch-manifest.json", launch)
    (retained / "container.log").write_bytes(b"first-attempt\n")
    incomplete = {
        "container_exit_code": 1,
        "execution_status": "failed",
        "generation": launch.generation,
        "launch": launch.to_payload(),
        "oom_killed": False,
        "research_outcome": "unavailable",
        "run_output_retained": False,
        "schema_version": "causal_alpha_v3_research_result_v1",
    }
    (retained / "research-result.json").write_bytes(_canonical(incomplete))
    partial = retained / "run"
    partial.mkdir()
    (partial / "partial.txt").write_text("partial", encoding="utf-8")
    copy_calls: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=1,
        ),
    )
    monkeypatch.setattr(
        module,
        "_container_logs",
        lambda _launch: (_ for _ in ()).throw(
            AssertionError("resume must preserve retained logs")
        ),
    )

    def copy(command, **_kwargs) -> None:
        call = tuple(command)
        copy_calls.append(call)
        assert call[:2] == ("docker", "cp")
        destination = Path(call[-1])
        destination.mkdir(parents=True, exist_ok=True)
        (destination / "artifact.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", copy)

    first = module.collect_generation(
        generation=launch.generation,
        state_root=state_root,
        retained_root=retained_root,
    )
    assert first["run_output_retained"] is True
    assert (retained / "run/artifact.json").is_file()
    assert not (retained / "run/partial.txt").exists()
    assert (retained / "container.log").read_bytes() == b"first-attempt\n"

    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("completed collection must not copy twice")
        ),
    )
    second = module.collect_generation(
        generation=launch.generation,
        state_root=state_root,
        retained_root=retained_root,
    )

    assert second == first
    assert len(copy_calls) == 1
    assert not any(path.name.startswith(".run-") for path in retained.iterdir())


def _prepared_start_inputs(tmp_path: Path, launch) -> tuple[Path, Path, Path, Path]:
    root = tmp_path / "project"
    state_root = tmp_path / "state"
    runtime_root = tmp_path / "runtime"
    compose = root / "docker/compose.causal-alpha-v3-research.yaml"
    for path in (
        compose,
        root / "examples/binance/universal-causal-alpha-v3-research.json",
        root / "examples/binance-multitimeframe/universal-u6-ppo.json",
        root / "uv.lock",
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture\n", encoding="utf-8")
    runtime_manifest = runtime_root / "runtime-manifest.json"
    runtime_manifest.parent.mkdir(parents=True, exist_ok=True)
    runtime_manifest.write_text("{}\n", encoding="utf-8")
    _write_launch(
        state_root / launch.generation / "launch-manifest.json",
        launch,
    )
    return root, state_root, runtime_root, compose


def _mock_start_identity(
    monkeypatch: pytest.MonkeyPatch,
    module,
    launch,
    *,
    source_digest: str | None = None,
) -> None:
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.setattr(module, "_git_status", lambda _root: "")
    monkeypatch.setattr(module, "_git", lambda *_args: launch.git_commit)
    monkeypatch.setattr(
        module,
        "load_universal_runtime_manifest",
        lambda _path: SimpleNamespace(manifest_digest=launch.runtime_manifest_digest),
    )
    monkeypatch.setattr(
        module,
        "source_tree_digest",
        lambda _root: source_digest or launch.source_tree_digest,
    )
    monkeypatch.setattr(
        module,
        "_sha256_file",
        lambda path: {
            "uv.lock": launch.lockfile_digest,
            "universal-causal-alpha-v3-research.json": (launch.research_config_digest),
            "universal-u6-ppo.json": launch.run_config_digest,
        }[path.name],
    )


def test_running_generation_rejects_current_identity_drift_before_docker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    root, state_root, runtime_root, compose = _prepared_start_inputs(tmp_path, launch)
    _mock_start_identity(
        monkeypatch,
        module,
        launch,
        source_digest="9" * 64,
    )
    monkeypatch.setattr(
        module,
        "_container_exists",
        lambda _name: (_ for _ in ()).throw(
            AssertionError("identity drift must fail before Docker inspection")
        ),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("identity drift must not mutate Docker")
        ),
    )

    with pytest.raises(RuntimeError, match="persisted launch identity"):
        module.start_generation(
            project_root=root,
            generation=launch.generation,
            compose_file=compose,
            runtime_artifact_root=runtime_root,
            state_root=state_root,
        )


def test_running_matching_generation_is_idempotent_without_docker_mutation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    root, state_root, runtime_root, compose = _prepared_start_inputs(tmp_path, launch)
    _mock_start_identity(monkeypatch, module, launch)
    monkeypatch.setattr(module, "_container_exists", lambda _name: True)
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=True,
            oom_killed=False,
            exit_code=0,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("idempotent start must not mutate Docker")
        ),
    )

    result = module.start_generation(
        project_root=root,
        generation=launch.generation,
        compose_file=compose,
        runtime_artifact_root=runtime_root,
        state_root=state_root,
    )

    assert result == launch


def test_operator_stop_copy_failure_resumes_through_collect(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import subprocess

    module = _module()
    launch = _launch(module)
    state_root = tmp_path / "state"
    retained_root = tmp_path / "retained"
    _write_launch(
        state_root / launch.generation / "launch-manifest.json",
        launch,
    )
    states = iter(
        (
            module.ContainerState(
                running=True,
                oom_killed=False,
                exit_code=0,
            ),
            module.ContainerState(
                running=False,
                oom_killed=False,
                exit_code=143,
            ),
        )
    )
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: next(states),
    )
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "stopped\n")

    def fail_copy(command, **_kwargs) -> None:
        if tuple(command)[:3] == ("docker", "container", "stop"):
            return
        assert tuple(command)[:2] == ("docker", "cp")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module, "_run", fail_copy)
    with pytest.raises(subprocess.CalledProcessError):
        module.stop_generation(
            generation=launch.generation,
            state_root=state_root,
            retained_root=retained_root,
        )

    retained = retained_root / launch.generation
    incomplete = json.loads(
        (retained / "research-result.json").read_text(encoding="utf-8")
    )
    assert incomplete["execution_status"] == "operator_stopped"
    assert incomplete["run_output_retained"] is False

    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=143,
        ),
    )
    monkeypatch.setattr(
        module,
        "_container_logs",
        lambda _launch: (_ for _ in ()).throw(
            AssertionError("retry must preserve the first retained log")
        ),
    )

    def copy(command, **_kwargs) -> None:
        assert tuple(command)[:2] == ("docker", "cp")
        destination = Path(tuple(command)[-1])
        (destination / "artifact.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", copy)
    result = module.collect_generation(
        generation=launch.generation,
        state_root=state_root,
        retained_root=retained_root,
    )

    assert result["execution_status"] == "operator_stopped"
    assert result["research_outcome"] == "unavailable"
    assert result["run_output_retained"] is True
    assert (retained / "run/artifact.json").is_file()
    assert (retained / "container.log").read_text() == "stopped\n"


def test_collect_recovers_launch_only_partial_retention(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    state_root = tmp_path / "state"
    retained_root = tmp_path / "retained"
    _write_launch(
        state_root / launch.generation / "launch-manifest.json",
        launch,
    )
    retained = retained_root / launch.generation
    retained.mkdir(parents=True)
    _write_launch(retained / "launch-manifest.json", launch)
    (retained / "container.log").write_text("preserved\n", encoding="utf-8")
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=1,
        ),
    )
    monkeypatch.setattr(
        module,
        "_container_logs",
        lambda _launch: (_ for _ in ()).throw(
            AssertionError("partial retry must preserve retained logs")
        ),
    )

    def copy(command, **_kwargs) -> None:
        destination = Path(tuple(command)[-1])
        (destination / "artifact.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", copy)
    result = module.collect_generation(
        generation=launch.generation,
        state_root=state_root,
        retained_root=retained_root,
    )

    assert result["execution_status"] == "failed"
    assert result["run_output_retained"] is True
    assert (retained / "container.log").read_text() == "preserved\n"
    assert (retained / "run/artifact.json").is_file()


def test_operator_stop_oom_retry_preserves_operator_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    module = _module()
    launch = _launch(module)
    state_root = tmp_path / "state"
    retained_root = tmp_path / "retained"
    _write_launch(
        state_root / launch.generation / "launch-manifest.json",
        launch,
    )
    retained = retained_root / launch.generation
    retained.mkdir(parents=True)
    _write_launch(retained / "launch-manifest.json", launch)
    (retained / "container.log").write_text("stopped-oom\n", encoding="utf-8")
    incomplete = {
        "container_exit_code": 137,
        "execution_status": "operator_stopped",
        "generation": launch.generation,
        "launch": launch.to_payload(),
        "oom_killed": True,
        "research_outcome": "unavailable",
        "run_output_retained": False,
        "schema_version": "causal_alpha_v3_research_result_v1",
    }
    (retained / "research-result.json").write_bytes(_canonical(incomplete))
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=True,
            exit_code=137,
        ),
    )
    monkeypatch.setattr(
        module,
        "_container_logs",
        lambda _launch: (_ for _ in ()).throw(
            AssertionError("retry must preserve the retained stop log")
        ),
    )

    def copy(command, **_kwargs) -> None:
        destination = Path(tuple(command)[-1])
        (destination / "artifact.json").write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(module, "_run", copy)
    result = module.collect_generation(
        generation=launch.generation,
        state_root=state_root,
        retained_root=retained_root,
    )

    assert result["execution_status"] == "operator_stopped"
    assert result["research_outcome"] == "unavailable"
    assert result["oom_killed"] is True
    assert result["run_output_retained"] is True
