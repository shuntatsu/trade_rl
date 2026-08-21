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
