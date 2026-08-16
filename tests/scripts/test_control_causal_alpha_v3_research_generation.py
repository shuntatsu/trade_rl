from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import control_causal_alpha_v3_research_generation as module


def _launch() -> module.CausalAlphaV3Launch:
    return module.CausalAlphaV3Launch(
        generation="causal-alpha-v3-v2-r1",
        container_name="trade-rl-causal-alpha-v3-causal-alpha-v3-v2-r1",
        image="trade-rl-causal-alpha-v3:aaaaaaaaaaaa-dddddddddddd",
        image_id="1" * 64,
        git_commit="a" * 40,
        source_tree_digest="b" * 64,
        lockfile_digest="c" * 64,
        runtime_manifest_digest="d" * 64,
        research_config_digest="e" * 64,
        run_config_digest="f" * 64,
        output_path="/workspace/var/runs/causal-alpha-v3-v2-r1",
    )


@pytest.mark.parametrize(
    "value",
    ("../x", "x/y", "x y", "", ".", "x;rm", "x$(id)"),
)
def test_generation_rejects_unsafe_segments(value: str) -> None:
    with pytest.raises(ValueError, match="generation"):
        module.validate_generation(value)


def test_generation_accepts_stable_identifier() -> None:
    assert module.validate_generation("causal-alpha-v3-v2-r1") == (
        "causal-alpha-v3-v2-r1"
    )


@pytest.mark.parametrize(
    ("code", "expected"),
    (
        (0, ("completed", "admitted")),
        (2, ("completed", "signal_rejected")),
        (3, ("completed", "selection_rejected")),
        (4, ("completed", "admission_rejected")),
        (1, ("failed", "unavailable")),
        (137, ("failed", "unavailable")),
    ),
)
def test_research_exit_code_classification(
    code: int, expected: tuple[str, str]
) -> None:
    assert module.classify_research_outcome(code) == expected


def test_operator_stop_never_becomes_scientific_rejection() -> None:
    assert module.classify_research_outcome(143, operator_stopped=True) == (
        "operator_stopped",
        "unavailable",
    )


def test_launch_payload_round_trips_strictly() -> None:
    launch = _launch()
    payload = launch.to_payload()

    assert payload["schema_version"] == "causal_alpha_v3_launch_v1"
    assert module.CausalAlphaV3Launch.from_payload(payload) == launch

    malformed = dict(payload)
    malformed["extra"] = "not allowed"
    with pytest.raises(ValueError, match="fields"):
        module.CausalAlphaV3Launch.from_payload(malformed)


def test_start_rejects_dirty_tree_before_docker_mutation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[tuple[str, ...]] = []
    monkeypatch.setattr(module, "_git_status", lambda _root: " M trade_rl/x.py")
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: calls.append(tuple(command)),
    )

    with pytest.raises(RuntimeError, match="clean Git tree"):
        module.start_generation(
            project_root=tmp_path,
            generation="causal-alpha-v3-v2-r1",
            compose_file=tmp_path / "docker/compose.causal-alpha-v3-research.yaml",
            runtime_artifact_root=tmp_path / "runtime",
            state_root=tmp_path / "state",
        )

    assert calls == []


def test_start_rejects_existing_generation_state_before_build(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    state = tmp_path / "state" / "causal-alpha-v3-v2-r1"
    state.mkdir(parents=True)
    monkeypatch.setattr(module, "_git_status", lambda _root: "")
    monkeypatch.setattr(module, "_git", lambda *_args: "a" * 40)

    with pytest.raises(FileExistsError, match="state"):
        module.start_generation(
            project_root=tmp_path,
            generation="causal-alpha-v3-v2-r1",
            compose_file=tmp_path / "docker/compose.causal-alpha-v3-research.yaml",
            runtime_artifact_root=tmp_path / "runtime",
            state_root=tmp_path / "state",
        )


def test_status_is_read_only_and_validates_container_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )
    calls: list[tuple[str, ...]] = []

    def capture(command, **_kwargs):
        call = tuple(command)
        calls.append(call)
        if call[:3] == ("docker", "container", "inspect"):
            return json.dumps(
                {
                    "State": {"Running": True, "OOMKilled": False, "ExitCode": 0},
                    "Config": {
                        "Image": launch.image,
                        "Labels": {
                            "trade-rl.kind": "causal-alpha-v3",
                            "trade-rl.generation": launch.generation,
                            "trade-rl.git-commit": launch.git_commit,
                            "trade-rl.runtime-manifest-digest": (
                                launch.runtime_manifest_digest
                            ),
                        },
                    },
                }
            )
        raise AssertionError(call)

    monkeypatch.setattr(module, "_run_capture", capture)

    payload = module.status_generation(
        generation=launch.generation,
        state_root=tmp_path / "state",
    )

    assert payload["container_status"] == "running"
    assert payload["generation"] == launch.generation
    assert all("stop" not in call and "rm" not in call and "cp" not in call for call in calls)


def test_status_rejects_foreign_container_identity(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )

    monkeypatch.setattr(
        module,
        "_run_capture",
        lambda *_args, **_kwargs: json.dumps(
            {
                "State": {"Running": False, "OOMKilled": False, "ExitCode": 2},
                "Config": {
                    "Image": launch.image,
                    "Labels": {
                        "trade-rl.kind": "causal-alpha-v3",
                        "trade-rl.generation": launch.generation,
                        "trade-rl.git-commit": "9" * 40,
                        "trade-rl.runtime-manifest-digest": (
                            launch.runtime_manifest_digest
                        ),
                    },
                },
            }
        ),
    )

    with pytest.raises(RuntimeError, match="identity"):
        module.status_generation(
            generation=launch.generation,
            state_root=tmp_path / "state",
        )


def test_collect_rejects_running_container_without_copying(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )
    copied: list[tuple[str, ...]] = []

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
        lambda command, **_kwargs: copied.append(tuple(command)),
    )

    with pytest.raises(RuntimeError, match="running"):
        module.collect_generation(
            generation=launch.generation,
            state_root=tmp_path / "state",
            retained_root=tmp_path / "retained",
        )

    assert copied == []


def test_collect_preserves_scientific_rejection_as_completed_research(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=2,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: commands.append(tuple(command)),
    )
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "signal rejected\n")

    result = module.collect_generation(
        generation=launch.generation,
        state_root=tmp_path / "state",
        retained_root=tmp_path / "retained",
    )

    assert result["execution_status"] == "completed"
    assert result["research_outcome"] == "signal_rejected"
    assert result["container_exit_code"] == 2
    retained = tmp_path / "retained" / launch.generation
    assert json.loads((retained / "research-result.json").read_text()) == result
    assert (retained / "container.log").read_text() == "signal rejected\n"
    assert any(call[:2] == ("docker", "cp") for call in commands)
    assert not any("rm" in call for call in commands)


def test_stop_marks_operator_stopped_and_retains_partial_evidence(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )
    states = iter(
        (
            module.ContainerState(running=True, oom_killed=False, exit_code=0),
            module.ContainerState(running=False, oom_killed=False, exit_code=143),
        )
    )
    commands: list[tuple[str, ...]] = []

    monkeypatch.setattr(module, "_inspect_container", lambda _launch: next(states))
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: commands.append(tuple(command)),
    )
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "stopped\n")

    result = module.stop_generation(
        generation=launch.generation,
        state_root=tmp_path / "state",
        retained_root=tmp_path / "retained",
    )

    assert result["execution_status"] == "operator_stopped"
    assert result["research_outcome"] == "unavailable"
    assert any(call[:3] == ("docker", "container", "stop") for call in commands)
    assert not any("rm" in call for call in commands)
