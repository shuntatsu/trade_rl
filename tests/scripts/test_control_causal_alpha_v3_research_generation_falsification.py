from __future__ import annotations

import json
import subprocess
from pathlib import Path

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


def _write_launch(tmp_path: Path) -> module.CausalAlphaV3Launch:
    launch = _launch()
    state = tmp_path / "state" / launch.generation
    state.mkdir(parents=True)
    (state / "launch-manifest.json").write_text(
        json.dumps(launch.to_payload()), encoding="utf-8"
    )
    return launch


def test_stop_rejects_already_terminal_container_to_preserve_research_outcome(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _write_launch(tmp_path)
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

    with pytest.raises(RuntimeError, match="already terminal"):
        module.stop_generation(
            generation=launch.generation,
            state_root=tmp_path / "state",
            retained_root=tmp_path / "retained",
        )

    assert not (tmp_path / "retained").exists()
    assert commands == []


def test_collect_oom_overrides_scientific_exit_code(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _write_launch(tmp_path)
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=True,
            exit_code=2,
        ),
    )
    monkeypatch.setattr(module, "_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "oom\n")

    result = module.collect_generation(
        generation=launch.generation,
        state_root=tmp_path / "state",
        retained_root=tmp_path / "retained",
    )

    assert result["execution_status"] == "failed"
    assert result["research_outcome"] == "unavailable"
    assert result["oom_killed"] is True


def test_collection_never_removes_source_container_or_volume(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _write_launch(tmp_path)
    commands: list[tuple[str, ...]] = []
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=0,
        ),
    )
    monkeypatch.setattr(
        module,
        "_run",
        lambda command, **_kwargs: commands.append(tuple(command)),
    )
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "done\n")

    module.collect_generation(
        generation=launch.generation,
        state_root=tmp_path / "state",
        retained_root=tmp_path / "retained",
    )

    flattened = [token for command in commands for token in command]
    assert "rm" not in flattened
    assert "volume" not in flattened


def test_copy_failure_retains_logs_launch_and_incomplete_result(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    launch = _write_launch(tmp_path)
    monkeypatch.setattr(
        module,
        "_inspect_container",
        lambda _launch: module.ContainerState(
            running=False,
            oom_killed=False,
            exit_code=1,
        ),
    )
    monkeypatch.setattr(module, "_container_logs", lambda _launch: "runtime failed\n")

    def fail_copy(command, **_kwargs):
        assert tuple(command)[:2] == ("docker", "cp")
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(module, "_run", fail_copy)

    with pytest.raises(subprocess.CalledProcessError):
        module.collect_generation(
            generation=launch.generation,
            state_root=tmp_path / "state",
            retained_root=tmp_path / "retained",
        )

    retained = tmp_path / "retained" / launch.generation
    assert (retained / "container.log").read_text() == "runtime failed\n"
    assert json.loads((retained / "launch-manifest.json").read_text()) == (
        launch.to_payload()
    )
    result = json.loads((retained / "research-result.json").read_text())
    assert result["execution_status"] == "failed"
    assert result["research_outcome"] == "unavailable"
    assert result["run_output_retained"] is False
