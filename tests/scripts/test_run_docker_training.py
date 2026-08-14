from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_module():
    path = (
        Path(__file__).resolve().parents[2]
        / "examples"
        / "binance-multitimeframe"
        / "run_docker_training.py"
    )
    spec = importlib.util.spec_from_file_location("run_docker_training", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_launcher_runs_sync_before_trainer() -> None:
    module = _load_module()
    commands = module.build_commands(Path("docker/compose.training.yaml"))

    assert commands[0][-1] == "market-data-sync"
    assert commands[1][-1] == "trainer"
    assert "--no-deps" not in commands[1]


def test_launcher_stops_when_sync_fails() -> None:
    module = _load_module()
    calls: list[tuple[str, ...]] = []

    def runner(command: tuple[str, ...]) -> int:
        calls.append(command)
        return 9

    result = module.run_training(
        compose_file=Path("docker/compose.training.yaml"),
        runner=runner,
    )

    assert result == 9
    assert len(calls) == 1
    assert calls[0][-1] == "market-data-sync"
