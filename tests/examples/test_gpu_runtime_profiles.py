from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "run_gpu_training_smoke_profiles", SMOKE
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _provenance(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TRADE_RL_GIT_COMMIT", "a" * 40)
    monkeypatch.setenv("TRADE_RL_GIT_DIRTY", "false")


def test_compatibility_profile_preserves_uncompiled_in_process_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provenance(monkeypatch)
    config = _load_smoke().build_smoke_config(
        timesteps=128,
        runtime_profile="compatibility",
    )

    assert config.training.sequence_compile is False
    assert config.training.sequence_compile_mode == "reduce-overhead"
    assert config.training.sequence_transfer_mode == "synchronous"
    assert config.training.vector_environment_mode == "in_process"


def test_accelerated_profile_enables_h3_and_h4_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _provenance(monkeypatch)
    config = _load_smoke().build_smoke_config(
        timesteps=128,
        runtime_profile="accelerated",
    )

    assert config.training.sequence_compile is True
    assert config.training.sequence_compile_mode == "reduce-overhead"
    assert config.training.sequence_transfer_mode == "pinned_non_blocking"
    assert config.training.vector_environment_mode == "subprocess"


def test_runtime_profile_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    _provenance(monkeypatch)
    with pytest.raises(ValueError, match="runtime_profile"):
        _load_smoke().build_smoke_config(
            timesteps=128,
            runtime_profile="unknown",
        )


def test_parser_defaults_to_compatibility() -> None:
    args = _load_smoke().build_parser().parse_args([])
    assert args.runtime_profile == "compatibility"
