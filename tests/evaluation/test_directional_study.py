import json
from pathlib import Path

import pytest

from trade_rl.evaluation import directional_study


def test_protocol_is_written_before_execution_and_cannot_be_replaced(
    tmp_path: Path,
) -> None:
    assert hasattr(directional_study, "reserve_study")
    root = tmp_path / "study"
    directional_study.reserve_study(
        root, {"maximum_drawdown": 0.2, "roster": ["breakout"]}
    )
    assert json.loads((root / "protocol.json").read_text())["maximum_drawdown"] == 0.2
    with pytest.raises(FileExistsError):
        directional_study.reserve_study(root, {"maximum_drawdown": 0.9})
    assert json.loads((root / "protocol.json").read_text())["maximum_drawdown"] == 0.2


def test_unknown_arm_is_rejected_before_any_training() -> None:
    with pytest.raises(ValueError, match="arm"):
        directional_study.validate_arm("ppo-seed-99")


def test_changed_stress_protocol_is_rejected_even_if_hash_is_rewritten(
    tmp_path: Path,
) -> None:
    from trade_rl.artifacts import canonical_json_bytes, content_digest

    expected = {"stresses": [{"cost_multiplier": 2.0}]}
    directional_study.reserve_study(tmp_path / "study", expected)
    changed = {"stresses": []}
    (tmp_path / "study" / "protocol.json").write_bytes(canonical_json_bytes(changed))
    (tmp_path / "study" / "protocol.digest.json").write_bytes(
        canonical_json_bytes({"digest": content_digest(changed)})
    )
    with pytest.raises(ValueError, match="protocol"):
        directional_study.validate_protocol(tmp_path / "study", expected)


def test_development_clock_requires_complete_exact_calendar_years() -> None:
    from dataclasses import replace

    import numpy as np

    from tests.evaluation.test_shared_cash_replay import _market

    market = _market(np.full((17546, 1), 100.0))
    clock = np.datetime64("2022-12-31T23", "ns") + np.arange(17546) * np.timedelta64(
        1, "h"
    )
    market = replace(market, timestamps=clock, available_at=clock[:, None])
    assert directional_study.development_indices(market) == (1, 17545)
    incomplete = replace(
        market,
        timestamps=clock + np.timedelta64(2, "h"),
        available_at=(clock + np.timedelta64(2, "h"))[:, None],
    )
    with pytest.raises(ValueError, match="clock"):
        directional_study.development_indices(incomplete)


def test_required_runtime_rejects_missing_or_wrong_frozen_trainers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    versions = {
        "lightgbm": "4.7.0",
        "stable-baselines3": "2.3.2",
        "torch": "2.4.1",
        "scikit-learn": "1.7.2",
    }
    imported: list[str] = []

    def version(name: str) -> str:
        if name not in versions:
            raise directional_study.metadata.PackageNotFoundError(name)
        return versions[name]

    def import_module(name: str) -> object:
        imported.append(name)
        return object()

    monkeypatch.setattr(directional_study.metadata, "version", version)
    monkeypatch.setattr(directional_study.importlib, "import_module", import_module)
    assert directional_study.validate_required_runtime() == versions
    assert imported == ["lightgbm", "stable_baselines3", "torch", "sklearn"]

    del versions["stable-baselines3"]
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        directional_study.validate_required_runtime()

    versions["stable-baselines3"] = "9.9.9"
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        directional_study.validate_required_runtime()


def test_prepare_fails_runtime_preflight_before_source_or_output_side_effects(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    loaded: list[Path] = []

    def fail_runtime() -> dict[str, str]:
        raise RuntimeError("stable-baselines3 runtime is unavailable")

    def forbidden_load(path: Path) -> object:
        loaded.append(path)
        raise AssertionError("source must not load before runtime preflight")

    monkeypatch.setattr(
        directional_study, "validate_required_runtime", fail_runtime, raising=False
    )
    monkeypatch.setattr(directional_study, "load_market_dataset_artifact", forbidden_load)
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        directional_study.prepare_study(tmp_path / "source", output)
    assert loaded == []
    assert not output.exists()
