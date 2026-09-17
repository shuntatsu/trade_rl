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
        directional_study,
        "official_activation_from_environment",
        lambda: {"workflow_run_id": 1},
        raising=False,
    )
    monkeypatch.setattr(
        directional_study, "validate_required_runtime", fail_runtime, raising=False
    )
    monkeypatch.setattr(
        directional_study, "load_market_dataset_artifact", forbidden_load
    )
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="stable-baselines3"):
        directional_study.prepare_study(tmp_path / "source", output)
    assert loaded == []
    assert not output.exists()


def _activation_environment() -> dict[str, str]:
    return {
        "GITHUB_ACTIONS": "true",
        "GITHUB_EVENT_NAME": "push",
        "GITHUB_REF_NAME": "run/issue640-directional-study-v1",
        "GITHUB_RUN_ID": "123456789",
        "GITHUB_RUN_NUMBER": "1",
        "GITHUB_RUN_ATTEMPT": "1",
        "GITHUB_SHA": "a" * 40,
        "GITHUB_WORKFLOW_REF": (
            "shuntatsu/trade_rl/.github/workflows/issue640-directional-study-v1.yml"
            "@refs/heads/run/issue640-directional-study-v1"
        ),
    }


def test_official_activation_accepts_only_first_exact_actions_run() -> None:
    environment = _activation_environment()
    activation = directional_study.official_activation_from_environment(environment)
    assert activation == {
        "schema_version": "directional_development_activation_v1",
        "workflow_run_id": 123456789,
        "workflow_run_number": 1,
        "workflow_run_attempt": 1,
        "workflow_event": "push",
        "workflow_ref": environment["GITHUB_WORKFLOW_REF"],
        "ref_name": "run/issue640-directional-study-v1",
        "head_sha": "a" * 40,
    }

    for key, value in (
        ("GITHUB_ACTIONS", "false"),
        ("GITHUB_EVENT_NAME", "workflow_dispatch"),
        ("GITHUB_REF_NAME", "scratch/directional-study"),
        ("GITHUB_RUN_NUMBER", "2"),
        ("GITHUB_RUN_ATTEMPT", "2"),
        ("GITHUB_SHA", "short"),
    ):
        changed = dict(environment)
        changed[key] = value
        with pytest.raises(RuntimeError, match="official activation"):
            directional_study.official_activation_from_environment(changed)


def test_prepare_rejects_nonofficial_activation_before_runtime_or_source(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    touched: list[str] = []

    def fail_activation() -> dict[str, object]:
        raise RuntimeError("official activation is unavailable")

    def forbidden_runtime() -> dict[str, str]:
        touched.append("runtime")
        raise AssertionError("runtime must not run after activation failure")

    def forbidden_load(path: Path) -> object:
        touched.append(f"load:{path}")
        raise AssertionError("source must not load after activation failure")

    monkeypatch.setattr(
        directional_study,
        "official_activation_from_environment",
        fail_activation,
        raising=False,
    )
    monkeypatch.setattr(
        directional_study, "validate_required_runtime", forbidden_runtime
    )
    monkeypatch.setattr(
        directional_study, "load_market_dataset_artifact", forbidden_load
    )
    output = tmp_path / "study"
    with pytest.raises(RuntimeError, match="official activation"):
        directional_study.prepare_study(tmp_path / "source", output)
    assert touched == []
    assert not output.exists()
