from __future__ import annotations

import json
import runpy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE_ROOT = ROOT / "examples" / "binance-multitimeframe"
RUNNER_PATH = EXAMPLE_ROOT / "run_full_research_state.py"
EXPECTED_DEFAULT_CANDIDATES = (
    (
        "target-weight-growth-gamma-one-ppo",
        "training-target-weight-growth-ppo.json",
    ),
    (
        "target-weight-constrained-growth-gamma-one",
        "training-target-weight-constrained-growth.json",
    ),
    (
        "target-weight-constrained-growth-discounted-168h",
        "training-target-weight-constrained-growth-discounted.json",
    ),
)


class _StopAfterConfigWrite(RuntimeError):
    pass


class _ObservationBuilder:
    def __init__(
        self,
        *,
        action_size: int,
        n_factors: int,
        finite_horizon: bool,
    ) -> None:
        del action_size, n_factors, finite_horizon

    def layout(self, dataset: object) -> object:
        del dataset
        return SimpleNamespace(size=1)


class _SequenceObservationBuilder:
    def schema_payload(self, dataset: object) -> dict[str, object]:
        del dataset
        return {"windows": ()}


def _runner_namespace() -> dict[str, Any]:
    return runpy.run_path(str(RUNNER_PATH))


def _candidate_rows(path: Path) -> tuple[tuple[str, str], ...]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return tuple(
        (str(candidate["name"]), str(candidate["run_file"]))
        for candidate in payload["candidates"]
    )


def test_default_full_research_uses_target_weight_growth_catalog() -> None:
    namespace = _runner_namespace()

    assert "_DEFAULT_WALK_FORWARD_TEMPLATE" in namespace
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    assert isinstance(template, Path)
    assert (
        template
        == (
            EXAMPLE_ROOT / "walk-forward-target-weight-constrained-growth.json"
        ).resolve()
    )
    assert _candidate_rows(template) == EXPECTED_DEFAULT_CANDIDATES


def test_develop_materializes_target_weight_default(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _runner_namespace()
    stages_type = namespace["BinanceFullResearchStages"]
    globals_dict = stages_type._develop.__globals__
    pipeline = globals_dict["pipeline"]
    captured: dict[str, Path] = {}
    resolution = SimpleNamespace(
        metadata=object(),
        execution_rule_histories=(),
        identity_evidence=object(),
        mode=object(),
        evidence_digest="a" * 64,
        write_artifacts=lambda output: None,
    )
    dataset_payload = {
        "dataset_id": "dataset-id",
        "artifact_digest": "artifact-digest",
    }
    dataset = SimpleNamespace(
        n_symbols=1,
        n_features=1,
        n_bars=100,
        dataset_id="dataset-id",
    )

    def build_dataset(**kwargs: object) -> dict[str, str]:
        del kwargs
        return dict(dataset_payload)

    def write_run_config(*, template_path: Path, output_path: Path) -> Path:
        captured["template_path"] = template_path
        captured["output_path"] = output_path
        raise _StopAfterConfigWrite

    monkeypatch.setitem(
        globals_dict,
        "_retain_runtime_promotion_report",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setitem(
        globals_dict,
        "require_separate_cache_root",
        lambda cache_root, work_root: cache_root,
    )
    monkeypatch.setitem(globals_dict, "BinanceMetadataMode", lambda value: value)
    monkeypatch.setitem(
        globals_dict,
        "BinancePublicTransport",
        lambda **kwargs: object(),
    )
    monkeypatch.setitem(
        globals_dict,
        "load_market_dataset_artifact",
        lambda path: dataset,
    )
    monkeypatch.setitem(globals_dict, "ObservationBuilder", _ObservationBuilder)
    monkeypatch.setitem(
        globals_dict,
        "SequenceObservationBuilder",
        _SequenceObservationBuilder,
    )
    monkeypatch.setattr(pipeline, "resolve_metadata", lambda **kwargs: resolution)
    monkeypatch.setattr(pipeline, "load_json", lambda path: {})
    monkeypatch.setattr(pipeline, "build_dataset", build_dataset)
    monkeypatch.setattr(pipeline, "policy_observation_count", lambda value: 1)
    monkeypatch.setattr(pipeline, "write_run_config", write_run_config)

    work_root = tmp_path / "run"
    work_root.mkdir()
    args = SimpleNamespace(
        cache_root=tmp_path / "cache",
        metadata_mode="test",
        conservative_static_path=None,
        training_template=None,
    )

    with pytest.raises(_StopAfterConfigWrite):
        stages_type(args)._develop(work_root)

    assert captured == {
        "template_path": (
            EXAMPLE_ROOT / "walk-forward-target-weight-constrained-growth.json"
        ),
        "output_path": work_root / "walk-forward-full.json",
    }


def test_training_full_is_available_only_through_explicit_template_selection() -> None:
    namespace = _runner_namespace()
    template = namespace["_DEFAULT_WALK_FORWARD_TEMPLATE"]
    example_template = namespace["_example_template"]

    assert "training-full.json" not in {
        run_file for _, run_file in _candidate_rows(template)
    }
    assert (
        example_template(
            "training-full.json",
            field="training template",
        )
        == (EXAMPLE_ROOT / "training-full.json").resolve()
    )
