from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from trade_rl.learning.causal_alpha_v6 import (
    CausalAlphaV6Candidate,
    CausalAlphaV6TargetConfig,
)
from trade_rl.workflows import universal_causal_alpha_v6_stage_entry as stage_entry
from trade_rl.workflows.universal_causal_alpha_v6_stage_execution import (
    execute_causal_alpha_v6_stage_callbacks,
)


class _WeakReferenceable:
    pass


class _RunLock:
    def __init__(self, _root: Path) -> None:
        pass

    def __enter__(self) -> None:
        return None

    def __exit__(self, *_args: object) -> None:
        return None


class _Store:
    def __init__(self, _root: Path, **_identity: object) -> None:
        pass

    def write_leaf(self, _name: str, _payload: object) -> None:
        pass


def test_v6_stage_callbacks_stop_before_forbidden_later_stage() -> None:
    calls: list[str] = []

    def stage(name: str, passed: bool = True):
        def run(*_args: object):
            calls.append(name)
            return SimpleNamespace(passed=passed)

        return run

    result = execute_causal_alpha_v6_stage_callbacks(
        prepare_v4=stage("prepare"),
        build_signal=stage("signal"),
        replay_and_select=stage("selection", passed=False),
        untouched_admission=stage("admission"),
    )
    assert calls == ["prepare", "signal", "selection"]
    assert result[-1] is None


def test_v6_fit_uses_exact_contract_start_cutoff(monkeypatch: Any) -> None:
    observed: list[int] = []
    sentinel = object()

    def fit(**kwargs: object):
        observed.append(int(kwargs["knowledge_cutoff"]))
        return sentinel

    monkeypatch.setattr(stage_entry, "fit_causal_alpha_v4", fit)
    prepared = SimpleNamespace(train_symbols=("S0",), samples={"S0": object()})
    assert stage_entry._fit_one(prepared, 123) is sentinel
    assert observed == [123]


def test_v6_candidate_paths_share_every_non_candidate_input(monkeypatch: Any) -> None:
    calls: list[dict[str, object]] = []

    def compile_target(forecast: object, **kwargs: object):
        calls.append({"forecast": forecast, **kwargs})
        return kwargs["candidate"]

    monkeypatch.setattr(stage_entry, "causal_alpha_v6_target_path", compile_target)
    forecast = object()
    uncertainty = {"4h": np.zeros(2), "24h": np.zeros(2), "72h": np.zeros(2)}
    costs = np.zeros(2)
    caps = np.ones(2)
    actionable = np.ones(2, dtype=np.bool_)
    paths = stage_entry._paired_target_paths(
        forecast=forecast,
        uncertainty=uncertainty,
        costs=costs,
        caps=caps,
        actionable=actionable,
        config=CausalAlphaV6TargetConfig(),
        initial_weight=0.05,
    )
    assert set(paths) == set(CausalAlphaV6Candidate)
    assert {call["candidate"] for call in calls} == set(CausalAlphaV6Candidate)
    for field in (
        "forecast",
        "uncertainty",
        "one_way_cost_rates",
        "liquidity_weight_caps",
        "actionable_mask",
        "config",
    ):
        assert calls[0][field] is calls[1][field]
    assert calls[0]["initial_weight"] == calls[1]["initial_weight"] == 0.05


def test_v6_contract_columns_require_one_shared_cutoff() -> None:
    prepared = SimpleNamespace(
        train_symbols=("S0", "S1"),
        nested_partitions={
            "S0": SimpleNamespace(signal_contracts=(SimpleNamespace(start=10),)),
            "S1": SimpleNamespace(signal_contracts=(SimpleNamespace(start=11),)),
        },
    )
    with pytest.raises(ValueError, match="cutoff scope drifted"):
        stage_entry._contract_column(prepared, "signal_contracts", 0)


def test_concrete_entry_releases_superseded_preparation_inputs_before_signal(
    monkeypatch: Any,
    tmp_path: Path,
) -> None:
    references: dict[str, weakref.ReferenceType[_WeakReferenceable]] = {}

    def prepare_adapter(**_kwargs: object) -> tuple[object, object, object]:
        context = _WeakReferenceable()
        runtime = _WeakReferenceable()
        prepared_v3 = _WeakReferenceable()
        setattr(
            prepared_v3,
            "execution_identity",
            SimpleNamespace(source_tree_digest="a" * 64),
        )
        references.update(
            context=weakref.ref(context),
            runtime=weakref.ref(runtime),
            prepared_v3=weakref.ref(prepared_v3),
        )
        return context, runtime, prepared_v3

    prepared = SimpleNamespace(
        run_manifest_digest="b" * 64,
        v4_context_manifest_digest="c" * 64,
        generator_code_digest="d" * 64,
    )
    monkeypatch.setitem(
        sys.modules,
        "trade_rl.workflows.universal_causal_alpha_v4_runtime_adapter",
        SimpleNamespace(prepare_causal_alpha_v4_runtime_adapter=prepare_adapter),
    )
    config = SimpleNamespace(
        target=CausalAlphaV6TargetConfig(),
        digest="e" * 64,
    )
    monkeypatch.setitem(
        sys.modules,
        "trade_rl.workflows.universal_causal_alpha_v6_runner",
        SimpleNamespace(
            CausalAlphaV6ResearchConfig=SimpleNamespace(from_json=lambda _path: config)
        ),
    )
    monkeypatch.setattr(
        stage_entry,
        "prepare_causal_alpha_v4_stage_data",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(stage_entry, "CausalAlphaV6RunLock", _RunLock)
    monkeypatch.setattr(stage_entry, "CausalAlphaV6ArtifactStore", _Store)
    sentinel = object()

    def run_pipeline(**_kwargs: object) -> object:
        gc.collect()
        assert all(reference() is None for reference in references.values())
        return sentinel

    monkeypatch.setattr(
        stage_entry,
        "run_universal_causal_alpha_v6_research_pipeline",
        run_pipeline,
    )
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    result = stage_entry.run_causal_alpha_v6_concrete_entry(
        config_path=config_path,
        run_config_path=tmp_path / "run.json",
        runtime_manifest_path=tmp_path / "runtime.json",
        v4_context_manifest_path=tmp_path / "context.json",
        frozen_metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "output",
    )
    assert result is sentinel
