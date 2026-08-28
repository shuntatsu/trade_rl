from __future__ import annotations

import gc
import sys
import weakref
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from trade_rl.workflows import universal_causal_alpha_v5_stage_entry as stage_entry


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


def test_concrete_entry_releases_superseded_preparation_inputs_before_stages(
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
    monkeypatch.setattr(
        stage_entry,
        "prepare_causal_alpha_v4_stage_data",
        lambda **_kwargs: prepared,
    )
    monkeypatch.setattr(stage_entry, "CausalAlphaV5RunLock", _RunLock)
    monkeypatch.setattr(stage_entry, "CausalAlphaV5ArtifactStore", _Store)

    sentinel = object()

    def run_pipeline(**_kwargs: object) -> object:
        gc.collect()
        assert all(reference() is None for reference in references.values())
        return sentinel

    monkeypatch.setattr(
        stage_entry,
        "run_universal_causal_alpha_v5_research_pipeline",
        run_pipeline,
    )

    result = stage_entry.run_causal_alpha_v5_concrete_entry(
        config_path=Path("examples/binance/universal-causal-alpha-v5-research.json"),
        run_config_path=tmp_path / "run.json",
        runtime_manifest_path=tmp_path / "runtime.json",
        v4_context_manifest_path=tmp_path / "context.json",
        frozen_metadata_root=tmp_path / "metadata",
        output_root=tmp_path / "output",
    )

    assert result is sentinel
