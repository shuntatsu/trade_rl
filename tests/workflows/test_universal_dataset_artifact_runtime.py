from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from trade_rl.artifacts.hashing import content_digest
from trade_rl.rl.universal_instrument_binding import InstrumentDatasetBinding


def _digest(label: str) -> str:
    return content_digest(label)


def _binding(symbol: str) -> InstrumentDatasetBinding:
    return InstrumentDatasetBinding(
        concrete_symbol=symbol,
        source_dataset_id=_digest(f"dataset:{symbol}"),
        symbol_dataset_digest=_digest(f"dataset:{symbol}"),
        execution_metadata_digest=_digest(f"metadata:{symbol}"),
        instrument_descriptor_digest=_digest(f"descriptor:{symbol}"),
        split="train",
    )


def test_dataset_artifact_factory_keeps_paths_and_loads_only_requested_symbol(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        UniversalDatasetArtifactEnvironmentFactory,
    )

    paths = {
        "AAAUSDT": tmp_path / "AAAUSDT",
        "BBBUSDT": tmp_path / "BBBUSDT",
    }
    loaded: list[Path] = []
    built: list[str] = []

    def load(path: Path) -> SimpleNamespace:
        loaded.append(path)
        symbol = path.name
        return SimpleNamespace(
            symbols=(symbol,),
            dataset_id=_digest(f"dataset:{symbol}"),
        )

    def build(
        dataset: Any,
        *,
        run_config: Any,
        normalizers: tuple[Any, Any],
    ) -> SimpleNamespace:
        del run_config, normalizers
        built.append(dataset.symbols[0])
        return SimpleNamespace(dataset=dataset)

    monkeypatch.setattr(module, "load_market_dataset_artifact", load)
    monkeypatch.setattr(module, "_build_universal_concrete_environment", build)
    factory = UniversalDatasetArtifactEnvironmentFactory(
        dataset_artifact_paths=paths,
        run_config=SimpleNamespace(),
        normalizers={
            "AAAUSDT": (SimpleNamespace(), SimpleNamespace()),
            "BBBUSDT": (SimpleNamespace(), SimpleNamespace()),
        },
    )

    assert not hasattr(factory, "datasets")
    result = factory(_binding("BBBUSDT"))

    assert result.dataset.symbols == ("BBBUSDT",)
    assert loaded == [paths["BBBUSDT"]]
    assert built == ["BBBUSDT"]


def test_dataset_artifact_factory_rejects_loaded_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    import trade_rl.workflows.universal_training_runner as module
    from trade_rl.workflows.universal_training_runner import (
        UniversalDatasetArtifactEnvironmentFactory,
    )

    path = tmp_path / "AAAUSDT"
    monkeypatch.setattr(
        module,
        "load_market_dataset_artifact",
        lambda _path: SimpleNamespace(
            symbols=("AAAUSDT",),
            dataset_id=_digest("wrong-dataset"),
        ),
    )
    factory = UniversalDatasetArtifactEnvironmentFactory(
        dataset_artifact_paths={"AAAUSDT": path},
        run_config=SimpleNamespace(),
        normalizers={"AAAUSDT": (SimpleNamespace(), SimpleNamespace())},
    )

    with pytest.raises(ValueError, match="identity"):
        factory(_binding("AAAUSDT"))
