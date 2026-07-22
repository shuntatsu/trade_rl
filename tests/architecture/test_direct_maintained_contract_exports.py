from __future__ import annotations

import ast
import inspect
from pathlib import Path

from trade_rl.simulation import MarketExecutor
from trade_rl.simulation.execution import MarketExecutor as DirectMarketExecutor
from trade_rl.simulation.execution_adapter import StatefulCompatibilityMarketExecutor
from trade_rl.studio.strict_telemetry import StrictStudioTelemetryReader
from trade_rl.studio.telemetry import StudioTelemetryReader
from trade_rl.telemetry import TrainingTelemetryRecord, TrainingTelemetryWriter
from trade_rl.telemetry.indexed_training import (
    IndexedTrainingTelemetryWriter,
    StrictTrainingTelemetryRecord,
)
from trade_rl.telemetry.training import (
    TrainingTelemetryRecord as DirectTrainingTelemetryRecord,
)
from trade_rl.telemetry.training import (
    TrainingTelemetryWriter as DirectTrainingTelemetryWriter,
)

ROOT = Path(__file__).resolve().parents[2]


def _source(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def _tree(path: str) -> ast.Module:
    return ast.parse(_source(path))


def test_simulation_public_executor_is_declared_by_execution_module() -> None:
    assert MarketExecutor is DirectMarketExecutor
    assert StatefulCompatibilityMarketExecutor is DirectMarketExecutor
    assert DirectMarketExecutor.__module__ == "trade_rl.simulation.execution"


def test_telemetry_public_contracts_are_declared_by_training_module() -> None:
    assert TrainingTelemetryRecord is DirectTrainingTelemetryRecord
    assert StrictTrainingTelemetryRecord is DirectTrainingTelemetryRecord
    assert DirectTrainingTelemetryRecord.__module__ == "trade_rl.telemetry.training"

    assert TrainingTelemetryWriter is DirectTrainingTelemetryWriter
    assert IndexedTrainingTelemetryWriter is DirectTrainingTelemetryWriter
    assert DirectTrainingTelemetryWriter.__module__ == "trade_rl.telemetry.training"


def test_studio_public_reader_is_declared_by_telemetry_module() -> None:
    assert StrictStudioTelemetryReader is StudioTelemetryReader
    assert StudioTelemetryReader.__module__ == "trade_rl.studio.telemetry"


def test_compatibility_modules_are_behavior_free_aliases() -> None:
    for path in (
        "trade_rl/simulation/execution_adapter.py",
        "trade_rl/telemetry/indexed_training.py",
        "trade_rl/studio/strict_telemetry.py",
    ):
        tree = _tree(path)
        assert not any(isinstance(node, ast.ClassDef) for node in tree.body), path
        assert not any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and not node.name.startswith("_")
            for node in tree.body
        ), path


def test_package_facades_import_canonical_public_modules() -> None:
    simulation = _source("trade_rl/simulation/__init__.py")
    telemetry = _source("trade_rl/telemetry/__init__.py")
    studio_api = _source("trade_rl/studio/api.py")

    assert "from trade_rl.simulation.execution import" in simulation
    assert "StatefulCompatibilityMarketExecutor" not in simulation
    assert "from trade_rl.telemetry.training import" in telemetry
    assert "indexed_training" not in telemetry
    assert "from trade_rl.studio.telemetry import" in studio_api
    assert "strict_telemetry" not in studio_api


def test_direct_executor_owns_stateful_compatibility_body() -> None:
    source = inspect.getsource(DirectMarketExecutor.execute_interval)
    assert "execute_target_statefully(" in source
    assert "compatibility_target_execution_v1" in source
    assert "_compatibility_order_book" in source
    assert "_target_weights(" not in _source("trade_rl/simulation/execution.py")


def test_direct_training_module_owns_strict_indexed_contract() -> None:
    source = _source("trade_rl/telemetry/training.py")
    assert "expected_generation" in source
    assert "_IndexedTrainingTelemetryWriter" in source
    assert "def _required_bool(" in source
    assert "must be a boolean" in source
    assert "self._handle: IO[str]" not in source


def test_direct_studio_reader_rejects_duplicate_seed_streams() -> None:
    source = inspect.getsource(StudioTelemetryReader._paths)
    assert "multiple telemetry streams claim seed" in source
    assert "candidate.is_symlink()" in source
