from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable

import pytest

ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = (
    ROOT
    / "examples"
    / "binance-multitimeframe"
    / "training_cuda_preflight.py"
)


def _load_preflight() -> Callable[[Path, Any], dict[str, object]]:
    spec = importlib.util.spec_from_file_location("training_cuda_preflight", PREFLIGHT)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load CUDA preflight module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.write_cuda_preflight_evidence


class _CudaProbe:
    def __init__(self, *, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available

    def get_device_name(self, index: int) -> str:
        assert index == 0
        return "NVIDIA Test GPU"

    def get_device_capability(self, index: int) -> tuple[int, int]:
        assert index == 0
        return (8, 9)

    def get_device_properties(self, index: int) -> SimpleNamespace:
        assert index == 0
        return SimpleNamespace(total_memory=24 * 1024**3)


def test_preflight_fails_closed_when_cuda_is_missing(tmp_path: Path) -> None:
    torch_probe = SimpleNamespace(cuda=_CudaProbe(available=False))

    with pytest.raises(RuntimeError, match="CUDA is required"):
        _load_preflight()(tmp_path / "cuda-preflight.json", torch_probe)


def test_preflight_writes_visible_cuda_device_evidence(tmp_path: Path) -> None:
    output = tmp_path / "cuda-preflight.json"
    torch_probe = SimpleNamespace(cuda=_CudaProbe(available=True))

    evidence = _load_preflight()(output, torch_probe)

    assert evidence == {
        "capability": [8, 9],
        "device_index": 0,
        "device_name": "NVIDIA Test GPU",
        "resolved_device": "cuda:0",
        "total_memory_bytes": 24 * 1024**3,
    }
    assert json.loads(output.read_text(encoding="utf-8")) == evidence
