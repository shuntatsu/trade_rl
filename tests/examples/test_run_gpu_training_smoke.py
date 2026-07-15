from __future__ import annotations

import importlib.util
from pathlib import Path
from types import ModuleType

ROOT = Path(__file__).resolve().parents[2]
SMOKE = ROOT / "examples" / "binance-multitimeframe" / "run_gpu_training_smoke.py"


def _load_smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("run_gpu_training_smoke", SMOKE)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not load GPU smoke module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_smoke_config_preserves_the_maintained_cuda_training_contract() -> None:
    config = _load_smoke().build_smoke_config(timesteps=128)

    assert config.training.device == "cuda"
    assert config.training.n_envs == 4
    assert config.training.policy_net_arch == (256, 256)
    assert config.training.asset_embedding_dim == 128
    assert config.training.global_embedding_dim == 128
    assert config.training.timesteps == 128
